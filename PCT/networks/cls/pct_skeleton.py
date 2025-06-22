import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
from PCT.misc.ops import FurthestPointSampler
from PCT.misc.ops import knn_point, index_points

def sample_and_group(npoint, nsample, xyz, points):
    """
    Input:
        npoint: int
        nsample: int  
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, 3]
        new_points: sampled points data, [B, npoint, nsample, D+3]
    """
    B, N, C = xyz.shape
    S = npoint 
    sampler = FurthestPointSampler(npoint)
    _, fps_idx = sampler(xyz) 
    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx) # [B, npoint, nsample, D]
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_xyz_norm, grouped_points_norm], dim=-1)
    return new_xyz, new_points

def farthest_point_sample(xyz, npoint):
    """
    最远点采样
    """
    B, N, C = xyz.shape
    centroids = jt.zeros((B, npoint), dtype=jt.int32)
    distance = jt.ones((B, N)) * 1e10
    farthest = jt.randint(0, N, (B,), dtype=jt.int32)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[jt.arange(B), farthest, :].view(B, 1, 3)
        dist = jt.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = jt.argmax(distance, -1, keepdims=False)[0]
    
    return centroids

class SkeletonStructurePrior(nn.Module):
    """骨骼结构先验知识模块"""
    def __init__(self, feat_dim, num_joints=22):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
        # Human skeletal hierarchy (SMPL format)
        # Parent-child relationships: each joint's parent index; -1 indicates the root joint
        self.bone_hierarchy = jt.array([
            -1,  # 0: pelvis (root)
            0,   # 1: left_hip
            0,   # 2: right_hip
            0,   # 3: spine1
            1,   # 4: left_knee
            2,   # 5: right_knee
            3,   # 6: spine2
            4,   # 7: left_ankle
            5,   # 8: right_ankle
            6,   # 9: spine3
            7,   # 10: left_foot
            8,   # 11: right_foot
            9,   # 12: neck
            9,   # 13: left_collar
            9,   # 14: right_collar
            12,  # 15: head
            13,  # 16: left_shoulder
            14,  # 17: right_shoulder
            16,  # 18: left_elbow
            17,  # 19: right_elbow
            18,  # 20: left_wrist
            19,  # 21: right_wrist
        ])

        self.bone_length_predictor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_joints),
            nn.Softplus() 
        )

        self.joint_angle_predictor = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_joints * 3), 
            nn.Tanh() 
        )
        
        # Symmetry constraint
        self.symmetry_pairs = [
            (1, 2),   # left_hip, right_hip
            (4, 5),   # left_knee, right_knee
            (7, 8),   # left_ankle, right_ankle
            (10, 11), # left_foot, right_foot
            (13, 14), # left_collar, right_collar
            (16, 17), # left_shoulder, right_shoulder
            (18, 19), # left_elbow, right_elbow
            (20, 21), # left_wrist, right_wrist
        ]
        
    def compute_bone_constraints(self, joint_positions, global_features):
        """计算骨骼约束损失"""
        B = joint_positions.shape[0]
        joint_pos = joint_positions.view(B, self.num_joints, 3)
        
        predicted_lengths = self.bone_length_predictor(global_features)  # [B, num_joints]
        
        bone_length_loss = 0.0
        for i in range(1, self.num_joints):  
            parent_idx = self.bone_hierarchy[i]
            if parent_idx >= 0:
                bone_vector = joint_pos[:, i] - joint_pos[:, parent_idx]
                actual_length = jt.norm(bone_vector, dim=1)
                expected_length = predicted_lengths[:, i]
                bone_length_loss += jt.mean((actual_length - expected_length) ** 2)
        
        return bone_length_loss / (self.num_joints - 1)
    
    def compute_symmetry_loss(self, joint_positions):
        """计算对称性损失"""
        B = joint_positions.shape[0]
        joint_pos = joint_positions.view(B, self.num_joints, 3)
        
        symmetry_loss = 0.0
        for left_idx, right_idx in self.symmetry_pairs:
            left_pos = joint_pos[:, left_idx]
            right_pos = joint_pos[:, right_idx]
            # Symmetry constraint: left and right joints should be symmetric along the x-axis
            left_mirrored = jt.stack([-left_pos[:, 0], left_pos[:, 1], left_pos[:, 2]], dim=1)
            symmetry_loss += jt.mean((left_mirrored - right_pos) ** 2)
        
        return symmetry_loss / len(self.symmetry_pairs)

class SkeletonAttention(nn.Module):
    """专门针对骨骼预测的注意力机制 """
    def __init__(self, feat_dim, num_heads=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        
        assert feat_dim % num_heads == 0

        self.q_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_linear = nn.Linear(feat_dim, feat_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        
    def execute(self, x):
        B, N, D = x.shape

        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = jt.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        attended = jt.matmul(attention, v)
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.out_linear(attended)

class ProgressiveSkeletonAttention(nn.Module):
    """渐进式骨骼注意力模块"""
    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        assert out_channels % num_heads == 0

        self.q_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.pos_encoding = nn.Conv1d(3, out_channels, 1, bias=False)

        self.output_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def execute(self, x, xyz):
        B, C, N = x.shape
        
        # Add positional encoding
        pos_feat = self.pos_encoding(xyz)
        x_with_pos = x + pos_feat

        q = self.q_conv(x_with_pos).view(B, self.num_heads, self.head_dim, N)
        k = self.k_conv(x_with_pos).view(B, self.num_heads, self.head_dim, N)  
        v = self.v_conv(x_with_pos).view(B, self.num_heads, self.head_dim, N)

        attention_scores = jt.matmul(q.transpose(2, 3), k) / np.sqrt(self.head_dim)
        attention_weights = self.softmax(attention_scores)

        attended = jt.matmul(v, attention_weights.transpose(2, 3))
        attended = attended.view(B, self.out_channels, N)

        output = self.output_conv(attended)
        if self.in_channels == self.out_channels:
            output = output + x
        
        output = self.relu(self.norm(output))
        return output

class CrossScaleSkeletonAttention(nn.Module):
    """跨尺度骨骼注意力机制"""
    def __init__(self, channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels

        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, 1, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU()
            ) for _ in range(num_scales)
        ])
        
        # cross-scale fusion
        self.cross_scale_fusion = nn.Sequential(
            nn.Conv1d(channels * num_scales, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )
        
        self.attention_weights = jt.randn((num_scales, num_scales)) * 0.1
        self.softmax = nn.Softmax(dim=-1)
        
    def adaptive_resize(self, feat, target_size):
        """简单的自适应调整尺寸"""
        current_size = feat.shape[2]
        if current_size == target_size:
            return feat
        elif current_size > target_size:

            step = current_size // target_size
            return feat[:, :, ::step][:, :, :target_size]
        else:

            repeat_factor = target_size // current_size
            remainder = target_size % current_size
            upsampled = feat.repeat_interleave(repeat_factor, dim=2)
            if remainder > 0:
                upsampled = concat([upsampled, feat[:, :, :remainder]], dim=2)
            return upsampled[:, :, :target_size]
        
    def execute(self, scale_features):
        """
        scale_features: list of [B, C, N_i] tensors from different scales
        """
        B = scale_features[0].shape[0]
        target_size = 256  
        

        projected_features = []
        for i, (feat, proj) in enumerate(zip(scale_features, self.scale_projections)):
            projected_feat = proj(feat)

            resized_feat = self.adaptive_resize(projected_feat, target_size)
            projected_features.append(resized_feat)
        

        attention_matrix = self.softmax(self.attention_weights)

        fused_features = []
        for i, feat_i in enumerate(projected_features):
            weighted_feat = jt.zeros_like(feat_i)
            for j, feat_j in enumerate(projected_features):
                weighted_feat += attention_matrix[i, j] * feat_j
            fused_features.append(weighted_feat)

        all_features = concat(fused_features, dim=1)
        final_features = self.cross_scale_fusion(all_features)
        
        return final_features

class SkeletonTransformer(nn.Module):
    """专门针对骨骼预测的Transformer模型"""
    def __init__(self, output_channels=256, num_joints=22):
        super().__init__()
        self.output_channels = output_channels
        self.num_joints = num_joints
        
        self.initial_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Multi-scale progressive feature extraction
        # Level 1: Full resolution (1024 points)
        self.level1_attention = nn.ModuleList([
            ProgressiveSkeletonAttention(256, 256, num_heads=8)
            for _ in range(2)
        ])
        
        # Level 2: Medium resolution (512 points)  
        self.level2_transform = nn.Conv1d(259, 384, 1, bias=False)
        self.level2_attention = nn.ModuleList([
            ProgressiveSkeletonAttention(384, 384, num_heads=8)
            for _ in range(3)
        ])
        
        # Level 3: Low resolution (256 points)
        self.level3_transform = nn.Conv1d(387, 512, 1, bias=False)
        self.level3_attention = nn.ModuleList([
            ProgressiveSkeletonAttention(512, 512, num_heads=8)
            for _ in range(4)
        ])


        self.dim_reduction_2 = nn.Conv1d(384, 256, 1, bias=False)
        self.dim_reduction_3 = nn.Conv1d(512, 256, 1, bias=False)
        
        self.cross_scale_attention = CrossScaleSkeletonAttention(
            channels=256, num_scales=3
        )

        self.feature_fusion = nn.Sequential(
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.global_feature_extractor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_channels)
        )
        
    def adaptive_pool_1d(self, x, target_size):
        """简单的自适应池化实现"""
        current_size = x.shape[2]
        if current_size == target_size:
            return x
        elif current_size > target_size:
            if current_size % target_size == 0:
                kernel_size = current_size // target_size
                B, C, N = x.shape
                x_reshaped = x.view(B, C, target_size, kernel_size)
                return jt.max(x_reshaped, dim=3)
            else:
                indices = jt.linspace(0, current_size-1, target_size).long()
                return x[:, :, indices]
        else:
            repeat_factor = target_size // current_size
            remainder = target_size % current_size
            upsampled = x.repeat_interleave(repeat_factor, dim=2)
            if remainder > 0:
                extra_indices = jt.linspace(0, current_size-1, remainder).long()
                extra_points = x[:, :, extra_indices]
                upsampled = concat([upsampled, extra_points], dim=2)
            return upsampled[:, :, :target_size]
        
    def execute(self, vertices):
        """
        vertices: [B, 3, N] - 输入点云
        """
        B, _, N = vertices.shape

        x = self.initial_conv(vertices)  # [B, 256, N]
        xyz = vertices  # [B, 3, N]

        x1 = x
        for attention_layer in self.level1_attention:
            x1 = attention_layer(x1, xyz)

        xyz_512, x1_grouped = sample_and_group(512, 32, xyz.permute(0,2,1), x1.permute(0,2,1))
        x2 = x1_grouped.mean(dim=2).permute(0, 2, 1)  # [B, 259, 512]
        x2 = self.level2_transform(x2)  # [B, 384, 512]
        
        for attention_layer in self.level2_attention:
            x2 = attention_layer(x2, xyz_512.permute(0,2,1))

        xyz_256, x2_grouped = sample_and_group(256, 32, xyz_512, x2.permute(0,2,1))
        x3 = x2_grouped.mean(dim=2).permute(0, 2, 1)  # [B, 387, 256]
        x3 = self.level3_transform(x3)  # [B, 512, 256]
        
        for attention_layer in self.level3_attention:
            x3 = attention_layer(x3, xyz_256.permute(0,2,1))

        x1_pooled = self.adaptive_pool_1d(x1, 256)  # [B, 256, 256]
        x2_reduced = self.dim_reduction_2(x2)  # [B, 256, 512] 
        x2_pooled = self.adaptive_pool_1d(x2_reduced, 256)  # [B, 256, 256]
        x3_reduced = self.dim_reduction_3(x3)  # [B, 256, 256]

        scale_features = [x1_pooled, x2_pooled, x3_reduced]
        fused_features = self.cross_scale_attention(scale_features)  # [B, 256, 256]

        fused_features = self.feature_fusion(fused_features)  # [B, 1024, 256]
        
        global_feat = jt.max(fused_features, 2)  # [B, 1024]

        global_output = self.global_feature_extractor(global_feat)  # [B, output_channels]
        
        return global_output

class SkeletonPoint_Transformer(nn.Module):
    """骨骼预测的Point Transformer - PTv3风格"""
    def __init__(self, output_channels=256):
        super().__init__()
        self.skeleton_transformer = PTv3SkeletonTransformer(output_channels)
        
    def execute(self, x):
        """
        x: [B, 3, N] 输入点云坐标
        """
        return self.skeleton_transformer(x)

# Point Cloud Serialization
class PointCloudSerialization(nn.Module):
    """点云序列化模块 - 基于PTv3的空间填充曲线"""
    def __init__(self, grid_size=0.02):
        super().__init__()
        self.grid_size = grid_size
        
    def z_order_encode(self, xyz):
        """Z-order空间填充曲线编码"""
        B, N, _ = xyz.shape

        grid_coords = (xyz / self.grid_size).int()

        x, y, z = grid_coords[..., 0], grid_coords[..., 1], grid_coords[..., 2]

        code = jt.zeros_like(x)
        for i in range(16): 
            mask = 1 << i
            code |= ((x & mask) << (2*i)) | ((y & mask) << (2*i+1)) | ((z & mask) << (2*i+2))
            
        return code
    
    def hilbert_encode(self, xyz):
        """Hilbert曲线编码（简化版本）"""
        return self.z_order_encode(xyz)
    
    def execute(self, xyz, pattern='z'):
        """
        执行点云序列化
        pattern: 'z' for Z-order, 'h' for Hilbert
        """
        if pattern == 'z':
            codes = self.z_order_encode(xyz)
        elif pattern == 'h':
            codes = self.hilbert_encode(xyz)
        else:
            codes = self.z_order_encode(xyz)

        sorted_indices = jt.argsort(codes, dim=-1)
        return sorted_indices

class xCPE(nn.Module):
    """高效条件位置编码 - 基于PTv3设计"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # use 1D convolution simulate sparse convolution
        self.pos_conv = nn.Sequential(
            nn.Conv1d(3, out_channels // 2, 1, bias=False),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(),
            nn.Conv1d(out_channels // 2, out_channels, 1, bias=False)
        )
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip_conv = None
            
    def execute(self, x, xyz):
        """
        x: [B, C, N] 特征
        xyz: [B, 3, N] 坐标
        """
        pos_encoding = self.pos_conv(xyz)
        
        # Skip connection
        if self.skip_conv is not None:
            x = self.skip_conv(x)
            
        return x + pos_encoding

# Patch Attention
class PatchAttention(nn.Module):
    """基于Patch的注意力机制 - PTv3风格"""
    def __init__(self, channels, num_heads=8, patch_size=64):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0

        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False) 
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.out_conv = nn.Conv1d(channels, channels, 1)

        # self.norm = nn.LayerNorm(channels)
        self.softmax = nn.Softmax(dim=-1)
        
    def patch_partition(self, x):
        """将特征分割为patches"""
        B, C, N = x.shape
        
        # Padding to make N divisible by patch_size
        pad_size = (self.patch_size - N % self.patch_size) % self.patch_size
        if pad_size > 0:
            x = jt.concat([x, x[:, :, :pad_size]], dim=2)
            
        N_padded = x.shape[2]
        num_patches = N_padded // self.patch_size
        
        # Reshape to patches: [B, C, num_patches, patch_size]
        x_patches = x.view(B, C, num_patches, self.patch_size)
        return x_patches, pad_size
        
    def patch_merge(self, x_patches, pad_size, original_N):
        """合并patches回原始形状"""
        B, C, num_patches, patch_size = x_patches.shape
        x = x_patches.view(B, C, -1)
        
        # Remove padding
        if pad_size > 0:
            x = x[:, :, :-pad_size]
            
        return x[:, :, :original_N]
        
    def execute(self, x):
        """
        x: [B, C, N]
        """
        B, C, N = x.shape
        original_N = N

        x_patches, pad_size = self.patch_partition(x) 
        B, C, num_patches, patch_size = x_patches.shape

        x_flat = x_patches.view(B * num_patches, C, patch_size)

        q = self.q_conv(x_flat)
        k = self.k_conv(x_flat)
        v = self.v_conv(x_flat)

        q = q.view(B * num_patches, self.num_heads, self.head_dim, patch_size).permute(0, 1, 3, 2) 
        k = k.view(B * num_patches, self.num_heads, self.head_dim, patch_size).permute(0, 1, 3, 2)
        v = v.view(B * num_patches, self.num_heads, self.head_dim, patch_size).permute(0, 1, 3, 2)

        scores = jt.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim) 
        attention = self.softmax(scores)

        attended = jt.matmul(attention, v) 
        attended = attended.permute(0, 1, 3, 2).contiguous().view(B * num_patches, C, patch_size)

        output = self.out_conv(attended)  

        output_patches = output.view(B, C, num_patches, patch_size)

        output = self.patch_merge(output_patches, pad_size, original_N)
        
        return output

class ShiftPatchInteraction(nn.Module):
    """Shift Patch交互模块"""
    def __init__(self, channels, patch_size=64):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        
        self.attention1 = PatchAttention(channels, patch_size=patch_size)
        self.attention2 = PatchAttention(channels, patch_size=patch_size)
        
    def shift_patches(self, x, shift_size):
        """移动patches"""
        B, C, N = x.shape
        shifted = jt.concat([x[:, :, shift_size:], x[:, :, :shift_size]], dim=2)
        return shifted
        
    def execute(self, x):
        """
        x: [B, C, N]
        """
        # Standard patch attention
        out1 = self.attention1(x)
        
        # Shifted patch attention
        shift_size = self.patch_size // 2
        x_shifted = self.shift_patches(x, shift_size)
        out2 = self.attention2(x_shifted)
        out2 = self.shift_patches(out2, -shift_size)  # Shift back
        
        return out1 + out2

class PTv3TransformerBlock(nn.Module):
    """PTv3风格的Transformer块"""
    def __init__(self, channels, num_heads=8, patch_size=64, mlp_ratio=4):
        super().__init__()
        self.channels = channels

        self.pos_encoding = xCPE(channels, channels)

        self.attention = ShiftPatchInteraction(channels, patch_size)

        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
 
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv1d(mlp_hidden, channels, 1),
        )
        
    def execute(self, x, xyz):
        """
        x: [B, C, N] 特征
        xyz: [B, 3, N] 坐标
        """

        x = self.pos_encoding(x, xyz)
        
        # Self-attention with residual
        residual = x
        x = self.norm1(x) 
        x = self.attention(x)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.norm2(x) 
        x = self.mlp(x)
        x = residual + x
        
        return x


class PTv3SkeletonTransformer(nn.Module):
    """基于PTv3设计的骨骼预测Transformer"""
    def __init__(self, output_channels=256, num_joints=22):
        super().__init__()
        self.output_channels = output_channels
        self.num_joints = num_joints

        self.serialization = PointCloudSerialization()

        self.stem = nn.Sequential(
            nn.Conv1d(3, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.encoder_layers = nn.ModuleList([
            nn.ModuleList([
                PTv3TransformerBlock(128, patch_size=128),
                PTv3TransformerBlock(128, patch_size=128)
            ]),

            nn.ModuleList([
                PTv3TransformerBlock(256, patch_size=64),
                PTv3TransformerBlock(256, patch_size=64)
            ]),

            nn.ModuleList([
                PTv3TransformerBlock(512, patch_size=32),
                PTv3TransformerBlock(512, patch_size=32),
                PTv3TransformerBlock(512, patch_size=32),
                PTv3TransformerBlock(512, patch_size=32),
                PTv3TransformerBlock(512, patch_size=32),
                PTv3TransformerBlock(512, patch_size=32)
            ]),

            nn.ModuleList([
                PTv3TransformerBlock(512, patch_size=16),
                PTv3TransformerBlock(512, patch_size=16)
            ])
        ])
        
        # Downsampling layers
        self.downsample_layers = nn.ModuleList([
            nn.Conv1d(128, 256, 1, bias=False),  
            nn.Conv1d(256, 512, 1, bias=False),  
            nn.Conv1d(512, 512, 1, bias=False),  
        ])
        

        self.final_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_channels)
        )
        
    def global_avg_pool_1d(self, x):
        """手动实现全局平均池化"""
        # x: [B, C, N] -> [B, C]
        return jt.mean(x, dim=2)
        
    def grid_pool(self, x, xyz, scale_factor=2):
        """Grid pooling for downsampling"""
        B, C, N = x.shape
        target_N = N // scale_factor
        
        # Simple downsampling by taking every scale_factor-th point
        indices = jt.arange(0, N, scale_factor)[:target_N]
        
        downsampled_x = x[:, :, indices]
        downsampled_xyz = xyz[:, :, indices]
        
        return downsampled_x, downsampled_xyz
        
    def execute(self, vertices):
        """
        vertices: [B, 3, N] 输入点云
        """
        B, _, N = vertices.shape
        xyz = vertices

        x = self.stem(vertices)

        stage_features = []
        
        for stage_idx, stage_layers in enumerate(self.encoder_layers):
            for layer in stage_layers:
                x = layer(x, xyz)
            
            stage_features.append(x)

            if stage_idx < len(self.downsample_layers):
                x = self.downsample_layers[stage_idx](x)
                x, xyz = self.grid_pool(x, xyz, scale_factor=2)

        global_feat = self.global_avg_pool_1d(x)

        output = self.final_mlp(global_feat) 
        return output 