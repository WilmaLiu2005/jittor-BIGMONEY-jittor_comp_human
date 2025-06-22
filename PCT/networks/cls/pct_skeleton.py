import jittor as jt
from jittor import nn  
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
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_xyz_norm, grouped_points_norm], dim=-1)
    return new_xyz, new_points

class SkeletonStructurePrior(nn.Module):
    """骨骼结构先验知识模块"""
    def __init__(self, feat_dim, num_joints=22):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_joints = num_joints
        
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
        
        self.symmetry_pairs = [
            (1, 2),
            (4, 5),
            (7, 8),
            (10, 11),
            (13, 14),
            (16, 17),
            (18, 19),
            (20, 21),
        ]
        
    def compute_bone_constraints(self, joint_positions, global_features):
        """计算骨骼约束损失"""
        B = joint_positions.shape[0]
        joint_pos = joint_positions.view(B, self.num_joints, 3)
        
        predicted_lengths = self.bone_length_predictor(global_features)
        
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
            left_mirrored = jt.stack([-left_pos[:, 0], left_pos[:, 1], left_pos[:, 2]], dim=1)
            symmetry_loss += jt.mean((left_mirrored - right_pos) ** 2)
        
        return symmetry_loss / len(self.symmetry_pairs)

class SkeletonAttention(nn.Module):
    """专门针对骨骼预测的注意力机制"""
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
        
        self.cross_scale_fusion = nn.Sequential(
            nn.Conv1d(channels * num_scales, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )
        
        self.attention_weights = jt.randn((num_scales, num_scales)) * 0.1
        self.softmax = nn.Softmax(dim=-1)
        
    def adaptive_resize(self, feat, target_size):
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
        
        self.level1_attention = nn.ModuleList([
            ProgressiveSkeletonAttention(256, 256, num_heads=8)
            for _ in range(2)
        ])
        
        self.level2_transform = nn.Conv1d(259, 384, 1, bias=False)
        self.level2_attention = nn.ModuleList([
            ProgressiveSkeletonAttention(384, 384, num_heads=8)
            for _ in range(3)
        ])
        
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
        B, _, N = vertices.shape
        
        x = self.initial_conv(vertices)
        xyz = vertices
        
        x1 = x
        for attention_layer in self.level1_attention:
            x1 = attention_layer(x1, xyz)
        
        xyz_512, x1_grouped = sample_and_group(512, 32, xyz.permute(0,2,1), x1.permute(0,2,1))
        x2 = x1_grouped.mean(dim=2).permute(0, 2, 1)
        x2 = self.level2_transform(x2)
        
        for attention_layer in self.level2_attention:
            x2 = attention_layer(x2, xyz_512.permute(0,2,1))
        
        xyz_256, x2_grouped = sample_and_group(256, 32, xyz_512, x2.permute(0,2,1))
        x3 = x2_grouped.mean(dim=2).permute(0, 2, 1)
        x3 = self.level3_transform(x3)
        
        for attention_layer in self.level3_attention:
            x3 = attention_layer(x3, xyz_256.permute(0,2,1))
        
        x1_pooled = self.adaptive_pool_1d(x1, 256)
        x2_reduced = self.dim_reduction_2(x2)
        x2_pooled = self.adaptive_pool_1d(x2_reduced, 256)
        x3_reduced = self.dim_reduction_3(x3)
        
        scale_features = [x1_pooled, x2_pooled, x3_reduced]
        fused_features = self.cross_scale_attention(scale_features)
        
        fused_features = self.feature_fusion(fused_features)
        
        global_feat = jt.max(fused_features, 2)
        
        global_output = self.global_feature_extractor(global_feat)
        
        return global_output

class SkeletonPoint_Transformer(nn.Module):
    """骨骼预测专用Point Transformer"""
    def __init__(self, output_channels=256):
        super().__init__()
        self.output_channels = output_channels
        self.skeleton_transformer = SkeletonTransformer(
            output_channels=output_channels, 
            num_joints=22
        )
    
    def execute(self, x):
        """
        x: [B, 3, N] - 输入点云
        返回: [B, output_channels] - 特征向量
        """
        return self.skeleton_transformer(x)
