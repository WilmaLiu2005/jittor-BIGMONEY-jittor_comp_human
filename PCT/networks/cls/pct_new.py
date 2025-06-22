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
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx) # [B, npoint, nsample, D]
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_xyz_norm, grouped_points_norm], dim=-1)
    return new_xyz, new_points

class StructureAwareAttention(nn.Module):
    """结构感知注意力 - 手动实现多头注意力"""
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
        
        # Offset-Attention 模块
        self.lbr = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True)
        )
        
        self.softmax = nn.Softmax(dim=-1)
        
    def execute(self, x):
        B, N, D = x.shape
        
        # 计算Q, K, V
        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = jt.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = self.softmax(scores)
        
        # 注意力加权特征
        fsa = jt.matmul(attention, v)  # (B, heads, N, head_dim)
        fsa = fsa.transpose(1, 2).contiguous().view(B, N, D)
        fsa = self.out_linear(fsa)
        
        # 差分 + LBR + 残差
        offset = x - fsa  # Fin - Fsa
        offset = offset.view(B * N, D)
        out = self.lbr(offset)  # LBR
        out = out.view(B, N, D)
        
        return out + x 
    
class CrossScaleAttention(nn.Module):
    """跨尺度注意力机制"""
    def __init__(self, channels, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels
        
        # 多尺度查询、键、值投影
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, 1, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU()
            ) for _ in range(num_scales)
        ])
        
        # 跨尺度融合
        self.cross_scale_fusion = nn.Sequential(
            nn.Conv1d(channels * num_scales, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )
        
        # 使用普通的初始化替代gauss
        self.attention_weights = jt.randn((num_scales, num_scales)) * 0.1
        self.softmax = nn.Softmax(dim=-1)
        
    def adaptive_resize(self, feat, target_size):
        """简单的自适应调整尺寸"""
        current_size = feat.shape[2]
        if current_size == target_size:
            return feat
        elif current_size > target_size:
            # 下采样
            step = current_size // target_size
            return feat[:, :, ::step][:, :, :target_size]
        else:
            # 上采样 (简单重复)
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
        target_size = 256  # 统一目标尺寸
        
        # 投影每个尺度的特征
        projected_features = []
        for i, (feat, proj) in enumerate(zip(scale_features, self.scale_projections)):
            projected_feat = proj(feat)
            # 调整到统一尺寸
            resized_feat = self.adaptive_resize(projected_feat, target_size)
            projected_features.append(resized_feat)
        
        # 跨尺度注意力权重
        attention_matrix = self.softmax(self.attention_weights)
        
        # 融合跨尺度特征
        fused_features = []
        for i, feat_i in enumerate(projected_features):
            weighted_feat = jt.zeros_like(feat_i)
            for j, feat_j in enumerate(projected_features):
                weighted_feat += attention_matrix[i, j] * feat_j
            fused_features.append(weighted_feat)
        
        # 最终融合
        all_features = concat(fused_features, dim=1)
        final_features = self.cross_scale_fusion(all_features)
        
        return final_features

class ProgressiveMultiScaleAttention(nn.Module):
    """渐进式多尺度注意力模块"""
    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        assert out_channels % num_heads == 0
        
        # 多头注意力组件
        self.q_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(in_channels, out_channels, 1)
        
        # 位置编码
        self.pos_encoding = nn.Conv1d(3, out_channels, 1, bias=False)
        
        # 输出投影
        self.output_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        # 软注意力
        self.softmax = nn.Softmax(dim=-1)
        
    def execute(self, x, xyz):
        B, C, N = x.shape
        
        # 添加位置编码
        pos_feat = self.pos_encoding(xyz)
        x_with_pos = x + pos_feat
        
        # 计算查询、键、值
        q = self.q_conv(x_with_pos).view(B, self.num_heads, self.head_dim, N)
        k = self.k_conv(x_with_pos).view(B, self.num_heads, self.head_dim, N)  
        v = self.v_conv(x_with_pos).view(B, self.num_heads, self.head_dim, N)
        
        # 计算注意力分数
        attention_scores = jt.matmul(q.transpose(2, 3), k) / np.sqrt(self.head_dim)
        attention_weights = self.softmax(attention_scores)
        
        # 应用注意力
        attended = jt.matmul(v, attention_weights.transpose(2, 3))
        attended = attended.view(B, self.out_channels, N)
        
        # 输出投影和残差连接
        output = self.output_conv(attended)
        if self.in_channels == self.out_channels:
            output = output + x
        
        output = self.relu(self.norm(output))
        return output

class FusedSkeletonTransformer(nn.Module):
    """简化的融合骨骼Transformer模型 - 移除未使用的骨骼结构模块"""
    def __init__(self, output_channels=256, num_joints=22):
        super().__init__()
        self.output_channels = output_channels
        self.num_joints = num_joints
        
        # 初始特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # 多尺度渐进式特征提取
        # Level 1: 全分辨率 (1024 points)
        self.level1_attention = nn.ModuleList([
            ProgressiveMultiScaleAttention(256, 256, num_heads=8)
            for _ in range(2)
        ])
        
        # Level 2: 中等分辨率 (512 points)  
        # 这里输入维度应该是259 (256特征 + 3坐标)
        self.level2_transform = nn.Conv1d(259, 384, 1, bias=False)
        self.level2_attention = nn.ModuleList([
            ProgressiveMultiScaleAttention(384, 384, num_heads=8)
            for _ in range(3)
        ])
        
        # Level 3: 低分辨率 (256 points)
        # 这里输入维度应该是387 (384特征 + 3坐标)
        self.level3_transform = nn.Conv1d(387, 512, 1, bias=False)
        self.level3_attention = nn.ModuleList([
            ProgressiveMultiScaleAttention(512, 512, num_heads=8)
            for _ in range(4)
        ])
        
        # 维度转换层
        self.dim_reduction_2 = nn.Conv1d(384, 256, 1, bias=False)
        self.dim_reduction_3 = nn.Conv1d(512, 256, 1, bias=False)
        
        # 跨尺度注意力融合
        self.cross_scale_attention = CrossScaleAttention(
            channels=256, num_scales=3
        )
        
        # 特征融合和输出
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # 全局特征提取
        self.global_feature_extractor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_channels)
        )
        
    def adaptive_pool_1d(self, x, target_size):
        """简单的自适应池化实现 - 使用Jittor可用的函数"""
        current_size = x.shape[2]
        if current_size == target_size:
            return x
        elif current_size > target_size:
            # 下采样
            if current_size % target_size == 0:
                # 整数倍下采样 - 使用手动reshape和max
                kernel_size = current_size // target_size
                B, C, N = x.shape
                # 重新组织为 [B, C, target_size, kernel_size] 然后取max
                x_reshaped = x.view(B, C, target_size, kernel_size)
                return jt.max(x_reshaped, dim=3)
            else:
                # 非整数倍的情况，使用均匀采样
                indices = jt.linspace(0, current_size-1, target_size).long()
                return x[:, :, indices]
        else:
            # 上采样 - 使用插值
            repeat_factor = target_size // current_size
            remainder = target_size % current_size
            upsampled = x.repeat_interleave(repeat_factor, dim=2)
            if remainder > 0:
                # 添加额外的点来达到目标尺寸
                extra_indices = jt.linspace(0, current_size-1, remainder).long()
                extra_points = x[:, :, extra_indices]
                upsampled = concat([upsampled, extra_points], dim=2)
            return upsampled[:, :, :target_size]
        
    def execute(self, vertices):
        """
        vertices: [B, 3, N] - 输入点云
        """
        B, _, N = vertices.shape
        
        # 初始特征提取
        x = self.initial_conv(vertices)  # [B, 256, N]
        xyz = vertices  # [B, 3, N]
        
        # Level 1: 全分辨率处理
        x1 = x
        for attention_layer in self.level1_attention:
            x1 = attention_layer(x1, xyz)
        
        # 下采样到Level 2 (512 points)
        xyz_512, x1_grouped = sample_and_group(512, 32, xyz.permute(0,2,1), x1.permute(0,2,1))
        # x1_grouped: [B, 512, 32, 259] (256特征 + 3坐标)

        # 重新组织特征维度
        x2 = x1_grouped.mean(dim=2)  # [B, 512, 259] 在邻域维度上取平均
        x2 = x2.permute(0, 2, 1)  # [B, 259, 512]

        x2 = self.level2_transform(x2)  # [B, 384, 512]
        
        for attention_layer in self.level2_attention:
            x2 = attention_layer(x2, xyz_512.permute(0,2,1))
        
        # 下采样到Level 3 (256 points)
        xyz_256, x2_grouped = sample_and_group(256, 32, xyz_512, x2.permute(0,2,1))
        # x2_grouped: [B, 256, 32, 387] (384特征 + 3坐标)

        x3 = x2_grouped.mean(dim=2)  # [B, 256, 387]
        x3 = x3.permute(0, 2, 1)  # [B, 387, 256]
        x3 = self.level3_transform(x3)  # [B, 512, 256]
        
        for attention_layer in self.level3_attention:
            x3 = attention_layer(x3, xyz_256.permute(0,2,1))
        
        # 统一特征维度用于跨尺度注意力
        x1_pooled = self.adaptive_pool_1d(x1, 256)  # [B, 256, 256]
        x2_reduced = self.dim_reduction_2(x2)  # [B, 256, 512] 
        x2_pooled = self.adaptive_pool_1d(x2_reduced, 256)  # [B, 256, 256]
        x3_reduced = self.dim_reduction_3(x3)  # [B, 256, 256]
        
        # 跨尺度注意力融合
        scale_features = [x1_pooled, x2_pooled, x3_reduced]
        fused_features = self.cross_scale_attention(scale_features)  # [B, 256, 256]
        
        # 特征融合
        fused_features = self.feature_fusion(fused_features)  # [B, 1024, 256]
        
        # 全局池化
        global_feat = jt.max(fused_features, 2)  # [B, 1024]
        
        # 全局特征
        global_output = self.global_feature_extractor(global_feat)  # [B, output_channels]
        
        return global_output

# 用于替换原有Point_Transformer的新模型
class EnhancedPoint_Transformer(nn.Module):
    """增强的点云Transformer，可直接替换原有的Point_Transformer"""
    def __init__(self, output_channels=256):
        super().__init__()
        self.fused_transformer = FusedSkeletonTransformer(
            output_channels=output_channels, 
            num_joints=22
        )
    
    def execute(self, x):
        """
        x: [B, 3, N] - 输入点云
        返回: [B, output_channels] - 特征向量
        """
        return self.fused_transformer(x) 