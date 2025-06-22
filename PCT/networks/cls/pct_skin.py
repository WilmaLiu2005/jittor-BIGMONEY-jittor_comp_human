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
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = concat([grouped_xyz_norm, grouped_points_norm], dim=-1)
    return new_xyz, new_points

def farthest_point_sample(xyz, npoint):
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
        
        self.pos_conv = nn.Sequential(
            nn.Conv1d(3, out_channels // 2, 1, bias=False),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(),
            nn.Conv1d(out_channels // 2, out_channels, 1, bias=False)
        )
        
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
        
        if self.skip_conv is not None:
            x = self.skip_conv(x)
            
        return x + pos_encoding

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
        
        self.softmax = nn.Softmax(dim=-1)
        
    def patch_partition(self, x):
        B, C, N = x.shape
        
        pad_size = (self.patch_size - N % self.patch_size) % self.patch_size
        if pad_size > 0:
            x = jt.concat([x, x[:, :, :pad_size]], dim=2)
            
        N_padded = x.shape[2]
        num_patches = N_padded // self.patch_size
        
        x_patches = x.view(B, C, num_patches, self.patch_size)
        return x_patches, pad_size
        
    def patch_merge(self, x_patches, pad_size, original_N):
        B, C, num_patches, patch_size = x_patches.shape
        x = x_patches.view(B, C, -1)
        
        if pad_size > 0:
            x = x[:, :, :-pad_size]
            
        return x[:, :, :original_N]
        
    def execute(self, x):
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
        B, C, N = x.shape
        shifted = jt.concat([x[:, :, shift_size:], x[:, :, :shift_size]], dim=2)
        return shifted
        
    def execute(self, x):
        out1 = self.attention1(x)
        
        shift_size = self.patch_size // 2
        x_shifted = self.shift_patches(x, shift_size)
        out2 = self.attention2(x_shifted)
        out2 = self.shift_patches(out2, -shift_size)
        
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
        x = self.pos_encoding(x, xyz)
        
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

class SkinWeightSmoothness(nn.Module):
    """皮肤权重平滑性约束"""
    def __init__(self, num_joints=22, k=8):
        super().__init__()
        self.num_joints = num_joints
        self.k = k
        
    def compute_smoothness_loss(self, skin_weights, xyz):
        B, N, _ = xyz.shape
        
        xyz_expanded = xyz.unsqueeze(2)
        xyz_tiled = xyz.unsqueeze(1)
        
        dist_matrix = jt.sum((xyz_expanded - xyz_tiled) ** 2, dim=3)
        
        _, knn_indices = jt.topk(-dist_matrix, k=self.k+1, dim=2)
        knn_indices = knn_indices[:, :, 1:]
        
        batch_indices = jt.arange(B).view(B, 1, 1).expand(B, N, self.k)
        neighbor_weights = skin_weights[batch_indices, knn_indices]
        
        current_weights = skin_weights.unsqueeze(2)
        smoothness_loss = jt.mean((current_weights - neighbor_weights) ** 2)
        
        return smoothness_loss
        
    def execute(self, skin_weights, xyz):
        return self.compute_smoothness_loss(skin_weights, xyz)

class PTv3SkinTransformer(nn.Module):
    """基于PTv3设计的皮肤权重预测Transformer"""
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
        
        self.smoothness_constraint = SkinWeightSmoothness(num_joints)
        
    def global_avg_pool_1d(self, x):
        """手动实现全局平均池化"""
        return jt.mean(x, dim=2)
        
    def grid_pool(self, x, xyz, scale_factor=2):
        B, C, N = x.shape
        target_N = N // scale_factor
        
        indices = jt.arange(0, N, scale_factor)[:target_N]
        
        downsampled_x = x[:, :, indices]
        downsampled_xyz = xyz[:, :, indices]
        
        return downsampled_x, downsampled_xyz
        
    def execute(self, vertices):
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
    
    def compute_smoothness_loss(self, skin_weights, xyz):
        return self.smoothness_constraint(skin_weights, xyz)

class SkinPoint_Transformer(nn.Module):
    """皮肤权重预测的Point Transformer"""
    def __init__(self, output_channels=256):
        super().__init__()
        self.skin_transformer = PTv3SkinTransformer(output_channels)
        
    def execute(self, x):
        """
        x: [B, 3, N] 输入点云坐标
        """
        return self.skin_transformer(x)
    
    def compute_smoothness_loss(self, skin_weights, xyz):
        return self.skin_transformer.compute_smoothness_loss(skin_weights, xyz)
