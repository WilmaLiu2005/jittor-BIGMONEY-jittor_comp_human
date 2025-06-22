import jittor as jt
from jittor import nn


def serialize_point_cloud(xyz, grid_size=0.05):
    B, N, C = xyz.shape
    coords = (xyz / grid_size).floor().int()
    keys = coords[..., 0] + coords[..., 1] * 1000 + coords[..., 2] * 1000000
    order = jt.argsort(keys, dim=-1)
    return order


def create_patches(x, patch_size):
    B, C, N = x.shape
    num_patches = N // patch_size
    x = x[:, :, :num_patches * patch_size]
    x = x.reshape((B, C, num_patches, patch_size)).permute(0, 2, 1, 3)
    return x


class SerializedAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.qkv = nn.Conv1d(dim, dim * 3, 1, bias=False)
        self.attn_proj = nn.Conv1d(dim, dim, 1, bias=False)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

    def execute(self, x):
        B, P, C, L = x.shape
        x = x.reshape(B * P, C, L)
        qkv = self.qkv(x).reshape(B * P, 3, self.heads, C // self.heads, L).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = jt.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = nn.softmax(attn, dim=-1)
        out = jt.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B * P, C, L)
        out = self.attn_proj(out).reshape(B, P, C, L)
        return out


class PTv3Block(nn.Module):
    def __init__(self, dim, patch_size=1024):
        super().__init__()
        self.xcpe = nn.Sequential(
            nn.Conv1d(3, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )
        self.attn = SerializedAttention(dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1)
        )
        self.patch_size = patch_size

    def execute(self, x, xyz):
        x = x + self.xcpe(xyz.permute(0, 2, 1))
        x_patches = create_patches(x, self.patch_size)
        x_patches = self.attn(x_patches)
        x = x_patches.reshape(x.shape)
        x = x + self.ffn(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def execute(self, x):
        return self.linear(x)


class PointTransformerV3(nn.Module):
    def __init__(self, output_channels=40):
        super().__init__()
        self.embedding = nn.Conv1d(3, 64, 1)

        self.encoder_channels = [64, 128, 256, 512]
        self.encoder_depths = [2, 2, 6, 2]

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(self.encoder_channels)):
            stage = nn.ModuleList([
                PTv3Block(self.encoder_channels[i])
                for _ in range(self.encoder_depths[i])
            ])
            self.encoders.append(stage)
            if i < len(self.encoder_channels) - 1:
                self.downsamples.append(
                    Downsample(self.encoder_channels[i], self.encoder_channels[i+1])
                )

        self.global_pool = lambda x: jt.max(x, dim=-1, keepdims=True)
        self.cls_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_channels)
        )

    def execute(self, x):
        xyz = x.permute(0, 2, 1)  # [B, N, 3]
        x = self.embedding(x)

        for i in range(len(self.encoders)):
            for block in self.encoders[i]:     # 手动逐个 block 执行
                x = block(x, xyz)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.global_pool(x).squeeze(-1)
        return self.cls_head(x)