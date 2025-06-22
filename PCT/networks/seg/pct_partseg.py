import jittor as jt 
from jittor import nn 
from jittor.contrib import concat 
import numpy as np 
import math 


class Point_Transformer_partseg(nn.Module):
    def __init__(self, part_num=50):
        super(Point_Transformer_partseg, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(scale=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(scale=0.2))

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def execute(self, x, cls_label):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = concat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max = jt.max(x, 2)
        x_avg = jt.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(batch_size,16,1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = concat((x_max_feature, x_avg_feature, cls_label_feature), 1) # 1024 + 64
        x = concat((x, x_global_feature), 1) # 1024 * 3 + 64 
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        return x



class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.conv.weight = self.k_conv.conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # sin-cos 编码参数
        self.num_pos_feats = channels  # 可调：也可设为 channels // 2
    
    # 加入位置编码
    def sinusoidal_positional_encoding(self, coords, num_feats):
        """
        coords: [B, 3, N]
        return: [B, num_feats, N]
        """
        B, _, N = coords.shape
        # 生成不同的 频率尺度（wave-length），确保每个维度有不同的频率
        div_term = jt.exp(jt.arange(0, num_feats, 2).float() * -(math.log(10000.0) / num_feats))
        div_term = div_term.view(1, 1, -1, 1)  # [1, 1, C//2, 1]
        coords = coords.unsqueeze(2)  # [B, 3, 1, N]

        pe = jt.zeros((B, 3, num_feats, N))
        for i in range(3):  # xyz，第奇数维用 cos，第偶数维用 sin
            pe[:, i, 0::2, :] = jt.sin(coords[:, i:i+1, :, :] * div_term)
            pe[:, i, 1::2, :] = jt.cos(coords[:, i:i+1, :, :] * div_term)
        pe = pe.view(B, -1, N)  # [B, 3*num_feats, N]
        return pe

    def execute(self, x):
        B, C, N = x.shape

        coords = x[:, :3, :]  # [B, 3, N]

        # 生成 sin-cos 编码
        pos_enc = self.sinusoidal_positional_encoding(coords, num_feats=C // 3)
        # 截断或投影编码维度以匹配输入特征
        if pos_enc.shape[1] > C:
            pos_enc = pos_enc[:, :C, :]
        elif pos_enc.shape[1] < C:
            pad = jt.zeros((B, C - pos_enc.shape[1], N))
            pos_enc = jt.concat([pos_enc, pad], dim=1)

        # 加入位置编码
        x_pe = x + pos_enc

        # 标准注意力路径
        x_q = self.q_conv(x_pe).permute(0, 2, 1)
        x_k = self.k_conv(x_pe)
        x_v = self.v_conv(x)

        energy = jt.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        x_r = jt.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
