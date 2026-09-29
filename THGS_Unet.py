import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F


# Visual State Space Block (VSSB) - 优化为双方向扫描
class VSSB(nn.Module):
    def __init__(self, dim, hidden_dim=96):
        super(VSSB, self).__init__()
        self.linear_in = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear_out = nn.Conv2d(hidden_dim, dim, 1)
        self.ssm = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = self.linear_in(x)  # [B, hidden_dim, H, W]
        x_dw = self.dwconv(x_in)  # [B, hidden_dim, H, W]

        # 双方向选择性扫描
        x_flat = x_dw.view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, hidden_dim]
        # 方向 1：左→右（行优先）
        x_ssm1 = self.ssm(x_flat)  # [B, HW, hidden_dim]
        # 方向 2：右→左（反向行优先）
        x_flat_reverse = torch.flip(x_flat, dims=[1])  # 反转序列
        x_ssm2 = self.ssm(x_flat_reverse)  # [B, HW, hidden_dim]
        x_ssm2 = torch.flip(x_ssm2, dims=[1])  # 恢复原始顺序
        # 融合双方向结果
        x_ssm = (x_ssm1 + x_ssm2) / 2.0  # 平均融合
        x_ssm = x_ssm.permute(0, 2, 1).view(B, -1, H, W)  # [B, hidden_dim, H, W]

        x_out = self.norm(x_ssm.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_out = self.linear_out(x_out + x_in)
        return x_out + x


# Channel-Aware Mamba Block (CAMB) - 解码器增强模块
class CAMB(nn.Module):
    def __init__(self, dim):
        super(CAMB, self).__init__()
        self.vssb = VSSB(dim)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_vssb = self.vssb(x)
        attn = self.channel_attn(x_vssb)
        x_attn = x_vssb * attn
        x_out = self.norm(x_attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x_out + x


# HyperGraphModule - 原 TransUNet 的超图模块
class HyperGraphModule(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super(HyperGraphModule, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_features)

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        sim = torch.bmm(x_flat, x_flat.transpose(1, 2))  # [B, N, N]
        attn = F.softmax(sim, dim=-1)
        x_new = torch.bmm(attn, x_flat)  # [B, N, C]
        x_new = self.fc2(F.relu(self.fc1(x_new)))
        x_new = x_new.permute(0, 2, 1).view(B, C, H, W)
        return x + x_new


# TransformerBlock - 原 TransUNet 的 Transformer 模块
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(drop)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> (h w) b c')
        pos_embed = repeat(self.pos_embed, '1 1 c -> (h w) 1 c', h=H, w=W)
        x = x + pos_embed
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)
        x = rearrange(x, '(h w) b c -> b c h w', h=H, w=W)
        return x


# DoubleConv - 使用 GroupNorm 替换 BatchNorm
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),  # 替换 BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# Down - 下采样模块
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Up - 上采样模块
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Out - 输出层
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# 主模型：THGS_Unet
class THGS_Unet(nn.Module):
    def __init__(self, in_channel=3, n_class=16, embed_dim=1024, num_transformer_blocks=1):
        super(THGS_Unet, self).__init__()

        # 编码器
        self.conv1 = DoubleConv(in_channel, 64)
        self.down_conv2 = Down(64, 128)
        self.down_conv3 = Down(128, 256)
        self.down_conv4 = Down(256, 512)
        self.down_conv5 = Down(512, embed_dim)

        # 瓶颈层：Transformer + Mamba + HyperGraph
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=8, mlp_ratio=4)
            for _ in range(num_transformer_blocks)
        ])
        self.vssb_bottleneck = VSSB(embed_dim)  # 使用优化后的 VSSB
        self.hypergraph = HyperGraphModule(embed_dim)  # 保留超图模块

        # 解码器
        self.up_conv6 = Up(embed_dim, 512)
        self.camb6 = CAMB(512)
        self.up_conv7 = Up(512, 256)
        self.camb7 = CAMB(256)
        self.up_conv8 = Up(256, 128)
        self.camb8 = CAMB(128)
        self.up_conv9 = Up(128, 64)
        self.camb9 = CAMB(64)
        self.out = Out(64, n_class)

    def forward(self, x):
        # 编码器
        x1 = self.conv1(x)  # [B, 64, H, W]
        x2 = self.down_conv2(x1)  # [B, 128, H/2, W/2]
        x3 = self.down_conv3(x2)  # [B, 256, H/4, W/4]
        x4 = self.down_conv4(x3)  # [B, 512, H/8, W/8]
        x5 = self.down_conv5(x4)  # [B, 1024, H/16, W/16]

        # 瓶颈层
        for blk in self.transformer_blocks:
            x5 = blk(x5)
        #x5 = self.vssb_bottleneck(x5)
        x5 = self.hypergraph(x5)

        # 解码器
        x6 = self.up_conv6(x5, x4)  # [B, 512, H/8, W/8]
        x6 = self.camb6(x6)
        x7 = self.up_conv7(x6, x3)  # [B, 256, H/4, W/4]
        x7 = self.camb7(x7)
        x8 = self.up_conv8(x7, x2)  # [B, 128, H/2, W/2]
        x8 = self.camb8(x8)
        x9 = self.up_conv9(x8, x1)  # [B, 64, H, W]
        x9 = self.camb9(x9)
        out = self.out(x9)  # [B, n_class, H, W]

        return out


# 示例使用
if __name__ == "__main__":
    model = THGS_Unet(in_channel=3, n_class=16, embed_dim=1024)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Output shape: {out.shape}")  # Expected: [2, 16, 256, 256]