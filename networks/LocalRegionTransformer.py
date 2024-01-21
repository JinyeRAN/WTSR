import torch
import torch.nn as nn
from torch.nn.modules.activation import PReLU
from torch.nn.modules.conv import Conv2d
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Involution(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 2
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels // reduction_ratio, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=channels // reduction_ratio, out_channels=kernel_size**2 * self.groups, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

    def forward(self, x):
        weight = self.conv2(self.relu(self.conv1(x if self.stride == 1 else self.avgpool(x))))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowsReverse(nn.Module):
    def __init__(self, num_feature, windows, stride):
        super(WindowsReverse, self).__init__()
        self.Windows = windows
        self.process = nn.Sequential(
            nn.Conv2d(num_feature, num_feature, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.PReLU()
        )
        self.stride = stride

    def forward(self, x, H, W):
        HF = H//self.stride
        WF = W//self.stride
        B = int(x.shape[0] / (HF * WF / self.Windows / self.Windows))
        x = x.view(B, HF // self.Windows, WF // self.Windows, self.Windows, self.Windows, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, HF, WF, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.process(x) + x
        x = x.permute(0, 2, 3, 1)
        return x


class WindowsPartition(nn.Module):
    def __init__(self, num_feature, windows, stride):
        super(WindowsPartition, self).__init__()
        self.Inv = Involution(num_feature, 3, stride)
        self.Con = nn.Conv2d(num_feature, num_feature, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fusion = nn.Sequential(
            nn.Conv2d(num_feature * 3, num_feature, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Conv2d(num_feature, num_feature, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        self.windows = windows

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x_inv = self.Inv(x)
        x_conv = self.Con(x_inv)
        x = self.fusion(torch.cat([x, x_inv, x_conv], dim=1))
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = x.view(B, H // self.windows, self.windows, W // self.windows, self.windows, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.windows, self.windows, C)
        return windows


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.3, proj_drop=0.3):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, num_features, WindowsSize, windows_stride, num_heads, drop_path):
        super(SelfAttentionBlock, self).__init__()
        self.window_size = WindowsSize // windows_stride
        self.windowpartion = WindowsPartition(num_features, self.window_size, windows_stride)
        self.windowreverse = WindowsReverse(num_features, self.window_size, windows_stride)
        self.attn = WindowAttention(num_features, to_2tuple(self.window_size), num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.mlp = Mlp(in_features=num_features, hidden_features=num_features*4, act_layer=nn.GELU, drop=0.)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        shortcut = x
        x = self.norm1(x)
        x_windows = self.windowpartion(x)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = self.windowreverse(attn_windows, H, W)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x