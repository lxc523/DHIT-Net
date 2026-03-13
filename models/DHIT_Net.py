import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import models.configs_DHIT_Net as configs
from einops import rearrange

import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Basic blocks
class CA(nn.Module):
    """Channel attention for 3D feature maps.
    Args:
        num_feat: number of channels.
        squeeze_factor: reduction ratio for the bottleneck.
    """
    def __init__(self, num_feat, squeeze_factor=16):
        super(CA, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

class Mlp(nn.Module):
    """Two-layer MLP with activation and dropout."""
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

# Core modules
class ConvStem3D(nn.Module):
    """3D conv stem with optional downsampling and residual shortcut."""
    def __init__(self, in_ch: int, out_ch: int, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act   = nn.GELU()

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)

        if downsample or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.act(x)

class Local_Context_Refiner(nn.Module):
    """Local context refiner using depthwise convs and channel attention."""
    def __init__(self, dim, expansion_ratio=4, drop=0., act_layer=nn.GELU, use_large_kernel=True):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.use_large_kernel = use_large_kernel

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()

        if self.use_large_kernel:
            self.dwc_local = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim)
            self.dwc_large = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=7, padding=9, dilation=3, groups=hidden_dim)
        else:
            self.dwc_compact = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)

        self.ca = CA(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W, T):
        B, L, C = x.shape

        x = self.fc1(x)
        x = self.act(x)

        x_spatial = x.transpose(1, 2).view(B, -1, H, W, T)

        if self.use_large_kernel:
            feat = self.dwc_local(x_spatial)
            feat = self.dwc_large(feat)
        else:
            feat = self.dwc_compact(x_spatial)

        feat = self.ca(feat)
        x_spatial = x_spatial + feat

        x = x_spatial.flatten(2).transpose(1, 2)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class HCA(nn.Module):
    """Hybrid context attention block (window attention + local refiner)."""
    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, rpe=True, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, use_large_kernel=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        self.laff = Local_Context_Refiner(
            dim=dim, expansion_ratio=mlp_ratio, drop=drop,
            act_layer=act_layer, use_large_kernel=use_large_kernel
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.H = None; self.W = None; self.T = None

    def forward(self, x, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        shortcut = x
        x1 = self.norm1(x)
        x1_reshaped = x1.view(B, H, W, T, C)

        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x1_reshaped = nnf.pad(x1_reshaped, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = x1_reshaped.shape

        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x1_reshaped, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x1_reshaped
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)

        if min(self.shift_size) > 0:
            attn_out = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            attn_out = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_h > 0:
            attn_out = attn_out[:, :H, :W, :T, :].contiguous()

        attn_out = attn_out.view(B, H * W * T, C)

        x = shortcut + self.drop_path(attn_out)

        x2 = self.laff(self.norm2(x), H, W, T)
        x = x + self.drop_path(x2)

        return x

# Interaction block
class DMIP(nn.Module):
    """Dual-modality interaction block for moving/fixed features."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        self.dw3 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.dw7 = nn.Conv3d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)

        self.pw_attn = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.Sigmoid()
        )

    def _get_attention_map(self, x):
        summed_feat = x + self.dw3(x) + self.dw5(x) + self.dw7(x)

        attn_map = self.pw_attn(summed_feat)

        return attn_map

    def forward(self, feat_m, feat_f):
        """Compute cross-gated features.
        Args:
            feat_m: moving features.
            feat_f: fixed features.
        Returns:
            Updated moving and fixed features.
        """
        att_m = self._get_attention_map(feat_m)

        att_f = self._get_attention_map(feat_f)

        out_f_new = att_m * feat_f

        out_m_new = att_f * feat_m

        return feat_m + out_m_new, feat_f + out_f_new

class SGTM_C(nn.Module):
    """Shallow channel fusion transform for paired features."""
    def __init__(self, channels: int):
        super().__init__()
        self.fuse_conv = nn.Conv3d(2 * channels, channels, kernel_size=1, bias=False)
        self.fuse_bn   = nn.BatchNorm3d(channels)
        self.spatial_att = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.out_bn = nn.BatchNorm3d(channels)

    def forward(self, feat_m, feat_f):
        fused = torch.cat([feat_m, feat_f], dim=1)
        fused = self.fuse_conv(fused)
        fused = self.fuse_bn(fused)
        avg_map = fused.mean(dim=1, keepdim=True)
        max_map, _ = fused.max(dim=1, keepdim=True)
        att = self.spatial_att(torch.cat([avg_map, max_map], dim=1))
        out = fused * att + fused
        return self.out_bn(out)

class SGTM_W(nn.Module):
    """Window-level transform with spatial gating."""
    def __init__(self, channels: int):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.out_bn = nn.BatchNorm3d(channels)

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        att = self.spatial_att(torch.cat([avg_map, max_map], dim=1))
        return self.out_bn(x * att + x)

# Transformer backbone
def window_partition(x, window_size):
    """Split a 5D tensor into non-overlapping 3D windows.
    Args:
        x: (B, H, W, T, C) tensor.
        window_size: tuple of window sizes.
    Returns:
        windows: (num_windows*B, ws, ws, ws, C).
    """
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def window_reverse(windows, window_size, H, W, L):
    """Reconstruct a 5D tensor from window blocks.
    Args:
        windows: window blocks.
        window_size: tuple of window sizes.
        H, W, T: target spatial sizes.
    Returns:
        x: (B, H, W, T, C).
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x

class WindowAttention(nn.Module):
    """Windowed multi-head attention with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))
        coords_flatten = torch.flatten(coords, 1)
        if rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PatchMerging(nn.Module):
    """Merge 2x2x2 neighborhoods to reduce resolution and expand channels."""
    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim

        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        B, L, C = x.shape
        assert L == H * W * T, f"Input feature has wrong size. Expect: {H*W*T}, Got: {L}"

        x = x.view(B, H, W, T, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = x.view(B, -1, 8 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """Stack of HCA blocks with optional downsampling."""
    def __init__(self, dim, depth, num_heads, window_size=(7, 7, 7), mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, rpe=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, pat_merg_rf=2, use_large_kernel=True):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            HCA(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_large_kernel=use_large_kernel
            ) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=pat_merg_rf)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]), slice(-self.window_size[2], -self.shift_size[2]), slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T

class SwinTransformer(nn.Module):
    """Hierarchical transformer backbone returning multi-scale features."""
    def __init__(self, pretrain_img_size=224, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=(7, 7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False, rpe=True,
                 out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False, pat_merg_rf=2):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.patch_embed = None

        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size_3 = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size_3[0], pretrain_img_size[1] // patch_size_3[1], pretrain_img_size[2] // patch_size_3[2]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            use_lka_strategy = True if i_layer < 2 else False

            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                rpe=rpe,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pat_merg_rf=pat_merg_rf,
                use_large_kernel=use_lka_strategy
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()
        self.init_weights()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            pass

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x, pre_embedded=True):
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        x = x.flatten(2).transpose(1, 2)

        if self.ape:
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = x + absolute_pos_embed.flatten(2).transpose(1, 2)

        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
        return outs

# Decoder blocks
class Conv3dReLU(nn.Sequential):
    """Conv-Norm-Act helper block."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        relu = nn.LeakyReLU(inplace=True)
        nm = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.InstanceNorm3d(out_channels)
        super(Conv3dReLU, self).__init__(conv, nm, relu)

class PixelShuffle3d(nn.Module):
    """3D pixel shuffle upsampling."""
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3
        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch_size, nOut, in_depth * self.scale, in_height * self.scale, in_width * self.scale)

class ConvergeHead(nn.Module):
    """Upsampling head using conv + pixel shuffle."""
    def __init__(self, in_dim, up_ratio, kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, (up_ratio**3)*in_dim, kernel_size, 1, 1)
        self.ps = PixelShuffle3d(up_ratio)
    def forward(self, x):
        return self.ps(self.conv(x))

class SR(nn.Module):
    """Upsample and refine with optional skip connection."""
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.use_skip = skip_channels > 0
        self.up = ConvergeHead(in_channels, 2, 3)
        in_ch = in_channels + skip_channels if self.use_skip else in_channels
        self.conv1 = Conv3dReLU(in_ch, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv3dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.use_skip and skip is not None:
            if skip.shape[2:] != x.shape[2:]:
                skip = nnf.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))

class RegistrationHead(nn.Module):
    """Regression head for dense displacement fields."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv.weight.shape))
        self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))
    def forward(self, x):
        return self.conv(x)

class SpatialTransformer(nn.Module):
    """Differentiable warper for 3D volumes."""
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return nnf.grid_sample(src, new_locs, align_corners=False, mode=self.mode)

# DHIT_Net model
class DHIT_Net(nn.Module):
    """Main registration network; returns warped image and flow."""
    def __init__(self, config):
        super(DHIT_Net, self).__init__()
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim
        c0 = embed_dim // 2

        self.stem = ConvStem3D(in_ch=1, out_ch=c0, downsample=True)
        self.mams1 = DMIP(channels=c0)

        self.stage2_conv = ConvStem3D(in_ch=c0, out_ch=c0*2, downsample=True)
        self.mams2 = DMIP(channels=c0*2)

        self.ftm_c = SGTM_C(channels=c0)

        self.embed_conv = nn.Conv3d(in_channels=4*c0, out_channels=embed_dim, kernel_size=1, bias=False)

        self.transformer = SwinTransformer(
            patch_size=config.patch_size, embed_dim=config.embed_dim,
            depths=config.depths, num_heads=config.num_heads, window_size=config.window_size,
            mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate, ape=config.ape, rpe=config.rpe,
            use_checkpoint=config.use_checkpoint, out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf
        )

        self.ftm_w1 = SGTM_W(channels=embed_dim * 4)
        self.ftm_w2 = SGTM_W(channels=embed_dim * 2)
        self.ftm_w3 = SGTM_W(channels=embed_dim)

        self.up0 = SR(embed_dim * 8, embed_dim * 4, skip_channels=embed_dim * 4 if self.if_transskip else 0, use_batchnorm=False)
        self.up1 = SR(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if self.if_transskip else 0, use_batchnorm=False)
        self.up2 = SR(embed_dim * 2, embed_dim, skip_channels=embed_dim if self.if_transskip else 0, use_batchnorm=False)
        self.up3 = SR(embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if self.if_convskip else 0, use_batchnorm=False)
        self.up4 = SR(embed_dim // 2, config.reg_head_chan, skip_channels=config.reg_head_chan if self.if_convskip else 0, use_batchnorm=False)

        self.f5_stem = Conv3dReLU(1, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.ftm_f5 = SGTM_C(channels=config.reg_head_chan)

        self.reg_head = RegistrationHead(in_channels=config.reg_head_chan, out_channels=3)
        self.spatial_trans = SpatialTransformer(config.img_size)

    def forward(self, x):
        source = x[:, 0:1, ...]
        moving = x[:, 0:1, ...]
        fixed  = x[:, 1:2, ...]
        f1_m = self.stem(moving)
        f1_f = self.stem(fixed)
        f1_m, f1_f = self.mams1(f1_m, f1_f)
        f4 = self.ftm_c(f1_m, f1_f) if self.if_convskip else None
        f2_m = self.stage2_conv(f1_m)
        f2_f = self.stage2_conv(f1_f)
        f2_m, f2_f = self.mams2(f2_m, f2_f)
        feat_cat = torch.cat([f2_m, f2_f], dim=1)
        x_embed = self.embed_conv(feat_cat)

        out_feats = self.transformer(x_embed, pre_embedded=True)

        if self.if_transskip:
            f1 = self.ftm_w1(out_feats[-2])
            f2 = self.ftm_w2(out_feats[-3])
            f3 = self.ftm_w3(out_feats[-4])
        else:
            f1 = f2 = f3 = None

        if self.if_convskip:
            feat_m0 = self.f5_stem(moving)
            feat_f0 = self.f5_stem(fixed)
            f5 = self.ftm_f5(feat_m0, feat_f0)
        else:
            f5 = None

        x = self.up0(out_feats[-1], f1)
        x = self.up1(x, f2)
        x = self.up2(x, f3)
        x = self.up3(x, f4)
        x = self.up4(x, f5)

        flow = self.reg_head(x)
        out = self.spatial_trans(source, flow)

        return out, flow

CONFIGS = {
    'DHIT_Net': configs.get_DHIT_Net_config(),
    'DHIT_Net-Small': configs.get_DHIT_NetSmall_config(),
}
