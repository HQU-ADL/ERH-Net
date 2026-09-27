# @Version  :1.0
# @Author   :High_cold
# @File     :DIA_v2.py
# @Time     :2025/8/14 上午11:23
# -*- coding: utf-8 -*-
"""
DIA-Module (Deformable Interactive Attention) — Improved
- Faithfully implements Eq.(16)–(21) from the paper.
- Robust gradient scaling via custom autograd (ScaleGrad).
- Flexible Down/Up choices; residual gating + temperature for mask.
- Unifies MS/PAN branch behavior via `branch` argument.
"""

from typing import Literal, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utility: gradient scaling
# ---------------------------
class _ScaleGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def scale_grad(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity in forward; scales incoming grad by `scale` in backward."""
    if scale == 1.0:
        return x
    return _ScaleGradFn.apply(x, scale)


# ---------------------------
# Modulation generator Mar(F, λ)
# ---------------------------
class ModulationGenerator(nn.Module):
    """
    A tiny conv head that produces [0,1] modulation coefficients from a 1-ch map.
    We zero-init the conv weight so sigmoid(conv)=0.5 at start (stable neutral gating).
    """
    def __init__(self, kernel_size: int = 3, use_norm: bool = False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=True)
        if use_norm:
            self.norm = nn.BatchNorm2d(1)
        else:
            self.norm = None
        # zero init => logits ~ 0 => sigmoid ~ 0.5
        nn.init.constant_(self.conv.weight, 0.0)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: torch.Tensor, grad_scale: float) -> torch.Tensor:
        """
        x: (B,1,H,W)  -> returns modulation in [0,1] with gradient scaled by grad_scale.
        """
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        out = torch.sigmoid(out)
        out = scale_grad(out, grad_scale)
        return out


# ---------------------------
# Main: DeformableAttention (DIA-Module)
# ---------------------------
class DeformableAttention(nn.Module):
    """
    Deformable Interactive Attention (DIA-Module), improved.

    Args:
        stride:   legacy arg (kept for compatibility; not used unless down_type='conv').
        distortionmode: enable/disable modulation (Mar). If False, it reduces to simple
                        mean/max fusion → mask.
        branch: 'ms' or 'pan'. Controls (λ1, λ2) ordering per paper.
                MS: λ1=0.1 (from F2_m), λ2=0.4 (from F2_a).
                PAN: opposite (λ1=0.4, λ2=0.1).
        lambdas: override (λ1, λ2). If None, use defaults based on `branch`.
        down_type: 'pool' | 'conv'. 'pool' uses Avg/MaxPool2d(2); 'conv' mimics your original conv-down.
        up_mode: 'bilinear' (default) | 'nearest'.
        temperature: temperature τ for mask = sigmoid(logits / τ).
        residual: if True, use y = x * (1 + beta*(mask - 0.5)) for safer training.
        beta_init: initial value of learnable beta when residual=True.
        conv_kernel: kernel size of the final 2→1 conv for mask generation.
        use_mod_norm: add BN after Mar conv (usually False for stability on small batches).
    """
    def __init__(
        self,
        stride: int = 1,
        distortionmode: bool = True,
        branch: Literal['ms', 'pan'] = 'ms',
        lambdas: Optional[Tuple[float, float]] = None,
        down_type: Literal['pool', 'conv'] = 'pool',
        up_mode: Literal['bilinear', 'nearest'] = 'bilinear',
        temperature: float = 1.0,
        residual: bool = True,
        beta_init: float = 0.0,
        conv_kernel: int = 3,
        use_mod_norm: bool = False,
    ):
        super().__init__()
        assert conv_kernel in (1, 3, 5, 7)

        # λ settings per branch (paper Sec. III-C / Eq.(18))
        # MS: λ1 (from F2_m) = 0.1, λ2 (from F2_a) = 0.4
        # PAN: opposite
        if lambdas is None:
            if branch == 'ms':
                lambdas = (0.1, 0.4)
            else:  # 'pan'
                lambdas = (0.4, 0.1)
        self.lambda1, self.lambda2 = lambdas
        self.distortionmode = distortionmode
        self.branch = branch
        self.temperature = float(temperature)
        self.residual = residual
        self.up_mode = up_mode
        self.down_type = down_type

        # Down: either pooling (default) or stride-2 convs to mimic original code
        if down_type == 'conv':
            self.down_avg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
            self.down_max = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        else:
            self.avg_pool2d = nn.AvgPool2d(kernel_size=2, stride=2)
            self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Modulation generators Mar(·, λ)
        if distortionmode:
            self.mar_from_avg = ModulationGenerator(kernel_size=3, use_norm=use_mod_norm)  # produces coeffs to mod F2_m
            self.mar_from_max = ModulationGenerator(kernel_size=3, use_norm=use_mod_norm)  # produces coeffs to mod F2_a

        # Mask head: Conv(Concat(Up(F3_a), Up(F3_m))) -> 1ch
        pad = (conv_kernel - 1) // 2
        self.mask_conv = nn.Conv2d(2, 1, kernel_size=conv_kernel, padding=pad)

        # Residual gating scale
        if residual:
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        else:
            self.register_parameter('beta', None)

        # Init
        nn.init.kaiming_normal_(self.mask_conv.weight, nonlinearity='linear')
        if self.mask_conv.bias is not None:
            nn.init.constant_(self.mask_conv.bias, 0.0)

    def _down(self, x_mean: torch.Tensor, x_max: torch.Tensor):
        if self.down_type == 'conv':
            d_mean = self.down_avg(x_mean)
            d_max = self.down_max(x_max)
        else:
            d_mean = self.avg_pool2d(x_mean)
            d_max = self.max_pool2d(x_max)
        return d_mean, d_max

    def _up_to(self, x: torch.Tensor, size_hw: Tuple[int, int]):
        return F.interpolate(x, size=size_hw, mode=self.up_mode, align_corners=False if self.up_mode == 'bilinear' else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Eq.(16): channel compression to 1ch maps with global smoothness / local texture properties
        # In practice (and your original implementation), this is mean/max across channel dim.
        F1_a = torch.mean(x, dim=1, keepdim=True)    # (B,1,H,W)
        F1_m, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)

        # Eq.(17): Down(·)
        F2_a, F2_m = self._down(F1_a, F1_m)  # (B,1,H/2,W/2)

        # Eq.(18): cross modulation with gradient scaling
        if self.distortionmode:
            # MS branch default: λ1 for coeffs generated FROM F2_m (to mod F2_a), λ2 for coeffs FROM F2_a (to mod F2_m)
            # PAN branch: opposite (handled by lambdas assignment in __init__)
            coeff_from_max = self.mar_from_max(F2_m, grad_scale=self.lambda1)  # -> mod F2_a
            coeff_from_avg = self.mar_from_avg(F2_a, grad_scale=self.lambda2)  # -> mod F2_m

            F3_a = coeff_from_max * F2_a
            F3_m = coeff_from_avg * F2_m
        else:
            # no modulation: pass-through
            F3_a, F3_m = F2_a, F2_m

        # Eq.(19)-(20): Up both, concat, conv -> mask in [0,1]
        F4_a = self._up_to(F3_a, (H, W))
        F4_m = self._up_to(F3_m, (H, W))
        mask_logits = self.mask_conv(torch.cat([F4_a, F4_m], dim=1))
        if self.temperature != 1.0:
            mask = torch.sigmoid(mask_logits / self.temperature)
        else:
            mask = torch.sigmoid(mask_logits)  # (B,1,H,W)

        # Eq.(21): apply mask (with safer residual gating if enabled)
        if self.residual:
            out = x * (1.0 + self.beta * (mask - 0.5))
        else:
            out = x * mask

        return out


# ---------------------------
# Convenience alias for PAN branch
# ---------------------------
class DeformableAttentionPAN(DeformableAttention):
    """Same as DeformableAttention but defaults to PAN branch (λ swapped)."""
    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault('branch', 'pan')
        super().__init__(**kwargs)


# ---------------------------
# (Optional) simple spatial/channel attentions kept for completeness
# ---------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        mask = torch.sigmoid(self.conv(a))
        return x * mask


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.avg_pool(x)
        a = self.fc2(self.relu(self.fc1(a)))
        w = torch.sigmoid(a)
        return x * w


# ---------------------------
# Quick self-test
# ---------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 32, 128, 128
    x = torch.randn(B, C, H, W)

    # MS branch (default) with modulation, pooling Down, bilinear Up, residual gating
    dia_ms = DeformableAttention(distortionmode=True, branch='ms')
    y_ms = dia_ms(x)
    print("MS out:", y_ms.shape)

    # PAN branch (λ swapped), nearest Up, conv Down (to mimic legacy)
    dia_pan = DeformableAttentionPAN(distortionmode=True, down_type='conv', up_mode='nearest')
    y_pan = dia_pan(x)
    print("PAN out:", y_pan.shape)

    # Ablate modulation
    dia_nomod = DeformableAttention(distortionmode=False, residual=False)
    y_nm = dia_nomod(x)
    print("No-mod out:", y_nm.shape)
