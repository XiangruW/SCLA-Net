"""Basic building blocks of the SCLA-Net lightweight backbone.

Every convolutional block follows the design described in Section
"Lightweight Backbone Network" of the manuscript:

* depthwise separable convolution (3x3x3 depthwise + 1x1x1 pointwise),
* layer normalization,
* Leaky ReLU activation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm3d(nn.Module):
    """Channel-wise layer normalization for channel-first 5D tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, D, H, W) -> (B, D, H, W, C) -> normalize -> back
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3)


def make_norm(norm: str, num_channels: int) -> nn.Module:
    norm = (norm or "layer").lower()
    if norm in ("layer", "layernorm", "ln"):
        return LayerNorm3d(num_channels)
    if norm in ("instance", "in"):
        return nn.InstanceNorm3d(num_channels, affine=True)
    if norm in ("batch", "bn"):
        return nn.BatchNorm3d(num_channels)
    raise ValueError(f"unknown norm: {norm}")


def make_act(act: str) -> nn.Module:
    act = (act or "leakyrelu").lower()
    if act in ("leakyrelu", "lrelu", "leaky"):
        return nn.LeakyReLU(0.01, inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act in ("gelu",):
        return nn.GELU()
    if act in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"unknown activation: {act}")


class DepthwiseSeparableConv3d(nn.Module):
    """3x3x3 depthwise convolution followed by a 1x1x1 pointwise convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: str = "layer",
        act: str = "leakyrelu",
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = make_norm(norm, out_channels)
        self.act = make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        return self.act(x)


class ConvNormAct3d(nn.Module):
    """Plain (1x1x1 by default) convolution with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        norm: str = "none",
        act: str = "none",
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.norm = make_norm(norm, out_channels) if norm and norm != "none" else nn.Identity()
        self.act = make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualDSBlock(nn.Module):
    """Two depthwise separable convolutions with an optional identity shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: str = "layer",
        act: str = "leakyrelu",
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv3d(in_channels, out_channels, stride, norm, act)
        self.conv2 = DepthwiseSeparableConv3d(out_channels, out_channels, 1, norm, act)
        self.use_residual = residual and (in_channels == out_channels) and stride == 1
        self.proj = (
            None
            if self.use_residual or not residual
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_norm(norm, out_channels),
            )
        )
        self.act = make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if (self.use_residual or not self.proj) else self.proj(x)
        out = self.conv2(self.conv1(x))
        if self.use_residual or self.proj is not None:
            out = self.act(out + identity)
        return out


class Downsample3d(nn.Module):
    """Stride-2 depthwise separable convolution used between encoder stages."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "layer") -> None:
        super().__init__()
        self.block = DepthwiseSeparableConv3d(in_channels, out_channels, stride=2, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
