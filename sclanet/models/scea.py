"""Structural Context Enhanced Attention (SCEA).

Implements the module described in Section "Structural Context Enhanced
Attention (SCEA)" of the manuscript.  SCEA consists of

1. *Structural Context Fusion* (SCF): directional pooling along the three
   orthogonal anatomical axes (depth / height / width) followed by lightweight
   depthwise convolution.  The three broadcast representations are fused with
   adaptive (softmax) weights.
2. *Structural Context Guided Attention Recalibration*: the fused context is
   injected into the key space of a query-key-value attention step; the
   resulting attention map recalibrates the input representation.

Implementation notes
--------------------
* The reduction ratio of the manuscript is applied to the query/key/value and
  to the projection applied to the recalibrated feature, so that the module
  stays lightweight at the bottleneck (``C / reduction`` channels).
* Full volumetric (``N x N``) attention is not tractable for 96^3 inputs, and
  the manuscript does not state a spatial reduction.  The attention is
  therefore computed on an adaptively pooled grid of at most
  ``attn_grid`` voxels per axis and the resulting map is trilinearly
  upsampled back to the input resolution.  With ``attn_grid`` larger than or
  equal to the feature size the computation is *exact*, which is the case for
  the bottleneck of 96^3 patches (6^3 voxels).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DepthwiseSeparableConv3d, make_act


class SCEA(nn.Module):
    """Structural Context Enhanced Attention module.

    Parameters
    ----------
    channels:
        Number of input (and output) channels.
    reduction:
        Channel reduction ratio applied to the attention projections.
    attn_grid:
        Maximum number of voxels per axis used to compute the attention.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        attn_grid: int = 8,
        min_width: int = 4,
        act: str = "leakyrelu",
    ) -> None:
        super().__init__()
        width = max(channels // max(reduction, 1), min_width)
        self.channels = channels
        self.width = width
        self.attn_grid = int(attn_grid)

        # ---- Structural Context Fusion -------------------------------------
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        self.dw_d = nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw_h = nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw_w = nn.Conv1d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.fuse = nn.Conv3d(3 * channels, 3, kernel_size=1, bias=True)

        # ---- Structural Context Guided Attention Recalibration --------------
        self.theta_q = nn.Conv3d(channels, width, 1, bias=False)
        self.theta_k = nn.Conv3d(channels, width, 1, bias=False)
        self.theta_v = nn.Conv3d(channels, width, 1, bias=False)
        self.theta_p = nn.Conv3d(channels, width, 1, bias=False)
        self.out = DepthwiseSeparableConv3d(width, channels, stride=1, act=act)

        for conv in (self.theta_q, self.theta_k, self.theta_v, self.theta_p):
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")

    # ------------------------------------------------------------------ #
    def structural_context_fusion(self, x: torch.Tensor) -> torch.Tensor:
        """Directional pooling + depthwise conv + adaptive fusion."""
        b, c, d, h, w = x.shape

        # (B, C, D, 1, 1) -> (B, C, D) -> depthwise 1D conv along the axis
        cd = self.dw_d(self.pool_d(x).reshape(b, c, d)).view(b, c, d, 1, 1)
        ch = self.dw_h(self.pool_h(x).reshape(b, c, h)).view(b, c, 1, h, 1)
        cw = self.dw_w(self.pool_w(x).reshape(b, c, w)).view(b, c, 1, 1, w)

        # broadcast the three directional representations to the volume
        cd = cd.expand(b, c, d, h, w)
        ch = ch.expand(b, c, d, h, w)
        cw = cw.expand(b, c, d, h, w)

        g = torch.softmax(self.fuse(torch.cat([cd, ch, cw], dim=1)), dim=1)
        return g[:, 0:1] * cd + g[:, 1:2] * ch + g[:, 2:3] * cw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape

        c_scf = self.structural_context_fusion(x)

        # ---- guided attention ------------------------------------------- #
        q = self.theta_q(x)
        k = self.theta_k(c_scf)
        v = self.theta_v(x)

        grid = tuple(min(int(s), self.attn_grid) for s in (d, h, w))
        n = grid[0] * grid[1] * grid[2]
        if grid != (d, h, w):
            q = F.adaptive_avg_pool3d(q, grid)
            k = F.adaptive_avg_pool3d(k, grid)
            v = F.adaptive_avg_pool3d(v, grid)

        qf = q.flatten(2)  # (B, C', n)
        kf = k.flatten(2)
        vf = v.flatten(2)

        attn = torch.softmax(
            torch.bmm(qf.transpose(1, 2), kf) / math.sqrt(qf.shape[1]), dim=-1
        )  # (B, n, n)
        fa = torch.bmm(attn, vf.transpose(1, 2))  # (B, n, C')
        fa = fa.transpose(1, 2).view(b, self.width, *grid)

        if grid != (d, h, w):
            fa = F.interpolate(fa, size=(d, h, w), mode="trilinear", align_corners=False)
        a_map = torch.sigmoid(fa)

        recalibrated = a_map * self.theta_p(x)
        return self.out(recalibrated)
