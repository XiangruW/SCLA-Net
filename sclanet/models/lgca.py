"""Localization-guided Cross Attention (LGCA).

Implements Section "Localization-guided Cross Attention" of the manuscript.
The SALT localization tokens act as queries and selectively aggregate
information from the structure-enhanced feature map::

    Q   = T W_Q,  K_f = F_hat W_K,  V_f = F_hat W_V
    A   = softmax(Q K_f^T / sqrt(d))        # (K, N)
    Z   = A V_f                             # (K, C)
    F_loc = A^T Z                           # (N, C)
    F_out = F_SCEA + reshape(F_loc)

The attention matrix is ``K x N`` so the module scales linearly with the
number of voxels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LGCA(nn.Module):
    """Localization-guided cross attention.

    Parameters
    ----------
    channels:
        Number of channels ``C`` of the structure-enhanced feature map.
    token_dim:
        Dimensionality of the SALT localization tokens (query dimension).
    out_channels:
        Number of output channels.  Defaults to ``channels``.
    residual:
        Add the aggregated localization information to the input feature.
    """

    def __init__(
        self,
        channels: int,
        token_dim: int = 256,
        out_channels: int | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__()
        out_channels = out_channels or channels
        self.channels = channels
        self.token_dim = token_dim
        self.out_channels = out_channels
        self.residual = residual

        self.w_q = nn.Linear(token_dim, token_dim, bias=False)
        self.w_k = nn.Linear(channels, token_dim, bias=False)
        self.w_v = nn.Linear(channels, out_channels, bias=False)
        self.proj = (
            nn.Identity()
            if out_channels == channels
            else nn.Conv3d(out_channels, channels, kernel_size=1, bias=False)
        )

    def forward(self, tokens: torch.Tensor, feature: torch.Tensor):
        """Apply localization-guided cross attention.

        Parameters
        ----------
        tokens: (B, K, token_dim) localization tokens.
        feature: (B, C, D, H, W) structure-enhanced feature map.

        Returns
        -------
        out: (B, C, D, H, W) fused feature map.
        attn: (B, K, N) localization-guided attention matrix.
        """
        b, c, d, h, w = feature.shape
        n = d * h * w

        q = self.w_q(tokens)  # (B, K, d)
        flat = feature.flatten(2).transpose(1, 2)  # (B, N, C)
        k = self.w_k(flat)  # (B, N, d)
        v = self.w_v(flat)  # (B, N, C_out)

        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]), dim=-1)
        z = torch.bmm(attn, v)  # (B, K, C_out)
        loc = torch.bmm(attn.transpose(1, 2), z)  # (B, N, C_out)

        loc = loc.transpose(1, 2).reshape(b, self.out_channels, d, h, w)
        loc = self.proj(loc)

        out = feature + loc if self.residual else loc
        return out, attn
