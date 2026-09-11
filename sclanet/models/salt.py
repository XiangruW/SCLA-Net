"""Structure-aware Localization Token Generator (SALT).

Implements Section "Structure-aware Localization Token Generator (SALT)" of
the manuscript:

* *Diversity-aware Anatomical Center Selection*: a lightweight response head
  predicts an anatomical response map, and K anatomical centers are selected
  greedily by maximizing

      O_d(x) = O_norm(x) * (1 + alpha * S_i) * R(x)

  where ``S_i`` is the local structural variation inside a 3x3x3
  neighbourhood and ``R(x) = 1 - exp(-d(x)^2 / (2 sigma^2))`` penalises
  spatial redundancy with respect to the already selected centers.
* *Adaptive Local Aggregation*: features inside a (2r+1)^3 neighbourhood of
  every center are aggregated with softmax weights produced by a lightweight
  MLP, a positional embedding of the center coordinates is added, and the
  result is projected to the token dimension.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DepthwiseSeparableConv3d


class PositionalEncoding3D(nn.Module):
    """Sinusoidal + MLP positional embedding of normalized 3D coordinates."""

    def __init__(self, channels: int, num_frequencies: int = 4) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        in_features = 3 + 3 * 2 * num_frequencies  # coords + sin/cos bands
        self.mlp = nn.Sequential(
            nn.Linear(in_features, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """``coords`` is (B, K, 3) with values in [-1, 1]."""
        features = [coords]
        for i in range(self.num_frequencies):
            freq = 2.0 ** i * math.pi
            features.append(torch.sin(freq * coords))
            features.append(torch.cos(freq * coords))
        return self.mlp(torch.cat(features, dim=-1))


class SALT(nn.Module):
    """Structure-aware Localization Token Generator.

    Parameters
    ----------
    channels:
        Number of channels of the deep feature map ``F_d``.
    num_tokens:
        Number of anatomical centers / localization tokens ``K``.
    token_dim:
        Dimensionality of the produced tokens.
    radius:
        Aggregation radius ``r`` (``r = 1`` gives a 3x3x3 neighbourhood).
    tau, sigma, alpha:
        Threshold, spatial weighting range and local-variation weight used by
        the center selection score.
    max_candidates:
        Safety limit on the size of the candidate pool.  ``None`` uses all
        voxels whose normalized response exceeds ``tau``.
    """

    def __init__(
        self,
        channels: int,
        num_tokens: int = 16,
        token_dim: int = 256,
        radius: int = 1,
        tau: float = 0.3,
        sigma: float = 2.0,
        alpha: float = 0.5,
        hidden: int = 64,
        max_candidates: int | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.radius = radius
        self.tau = tau
        self.sigma = sigma
        self.alpha = alpha
        self.max_candidates = max_candidates

        # response prediction head psi(.) - kept lightweight (depthwise separable)
        self.head = nn.Sequential(
            DepthwiseSeparableConv3d(
                channels, max(channels // 2, 8), stride=1, norm="layer", act="leakyrelu"
            ),
            nn.Conv3d(max(channels // 2, 8), 1, kernel_size=1, bias=True),
        )

        # lightweight MLP g(.) producing one scalar per neighbouring feature
        self.g = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        self.pos_embed = PositionalEncoding3D(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, token_dim),
            nn.LayerNorm(token_dim),
        )

        self.offsets = self._neighbourhood_offsets(radius)
        self.num_neighbours = len(self.offsets)

    @staticmethod
    def _neighbourhood_offsets(radius: int) -> List[Tuple[int, int, int]]:
        return [
            (dz, dy, dx)
            for dz in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _local_variation(f: torch.Tensor, radius: int = 1) -> torch.Tensor:
        """Local structural variation, normalized to [0, 1] per volume.

        The variation is computed from a 3x3x3 average-pooled neighbourhood
        (E[F^2] - E[F]^2), averaged over channels.
        """
        k = 2 * radius + 1
        pad = radius
        mean = F.avg_pool3d(f, k, stride=1, padding=pad, count_include_pad=False)
        mean_sq = F.avg_pool3d(f * f, k, stride=1, padding=pad, count_include_pad=False)
        var = (mean_sq - mean * mean).clamp_min(0.0).mean(dim=1, keepdim=True)  # (B,1,D,H,W)
        b = var.shape[0]
        flat = var.view(b, -1)
        vmin = flat.min(dim=1, keepdim=True)[0]
        vmax = flat.max(dim=1, keepdim=True)[0]
        return ((flat - vmin) / (vmax - vmin + 1e-6)).view_as(var)

    def select_centers(
        self,
        response: torch.Tensor,
        variation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy diversity-aware center selection.

        Parameters
        ----------
        response: (D, H, W) normalized anatomical response map ``O_norm``.
        variation: (D, H, W) normalized local structural variation ``S``.

        Returns
        -------
        centers: (K, 3) integer coordinates.
        keep: (K,) boolean mask, ``True`` for centers obtained from the
            candidate pool and ``False`` for padded entries.
        """
        device = response.device
        shape = torch.tensor(response.shape, device=device)
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(shape[0], device=device),
                torch.arange(shape[1], device=device),
                torch.arange(shape[2], device=device),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 3)  # (N, 3), ordering (z, y, x)

        flat_resp = response.reshape(-1)
        flat_var = variation.reshape(-1)

        candidates = torch.nonzero(flat_resp > self.tau, as_tuple=False).squeeze(-1)
        if candidates.numel() < self.num_tokens:
            k = min(self.num_tokens, flat_resp.numel())
            candidates = torch.topk(flat_resp, k=k).indices
        if self.max_candidates is not None and candidates.numel() > self.max_candidates:
            top = torch.topk(flat_resp[candidates], k=self.max_candidates).indices
            candidates = candidates[top]

        cand_coords = grid[candidates].float()  # (M, 3)
        cand_resp = flat_resp[candidates]
        cand_var = flat_var[candidates]

        num_tokens = min(self.num_tokens, cand_coords.shape[0])
        selected: List[int] = []
        selected_scores = torch.full((cand_coords.shape[0],), -torch.inf, device=device)

        first = torch.argmax(cand_resp)
        selected.append(int(first))

        centers = torch.zeros((self.num_tokens, 3), dtype=torch.long, device=device)
        keep = torch.zeros(self.num_tokens, dtype=torch.bool, device=device)
        centers[0] = cand_coords[selected[0]].long()
        keep[0] = True

        for k in range(1, self.num_tokens):
            if len(selected) < num_tokens:
                sel_coords = cand_coords[selected]  # (|S|, 3)
                dist = torch.cdist(cand_coords, sel_coords).amin(dim=1)  # (M,)
                redundancy = 1.0 - torch.exp(-(dist ** 2) / (2.0 * self.sigma ** 2))
                scores = cand_resp * (1.0 + self.alpha * cand_var) * redundancy
                scores[selected] = -torch.inf
                nxt = int(torch.argmax(scores))
                if not torch.isfinite(scores[nxt]):
                    # candidate pool exhausted: pad with the last valid center
                    centers[k] = centers[selected[-1]]
                    keep[k] = False
                    continue
                selected.append(nxt)
                selected_scores[nxt] = scores[nxt]
                centers[k] = cand_coords[nxt].long()
                keep[k] = True
            else:
                centers[k] = centers[selected[-1]]
                keep[k] = False

        return centers, keep

    def aggregate(
        self,
        f: torch.Tensor,
        centers: torch.Tensor,
        keep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive local aggregation around the centers + positional encoding."""
        b, c, d, h, w = f.shape
        r = self.radius
        num_tokens = centers.shape[0]

        padded = F.pad(f, (r, r, r, r, r, r), mode="replicate")
        dp, hp, wp = d + 2 * r, h + 2 * r, w + 2 * r

        offsets = torch.as_tensor(self.offsets, device=f.device, dtype=torch.long)  # (27, 3)
        idx = centers[:, None, :] + offsets[None, :, :]  # (K, 27, 3)
        flat_idx = (idx[..., 0] * hp + idx[..., 1]) * wp + idx[..., 2]  # (K, 27)

        flat_feat = padded.reshape(b, c, -1)
        gathered = flat_feat[:, :, flat_idx.reshape(-1)]  # (B, C, K*27)
        gathered = gathered.view(b, c, num_tokens, self.num_neighbours)

        # softmax weights over the neighbourhood
        nb = gathered.permute(0, 2, 3, 1).reshape(b * num_tokens * self.num_neighbours, c)
        e = self.g(nb).view(b, num_tokens, self.num_neighbours)
        weights = torch.softmax(e, dim=-1)  # (B, K, 27)

        fk = torch.einsum("bkn,bckn->bkc", weights.to(gathered.dtype), gathered)  # (B, K, C)

        # normalized coordinates in [-1, 1] for the positional embedding
        size = torch.tensor([d - 1, h - 1, w - 1], device=f.device, dtype=torch.float32).clamp_min(1)
        norm_coords = centers.float() / size * 2.0 - 1.0  # (K, 3)
        pos = self.pos_embed(norm_coords.unsqueeze(0).expand(b, -1, -1))  # (B, K, C)

        tokens = self.proj(fk + pos)
        return tokens, weights

    # ------------------------------------------------------------------ #
    def forward(self, f: torch.Tensor):
        """Compute localization tokens.

        Parameters
        ----------
        f: (B, C, D, H, W) deepest semantic feature ``F_d``.

        Returns
        -------
        tokens: (B, K, token_dim)
        response_logits: (B, 1, D, H, W) pre-sigmoid anatomical response map
            (``sigmoid`` of this map is ``O = sigma(psi(F_d))``; the logits are
            returned so that the auxiliary loss can be evaluated in a
            numerically stable way)
        centers: (B, K, 3) selected anatomical centers
        weights: (B, K, 27) adaptive aggregation weights
        """
        b, _, d, h, w = f.shape
        response_logits = self.head(f)  # psi(F_d)
        response = torch.sigmoid(response_logits)

        variation = self._local_variation(f.detach(), radius=1)  # (B, 1, D, H, W)

        tokens, centers_all, weights_all = [], [], []
        for i in range(b):
            r = response[i, 0]
            r_norm = (r - r.min()) / (r.max() - r.min() + 1e-6)
            centers, keep = self.select_centers(r_norm, variation[i, 0])
            tok, wgt = self.aggregate(f[i : i + 1], centers, keep)
            tokens.append(tok)
            centers_all.append(centers)
            weights_all.append(wgt)

        return (
            torch.cat(tokens, dim=0),
            response_logits,
            torch.stack(centers_all, dim=0),
            torch.cat(weights_all, dim=0),
        )
