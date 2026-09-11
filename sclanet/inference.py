"""Sliding-window inference (Section "Experimental Setup").

During inference a sliding window of ``96 x 96 x 96`` with ``0.5`` overlap is
used and overlapping predictions are averaged.  No test-time augmentation is
applied.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _window_positions(length: int, window: int, step: int):
    if window >= length:
        return [0]
    positions = list(range(0, length - window + 1, step))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


@torch.no_grad()
def sliding_window_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    window: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Run the model over a sliding window and average the logits.

    Parameters
    ----------
    model: network returning a dictionary with a ``"logits"`` entry.
    image: (B, C, D, H, W) input volume (already preprocessed).
    window: size of the sliding window.
    overlap: fractional overlap between neighbouring windows.

    Returns
    -------
    Averaged logits of shape (B, num_classes, D, H, W).
    """
    model.eval()
    b, _, d, h, w = image.shape
    win = [min(int(wi), si) for wi, si in zip(window, (d, h, w))]
    steps = [max(int(wi * (1.0 - overlap)), 1) for wi in win]

    positions = [_window_positions(s, wi, st) for s, wi, st in zip((d, h, w), win, steps)]

    with torch.no_grad():
        probe = model(image[:1, :, : win[0], : win[1], : win[2]])
    n_cls = num_classes or probe["logits"].shape[1]
    del probe

    accum = torch.zeros((b, n_cls, d, h, w), device=image.device, dtype=torch.float32)
    counts = torch.zeros((1, 1, d, h, w), device=image.device, dtype=torch.float32)

    for z in positions[0]:
        for y in positions[1]:
            for x in positions[2]:
                patch = image[:, :, z : z + win[0], y : y + win[1], x : x + win[2]]
                logits = model(patch)["logits"].float()
                accum[:, :, z : z + win[0], y : y + win[1], x : x + win[2]] += logits
                counts[:, :, z : z + win[0], y : y + win[1], x : x + win[2]] += 1.0

    return accum / counts.clamp_min(1.0)


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    image: torch.Tensor,
    window: Sequence[int] = (96, 96, 96),
    overlap: float = 0.5,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Return the argmax segmentation of a volume (B, D, H, W)."""
    logits = sliding_window_logits(model, image, window, overlap, num_classes)
    return torch.argmax(logits, dim=1)
