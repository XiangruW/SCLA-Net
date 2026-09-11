"""Loss functions.

Total objective (Section "Localization-aware Optimization Loss")::

    L = L_seg + lambda_loc * L_loc,     L_seg = L_Dice + L_CE

* ``L_Dice``  soft Dice loss over the foreground classes.
* ``L_CE``    voxel-wise cross entropy.
* ``L_loc``   binary cross entropy between the predicted anatomical response
  map ``O`` and the anatomical importance target map ``H``.

The manuscript states that ``H`` is constructed from the ground-truth mask
``Y`` but does not describe the construction.  This implementation uses the
smoothed binary foreground-union map (a 3x3x3 Gaussian filter applied to
``Y > 0``), which is the setting used for the results in the paper.  The
``target_mode`` option allows ``binary`` (no smoothing) as well.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float32):
    """(B, D, H, W) integer labels -> (B, C, D, H, W) one-hot tensor."""
    labels = labels.long().unsqueeze(1)
    out = torch.zeros(
        (labels.shape[0], num_classes, *labels.shape[2:]), device=labels.device, dtype=dtype
    )
    return out.scatter_(1, labels, 1.0)


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-5):
    """Soft Dice loss over all classes except the background (channel 0)."""
    probs = torch.softmax(logits, dim=1)
    if targets.shape[1] != probs.shape[1]:
        targets = one_hot(targets, probs.shape[1], probs.dtype)
    num_classes = probs.shape[1]
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * targets, dims)
    cardinality = torch.sum(probs + targets, dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice[1:].mean() if num_classes > 1 else 1.0 - dice.mean()


def gaussian_kernel3d(sigma: float = 1.0, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = max(int(round(2 * sigma)), 1)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    k3 = kernel[:, None, None] * kernel[None, :, None] * kernel[None, None, :]
    return k3


def anatomical_target(labels: torch.Tensor, mode: str = "gaussian", sigma: float = 1.0):
    """Anatomical importance target map ``H`` derived from the mask ``Y``."""
    foreground = (labels > 0).to(torch.float32).unsqueeze(1)  # (B, 1, D, H, W)
    if mode in (None, "binary", "none"):
        return foreground
    if mode == "gaussian":
        k3 = gaussian_kernel3d(sigma, device=labels.device, dtype=foreground.dtype)
        c = k3.shape[0] // 2
        kernel = k3.view(1, 1, *k3.shape).repeat(1, 1, 1, 1, 1)
        return F.conv3d(F.pad(foreground, (c, c, c, c, c, c), mode="replicate"), kernel)
    raise ValueError(f"unknown target mode: {mode}")


class SCLANetLoss(nn.Module):
    """Dice + cross entropy + localization-aware auxiliary loss.

    Parameters
    ----------
    lambda_loc:
        Weight ``lambda_loc`` of the auxiliary localization loss.
    target_mode / target_sigma:
        Construction of the anatomical importance target map ``H``.
    """

    def __init__(
        self,
        lambda_loc: float = 0.5,
        target_mode: str = "gaussian",
        target_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_loc = float(lambda_loc)
        self.target_mode = target_mode
        self.target_sigma = float(target_sigma)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_logits: torch.Tensor | None = None,
    ):
        dice = soft_dice_loss(logits, labels)
        ce = F.cross_entropy(logits, labels.long())
        parts: Dict[str, torch.Tensor] = {"dice": dice.detach(), "ce": ce.detach()}

        total = dice + ce
        if response_logits is not None and self.lambda_loc > 0:
            target = anatomical_target(labels, self.target_mode, self.target_sigma)
            if target.shape[2:] != response_logits.shape[2:]:
                # the response map lives on the bottleneck grid: bring the
                # anatomical importance target to the same resolution
                target = F.interpolate(
                    target, size=response_logits.shape[2:], mode="trilinear", align_corners=False
                )
            loc = F.binary_cross_entropy_with_logits(response_logits, target)
            parts["loc"] = loc.detach()
            total = total + self.lambda_loc * loc
        return total, parts
