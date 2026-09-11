"""Evaluation metrics: Dice similarity coefficient and 95% Hausdorff distance."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Dice similarity coefficient between two binary masks."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0
    intersection = np.logical_and(pred, target).sum()
    return float(2.0 * intersection / (pred.sum() + target.sum() + eps))


def _surface(mask: np.ndarray, spacing: Sequence[float]):
    from scipy import ndimage

    mask = mask.astype(bool)
    if mask.sum() == 0:
        return None
    eroded = ndimage.binary_erosion(
        mask, ndimage.generate_binary_structure(3, 1), border_value=0
    )
    surface = mask ^ eroded
    if surface.sum() == 0:
        surface = mask
    return surface


def hd95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    percentile: float = 95.0,
) -> float:
    """95th percentile of the symmetric Hausdorff distance (in millimetres)."""
    from scipy import ndimage

    pred_surface = _surface(pred, spacing)
    target_surface = _surface(target, spacing)
    if pred_surface is None and target_surface is None:
        return 0.0
    if pred_surface is None or target_surface is None:
        # one of the two masks is empty -> use the spacing-implied maximum
        shape = np.asarray(pred.shape, dtype=np.float64)
        return float(np.linalg.norm(shape * np.asarray(spacing)))
    dist_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    d1 = dist_to_target[pred_surface]
    d2 = dist_to_pred[target_surface]
    return float(np.percentile(np.concatenate([d1, d2]), percentile))


def region_to_mask(labels: np.ndarray, region: Iterable[int]) -> np.ndarray:
    """Binary mask of a (possibly nested) region defined by a set of labels."""
    region = list(region)
    return np.isin(labels, region)


def evaluate_regions(
    prediction: np.ndarray,
    target: np.ndarray,
    regions: Mapping[str, Sequence[int]],
    spacing: Sequence[float] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Dice and HD95 for every named region of a task.

    ``regions`` maps a region name to the list of label values it contains,
    e.g. ``{"WT": [1, 2, 3], "TC": [2, 3], "NET": [2]}``.  When ``spacing`` is
    given the HD95 is expressed in millimetres, otherwise in voxels.
    """
    results: Dict[str, Dict[str, float]] = {}
    for name, labels in regions.items():
        pred_mask = region_to_mask(prediction, labels)
        target_mask = region_to_mask(target, labels)
        scores = {"dice": dice_score(pred_mask, target_mask)}
        if spacing is not None:
            scores["hd95"] = hd95(pred_mask, target_mask, spacing)
        else:
            scores["hd95"] = hd95(pred_mask, target_mask, (1.0, 1.0, 1.0))
        results[name] = scores
    return results


def summarize(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Average Dice / HD95 over all regions of a case."""
    if not results:
        return {"dice": 0.0, "hd95": 0.0}
    n = len(results)
    return {
        "dice": float(sum(r["dice"] for r in results.values()) / n),
        "hd95": float(sum(r["hd95"] for r in results.values()) / n),
    }
