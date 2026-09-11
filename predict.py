"""Run sliding-window inference on a single volume.

::

    # on a preprocessed .npy volume
    python predict.py --config configs/btcv13.yaml --ckpt checkpoints/btcv13/best.pth \
        --input data/preprocessed/btcv13/img0027_image.npy --out pred_img0027.nii.gz

    # directly on a NIfTI file (preprocessed on the fly)
    python predict.py --config configs/btcv13.yaml --ckpt best.pth \
        --input /path/to/img0027.nii.gz --out pred_img0027.nii.gz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from sclanet.inference import predict_volume
from sclanet.models import build_model
from sclanet.utils import load_checkpoint, load_config, resolve_task_config


def parse_args():
    p = argparse.ArgumentParser(description="SCLA-Net inference")
    p.add_argument("--config", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--variant", default=None, choices=["all", "scea", "salt", "lgca", "baseline"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True, help=".npy or .nii/.nii.gz volume")
    p.add_argument("--out", default="prediction.nii.gz")
    p.add_argument("--set", action="append", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = resolve_task_config(load_config(args.config, args.set), args.task)
    if args.variant:
        cfg["variant"] = args.variant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spacing = None
    if args.input.endswith(".npy"):
        image = np.load(args.input).astype(np.float32)
        if image.ndim == 3:
            image = image[None]
    else:
        from sclanet.data.preprocess import normalize_intensity, resample_volume, load_nifti

        data, spacing = load_nifti(args.input)
        data, spacing = resample_volume(data, spacing, cfg["data"].get("spacing"))
        data = normalize_intensity(
            data,
            mode=cfg["data"].get("intensity", "hu"),
            hu_window=cfg["data"].get("hu_window", (-175.0, 250.0)),
            percentile=cfg["data"].get("percentile", (0.5, 99.5)),
        )
        image = np.moveaxis(data, -1, 0) if data.ndim == 4 else data[None]

    model = build_model(cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()

    prediction = predict_volume(
        model,
        torch.from_numpy(image.astype(np.float32))[None].to(device),
        cfg["inference"]["window"],
        cfg["inference"]["overlap"],
        cfg["model"]["num_classes"],
    )[0].cpu().numpy().astype(np.int16)

    out = args.out
    if out.endswith(".nii") or out.endswith(".nii.gz"):
        import nibabel as nib

        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        affine = np.diag([float(spacing[2]), float(spacing[1]), float(spacing[0]), 1.0])
        nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine), out)
    else:
        np.save(out, prediction)
    print(f"prediction written to {os.path.abspath(out)}  shape={prediction.shape}")


if __name__ == "__main__":
    main()
