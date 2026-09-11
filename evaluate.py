"""Evaluate a trained SCLA-Net checkpoint with sliding-window inference.

::

    python evaluate.py --config configs/btcv13.yaml --ckpt checkpoints/btcv13/best.pth
    python evaluate.py --config configs/msd.yaml --task task03_liver --ckpt <path>

The Dice similarity coefficient and the 95% Hausdorff distance are computed
for every region and then averaged, matching the protocol in the manuscript.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from sclanet.data import load_meta, read_split_file
from sclanet.data.splits import BTCV13_REGIONS, MSD_TASKS
from sclanet.inference import predict_volume
from sclanet.metrics import evaluate_regions, summarize
from sclanet.models import build_model
from sclanet.utils import load_checkpoint, load_config, resolve_task_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SCLA-Net")
    p.add_argument("--config", required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--variant", default=None, choices=["all", "scea", "salt", "lgca", "baseline"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--preprocessed", default=None)
    p.add_argument("--split-dir", default=None)
    p.add_argument("--out", default=None, help="optional JSON file for the per-case results")
    p.add_argument("--set", action="append", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = resolve_task_config(load_config(args.config, args.set), args.task)
    if args.variant:
        cfg["variant"] = args.variant
    if args.preprocessed:
        cfg["data"]["preprocessed"] = args.preprocessed
    split_dir = args.split_dir or cfg["data"]["split_dir"]

    name = cfg["experiment"] + (f"_{cfg['task']}" if cfg.get("task") else "")
    cases = read_split_file(os.path.join(split_dir, f"{name}_{args.split}.txt"))

    regions = cfg.get("regions")
    if regions is None:
        regions = (
            MSD_TASKS[cfg["task"]]["regions"] if cfg.get("task") else BTCV13_REGIONS
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()

    meta = load_meta(cfg["data"]["preprocessed"])
    window = cfg["inference"]["window"]
    overlap = cfg["inference"]["overlap"]
    num_classes = cfg["model"]["num_classes"]

    per_case, all_dice, all_hd95 = {}, [], []
    for case in cases:
        image = np.load(os.path.join(cfg["data"]["preprocessed"], f"{case}_image.npy")).astype(
            np.float32
        )
        label = np.load(os.path.join(cfg["data"]["preprocessed"], f"{case}_label.npy")).astype(
            np.int64
        )
        if image.ndim == 3:
            image = image[None]
        spacing = meta.get(case, {}).get("spacing")

        pred = predict_volume(
            model, torch.from_numpy(image)[None].to(device), window, overlap, num_classes
        )[0].cpu().numpy()

        results = evaluate_regions(pred, label, regions, spacing=spacing)
        summary = summarize(results)
        per_case[case] = {"regions": results, "mean": summary}
        all_dice.append(summary["dice"])
        all_hd95.append(summary["hd95"])
        print(
            f"{case}: DSC={summary['dice'] * 100:.2f}  HD95={summary['hd95']:.2f} mm  "
            + "  ".join(f"{k}={v['dice'] * 100:.2f}" for k, v in results.items())
        )

    mean_dice = float(np.mean(all_dice) * 100)
    mean_hd95 = float(np.mean(all_hd95))
    print(f"\n===== {name} ({args.split}) =====")
    print(f"mean DSC : {mean_dice:.2f}")
    print(f"mean HD95: {mean_hd95:.2f} mm")
    print(f"cases    : {len(cases)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {"mean_dice": mean_dice, "mean_hd95": mean_hd95, "cases": per_case},
                fh,
                indent=2,
                sort_keys=True,
            )
        print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
