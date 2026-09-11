"""Preprocess raw NIfTI data into the ``.npy`` format used for training.

The datasets themselves are **not** distributed with this repository; download
them from their official sources and point ``--raw-dir`` at the extracted
files (see the README for the access routes and licences):

* BTCV13 - Synapse multi-organ abdominal CT benchmark
* MSD    - Medical Segmentation Decathlon

Examples
--------
::

    # BTCV13: <raw-dir>/img0001.nii.gz, <raw-dir>/label0001.nii.gz, ...
    python scripts/prepare_data.py --config configs/btcv13.yaml \
        --raw-dir /data/BTCV/Training --out data/preprocessed/btcv13

    # MSD: <raw-dir>/Task03_Liver/imagesTr/*.nii.gz
    python scripts/prepare_data.py --config configs/msd.yaml --task task03_liver \
        --raw-dir /data/MSD --out data/preprocessed/msd
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sclanet.data.preprocess import preprocess_dataset  # noqa: E402
from sclanet.data.splits import MSD_TASKS  # noqa: E402
from sclanet.utils import load_config, resolve_task_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess BTCV13 / MSD volumes")
    p.add_argument("--config", required=True)
    p.add_argument("--task", default=None, help="MSD task name (configs/msd.yaml)")
    p.add_argument("--raw-dir", required=True, help="directory with the raw NIfTI files")
    p.add_argument("--out", required=True, help="output directory for the .npy files")
    p.add_argument("--limit", type=int, default=None, help="only preprocess the first N cases")
    p.add_argument("--prefix-task", action="store_true", help="prefix MSD case ids with the task name")
    return p.parse_args()


def collect_btcv(raw_dir: str):
    cases = []
    for image_path in sorted(glob.glob(os.path.join(raw_dir, "**", "img*.nii.gz"), recursive=True)):
        stem = os.path.basename(image_path).replace("img", "").replace(".nii.gz", "")
        label_path = None
        for cand in (
            os.path.join(os.path.dirname(image_path), f"label{stem}.nii.gz"),
            os.path.join(os.path.dirname(image_path), f"labelsTr", f"label{stem}.nii.gz"),
        ):
            if os.path.exists(cand):
                label_path = cand
                break
        cases.append({"id": f"img{stem}", "image": image_path, "label": label_path})
    return cases


def collect_msd(raw_dir: str, task: str, prefix_task: bool = False):
    task_cfg = MSD_TASKS[task]
    image_dir = os.path.join(raw_dir, task_cfg["dir"], "imagesTr")
    label_dir = os.path.join(raw_dir, task_cfg["dir"], "labelsTr")

    cases = []
    for first in sorted(glob.glob(os.path.join(image_dir, "*_0000.nii.gz"))):
        base = os.path.basename(first).replace("_0000.nii.gz", "")
        label_path = os.path.join(label_dir, f"{base}.nii.gz")
        case_id = f"{task}_{base}" if prefix_task else base
        cases.append(
            {
                "id": case_id,
                "image": first,
                "label": label_path if os.path.exists(label_path) else None,
                "modalities": sorted(glob.glob(os.path.join(image_dir, f"{base}_*.nii.gz"))),
            }
        )
    return cases


def main():
    args = parse_args()
    cfg = resolve_task_config(load_config(args.config), args.task)
    pre_cfg = dict(cfg["data"])

    if cfg.get("task"):
        cases = collect_msd(args.raw_dir, cfg["task"], args.prefix_task)
        pre_cfg["intensity"] = cfg["data"].get("intensity", "hu")
    else:
        cases = collect_btcv(args.raw_dir)

    if not cases:
        raise SystemExit(f"no cases found in '{args.raw_dir}'")

    if args.limit:
        cases = cases[: args.limit]

    print(f"preprocessing {len(cases)} cases -> {args.out}")
    print(f"intensity={pre_cfg.get('intensity')} spacing={pre_cfg.get('spacing')} "
          f"hu_window={pre_cfg.get('hu_window')}")
    preprocess_dataset(cases, args.out, pre_cfg)
    print("done.")


if __name__ == "__main__":
    main()
