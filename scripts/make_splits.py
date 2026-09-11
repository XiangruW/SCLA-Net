"""Generate the train/validation split files stored in ``splits/``.

The splits are fully deterministic:

* **BTCV13** - cases ``img0001`` - ``img0024`` for training and
  ``img0025`` - ``img0030`` for validation (24/6 protocol).  These identifiers
  are fixed by the Synapse BTCV release, so the split can be written without
  the raw data (``--btcv-only``).
* **MSD** - for each task the case identifiers are sorted, shuffled with a
  fixed seed (default ``0``) and divided into 80% training / 20% validation.

Examples
--------
::

    python scripts/make_splits.py --btcv-only
    python scripts/make_splits.py --config configs/msd.yaml --raw-dir /data/MSD
    python scripts/make_splits.py --config configs/msd.yaml --task task03_liver --raw-dir /data/MSD
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sclanet.data.splits import (  # noqa: E402
    MSD_TASKS,
    btcv13_split,
    list_btcv13_cases,
    list_msd_cases,
    make_split,
    write_split_file,
)


def parse_args():
    p = argparse.ArgumentParser(description="Create split files")
    p.add_argument("--config", default="configs/msd.yaml")
    p.add_argument("--task", default=None, help="single MSD task (default: all ten)")
    p.add_argument("--raw-dir", default=None, help="directory with the raw datasets")
    p.add_argument("--out-dir", default="splits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--btcv-only", action="store_true", help="write the BTCV13 split only")
    return p.parse_args()


def main():
    args = parse_args()

    if args.raw_dir:
        cases = list_btcv13_cases(args.raw_dir)
        if not cases and args.btcv_only:
            cases = [f"img{i:04d}" for i in range(1, 31)]
    else:
        cases = [f"img{i:04d}" for i in range(1, 31)]

    split = btcv13_split(cases)
    write_split_file(os.path.join(args.out_dir, "btcv13_train.txt"), split["train"])
    write_split_file(os.path.join(args.out_dir, "btcv13_val.txt"), split["val"])
    print(f"btcv13: train={len(split['train'])} val={len(split['val'])}")

    if args.btcv_only:
        return

    if not args.raw_dir:
        print("no --raw-dir given: MSD splits are derived from the raw data and were skipped")
        return

    tasks = [args.task] if args.task else list(MSD_TASKS)
    for task in tasks:
        task_cases = list_msd_cases(args.raw_dir, task)
        if not task_cases:
            print(f"{task}: no cases found, skipped")
            continue
        split = make_split(task_cases, val_ratio=args.val_ratio, seed=args.seed)
        name = f"msd_{task}"
        write_split_file(os.path.join(args.out_dir, f"{name}_train.txt"), split["train"])
        write_split_file(os.path.join(args.out_dir, f"{name}_val.txt"), split["val"])
        print(f"{task}: train={len(split['train'])} val={len(split['val'])}")


if __name__ == "__main__":
    main()
