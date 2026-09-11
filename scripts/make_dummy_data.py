"""Create a tiny synthetic dataset for smoke tests.

No real data is downloaded: random volumes containing a few spherical blobs
are written in the same ``.npy`` layout as :mod:`scripts.prepare_data`, so
that the full pipeline (training loop, sliding-window inference, evaluation)
can be exercised without the datasets.

::

    python scripts/make_dummy_data.py --out data/dummy/btcv13 --cases 4
    python train.py --config configs/btcv13.yaml \
        --preprocessed data/dummy/btcv13 --split-dir data/dummy/btcv13/splits \
        --iterations 10 --set train.val_interval=5
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Create a synthetic dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--cases", type=int, default=4)
    p.add_argument("--shape", type=int, nargs=3, default=(96, 96, 96))
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--classes", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def blobs(shape, num_blobs, rng):
    label = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    for i in range(1, num_blobs + 1):
        center = rng.integers(8, np.array(shape) - 8)
        radius = rng.integers(6, max(7, shape[0] // 6))
        mask = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 < radius ** 2
        label[mask] = i
    return label


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    meta = {}
    cases = []
    for i in range(1, args.cases + 1):
        case = f"img{i:04d}"
        cases.append(case)
        label = blobs(tuple(args.shape), args.classes - 1, rng)
        image = rng.random((args.channels, *args.shape)).astype(np.float32) * 0.5
        image = image + (label[None] > 0).astype(np.float32) * 0.5  # blobs are "brighter"
        np.save(os.path.join(args.out, f"{case}_image.npy"), image.astype(np.float16))
        np.save(os.path.join(args.out, f"{case}_label.npy"), label)
        meta[case] = {"spacing": [1.0, 1.0, 1.0], "shape": list(image.shape)}

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    split_dir = os.path.join(args.out, "splits")
    os.makedirs(split_dir, exist_ok=True)
    n_val = max(1, args.cases // 4)
    with open(os.path.join(split_dir, "btcv13_train.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(cases[:-n_val]) + "\n")
    with open(os.path.join(split_dir, "btcv13_val.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(cases[-n_val:]) + "\n")

    print(f"wrote {args.cases} synthetic cases to {args.out}")


if __name__ == "__main__":
    main()
