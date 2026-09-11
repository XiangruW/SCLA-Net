"""Report the number of parameters and the computational cost of SCLA-Net.

::

    python scripts/model_stats.py --config configs/btcv13.yaml
    python scripts/model_stats.py --config configs/btcv13.yaml --variant baseline

The manuscript reports 6.52 M parameters and 34.70 GFLOPs for the complete
model (and 1.26 M / 16.51 GFLOPs for the lightweight backbone baseline) at an
input size of 96 x 96 x 96.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from sclanet.models import build_model  # noqa: E402
from sclanet.utils import count_flops, count_parameters, load_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SCLA-Net model statistics")
    p.add_argument("--config", required=True)
    p.add_argument("--variant", default=None, choices=["all", "scea", "salt", "lgca", "baseline"])
    p.add_argument("--input-size", type=int, nargs=3, default=(96, 96, 96))
    p.add_argument("--all-variants", action="store_true")
    return p.parse_args()


def report(cfg, variant, input_size):
    model = build_model(cfg, variant=variant)
    model.eval()
    in_channels = cfg["model"].get("in_channels", 1)
    params = count_parameters(model)
    flops = count_flops(model, (1, in_channels, *input_size))
    print(f"{variant:<10} params = {params / 1e6:7.2f} M   GFLOPs = {flops:7.2f}")
    return params, flops


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print(f"config: {args.config}   input size: {tuple(args.input_size)}")
    if args.all_variants:
        for variant in ("baseline", "scea", "salt", "lgca", "all"):
            report(cfg, variant, args.input_size)
    else:
        report(cfg, args.variant, args.input_size)


if __name__ == "__main__":
    main()
