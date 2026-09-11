"""Measure training speed and peak GPU memory for a configuration.

Useful to decide whether a given GPU can run the paper's protocol (40 000
iterations, batch size 2, 96^3 patches) and how long it takes.

::

    python scripts/benchmark.py --config configs/btcv13.yaml
    python scripts/benchmark.py --config configs/btcv13.yaml --batch-size 1 --amp
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from sclanet.losses import SCLANetLoss  # noqa: E402
from sclanet.models import build_model  # noqa: E402
from sclanet.utils import amp_autocast, amp_grad_scaler, count_parameters, load_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SCLA-Net speed / memory benchmark")
    p.add_argument("--config", required=True)
    p.add_argument("--variant", default=None, choices=["all", "scea", "salt", "lgca", "baseline"])
    p.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    p.add_argument("--patch-size", type=int, nargs=3, default=None)
    p.add_argument("--iters", type=int, default=5, help="timed iterations (after 1 warm-up)")
    p.add_argument("--amp", action="store_true", help="use mixed precision")
    p.add_argument("--cpu", action="store_true", help="benchmark on CPU")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.variant:
        cfg["variant"] = args.variant

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(args.amp) and device.type == "cuda"

    patch = tuple(args.patch_size or cfg["data"].get("patch_size", (96, 96, 96)))
    batch_size = int(args.batch_size or cfg["train"].get("batch_size", 2))
    channels = int(cfg["model"].get("in_channels", 1))
    num_classes = int(cfg["model"]["num_classes"])

    model = build_model(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = SCLANetLoss(
        lambda_loc=float(cfg.get("loss", {}).get("lambda_loc", 0.5)),
        target_mode=cfg.get("loss", {}).get("target", "gaussian"),
    )
    scaler = amp_grad_scaler(device.type, enabled=amp)

    print(f"device        : {device.type}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"parameters    : {count_parameters(model) / 1e6:.2f} M")
    print(f"input         : {batch_size} x {channels} x {' x '.join(map(str, patch))}"
          f"   (amp={'on' if amp else 'off'})")

    x = torch.randn(batch_size, channels, *patch, device=device)
    y = torch.randint(0, num_classes, (batch_size, *patch), device=device)

    def full_step():
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(device.type, enabled=amp):
            out = model(x)
            loss, _ = criterion(out["logits"], y, out.get("response_logits"))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        full_step()  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.iters):
            full_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = (time.time() - t0) / args.iters
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA out of memory with this configuration.")
        print("Try:  --batch-size 1  (and train.grad_accum_steps=2 to keep an")
        print("      effective batch size of 2), or reduce data.patch_size.")
        raise SystemExit(1)

    iterations = int(cfg["train"].get("iterations", 40000))
    hours = seconds * iterations / 3600.0
    print(f"time / iteration : {seconds:.3f} s")
    print(f"estimated total  : {hours:.2f} h for {iterations} iterations"
          f" (validation excluded)")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"peak GPU memory  : {peak:.2f} GB / {total:.2f} GB")
        if peak > 0.9 * total:
            print("NOTE: close to the memory limit - consider "
                  "--batch-size 1 with train.grad_accum_steps=2")


if __name__ == "__main__":
    main()
