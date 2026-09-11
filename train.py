"""Train SCLA-Net.

Example
-------
::

    python train.py --config configs/btcv13.yaml
    python train.py --config configs/msd.yaml --task task03_liver

Hyper-parameters come from the YAML configuration (AdamW, learning rate
1e-4, weight decay 1e-5, batch size 2, 40 000 iterations, mixed precision,
96^3 patches, best checkpoint selected by validation DSC).
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from sclanet.data import list_btcv13_cases, list_msd_cases, make_loader, read_split_file
from sclanet.data.splits import MSD_TASKS, btcv13_split, make_split, write_split_file
from sclanet.losses import SCLANetLoss
from sclanet.metrics import evaluate_regions, summarize
from sclanet.models import build_model
from sclanet.utils import (
    AverageMeter,
    Logger,
    amp_autocast,
    amp_grad_scaler,
    gpu_memory_gb,
    load_checkpoint,
    load_config,
    resolve_task_config,
    save_checkpoint,
    save_config,
    set_seed,
)
from sclanet.inference import predict_volume


def parse_args():
    p = argparse.ArgumentParser(description="Train SCLA-Net")
    p.add_argument("--config", required=True, help="path to a YAML configuration file")
    p.add_argument("--task", default=None, help="MSD task name (see configs/msd.yaml)")
    p.add_argument("--variant", default=None, choices=["all", "scea", "salt", "lgca", "baseline"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--preprocessed", default=None, help="override data.preprocessed")
    p.add_argument("--split-dir", default=None, help="override data.split_dir")
    p.add_argument("--raw-dir", default=None, help="override data.raw_dir")
    p.add_argument("--set", action="append", default=[], help="override config values, key=value")
    return p.parse_args()


def resolve_split(cfg, split_dir):
    """Load the checked-in split files, or derive the split if they are missing."""
    name = cfg["experiment"] + (f"_{cfg['task']}" if cfg.get("task") else "")
    train_file = os.path.join(split_dir, f"{name}_train.txt")
    val_file = os.path.join(split_dir, f"{name}_val.txt")
    if os.path.exists(train_file) and os.path.exists(val_file):
        return read_split_file(train_file), read_split_file(val_file)

    print(f"[split] {train_file} not found - deriving the split from the raw data")
    if cfg["experiment"] == "btcv13":
        cases = list_btcv13_cases(cfg["data"]["raw_dir"])
        split = btcv13_split(cases)
    else:
        cases = list_msd_cases(cfg["data"]["raw_dir"], cfg["task"])
        split = make_split(cases, val_ratio=cfg["data"].get("val_ratio", 0.2), seed=cfg["seed"])
    return split["train"], split["val"]


@torch.no_grad()
def validate(model, cases, cfg, device, compute_hd95_cases: int = 3):
    """Sliding-window validation; returns the mean DSC and HD95 over the cases."""
    num_classes = cfg["model"]["num_classes"]
    window = cfg["inference"]["window"]
    overlap = cfg["inference"]["overlap"]
    regions = cfg.get("regions") or {f"class{i}": [i] for i in range(1, num_classes)}

    dices, hd95s = [], []
    for i, case in enumerate(cases):
        image = torch.from_numpy(
            __import__("numpy").load(os.path.join(cfg["data"]["preprocessed"], f"{case}_image.npy"))
        ).float()
        label = __import__("numpy").load(
            os.path.join(cfg["data"]["preprocessed"], f"{case}_label.npy")
        ).astype("int64")
        if image.ndim == 3:
            image = image[None]
        image = image[None].to(device)

        pred = predict_volume(model, image, window, overlap, num_classes)[0].cpu().numpy()

        if i < compute_hd95_cases:
            results = evaluate_regions(pred, label, regions, spacing=None)
        else:
            results = {
                name: {"dice": evaluate_regions(pred, label, {name: labels})[name]["dice"], "hd95": 0.0}
                for name, labels in regions.items()
            }
        summary = summarize(results)
        dices.append(summary["dice"])
        hd95s.append(summary["hd95"])
        print(f"    {case}: DSC={summary['dice'] * 100:.2f}  HD95={summary['hd95']:.2f}")

    return sum(dices) / max(len(dices), 1), sum(hd95s) / max(len(hd95s), 1)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    cfg = resolve_task_config(cfg, args.task)
    if args.variant:
        cfg["variant"] = args.variant
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.iterations is not None:
        cfg["train"]["iterations"] = args.iterations
    if args.preprocessed:
        cfg["data"]["preprocessed"] = args.preprocessed
    if args.split_dir:
        cfg["data"]["split_dir"] = args.split_dir
    if args.raw_dir:
        cfg["data"]["raw_dir"] = args.raw_dir
    if cfg.get("task") and "regions" not in cfg:
        cfg["regions"] = MSD_TASKS[cfg["task"]]["regions"]

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = cfg["train"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(os.path.join(ckpt_dir, "train.log"))
    logger(f"config: {args.config} task={cfg.get('task')} variant={cfg.get('variant', 'all')}")
    logger(f"device: {device}")
    save_config(cfg, os.path.join(ckpt_dir, "config.yaml"))

    train_cases, val_cases = resolve_split(cfg, cfg["data"]["split_dir"])
    logger(f"train cases: {len(train_cases)}  val cases: {len(val_cases)}")
    if not os.path.exists(os.path.join(cfg["data"]["preprocessed"], "meta.json")):
        raise FileNotFoundError(
            f"preprocessed data not found in '{cfg['data']['preprocessed']}'. "
            "Run scripts/prepare_data.py first."
        )

    train_loader = make_loader(
        train_cases,
        cfg["data"]["preprocessed"],
        cfg["data"],
        training=True,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 0),
    )

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger(f"trainable parameters: {n_params / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = amp_grad_scaler(device.type, enabled=use_amp)
    if device.type == "cuda":
        logger(f"GPU: {torch.cuda.get_device_name(0)}  ({gpu_memory_gb():.1f} GB)")

    criterion = SCLANetLoss(
        lambda_loc=float(cfg["loss"]["lambda_loc"]),
        target_mode=cfg["loss"].get("target", "gaussian"),
        target_sigma=float(cfg["loss"].get("target_sigma", 1.0)),
    )

    start_iter, best_dice = 0, -1.0
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_iter = int(payload.get("iteration", 0))
        best_dice = float(payload.get("best_dice", -1.0))
        logger(f"resumed from {args.resume} at iteration {start_iter}")

    iterations = int(cfg["train"]["iterations"])
    val_interval = int(cfg["train"].get("val_interval", 500))
    grad_accum = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    if grad_accum > 1:
        logger(
            f"gradient accumulation: {grad_accum} micro-batches "
            f"(effective batch size {grad_accum * int(cfg['train']['batch_size'])})"
        )
    model.train()
    meter = AverageMeter()
    t0 = time.time()

    iterator = iter(train_loader)
    for it in range(start_iter + 1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            with amp_autocast(device.type, enabled=use_amp):
                out = model(image)
                loss, parts = criterion(out["logits"], label, out.get("response_logits"))
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            meter.update(loss.item() * grad_accum)

        scaler.step(optimizer)
        scaler.update()

        if it % 50 == 0:
            speed = (it - start_iter) / max(time.time() - t0, 1e-6)
            logger(
                f"iter {it}/{iterations}  loss {meter.avg:.4f} "
                f"(dice {parts['dice'].item():.3f} ce {parts['ce'].item():.3f}"
                + (f" loc {parts['loc'].item():.3f}" if "loc" in parts else "")
                + f")  {speed:.1f} it/s"
            )
            meter.reset()

        if it % val_interval == 0 or it == iterations:
            logger(f"validating at iteration {it} ...")
            dice, hd95 = validate(model, val_cases, cfg, device)
            logger(f"iter {it}: val DSC = {dice * 100:.2f}  HD95 = {hd95:.2f}")
            save_checkpoint(os.path.join(ckpt_dir, "last.pth"), model, optimizer, it, best_dice, cfg)
            if dice > best_dice:
                best_dice = dice
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pth"), model, optimizer, it, best_dice, cfg
                )
                logger(f"  new best model (DSC {best_dice * 100:.2f})")
            model.train()

    logger(f"training finished, best validation DSC = {best_dice * 100:.2f}")


if __name__ == "__main__":
    main()
