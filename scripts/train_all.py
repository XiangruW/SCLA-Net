"""Train every experiment of the paper one after another.

::

    python scripts/train_all.py                       # all eleven experiments
    python scripts/train_all.py --only btcv13,msd:task03_liver
    python scripts/train_all.py --dry-run             # show the commands only
    python scripts/train_all.py --resume              # continue finished runs from last.pth
    python scripts/train_all.py --iterations 2000     # shorter schedule (debugging)

Experiments whose ``best.pth`` already exists are skipped unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [("configs/btcv13.yaml", None)] + [
    ("configs/msd.yaml", task)
    for task in (
        "task01_braintumour",
        "task02_heart",
        "task03_liver",
        "task04_hippocampus",
        "task05_prostate",
        "task06_lung",
        "task07_pancreas",
        "task08_hepaticvessel",
        "task09_spleen",
        "task10_colon",
    )
]


def run_name(config: str, task: str | None) -> str:
    base = os.path.basename(config).replace(".yaml", "")
    return f"{base}:{task}" if task else base


def checkpoint_dir(config: str, task: str | None) -> str:
    base = os.path.basename(config).replace(".yaml", "")
    return os.path.join(ROOT, "checkpoints", base if task is None else os.path.join(base, task))


def parse_args():
    p = argparse.ArgumentParser(description="Train all SCLA-Net experiments")
    p.add_argument("--only", default=None, help="comma separated list, e.g. btcv13,msd:task03_liver")
    p.add_argument("--force", action="store_true", help="retrain experiments that already have a checkpoint")
    p.add_argument("--resume", action="store_true", help="resume from last.pth when it exists")
    p.add_argument("--dry-run", action="store_true", help="only print the commands")
    p.add_argument("--stop-on-error", action="store_true", help="abort on the first failing run")
    p.add_argument("--iterations", type=int, default=None, help="override train.iterations")
    p.add_argument("--extra", action="append", default=[], help="extra --set key=value arguments")
    return p.parse_args()


def main():
    args = parse_args()
    selected = RUNS
    if args.only:
        wanted = {item.strip() for item in args.only.split(",")}
        selected = [r for r in RUNS if run_name(*r) in wanted or os.path.basename(r[0])[:-5] in wanted]
        if not selected:
            raise SystemExit(f"nothing matches --only {args.only}")

    print(f"{len(selected)} experiment(s) selected")
    failures = []
    for config, task in selected:
        name = run_name(config, task)
        ckpt_dir = checkpoint_dir(config, task)
        best = os.path.join(ckpt_dir, "best.pth")
        last = os.path.join(ckpt_dir, "last.pth")

        if os.path.exists(best) and not args.force:
            print(f"[skip] {name}: {best} already exists (use --force to retrain)")
            continue

        cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--config", os.path.join(ROOT, config)]
        if task:
            cmd += ["--task", task]
        if args.resume and os.path.exists(last):
            cmd += ["--resume", last]
        if args.iterations:
            cmd += ["--iterations", str(args.iterations)]
        for item in args.extra:
            cmd += ["--set", item]

        print(f"\n=== {name} ===\n{' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue

        t0 = time.time()
        result = subprocess.run(cmd, cwd=ROOT)
        minutes = (time.time() - t0) / 60.0
        if result.returncode != 0:
            failures.append(name)
            print(f"[fail] {name} after {minutes:.1f} min")
            if args.stop_on_error:
                break
        else:
            print(f"[done] {name} in {minutes:.1f} min")

    if failures:
        print(f"\nfailed runs: {', '.join(failures)}")
        sys.exit(1)
    print("\nall selected experiments finished")


if __name__ == "__main__":
    main()
