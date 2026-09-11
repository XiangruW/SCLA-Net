"""Configuration, seeding, logging and checkpoint utilities."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from typing import Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def load_config(path: str, overrides=None) -> Dict:
    """Load a YAML configuration file.

    ``overrides`` is an iterable of ``key=value`` strings, where ``key`` may
    use dotted notation (``train.lr=0.0005``).  Values are parsed with
    :func:`ast.literal_eval` when possible.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got '{item}'")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            import ast

            node[parts[-1]] = ast.literal_eval(value)
        except Exception:
            node[parts[-1]] = value
    return cfg


def save_config(cfg: Dict, path: str) -> None:
    import yaml

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_task_config(cfg: Dict, task: str | None) -> Dict:
    """Merge the task specific section of an MSD-style configuration.

    A configuration may contain a ``tasks`` mapping; ``resolve_task_config``
    returns the global settings merged with ``tasks[task]``.
    """
    if not task:
        return cfg
    tasks = cfg.get("tasks", {})
    if task not in tasks:
        raise KeyError(f"task '{task}' not found; available: {sorted(tasks)}")
    base = {k: v for k, v in cfg.items() if k != "tasks"}
    merged = merge(base, tasks[task])
    merged["task"] = task
    return merged


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
class AverageMeter:
    """Track the running mean of a scalar."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class Logger:
    """Minimal console + file logger."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model, optimizer=None, iteration: int = 0, best: float = -1.0, cfg=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "iteration": iteration,
        "best_dice": best,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if cfg is not None:
        payload["config"] = cfg
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu") -> Dict:
    payload = torch_load(path, map_location=map_location)
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


# --------------------------------------------------------------------------- #
# mixed precision and checkpoint helpers (torch >= 1.12 compatible)
# --------------------------------------------------------------------------- #
def amp_autocast(device_type: str, enabled: bool = True):
    """``torch.amp.autocast`` with a fallback for torch < 2.0."""
    if not enabled:
        import contextlib

        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type, enabled=True)
    if str(device_type).startswith("cuda"):
        return torch.cuda.amp.autocast(enabled=True)
    import contextlib

    return contextlib.nullcontext()


def amp_grad_scaler(device_type: str, enabled: bool = True):
    """``torch.amp.GradScaler`` with a fallback for torch < 2.0."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled and str(device_type).startswith("cuda"))


def torch_load(path: str, map_location="cpu"):
    """``torch.load`` that also works on torch < 2.0 (no ``weights_only``)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def gpu_memory_gb() -> float:
    """Total memory of the first CUDA device in GB (0 when running on CPU)."""
    if torch is None or not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / 1024 ** 3


# --------------------------------------------------------------------------- #
# model statistics
# --------------------------------------------------------------------------- #
def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_shape=(1, 1, 96, 96, 96)) -> float:
    """Approximate GFLOPs for a forward pass (convolution and linear layers).

    The counter follows the usual convention of counting multiply-accumulate
    operations as two floating point operations and ignores normalization and
    attention layers, which is accurate enough for comparing configurations.
    """
    total = {"flops": 0.0}

    def conv_hook(module, inputs, outputs):
        out = outputs
        kernel = np.prod(module.kernel_size)
        total["flops"] += 2.0 * out.numel() * (module.in_channels / module.groups) * kernel

    def linear_hook(module, inputs, outputs):
        total["flops"] += 2.0 * outputs.numel() * module.in_features

    handles = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    return total["flops"] / 1e9
