# Training guide

Everything needed to produce the checkpoints of the paper. The protocol is fixed by
`configs/*.yaml` (AdamW, lr 1e-4, weight decay 1e-5, batch size 2, 40 000 iterations, mixed
precision, 96³ patches, best checkpoint selected by validation DSC), so the only decisions left
are *where* to run and *which* experiments.

---

## 1. Environment

Any environment with **PyTorch ≥ 1.12** (CPU or CUDA), `numpy`, `scipy` and `PyYAML` works. The
code contains compatibility shims for the mixed-precision and checkpoint APIs, so torch 1.12 and
torch 2.x behave identically.

```bash
pip install -r requirements.txt
```

Known pitfalls seen on Windows workstations:

* **numpy 2.x with torch ≤ 1.13** – the numpy↔torch bridge is broken
  (`RuntimeError: Numpy is not available`). Fix it inside the environment:

  ```bash
  pip install "numpy<2"
  ```

* **broken `triton` install** – `torch.optim.AdamW` fails with
  `ModuleNotFoundError: No module named 'triton.backends'`. Either repair/remove `triton` or use
  another environment.

Check that the environment is usable before a long run:

```bash
python -c "import torch, numpy as np; print(torch.__version__, torch.cuda.is_available(), torch.from_numpy(np.zeros(2)).shape)"
python scripts/model_stats.py --config configs/btcv13.yaml
python scripts/benchmark.py   --config configs/btcv13.yaml --amp      # speed + peak memory
```

---

## 2. How long does training take?

Measured with `scripts/benchmark.py` (96³ patches, batch size 2, mixed precision):

| Hardware | s / iteration | 40 000 iterations | peak GPU memory |
|---|---|---|---|
| RTX 4060 Laptop (8 GB) | 0.81 | ≈ 9.0 h | 3.19 GB |
| RTX 4060 Laptop, batch 1 | 0.40 | ≈ 4.5 h | 1.61 GB |
| CPU (16 threads) | ≈ 25 | ≈ 11 days | – |

The model fits comfortably in 8 GB at the paper's patch size and batch size; no gradient
checkpointing or model parallelism is required. Validation (sliding-window inference over the validation cases)
adds a few minutes per checkpoint and is included in `train.val_interval`.

---

## 3. Data

Download the datasets from their official sources (see section 3 of the README), then preprocess
once — the `.npy` files are small and make training I/O bound at most:

```bash
# BTCV13 (30 labelled volumes)
python scripts/prepare_data.py --config configs/btcv13.yaml \
    --raw-dir /path/to/BTCV/Training --out data/preprocessed/btcv13

# MSD: repeat for the ten tasks (Task01 ... Task10)
python scripts/prepare_data.py --config configs/msd.yaml --task task03_liver \
    --raw-dir /path/to/MSD --out data/preprocessed/msd

# split files
python scripts/make_splits.py --btcv-only
python scripts/make_splits.py --config configs/msd.yaml --raw-dir /path/to/MSD
```

---

## 4. Running the experiments

Single experiment:

```bash
python train.py --config configs/btcv13.yaml
python train.py --config configs/msd.yaml --task task03_liver
```

All eleven experiments of the paper, one after another (skipping the ones that are already
finished, `last.pth` is resumed automatically with `--resume`):

```bash
python scripts/train_all.py                 # everything
python scripts/train_all.py --only btcv13,msd:task03_liver
python scripts/train_all.py --dry-run       # print the commands without running them
```

Interrupted runs are resumed from the last checkpoint:

```bash
python train.py --config configs/btcv13.yaml --resume checkpoints/btcv13/last.pth
```

### Smaller GPUs (< 8 GB)

Keep the *effective* batch size of 2 by accumulating two micro-batches of one volume:

```bash
python train.py --config configs/btcv13.yaml \
    --set train.batch_size=1 --set train.grad_accum_steps=2
```

`scripts/benchmark.py` reports the peak memory of the current configuration and warns when it is
close to the device limit.

---

## 5. Outputs

```
checkpoints/<experiment>/
├── best.pth      model with the highest validation DSC
├── last.pth      latest model (used by --resume)
├── config.yaml   fully resolved configuration of the run
└── train.log     training log (loss components, validation DSC/HD95 per checkpoint)
```

`best.pth` contains the model state dict, the optimizer state, the iteration, the best validation
DSC and the resolved configuration, i.e. everything required to reproduce a reported number. To
verify a checkpoint:

```bash
python evaluate.py --config configs/btcv13.yaml --ckpt checkpoints/btcv13/best.pth --out results.json
```

---

## 6. Training schedule of the paper

| Experiment | Config / task | Reported DSC |
|---|---|---|
| BTCV13 | `configs/btcv13.yaml` | 81.77 (13-organ mean), HD95 8.08 mm |
| MSD ×10 | `configs/msd.yaml --task taskXX_*` | see Tables 3 and 4 of the paper |
| Ablations | `--variant baseline/scea/salt/lgca` | see Table 5 of the paper |

Re-running every experiment therefore takes roughly 11 × 9 h ≈ 4 days on a single RTX 4060; on a
cloud GPU (RTX 4090/3090) the same schedule finishes in about one day, and individual
experiments can be trained in parallel on several devices — the runs are completely independent
of each other.
