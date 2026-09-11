# SCLA-Net

Official implementation of

> **SCLA-Net: A Lightweight Structure-aware Context and Localization Aggregation Network
> for 3D Medical Image Segmentation**

SCLA-Net is a lightweight 3D segmentation network for CT and MRI volumes. It combines three
components inside a depthwise-separable encoder–decoder backbone:

| Component | Section of the paper | Code |
|---|---|---|
| **SCEA** – Structural Context Enhanced Attention: directional pooling along the three orthogonal anatomical axes, adaptive fusion of the resulting contexts and structure-guided attention recalibration | *Structural Context Enhanced Attention (SCEA)* | `sclanet/models/scea.py` |
| **SALT** – Structure-aware Localization Token Generator: anatomical response map, diversity-aware center selection and adaptive local aggregation into `K` localization tokens | *Structure-aware Localization Token Generator (SALT)* | `sclanet/models/salt.py` |
| **LGCA** – Localization-guided Cross Attention: the SALT tokens act as queries that selectively aggregate structure-enhanced features | *Localization-guided Cross Attention* | `sclanet/models/lgca.py` |
| **LAL** – Localization-aware auxiliary loss (`L = L_Dice + L_CE + lambda_loc * L_BCE`) | *Localization-aware Optimization Loss* | `sclanet/losses.py` |

Everything needed to reproduce the experiments (preprocessing, exact data splits, training and
inference configuration, evaluation metrics) is contained in this repository.

---

## 1. Repository layout

```
SCLA-Net/
├── configs/                 YAML configuration files (all hyper-parameters)
│   ├── btcv13.yaml          BTCV13
│   └── msd.yaml             the ten MSD tasks (per-task label definitions)
├── splits/                  train/validation split files (one case id per line)
├── sclanet/
│   ├── models/              network, SCEA, SALT, LGCA, backbone blocks
│   ├── data/                preprocessing, dataset, augmentation, splits
│   ├── losses.py            Dice + CE + localization-aware loss
│   ├── metrics.py           DSC and HD95
│   ├── inference.py         sliding-window inference
│   └── utils.py             config, seeding, logging, checkpoints, FLOPs counter
├── scripts/
│   ├── prepare_data.py      raw NIfTI -> preprocessed .npy (datasets are NOT downloaded)
│   ├── make_splits.py       writes the exact train/val split files
│   ├── make_dummy_data.py   synthetic data for smoke tests
│   └── model_stats.py       parameters / GFLOPs of every ablation variant
├── train.py                 training entry point
├── evaluate.py              sliding-window evaluation (DSC, HD95)
├── predict.py               single-volume inference
└── RELEASE.md               how to publish this code and get a Zenodo DOI
```

---

## 2. Installation

The code runs with Python ≥ 3.9 and PyTorch ≥ 2.0 (CPU or CUDA).

```bash
conda create -n sclanet python=3.10
conda activate sclanet
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or the CPU wheel
```

`requirements.txt` only lists `torch`, `numpy`, `scipy`, `nibabel` and `PyYAML`; no additional
framework (MONAI, nnU-Net, …) is required.

---

## 3. Data

**The datasets are not redistributed with this repository**; download them from their official
sources and comply with the respective licences.

| Dataset | Content | Access route | Licence / terms |
|---|---|---|---|
| **BTCV13** | 30 labelled abdominal CT volumes, 13 organs | Synapse: <https://www.synapse.org/Synapse:syn3193805> (MICCAI 2015 Multi-Atlas Abdomen Labeling Challenge) | Synapse Terms and Conditions of Use, free for research use |
| **MSD** | 10 segmentation tasks (brain tumour, heart, liver, hippocampus, prostate, lung, pancreas, hepatic vessel, spleen, colon) | <http://medicaldecathlon.com/> (also mirrored on Synapse: `syn26350678`) | Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0) |

Both datasets are fully de-identified by their providers.

### Preprocessing

```bash
# BTCV13
python scripts/prepare_data.py --config configs/btcv13.yaml \
    --raw-dir /path/to/BTCV/Training --out data/preprocessed/btcv13

# MSD (one command per task, or loop over the ten tasks)
python scripts/prepare_data.py --config configs/msd.yaml --task task03_liver \
    --raw-dir /path/to/MSD --out data/preprocessed/msd
```

The pipeline is implemented in `sclanet/data/preprocess.py` and performs, in order:

1. reorientation to the canonical (RAS) axis order,
2. optional resampling to a target voxel spacing (`data.spacing`, `null` = native spacing),
3. intensity preprocessing
   * CT tasks: clipping to **[-175, 250] HU** followed by min-max normalization to **[0, 1]**,
   * MRI tasks (MSD Task01/02/04/05): percentile clipping (0.5–99.5%) + min-max normalization,
4. storage as `float16` `.npy` files plus a `meta.json` that records the voxel spacing used to
   express the HD95 in millimetres.

Multi-modal tasks are supported: MSD Task01 uses all four MRI modalities and Task05 both
channels (`model.in_channels` in `configs/msd.yaml`).

### Exact data splits

The split files used for every experiment are stored in `splits/` and are fully deterministic:

* **BTCV13** – `img0001`–`img0024` train, `img0025`–`img0030` validation (24/6 protocol).
  These identifiers are fixed by the Synapse release, so the files are shipped with the
  repository.
* **MSD** – for each task the labelled case identifiers are sorted, shuffled with a fixed seed
  (`--seed 0`) and divided into 80% training / 20% validation. Run

  ```bash
  python scripts/make_splits.py --config configs/msd.yaml --raw-dir /path/to/MSD
  ```

  once after downloading the data; the resulting `splits/msd_<task>_{train,val}.txt` files are
  the exact splits used in the paper (the rule is deterministic, so they are identical on every
  machine). `train.py` will also derive the split automatically if the files are absent.

---

## 4. Training

```bash
# BTCV13, full model
python train.py --config configs/btcv13.yaml

# a single MSD task
python train.py --config configs/msd.yaml --task task03_liver
```

The configuration mirrors the *Experimental Setup* section of the paper:

| Setting | Value |
|---|---|
| patch size | 96 × 96 × 96 |
| batch size | 2 |
| optimizer | AdamW |
| learning rate | 1 × 10⁻⁴ |
| weight decay | 1 × 10⁻⁵ |
| iterations | 40 000 |
| mixed precision | enabled |
| model selection | highest validation DSC |
| inference | sliding window 96³, 0.5 overlap, overlapping predictions averaged, no TTA |

Any value can be overridden on the command line, e.g.

```bash
python train.py --config configs/btcv13.yaml --set train.lr=5e-5 --set model.num_tokens=32
```

Checkpoints, the resolved configuration and the training log are written to
`checkpoints/<experiment>/` (`best.pth`, `last.pth`, `config.yaml`, `train.log`).

To train every experiment of the paper in one resumable command, and for hardware guidance
(measured speed, peak GPU memory, gradient accumulation on small GPUs), see
[`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md):

```bash
python scripts/train_all.py                                     # all eleven experiments
python scripts/benchmark.py --config configs/btcv13.yaml --amp   # speed + peak GPU memory
```

### Ablation variants

The component-wise ablation of the paper is reproduced with `--variant`:

```bash
python train.py --config configs/btcv13.yaml --variant baseline   # lightweight backbone only
python train.py --config configs/btcv13.yaml --variant scea       # + SCEA
python train.py --config configs/btcv13.yaml --variant salt       # + SALT (LAL enabled)
python train.py --config configs/btcv13.yaml --variant lgca       # + LGCA with learnable queries
python train.py --config configs/btcv13.yaml --variant all        # full model
```

The token count `K` and the aggregation radius `r` of Table "Ablation of K and r" are controlled
by `model.num_tokens` and `model.salt_kwargs.radius`; the localization loss weight is
`loss.lambda_loc`.

---

## 5. Evaluation

```bash
python evaluate.py --config configs/btcv13.yaml --ckpt checkpoints/btcv13/best.pth
python evaluate.py --config configs/msd.yaml --task task03_liver --ckpt checkpoints/msd/task03_liver/best.pth
```

Per-case and per-region Dice and HD95 (in millimetres, using the voxel spacing stored during
preprocessing) are printed and can be exported with `--out results.json`. For MSD Task01 the
overlapping regions WT = {1,2,3}, TC = {2,3} and NET = {2} are evaluated exactly as in the paper;
the remaining tasks use the label definitions listed in `sclanet/data/splits.py`.

---

## 6. Inference on a single volume

```bash
python predict.py --config configs/btcv13.yaml --ckpt checkpoints/btcv13/best.pth \
    --input /path/to/img0027.nii.gz --out pred_img0027.nii.gz
```

---

## 7. Implementation notes

The paper describes the architecture and the training protocol; the following choices had to be
made where the text leaves room for interpretation. They are collected here so that the results
can be reproduced unambiguously:

1. **SCEA attention.** Full `N × N` spatial attention over a 96³ volume is not tractable, and the
   paper does not state a spatial reduction. The attention is computed on an adaptively pooled
   grid with at most `model.scea_attn_grid` voxels per axis and the resulting attention map is
   trilinearly upsampled. For the bottleneck (6³ voxels at 96³ patches) with the default
   `attn_grid: 8` no pooling is applied, i.e. the computation is exact there.
2. **SCEA reduction ratio.** `scea_reduction` (16 in the encoder/decoder stages, 4 at the
   bottleneck) is applied to the query/key/value projections and to the projection of the
   recalibrated feature, and the module output is projected back to `C` channels by a depthwise
   separable convolution.
3. **Anatomical importance target map `H`.** The paper states that `H` is constructed from the
   ground-truth mask `Y` but does not specify the construction. The default is the Gaussian
   smoothed binary foreground-union map (`loss.target: gaussian`, `loss.target_sigma: 1.0`);
   `binary` is available as an alternative. `H` is resampled to the resolution of the response
   map before the BCE is evaluated.
4. **`lambda_loc`.** The weight of the auxiliary localization loss is not given in the paper;
   the released configuration uses `0.5` (`loss.lambda_loc`).
5. **Affine augmentation probability.** Probabilities are given for flipping (p = 0.1), 90°
   rotation (p = 0.1) and intensity shifting (σ = 0.1, p = 0.5); the affine augmentation is
   applied with `p = 0.1` (`data.augment.affine_p`), rotation ≤ π/30 and scaling in [0.9, 1.1].
6. **Positional embedding of the centers.** A sinusoidal/Fourier encoding of the normalized
   `(z, y, x)` coordinates followed by a two-layer MLP (`sclanet/models/salt.py`).
7. **Center selection is non-differentiable**, as stated in the paper: gradients reach SALT
   through the features gathered around the selected centers and through the auxiliary loss on
   the dense response map, not through the `argmax` steps.
8. **Patch sampling** prefers patches with a foreground fraction above 2% (up to 20 attempts);
   the foreground fraction of the sampled patches is not specified in the paper.

---

## 8. Trained weights

The trained checkpoints are **not** shipped with this repository, because each file is 45–90 MB.
They are available from the corresponding author on reasonable request. Everything needed to
reproduce them is public: the network code, the preprocessing pipeline, the exact data splits,
the configuration files and the training script (`scripts/train_all.py`, see
[`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md)).

[`MODEL_ZOO.md`](MODEL_ZOO.md) is the model card of the release: it lists the expected artifact
names, the configuration each checkpoint belongs to, the metric reported in the paper and the
commands used to package and publish them once they are ready.

```bash
# once a checkpoint has been obtained, place it at checkpoints/<experiment>/best.pth
python evaluate.py --config configs/btcv13.yaml --ckpt checkpoints/btcv13/best.pth
```

A checkpoint contains the model state dict, the optimizer state, the iteration, the best
validation DSC and the fully resolved configuration of the run, so every reported number can be
traced back to the exact training settings.

---

## 9. Acknowledgements

This work was supported by the Liaoning Provincial Science and Technology Programme
(Grant No. 2023JH1/10400091) and the Liaoning Provincial Department of Education Project
(Grant No. LJ212410142096).

## 10. Licence

Released under the MIT Licence (see `LICENSE`). The datasets keep their own licences
(BTCV13: Synapse terms of use; MSD: CC BY-SA 4.0).
