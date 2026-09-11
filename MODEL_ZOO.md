# Model zoo — trained weights

The checkpoints of the paper are released as **release assets** (GitHub) and as a **Zenodo
deposit**; they are not stored in the git history because each file is 45–90 MB. This page is the
model card of the release: it lists the expected artifacts, where they are hosted and how to
verify them.

> **Status.** This repository ships the complete, verified training and evaluation pipeline. The
> checkpoints below are produced with `python scripts/train_all.py` (see
> `docs/TRAINING_GUIDE.md`, ≈ 9 h per experiment on an 8 GB GPU). The weights are **available from
> the corresponding author on reasonable request**; they are uploaded here (together with the
> Zenodo record) once the remaining experiments have been trained.

---

## 1. Artifacts

| # | Experiment | Config | Checkpoint | Reported (paper) | SHA256 |
|---|---|---|---|---|---|
| 1 | BTCV13 | `configs/btcv13.yaml` | `btcv13/best.pth` | DSC 81.77, HD95 8.08 mm | – |
| 2 | MSD Task01 Brain Tumour | `configs/msd.yaml --task task01_braintumour` | `msd/task01_braintumour/best.pth` | Avg. DSC 79.21 | – |
| 3 | MSD Task02 Heart | `… task02_heart` | `msd/task02_heart/best.pth` | DSC 93.01 | – |
| 4 | MSD Task03 Liver | `… task03_liver` | `msd/task03_liver/best.pth` | Avg. DSC 78.93 | – |
| 5 | MSD Task04 Hippocampus | `… task04_hippocampus` | `msd/task04_hippocampus/best.pth` | Avg. DSC 87.98 | – |
| 6 | MSD Task05 Prostate | `… task05_prostate` | `msd/task05_prostate/best.pth` | Avg. DSC 65.43 | – |
| 7 | MSD Task06 Lung | `… task06_lung` | `msd/task06_lung/best.pth` | DSC 73.98 | – |
| 8 | MSD Task07 Pancreas | `… task07_pancreas` | `msd/task07_pancreas/best.pth` | Avg. DSC 52.12 | – |
| 9 | MSD Task08 Hepatic Vessel | `… task08_hepaticvessel` | `msd/task08_hepaticvessel/best.pth` | Avg. DSC 54.37 | – |
| 10 | MSD Task09 Spleen | `… task09_spleen` | `msd/task09_spleen/best.pth` | DSC 96.75 | – |
| 11 | MSD Task10 Colon | `… task10_colon` | `msd/task10_colon/best.pth` | DSC 49.87 | – |
| 12–16 | Ablation (`baseline`, `scea`, `salt`, `lgca`, `all`) on BTCV13 | `configs/btcv13.yaml --variant …` | `btcv13_<variant>/best.pth` | see Table 5 | – |

Every checkpoint stores the model weights, the optimizer state, the iteration, the best validation
DSC and the fully resolved configuration of the run.

---

## 2. How to use a checkpoint

```bash
# evaluation (DSC + HD95 for every region, matching the tables of the paper)
python evaluate.py --config configs/btcv13.yaml --ckpt checkpoints/btcv13/best.pth --out results.json

# single volume inference
python predict.py --config configs/btcv13.yaml --ckpt checkpoints/btcv13/best.pth \
    --input /path/to/img0027.nii.gz --out pred_img0027.nii.gz
```

The expected directory layout in the repository is `checkpoints/<experiment>/best.pth`.

---

## 3. Model card

| Field | Value |
|---|---|
| Architecture | SCLA-Net: depthwise-separable encoder–decoder with SCEA, SALT and LGCA (`sclanet/models/`) |
| Parameters | printed by `scripts/model_stats.py` for the released configuration |
| Input | 96 × 96 × 96 patches, intensity normalized to [0, 1] (CT: clipped to [-175, 250] HU) |
| Output | per-voxel class probabilities (14 classes for BTCV13) |
| Training data | BTCV13 (24 volumes) or the training part of the respective MSD task, see `splits/` |
| Training protocol | AdamW, lr 1e-4, weight decay 1e-5, batch size 2, 40 000 iterations, mixed precision |
| Inference | 96³ sliding window, 0.5 overlap, averaging, no test-time augmentation |
| Licence | MIT (weights); the datasets keep their own licences |
| Limitations | trained for abdominal CT / the specific MSD tasks; not validated for other modalities, scanners or pathologies |

---

## 4. Publishing a new checkpoint

```bash
# 1. package the weights with their configuration
zip -j btcv13.zip checkpoints/btcv13/best.pth checkpoints/btcv13/config.yaml

# 2. checksums (fill the SHA256 column above)
sha256sum btcv13.zip > btcv13.zip.sha256

# 3. attach to the GitHub release
gh release upload v1.0.0 btcv13.zip btcv13.zip.sha256

# 4. (optional) the Zenodo deposit created from the same release gets the DOI
```

After uploading, update section 1 of this file (status, SHA256) and the DOI placeholders in
`README.md`, `CITATION.cff` and `.zenodo.json`.
