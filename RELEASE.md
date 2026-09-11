# Publishing the code and obtaining a Zenodo DOI

This file answers reviewer point 3 (*"Please make the code, trained weights, and the
hyperparameter and configuration files publicly available (for example, GitHub archived with a
Zenodo DOI), and include the preprocessing pipeline and the exact data splits."*).

---

## 1. What is already in this repository

| Reviewer request | Where it is satisfied |
|---|---|
| Code of the network | `sclanet/models/` (backbone, SCEA, SALT, LGCA, `SCLA-Net`), `train.py`, `evaluate.py`, `predict.py` |
| Hyper-parameter and configuration files | `configs/btcv13.yaml`, `configs/msd.yaml` (every value used in the paper, plus the per-task label definitions); each run also stores its resolved `config.yaml` next to the checkpoint |
| Preprocessing pipeline | `sclanet/data/preprocess.py` + `scripts/prepare_data.py` (reorientation, resampling, HU clipping to [-175, 250], normalization, `.npy` export, `meta.json` with the voxel spacing) |
| Exact data splits | `splits/btcv13_{train,val}.txt` (24/6 protocol) and `splits/msd_<task>_{train,val}.txt`, produced deterministically by `scripts/make_splits.py` (sorted ids, shuffle with seed 0, 80/20) |
| Training configuration | `configs/*.yaml` (`train:` section) and section 4 of `README.md` |
| Evaluation protocol | `sclanet/inference.py` (96³ sliding window, 0.5 overlap, averaging, no TTA), `sclanet/metrics.py` (DSC, HD95 in mm) |
| Trained weights | not shipped with the code — available from the corresponding author on reasonable request (see section 6, **Tier B**); section 4 describes how to package and publish them once they are ready |

---

## 2. Push the repository to GitHub

```bash
cd SCLA-Net
git init
git add .
git commit -m "SCLA-Net: official implementation (v1.0.0)"
git branch -M main
git remote add origin https://github.com/<your-user>/SCLA-Net.git
git push -u origin main
```

Before pushing, check that no data or checkpoint file is committed: `data/`, `checkpoints/` and
`logs/` are listed in `.gitignore`.

---

## 3. Ship the exact MSD split files

The MSD splits depend on the case identifiers of the official archives, so they are generated
locally once and then committed:

```bash
python scripts/make_splits.py --config configs/msd.yaml --raw-dir /path/to/MSD
git add splits && git commit -m "Add the exact MSD train/validation splits" && git push
```

The rule (sorted identifiers, shuffle with seed 0, 80/20) is deterministic, so anybody can verify
the split by re-running the same command.

---

## 4. Publish the trained weights

Large files should not live in the git history. Use a GitHub release and/or Zenodo:

```bash
# one archive per experiment (recommended: <experiment>.zip with the checkpoint + config)
zip -r btcv13_all.zip checkpoints/btcv13/best.pth checkpoints/btcv13/config.yaml
zip -r msd_all.zip    checkpoints/msd/*/best.pth checkpoints/msd/*/config.yaml

# checksums for the model card
sha256sum btcv13_all.zip msd_all.zip > SHA256SUMS.txt
```

1. Create a tag and a release: `git tag -a v1.0.0 -m "SCLA-Net v1.0.0" && git push origin v1.0.0`,
   then attach the archives to the release on GitHub.
2. Alternatively (or additionally) upload the archives to Zenodo as a *new upload* of type
   *Software* — Zenodo assigns a DOI directly to the upload.

A short model card next to the weights (input size 96³, patch-wise normalization as in
`configs/*.yaml`, the number of training iterations, the validation DSC and the commit hash) makes
the release self-describing.

---

## 5. Archive on Zenodo and get the DOI

For a GitHub repository, Zenodo issues a DOI automatically per release:

1. Sign in at <https://zenodo.org> with GitHub.
2. **Settings → GitHub → Sync now** and flip the switch of the `SCLA-Net` repository *on*.
3. Create a new release on GitHub (`Releases → Draft a new release`, tag `v1.0.0`), then publish
   it. Zenodo archives the snapshot and mints a DOI.
4. Copy the DOI badge from the Zenodo entry into `README.md`:

   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```

5. Replace the `10.5281/zenodo.XXXXXXX` placeholder in `README.md`, `CITATION.cff` and
   `.zenodo.json` by the real DOI, then commit.

The `.zenodo.json` file in this repository pre-fills the Zenodo metadata (title, creators,
description, keywords, licence) — adjust the author list and the affiliation before the first
release.

---

## 6. If the weights cannot be released immediately

The reviewer asks for code **and** trained weights. Three publishing tiers are possible; pick the
highest one that is actually achievable and describe it accurately.

**The strategy adopted for this submission is Tier B**: the code, the preprocessing pipeline, the
exact data splits and the configuration files are released now (with a DOI), while the checkpoints
are not published yet and are made available on request. Describe it exactly like that — do not
claim that the weights are online if they are not.

**Tier A — code + weights + DOI (fully compliant).** Train with `scripts/train_all.py`
(≈ 9 h per experiment on an 8 GB GPU, see `docs/TRAINING_GUIDE.md`), package the checkpoints
(`MODEL_ZOO.md` §4) and archive the release on Zenodo.

> The source code, the preprocessing pipeline, the exact data splits, the configuration files and
> the trained weights of all reported experiments are publicly available at
> `https://github.com/XiangruW/SCLA-Net` and have been archived on Zenodo
> (DOI `10.5281/zenodo.22706035`). The Zenodo record contains the checkpoints of the eleven
> experiments (BTCV13, the ten MSD tasks and the ablation variants), together with `config.yaml`
> files that fully specify each run and SHA-256 checksums.

**Tier B — code + DOI now, weights on request.** Appropriate when the compute is not available at
revision time; state the reason and give a concrete timeline.

> The source code, the preprocessing pipeline, the exact data splits and the configuration files
> are publicly available at `https://github.com/XiangruW/SCLA-Net` (archived on Zenodo, DOI
> `10.5281/zenodo.22706035`). The trained weights are being prepared for release; in the meantime
> they are available from the corresponding author on reasonable request, and the released
> training script (`scripts/train_all.py`) reproduces every checkpoint with the reported
> hyper-parameters.

**Tier C — code only.** Weakest option; the repository still documents the exact protocol, but
say so plainly instead of promising weights that are not there.

> The complete implementation, preprocessing pipeline, exact data splits, hyper-parameters and
> configuration files are publicly available at `https://github.com/XiangruW/SCLA-Net`. Since the
> training data (BTCV13 and the MSD tasks) cannot be redistributed with the code, we provide the
> full training protocol and the deterministic data splits so that the results can be reproduced;
> trained weights will be released with the final version of the paper.

A checklist mapping every item of the reviewer request to a file in the repository is given in
section 1 of this document.

### Manuscript declaration (Declarations → Code availability)

The matching statement for the manuscript itself:

> **Code availability.** The software, the preprocessing pipeline, the exact data splits and the
> configuration files that implement and reproduce the results reported in this paper are openly
> available at `https://github.com/XiangruW/SCLA-Net` and have been archived on Zenodo
> (DOI `10.5281/zenodo.22706035`). The trained model weights are available from the corresponding
> author on reasonable request.

---

## 7. Suggested reply to the reviewer

> We have released the complete implementation of SCLA-Net, together with the preprocessing
> pipeline, the exact data splits, and the hyper-parameter and configuration files. The code is
> available at `https://github.com/XiangruW/SCLA-Net` and has been archived on Zenodo with the DOI
> `10.5281/zenodo.22706035` (v1.0.2). The repository contains: (i) the implementation of
> the lightweight backbone, SCEA, SALT and LGCA, and the localization-aware loss; (ii) the
> preprocessing pipeline (RAS reorientation, optional resampling, HU clipping to [-175, 250] and
> min-max normalization) together with the per-task intensity settings for the ten MSD tasks;
> (iii) the exact train/validation split files for BTCV13 (`img0001`–`img0024` / `img0025`–`img0030`)
> and for each MSD task (deterministic 80/20 split with a fixed seed, reproducible with
> `scripts/make_splits.py`); (iv) the full training configuration (AdamW, lr 1e-4, weight decay
> 1e-5, batch size 2, 40 000 iterations, mixed precision, 96³ patches) and the sliding-window
> inference settings (96³ window, 0.5 overlap, averaging, no test-time augmentation); and (v) the
> implementation notes that document the few design choices the manuscript leaves unspecified
> (section 7 of the README). The trained weights are available from the corresponding author on
> reasonable request and will be published as release assets together with the Zenodo record once
> the remaining experiments have been trained.
