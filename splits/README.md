# Train / validation splits

Every file contains one case identifier per line. They are the exact splits behind the numbers
reported in the paper and are reproduced deterministically by `scripts/make_splits.py`.

## BTCV13 (24 / 6)

| file | cases |
|---|---|
| `btcv13_train.txt` | `img0001` … `img0024` (24 volumes) |
| `btcv13_val.txt` | `img0025` … `img0030` (6 volumes) |

The identifiers are fixed by the Synapse BTCV release (MICCAI 2015 Multi-Atlas Abdomen Labeling
Challenge, `syn3193805`), so these two files are shipped with the repository and no raw data is
required to reproduce them.

## Medical Segmentation Decathlon (80 / 20 per task)

| file | task |
|---|---|
| `msd_task01_braintumour_{train,val}.txt` | Task01 Brain Tumour |
| `msd_task02_heart_{train,val}.txt` | Task02 Heart |
| `msd_task03_liver_{train,val}.txt` | Task03 Liver |
| `msd_task04_hippocampus_{train,val}.txt` | Task04 Hippocampus |
| `msd_task05_prostate_{train,val}.txt` | Task05 Prostate |
| `msd_task06_lung_{train,val}.txt` | Task06 Lung |
| `msd_task07_pancreas_{train,val}.txt` | Task07 Pancreas |
| `msd_task08_hepaticvessel_{train,val}.txt` | Task08 Hepatic Vessel |
| `msd_task09_spleen_{train,val}.txt` | Task09 Spleen |
| `msd_task10_colon_{train,val}.txt` | Task10 Colon |

The case identifiers of the MSD archives are listed, sorted, shuffled with a **fixed seed (0)**
and cut at 80% for training / 20% for validation:

```bash
python scripts/make_splits.py --config configs/msd.yaml --raw-dir /path/to/MSD
```

Only the labelled cases (`imagesTr`, i.e. the cases for which the challenge provides a mask in
`labelsTr`) are used; the unlabelled test cases of the challenge are never seen during training
or validation. Because sorting and seeding are fixed, re-running the command on the official
archives reproduces exactly the same files, which makes the split verifiable independently.

`train.py` falls back to this same deterministic rule if a split file is missing, so training
always uses the same partition.
