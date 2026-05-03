# Table 1: Dataset Statistics

*Auto-generated from `data/labels/segment_labels.csv` and `annotations.csv`.*
*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table1.py`*

## EEG Segments by Subtype

| | LPD | GPD | LRDA | GRDA | Total |
|---|---:|---:|---:|---:|---:|
| Segments (non-excluded) | 4,201 | 3,348 | 1,489 | 3,387 | 12,425 |
| Excluded | 115 | 102 | 483 | 103 | 803 |
| Unique patients | 3,955 | 3,093 | 1,462 | 3,220 | 11,729 |

All segments: 19-channel monopolar EEG, 10 seconds at 200 Hz (2,000 samples). Common average reference montage.

## Label Coverage by Subtype and Task

| Label type | LPD | GPD | LRDA | GRDA | Total |
|---|---:|---:|---:|---:|---:|
| Expert-reviewed frequency | 1,530 | 1,110 | 654 | 1,386 | 4,680 |
| Algorithm frequency | — | — | 1,310 | 3,222 | 4,532 |
| Any frequency | 1,530 | 1,110 | 1,408 | 3,328 | 7,376 |
| Discharge timing | 1,372 | 1,504 | 29 | 182 | 3,087 |
| Wave timing | — | — | 189 | 313 | 502 |
| Channel involvement / spatial | 381 | 264 | 29 | 182 | 856 |
| Laterality | 1,364 | 253 | 1,089 | 789 | 3,495 |
| IIIC crowd votes (≥10 raters) | 1,848 | 1,025 | 319 | 477 | 3,669 |

## Rater Contributions (annotations.csv)

| Rater | LPD | GPD | LRDA | GRDA | Total |
|---|---:|---:|---:|---:|---:|
| LB | 303 | 262 | 210 | 285 | 1,060 |
| MW | 1,888 | 1,944 | 1,079 | 1,983 | 7,547 |
| PH | 303 | 262 | 210 | 285 | 1,060 |
| SZ | 303 | 262 | 210 | 285 | 1,060 |
| **Total annotations** | | | | | **10,727** |

## Model Training Data

| Model | Segments | Selection criteria |
|---|---:|---|
| ChannelPD-Net | 2,640 | LPD/GPD with expert frequency |
| HemiCET-UNet | 675 | Ground-truth discharge timing |
| CET-UNet | 675 | Ground-truth discharge timing |

## Data Provenance

| Source | Description |
|---|---|
| IIIC crowd-labeled | 10-min recordings, center 10s extracted; ≥10 expert votes per segment |
| MW-labeled | From pattern-specific S3 folders; single-rater classification |
| Expert dataset | 38-patient dataset; 4 raters (LB, PH, SZ, MW) |
