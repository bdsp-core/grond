# Table 1: Dataset Statistics

*Auto-generated from `data/labels/segment_labels.csv` and `annotations.csv`.*
*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table1.py`*

## EEG Segments by Subtype

| | LPD | GPD | LRDA | GRDA | Total |
|---|---:|---:|---:|---:|---:|
| Segments (non-excluded) | 4,170 | 3,337 | 1,408 | 3,323 | 12,238 |
| Excluded | 146 | 113 | 564 | 167 | 990 |
| Unique patients | 3,953 | 3,086 | 1,381 | 3,159 | — |

All segments: 19-channel monopolar EEG, 10 seconds at 200 Hz (2,000 samples). Common average reference montage.

## Label Coverage by Subtype and Task

| Label type | LPD | GPD | LRDA | GRDA | Total |
|---|---:|---:|---:|---:|---:|
| Expert-reviewed frequency | 1,499 | 1,539 | 654 | 1,381 | 5,073 |
| Algorithm frequency | — | — | 754 | 1,942 | 2,696 |
| Any frequency | 1,505 | 1,545 | 1,408 | 3,323 | 7,781 |
| Discharge timing | 917 | 1,036 | 29 | 177 | 2,159 |
| Wave timing | — | — | 189 | 313 | 502 |
| Channel involvement / spatial | 352 | 260 | 29 | 177 | 818 |
| Laterality | 1,336 | 249 | 1,039 | 789 | 3,413 |
| IIIC crowd votes (≥10 raters) | 1,846 | 1,024 | 239 | 420 | 3,529 |

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
| ChannelPD-Net | 3,038 | LPD/GPD with expert frequency |
| HemiCET-UNet | 675 | Ground-truth discharge timing |
| CET-UNet | 675 | Ground-truth discharge timing |

## Data Provenance

| Source | Description |
|---|---|
| IIIC crowd-labeled | 10-min recordings, center 10s extracted; ≥10 expert votes per segment |
| MW-labeled | From pattern-specific S3 folders; single-rater classification |
| Expert dataset | 38-patient dataset; 4 raters (LB, PH, SZ, MW) |
