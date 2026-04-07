# Table 1: Dataset Statistics

## EEG Segments by Subtype

| | LPD | GPD | LRDA | GRDA | Total |
|---|---:|---:|---:|---:|---:|
| Segments | 4,170 | 3,337 | 1,408 | 3,323 | 12,238 |
| Unique patients | 3,953 | 3,086 | 1,381 | 3,159 | — |

All segments: 19-channel monopolar EEG, 10 seconds at 200 Hz (2,000 samples). Common average reference montage.

## Label Coverage

| Label type | LPD | GPD | LRDA | GRDA | Total |
|---|---:|---:|---:|---:|---:|
| Expert frequency | 1,505 | 1,545 | 654 | 1,381 | 5,085 |
| Spatial annotations | — | — | — | — | 669 |
| Laterality | — | — | — | — | 3,418 |
| Discharge timing | ~1,100 | ~1,060 | — | — | 2,159 |
| Wave timing | — | — | 189 | 279 | 502 |
| IIIC crowd votes (≥10 raters) | 1,846 | 1,024 | 275 | 433 | 3,578 |

## Data Provenance

| Source | Segments | Description |
|---|---:|---|
| IIIC crowd-labeled | 4,088 | 10-min recordings, center 10s extracted; ≥10 expert votes per segment |
| MW-labeled | ~8,000 | From pattern-specific S3 folders; single-rater classification |
| Expert dataset | 328 | 38-patient dataset; 4 raters (LB, PH, SZ, MW) |
