# Independent expert v1 — raw input archive

This folder is the audit trail for the independent-expert annotation
study described in [paper_materials/independent_expert_tasks/](../../../../paper_materials/independent_expert_tasks/).
The canonical labels live in `data/labels/labels.csv` tagged
`round='independent_expert_v1'`; this folder preserves the original
JSON exports from each rater so the ingest is reproducible and any
future re-analysis can re-derive the canonical labels from source.

## Layout

```
independent_expert_v1/
├── README.md           ← this file
├── MW/                 ← reserved for any LRDA/GRDA frequency catch-up
├── SZ/                 ← Sahar Zafar's exports (received 2026-04-27)
│   ├── lpd_freq_timing_batch1_results.json    (200 entries)
│   ├── gpd_freq_timing_batch1_results.json    (200 entries)
│   ├── rda_freq_labeling_results-2.json       (400 entries: LRDA + GRDA combined)
│   └── rda_freq_labeling_results.json         (200 LRDA only — superseded earlier export, kept for audit)
└── TZ/                 ← Tianyu Zhang's exports (received 2026-04-13)
    ├── lpd_freq_timing_results_TZ.json        (200 entries)
    ├── gpd_freq_timing_results_TZ.json        (200 entries)
    ├── lrda_freq_labeling_results_TZ.json     (200 entries)
    └── grda_freq_labeling_results_TZ.json     (200 entries)
```

The 200-segment subsets per task are defined in
[paper_materials/independent_expert_tasks/<subtype>/manifest.csv](../../../../paper_materials/independent_expert_tasks/);
each entry in the JSON files corresponds to one row of that manifest.

## Schemas

### TZ PD viewer files (`{lpd,gpd}_freq_timing_results_TZ.json`)

Top-level: `{ patient_id: entry }`. Each entry has:

| field | description |
|---|---|
| `segment_id`, `patient_id` | segment identifier |
| `est_freq` | algorithm's pre-filled frequency |
| `selected_freq` | rater's chosen frequency (Hz) |
| `laterality` | `'left'` / `'right'` (LPD only; null for GPD) |
| `laterality_index` | algorithm's pre-filled laterality score |
| `global_times` | list of discharge times in seconds (or `[]` if rejected) |
| `rejected` | bool |
| `review_status` | `'rejected'`, `'ground_truth'`, etc. |

### TZ RDA viewer files (`{lrda,grda}_freq_labeling_results_TZ.json`)

Top-level: `{ mat_file_stem: entry }`. Each entry has:

| field | description |
|---|---|
| `mat_file`, `patient_id`, `segment_id`, `subtype` | segment identifiers |
| `freq` | rater's chosen frequency (Hz) |
| `laterality` | `'left'` / `'right'` / null (LRDA only) |
| `w05_freq`, `w05_laterality` | algorithm's W05 / NB-Hilbert prediction |
| `tautan_freq` | predecessor signal-processing baseline prediction |
| `action` | `'accept'` for retained labels; `'reject_not_rda'` for rejections |

### SZ files

Same schemas as TZ. The combined `rda_freq_labeling_results-2.json`
holds 400 entries (200 LRDA + 200 GRDA) discriminated by the
per-entry `subtype` field. The `rda_freq_labeling_results.json`
(200 LRDA only) is an earlier partial export and is **superseded**
by the `-2` file; it is kept here for audit but not used by the
ingester.

## Ingest

Run:

```bash
conda run -n morgoth python code/data_management/ingest_independent_expert_v1.py
```

This appends rows to `data/labels/labels.csv` with
`round='independent_expert_v1'`. The script is idempotent — re-running
it after the data is already ingested adds zero rows.

The ingester maps:

| JSON field | labels.csv `label_type` | Notes |
|---|---|---|
| `selected_freq` (PD) / `freq` (RDA) | `frequency_hz` | only for accepted entries |
| `laterality` | `laterality` | only LPD and LRDA (string `'left'` / `'right'`) |
| `global_times` | `discharge_times` | JSON-encoded list, PD only |

Rejected entries are skipped (absence of a row = "not labeled by this rater").

After ingesting, run:

```bash
conda run -n morgoth python code/data_management/build_segment_labels.py
```

to regenerate `data/labels/segment_labels.csv` (the consolidated view).

## Coverage

After ingest, coverage of the 200-segment subsets per (subtype, rater, label_type):

| subtype | rater | frequency_hz | laterality | discharge_times |
|---|---|---:|---:|---:|
| LPD  | MW | 130 | 85  | 86  |
| LPD  | SZ | 170 | 168 | 168 |
| LPD  | TZ | 179 | 179 | 179 |
| GPD  | MW | 150 | —   | 127 |
| GPD  | SZ | 190 | —   | 190 |
| GPD  | TZ | 187 | —   | 187 |
| LRDA | MW | 0   | 126 | —   |
| LRDA | SZ | 112 | 112 | —   |
| LRDA | TZ | 144 | 144 | —   |
| GRDA | MW | 0   | —   | —   |
| GRDA | SZ | 130 | —   | —   |
| GRDA | TZ | 160 | —   | —   |

(— means the field is not applicable for that subtype — GPD is bilateral
by definition, GRDA is generalized, RDA tasks have no discharge timing.)

**Known gap:** MW has zero frequency labels for the 200-segment LRDA
and GRDA subsets, because those manifests were stratified on
`algo_freq_hz` rather than on MW labels. To complete the 4-way
(MW × SZ × TZ × algo) frequency comparison on RDA, MW would need to
label those 400 segments. Without that, RDA-frequency analysis
includes only SZ-vs-TZ-vs-algo. MW LPD/GPD frequency coverage is
also less than 200/200 because MW only reviewed a subset during the
original annotation rounds.
