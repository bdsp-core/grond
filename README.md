# Automated Frequency Estimation for Periodic and Rhythmic EEG Patterns (LPD, GPD, LRDA, GRDA)

Algorithms for estimating the frequency of periodic discharges (PD) and rhythmic delta activity (RDA) in continuous EEG, developed for ICU EEG monitoring at Massachusetts General Hospital.

**Paper**: Tautan AM, Jing J, Basovic L, Hadar PN, Sartipi S, Fernandes MP, Kim J, Struck AF, Westover MB, Zafar SF. "Automated estimation of frequency and spatial extent of periodic and rhythmic epileptiform activity from continuous electroencephalography data." *Journal of Neural Engineering*, 22(6), 2025. [PubMed](https://pubmed.ncbi.nlm.nih.gov/41330044/)

## Status

| Task | Metric | N patients | Performance | Method |
|------|--------|-----------|-------------|--------|
| PD frequency (IPI) | Spearman rho | 581 | **0.819** | HemiCET+DP (8ch) |
| PD frequency (direct) | Spearman rho | 594 | **0.744** | CNN+Attention direct |
| Discharge timing | F1 | 675 | **0.717** | Full 18ch pipeline |
| Discharge timing (hemi) | F1 | 675 | **0.699** | HemiCET+DP (8ch) |
| Timing accuracy | MAE | 675 | **5.0 ms** (median) | HemiCET+DP (8ch) |
| Subtype (LPD vs GPD) | AUC | 594 | **0.931** | RF 300 trees |
| Laterality (L vs R) | AUC | 437 | **0.98** | CNN+Attention PD prob |
| Channel PD detection | AUC | 815 | **0.870** | CNN+Attention |
| RDA frequency | Spearman rho | 23 | **0.840** | FFT baseline |

**All methods use EEG-only input** — no gold standard labels provided as algorithm input. See [APPROACH_REVIEW_v13.md](APPROACH_REVIEW_v13.md) for details.

## Overview

The system extracts four properties from 10-second EEG segments:

1. **Frequency** of the epileptiform pattern (Hz)
2. **Spatial extent** (proportion of brain regions affected)
3. **Laterality** (unilateral, bilateral asymmetric, bilateral symmetric)
4. **Verbal description** following ACNS 2021 standardized nomenclature

Supported pattern types: LPD, GPD, LRDA, GRDA.

## Installation

```bash
git clone https://github.com/bdsp-core/IIIC-Frequency-Analysis-2.git
cd IIIC-Frequency-Analysis-2
conda env create -f code/environment.yml
conda activate foe
```

Key dependencies: Python 3.8, MNE 1.0.3, NumPy 1.24, SciPy 1.10, fooof 1.0, pyhht 0.1.

## Data Access

EEG data and expert annotations are stored on AWS S3 (not in this git repository due to size):

```bash
aws s3 sync s3://bdsp-opendata-credentialed/iiic-freq3/data/ data/
```

This requires AWS credentials with access to the `bdsp-opendata-credentialed` bucket. To request access, visit the [Brain Data Science Platform (BDSP)](https://bdsp.io).

The `data/` directory contains:
- `eeg/` — ~9,600+ .mat files (10s bipolar EEG segments, 18ch at 200 Hz, standardized format)
- `labels/` — canonical label files (see below)
- `dl_cache/`, `cet_cache/`, `pd_channel_cache/`, `hemi_cache/` — model weights
- `_archive/` — raw source annotation files

### Data Structure

```
data/
├── eeg/                  ~9,600 .mat files (18ch × 2000 samples, keys: data, Fs)
├── labels/
│   ├── patients.csv          Patient registry (2,865 rows: subtype, freq, laterality, exclusions, expert votes)
│   ├── segments.csv          Segment registry (3,313 rows: segment metadata, file paths)
│   ├── annotations.csv       Per-rater annotations (3,821 rows)
│   ├── discharge_times.json  Discharge timing labels (675 ground truth + 21 BIPD awaiting)
│   ├── list_events_20241129.xlsx  IIIC expert vote data (47,330 segments, 2,562 patients)
│   └── orphan_eeg_catalog.json   EEG files not yet in patients.csv
├── cet_cache/            CET-UNet model weights (5-fold)
├── pd_channel_cache/     CNN+Attention model weights (5-fold)
├── hemi_cache/           HemiCET model weights + experiment results
├── dl_cache/             External segment pool
└── _archive/             Raw source annotation files (by task/round)
```

### Current Label Coverage

| Type | Active | Frequency | Discharge Timing | Laterality |
|------|--------|-----------|-----------------|------------|
| **LPD** | 437 | 437 ✓ | 437 ✓ | 437 ✓ |
| **GPD** | 207 | 207 ✓ | 207 ✓ | N/A |
| **BIPD** | 21 | — | awaiting | N/A |
| **LRDA** | 99 | 4 | 4 | 0 |
| **GRDA** | 119 | 14 | 14 | N/A |

LPD and GPD are **fully labeled** with expert-reviewed frequency, discharge timing, and laterality (for LPD). Labels have been through 3 rounds of review using HemiCET model-assisted correction.

An additional ~8,000 IIIC dataset segments (LPD, GPD, LRDA, GRDA) are being downloaded from S3 with IIIC expert vote data for training.

### Label Management

The three files in `data/labels/` are the canonical label set for the project. All analysis code reads from these files. When new annotations are collected, they are integrated into these files following this process:

**The three canonical label files:**

- **`patients.csv`** — One row per patient. Contains: `patient_id`, `subtype`, `subtype_original`, `n_segments`, `n_raters`, `raters`, `gold_standard_freq`, `gold_standard_freq_original`, `excluded`, `exclusion_reason`, `laterality`, `laterality_original`, `subtype_rater`, `laterality_rater`. This is where patient-level labels live (frequency, laterality, exclusion status). See "Label versioning" below for the `_original` vs active column convention.
- **`annotations.csv`** — One row per segment per rater (long format). Contains: `segment_id`, `patient_id`, `rater`, `frequency_hz`, `no_pd`, `skipped`, `spatial_extent`, `spatial_channels`, `annotation_date`, `annotation_round`, `notes`. This is where individual rater judgments live.
- **`segments.csv`** — One row per segment. Contains: `segment_id`, `patient_id`, `subtype`, `subtype_source`, `mat_file`, `duration_sec`, `fs`, `n_channels`, `montage`, `original_source`, `original_filename`. This is the segment registry (metadata, not labels).

**How to add new labels:**

1. Save the raw source annotation file to `data/_archive/<task_name>/` (e.g., `data/_archive/lpd_laterality/lpd_laterality_annotations.csv`). This preserves the original data as received.
2. Write a script or use inline Python to merge the new annotations into the appropriate canonical file(s):
   - Patient-level labels (e.g., laterality, exclusions) → add/update columns in `patients.csv`
   - Per-segment per-rater labels (e.g., frequency ratings) → add rows to `annotations.csv`
   - New segments → add rows to `segments.csv`
3. Verify the merge: check row counts, confirm no patients are missing or duplicated, and spot-check values.
4. Labels that an annotator marked as "skip" should be left blank (not stored as "skip") in the canonical files.

**Existing archive directories** in `data/_archive/`: `canonical_dataset`, `dataset_eeg`, `pd_expert_raw`, `pd_expert_review`, `pd_mw_catchup`, `pd_round1_candidates`, `pd_round2`, `pd_round3`, `pd_round4`, `rda_round1`, `lpd_laterality`, `misclass_review`.

**Current laterality status:** LPD patients are fully annotated (95 left, 75 right, 3 bilateral, 13 skipped). LRDA patients still need laterality annotation.

### Label Versioning

`patients.csv` maintains two versions of each label:

- **Active columns** (`subtype`, `gold_standard_freq`, `laterality`) — the current best ground truth. These get updated when corrections are applied after reviewing algorithm errors. All evaluation code uses these columns.
- **Original columns** (`subtype_original`, `gold_standard_freq_original`, `laterality_original`) — the annotator's first-pass labels, frozen at the time of initial annotation. These are **never updated**, even when corrections are applied.

This dual-column design supports three types of comparison:

| Comparison | Use case | Columns |
|-----------|----------|---------|
| Expert vs expert (IRR) | Inter-rater reliability | `*_original` for each rater |
| Algorithm vs noisy expert | Fair comparison (algorithm never saw corrections) | algorithm vs `*_original` |
| Algorithm vs ground truth | Best-available performance metric | algorithm vs active columns |

**Rater provenance** is tracked via `subtype_rater` and `laterality_rater`:
- `"original"` — label came from the original dataset (folder structure or prior annotation)
- `"MW"` — label was set or corrected by MW
- Future raters (e.g., `"SZ"` for Sahar) will be added when their labels arrive

**Correction workflow:**

1. Run `code/generate_misclass_reviewer.py` to build an HTML viewer of algorithm errors
2. Review cases in the browser, annotating corrections via the buttons/keyboard
3. Export the corrections CSV from the viewer
4. Save the CSV to `data/_archive/misclass_review/` with date and rater (e.g., `label_corrections_mw_20260319.csv`)
5. Run a merge script to apply corrections to the **active columns** in `patients.csv` (the `_original` columns are never touched)
6. Re-run the evaluation to measure the impact of cleaner labels

## Repository Structure

```
code/
├── Signal Processing
│   ├── pd_pointiness_acf.py          Core SP feature extraction (ACF, pointiness, etc.)
│   ├── pd_detector/                  Original McGraw et al. PD detector
│   ├── pd_detector_alternate/        Enhanced PD detector (APD + z-score peak detection)
│   └── rda_detector/                 RDA detectors (FFT/FOOOF + Hilbert-Huang)
│
├── Optimization Frameworks
│   ├── optimization_harness_v2.py    PD evaluation framework (LOPO cross-validation)
│   ├── rda_optimization_harness.py   RDA evaluation framework
│   ├── update_dashboard_v2.py        Regenerate PD optimization dashboard
│   └── update_rda_dashboard.py       Regenerate RDA optimization dashboard
│
├── Experiment Scripts
│   ├── exp_t1_*.py                   Feature engineering experiments
│   ├── exp_t2_*.py                   Model selection experiments (GBM, KNN, etc.)
│   ├── exp_t3_*.py                   CNN embedding + timing experiments
│   ├── exp_rda_*.py                  RDA-specific experiments
│   ├── r3_*.py through r12_*.py      Round-specific experiment scripts
│   └── run_baselines.py              Baseline method evaluation
│
├── Deep Learning (dl/)
│   ├── model.py                      CNN architecture
│   ├── train_phase1.py               Phase 1: pretrain on weak labels
│   ├── train_phase2.py               Phase 2: fine-tune on expert labels
│   └── evaluate.py                   Evaluation utilities
│
├── Visualization & Annotation
│   ├── browse_results.py             Interactive EEG browser with verbal descriptions
│   ├── generate_test_images.py       Generate test-case images (25 per pattern)
│   ├── generate_figures.py           Publication figures
│   ├── imageGeneration/              EEG plotting utilities
│   └── generate_*_viewer.py          Annotation review viewers
│
├── Analysis
│   ├── extract_frequency_spatial_extent.py   Main batch extraction
│   ├── extract_with_laterality.py            Laterality-enhanced extraction
│   ├── evaluate_methods.py                   Method comparison
│   └── irr_analysis_*.ipynb                  Inter-rater reliability notebooks
│
└── environment.yml                   Conda environment specification

docs/                                 Archived approach review documents (v1-v6)
APPROACH_REVIEW_v11.md                Current approach, results, and system architecture
DESCRIPTION_RULES.md                  Verbal description rules (ACNS 2021)
QUICKSTART.md                         Getting started guide
DATASET_INFO.md                       Detailed data access instructions
TESTING.md                            Testing guide
test_case_images/                     100 example images (25 per pattern) + raw EEG
```

## Usage

### Run Experiments

```bash
conda activate foe

# Run a PD frequency experiment
python code/exp_t1_expanded_features.py

# Run an RDA experiment
python code/exp_rda_task_a.py

# Update the optimization dashboard
python code/update_dashboard_v2.py
```

### Process Full Dataset

```bash
cd code
python extract_frequency_spatial_extent.py
```

Processes all EEG files, runs multiple detector variants, saves results to `results/`.

### Interactive Browser

```bash
cd code
python browse_results.py --event lrda
```

Controls: arrow keys to navigate, 1-4 to switch pattern types, Q to quit.

### View Dashboards

Results dashboards (when generated) are in `results/`:
- `optimization_dashboard_v2.html` -- PD optimization results
- `rda_dashboard.html` -- RDA optimization results

## Algorithm Details

### PD Detectors (LPD/GPD)

Best model: CNN+Temporal Attention on raw single-channel waveforms with PD-weighted aggregation (Spearman 0.640 on 594 patients). Best handcrafted: RF 200 trees on 6 SP features (Spearman 0.604).

- `pd_detect` -- Original McGraw et al. detector
- `pd_detect_alternate (pk_detect='apd')` -- Adaptive peak detection (best)
- `pd_detect_alternate (pk_detect='zscore')` -- Z-score based

### RDA Detectors (LRDA/GRDA)

Best model: FFT peak frequency (Spearman 0.840).

- `rda1a_fft` -- FFT + FOOOF frequency modeling
- `rda1b_fft` -- FFT with enhanced peak selection (best)
- `rda2_hht` -- Hilbert-Huang Transform

### Verbal Descriptions

Generates ACNS 2021 standardized descriptions, e.g.:
```
LRDA at 2.1 Hz, unilateral left; maximal in the centro-parietal and temporal regions.
GRDA at 1.8 Hz, frontally predominant.
GPD at 1.5 Hz, no regional predominance.
```

See [DESCRIPTION_RULES.md](DESCRIPTION_RULES.md) for the complete rule set.

## Citation

```bibtex
@article{tautan2025automated,
  title={Automated estimation of frequency and spatial extent of periodic and
         rhythmic epileptiform activity from continuous electroencephalography data},
  author={T{\u{a}}u{\c{t}}an, Alexandra-Maria and Jing, Jin and Basovic, Lara
          and Hadar, Parimala Nallappan and Sartipi, Sahar and Fernandes, Marcos Paulo
          and Kim, Jonathan and Struck, Aaron F and Westover, M Brandon and Zafar, Sahar F},
  journal={Journal of Neural Engineering},
  volume={22},
  number={6},
  year={2025},
  month={dec},
  doi={10.1088/1741-2552/ae2716},
  pmid={41330044},
  url={https://pubmed.ncbi.nlm.nih.gov/41330044/}
}
```

## License

CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International). See [LICENSE.txt](LICENSE.txt).

- Academic/research use: free with citation
- Commercial use: contact authors for licensing

Data provided by BDSP under their Data Use Agreement.

## Contact

- **Data access**: [bdsp.io](https://bdsp.io) or support@bdsp.io
- **Code issues**: [GitHub Issues](https://github.com/bdsp-core/IIIC-Frequency-Analysis-2/issues)
- **Research collaboration**: Contact the corresponding authors via the publication

## Acknowledgments

Massachusetts General Hospital, Department of Neurology; Brain Data Science Platform (BDSP); Harvard Medical School. We thank expert EEG annotators LB, PH, and SZ for their careful review.
