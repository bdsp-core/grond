# Automated Frequency Estimation for Periodic and Rhythmic EEG Patterns (LPD, GPD, LRDA, GRDA)

Algorithms for estimating the frequency of periodic discharges (PD) and rhythmic delta activity (RDA) in continuous EEG, developed for ICU EEG monitoring at Massachusetts General Hospital.

**Paper**: Tautan AM, Jing J, Basovic L, Hadar PN, Sartipi S, Fernandes MP, Kim J, Struck AF, Westover MB, Zafar SF. "Automated estimation of frequency and spatial extent of periodic and rhythmic epileptiform activity from continuous electroencephalography data." *Journal of Neural Engineering*, 22(6), 2025. [PubMed](https://pubmed.ncbi.nlm.nih.gov/41330044/)

## Status

| Task | Metric | N patients | Performance |
|------|--------|-----------|-------------|
| PD frequency (LPD + GPD) | Spearman rho | 335 | **0.686** |
| RDA frequency (GRDA + LRDA) | Spearman rho | 23 | **0.840** |
| Timing | Per 10s segment | -- | 33 ms |

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
- `eeg/` — 2,246 .mat files (10s bipolar EEG segments at 200 Hz)
- `labels/` — `segments.csv`, `annotations.csv` (long-format expert ratings), `patients.csv`
- `dl_cache/` — CNN model weights and external segment pool
- `_archive/` — previous round-specific data directories

### Data Structure

```
data/
├── eeg/              2,241 .mat files (10s segments, 200 Hz, 19 channels)
├── labels/
│   ├── segments.csv      Segment registry (segment metadata, file paths)
│   ├── annotations.csv   Expert ratings (long format: per-segment, per-rater)
│   └── patients.csv      Patient summary (gold standard freq, laterality, exclusions)
├── dl_cache/         CNN weights, segment pool
├── rda_cache/        Precomputed RDA features
├── templates_*.npy   Template banks for matched filtering
└── _archive/         Source annotation files (organized by round/task)
```

### Label Management

The three files in `data/labels/` are the canonical label set for the project. All analysis code reads from these files. When new annotations are collected, they are integrated into these files following this process:

**The three canonical label files:**

- **`patients.csv`** — One row per patient. Contains: `patient_id`, `subtype` (lpd/gpd/lrda/grda), `n_segments`, `n_raters`, `raters`, `gold_standard_freq`, `excluded`, `exclusion_reason`, `laterality`. This is where patient-level labels live (frequency, laterality, exclusion status).
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

**Existing archive directories** in `data/_archive/`: `canonical_dataset`, `dataset_eeg`, `pd_expert_raw`, `pd_expert_review`, `pd_mw_catchup`, `pd_round1_candidates`, `pd_round2`, `pd_round3`, `pd_round4`, `rda_round1`, `lpd_laterality`.

**Current laterality status:** LPD patients are fully annotated (95 left, 75 right, 3 bilateral, 13 skipped). LRDA patients still need laterality annotation.

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
APPROACH_REVIEW_v7.md                 Current optimization approach and results
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

Best model: GBM on 6 signal-processing features (Spearman 0.686).

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
