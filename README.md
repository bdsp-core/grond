# Automated Frequency Estimation for Periodic and Rhythmic EEG Patterns (LPD, GPD, LRDA, GRDA)

Algorithms for estimating the frequency of periodic discharges (PD) and rhythmic delta activity (RDA) in continuous EEG, developed for ICU EEG monitoring at Massachusetts General Hospital.

**Paper**: Tautan AM, Jing J, Basovic L, Hadar PN, Sartipi S, Fernandes MP, Kim J, Struck AF, Westover MB, Zafar SF. "Automated estimation of frequency and spatial extent of periodic and rhythmic epileptiform activity from continuous electroencephalography data." *Journal of Neural Engineering*, 22(6), 2025. [PubMed](https://pubmed.ncbi.nlm.nih.gov/41330044/)

## Status

| Task | Metric | N | Performance | Method |
|------|--------|---|-------------|--------|
| **PD Unified Pipeline** | | | | **PDCharacterizer** |
| PD discharge timing | F1 | 651 | **0.684** | HemiCET+DP (CNN-weighted) |
| PD frequency (IPI) | Spearman ρ | 500 | **0.681** | CNN+ACF → IPI |
| PD spatial localization | Composite | 465 | **0.811** | Hybrid-PLV (CNN ref + PLV) |
| PD spatial localization | Mean AUC | 465 | **0.814** | Hybrid-PLV |
| PD spatial inter-rater | Jaccard | 220 | **0.731** vs human 0.751 (97.3%) | Hybrid-PLV @ threshold 0.38 |
| Laterality (L vs R) | AUC | 423 | **0.963** | ChannelPD-Net hemisphere |
| Channel PD detection | AUC | 815 | **0.870** | ChannelPD-Net |
| Subtype (LPD vs GPD) | AUC | 594 | **0.931** | RF 300 trees |
| BIPD vs GPD | AUC | 2,305 | **0.840** | HemiCET+GBT (screening) |
| **RDA Analysis** | | | | |
| LRDA vs GRDA | AUC | 4,253 | **0.837** | W05_DomOnly_IterRefine |
| RDA frequency | Spearman ρ | 4,253 | **0.686** | W07_AutoChannel_FreqAgreement |
| LRDA+GRDA+freq (unified) | AUC / ρ | 4,253 | **0.837 / 0.635** | W05_DomOnly_IterRefine |
| LRDA+GRDA+freq (best freq) | AUC / ρ | 4,253 | **0.809 / 0.682** | V04_PLVSelected |
| RDA freq labels reviewed | — | 993 | MW-reviewed | V22 viewer + MW correction |
| LRDA laterality reviewed | — | 1,374 | MW-reviewed | 727 left, 308 right, 338 not-LRDA |

**All methods use EEG-only input** — no gold standard labels provided as algorithm input. See [APPROACH_REVIEW_v16.md](APPROACH_REVIEW_v16.md) for details.

### Unified PD Pipeline (PDCharacterizer)

The PD analysis uses a unified pipeline where a single per-channel CNN (**ChannelPD-Net**) serves triple duty: laterality detection, spatial reference selection, and evidence channel weighting. Laterality detection feeds forward into both downstream modules — constraining the spatial localizer to seed from the ipsilateral hemisphere and restricting the discharge detector to ipsilateral channels. See [paper_materials/unified_pd_pipeline.md](paper_materials/unified_pd_pipeline.md) for full architecture description and figure specification.

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
- `eeg/` — ~11,800 .mat files (10s bipolar EEG segments, 18ch at 200 Hz)
- `labels/` — canonical label files (see below)
- `dl_cache/`, `cet_cache/`, `pd_channel_cache/`, `hemi_cache/` — model weights
- `_archive/` — raw source annotation files

### Data Structure

```
data/
├── eeg/                  ~11,800 .mat files (18ch × 2000 samples @ 200 Hz)
├── labels/
│   ├── segment_labels.csv    Canonical labels — one row per EEG segment (11,817 rows)
│   ├── annotations.csv       Per-rater detailed annotations (5,160 rows)
│   ├── segments.csv          File registry: segment_id → mat_file mapping
│   ├── discharge_times.json  PD per-discharge timing (712 cases)
│   ├── rda_wave_labels.json  RDA per-wave timing (549 cases)
│   ├── channel_involvement.json    Spatial ground truth (594 cases)
│   ├── channel_pseudolabels.json   Channel detection training labels
│   └── archive_labels/       Raw source files (IIIC votes, deprecated patients.csv, etc.)
├── cet_cache/            CET-UNet model weights (5-fold)
├── pd_channel_cache/     CNN+Attention model weights (5-fold)
├── hemi_cache/           HemiCET model weights + experiment results
├── dl_cache/             External segment pool
└── _archive/             Raw source annotation files (by task/round)
```

### Label System

**All labels live at the segment level.** The canonical label file is `segment_labels.csv` — one row per EEG file on disk, consolidating all label sources.

**`segment_labels.csv`** columns:

| Column group | Columns | Description |
|-------------|---------|-------------|
| Identity | `mat_file`, `segment_id`, `patient_id` | File and patient identifiers |
| Subtype | `subtype`, `subtype_source` | Best subtype label and how it was assigned |
| IIIC votes | `iiic_vote_{other,seizure,lpd,gpd,lrda,grda}`, `iiic_n_votes`, `iiic_plurality`, `iiic_plurality_frac` | Per-segment expert vote vector (195 segments matched) |
| Frequency | `mw_freq`, `mw_freq_rater`, `auto_freq` | MW-reviewed vs algorithm-assigned frequency |
| Spatial | `spatial_channels`, `spatial_raters` | Brain region annotations |
| Laterality | `laterality`, `laterality_rater` | Left/right/bilateral |
| Exclusion | `excluded`, `exclusion_reason` | Exclusion flags |
| Audit trail | `subtype_original`, `freq_original`, `laterality_original` | Pre-correction labels (never updated) |
| Other labels | `has_discharge_timing`, `has_wave_timing`, `has_channel_involvement` | Boolean flags for JSON label files |
| Provenance | `original_source`, `original_filename`, `annotators` | Where the EEG came from, who annotated it |

**Supporting files:**

- **`annotations.csv`** — Per-segment per-rater annotations (frequency, spatial channels). One row per rater per segment. Includes `mat_file` for direct lookup.
- **`segments.csv`** — File registry mapping `segment_id` → `mat_file` with physical metadata (montage, sampling rate, provenance).
- **JSON files** — Specialized per-event label types (discharge timing, wave timing, channel involvement).

**To regenerate `segment_labels.csv`:** Run `python code/data_management/build_segment_labels.py`. This consolidates all sources (segments.csv, annotations.csv, IIIC votes, JSON files) into the single canonical file.

**To add new labels:** Add rows to `annotations.csv`, then re-run `build_segment_labels.py`.

### Label Coverage

| Label type | Segments |
|-----------|----------|
| IIIC per-segment votes | 195 |
| MW/expert frequency | 3,607 |
| Auto-assigned frequency | 4,368 |
| Spatial annotations | 965 |
| Laterality | 2,674 |
| Discharge timing | 2,400 |
| Wave timing | 549 |
| Channel involvement | 2,228 |

## Repository Structure

```
code/
├── Core Pipelines (root — 16 files, imported by everything)
│   ├── pd_characterizer.py           Unified PD pipeline (main entry point)
│   ├── discharge_detector.py         HemiCET+DP discharge detection
│   ├── pd_pointiness_acf.py          Core signal processing (ACF, pointiness, getBanana)
│   ├── bipd_detector.py              BIPD vs GPD classification
│   ├── optimization_harness_v2.py    PD evaluation framework (LOPO CV)
│   ├── rda_optimization_harness.py   RDA evaluation framework
│   └── browse_results.py             Interactive EEG browser
│
├── models/                           Neural network architectures & training
│   ├── cet_model/                    CET-UNet (per-channel evidence, 13 files)
│   ├── hemi_detector/                HemiCET (8-channel hemisphere, 16 files)
│   ├── pd_channel_detector/          ChannelPD-Net (per-channel PD prob, 12 files)
│   ├── unified_model/                Unified multi-task model (4 files)
│   ├── pdnet_v2/                     PDNet v2 architecture (5 files)
│   └── dl/                           General deep learning utilities (14 files)
│
├── detectors/                        Pattern-specific detectors
│   ├── pd_detector/                  Original McGraw et al. PD detector (9 files)
│   ├── pd_detector_alternate/        Enhanced PD detector (2 files)
│   └── rda_detector/                 RDA detectors: FFT/FOOOF/Hilbert (6 files)
│
├── contests/                         Contest frameworks & methods
│   ├── rda_contest/                  RDA analysis contest (45 methods, 11 files)
│   ├── spatial_contest/              PD spatial localization (26 methods, 9 files)
│   └── lateralization_contest/       LRDA vs GRDA classification (76 methods, 24 files)
│
├── experiments/                      Historical experiment scripts (60 files)
│   ├── pd_frequency/                 PD frequency optimization (11 files)
│   ├── pd_timing/                    PD timing & laterality experiments (6 files)
│   ├── rda/                          RDA-specific experiments (5 files)
│   ├── rounds/                       Round-based experiments r3-r12 (27 files)
│   └── misc/                         One-off experiments (12 files)
│
├── generators/                       HTML viewer/dashboard/figure builders (30 files)
│   ├── labeling/                     Annotation tool generators (9 files)
│   ├── review/                       Review tool generators (9 files)
│   ├── dashboards/                   Dashboard generators (3 files)
│   └── figures/                      Publication figure generators (9 files)
│
├── evaluation/                       Evaluation & validation scripts (11 files)
├── data_management/                  Data harvesting, download, cleanup (16 files)
├── visualization/                    Plotting & interactive browsing (6 files)
├── label_pipeline/                   Label management tools (10 files)
├── archive/                          Superseded scripts (7 files)
└── imageGeneration/                  EEG plotting utilities (2 files)

docs/                                 Archived approach review documents (v1-v6)
paper_materials/                      Paper writeup, figures, pipeline specs
APPROACH_REVIEW_v15.md                Current approach, results, and system architecture
```

## Usage

### Run Experiments

```bash
conda activate foe

# Run a PD frequency experiment
python code/experiments/pd_frequency/exp_t1_expanded_features.py

# Run an RDA experiment
python code/experiments/rda/exp_rda_task_a.py

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

### Results Organization

The `results/` directory contains 53 interactive HTML viewers, 6 contest leaderboards, and 267 contest result JSONs, organized by function:

```
results/
├── labeling_tools/          17 HTML — interactive annotation tools for MW
│   ├── pd_timing/           PD discharge timing (hpp_labeler, timing_correction)
│   ├── pd_frequency/        PD frequency annotation
│   ├── pd_laterality/       LPD laterality labeling
│   ├── rda_frequency/       RDA frequency (lrda_labeler, rda_freq_viewer, 7 batch viewers)
│   ├── rda_waves/           RDA wave timing annotation
│   └── bipd/                BIPD discharge labeling
│
├── review_tools/             8 HTML — label review and correction
│   ├── pd_review/           Misclassification, timing, spatial, laterality review
│   └── bipd_review/         BIPD screening and review
│
├── dashboards/              12 HTML — monitoring and status
│   ├── dataset/             Dataset overview, download status, data harvesting
│   ├── training/            Model training curves, HemiCET optimization
│   └── results/             PD optimization dashboards
│
├── leaderboards/             6 HTML + 267 JSON — contest results
│   ├── rda_contest/         RDA analysis contest (45 methods)
│   ├── spatial_contest/     PD spatial localization contest (26 methods)
│   └── lateralization/      LRDA vs GRDA classification (3 contest rounds)
│
├── analysis/                 9 HTML — result visualization
│   ├── frequency/           Frequency scatter plots, method comparisons
│   ├── evidence/            HPP vs CET evidence, consistency, timing
│   └── spatial/             Laterality visualization
│
└── archive/                 Old/superseded results and optimization run histories
```

### View Dashboards

Key dashboards:
- `results/dashboards/results/optimization_dashboard_v2.html` — PD optimization results
- `results/dashboards/dataset/rda_dashboard.html` — RDA data overview
- `results/leaderboards/rda_contest_leaderboard.html` — RDA analysis contest (45 methods)
- `results/leaderboards/spatial_contest_leaderboard.html` — Spatial localization contest (26 methods)

## Algorithm Details

### PD Unified Pipeline (PDCharacterizer)

Standalone callable: `code/pd_characterizer.py`. A single per-channel CNN (**ChannelPD-Net**) serves as the backbone for all PD tasks:

- **Laterality**: Compare hemisphere mean PD probabilities (AUC = 0.963)
- **Spatial**: Hybrid-PLV — CNN picks ipsilateral reference channels, PLV finds connected regions (Composite = 0.811). Inter-rater agreement: Model Jaccard 0.731 vs human 0.751 (97.3% of expert agreement, N=220 with 3-rater ground truth)
- **Timing**: HemiCET+DP — CNN-weighted evidence aggregation + dynamic programming (F1 = 0.684)
- **Frequency**: CNN+ACF ensemble → IPI-derived (Spearman ρ = 0.681)

Laterality detection feeds forward into both spatial localizer (ipsilateral seed) and discharge detector (hemisphere selection). See `paper_materials/unified_pd_pipeline.md` for full architecture.

### RDA Pipeline (LRDA/GRDA)

The RDA analysis pipeline classifies LRDA vs GRDA, determines laterality, and estimates frequency from a single 10-second bipolar EEG segment, processing hemispheres independently.

Best unified method: **W05_DomOnly_IterRefine** — two-pass iterative narrowband refinement with frequency estimated strictly from the predicted-dominant hemisphere. Achieves AUC 0.837 (LRDA vs GRDA classification) and Spearman ρ=0.635 (frequency estimation).

76 methods benchmarked across 5 contest rounds. See [APPROACH_REVIEW_v16.md](APPROACH_REVIEW_v16.md) Appendix A for full results and the [V5 leaderboard](results/v4_lateralization_leaderboard.html).

### Verbal Descriptions

Generates ACNS 2021 standardized descriptions, e.g.:
```
LRDA at 2.1 Hz, unilateral left; maximal in the centro-parietal and temporal regions.
GRDA at 1.8 Hz, frontally predominant.
GPD at 1.5 Hz, no regional predominance.
```

See [DESCRIPTION_RULES.md](DESCRIPTION_RULES.md) for the complete rule set.

### Paper Figures

Publication-quality figures in `paper_materials/` showing LPD, GPD, LRDA, and GRDA characterization examples. Each figure presents 3 cases (easy/medium/hard) stratified by IIIC expert agreement level. Panels include EEG with discharge markers, MNE-interpolated topoplots (inferno colormap), and ACNS 2021 verbal descriptions.

```bash
conda run -n morgoth python paper_materials/render_figures.py --pick '{"lpd":[1,15,9],"gpd":[17,2,9],"lrda":[11,3,6],"grda":[0,4,9]}'
```

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
