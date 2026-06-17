# GROND: Automated Characterization of Periodic and Rhythmic EEG Patterns

**GROND** (Generalized Rhythmic and Oscillatory Neurophysiology Descriptor) is an automated system for joint characterization — lateralization, spatial localization, discharge timing, and frequency estimation — of periodic discharges (PD) and rhythmic delta activity (RDA) in continuous EEG. It is built around two complementary pipelines: the **PD-Profiler** (LPD, GPD, BIPD) and the **RDA-Profiler** (LRDA, GRDA), each with its own README sections below. GROND was developed for critical-care EEG monitoring at Massachusetts General Hospital and Beth Israel Deaconess Medical Center.

**Manuscript**: Jing J, Sun C, Zhang T, Byrd M, T\u{a}u\c{t}an AM, Basovic L, Hadar PN, Fernandes MP, Goldenholz D, Kim J, Struck AF, Zafar SF, Westover MB. "GROND: Automated Characterization of Periodic and Rhythmic EEG Patterns." *Journal of Neural Engineering* — manuscript in preparation. Source LaTeX in [paper_materials/manuscript/manuscript.tex](paper_materials/manuscript/manuscript.tex); built PDF at [paper_materials/manuscript/manuscript.pdf](paper_materials/manuscript/manuscript.pdf).

**Predecessor**: T\u{a}u\c{t}an AM, Jing J, Basovic L, Hadar PN, Sartipi S, Fernandes MP, Kim J, Struck AF, Westover MB, Zafar SF. "Automated estimation of frequency and spatial extent of periodic and rhythmic epileptiform activity from continuous electroencephalography data." *Journal of Neural Engineering*, 22(6):066027, 2025. [doi:10.1088/1741-2552/ae2716](https://doi.org/10.1088/1741-2552/ae2716). The present system substantially improves on this prior work across all characterization tasks.

## Status

Headline numbers from the manuscript (see [paper_materials/manuscript/manuscript.tex](paper_materials/manuscript/manuscript.tex) for full Methods, sample-size reconciliation, and confidence intervals).

| Task | Metric | N | Performance | Method |
|------|--------|---|-------------|--------|
| **PD-Profiler** | | | | |
| PD hemisphere lateralization | AUC | 1,274 | **0.989** | ChannelPD-Net mean probability |
| LPD vs. GPD classification | AUC | 7,037 | **0.911** | RF on per-channel probs + features |
| 3-way LPD/GPD/BIPD (macro) | AUC | 5,064 | **0.862** | RF on probs + timing features |
| BIPD vs. GPD | AUC | 2,308 | **0.937** | GBT on timing features |
| PD discharge timing | F1 | 576 | **0.817** (mean timing MAE 3.12 ms, median 0 samples) | HemiCET-UNet + DP |
| PD frequency (LPD / GPD) | Spearman ρ | 1,103 / 1,099 | **0.772 / 0.819** | IPI from HemiCET-UNet + DP |
| PD spatial extent | Jaccard | 211 | **0.731** (expert–expert 0.751, 97.3%) | Hybrid-PLV @ threshold 0.38 |
| PD spatial localization | — | — | Discharge-locked topoplot | Laplacian–GFP alignment + morgoth-viewer regions |
| **RDA-Profiler** | | | | |
| LRDA vs. GRDA | AUC | 4,253 | **0.837** | NB-Hilbert (iterative narrowband Hilbert refinement) |
| RDA frequency (LRDA / GRDA) | Spearman ρ | 721 / 1,453 | **0.719 / 0.795** | NB-Hilbert (V12) |
| RDA spatial extent | ICC(3,1) | 211 | **0.371** (expert–expert 0.373) | RDA-PLV @ threshold 0.15 |
| **Independent-expert IRR (3-rater majority-accept consensus, V14)** | | | | |
| LPD frequency (mean EA ICC vs mean EE ICC) | ICC(3,1) | 187 | **0.916 vs 0.866** (Δ=+0.051, *p*<0.001, **EA above EE**) | V12 (NB-Hilbert) |
| GPD frequency | ICC(3,1) | 197 | **0.975 vs 0.966** (Δ=+0.009, *p*<0.001, **EA above EE**) | V12 |
| GRDA frequency | ICC(3,1) | 123 | **0.942 vs 0.944** (Δ=−0.002, *p*=0.55, tie at EE ceiling) | V12 |
| LRDA frequency | ICC(3,1) | 150 | **0.815 vs 0.894** (Δ=−0.079, *p*=0.048) | V12 retuned |
| LRDA laterality | Cohen κ | 150 | **0.948 vs 0.983** (Δ=−0.035, *p*=0.116, n.s.) | V14 amplitude/rhythmicity hybrid |
| LPD laterality | Cohen κ | 156 | **0.948 vs 0.970** (Δ=−0.022, *p*=0.29, tie) | V12 |

**All methods use EEG-only input** — no gold standard labels provided as algorithm input. See [docs/history/APPROACH_REVIEW_v17.md](docs/history/APPROACH_REVIEW_v17.md) for the development log and [paper_materials/manuscript/manuscript.tex](paper_materials/manuscript/manuscript.tex) for the formal Methods.

### PD-Profiler

The PD characterization pipeline (referred to in code as `PDProfiler`) uses a single per-channel CNN (**ChannelPD-Net**) as a backbone serving three roles: laterality detection, spatial reference selection, and evidence channel weighting. Laterality detection feeds forward into both downstream modules — constraining the spatial localizer to seed from the ipsilateral hemisphere and restricting the discharge detector to ipsilateral channels. See [paper_materials/manuscript/manuscript.tex](paper_materials/manuscript/manuscript.tex) §2.3 + appendix B for the formal mathematical specification.

## Overview

The system extracts four properties from 10-second EEG segments:

1. **Frequency** of the epileptiform pattern (Hz)
2. **Spatial extent** (proportion of brain regions affected)
3. **Laterality** (unilateral, bilateral asymmetric, bilateral symmetric)
4. **Verbal description** following ACNS 2021 standardized nomenclature

Supported pattern types: LPD, GPD, LRDA, GRDA.

## Installation

The repo uses **two conda environments** for different parts of the pipeline. Most work needs `morgoth`; only a few legacy signal-processing scripts need `foe`.

```bash
git clone https://github.com/bdsp-core/grond.git
cd grond

# morgoth (Python 3.11): PyTorch model training/inference, figure
# generation, the full GROND (PD-Profiler + RDA-Profiler) pipelines.
conda env create -f code/morgoth.yml
conda activate morgoth

# foe (Python 3.8): legacy signal-processing scripts that depend on
# fooof, pyhht, pingouin, statsmodels (mostly under code/archive/).
# Optional unless you are reproducing those specific experiments.
conda env create -f code/environment.yml
```

| Environment | Python | Key dependencies | Used for |
|---|---|---|---|
| `morgoth` | 3.11 | torch 2.10, numpy 2.4, scipy 1.17, mne 1.11, scikit-learn 1.8 | Default for everything in this README |
| `foe` | 3.8 | numpy 1.24, scipy 1.10, mne 1.6, fooof 1.0, pyhht 0.1, pingouin | Legacy frequency-estimation experiments under `code/archive/` |

## Data Access

The entire dataset — every EEG segment, every expert annotation, every algorithm prediction, and the IRR/discharge-timing cohort flags — is bundled into a single self-contained HDF5 file on S3:

```bash
# Download the GROND data bank (1.69 GB, ~14,300 segments)
aws s3 cp s3://bdsp-opendata-credentialed/grond/grond_data.h5 data/grond_data.h5
```

This requires AWS credentials with access to the `bdsp-opendata-credentialed` bucket. To request access, visit the [Brain Data Science Platform (BDSP)](https://bdsp.io).

Pre-trained model weights live in this git repository under `data/hemi_cache/`, `data/pd_channel_cache/`, `data/cet_cache/`, and `data/models/`. Once you have `grond_data.h5` + the git repo, you can reproduce every figure and every number in the manuscript with `python paper_materials/generate_all_figures.py`.

### S3 Data Contents

```
s3://bdsp-opendata-credentialed/grond/
├── grond_data.h5                              (1.69 GB) — the entire dataset
└── independent_expert_tasks/                  (470 MB) — labeling viewers
    ├── lpd_task.html
    ├── gpd_task.html
    ├── lrda_task.html
    └── grda_task.html
```

### `grond_data.h5` schema

```
/segments/{segment_id}/
    eeg              (19, 2000) float32 monopolar referential EEG at 200 Hz
    attrs            patient_id, subtype, eeg_source, mat_file,
                     has_discharge_timing
/labels/{segment_id}/
    discharge_times  (N,) float32 — per-discharge times (PD-GT only)
    attrs            freq_hz_consensus, laterality_consensus,
                     spatial_channels, freq_per_rater_json, …
/predictions/pdchar/{segment_id}/    — PD-Profiler outputs
/predictions/tautan/{segment_id}/    — Tăuțan-baseline outputs
/predictions/rda_plv/{segment_id}/   — RDA-PLV spatial-extent outputs
/cohorts/
    segment_ids      (N,) str
    pd_gt            (N,) bool
    irr_canonical    (N,) bool
/metadata/
    channel_names_mono       (19,) str
    bipolar_pair_definitions_json    JSON — derive the canonical
                                     18-channel bipolar banana on demand
```

The bank stores **19-channel monopolar referential** EEG as the canonical format. Downstream code (PD-Profiler, build_fig2, etc.) derives whichever montage it needs at runtime — bipolar banana for the discharge detector + most plotting, Laplacian for the discharge-locked topography, common-average for some analyses.

### Local Data Structure

```
data/
├── eeg/                       14,341 .mat files (19-ch monopolar × 2000 samples @ 200 Hz)
├── grond_data.h5              ← all of the above bundled (download from S3, see Data Access)
├── labels/
│   ├── labels.csv             Unified per-rater labels (48,727 rows)
│   │                          One row per (segment, rater, label_type)
│   ├── segments.csv           Segment registry (14,369 rows; algo predictions + metadata)
│   ├── segment_labels.csv     Consolidated summary (one row per segment)
│   ├── discharge_times.json   PD per-discharge timing (593 PD ground-truth: 423 LPD + 170 GPD)
│   ├── rda_wave_labels.json   RDA per-wave timing
│   ├── channel_involvement.json   PD spatial ground truth
│   ├── channel_pseudolabels.json  Channel detection training labels
│   └── archive_labels/        Raw labeling session outputs, backups, deprecated files
├── hemi_cache/hemi_cet_v2/    HemiCET-UNet weights (canonical, 5-fold)
├── pd_channel_cache/          ChannelPD-Net weights (5-fold)
├── cet_cache/                 Legacy CET weights (superseded by HemiCET)
└── models/                    BIPD GBT, LPD-vs-GPD RF, 3-way RF
```

### Label System

The label system has three tiers:

1. **`labels.csv`** (48,727 rows) — the unified per-rater label store. Each row is one (segment, rater, label_type, value) tuple. All human annotations — frequency, spatial extent, spatial channels, discharge timing, wave timing, pattern class, laterality — live here. This is where new annotations go.

2. **`segments.csv`** (14,369 rows) — segment registry. One row per EEG file on disk. Contains physical metadata (montage, sampling rate, duration) and algorithm predictions (algo_freq_hz, pdchar_freq_hz, tautan_freq_hz). No human labels.

3. **`segment_labels.csv`** (14,369 rows) — consolidated read-only summary. One row per segment, aggregating expert labels from labels.csv with algorithm predictions from segments.csv. **Regenerated** by `python code/data_management/build_segment_labels.py`.

**JSON label files** store per-event annotations that don't fit a single-value-per-segment model:
- **`discharge_times.json`** — per-discharge peak times within each PD segment
- **`rda_wave_labels.json`** — per-wave times within each RDA segment
- **`channel_involvement.json`** — per-channel binary involvement labels

**To add new labels:** Add rows to `labels.csv`, then re-run `build_segment_labels.py`.

### Label Coverage by Subtype

| Label type | LPD | GPD | LRDA | GRDA | Total |
|---|---:|---:|---:|---:|---:|
| Expert-reviewed frequency | 1,499 | 1,539 | 654 | 1,381 | 5,073 |
| Discharge timing | 917 | 1,036 | — | — | 1,953 |
| Wave timing | — | — | 189 | 313 | 502 |
| Channel involvement / spatial | 352 | 260 | 29 | 177 | 818 |
| Laterality | 1,336 | 249 | 1,039 | 789 | 3,413 |
| IIIC crowd votes (≥10 raters) | 1,846 | 1,024 | 239 | 420 | 3,529 |
| **Total MW annotations** | **1,888** | **1,944** | **1,079** | **1,983** | **7,547** |

## Repository Structure

```
code/
├── Core Pipeline
│   ├── pd_profiler.py           Unified PD pipeline (main entry point)
│   ├── discharge_detector.py         HemiCET+DP discharge detection
│   ├── pd_pointiness_acf.py          Signal processing (ACF, pointiness, getBanana)
│   ├── bipd_detector.py              BIPD vs GPD classification
│   ├── rda_spatial_extent.py         RDA-PLV spatial extent
│   └── browse_results.py             Interactive EEG browser
│
├── Models & Training
│   ├── cet_model/                    CET-UNet — per-channel discharge evidence
│   ├── hemi_detector/                HemiCET — 8-channel hemisphere evidence
│   ├── pd_channel_detector/          ChannelPD-Net — per-channel PD detection
│   ├── unified_model/                Unified multi-task model
│   ├── pdnet_v2/                     PDNet v2 architecture
│   └── e2e_model/                    End-to-end comparison model
│
├── Data & Labels
│   ├── data_management/              Data harvesting, download, segment extraction
│   └── label_pipeline/               Label management, pseudolabel generation
│
├── Evaluation
│   └── evaluation/                   Evaluation & validation scripts
│
├── Tools
│   ├── generators/labeling/          Interactive annotation tool generators (11)
│   ├── generators/review/            Label review tool generators (9)
│   ├── generators/dashboards/        Summary dashboards (3)
│   ├── generators/figures/           Analysis figure generators
│   └── visualization/               Plotting & interactive browsing
│
└── archive/                          Historical: experiments, contests, legacy code

paper_materials/
├── manuscript.tex                    Main manuscript (Journal of Neural Engineering format)
├── appendix_math_methods.tex         Appendix B: full mathematical specifications
├── iopjournal.cls                    IOP Publishing LaTeX class file
├── figures/                          Publication figures (Figs 1-9 main, S1-S2 supplement)
├── tables/                           Publication tables (Tables 1-7)
├── methods/                          Formal math writeups (10 methods, source for appendix B)
├── build_fig2.py                     Fig 2: PD pipeline (fully from data)
├── build_fig3.py                     Fig 3: RDA pipeline (fully from data)
├── render_figures.py                 Figs 5-8: characterization examples
├── generate_all_figures.py           Wrapper: generate all figures
├── generate_all_tables.py            Wrapper: verify all tables
├── figure_legends.md                 Figure legends for all 11 figures
└── archive/                          Optimization logs, old figure versions

docs/
├── plans/                            Implementation/design plans (BIPD, HEMI, FIGURE, ...)
├── history/                          APPROACH_REVIEW versions (v2-v17)
├── references/                       ACNS reference PDFs and Tăuțan 2025
├── DATASET_INFO.md
├── DESCRIPTION_RULES.md
├── QUICKSTART.md
└── TESTING.md
```

## Quick Start

### Reproduce All Figures

```bash
conda activate morgoth

# Generate all publication figures
python paper_materials/generate_all_figures.py

# Generate a single figure
python paper_materials/generate_all_figures.py --figure 2

# Verify all tables
python paper_materials/generate_all_tables.py
```

### Run on New Data

```bash
conda activate morgoth

# PD characterization (laterality, timing, frequency, spatial, verbal description)
python -c "
from code.pd_profiler import PDProfiler
import scipy.io as sio
eeg = sio.loadmat('your_segment.mat')['data']  # 19×2000 monopolar referential
result = PDProfiler().characterize(eeg, subtype='lpd')  # PDProfiler does any
                                                       # mono→bipolar / Laplacian
                                                       # derivation it needs
print(result)
"
```

### Retrain Models

```bash
conda activate morgoth

# Train ChannelPD-Net (5-fold CV)
python code/pd_channel_detector/train_cnn_attention.py

# Train HemiCET-UNet (5-fold CV)
python code/hemi_detector/train_hemi_cet.py

# Train CET-UNet (5-fold CV)
python code/cet_model/train_cet.py
```

### Interactive Labeling

```bash
# Generate labeling tool for PD discharge timing
python code/generators/labeling/generate_hpp_labeler.py

# Generate labeling tool for RDA frequency
python code/generators/labeling/generate_rda_freq_labeler.py
```

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

### PD-Profiler

Standalone callable: `code/pd_profiler.py` (class `PDProfiler`). A single per-channel CNN (**ChannelPD-Net**) serves as the backbone for all PD tasks:

- **Laterality**: Compare hemisphere mean PD probabilities (AUC = 0.989, n = 1,274)
- **Spatial**: Hybrid-PLV — CNN picks ipsilateral reference channels, PLV finds connected regions. Inter-rater agreement: Model Jaccard 0.731 vs human 0.751 (97.3% of expert–expert agreement, n = 211 with 3-rater ground truth)
- **Timing**: HemiCET-UNet + DP — learned evidence trace + dynamic programming with periodic prior (F1 = 0.817, mean timing MAE = 3.12 ms, median 0 samples, n = 576)
- **Frequency**: IPI from detected discharges (Spearman ρ = 0.786 LPD / 0.846 GPD)

Laterality detection feeds forward into both spatial localizer (ipsilateral seed) and discharge detector (hemisphere selection). See `paper_materials/manuscript/manuscript.tex` §2.3 + appendix B for the formal mathematical specification.

### RDA-Profiler (LRDA/GRDA)

The RDA characterization pipeline classifies LRDA vs GRDA, determines laterality, and estimates frequency from a single 10-second bipolar EEG segment.

**Frequency** (V12): two-pass iterative narrowband Hilbert refinement (NB-Hilbert; originally `W05_DomOnly_IterRefine` in the contest log) with hyperparameters retuned (pass-1 bandpass 0.5–4.5 Hz, pass-2 narrowband half-width 0.5 Hz, top-3 dominant-hemisphere channels averaged, frequency search cap 4.5 Hz) against the 150-segment majority-accept independent-expert consensus dataset. On the labels.csv-canonical labeled cohort (after MW's two-pass May 2026 disagreement-review of all 425 LRDA segments where the V14 algorithm disagreed with any rater by >0.25 Hz, then a second pass on the 132 cases where MW specifically disagreed by >0.5 Hz), V12 NB-Hilbert reaches Spearman ρ of 0.719 (LRDA, n=721, MAE 0.225 Hz) and 0.795 (GRDA, n=1,453, MAE 0.188 Hz). On the original 4,253-segment LRDA vs. GRDA cohort the LRDA-vs-GRDA discriminator reaches AUC 0.837.

**Laterality** (V14, amplitude / rhythmicity hybrid): the V12 laterality call is amplitude-based (which hemisphere has more narrowband envelope amplitude at the estimated rhythm frequency) and is fallible whenever artifact, slow drift, or asymmetric volume conduction inflates one hemisphere's amplitude even when the rhythm itself is on the other side. V14 defaults to the V12 amplitude call but **overrides only when four amplitude-normalized rhythmicity measures unanimously disagree**: per-channel Q-factor (f<sub>peak</sub>/FWHM near est_freq), within-hemisphere phase-locking value at est_freq, narrowband peak-amplitude consistency, and per-hemisphere spectral peak prominence. The override is parameter-free. On the 156-segment consensus set it flips 4 of 156 V12 calls and lifts mean expert–algorithm κ from 0.927 to 0.953 (per-rater: MW 0.910→0.935; SZ 0.946→0.982; TZ 0.912→0.941).

76 methods benchmarked across 5 contest rounds. See [docs/history/APPROACH_REVIEW_v17.md](docs/history/APPROACH_REVIEW_v17.md) appendix A for the full leaderboard, and `paper_materials/manuscript/manuscript.tex` appendix A for the contest naming scheme. The V12/V14 work is documented in [paper_materials/independent_expert_tasks/lrda_path_c_plan.md](paper_materials/independent_expert_tasks/lrda_path_c_plan.md).

### Verbal Descriptions

Generates ACNS 2021 standardized descriptions, e.g.:
```
LRDA at 2.1 Hz, unilateral left; maximal in the centro-parietal and temporal regions.
GRDA at 1.8 Hz, frontally predominant.
GPD at 1.5 Hz, no regional predominance.
```

See [docs/DESCRIPTION_RULES.md](docs/DESCRIPTION_RULES.md) for the complete rule set.

### Paper Figures

Publication-quality figures in `paper_materials/` showing LPD, GPD, LRDA, and GRDA characterization examples. Each figure presents 3 cases (easy/medium/hard) stratified by IIIC expert agreement level. Panels include EEG with discharge markers, MNE-interpolated topoplots (inferno colormap), and ACNS 2021 verbal descriptions.

```bash
conda run -n morgoth python paper_materials/render_figures.py --pick '{"lpd":[1,15,9],"gpd":[17,2,9],"lrda":[11,3,6],"grda":[0,4,9]}'
```

## Citation

If you use the GROND data bank, model weights, or pipeline, please cite:

```bibtex
@article{jing2026grond,
  title={GROND: Automated Characterization of Periodic and Rhythmic EEG Patterns},
  author={Jing, Jin and Sun, Chenxi and Zhang, Tianyu and Byrd, Matt and
          T{\u{a}}u{\c{t}}an, Alexandra-Maria and Basovic, Lara
          and Hadar, Parimala Nallappan and Fernandes, Marcos Paulo
          and Goldenholz, Daniel and Kim, Jonathan and Struck, Aaron F
          and Zafar, Sahar F and Westover, M Brandon},
  journal={Journal of Neural Engineering},
  year={2026},
  note={Manuscript in preparation}
}
```

The predecessor frequency / spatial-extent paper:

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
- **Code issues**: [GitHub Issues](https://github.com/bdsp-core/grond/issues)
- **Research collaboration**: Contact the corresponding authors via the publication

## Acknowledgments

Massachusetts General Hospital, Department of Neurology; Brain Data Science Platform (BDSP); Harvard Medical School. We thank expert EEG annotators LB, PH, and SZ for their careful review.
