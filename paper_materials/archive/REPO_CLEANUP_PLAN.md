# Repository Cleanup Plan for Publication

## Goals

The published repo should allow users to:
1. **Retrain models** from scratch (data + labels → trained models)
2. **Label data** using interactive tools (RDA and PD tasks)
3. **View algorithm results** on existing data
4. **Reproduce all figures and tables** from either raw data or intermediate results
5. **Run final models on new data** easily

## Proposed Directory Structure

```
code/
├── pipeline/                    # Core inference pipeline (run on new data)
│   ├── pd_profiler.py      # Main PD entry point
│   ├── discharge_detector.py    # HemiCET+DP discharge detection
│   ├── rda_characterizer.py     # Main RDA entry point (W05)
│   ├── rda_spatial_extent.py    # RDA-PLV spatial
│   ├── bipd_detector.py         # BIPD classification
│   └── pd_pointiness_acf.py     # Signal processing utilities
│
├── models/                      # Model definitions + training scripts
│   ├── channelpdnet/            # ChannelPD-Net (CNN+Attention)
│   │   ├── model.py
│   │   └── train.py
│   ├── hemicet/                 # HemiCET-UNet
│   │   ├── model.py
│   │   └── train.py
│   ├── cet_unet/                # CET-UNet (per-channel evidence)
│   │   ├── model.py
│   │   └── train.py
│   └── train_all.py             # Wrapper: train all models sequentially
│
├── evaluation/                  # Evaluation and metrics
│   ├── eval_frequency.py
│   ├── eval_timing.py
│   ├── eval_spatial.py
│   ├── eval_lateralization.py
│   └── eval_all.py              # Wrapper: run all evaluations
│
├── labeling/                    # Interactive labeling tools
│   ├── pd_timing_labeler.py
│   ├── pd_frequency_labeler.py
│   ├── rda_frequency_labeler.py
│   ├── spatial_labeler.py
│   ├── laterality_labeler.py
│   └── bipd_labeler.py
│
├── data_management/             # Data curation and label management
│   ├── build_segment_labels.py
│   ├── label_status_report.py
│   └── harvest_segments.py
│
├── visualization/               # Result viewers and plotting
│   ├── browse_results.py
│   └── plotting_utils.py
│
└── archive/                     # Historical code (experiments, contests, legacy)
    ├── experiments/
    ├── lateralization_contest/
    ├── rda_contest/
    ├── spatial_contest/
    ├── pd_detector/
    └── optimization_harnesses/

paper_materials/
├── figures/                     # Output figures
├── tables/                      # Output tables
├── methods/                     # Math writeups
├── build_fig2.py                # Fig 2 generator
├── build_fig3.py                # Fig 3 generator
├── render_figures.py            # Figs 5-8 generator
├── generate_all_figures.py      # Wrapper: generate all figures
├── generate_all_tables.py       # Wrapper: generate all tables
└── archive/                     # Optimization logs, old figure versions
```

## Phase 1: Archive experimental/contest code
- Move experiments/, *_contest/, pd_detector*, dl/ → code/archive/
- Move optimization harnesses → code/archive/
- Archive old APPROACH_REVIEW versions (v7-v16)
- Archive paper_materials/optimization_logs/

## Phase 2: Create wrapper scripts
- generate_all_figures.py — runs all figure generators in order
- generate_all_tables.py — generates all tables
- train_all.py — trains all models sequentially with learning curves
- eval_all.py — runs all evaluations and prints summary

## Phase 3: Consolidate duplicates
- Reduce labeling generators to essential set
- Keep only latest training variant per model
- Consolidate visualization utilities

## Phase 4: Documentation
- Update README with clear reproduction instructions
- Add QUICKSTART.md for running on new data
