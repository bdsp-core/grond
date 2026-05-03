# Reproducibility — training data, scripts, and seeds

Reference for retraining every model in the manuscript from scratch and
reproducing the published numbers.

## Data

- **Local cache:** `data/eeg/` holds 13,556 `.mat` segment files (≈4.1 GB),
  one per row in [`data/labels/segment_labels.csv`](../data/labels/segment_labels.csv).
- **AWS S3 mirror:** `s3://bdsp-opendata-credentialed/iiic-freq3/` (use the
  `opendata` profile for read, `opendata-write` for write). Contains:
  - `data/eeg/` — full 13,556-file segment store
  - `data/labels/` — every CSV/JSON the training scripts depend on
  - `data/pd_channel_cache/`, `data/hemi_cache/` — model checkpoint cache
  - `data/bipd_cache/`, `data/cet_cache/`, `data/e2e_cache/`, `data/pdnet_v2_cache/`,
    `data/rda_cache/`, `data/unified_model_cache/` — feature/inference caches
- **Verifier:** [`code/evaluation/verify_local_data.py`](../code/evaluation/verify_local_data.py)
  cross-references every `mat_file` in `segment_labels.csv` against the
  on-disk tree, validates every label JSON parses, and confirms every
  expected checkpoint exists. Writes a JSON report to
  `results/verify_local_data_report.json`.

## Models

| Model | Training script | Final checkpoint | CV | Seed |
|---|---|---|---|---|
| ChannelPD-Net | [`code/pd_channel_detector/train_cnn_attention.py`](../code/pd_channel_detector/train_cnn_attention.py) | `data/pd_channel_cache/cnn_attn_fold{0..4}.pt` | 5-fold patient-stratified `StratifiedKFold` | `cfg['seed']=42` (`torch.manual_seed`, `np.random.seed`) |
| HemiCET-UNet | [`code/hemi_detector/train_hemi_cet.py`](../code/hemi_detector/train_hemi_cet.py) | `data/hemi_cache/hemi_cet_v2/hemi_cet_fold{0..4}.pt` | 5-fold patient-stratified | `SEED=42` (numpy + torch + CUDA) |
| LPD-vs-GPD RF (300 trees) | [`code/evaluation/eval_subtype_classification.py`](../code/evaluation/eval_subtype_classification.py) | `data/models/lpd_vs_gpd_rf.pkl` | 5-fold `StratifiedGroupKFold` | `random_state=42` |
| 3-way LPD/GPD/BIPD RF | [`code/evaluation/eval_3way_classification.py`](../code/evaluation/eval_3way_classification.py) | `data/models/three_way_rf.pkl` | 5-fold `StratifiedGroupKFold` (class-balanced) | `random_state=42` |
| BIPD GBT (synthetic-trained) | [`code/bipd_detector.py`](../code/bipd_detector.py) | `data/models/bipd_gbt.pkl` | 5-fold `StratifiedKFold` on synthetic | `np.random.RandomState(42)` for synth gen + `seed=42` for LightGBM |
| LRDA laterality classifier | [`code/evaluation/train_lrda_laterality_classifier.py`](../code/evaluation/train_lrda_laterality_classifier.py) | `data/labels/independent_expert_v1/lrda_laterality_classifier.pkl` | 5-fold patient-grouped | `random_state=42` |
| LRDA hard-case classifier | [`code/evaluation/train_lrda_hard_case_classifier.py`](../code/evaluation/train_lrda_hard_case_classifier.py) | `data/labels/independent_expert_v1/hard_case_classifier.pkl` | 5-fold | `random_state=42` |
| NB-Hilbert RDA frequency (V12) | [`code/evaluation/lrda_freq_hyperparam_sweep.py`](../code/evaluation/lrda_freq_hyperparam_sweep.py) | n/a (signal processing, no model) | n/a | n/a |

## Cross-derived label files

- [`data/labels/channel_pseudolabels.json`](../data/labels/channel_pseudolabels.json)
  — built by [`code/label_pipeline/build_channel_pseudolabels.py`](../code/label_pipeline/build_channel_pseudolabels.py)
  from four upstream sources (priority order):
  1. MW ground-truth from `channel_involvement.json` (`review_status==ground_truth`)
  2. Spatial annotation channels from `annotations.csv`
  3. Human laterality from `archive_labels/patients.csv`
  4. Predicted laterality from `archive_labels/channel_involvement_predictions.json`
  The build script was patched in May 2026 to point at `archive_labels/`
  for inputs (3) and (4) after those files were moved during repo cleanup.

## Known sources of nondeterminism

- **Apple Silicon MPS backend.** Some torch ops on MPS are nondeterministic
  even with `torch.manual_seed`; per-fold AUCs may vary by ~0.001 across
  reruns. CUDA + `torch.use_deterministic_algorithms(True)` is fully
  reproducible if available.
- **No `PYTHONHASHSEED` is exported** by training scripts. Set it in the
  shell (`PYTHONHASHSEED=0`) before invoking the trainers if bit-for-bit
  reproducibility is required.
- **The contest grid-search outputs** (e.g. for the 192-experiment LRDA
  frequency sweep) are not persisted as a single canonical JSON; rerunning
  `lrda_freq_hyperparam_sweep.py` regenerates them from scratch.

## Comparison harness

After re-training, run [`code/evaluation/freq_table_cis.py`](../code/evaluation/freq_table_cis.py)
and [`paper_materials/build_fig_irr_bars.py`](../paper_materials/build_fig_irr_bars.py)
to regenerate the manuscript's Table 4 numbers and Figure 5; cross-check
against the published values in `paper_materials/manuscript.tex`. The
existing checkpoints in `data/pd_channel_cache/` and `data/hemi_cache/`
correspond to the published numbers. A planned `compare_retrain_to_published.py`
will diff per-row and tag each within-tolerance.
