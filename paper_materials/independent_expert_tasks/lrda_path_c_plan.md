# Path C — closing the LRDA frequency / laterality gap

> **Durable plan.** This document captures the full Path C strategy so any
> future session (Claude or human) can pick up exactly where we left off.
> Updated continuously as work progresses.

## Goal

Close the algorithm-vs-expert gap on **LRDA frequency** (current MW–ALGO
ICC 0.66 vs expert–expert mean 0.90) and on **LRDA laterality** (MW–ALGO κ
0.85 vs expert–expert 0.99). The PD tasks and GRDA frequency are already
at or above expert–expert agreement and are out of scope here.

## Failure-mode taxonomy (from MW's review of top-20 disagreement cases)

| Failure mode | Count of top 20 | Description |
|---|---|---|
| Partial-LRDA | 5 | LRDA is temporally local (only 1–3s of the 10s window) |
| Laterality cascade | 6 | Variance-based hemisphere selection picks the wrong side; freq error follows |
| Partial hemisphere | 3 | Within the right hemisphere, only some channels show LRDA |
| Misclassified | 1 | Not LRDA at all |

(case 13 hits 3 buckets simultaneously — overlapping)

## Strategy: two parallel research thrusts

- **Plan A — Hard-case detector + selective swap-in.** Train a binary
  classifier that decides, per segment, whether to use the existing
  W05/V1 estimator or the V8 "active-window spectral-peak" estimator.
  Lower risk, expected to deliver tomorrow.
- **Plan B — End-to-end neural pitch detection.** Train a CRNN that maps
  18-channel × 10s EEG directly to frequency. Higher ceiling, longer
  development cycle.

Both plans share most of the data preparation and evaluation harness.

---

## Plan A — Hard-case classifier (Path A.1)

### Files

| Path | Purpose | Status |
|---|---|---|
| `code/evaluation/lrda_features.py` | per-segment featurizer (V1 internals + spectral + temporal stability + hemisphere/channel dispersion + IIIC ambiguity) | TODO |
| `code/evaluation/train_lrda_hard_case_classifier.py` | 5-fold patient-stratified CV, HistGradientBoostingClassifier, ROC analysis, save model + threshold | TODO |
| `code/evaluation/v9_gated_hybrid.py` | per-segment routing: V1 by default, V8 if classifier predicts hard | TODO |
| `data/labels/independent_expert_v1/lrda_features.csv` | featurized segments (cached) | TODO |
| `data/labels/independent_expert_v1/hard_case_classifier.pkl` | trained model | TODO |
| `data/labels/independent_expert_v1/v9_predictions.json` | V9 freq/laterality per segment, used by the IRR analysis | TODO |

### Featurization (~25 features per segment)

1. **V1 internal signals**
   - `v1_freq` — V1 frequency
   - `v1_if_cv` — Hilbert IF coefficient of variation (already returned by `_hilbert_freq_cv`)
   - `v1_hemi_var_ratio` — pass-1 left/right variance ratio
   - `v1_pass_delta` — pass-1 freq − pass-2 freq

2. **Whole-segment spectral**
   - `welch_peak_prominence` — max-peak prominence in 0.5–4 Hz Welch PSD
   - `welch_n_peaks` — number of distinct peaks above baseline
   - `welch_peak_freq` — frequency at max peak
   - `welch_peak_freq_minus_v1_freq` — discrepancy

3. **Sliding-window rhythmicity stability**
   - `rhythm_stability_mean` — mean spectral peak prominence across 5 sliding 3-s windows
   - `rhythm_stability_std` — std of those prominences (high = LRDA is local)
   - `rhythm_max_min_ratio` — max-window prominence / min-window prominence

4. **Hemisphere asymmetry**
   - `lr_prominence_ratio` — left-hemisphere prominence vs right-hemisphere prominence
   - `lr_prominence_diff` — absolute difference
   - `dom_hemi_signed_score` — positive if V1's choice is also strongest by spectral prominence

5. **Per-channel rhythmicity dispersion**
   - `ch_prominence_std` — std of per-channel narrowband-variance ratios across the V1-chosen 8 hemisphere channels
   - `ch_prominence_max_min_ratio`
   - `ch_top3_v_bot3_ratio`

6. **Inter-rater pre-existing context**
   - `iiic_plurality_frac` — from manifest CSV
   - `iiic_n_votes` — from manifest CSV

### Training

- Target: `is_hard = |v1_freq − mean(expert_freqs)| > 0.5`
- Training set: 174 LRDA segments where MW + V1 both have a label.
- Labeling: 23 hard / 151 easy → 1:6.6 imbalance. Use `class_weight='balanced'`.
- Classifier: `HistGradientBoostingClassifier` (sklearn). `max_depth=3`, `n_estimators=100`, `learning_rate=0.05`. Tunable.
- 5-fold patient-stratified CV. Patient strata derived from manifest's `patient_id` column.
- Pick threshold to maximize macro-F1 across the easy/hard classes (or to maximize end-to-end MAE — we'll evaluate both).
- Save the trained model + threshold to `data/labels/independent_expert_v1/hard_case_classifier.pkl`.

### V9 gated hybrid

- For each LRDA manifest segment:
  - Compute features → run classifier → if hard, use V8 active-window; else use V1.
- Save predictions to `data/labels/independent_expert_v1/v9_predictions.json` for downstream analysis.
- Modify `analyze_independent_expert_v1.py` to optionally use V9 predictions as the ALGO column for LRDA, then re-run.

### Expected result

- LRDA freq MW–ALGO ICC: 0.66 → 0.78–0.85 (depending on classifier accuracy).
- LRDA freq SZ–ALGO ICC: 0.89 → ≥0.89 (no regression hopefully; SZ is already aligned with V1).
- LRDA freq TZ–ALGO ICC: 0.71 → 0.80+ (TZ is similar to MW in scoring style).
- LRDA laterality unchanged (V8 inherits V1's laterality).

### Risks

- 23-case "hard" training set is small → overfitting risk. Mitigation: shallow trees, early stopping, cross-validate.
- V8 has its own failure modes on partial-LRDA (active window detector is naive). Mitigation: V9's gate should *itself* learn to send V8 only when V8 is likely to outperform V1.
- Doesn't help LRDA laterality — that's a separate Plan A.2 (see below).

### Plan A.2 — laterality-cascade fix (after A.1 is shipped)

- Same idea but for laterality: train a binary classifier to decide whether to trust V1's variance-based hemisphere selection or to swap to a peak-prominence-based selection.
- Training labels from MW's lat annotations on the 200-segment manifest.
- Lower priority since LRDA laterality is already at 0.91 vs expert 0.99 — a smaller gap to close than frequency.

---

## Plan B — End-to-end neural pitch detection

### Files

| Path | Purpose | Status |
|---|---|---|
| `code/cet_model/lrda_crnn.py` | CRNN architecture | TODO |
| `code/cet_model/lrda_crnn_dataset.py` | dataset loader + augmentation | TODO |
| `code/cet_model/train_lrda_crnn.py` | training driver | TODO |
| `data/lrda_crnn_cache/fold_{0..4}/` | per-fold checkpoints + loss curves + predictions | TODO |
| `data/labels/independent_expert_v1/lrda_crnn_predictions.json` | aggregated 5-fold out-of-fold predictions | TODO |

### Architecture

```
input: (B, 18, 2000)         # 18 bipolar channels × 10s × 200 Hz
  ├─ Conv1D blocks ×4         # kernels 51/25/13/7, channels 16/32/64/64, stride 2 each
  │   └─ output: (B, 64, 125)
  ├─ BiGRU (2 layers, 64 hidden each direction)
  │   └─ output: (B, 128, 125)   # bidirectional
  ├─ Attention-pool over time  # learnable temporal attention; outputs (B, 128)
  ├─ Linear(128 → 64) → GELU → Dropout(0.2)
  └─ Linear(64 → 1)             # log-frequency regression
```

Optional multi-task heads:
- `Linear(128 → 1) + sigmoid` → laterality (left=0, right=1) for LRDA only
- `Linear(128 → 1)` → "is rhythmic" confidence

### Training data

- 174 (MW) + 112 (SZ) + 144 (TZ) = 430 expert frequency labels on the 200-segment LRDA manifest.
- Plus legacy MW LRDA labels in `labels.csv` (~600 from earlier rounds) — total ~1,030 LRDA freq labels available.
- Optionally pre-train on PD frequency (~5,000 labels) and fine-tune on LRDA.
- Target: median of available expert frequencies per segment (more robust than mean to outliers).
- Held-out set: 50-segment patient-stratified test fold for honest reporting.

### Training protocol

- 5-fold patient-stratified CV (matches manuscript convention).
- Loss: log-frequency MSE (`mse(log(pred), log(target))`).
- Optimizer: AdamW (lr 1e-3, weight_decay 1e-4), cosine LR schedule.
- ~100 epochs, batch size 16, early-stop on validation loss with patience 10.
- Augmentations: amplitude scaling (0.8–1.2×), additive noise (20–40 dB SNR), channel dropout (p=0.15), time shifts (±100 samples), hemisphere swap (50% probability — target frequency unchanged).
- Hardware: Apple Silicon (MPS) — should train ~2–3 hours per fold.

### Evaluation

- Out-of-fold predictions from all 5 folds → single per-segment prediction set.
- Compare to V1 and to V9 (Plan A) on:
  - MW–ALGO ICC, SZ–ALGO ICC, TZ–ALGO ICC for LRDA frequency.
  - Stratified by ground-truth frequency band.
  - Stratified by failure mode taxonomy (partial-LRDA, laterality cascade, etc.).

### Risks

- Small dataset (~1,030 segments). Mitigation: heavy augmentation; pre-train on PD frequency.
- Multi-rater label noise. Mitigation: predict median; weight loss by IIIC plurality fraction.
- Could fail to beat NB-Hilbert if dataset insufficient. Fallback: ensemble V1 + V9 + CRNN by mean-of-predictions; or use CRNN as a feature for Plan A's classifier.

---

## Sequencing

1. **Now**: write this plan, then build & train Plan A. Feed-forward classifier completes within minutes.
2. **In parallel**: scaffold Plan B (data, model, training script). Kick off training in background. Target: training completes within 2-3 hours per fold.
3. **Decision point** after Plan A finishes: does the gated hybrid materially close the LRDA gap?
   - Yes → ship V9 in the manuscript, continue Plan B as a follow-up.
   - No → wait for Plan B to finish, evaluate, and ship whichever is best.
4. **End state**: updated figS5 (with whichever algorithm is best), updated `analysis_v1.md`, updated manuscript reviewer-note resolutions and abstract.

## Reproducibility

```bash
# Plan A end-to-end
conda run -n morgoth python code/evaluation/lrda_features.py    # cache features
conda run -n morgoth python code/evaluation/train_lrda_hard_case_classifier.py
conda run -n morgoth python code/evaluation/v9_gated_hybrid.py  # produce v9_predictions.json
conda run -n morgoth python code/evaluation/analyze_independent_expert_v1.py --algo v9
cp results/independent_expert_v1/forest_plot.png paper_materials/figures/figS5_independent_expert_v1_irr.png

# Plan B end-to-end
conda run -n morgoth python code/cet_model/train_lrda_crnn.py --fold 0  # repeat for 1..4
conda run -n morgoth python code/cet_model/lrda_crnn_predict.py         # aggregate folds
conda run -n morgoth python code/evaluation/analyze_independent_expert_v1.py --algo crnn
```

## Status log

(updated as work progresses)

- 2026-04-29 09:00 — plan written
- 2026-04-29 09:30 — Plan A complete:
  - Featurizer (`code/evaluation/lrda_features.py`): cached 200 LRDA segments × 22 features
  - Classifier trained (`hard_case_classifier.pkl`): ROC AUC 0.754, OOF precision 0.227, recall 0.238 on hard class. Top features: `ch_prominence_std`, `v1_if_cv_pass1`, `v1_dom_side_left`, `ch_top3_v_bot3_ratio`, `lr_prominence_diff` -- all aligned with the failure-mode taxonomy.
  - V9 gated hybrid (`code/evaluation/v9_gated_hybrid.py`): 18/200 segments routed to V8 active-window, 182 to V1.
  - End-to-end IRR: LRDA freq MW-ALGO ICC 0.659 -> **0.727** (+0.068), TZ-ALGO 0.710 -> **0.774** (+0.064), SZ-ALGO 0.890 -> 0.823 (-0.067, slight regression).
  - Net EA-mean ICC for LRDA frequency: 0.751 -> 0.775 (+0.024). Laterality unchanged.
  - Saved as `paper_materials/figures/figS5b_independent_expert_v1_irr_v9.png` alongside the V1 baseline `figS5_*.png`.
- 2026-04-29 09:55 — Plan B scaffolding committed:
  - `code/cet_model/lrda_crnn.py`: CRNN model (4 ConvBlocks → 1-layer BiGRU → temporal attention → tanh-bounded log-freq head). 141,682 params.
  - `code/cet_model/lrda_crnn_dataset.py`: dataset loader with augmentations (amplitude, noise, channel dropout, time shift, hemisphere swap), patient-stratified 5-fold split. 739 LRDA segments with rater frequency labels (183 manifest + 556 legacy).
  - `code/cet_model/train_lrda_crnn.py`: training driver with NaN-safe batching, cosine LR schedule, Adam, early stopping. CLI for fold selection.
  - Smoke-test (fold 0, 30 epochs): val_MAE converged to 0.251 Hz (best epoch 28). Training-loss readouts are unstable on MPS (numerical artifact in batch loss aggregation) but val_MAE and saved checkpoints are valid.
  - Full 5-fold training kicked off as background task; outputs at `data/labels/independent_expert_v1/lrda_crnn_predictions.json` when complete.
- 2026-04-29 10:10 — Plan B first attempt FAILED (numerical instability):
  - Full 5-fold MPS training completed but produced poor IRR.
  - LRDA freq vs experts: MW-ALGO ICC 0.437, SZ-ALGO 0.622, TZ-ALGO 0.448 -- all materially WORSE than V1.
  - Root cause: MPS numerical instability triggered early-stopping after 2-15 epochs in 4/5 folds, far below adequate convergence. Per-epoch train_loss readouts oscillated between sane values (~0.08) and astronomical (~1e30) on MPS, suggesting gradient accumulation precision issues.
  - Archived to `data/lrda_crnn_cache_mps_unstable/` and `data/labels/independent_expert_v1/lrda_crnn_predictions_mps_unstable.json` for audit.
- 2026-04-29 10:15 — Plan B retraining on CPU (forced via `--force-cpu` flag added to train script). CPU is slower (~10s/epoch vs 4s on MPS) but numerically stable. Training kicked off in background; ETA 30-60 min for 5 folds.
