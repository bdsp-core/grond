# Frequency Estimation for Periodic EEG Patterns: Review v14

## Critical Methodological Principle

**No method may use gold standard labels as input.** All algorithms operate from raw EEG only.

## Dataset (as of 2026-03-23)

### Labeled Data

| Type | Active | Freq | Timing | Laterality | Status |
|------|--------|------|--------|------------|--------|
| **LPD** | 2,433 | 437 | 437 | 437 | Freq/timing complete for core set |
| **GPD** | 2,272 | 207 | 207 | N/A | Freq/timing complete for core set |
| **BIPD** | 16 | — | 16 (L+R) | N/A | **Detected + classified** |
| **LRDA** | 170 | 68 | — | 68 | Freq+laterality labeled |
| **GRDA** | 119 | 14 | 14 | N/A | Mostly unlabeled |

**Discharge timing labels**: 675 ground truth cases (437 LPD + 207 GPD + 31 other), expert-reviewed through 3 rounds of model-assisted correction.

**BIPD review**: 214 cases reviewed by MW (69 from original BIPD/GPD pool + 145 high-probability model candidates). Led to discovery of 16 confirmed BIPDs, reclassification of 22 cases as LPD, and exclusion of 42 cases as non-PD.

**LRDA labels**: 68 cases with MW-reviewed frequency and laterality (left/right/bilateral). 97 additional RDA cases harvested but not yet frequency-labeled.

### IIIC Dataset Integration

~8,260 segments integrated from the IIIC S3 dataset:
- LPD: 2,144 segments → per-hemisphere detection cached for 2,750 patients
- GPD: 2,167 segments → per-hemisphere detection cached for 2,357 patients
- LRDA: 1,311 segments
- GRDA: 2,638 segments

### EEG Data

~9,600 standardized .mat files (18 channels × 2000 samples at 200 Hz).

## What Changed Since v13

### 1. BIPD Detection Pipeline Implemented

Built a two-stage BIPD vs GPD classifier per BIPD_PLAN.md:

**Stage 1**: Run HemiCET+DP independently on each hemisphere of every GPD case to extract per-hemisphere discharge times and frequencies. Cached results for 2,357 GPD patients and 2,750 LPD patients.

**Stage 2**: Compute timing-sequence features from L/R discharge times, train gradient boosted tree classifier on synthetic data, evaluate on real cases.

#### Feature Engineering (21 features)

From two timing sequences T_L and T_R:

- **Frequency features** (4): f_L, f_R, freq_ratio, freq_diff, log_freq_ratio
- **Phase relationship** (5): nearest_delay_median/std/iqr/mad, phase_consistency
- **Cross-correlation** (3): xcorr_peak, xcorr_peak_lag, xcorr_ratio
- **Independence** (3): ipi_correlation, matched_fraction, unmatched_L/R
- **Count** (3): n_L, n_R, count_ratio, total_discharges

#### Synthetic Training Data

Generated ~45,800 synthetic examples from real GPD/LPD timing sequences:

| Type | Source | N examples |
|------|--------|-----------|
| **GPD-like (negative)** | Real GPD pairs | ~2,357 |
| | Duplicated LPD + jitter (σ=10-30ms) | ~5,500 |
| | GPD + propagation delay (15-45ms) | ~7,000 |
| **BIPD-like (positive)** | Cross-patient LPD pairs | ~8,000 |
| | Phase-shifted GPD | ~4,000 |
| | Frequency-scaled GPD | ~4,000 |
| | Same-freq cross-patient LPD | ~2,000 |

Empirical jitter (from real GPD L-R delays): 122.4 ms median std.

#### Performance

| Metric | Value |
|--------|-------|
| Synthetic CV AUC (5-fold) | **0.920** |
| Real data AUC | **0.840** |
| Sensitivity (BIPD) | **10/16 (63%)** |
| Specificity (GPD) | **1,976/2,289 (86%)** |

**Top feature importances**: freq_diff (51%), log_freq_ratio (19%), nearest_delay_median (14%), phase_consistency (3%).

**PPV analysis**: At no threshold does PPV exceed 20%. With only 16 BIPDs among 2,305 cases (0.7% base rate), even small false positive rates overwhelm the true positives. The classifier is a **screening tool** for expert review, not an autonomous classifier.

**Key finding**: Of 214 cases reviewed by MW, only 16 are true BIPDs. Many high-probability "BIPD" predictions are actually GPDs with asymmetric discharge counts (detection artifact: left hemisphere consistently detects more discharges than right). True BIPD requires visually confirming independent timing between hemispheres.

### 2. BIPD Label Cleanup

Two rounds of expert review using an interactive HTML reviewer:

| Round | Reviewed | GPD | BIPD | LPD | Reject |
|-------|----------|-----|------|-----|--------|
| 1 (original pool) | 69 | 34 | 14 | 16 | 5 |
| 2 (model candidates, prob>0.7) | 145 | 100 | 2 | 6 | 37 |
| **Total** | **214** | **134** | **16** | **22** | **42** |

**Impact on labels**:
- 78 cases relabeled (33 from round 1 + 45 from round 2)
- 22 GPD → LPD (actually unilateral)
- 42 GPD/BIPD → Rejected (not periodic discharges at all)
- 6 BIPD → GPD (actually generalized)

### 3. LRDA Labeling Tool Built

Built an interactive HTML labeling tool for LRDA frequency and laterality annotation:

- Canvas-based EEG viewer with channels arranged: L-lateral, L-parasag, midline, R-parasag, R-lateral
- NVO (Narrowband Variance Optimization) pre-estimates frequency using sliding-window bandpass VE search
- Per-channel narrowband filtered overlay shows how well the selected frequency matches the actual LRDA
- Per-channel VE% shown as colored bars (blue=left, red=right, green=midline)
- Laterality computed from VE laterality index
- Wave triplet marking (onset → peak → offset) for future morphology analysis
- Frequency buttons (0.5–3.5 Hz in 0.05 Hz steps) with keyboard shortcuts

**NVO algorithm**: For each candidate frequency, bandpass filter the signal at f±0.3Hz, compute VE = var(filtered)/var(original) per channel. Use a 3-second sliding window to handle intermittent RDA. Score = max(top-3 left channels, top-3 right channels) VE. Best frequency = highest scoring.

**68 LRDA cases labeled** with frequency and laterality (left/right/bilateral). 97 additional cases rejected as non-RDA.

### 4. BIPD Reviewer Tool

Interactive HTML tool for reviewing BIPD vs GPD classification:
- EEG display with hemisphere grouping (L-lateral, L-parasag, midline, R-parasag, R-lateral)
- Left hemisphere discharge times shown as red vertical lines (L hemisphere channels only)
- Right hemisphere discharge times shown as blue vertical lines (R hemisphere channels only)
- Lines meet at midline channels
- Model prediction (BIPD probability) displayed
- Keyboard shortcuts: 1=GPD, 2=BIPD, 3=LPD, 4=Reject
- Auto-advance after labeling, localStorage persistence, JSON export

### 5. Master Labels Updated

All corrections from BIPD and LRDA review rounds merged into canonical label files:
- **patients.csv**: 2,865+ rows, subtype/laterality/exclusion fields updated
- **segments.csv**: subtypes updated for reclassified cases
- **bipd_review_labels_mw.json**: 214 BIPD review decisions
- **lrda_labels_mw.json**: 68 LRDA frequency + laterality labels

## Architecture

### PD Detection: Two Approaches

#### Full 18-Channel Pipeline (F1=0.717)

```
EEG (18ch × 2000) ──┬── CNN+Attention (freq) ──→ f_cnn ──┐
                     ├── ACF on pointiness ──→ f_acf ──────┤── 0.8×f_cnn + 0.2×f_acf ──→ f_est
                     ├── HPP evidence (pointiness+TKEO) ───┤
                     └── CET-UNet evidence ────────────────┤── product-boost ──→ DP ──→ times
```

#### HemiCET (8-Channel, Best Timing, F1=0.699)

```
EEG (8ch hemisphere) ──→ HemiCET-UNet ──→ evidence ──→ DP ──→ times
                    ──→ CNN+Attention (freq) ──→ f_est ──↗
```

### BIPD Detection: Two-Stage Pipeline

```
EEG (18ch) ──┬── Left 8ch  ──→ HemiCET+DP ──→ {t_L1, t_L2, ...}, f_L ──┐
             └── Right 8ch ──→ HemiCET+DP ──→ {t_R1, t_R2, ...}, f_R ──┤
                                                                         ├── 21 timing features ──→ GBT ──→ P(BIPD)
```

### LRDA Frequency: NVO

```
EEG (18ch) ──→ lowpass 5Hz ──→ per-freq bandpass (f±0.3Hz) ──→ VE per channel ──→ sliding window max ──→ best freq
```

## Method Names

| Abbreviation | Full Name | What it does |
|-------------|-----------|-------------|
| **HPP** | Hidden Point Process | MAP inference via DP for discharge timing |
| **CET** | CNN Evidence Trace | Learned per-channel discharge evidence (U-Net) |
| **HemiCET** | Hemisphere CET | 8-channel CET-UNet for single-hemisphere evidence |
| **NVO** | Narrowband Variance Optimization | Bandpass VE search for RDA frequency |
| **GBT** | Gradient Boosted Trees | BIPD vs GPD classifier on timing features |

## Nine Tasks Status

| # | Task | Status | Best Method | Performance |
|---|------|--------|-------------|-------------|
| 1 | LPD vs GPD | **Done** | RF 300 | AUC 0.931 |
| 2 | LRDA vs GRDA | TODO | — | — |
| 3 | PD channel ID | Partial | CNN+Attention | AUC 0.870 |
| 4 | RDA channel ID | Pseudolabels | CNN | AUC 0.842 |
| 5 | PD discharge timing | **Done** | Full pipeline / HemiCET | F1 0.717 / 0.699 |
| 6 | RDA wave timing | TODO | — | — |
| 7 | RDA frequency | **Partial** (68 cases) | NVO | Pending eval |
| 8 | PD frequency | **Done** | HemiCET IPI | ρ 0.819 |
| 9 | BIPD detection | **Done** | HemiCET+GBT | AUC 0.840, Sens 63% |

## Next Steps

1. **Evaluate NVO on LRDA** — now have 68 labeled cases (up from 23), run leave-one-out frequency estimation
2. **Retrain BIPD classifier** with additional labeled BIPDs if more are found
3. **LRDA/GRDA classification** (Task 2) — laterality labels now available for 68 cases
4. **RDA wave timing** (Task 6) — triplet marking infrastructure built, needs more annotations
5. **Self-supervised pretraining** for HemiCET using all ~9,600 EEG segments
6. **Generate paper figures** per PAPER_ROADMAP.md
