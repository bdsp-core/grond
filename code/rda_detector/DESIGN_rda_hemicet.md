# RDA HemiCET+DP: Adapting the PD Pipeline for Rhythmic Delta Activity

## Motivation

The HemiCET+DP pipeline achieves F1=0.891 for detecting periodic discharges (LPD/GPD) and estimating their frequency. RDA (LRDA/GRDA) presents a structurally similar problem: find approximately-periodic repeating waveforms within a 10-second EEG segment and estimate their frequency. Current spectral methods (VE, NVO, FOOOF, FFT peak, ACF) work sometimes but not reliably — they struggle with intermittent patterns, amplitude modulation, and harmonic confusion.

The core DP framework (evidence trace → candidate peaks → Viterbi with periodicity prior → template refinement) should transfer directly, since it models exactly the right structure: quasi-periodic events with occasional skips.

## What transfers directly from the PD pipeline

1. **Dynamic Programming** — approximately-periodic prior, skip tolerance (max_skip=3), Viterbi-style forward pass + backtracking. RDA has the same temporal structure.
2. **Pipeline architecture** — evidence → threshold → candidates → DP → EM template refinement → confidence filter.
3. **Hemisphere-based processing** — LRDA is lateralized (like LPD); GRDA is bilateral (like GPD). Same routing logic applies.
4. **5-fold patient-stratified CV** — same evaluation framework.
5. **Frequency = 1/median(IPI)** — same output computation.

## What needs adaptation

### 1. Evidence Trace Generation

**PD version:** HemiCET U-Net trained on sharp Gaussian targets (σ=10ms = 2 samples) at discharge peaks.

**RDA adaptation:** Two options, in order of complexity:

**Option A — Signal-processing evidence (no model, baseline):**
- Bandpass filter around estimated frequency ± 0.5 Hz (use NVO/ACF estimate)
- Rectify + smooth (or Hilbert envelope)
- Average across top channels per hemisphere
- This may work well for clean RDA since the signal is more sinusoidal

**Option B — Retrained RhythmiCET U-Net:**
- Same architecture (8ch → 32 → 64 → 128 → 128, skip connections, ~525K params)
- Train on RDA wave peak annotations with wider Gaussian targets (σ=20-50ms) to match rounder morphology
- Loss: BCE(pos_weight=~10) + sharpness penalty + floor loss (same structure)
- This should handle intermittent RDA, amplitude variation, and noise better than bandpass alone

### 2. Frequency Estimation Branch

**PD version:** Per-channel CNN+Attention (50K params) → PD-weighted average → 0.8*CNN + 0.2*ACF blend.

**RDA adaptation:**
- ACF should work *better* for RDA — rhythmic signals are what ACF was designed for
- CNN could be retrained, but simpler spectral methods may suffice as the initial frequency estimate
- **Proposed:** Use NVO best frequency (already implemented in `generate_lrda_labeler.py`) or ACF as primary estimate, optionally blend with CNN if retrained
- Frequency range: same 0.5-3.5 Hz

### 3. Candidate Peak Extraction

**PD version:** `find_peaks` on evidence trace, height > 5% of max, min_distance = 0.2*T.

**RDA adaptation:**
- Same approach but on bandpass-filtered signal peaks rather than pointiness peaks
- For Option A: find peaks of the filtered signal directly (zero-crossing midpoints or local maxima)
- For Option B: find peaks of the RhythmiCET evidence trace (same as PD)
- min_distance constraint scales with expected period, same as now

### 4. DP Parameters

| Parameter | PD value | RDA suggested | Rationale |
|-----------|----------|---------------|-----------|
| α (timing penalty) | 1.5 | 1.0-1.2 | RDA periods slightly more variable |
| β (skip penalty) | 0.3 | 0.3-0.5 | RDA may have fewer skips, penalize more |
| λ (existence cost) | 0.05 | 0.05 | Similar |
| Node score exponent | 1.5 | 1.0-1.2 | Less discrimination needed for rounder peaks |
| max_skip | 3 | 3 | Same |
| peak_height_frac | 0.05 | 0.10 | RDA evidence may be noisier, raise threshold |

These should be optimized empirically, same as C1 parameter sweep for PD.

## Auto-Labeling Strategy for Training Data

### The bottleneck: per-wave timing annotations

We need per-wave peak times to train RhythmiCET. Manual annotation of thousands of waves would be prohibitive. But we have a key advantage: **550 RDA cases already have expert-verified frequency estimates** (from `rda_freq_annotations_mw.csv`), and we know which cases are clean LRDA/GRDA.

### Proposed auto-labeling pipeline

For each case with a known frequency f_est:

1. **Detrend** — remove linear/polynomial trend per channel
2. **Bandpass filter** — narrow band around f_est (e.g., f_est ± 0.3 Hz), using a zero-phase Butterworth filter. The known frequency lets us choose a tight passband that isolates the RDA rhythm.
3. **Find peaks** — local maxima of the filtered signal. With a tight bandpass, peaks should correspond to individual RDA waves.
4. **Find zero crossings** — ascending zero crossings give wave onsets; descending give wave offsets. This yields full triplets (onset, peak, offset) for free.
5. **Cross-channel consensus** — average filtered signal across top-N involved channels, then find peaks on the average. Alternatively, find peaks per channel and cluster across channels (within ±25ms).
6. **Quality check** — compute IPI CV on the detected peaks. Cases with IPI CV < 0.25 are likely clean and can be accepted en bloc. Cases with higher variability need review.

### Review workflow

- **Tier 1 (auto-accept):** IPI CV < 0.20, all peaks within ±20% of expected period. Accept without review. Estimated ~60-70% of cases.
- **Tier 2 (quick review):** IPI CV 0.20-0.35. Show in HTML viewer for MW to accept/reject en bloc. Estimated ~20% of cases.
- **Tier 3 (manual edit):** IPI CV > 0.35 or visual outliers. Show in HTML viewer with add/delete capability (reuse existing timing review viewer infrastructure). Estimated ~10-15% of cases.

### Adapting the existing viewer

The LRDA labeler (`generate_lrda_labeler.py`) already has wave triplet annotation support (onset/peak/offset click modes, W key toggle, Del to remove). The timing review viewer (`generate_timing_review_viewer.py`) has add/delete functionality for discharge times. We can combine these:

- Pre-populate with auto-detected wave peaks (displayed as vertical lines + dots)
- Show the bandpass-filtered signal as an overlay for visual confirmation
- Add/delete individual peaks as needed
- Accept/reject buttons for en-bloc decisions

## Data Inventory

| Source | Cases | Notes |
|--------|-------|-------|
| MW RDA freq annotations | 550 | Expert frequency, can auto-label peaks |
| MW LRDA labels (ground truth) | 78 | Have frequency + laterality, no triplets yet |
| Multi-expert RDA annotations | 222 | From rda_optimization_harness, ≥2 raters |
| LRDA segments (total) | 1,674 | Available for expansion |
| GRDA segments (total) | 3,296 | Available for expansion |

**Starting dataset:** 550 cases with expert frequency → auto-label → review → ~400-500 usable cases with per-wave timing. This is comparable to the 675 PD cases used for HemiCET.

## Implementation Plan

### Phase 0: Auto-labeling pipeline
1. Write `auto_label_rda_waves.py` — bandpass + peak detection for all 550 annotated cases
2. Compute quality metrics (IPI CV, peak consistency) per case
3. Generate review viewer for Tier 2/3 cases
4. MW reviews → produce `rda_discharge_times.json` (same format as PD `discharge_times.json`)

### Phase 1: Signal-processing baseline (no neural model)
1. Implement `rda_pipeline_baseline.py` using:
   - Frequency estimate from NVO or ACF
   - Evidence trace from bandpass envelope
   - Existing DP framework (import from hemi_detector)
   - Template refinement + confidence filter
2. Evaluate on the 550-case dataset (LOPO CV)
3. This establishes the baseline to beat

### Phase 2: RhythmiCET training
1. Adapt `hemi_detector/dataset.py` for RDA targets (wider Gaussians)
2. Train U-Net on auto-labeled RDA data (same architecture, adjusted loss)
3. Evaluate RhythmiCET evidence + DP vs. bandpass evidence + DP
4. Optimize DP parameters for RDA (α, β, λ, exponent sweep)

### Phase 3: Unified evaluation
1. Compare against existing methods (VE, NVO, FOOOF, FFT peak, HHT) on same test set
2. Report Spearman ρ, timing MAE, and per-wave F1
3. Test generalization: train on LRDA, test on GRDA (and vice versa)

## Key Risks

1. **Auto-labeling quality** — If many cases have poor auto-labels, the review burden grows. Mitigation: start with the cleanest cases (lowest IPI CV from existing spectral estimates).
2. **RDA morphology variation** — RDA waves can evolve (frequency shifts, morphology changes within a segment). The DP's skip tolerance and template refinement help, but may need wider tolerance.
3. **Harmonic confusion** — Spectral methods often pick up harmonics/subharmonics. The DP approach should be more robust since it finds actual peaks, but the initial frequency estimate matters.
4. **Fewer sharp landmarks** — PD discharges have clear onset points; RDA waves are smoother. Evidence peaks may be broader and noisier. The wider Gaussian targets and lower node score exponent should help.

## Files Reference

| Existing file | Relevance |
|---------------|-----------|
| `code/hemi_detector/` | HemiCET model, dataset, training code — adapt for RDA |
| `code/rda_detector/rda1a_fft.py` | Existing FOOOF-based frequency estimation |
| `code/rda_detector/rda1b_fft.py` | Enhanced FFT frequency estimation |
| `code/rda_detector/rda2_hht.py` | HHT-based frequency estimation |
| `code/rda_optimization_harness.py` | RDA evaluation framework (222 multi-expert cases) |
| `code/generate_lrda_labeler.py` | LRDA labeler with wave triplet annotation support |
| `code/label_pipeline/generate_timing_review_viewer.py` | Timing review viewer with add/delete |
| `data/labels/archive_labels/rda_freq_annotations_mw.csv` | 550 expert frequency annotations |
| `data/labels/archive_labels/lrda_labels_mw.json` | 78 ground truth LRDA labels |
