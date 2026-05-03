# RDA Lateralization: Findings and Development Plan

## Background

This document summarizes findings from analyzing the predecessor [IIIC-Frequency-Analysis-2](https://github.com/bdsp-core/IIIC-Frequency-Analysis-2) repository and outlines a plan to extend it with continuous per-channel RDA scores and a laterality index. The plan was subsequently implemented in the present (`grond`) repo.

The repo accompanies: Tautan et al. 2025, "Automated estimation of frequency and spatial extent of periodic and rhythmic epileptiform activity from continuous electroencephalography data," J. Neural Eng. 22, 066027.

## What the Code Does

The repo provides automated algorithms to detect and characterize four types of epileptiform activity from 10-second EEG segments (19 raw channels, 200 Hz):

- **LRDA** (Lateralized Rhythmic Delta Activity)
- **GRDA** (Generalized Rhythmic Delta Activity)
- **LPD** (Lateralized Periodic Discharges)
- **GPD** (Generalized Periodic Discharges)

For each segment, the algorithms output:
- **Frequency** of the epileptiform activity (Hz)
- **Spatial extent** (fraction of channels affected, 0–1)
- **Spatial areas** (list of affected brain regions)
- **Event type** classification (LRDA vs GRDA, based on spatial extent > 80% threshold)

## Key Algorithm: RDA1b-FFT (Best Performer for RDA)

Located in `code/rda_detector/rda1b_fft.py`. This is the best-performing algorithm for RDA according to the paper's ICC and MAE metrics.

### Processing Pipeline

1. **Preprocessing**: 60 Hz notch filter + 0.5–40 Hz bandpass filter
2. **Re-referencing**: Raw 19 channels → 18 bipolar channels (longitudinal banana montage)
3. **Channel validation** (optional): Reject channels where variance is disproportionate to signal range
4. **Per-channel spectral analysis**:
   - Compute power spectral density (PSD) via FFT
   - Fit FOOOF model (fitting oscillations and one-over-f) to decompose spectrum into aperiodic (1/f) and periodic (peaks) components
   - Extract peak parameters: center frequency, power, bandwidth, fit error, R²
5. **Peak prominence test** (the key step for RDA detection):
   - For each channel's detected peak, compute mean spectral power within the peak bandwidth
   - Compute mean spectral power in the rest of the 0.5–3 Hz delta band (excluding the peak)
   - If peak power > baseline power → channel has RDA (binary decision)
6. **Aggregation**: Median frequency across detected channels, count of channels, region mapping

### Channel-to-Region Mapping

The 18 bipolar channels map to 8 brain regions:

| Region | Hemisphere | Channels |
|--------|-----------|----------|
| LF (Left Frontal) | Left | Fp1-F7, Fp1-F3, F3-C3, F7-T3 |
| RF (Right Frontal) | Right | Fp2-F8, Fp2-F4, F4-C4, F8-T4 |
| LT (Left Temporal) | Left | T3-T5, T5-O1 |
| RT (Right Temporal) | Right | T4-T6, T6-O2 |
| LCP (Left Central-Parietal) | Left | C3-P3 |
| RCP (Right Central-Parietal) | Right | C4-P4 |
| LO (Left Occipital) | Left | P3-O1 |
| RO (Right Occipital) | Right | P4-O2 |

Left hemisphere channels: Fp1-F7, Fp1-F3, F3-C3, F7-T3, T3-T5, T5-O1, C3-P3, P3-O1 (indices 0–3, 6–7, 9–10 in bipolar_channels)
Right hemisphere channels: Fp2-F8, Fp2-F4, F4-C4, F8-T4, T4-T6, T6-O2, C4-P4, P4-O2 (indices 4–5, 8, 11–14 in bipolar_channels)
Midline channels: Fz-Cz, Cz-Pz (indices 16–17)

### The Critical Code (line 121–137 of rda1b_fft.py)

```python
for pk_idx, pk in enumerate(pks):
    closest_value = min(freqs, key=lambda x: abs(x - pk))

    # Spectral power within the peak bandwidth
    select_pk_bw = (freqs > closest_value - (bw[pk_idx]/2)) & (freqs < closest_value + (bw[pk_idx]/2))

    # Mean baseline power in 0.5-3 Hz delta band, excluding the peak
    condition_proeminence = np.mean(spectra[pk_idx, (freqs<=3) & (freqs>=0.5) & ~select_pk_bw])

    # Binary decision: peak > baseline?
    if np.mean(spectra[pk_idx, select_pk_bw]) > condition_proeminence:
        idx2.append(pk_idx)
```

Currently this comparison is reduced to a boolean. The ratio `peak_power / baseline_power` is a natural continuous "RDA-ishness" score that is computed but discarded.

## Reproduction Results

We successfully reproduced the paper's key results using the pre-computed algorithm outputs and expert annotations. The reproduction script is at `reproduce_results.py`.

### Table 1 MAE (segments with 100% annotator agreement on classification)

**RDA Frequency MAE** (ours / paper):

| Algorithm | LRDA | Paper | GRDA | Paper |
|-----------|------|-------|------|-------|
| RDA1a-FFT | 0.19 | 0.18 | 0.08 | 0.24 |
| RDA1b-FFT | 0.12 | 0.13 | 0.09 | 0.26 |
| RDA2-HHT | 0.24 | 0.13 | 0.52 | 0.46 |

**PD Spatial MAE** (near-exact matches):

| Algorithm | LPD | Paper | GPD | Paper |
|-----------|-----|-------|-----|-------|
| PD1 | 0.53 | 0.53 | 0.01 | 0.01 |
| PD2a | 0.16 | 0.17 | 0.40 | 0.40 |
| PD2b | 0.59 | 0.59 | 0.02 | 0.01 |

### ICC (Figures 5 & 6, using ICC3k)

RDA ICC values reproduced exactly: LRDA ee-IRR Freq=88% (paper: 88%), RDA1b-FFT=91% (paper: 91%), GRDA ee-IRR=92% (paper: 92%), RDA1b-FFT=96% (paper: 96%).

## Development Plan: Continuous Per-Channel RDA Score

### Goal

Extend the RDA1b-FFT detector to output a continuous per-channel score quantifying how strongly each channel exhibits rhythmic delta activity, enabling principled lateralization analysis.

### Proposed RDA Score

For each bipolar channel, define:

```
rda_score = mean_peak_power / mean_baseline_power
```

Where:
- `mean_peak_power` = mean PSD within the detected peak ± bandwidth/2
- `mean_baseline_power` = mean PSD in the 0.5–3 Hz delta band excluding the peak region

Interpretation:
- `rda_score = 1.0` → peak power equals baseline (no RDA evidence)
- `rda_score > 1.0` → peak exceeds baseline (current binary threshold for "has RDA")
- `rda_score = 3.0` → peak is 3x the baseline (strong RDA)
- `rda_score < 1.0` or `NaN` → no valid peak detected

Channels with no detected FOOOF peak get `rda_score = NaN` (or 0, depending on convention).

### Proposed Laterality Index

```
laterality_index = (mean_right_scores - mean_left_scores) / (mean_right_scores + mean_left_scores)
```

Range: -1 (fully left-lateralized) to +1 (fully right-lateralized), 0 = symmetric.

Use mean of `rda_score` values across left-hemisphere and right-hemisphere channels. Midline channels (Fz-Cz, Cz-Pz) excluded from the laterality calculation. For channels with no detected peak, use `rda_score = 1.0` (neutral) so they don't bias the index.

### Implementation Plan

#### Step 1: Modify `fcn_rdafooof_enhanced` to return per-channel scores

In `code/rda_detector/rda1b_fft.py`, the inner function `fcn_rdafooof_enhanced` currently returns only the channels that pass the binary threshold. Modify it to also return an 18-element array of RDA scores (one per bipolar channel).

Changes needed in `fcn_rdafooof_enhanced`:
- Initialize `rda_scores = np.full(len(seg), np.nan)` at the top
- Inside the prominence loop, compute `rda_scores[pk_idx] = mean_peak_power / mean_baseline_power` for every channel (not just those that pass)
- Return `rda_scores` as an additional output

#### Step 2: Modify `rda1b_fft` to propagate per-channel scores

Add to the returned `data_obj`:
- `channel_rda_scores`: dict mapping each bipolar channel name → its RDA score
- `channel_frequencies`: dict mapping each bipolar channel name → its detected peak frequency (or NaN)
- `laterality_index`: computed from left vs right hemisphere scores

#### Step 3: Create a new wrapper function

```python
def analyze_rda(segment, fs, channel_filter=1):
    """
    Analyze an EEG segment for rhythmic delta activity.

    Parameters:
        segment: numpy array, shape (19, n_samples) — raw 19-channel EEG at fs Hz
        fs: sampling frequency (typically 200)
        channel_filter: 1 to reject noisy channels, 0 to include all

    Returns:
        dict with keys:
            # Existing outputs (backward compatible)
            type_event: 'LRDA', 'GRDA', or NaN
            event_frequency: median frequency across detected channels (Hz)
            spatial_extent: fraction of channels with RDA (0-1)
            spatial_areas: list of region labels with RDA

            # New per-channel outputs
            channel_scores: dict of {channel_name: rda_score} for all 18 channels
            channel_frequencies: dict of {channel_name: peak_freq_hz} for all 18 channels
            region_scores: dict of {region_name: mean_rda_score} for 8 regions

            # Lateralization
            laterality_index: float in [-1, +1], negative=left, positive=right
            left_mean_score: mean RDA score across left hemisphere channels
            right_mean_score: mean RDA score across right hemisphere channels
    """
```

#### Step 4: Update the extraction script

Modify `extract_frequency_spatial_extent.py` to call the new function and save the additional columns (per-channel scores, laterality index) to the results CSV.

#### Step 5: Validate

- Run on the existing LRDA/GRDA dataset
- Verify that the binary RDA detection results are unchanged (backward compatible)
- Check that LRDA segments tend to have laterality_index != 0 and GRDA segments tend to have laterality_index near 0
- Examine whether laterality_index correlates with expert spatial_area annotations

### File Structure for New Project

```
new-repo/
├── code/
│   ├── rda_detector/
│   │   ├── rda1b_fft.py          # Modified with per-channel scores
│   │   ├── rda1a_fft.py          # Unchanged
│   │   └── rda2_hht.py           # Unchanged
│   ├── analyze_rda.py            # New wrapper function (Step 3)
│   ├── extract_with_laterality.py # Updated extraction script (Step 4)
│   └── validate_laterality.py    # Validation script (Step 5)
├── data/                         # Symlink or copy from BDSP
├── results/                      # Will contain new results with laterality
└── LATERALIZATION_PLAN.md        # This file
```

### Dependencies

The core analysis requires: numpy, scipy, matplotlib, pandas, mne, fooof, tqdm, hdf5storage, h5py.

Note: `fooof` requires Python < 3.10. The original environment uses Python 3.8.19. The conda environment spec is in `code/environment.yml`. On macOS ARM, creating from the yml works with `conda env create -f code/environment.yml` if you have conda/miniforge installed.

### Key Code References

- Main detector: `code/rda_detector/rda1b_fft.py` — function `rda1b_fft()` (line 154) and `fcn_rdafooof_enhanced()` (line 63)
- Channel definitions: `bipolar_channels` list (line 19) and `mono_channels` (line 21)
- Region mapping logic: lines 228–244 of `rda1b_fft.py`
- LRDA vs GRDA threshold: line 248 (`len(channels)/seg.shape[0] > 0.8`)
- Prominence test (where the score should be extracted): lines 128–137
- Extraction script: `code/extract_frequency_spatial_extent.py`
- Paper: `Tăuțan_2025_J._Neural_Eng._22_066027.pdf` in the repo root
