# Plan: RDA Spatial Localization via Narrowband Amplitude Topography

## Motivation

For PDs, we developed discharge-locked topographic localization: extract the 19-channel voltage at each discharge peak, average across discharges, and visualize the resulting topography. This works because PDs are discrete events with well-defined peaks.

RDA is fundamentally different — it's continuous rhythmic delta activity without sharp peaks. There's no single "event" to lock to. However, the same underlying question applies: **where on the scalp is the rhythmic activity maximal?**

## Key Insight

For RDA, the **narrowband amplitude topography IS the localization map**. We don't need event-locked averaging because the pattern is quasi-stationary — it's present throughout the segment. The spatial distribution of narrowband (delta-band) amplitude across channels directly answers "where is the RDA?"

This is closely related to what we already compute for RDA spatial extent (PLV×Amplitude), but reformulated as a topographic map suitable for the same visualization and verbal description pipeline we built for PDs.

## Proposed Algorithm

### Step 1: Frequency Estimation
Use W05_DomOnly_IterRefine (already implemented):
- Bandpass 0.5–3.5 Hz
- Coarse lateralization via variance
- Hilbert frequency from top-3 dominant hemisphere channels
- Iterative narrowband refinement

### Step 2: Narrowband Filtering
Bandpass at estimated_frequency ± 0.4 Hz (same as current RDA pipeline).

### Step 3: Per-Channel Amplitude Envelope
For each of the 19 monopolar channels:
- Compute Hilbert analytic signal of the narrowband-filtered data
- Take the amplitude envelope: `amp[ch] = mean(abs(hilbert(narrowband[ch])))`
- This gives a 19-element vector representing "how much rhythmic activity at the target frequency exists at each electrode"

### Step 4: Laplacian Transform (optional, for CSD approximation)
Apply the same spatial Laplacian we use for PDs:
- Each channel's amplitude minus mean of its neighbors' amplitudes
- This sharpens the topography, removing volume-conducted spread
- Particularly important for LRDA where we need to distinguish the involved hemisphere from passively-conducted contralateral activity

### Step 5: Topographic Visualization
Same pipeline as PD discharge-locked topography:
- MNE spherical spline interpolation
- RdBu_r colormap (or single-color since amplitudes are non-negative)
- Toggle between monopolar (average ref) and Laplacian views
- Channel names with adaptive text color

### Step 6: Verbal Description
Same morgoth-viewer `describe_ied_topoplot()` function:
- Input: 19-channel amplitude vector (absolute values)
- Output: standardized regional descriptor (e.g., "left temporal", "bilateral frontal (left predominant)")
- Laterality from W05 (primary L/R determination)

## Differences from PD Approach

| Aspect | PD (discharge-locked) | RDA (narrowband amplitude) |
|--------|----------------------|---------------------------|
| **Input** | Voltage at discharge peaks | Mean amplitude envelope of narrowband signal |
| **Averaging** | Across discrete events (5–20 discharges) | Across time (continuous, full 10s segment) |
| **Alignment** | GFP-based, two-pass template refinement | Not needed (amplitude envelope is already smooth) |
| **Phantom suppression** | GFP²-weighted averaging | Not needed (no discrete events to misdetect) |
| **Noise robustness** | √N averaging across discharges | Inherent in envelope computation (smooths fast noise) |
| **Polarity** | Both positive and negative voltage | Always non-negative (amplitude envelope) |
| **Colormap** | RdBu_r (diverging, shows polarity) | Hot/inferno (sequential, amplitude only) |
| **Laterality source** | PDCharacterizer | W05_DomOnly_IterRefine |

## Colormap Choice

Since RDA amplitude is always non-negative (it's an envelope), we should use a **sequential colormap** (e.g., `inferno`, `hot`, or `YlOrRd`) rather than the diverging RdBu_r used for PDs. Red = high amplitude = strong RDA. This avoids the confusing polarity issues we encountered with PDs.

## Implementation Plan

### Phase 1: Core Algorithm
- [ ] Create `rda_spatial_localization.py` in `code/`
  - Function: `rda_localization_topo(mono_19ch, freq_hz)` → returns amplitude topography (19,) and Laplacian topography (19,)
  - Reuse existing functions: `_bandpass()`, `compute_laplacian()` from `rda_spatial_extent.py`

### Phase 2: Topoplot Generation
- [ ] Adapt `generate_topoplot_b64()` for sequential colormap
- [ ] Use `describe_ied_topoplot()` from morgoth-viewer for verbal description
- [ ] Generate both monopolar and Laplacian topoplots

### Phase 3: Viewer
- [ ] Build HTML viewer similar to PD discharge topo viewer
  - EEG traces with narrowband overlay (green)
  - Topoplot on right
  - Verbal description below
  - 3 montage toggle (bipolar, average ref, Laplacian)
  - Toggle monopolar/Laplacian topoplot
  - 200 cases: 100 LRDA + 100 GRDA with ≥10 IIIC votes and MW frequency labels

### Phase 4: Evaluation
- [ ] Compare narrowband amplitude topography against expert spatial extent labels
- [ ] Compute correlation between topo-derived spatial extent (count channels above threshold) and expert labels
- [ ] Qualitative review: do the topoplots match clinical impression?

## Relationship to Existing PLV×Amp Method

The current `rda_spatial_extent.py` already computes something closely related:
- PLV×Amplitude per channel = phase coherence with reference × narrowband amplitude
- This is effectively a "phase-coherent narrowband amplitude" — it measures both "is this channel rhythmic?" (PLV) and "how strong?" (amplitude)

The proposed narrowband amplitude topography is simpler — just the amplitude envelope without the PLV component. PLV is useful for spatial *extent* (binary: involved or not) because it helps reject channels that have delta power but aren't phase-locked to the RDA. But for *localization* (where is the maximum?), pure amplitude may be sufficient since the largest amplitude almost always corresponds to the generator.

We should compare:
1. **Pure narrowband amplitude** — simplest, just the envelope
2. **PLV×Amplitude** — our current metric, more selective
3. **Laplacian of narrowband amplitude** — sharpest localization

And pick whichever gives the most clinically plausible localizations.

## Timeline

This is a relatively straightforward adaptation of existing code. The core algorithm (steps 1–3) is essentially already implemented in `rda_spatial_extent.py`. The main new work is:
1. Reformulating the output as a 19-channel monopolar topography (not 18-channel bipolar scores)
2. Generating the topoplots and verbal descriptions
3. Building the viewer

Estimated effort: ~2 hours to implement, ~1 hour to evaluate.
