# Figure Plan: PDCharacterizer Paper

## Main Text Figures

### Figure 1: The Problem — Examples of Periodic and Rhythmic EEG Patterns
- **Current file**: `fig0_eeg_examples.png`
- **Generator**: `paper_materials/generate_fig0_examples.py`
- **Content**: 3×2 grid of raw EEG examples in average reference montage. No algorithm markup.
  - A: Clear LPD (96% agreement, 26 votes)
  - B: Clear GPD (90% agreement, 20 votes)
  - C: Clear LRDA (92% agreement, 13 votes)
  - D: Clear GRDA (86% agreement, 14 votes)
  - E: Ambiguous LPD (58% agreement — LPD vs seizure vs other)
  - F: Ambiguous GRDA (50% agreement — GRDA vs LPD/GPD/LRDA)
- **Paper section**: Introduction
- **Purpose**: Illustrate the 4 IIIC pattern types and the challenge of classification when experts disagree
- **Verbal description for PaperBanana**:
  A 3×2 grid of 10-second EEG recordings displayed in common average reference montage with 19 electrodes (standard 10-20 system). The layout groups channels as: left parasagittal (Fp1, F3, C3, P3), left temporal (F7, T3, T5, O1), midline (Fz, Cz, Pz), right parasagittal (Fp2, F4, C4, P4), right temporal (F8, T4, T6, O2), with small gaps between groups. Each panel shows black traces on white background, with a scale bar (100 µV vertical, 1 second horizontal) in the lower right corner. Panel labels A-F are bold in the upper left of each panel. A small subtitle below each label states the pattern type, agreement percentage, and number of voters. No annotations, markers, or algorithmic output should appear — these are raw EEG traces only. Time axis 0-10 seconds at bottom. The top 4 panels (A-D) show clear examples that most experts agree on; the bottom 2 panels (E-F) show ambiguous cases where expert opinions diverge. All panels should be the same size and have consistent gain/scaling.

---

### Figure 2: PD Characterization Pipeline — ChannelPD-Net + HemiCET+DP + Discharge-Locked Topography
- **Current file**: `figS2_hemicet_composite.png` (needs updating)
- **Generator**: needs new script or manual update
- **Content**: Three-panel composite:
  - Left panel: Real EEG input (8-channel hemisphere, from one of our LPD examples)
  - Center panel: Architecture flowchart showing the full PD pipeline:
    - ChannelPD-Net (per-channel CNN → 18 PD probs + 18 freq estimates)
    - ↓ feeds into 3 parallel branches:
      1. Laterality Detection (L vs R hemisphere mean PD prob → AUC 0.963)
      2. HemiCET+DP Discharge Detection (CNN evidence → DP with periodic prior → discharge times → IPI frequency)
      3. Discharge-Locked Topographic Localization (GFP-aligned averaging → Laplacian topoplot → morgoth-viewer verbal description)
  - Right panel: Output visualization — EEG with discharge markers (ipsilateral only) + Laplacian topoplot + verbal description
- **Paper section**: Methods 3.1-3.4
- **Purpose**: Show the unified PD pipeline from input to all 4 outputs (laterality, spatial localization, timing, frequency)
- **Verbal description for PaperBanana**:
  A three-panel horizontal composite figure. LEFT PANEL: A real 8-channel EEG recording from one hemisphere showing periodic discharges — this should be actual EEG data, not a schematic. The channels should be labeled (e.g., Fp1-F7, F7-T3, etc. for bipolar, or Fp1, F3, C3, P3, F7, T3, T5, O1 for monopolar). The EEG should show clear periodic sharp transients. CENTER PANEL: A flowchart/architecture diagram with colored boxes. The flow starts at the top with "Input: 10s, 19-channel Monopolar EEG (200 Hz)" and flows downward. First box (blue): "ChannelPD-Net" — per-channel CNN+Attention, producing 18 PD probabilities and 18 frequency estimates. This feeds into three parallel branches via dashed arrows: Branch 1 (green, left): "Laterality Detection" — compare L vs R hemisphere mean PD probability → output: Left/Right, AUC=0.963. Branch 2 (red, center): "HemiCET+DP" — 8-channel hemisphere CET-UNet → evidence trace → CNN+ACF frequency prior → dynamic programming with periodic prior → discharge times → IPI frequency. Sub-boxes show: evidence generation, candidate extraction, DP optimization, EM template refinement, confidence filtering. Branch 3 (orange, right): "Discharge-Locked Topographic Localization" — at each detected discharge time, extract 19-channel voltage → Laplacian-GFP alignment → two-pass template refinement → GFP²-weighted averaging → MNE spherical spline topoplot → morgoth-viewer 16-region verbal description. Bottom (outputs): four output boxes: Laterality (Left/Right), Spatial Localization (topoplot + "left frontotemporal"), Discharge Times (t₁, t₂, ..., tₖ), Frequency (1/median(IPI)). RIGHT PANEL: The output visualization for a real LPD case — top shows 18-channel EEG with vertical red dashed lines marking detected discharge times (only on the involved hemisphere channels), bottom shows the Laplacian topoplot (inferno colormap) with electrode labels, and below that the verbal description text (e.g., "LPD, left sided (unilateral), at 1.5 Hz, left frontotemporal").

---

### Figure 3: RDA Characterization Pipeline — W05 Iterative Hilbert + PLV×Amplitude + Narrowband Amplitude Topography
- **Current file**: `figS5_hilbert_cv_composite.png` (needs updating)
- **Generator**: needs new script or manual update
- **Content**: Similar three-panel composite for RDA:
  - Left panel: Real LRDA EEG input showing rhythmic delta
  - Center panel: Architecture flowchart:
    - Pass 1: Coarse lateralization (variance per hemisphere) → dominant side → Hilbert frequency from top-3 channels
    - Pass 2: Narrowband at estimated freq ± 0.4 Hz → refined lateralization (envelope amplitude) → refined frequency
    - Spatial: PLV×Amplitude per channel (phase coherence × narrowband amplitude) → threshold → spatial extent
    - Localization: Narrowband amplitude envelope per monopolar channel → Laplacian → topoplot → verbal description
  - Right panel: Output — EEG with narrowband overlay + Laplacian topoplot + verbal description
- **Paper section**: Methods 3.5-3.6
- **Purpose**: Show the RDA pipeline, emphasizing the iterative refinement and amplitude-weighted phase coherence
- **Verbal description for PaperBanana**:
  A three-panel horizontal composite figure, same layout as Figure 2 but for RDA patterns. LEFT PANEL: A real 19-channel LRDA EEG recording in average reference montage showing rhythmic delta activity that is clearly lateralized (larger amplitude on one side). Channels labeled with standard 10-20 names. CENTER PANEL: A flowchart showing the two-pass iterative approach. Top: "Input: 10s, 19-channel Monopolar EEG (200 Hz)". Pass 1 (light blue box): "Coarse Analysis" — bandpass 0.5-3.5 Hz → mean variance per hemisphere → identify dominant side → Hilbert instantaneous frequency from top-3 dominant channels → coarse frequency estimate. Arrow down to Pass 2 (darker blue box): "Narrowband Refinement" — bandpass at estimated_freq ± 0.4 Hz → refined lateralization via envelope amplitude → refined Hilbert frequency from dominant hemisphere. Two output branches: Branch 1 (green): "Spatial Extent" — PLV×Amplitude per channel — for each channel, compute phase coherence with dominant-hemisphere reference AND narrowband amplitude, multiply → 18 channel scores → threshold → count/18. Note: amplitude weighting downweights contralateral volume-conducted signals. Branch 2 (orange): "Topographic Localization" — per-channel narrowband Hilbert amplitude envelope → Laplacian transform → MNE topoplot (inferno) → describe_ied_topoplot() → verbal description. Bottom outputs: Laterality (L/R), Spatial Extent (0-1), Frequency (Hz), Localization (topoplot + "left temporal"). RIGHT PANEL: Output visualization for a real LRDA case — EEG traces with green narrowband overlay at the estimated frequency, Laplacian topoplot (inferno), and verbal description (e.g., "LRDA, left sided (unilateral), at 1.5 Hz, left temporal").

---

### Figure 4: Frequency Estimation Results — Scatter Plots
- **Current file**: `fig6_frequency_scatter.png`
- **Generator**: `paper_materials/generate_fig6.py`
- **Content**: 2×4 scatter plot grid. Rows: PDCharacterizer/W05 (top) vs Tautan et al. (bottom). Columns: LPD, GPD, LRDA, GRDA. Each panel shows predicted frequency (y) vs expert frequency (x) with Spearman ρ and MAE. Black stars = ≥3 expert raters, colored circles = 1-2 raters. IIIC-standard subtype colors.
- **Paper section**: Results 4.1
- **Purpose**: Primary quantitative result for frequency estimation
- **No PaperBanana update needed** — this is a pure data figure generated from code.

---

### Figure 5: LPD Characterization Examples
- **Current file**: `fig1_lpd_characterization.png`
- **Generator**: `paper_materials/render_figures.py` from `figure_lpd_examples_data.json`
- **Content**: 3 rows (Easy 96% / Medium 78% / Hard 60%), each showing average-reference EEG with discharge markers + Laplacian topoplot + verbal description
- **Paper section**: Results 4.2
- **Verbal description for PaperBanana**:
  Three rows, each showing one LPD case at decreasing levels of inter-rater agreement (Easy/Medium/Hard). Each row is divided into left (75% width) and right (25% width). LEFT: 19-channel EEG in average reference montage with channels grouped as L parasagittal, L temporal, midline, R parasagittal, R temporal. Black traces on white background. Light blue shading on the involved hemisphere's channels. Red dashed vertical lines at detected discharge times. Thin red discharge time labels at the top. Scale bar in lower right. A difficulty badge in upper right corner (green "EASY", orange "MEDIUM", red "HARD" with agreement percentage). RIGHT: Top — Laplacian topoplot (inferno colormap, circular head outline with nose at top, electrode names in adaptive black/white text). Title "Laplacian topography" above. Bottom — verbal description text (e.g., "LPD, left sided (unilateral), at 1.1 Hz, left posterior temporal"). Figure title "LPD Characterization Examples" centered at top.

---

### Figure 6: GPD Characterization Examples
- **Current file**: `fig2_gpd_characterization.png`
- **Same format as Figure 5** but for GPD. Bilateral shading (both hemispheres + midline). No laterality in verbal description. Same layout.

---

### Figure 7: LRDA Characterization Examples
- **Current file**: `fig3a_lrda_characterization.png`
- **Same format as Figure 5** but for LRDA. Green narrowband overlay instead of red discharge markers. Laplacian topoplot from narrowband amplitude (not discharge-locked). Unilateral shading.

---

### Figure 8: GRDA Characterization Examples
- **Current file**: `fig3b_grda_characterization.png`
- **Same format as Figure 6** but for GRDA. Green narrowband overlay. Bilateral shading. Topoplot from narrowband amplitude.

---

## Supplemental Figures

### Figure S1: Inter-Rater Reliability Comparison (ICC/PA)
- **Current file**: `figS8_irr_comparison.png`
- **Generator**: `paper_materials/generate_fig_irr.py`
- **Content**: 2×4 grid of ICC and PA bars comparing expert-expert vs expert-algorithm agreement
- **Purpose**: Direct comparison with Tautan et al. methodology. Supplemental because PA binning is suboptimal for continuous variables, and spatial extent expert labels are noisy.

### Figure S2: Spatial Extent Scatter Plots
- **Current file**: `figS9_spatial_scatter.png`
- **Generator**: `paper_materials/generate_fig_spatial_scatter.py`
- **Content**: 2×4 scatter plots of predicted vs expert spatial extent, per-rater dots
- **Purpose**: Shows spatial extent prediction quality but also illustrates poor expert consistency — motivates the topographic localization approach.

### Figure S3: Spatial Extent Threshold Optimization
- **Current file**: `threshold_sweep_spatial.png`
- **Generator**: `paper_materials/generate_threshold_sweep.py`
- **Content**: MAE/Pearson/ICC vs threshold curves for PD and RDA spatial extent
- **Purpose**: Documents threshold selection.

---

## Figures to Delete
- `fig4_system_pipeline.png` — redundant with updated Fig 2, has typos
- `fig4_system_pipeline_alt.png` — alternate version, also redundant
- `figS1_hpp_algorithm.png` — old HPP schematic, superseded by HemiCET
- `figS3_timing_examples.png` — timing examples, could be useful but not essential
- `figS4_evidence_comparison.png` — evidence comparison, technical detail
- `figS7_failure_cases.png` — failure cases, could move to supplemental if desired

---

## File Renaming Plan

| Current | New | Notes |
|---------|-----|-------|
| `fig0_eeg_examples.png` | `fig1_eeg_examples.png` | |
| `figS2_hemicet_composite.png` | `fig2_pd_pipeline.png` | needs regeneration |
| `figS5_hilbert_cv_composite.png` | `fig3_rda_pipeline.png` | needs regeneration |
| `fig6_frequency_scatter.png` | `fig4_frequency_scatter.png` | |
| `fig1_lpd_characterization.png` | `fig5_lpd_characterization.png` | |
| `fig2_gpd_characterization.png` | `fig6_gpd_characterization.png` | |
| `fig3a_lrda_characterization.png` | `fig7_lrda_characterization.png` | |
| `fig3b_grda_characterization.png` | `fig8_grda_characterization.png` | |
| `figS8_irr_comparison.png` | `figS1_irr_comparison.png` | |
| `figS9_spatial_scatter.png` | `figS2_spatial_scatter.png` | |
| `threshold_sweep_spatial.png` | `figS3_threshold_sweep.png` | |

---

## What Still Needs Work

1. **Fig 2 (PD pipeline)**: Update figS2_hemicet_composite with real EEG input, add discharge-locked topo branch, show ipsilateral-only markers
2. **Fig 3 (RDA pipeline)**: Update figS5_hilbert_cv with W05 iterative method, add PLV×Amp spatial, add narrowband amplitude topo
3. **Dataset summary table**: Not a figure — create as Table 1 in the paper (label counts by subtype and rater)
4. **Comparison table**: Table 2 — our method vs Tautan et al. across all metrics
