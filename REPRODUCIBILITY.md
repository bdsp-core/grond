# GROND reproducibility

This document explains how to reproduce the figures, tables, and statistical results in the GROND manuscript.

## Quick start

```sh
# 1. Clone the repo + initialize submodules (manuscript lives in a submodule)
git clone https://github.com/bdsp-core/grond
cd grond
git submodule update --init --recursive

# 2. Create the conda environment
conda env create -f environment.yml
conda activate morgoth

# 3. Get the EEG data bank (grond_data.h5, 1.67 GB)
#    Option A: download from S3 (BDSP credentials required):
aws s3 cp s3://bdsp-opendata-credentialed/grond/grond_data.h5 data/grond_data.h5
#    Option B: rebuild from data/eeg/ + data/labels/ (no S3 needed):
python code/data_management/build_grond_h5_bank.py --out data/grond_data.h5

# 3a. (Optional) Download the canonical labeling viewers (4 self-contained HTMLs,
#     470 MB total) — the actual tools used by the four raters to score the
#     200-per-subtype IRR cohort. Useful for inspecting how labels were created.
aws s3 cp --recursive s3://bdsp-opendata-credentialed/grond/independent_expert_tasks/ \
    paper_materials/independent_expert_tasks/viewers/

# 4. Regenerate every figure and table
python paper_materials/generate_all_figures.py

# 5. Build the manuscript PDF
cd paper_materials/manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex   # second pass for refs
```

## Data layout

| Artifact | Path | Contents |
|---|---|---|
| EEG bank | `data/grond_data.h5` | All 14,341 10-second EEG segments + labels + algorithm predictions + cohort flags. Self-contained, single-file. See "Data bank schema" below. |
| EEG individual files | `data/eeg/{pid}_seg000.mat` | One .mat per segment. Same content as the h5; kept for backward compatibility with the optimization harness. |
| Per-segment label index | `data/labels/segments.csv` | One row per segment: patient_id, subtype, source, .mat path, label-availability flags. |
| Frequency / laterality long-form | `data/labels/labels.csv` | Long-format expert annotations: one row per (rater, label_type, segment). |
| Subtype + IIIC crowd votes | `data/labels/segment_labels.csv` | Per-segment IIIC vote distribution and consensus subtype. |
| Discharge timing | `data/labels/discharge_times.json` | PD-GT cohort discharge-timing annotations (657 PD-GT patients). |
| IRR-cohort manifests | `paper_materials/independent_expert_tasks/{lpd,gpd,lrda,grda}/manifest.csv` | The 200-per-subtype canonical IRR cohort (MW/SZ/TZ/AS) and its stratification. |
| Trained models | `data/hemi_cache/hemi_cet_v2/`, `data/pd_channel_cache/`, `data/cet_cache/` | 5-fold ensembles. |

## Data bank schema (`data/grond_data.h5`)

The h5 file is self-contained: every label, prediction, and cohort flag is keyed by `segment_id` (the .mat basename without extension).

```
/metadata/                              # global attrs and channel definitions
   channel_names_bipolar, channel_names_mono, bipolar_pair_definitions_json,
   fs_hz=200, n_samples_per_segment=2000, n_bipolar_channels=18, citation

/segments/{segment_id}/                 # 14,341 segments
   eeg                                  # (18, 2000) float32 bipolar EEG at 200 Hz
   attrs: patient_id, subtype, eeg_source, mat_file, has_discharge_timing

/labels/{segment_id}/
   discharge_times                      # (N,) float32 — discharge times in seconds (PD-GT only)
   attrs: freq_hz_consensus, freq_per_rater_json, laterality_consensus,
          laterality_per_rater_json, spatial_channels, review_status,
          review_source, gt_laterality, selected_freq

/predictions/pdchar/{segment_id}/       # PD-Profiler outputs
   attrs: pdchar_freq_hz, pdchar_laterality, pdchar_spatial_extent

/predictions/tautan/{segment_id}/       # Täutan et al. (2025) baseline outputs
   attrs: tautan_freq_hz, tautan_spatial_extent

/predictions/rda_plv/{segment_id}/      # RDA-PLV spatial-extent outputs
   attrs: rda_plv_spatial_extent

/cohorts/
   segment_ids                          # (N,) str — order key for the flag arrays
   pd_gt                                # (N,) bool — discharge-timing GT cohort
   irr_canonical                        # (N,) bool — canonical IRR cohort (MW/SZ/TZ/AS)
```

## Pipeline components

The PD-Profiler is a pipeline of trained neural nets + signal processing:

| Component | Type | Input → Output | Subtask |
|---|---|---|---|
| **ChannelPD-Net** (`code/pd_channel_detector/`) | CNN+Attention, 5-fold ensemble | 1 × 2000 (single bipolar channel) → (P(PD), log-freq) | Lateralization (mean L vs R prob), spatial selection input |
| **HemiCET-UNet** (`code/hemi_detector/`) | 1D U-Net, 5-fold ensemble | 8 × 2000 (hemisphere) → 2000-sample evidence trace | Discharge timing |
| **DP + EM + ACF** (`code/discharge_detector.py`) | Signal processing | Evidence trace + ACF freq prior → discrete discharge times | Timing + IPI-derived frequency |
| **Discharge-locked Laplacian topography** (`code/pd_profiler.py`) | Signal processing | 19-ch monopolar + DP times → spatial map | PD spatial extent |
| **BIPD GBT classifier** (`code/bipd_detector.py`) | LightGBM / sklearn GBT | 29 features → LPD/GPD/BIPD | 3-way classification |

The RDA-Profiler (`code/rda_detector/`) is fully signal-processing — iterative narrowband Hilbert refinement (NB-Hilbert) for frequency and lateralization, with PLV analysis for spatial extent. No trainable parameters.

## Trained model weights

| File | Used by | Training script |
|---|---|---|
| `data/hemi_cache/hemi_cet_v2/hemi_cet_fold{0..4}.pt` | HemiCET-UNet 5-fold ensemble | `code/hemi_detector/train_hemi_cet.py` |
| `data/pd_channel_cache/cnn_attn_fold{0..4}.pt` | ChannelPD-Net 5-fold ensemble | `code/pd_channel_detector/train_cnn_attention.py` |
| `data/cet_cache/cet_unet_fold{0..4}.pt` | CET-UNet (legacy variant; production uses HemiCET) | `code/cet_model/train_cet.py` |
| `data/models/bipd_gbt.pkl` | BIPD GBT classifier | `code/bipd_detector.py --train` |

To retrain HemiCET-UNet on the current labels (used for the headline discharge-timing F1):
```sh
conda run -n morgoth python code/hemi_detector/train_hemi_cet.py
# ~8 minutes on Apple Silicon MPS for 5 folds × 80 epochs
```

## Regenerating individual results

| Result | Script | Inputs |
|---|---|---|
| Discharge timing F1 / sens / prec / freq ρ | `code/evaluation/verify_c1_repro.py` | `grond_data.h5` + hemi_cache + pd_channel_cache |
| Architecture comparison (Table 6) | `code/cet_model/eval_all_methods.py` | `grond_data.h5` + all model caches |
| Tautan PHLBSZ cohort (Table 5) | `code/evaluation/analyze_phlbsz_cohort.py` | `grond_data.h5` + tautan predictions |
| BIPD detection (Table 7) | `code/bipd_detector.py` | hemi_cache + DP pipeline |
| fig 0 (EEG examples) | `paper_materials/generate_fig0_examples.py` | `grond_data.h5` + IRR manifests |
| fig 1 (PD pipeline) | `paper_materials/build_fig2.py` | `grond_data.h5` + model weights |
| fig 2 (RDA pipeline) | `paper_materials/draw_panel_b_rda.py` | `grond_data.h5` + model weights |
| fig irr_main (EE vs EA bars) | `paper_materials/build_fig_irr_bars.py` | `grond_data.h5` + `results/independent_expert_v1/summary.json` |
| fig 5 (IRR) | `paper_materials/generate_fig_irr.py` | `grond_data.h5` + `results/independent_expert_v1/` |
| Tables 1–7 | `paper_materials/tables/generate_table{N}.py` | `grond_data.h5` |

One-shot regeneration of every figure and table:
```sh
python paper_materials/generate_all_figures.py
```

## Cohort definitions

- **PD-GT discharge-timing cohort** (n = 657 patients): all entries in `discharge_times.json` with `review_status == 'ground_truth'`, `subtype ∈ {lpd, gpd}`, and ≥ 2 discharge timestamps. Of these, 649 of 657 have resolvable EEG in the current `data/eeg/` layout.
- **Canonical IRR cohort** (n = 800 segments, 200 per subtype): defined by the four `paper_materials/independent_expert_tasks/{lpd,gpd,lrda,grda}/manifest.csv` files. Scored independently by raters MW, SZ, TZ, AS.
- **Täutan PHLBSZ cohort** (n = 38 patients): the prior published Täutan et al. cohort, used for the head-to-head comparison in Table 5. Scored by PH, LB, SZ, MW.

## Known limitations

Eight PD-GT discharge-timing patients (4 LPD + 4 GPD) have ground-truth timing labels in `discharge_times.json` but their source 10-second EEG segments could not be located in any of the searched archives (data/eeg, S3 `iiic-freq3/`, S3 `morgoth1/data/internal_dataset/`, local SSD caches). The exact bare-PID filenames to search for are listed in `paper_materials/reproducibility/missing_pd_segments_to_find.txt`. These 8 patients are excluded from the timing-F1 denominator (n = 649 resolvable / 657 with labels). If you find any of these files, drop the .mat into `data/eeg/` and append a row to `segments.csv`; the verifier will pick it up automatically.

## Citation

If you use the GROND data bank, please cite:

> Westover MB, Sun C, Zafar SF, Aboul Nour H, Ge W, Türeli A, et al. *GROND: Automated characterization of periodic discharges and rhythmic delta activity in critical-care EEG.* (2026).
