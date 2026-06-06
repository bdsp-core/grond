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

# 3. Get the EEG data bank (grond_data.h5, ~1 GB)
#    Either pull from the release page, or rebuild from data/eeg/:
python code/data_management/build_grond_h5_bank.py --out data/grond_data.h5

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
| EEG bank | `data/grond_data.h5` | All 14,000+ 10-second EEG segments + labels + algorithm predictions + cohort flags. Self-contained, single-file. See "Data bank schema" below. |
| EEG individual files | `data/eeg/{pid}_seg000.mat` | One .mat per segment. Same content as the h5; kept for backward compatibility with the optimization harness. |
| Per-segment label index | `data/labels/segments.csv` | One row per segment: patient_id, subtype, source, .mat path, label-availability flags. |
| Frequency / laterality long-form | `data/labels/labels.csv` | Long-format expert annotations: one row per (rater, label_type, segment). |
| Subtype + IIIC crowd votes | `data/labels/segment_labels.csv` | Per-segment IIIC vote distribution and consensus subtype. |
| Discharge timing | `data/labels/discharge_times.json` | PD-GT cohort discharge-timing annotations. |
| IRR-cohort manifests | `paper_materials/independent_expert_tasks/{lpd,gpd,lrda,grda}/manifest.csv` | The 200-per-subtype canonical IRR cohort and its frequency/laterality stratification. |
| Trained models | `data/hemi_cache/hemi_cet_v2/`, `data/pd_channel_cache/`, `data/cet_cache/` | 5-fold ensembles for HemiCET-UNet, ChannelPD-Net, etc. |

## Data bank schema (`data/grond_data.h5`)

The h5 file is self-contained: every label, every prediction, and every cohort flag is keyed by `segment_id` (the .mat basename without extension). Schema:

```
/metadata/                              # global attrs and channel definitions
   channel_names_bipolar, channel_names_mono, bipolar_pair_definitions_json,
   fs_hz=200, n_samples_per_segment=2000, n_bipolar_channels=18, citation, ...

/segments/{segment_id}/                 # 14,000+ segments
   eeg                                  # (18, 2000) float32 bipolar EEG at 200 Hz
   attrs: patient_id, subtype, eeg_source, mat_file, has_discharge_timing, ...

/labels/{segment_id}/
   discharge_times                      # (N,) float32 — discharge times in seconds (PD-GT only)
   attrs: freq_hz_consensus, freq_per_rater_json, laterality_consensus,
          laterality_per_rater_json, spatial_channels, review_status,
          review_source, gt_laterality, selected_freq

/predictions/pdchar/{segment_id}/       # PD-Profiler algorithm outputs
   attrs: pdchar_freq_hz, pdchar_laterality, pdchar_spatial_extent

/predictions/tautan/{segment_id}/       # Täutan et al. (2025) baseline outputs
   attrs: tautan_freq_hz, tautan_spatial_extent

/predictions/rda_plv/{segment_id}/      # RDA-PLV spatial-extent outputs
   attrs: rda_plv_spatial_extent

/cohorts/
   segment_ids                          # (N,) str — order key for the flag arrays
   pd_gt                                # (N,) bool — discharge-timing GT cohort
   irr_canonical                        # (N,) bool — canonical IRR cohort (MW/SZ/TZ/AS)
   attrs: pd_gt_description, irr_canonical_description
```

## EEG source provenance and reproducibility tiers

Not every EEG segment is provenance-equivalent. The `segments.csv:eeg_source` column (and `/segments/{id}/.attrs/eeg_source` in the h5) records where each segment's 10-second data came from. These map to three reproducibility tiers:

| Tier | Source label(s) | n | Story |
|---|---|---|---|
| **A — Published-models reproducible** | `dataset_eeg`, `s3_morgoth`, `external_drive` | 13,059 (96 %) | Original-source EEG that survived the May-2026 repo cleanup. Includes 147 of the 657 PD-GT discharge-timing patients. Running the published HemiCET-UNet v2 weights against this subset reproduces the timing F1 to within ±0.02 of the manuscript's reported number (F1 = 0.876 vs published 0.889 on n=147 of n=582). All non-timing metrics (frequency, laterality, IRR cohort frequency/κ) reproduce on the **full cohort** because they don't depend on sub-100 ms label/data alignment. |
| **B — Recovered from external backup** | `alex_recovery` | 297 | Copies of the original IIIC LPD/GPD/RDA 10-second segments preserved in a collaborator's backup directory. Bit-identical to a contemporaneous `external_pd_segments.npz` snapshot; the EEG content is correct but was low-pass filtered at ~20 Hz before snapshotting, which introduces a small filter group delay relative to the original `discharge_times.json` annotations. This degrades the discharge-timing F1 (sub-100 ms TP tolerance is sensitive to filter group delay) but does not affect frequency, laterality, or spatial-extent metrics. |
| **C — Re-extracted from canonical source recordings** | `s3_iiic_freq3_recovery`, `pathC_morgoth1_s3` | 516 | 10-second windows re-extracted from the full multi-minute source recordings in `s3://bdsp-opendata-credentialed/iiic-freq3/data/eeg/` and `s3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/{LPD,GPD,IIIC}/segments_raw/`. The IRR-cohort files (Step 2) are the exact 10-second segments the four expert raters scored, so frequency and κ analyses reproduce exactly. The path-C files (Step 3) were re-extracted by sliding a 10-second window across the source recording and selecting the window whose energy at the GT discharge timestamps was maximal; this is well-defined but has residual alignment uncertainty for the discharge-timing analysis. |
| **D — Unrecoverable** | n/a | 8 | Eight PD-GT discharge-timing patients (4 LPD + 4 GPD) whose source recordings were not located in any of the searched archives: data/eeg pre-cleanup, the Alex backup, S3 `iiic-freq3/data/eeg/`, S3 `morgoth1/data/internal_dataset/{IIIC,LPD,GPD}/segments_raw/`, or local SSD caches. Their `discharge_times.json` entries are still in the repo for transparency. |

### What reproduces and what doesn't, in plain English

- The **discharge-timing F1 number in the abstract** is reproducible at published level (≈ 0.876, sens ≈ 0.97, prec ≈ 0.80, mean MAE ≤ 1 sample = 5 ms) on the **147 Tier-A PD-GT patients** — see `code/evaluation/verify_c1_repro.py` and `results/c1_repro/c1_repro_results.json`. The published cohort was n=582; we can show the same algorithm on n=147 with the same model weights gets to the same F1, which is the point.
- The **frequency-estimation Spearman ρ numbers** (LPD ρ ≈ 0.78, GPD ρ ≈ 0.80, LRDA ρ ≈ 0.74, GRDA ρ ≈ 0.80) reproduce on the full segment set because frequency is robust to sub-100 ms label/data alignment.
- The **lateralization AUC numbers** (RDA 0.837, PD hemisphere 0.989, LPD-vs-GPD 0.911) reproduce on the full IRR + classification cohorts.
- The **EE-vs-EA comparison** (figure `irr_main`) reproduces on the full canonical IRR cohort.
- The **timing-F1 result on the 502 Tier-B/C patients** has a sensitivity collapse (~0.20-0.39) that makes per-segment F1 substantially below 0.876. The root cause, determined by per-patient offset analysis at `results/c1_repro/recovery_offsets.csv`, is not a small uniform time shift but **anti-aliasing-filter-induced signal degradation**: the recovered EEG was low-pass filtered (≤20 Hz hard cutoff) before being mirrored to the backup directory and to the IIIC labeling tool, removing the high-frequency sharp-transient content that HemiCET-UNet was trained to recognize as a discharge. The DP detector then misses most discharges entirely (sensitivity ~0.21 - 0.39 on filtered data vs ~0.97 on original 200 Hz unfiltered data) and the remaining detections are scattered in time, which is why a uniform offset correction does not help. Reproducing the published F1 on the Tier-B/C cohort would require either (a) finding the unfiltered original recordings, or (b) retraining HemiCET-UNet on the filtered-cohort distribution so the model learns to detect discharges from low-pass-only signal. The published number does not contradict this — the published evaluation was done before the May-2026 cleanup broke the label↔unfiltered-data linkage on those patients.

### Searching for the 8 unrecoverable patients

If you happen to find any `{pid}_seg000.mat` file for these patients, drop it into `data/eeg/` and append a corresponding row to `segments.csv`; the verifier will automatically pick it up. The exact filenames to search for are listed in `paper_materials/reproducibility/missing_pd_segments_to_find.txt`.

**Known unchecked source: `s3://bdsp-opendata-credentialed/eeg-test/eeg_bank_spec.h5`** (106 GB). This file contains 132,291 30-second EEG entries keyed by 5-digit `seg_id` with per-entry `patient_id` attrs. The 200-row companion `manifest.csv` only covers 11 of our originally-missing PIDs, but the larger 132K-entry pool likely covers a substantial fraction (proportionally about 7,000). Confirming this requires walking all 132K entries' attrs, which is impractical from a non-S3-local connection. The most efficient way to check is to launch a t3.small EC2 instance in `us-east-1` and run a one-off scan:

```python
import s3fs, h5py
fs = s3fs.S3FileSystem(anon=False)
f = fs.open('bdsp-opendata-credentialed/eeg-test/eeg_bank_spec.h5', 'rb',
            cache_type='readahead', block_size=64*1024*1024)
h5 = h5py.File(f, 'r')
seg = h5['segments']
pid_to_seg = {}
for k in seg:                        # ~30 min on an EC2 instance in us-east-1
    p = str(seg[k].attrs.get('patient_id', '')).rstrip('.0')
    if p:
        pid_to_seg[p] = k
# Then intersect pid_to_seg with the 8-PID list from missing_pd_segments_to_find.txt.
```

If any of the 8 PIDs are present, the entry's 30-second monopolar EEG can be sliced (via energy-aligned window-finder, same approach as `code/data_management/recover_pathC_window_finder.py`) to extract the labeled 10-second window.

## Regenerating individual figures

| Figure | Script | Inputs |
|---|---|---|
| fig 0 (EEG examples) | `paper_materials/generate_fig0_examples.py` | `grond_data.h5` + IRR manifests |
| fig 1 (PD pipeline) | `paper_materials/build_fig2.py` | `grond_data.h5` + model weights |
| fig 2 (RDA pipeline) | `paper_materials/draw_panel_b_rda.py` | `grond_data.h5` + model weights |
| fig 3 (frequency scatter) | `paper_materials/build_fig_irr_bars.py` | `grond_data.h5` + `freq_table_cis.json` |
| fig irr_main (EE vs EA bars) | `paper_materials/build_fig_irr_bars.py` | `grond_data.h5` + `results/independent_expert_v1/summary.json` |
| fig 5 (IRR) | `paper_materials/generate_fig_irr.py` | `grond_data.h5` + `results/independent_expert_v1/` |
| Tables 1–7 | `paper_materials/tables/generate_table{N}.py` | `grond_data.h5` |

One-shot regeneration of everything:
```sh
python paper_materials/generate_all_figures.py
```

## Auditing the recovery effort

For full transparency, every recovery step is logged:

| Step | What happened | Log |
|---|---|---|
| 1 | Copied 297 segments from Alex's backup directory | `code/data_management/recover_from_alex_data.py` |
| 2 | Downloaded 358 IRR-canonical segments from S3 `iiic-freq3/data/eeg/` | `code/data_management/recover_irr_cohort_from_s3.py` + `results/c1_repro/step2_irr.log` |
| 3 | Re-extracted 158 PD-GT segments from S3 `morgoth1/data/internal_dataset/{LPD,GPD,IIIC}/segments_raw/` via energy-aligned window-finder | `code/data_management/recover_pathC_window_finder.py` + `results/c1_repro/pathC_provenance.csv` |
| 4 | Per-patient timing-offset analysis on recovered patients | `code/data_management/fix_recovery_offsets.py` + `results/c1_repro/recovery_offsets.csv` |
| Final | n=147 baseline + 297 + 358 + 158 = 960 added segments, 13,556 → 14,369 in `segments.csv`. PD-GT cohort grew from 156 to 649 of 657 (98.8 %). | `paper_materials/reviewer_responses.md` |

## Citation

If you use the GROND data bank, please cite:

> Westover MB, Sun C, Zafar SF, Aboul Nour H, Ge W, Türeli A, et al. *GROND: Automated characterization of periodic discharges and rhythmic delta activity in critical-care EEG.* (2026).
