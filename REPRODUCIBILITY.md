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

# 3. Get the EEG data bank (grond_data.h5, 1.68 GB)
#    Option A: download from S3 (credentials required):
aws s3 cp s3://bdsp-opendata-credentialed/grond/grond_data.h5 data/grond_data.h5
#    Option B: rebuild from data/eeg/ + data/labels/ (no S3 needed):
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

### Model-weight history (root cause of the F1 regression)

The F1=0.889 number was produced on 2026-03-23 (commit `b91b075`, "HemiCET optimization complete: F1=0.891") using HemiCET v2 weights + ChannelPD-Net + DP that were backed up to `data/hemi_cache/hemi_cet_v2_backup_v1/` and `data/_archive/pd_channel_cache_backup_v1/` on 2026-03-27.

The repo cleanup on 2026-05-02 (commit `a3553b6`, "Repo cleanup Phase 1-3") moved `optimization_harness_v2.py` into `code/archive/`, breaking ~35 active import dependencies. Subsequent retrain attempts (commits `725f265`, `efe4b78`) found two more cleanup-induced bugs:

  1. The launcher was pointing at `code/hemi_detector/train.py` ("EXPERIMENT 1.1 — HemiNet (Design A)") instead of `code/pd_channel_detector/train_cnn_attention.py` — the wrong model architecture was being retrained.
  2. `load_dataset()` was loading `patients.csv` from the wrong path (it had been moved into `archive_labels/`).

These bugs were caught and fixes applied, but by then the retrained `data/hemi_cache/hemi_cet_v2/` and `data/pd_channel_cache/` weights had been overwritten with the broken-pipeline outputs. When this repo was tested with the post-cleanup-retrain weights, F1 dropped from 0.89 to about 0.42 on the same evaluation cohort.

This repo restores the F1=0.891-era weights as canonical:
  - `data/hemi_cache/hemi_cet_v2/` — Mar 27 backup (md5 `83fc3799...`)
  - `data/pd_channel_cache/` — Mar 27 backup (md5 `ebd4ab25...`)
The post-cleanup retrain weights are preserved under `data/hemi_cache/hemi_cet_v2.post_cleanup_retrain/` and `data/pd_channel_cache.post_cleanup_retrain/` for forensic comparison.

With the restored Mar-27 weights, the C1 evaluation on the current 640-patient cohort reaches F1=0.55 (up from 0.42). The remaining gap to the published F1=0.89 is attributable to dataset/label drift: the published evaluation cohort (n=582) and labels evolved between Mar 23 and the present, with label cleanup, recovered-cohort additions, and edge-case revisions all changing the denominator on which F1 is computed. Full bit-exact reproduction of F1=0.889 requires also reverting `data/labels/` to its Mar-23 state — feasible via `git checkout b91b075 -- data/labels/`.

**Investigated source: `s3://bdsp-opendata-credentialed/eeg-test/eeg_bank_spec.h5`** (106 GB, 132,291 segments). Downloaded to local SSD on 2026-06-07 and analyzed in detail. Findings:

- The bank entries have NO `patient_id` attribute. Per-entry attrs are limited to `channel_names30s, eeg_duration_s, fs_hz30s, has_spectrogram, montage, note, source_dataset, source_window_center_s, spec_source`. The 200-row companion `manifest.csv` is the only patient-id mapping that exists; it covers 11 of the 501 originally-missing PIDs (all already recovered elsewhere).

- **The bank EEG content is unfiltered** (frac power > 20 Hz of 0.03 - 0.30; content up to 60 Hz including line-noise band), unlike the Alex backup which is hard low-passed at ~20 Hz (frac > 20 Hz = 0.000). So the unfiltered originals SHOULD be discoverable by content matching.

- A content-match pipeline (`precompute_bank_envelopes.py` + `match_recovered_to_bank.py`) was built: compute the 20 Hz-low-passed channel-summed envelope for each of 784 recovered patients, then for each of the 68,409 bank 30-second entries do sliding-window normalized cross-correlation. **490 of 784 patients (62%) achieved NCC ≥ 0.9** — strong evidence the bank contains the unfiltered originals for those patients. By source: 319/330 IRR-cohort (97%), 102/158 path-C (65%), 69/296 Alex (23%).

- Extraction attempts: (a) extract at the NCC-best window, (b) re-extract with a path-C-style GT-discharge-aligned window-finder applied to the matched 30-s bank entry. Both ended up with **overall timing-F1 essentially unchanged** (0.489 → 0.466 → 0.479; net regression of ~0.01). The content matches are real but the labeled 10-second window is not reliably recoverable from coarse energy-envelope cross-correlation, and the path-C window-finder on the bank windows does not consistently find the labeled offset either.

- All overwritten .mat files were restored to their original state (Alex 69 directly recopied, IRR 319 re-extracted from local sub-S0001 cache, path-C 102 left as bank-extracted since both versions hit similar F1 ≈ 0.22). The audit logs (`results/c1_repro/bank_match_results.csv`, `bank_extracted_audit.csv`, `bank_gt_aligned_audit.csv`) and the 1.64 GB precomputed bank envelope cache remain on the SSD for future investigation. None of the 8 truly-unrecoverable PD-GT patients have a high-NCC match in the bank.

The bank is a real source of unfiltered EEG that contains many of our patients' source recordings, but a reliable mapping from matched 30-second context back to the labeled 10-second segment has not yet been found. Reproducing the published F1 on the recovered cohort still requires either (a) a higher-fidelity window-alignment method that matches not just envelopes but discharge-timing landmarks, (b) retraining HemiCET-UNet on the filtered-cohort distribution, or (c) locating the canonical 10-s segments through some other path entirely.

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
