# Unified PD Model: Implementation Plan

## Vision

A single end-to-end model that jointly performs:
1. **Subtype classification** — LPD vs GPD (and eventually BIPDs)
2. **Frequency estimation** — Hz, derived from actual discharge timing
3. **Spatial localization** — which channels contain PDs
4. **Discharge timing** — precise times t_1, t_2, ... per involved channel

By solving all four tasks simultaneously with shared representations, each task reinforces the others. The model learns "what PDs look like" (spatial + temporal pattern) as a foundation, then reads off subtype, frequency, localization, and timing as downstream consequences.

## What We've Learned So Far

| Finding | Implication for the plan |
|---------|------------------------|
| CNN+Attention (Spearman 0.640) beats handcrafted features (0.604) | Raw waveform learning works — the unified model should operate on raw EEG |
| PD probability ⊥ frequency error | PD detection and frequency estimation use different signal aspects — both need explicit supervision |
| Bootstrapped discharge timing failed (ρ = -0.08) | **Clean timing labels are essential** — the model can't learn from noisy targets |
| Channel-level PD detection works well (0.989 patient AUC) | The spatial localization piece is already strong — needs review, not reinvention |
| Laterality index alone gets 0.920 AUC | Simple asymmetry features are powerful — the model should have access to both hemispheres |
| Temporal attention helps (+0.036 Spearman) | The model benefits from learning WHERE in time to focus |
| Feature engineering hit a ceiling (all within 0.003) | More features won't help — better labels will |

## The Label Improvement Pipeline

This is the foundation. Better labels → better model → better labels (virtuous cycle).

### Stage 1: Discharge Timing Labels

**Goal**: Accurate discharge time markers on every channel of every case.

**Step 1.1: Initial automated marking using hidden point process (HPP)**

We have an existing HPP approach that models periodic discharges as a point process with regular inter-event intervals. This provides a principled statistical estimate of discharge times, superior to simple pointiness peak detection.

Implementation:
- Script: `code/label_pipeline/hpp_discharge_marking.py`
- Input: all segments in `data/eeg/`, channel-level PD labels from our detector
- For each PD-positive channel: run HPP to estimate discharge times
- Output: `data/labels/discharge_times_hpp.json` — per-patient, per-segment, per-channel list of discharge times (in seconds)
- Also compute: estimated frequency (1/median IPI), regularity (CV of IPIs)

**Step 1.2: Build review viewer for MW first pass (correct/incorrect)**

Script: `code/label_pipeline/generate_timing_review_viewer.py`

HTML viewer showing:
- EEG trace (morgoth style, black on white)
- Red dots at HPP-estimated discharge times per channel
- Vertical sync lines connecting simultaneous discharges
- IPI bar showing inter-discharge intervals
- For each case: buttons **Correct** (keyboard C) / **Incorrect** (keyboard I)
- Auto-advance after marking
- Export CSV: `patient_id, segment_id, channel_idx, status (correct/incorrect)`
- Progress bar, filter (all/unmarked/correct/incorrect)

**Step 1.3: MW first pass review**

MW reviews all cases, marking each channel's timing as correct or incorrect.
- Correct cases → stored as training data immediately
- Incorrect cases → queued for second pass

Expected: many cases will be correct (HPP is more principled than pointiness peaks). The first pass is fast — just binary judgment, no editing.

**Step 1.4: Build correction viewer for MW second pass**

Script: `code/label_pipeline/generate_timing_correction_viewer.py`

For incorrect cases, an interactive HTML viewer where MW can:
- Click to add discharge markers
- Click existing markers to remove them
- Drag markers to adjust timing
- Keyboard shortcuts: A = add at cursor, D = delete nearest, arrow keys to navigate
- Show current IPI and frequency as markers are adjusted
- Export corrected times as JSON

**Step 1.5: MW second pass — correct the incorrect cases**

MW edits the timing markers on all cases marked incorrect in Step 1.3.

**Step 1.6: Merge and store final timing labels**

- Merge correct (pass 1) + corrected (pass 2) into canonical label file
- Store in `data/labels/discharge_times.json` with format:
```json
{
  "patient_id": {
    "segment_id": {
      "channel_idx": {
        "times": [0.45, 1.32, 2.18, ...],
        "source": "hpp_reviewed" | "manual_corrected",
        "reviewer": "mw"
      }
    }
  }
}
```

### Stage 2: Spatial Localization Labels

**Goal**: Accurate per-channel PD involvement labels for every case.

**Step 2.1: Run current best channel-level PD detector on all cases**

- Use CNN+Attention model (0.870 channel AUC)
- For each patient: predict PD probability per channel
- Threshold at 0.5 to get binary involved/not-involved
- Output: `data/labels/channel_pd_predictions.json`

**Step 2.2: Build spatial review viewer for MW first pass**

Script: `code/label_pipeline/generate_spatial_review_viewer.py`

HTML viewer showing:
- EEG with channels color-coded: green = predicted PD+, gray = predicted PD-
- Overall predicted laterality (left/right/bilateral) shown prominently
- Buttons: **Correct** (C) / **Incorrect** (I)
- Progress bar, filter

**Step 2.3: MW first pass — correct/incorrect**

Fast binary review. Correct cases stored as training data.

**Step 2.4: Build spatial correction viewer for MW second pass**

For incorrect cases:
- Click channels to toggle involved/not-involved
- Keyboard: channel number keys to toggle
- Show updated laterality as channels are toggled

**Step 2.5: MW second pass — correct the incorrect**

**Step 2.6: Merge into canonical labels**

Store in `data/labels/channel_involvement.json`:
```json
{
  "patient_id": {
    "segment_id": {
      "involved_channels": [0, 1, 2, 3, 8, 9, 10, 11],
      "source": "cnn_reviewed" | "manual_corrected",
      "reviewer": "mw"
    }
  }
}
```

### Stage 3: Frequency and Subtype Label Refinement

**Goal**: Clean up remaining label noise using the current best models.

**Step 3.1: Run current best models, identify disagreements**

- Frequency: CNN+Attention predictions vs gold standard (flag |error| > 0.3 Hz)
- Subtype: RF 300 predictions vs gold standard (flag disagreements)
- Generate a unified review set

**Step 3.2: Build combined review viewer**

Similar to existing `misclass_reviewer.html` but with the updated models and filtered to cases where model and label disagree.

**Step 3.3: MW review and correct**

Update the active columns in `patients.csv` (preserve `_original`).

## The Unified Model

### Architecture: Multi-Channel Spatial-Temporal Attention Network

```
Input: 18 channels × 2000 samples (10s at 200Hz)

┌─────────────────────────────────────────────────┐
│ Per-Channel Encoder (shared weights)             │
│   Conv1d(1→16, k=51, s=2) → BN → GELU          │
│   Conv1d(16→32, k=25, s=2) → BN → GELU         │
│   Conv1d(32→64, k=13, s=2) → BN → GELU         │
│   Conv1d(64→64, k=7, s=2) → BN → GELU          │
│   Output: (18, 64, 125) — 18 channels × 64 features × 125 time steps
└─────────────┬───────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────┐
│ Temporal Attention (per channel)                  │
│   Conv1d(64→1, k=1) → softmax → weights (18, 125)│
│   Weighted pool → channel embeddings (18, 64)     │
└─────────────┬───────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────┐
│ Spatial Attention (across channels)               │
│   MLP(64→32→1) per channel → softmax → (18,)     │
│   Weighted pool → patient embedding (64,)         │
└─────────────┬───────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────┐
│ Task Heads                                        │
│                                                   │
│ Head 1: Subtype                                   │
│   patient_embedding → Linear(64→1) → Sigmoid      │
│   Output: P(GPD) — binary classification          │
│                                                   │
│ Head 2: Frequency                                 │
│   patient_embedding → Linear(64→1)                │
│   Output: log(frequency in Hz)                    │
│                                                   │
│ Head 3: Channel Involvement (per channel)         │
│   channel_embeddings → Linear(64→1) → Sigmoid     │
│   Output: P(PD) per channel — 18 values           │
│                                                   │
│ Head 4: Discharge Timing (per channel, per time)  │
│   Decoder from encoder features (U-Net skip conn) │
│   Output: discharge probability (18, 2000)        │
│   Peak-pick → discharge times per channel         │
└─────────────────────────────────────────────────┘
```

### Multi-Task Loss

```
L = λ_sub * BCE(subtype_pred, subtype_label)
  + λ_freq * MSE(log_freq_pred, log_freq_label)
  + λ_chan * BCE(channel_pred, channel_label)    [per channel]
  + λ_time * BCE(timing_pred, timing_label)      [per sample, per channel, masked to involved channels]
```

Default weights: λ_sub=1.0, λ_freq=1.0, λ_chan=0.5, λ_time=0.5

The timing loss is only computed on channels marked as involved (masking out non-PD channels). The frequency loss uses the timing-derived frequency when timing labels are available, gold standard frequency otherwise.

### Training Strategy

**Phase A: Pre-train encoder + Heads 1-3 (no timing)**

Use existing labels (subtype, frequency, channel involvement). This is essentially a multi-channel version of our CNN+Attention that already works.
- 5-fold patient-stratified CV
- 30 epochs, batch 32 (18 channels per sample = larger memory)
- Validate on all 3 tasks

**Phase B: Add Head 4 (timing) with clean labels from Stage 1**

Freeze encoder (or fine-tune with low LR), train the decoder for discharge timing.
- Only on cases with reviewed timing labels
- Use timing-derived frequency as additional frequency supervision signal

**Phase C: End-to-end fine-tuning**

Unfreeze everything, train all 4 tasks jointly with reduced LR.
- The timing labels provide a richer gradient signal for the encoder
- The frequency head can now be supervised by BOTH gold standard freq AND timing-derived freq

### Why Joint Training Helps Each Task

| Task | Benefits from... |
|------|-----------------|
| **Subtype** | Spatial attention learns GPD=bilateral, LPD=lateralized from channel involvement labels |
| **Frequency** | Timing labels provide exact IPI → freq, removing ambiguity. PD-weighted channels focus on the right signal. |
| **Localization** | Timing labels on involved channels reinforce which channels have PDs. Subtype constrains spatial pattern. |
| **Timing** | Channel involvement masks out irrelevant channels. Frequency provides expected IPI range. Subtype informs expected morphology. |

### Inference Pipeline

For a new 10-second EEG segment:
1. Run all 18 channels through the encoder
2. Temporal attention → per-channel focus
3. Spatial attention → which channels matter
4. Head 1: LPD or GPD?
5. Head 3: which channels involved?
6. Head 4: discharge times on involved channels
7. Head 2: frequency (also derivable from Head 4 discharge times)

**BIPD detection** (future): if Head 3 shows independent left AND right involvement with different timing patterns (non-synchronized discharges), flag as BIPD candidate. This falls out naturally from the spatial + temporal structure — no new training data needed for initial detection, just a post-hoc rule on the model outputs.

## Implementation Sequence

### Sprint 1: Label Infrastructure (MW effort: ~2-3 days of reviewing)

| Step | Script to build | MW review effort | Output |
|------|----------------|-----------------|--------|
| 1.1 | hpp_discharge_marking.py | None (automated) | Initial timing marks |
| 1.2 | generate_timing_review_viewer.py | None (build tool) | Review viewer |
| 1.3 | — | **~4-6 hours** (fast binary) | Correct/incorrect marks |
| 1.4 | generate_timing_correction_viewer.py | None (build tool) | Correction viewer |
| 1.5 | — | **~4-8 hours** (editing) | Corrected timing labels |
| 1.6 | merge_timing_labels.py | None (automated) | Final timing labels |

### Sprint 2: Spatial Labels (MW effort: ~1-2 days)

| Step | Script to build | MW review effort | Output |
|------|----------------|-----------------|--------|
| 2.1 | predict_channel_involvement.py | None (automated) | Initial spatial predictions |
| 2.2 | generate_spatial_review_viewer.py | None (build tool) | Review viewer |
| 2.3 | — | **~2-3 hours** (fast binary) | Correct/incorrect marks |
| 2.4 | generate_spatial_correction_viewer.py | None (build tool) | Correction viewer |
| 2.5 | — | **~2-4 hours** (toggling) | Corrected spatial labels |
| 2.6 | merge_spatial_labels.py | None (automated) | Final spatial labels |

### Sprint 3: Frequency/Subtype Label Refinement (MW effort: ~half day)

| Step | Script to build | MW review effort | Output |
|------|----------------|-----------------|--------|
| 3.1 | identify_disagreements.py | None (automated) | Disagreement cases |
| 3.2 | generate_refinement_viewer.py | None (build tool) | Review viewer |
| 3.3 | — | **~2-3 hours** | Refined labels |

### Sprint 4: Unified Model (no MW effort — pure engineering)

| Step | What | Time estimate |
|------|------|--------------|
| 4.1 | Build multi-channel architecture | 1 session |
| 4.2 | Phase A training (heads 1-3) | ~2-3 hours training |
| 4.3 | Phase B training (add timing head) | ~2-3 hours training |
| 4.4 | Phase C end-to-end fine-tuning | ~2-3 hours training |
| 4.5 | Evaluation + comparison with current methods | 1 session |
| 4.6 | Build comprehensive results viewer | 1 session |

### Sprint 5: BIPD Extension (future)

| Step | What |
|------|------|
| 5.1 | Collect BIPD training cases |
| 5.2 | Add BIPD detection rule: independent L/R timing patterns |
| 5.3 | Fine-tune model with BIPD examples |

## Expected Outcomes

### With clean timing labels (Sprints 1-3):
- Discharge timing Spearman: -0.08 → **0.6+** (clean labels vs noisy bootstrap)
- Frequency from IPI: should match or exceed CNN+Attention (0.640) since it's derived from actual discharge times rather than learned abstraction
- Channel localization: 0.870 AUC → **0.92+** with reviewed labels

### With unified model (Sprint 4):
- Frequency: 0.640 → **0.68-0.72** (timing supervision provides exact IPI signal)
- Subtype: 0.931 AUC → **0.94+** (spatial attention learns bilateral/lateralized patterns)
- Localization: **0.95+** (reinforced by timing and subtype)
- Timing: **accurate** (trained on clean human-reviewed labels)

### With BIPD extension (Sprint 5):
- Novel capability: detect bilateral independent PDs from the spatial + temporal structure
- No existing automated method does this

## File Organization

```
code/label_pipeline/
├── hpp_discharge_marking.py              Step 1.1
├── generate_timing_review_viewer.py      Step 1.2
├── generate_timing_correction_viewer.py  Step 1.4
├── merge_timing_labels.py               Step 1.6
├── predict_channel_involvement.py        Step 2.1
├── generate_spatial_review_viewer.py     Step 2.2
├── generate_spatial_correction_viewer.py Step 2.4
├── merge_spatial_labels.py              Step 2.6
├── identify_disagreements.py            Step 3.1
└── generate_refinement_viewer.py        Step 3.2

code/unified_model/
├── model.py                  Multi-channel spatial-temporal attention network
├── dataset.py                Dataset loader (all 4 label types)
├── train_phase_a.py          Pre-train encoder + heads 1-3
├── train_phase_b.py          Add timing head
├── train_phase_c.py          End-to-end fine-tuning
├── evaluate.py               Comprehensive evaluation
└── inference.py              Inference pipeline

data/labels/
├── discharge_times.json      Per-channel discharge timing (Stage 1 output)
├── channel_involvement.json  Per-channel PD involvement (Stage 2 output)
├── patients.csv              Updated frequency/subtype (Stage 3 output)
├── segments.csv              Unchanged
└── annotations.csv           Updated with new annotations

results/
├── timing_review_viewer.html
├── timing_correction_viewer.html
├── spatial_review_viewer.html
├── spatial_correction_viewer.html
└── unified_model_results.html
```

## Critical Dependencies

1. **HPP discharge marking code** — needs to exist or be built. MW mentioned "hidden point process modeling that we did earlier." Need to locate this code or re-implement.
2. **Interactive marker editing in HTML** — the timing correction viewer (Step 1.4) is the most complex UI component. Click-to-add, drag-to-move, click-to-delete discharge markers on an EEG trace.
3. **Training time** — the unified 18-channel model with timing decoder will be large. May need to reduce to 3-fold CV or use a validation set instead of full CV for development.
4. **Memory** — 18 channels × 2000 samples × batch_size. May need gradient accumulation or smaller batches.

## Success Criteria

| Task | Current best | Target | Metric |
|------|-------------|--------|--------|
| Frequency | 0.640 | **0.70+** | Combined Spearman |
| Subtype | 0.931 | **0.95+** | AUC |
| Localization | 0.870 | **0.95+** | Channel AUC |
| Timing | -0.08 | **0.60+** | IPI-freq Spearman |
| BIPD detection | N/A | **feasible** | Proof of concept |
