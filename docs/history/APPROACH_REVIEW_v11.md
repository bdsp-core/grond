# Frequency Estimation for Periodic EEG Patterns: Review v11

## Critical Methodological Principle

**No method may use gold standard labels as input.** All algorithms operate from raw EEG only. Previous evaluations inflated HPP timing F1 from ~0.65 to ~0.77 by passing gold standard frequency as input.

## Problem & Data

Comprehensive characterization of periodic discharges (LPD, GPD) and rhythmic delta activity (LRDA, GRDA) in 10-second, 18-channel bipolar EEG at 200 Hz.

### Dataset (as of 2026-03-21)

| | Patients | Labels |
|---|---|---|
| LPD | 450 | Freq, timing (complete), spatial (partial), laterality |
| GPD | 169 | Freq, timing (complete), spatial (partial) |
| LRDA | 100 + 70 harvested | Subtype |
| GRDA | 120 + harvesting | Subtype |
| BIPD | 100 harvested | Subtype (new) |
| Other (controls) | 20+ harvesting | True negatives |
| Hi-freq LPD (>2.5 Hz) | 292 harvested | Awaiting annotation |

**Discharge timing labels**: 576 LPD+GPD patients with MW-reviewed discharge times, refined through CET-UNet comparison. 184 cases (32%) had labels updated based on CET-UNet review — labels are no longer biased toward pointiness peaks. This is the definitive gold standard.

### What changed since v10
- **CET-UNet (CNN Evidence Trace with U-Net)**: Produces frame-level discharge evidence from raw EEG, trained on MW timing labels with sharp targets (σ=10ms), skip connections, and auxiliary HPP floor loss
- **CET-UNet labels are more accurate**: MW accepted CET-UNet timing in 18% of cases, edited 14%, kept original 68%. 184 total label updates.
- **max(HPP, CET) evidence combination**: Taking element-wise maximum of handcrafted and CNN evidence captures discharges either method finds. This is the key innovation.
- **Fair evaluation framework**: All methods tested EEG-only. No gold standard frequency as input.
- **Parameter optimization in fair setting**: HPP DP parameters re-optimized for noisy (CNN-estimated) frequency prior: α=1.275, λ=0.05, β=0.3
- **MPS GPU acceleration**: ~15× speedup for CNN training
- **Data harvesting**: BIPDs (100), LRDA/GRDA (~70), Other controls (~20), hi-freq LPDs (292)
- **Unified model attempted then abandoned**: Separate specialized models outperform

## Best System: max(HPP, CET-UNet) + CNN Freq + HPP DP

### Architecture Overview

```
EEG (18ch × 2000) ──┬── CNN+Attention (freq est) ──→ f_est
                     │
                     ├── Handcrafted evidence ──→ E_hpp(t) ──┐
                     │   (pointiness + TKEO)                  │
                     │                                        ├── max() ──→ E(t) ──→ HPP DP ──→ discharge times
                     └── CET-UNet evidence ──→ E_cet(t) ──────┘
                         (learned, U-Net)
```

1. **Frequency estimation**: CNN+Attention model estimates f_est from raw EEG (PD-weighted across channels)
2. **Dual evidence**: handcrafted (pointiness+TKEO) AND learned (CET-UNet) evidence traces
3. **max() combination**: takes the best of both at each time point
4. **HPP DP inference**: dynamic programming with approximately-periodic prior, using f_est as expected period
5. **EM template refinement**: case-specific waveform template cross-correlation

See detailed architecture description in the companion document.

## Current Results (Updated Gold Standard, EEG-Only)

### HPP-only (handcrafted evidence, gold freq — REFERENCE)

| Metric | Value |
|--------|-------|
| Sensitivity | 0.693 |
| Precision | 0.834 |
| F1 | 0.757 |
| Freq ρ (algo IPI vs MW IPI) | 0.956 |
| Freq ρ (algo IPI vs gold standard) | 0.965 |
| MW IPI vs gold standard | 0.965 |
| Timing accuracy | 5.9 ms |

Note: This uses gold standard frequency as input (reference only).

### Fair EEG-Only Methods (updated gold standard, 593 cases)

| Method | Evidence | Freq Source | DP Params | Sens | Prec | **F1** | **Freq ρ** |
|--------|----------|-------------|-----------|------|------|--------|-----------|
| **max(HPP,CET)+CNN freq+opt** | Combined | CNN | Optimized | 0.774 | 0.675 | **0.721** | **0.753** |
| HPP + CNN freq | Handcrafted | CNN | Default | 0.578 | 0.728 | 0.645 | 0.716 |
| HPP + bootstrap | Handcrafted | ACF→IPI | Default | 0.615 | 0.658 | 0.636 | 0.453 |
| CET + bootstrap | CET-UNet | ACF→IPI | Default | 0.499 | 0.746 | 0.598 | 0.531 |
| CET + CNN freq | CET-UNet | CNN | Default | 0.451 | 0.735 | 0.559 | 0.742 |

Optimized DP parameters: α=1.275, λ=0.05, β=0.3 (from parameter sweep).

### Frequency Estimation Comparison (EEG-Only)

| Method | Spearman | MAE |
|--------|----------|-----|
| FFT peak (Alexandra's baseline) | 0.353 | 0.561 |
| CNN+Attention direct | **0.744** | 0.266 |
| IPI from HPP+CNN_freq timing | 0.688 | 0.262 |

### Other Tasks

| Task | Method | Performance | Input |
|------|--------|-------------|-------|
| Subtype (LPD vs GPD) | RF 300 | AUC 0.931 | EEG only |
| Laterality (L vs R) | GBM balanced | AUC 0.957 | EEG only |
| Channel PD detection | CNN+Attention | AUC 0.870 | EEG only |
| Channel RDA detection | Pseudolabels | AUC 0.842 | EEG only |

## Key Findings

### 1. max(HPP, CET) evidence combination is the key innovation

Neither handcrafted nor CNN evidence alone is best. The max combination captures:
- Sharp discharges (pointiness excels)
- Broad/subtle discharges (CNN excels)
- F1 improved from 0.653 → 0.706 (+0.053)

### 2. Labels were biased toward pointiness peaks

MW's full CET-UNet review updated 184/576 cases (32%). The CET-UNet found more accurate discharge locations in many cases — previous F1 metrics penalized the CNN unfairly.

### 3. CNN frequency estimation is the bottleneck

The gap between gold-freq HPP (F1=0.757) and CNN-freq HPP+CET (F1=0.706) is primarily due to imperfect frequency estimation (ρ=0.744 vs 1.0). Improving frequency estimation is the single highest-leverage improvement.

### 4. IPI-derived frequency loses information vs direct CNN frequency

CNN direct frequency (ρ=0.744) > IPI from HPP timing (ρ=0.688). The timing→IPI→frequency conversion loses accuracy because the timing detection isn't perfect.

### 5. Loose periodic prior works better with noisy frequency

Optimized α=1.275 (vs 3.0 default) because CNN frequency estimates are imperfect. The DP needs more flexibility.

### 6. Parameter optimization matters hugely

Sweeping evidence types + DP parameters yielded +0.053 F1 improvement. The sweep tested max/mean/weighted evidence combinations and alpha/lambda/beta/peak threshold.

## Method Names

| Abbreviation | Full Name | What it does |
|-------------|-----------|-------------|
| **HPP** | Hidden Point Process | MAP inference via DP for discharge timing |
| **CET** | CNN Evidence Trace | Learned per-channel discharge evidence (U-Net) |
| **NVO** | Narrowband Variance Optimization | Sinusoidal fitting for RDA frequency (TODO) |
| **SPF** | Signal Processing Features | Handcrafted features (pointiness, ACF, FFT, TKEO) |

## Nine Tasks (Paper Roadmap Status)

| # | Task | Status | Best Method | Performance |
|---|------|--------|-------------|-------------|
| 1 | LPD vs GPD | **Done** | RF 300 | AUC 0.931 |
| 2 | LRDA vs GRDA | TODO | — | — |
| 3 | PD channel ID | Partial (304 GT) | CNN+Attention | AUC 0.870 |
| 4 | RDA channel ID | Pseudolabels only | CNN | AUC 0.842 |
| 5 | PD discharge timing | **Done** | max(HPP,CET)+CNN_freq+DP | F1 0.706 |
| 6 | RDA wave timing | TODO | — | — |
| 7 | RDA frequency | TODO | FFT baseline | ρ 0.840 (23 pts) |
| 8 | PD frequency | **Done** | CNN+Attention direct | ρ 0.744 |
| 9 | BIPD analysis | Waiting for data | — | — |

## Next Steps

1. **Complete fair evaluation** with updated gold standard labels
2. **RDA tasks**: NVO implementation, HPP adaptation for RDA waves, LRDA laterality annotation
3. **Integrate harvested data**: 292 hi-freq LPDs, 100 BIPDs, 70+ LRDA, Other controls
4. **Improve frequency estimation**: this is the biggest lever for improving timing
5. **Generate paper figures**: see PAPER_ROADMAP.md
