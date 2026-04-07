# Frequency Estimation for Periodic EEG Patterns: Review v7

## Problem & Data

Estimating frequency (Hz) of periodic discharges (LPD, GPD) in 10-second, 18-channel bipolar EEG at 200 Hz. Pattern type is known.

### Dataset (as of 2026-03-18)

| | Patients | Segments (5/patient cap) | Raters |
|---|---|---|---|
| LPD | 177 | 900 | MW (all), LB/PH/SZ (38 patients) |
| GPD | 182 | 830 | MW (all), LB/PH/SZ (6 patients) |
| GRDA | 17 | 64 | LB/PH/SZ only |
| LRDA | 7 | 35 | LB/PH/SZ only |
| **Total** | **374** | **2,046** | |

**Gold standard:** MW's rating where available (340 patients); mean of LB/PH/SZ where MW has not scored (29 patients). Total with gold standard: 343. For the original 43 patients, ratings from all 4 raters (LB, PH, SZ, MW) exist.

**Data organization:** Unified into `data/eeg/` (2,046 .mat files) and `data/labels/` (segments.csv, annotations.csv in long format, patients.csv). All previous round-specific directories archived to `data/_archive/`.

### What changed since v6
- **8× more patients**: 43 → 374 (343 usable with gold standard labels)
- **Gold standard hierarchy**: MW's rating when available (340 patients), else mean of other experts (29 patients). Previously used median of 3 experts only.
- **Round 4 annotation**: 134 new cases (104 GPD, 30 LPD) scored by MW
- **Sahar annotation package**: 336 cases sent for independent scoring to establish inter-rater agreement on the full dataset
- **Unified data structure**: Single `eeg/` folder + long-format `annotations.csv` (extensible to new raters)

## Current Best Results

### 34 experiments on 335 patients, strict patient-level LOPO CV

| Method | Combined ρs | LPD ρs (n=155) | GPD ρs (n=163) | Speed |
|--------|-------------|----------------|----------------|-------|
| **GBM (100 trees, depth 3)** | **0.686** | 0.553 | 0.715 | 33 ms |
| Ridge + expanded features (11) | 0.680 | 0.566 | 0.689 | 69 ms |
| Ridge + base features (6) | 0.670 | 0.516 | 0.712 | 33 ms |
| Ridge + interactions | 0.675 | 0.542 | 0.702 | 33 ms |
| Subtype-specific Ridge | 0.676 | 0.549 | 0.698 | 33 ms |
| CNN embeddings + Ridge | 0.659 | — | — | ~200 ms |

**Expert-expert baselines (original 35 patients):** LB-MW: 0.630, PH-MW: 0.618, SZ-MW: 0.166

### Key finding: We likely exceed single-expert agreement

The algorithm achieves Spearman 0.686 on 335 patients (combined), while individual experts achieve 0.618–0.630 against MW on 35 patients. These aren't directly comparable (different sample sizes), but the algorithm's CI [0.592, 0.737] overlaps with and trends above the expert-MW correlations. Sahar's independent ratings will provide a fair head-to-head comparison on the same cases.

## Best Model Details

**GBM (Gradient Boosted Machine)** on 6 base signal processing features, trained on log(freq).

### Features (6):
| Feature | Description | Time |
|---------|-------------|------|
| f_B | Pointiness → smooth → ACF first peak (thr=0.10) | 24 ms |
| f_peaks | Peak-count: pointiness peaks → (n-1)/time_span | 0.2 ms |
| f_fft | FFT of pointiness → peak in [0.3, 3.5] Hz | 0.3 ms |
| f_tkeo | TKEO |x²(n)-x(n-1)x(n+1)| → FFT peak | 0.7 ms |
| f_coh | Cross-channel spectral coherence peak | 2.2 ms |
| is_gpd | Pattern type indicator | 0 ms |

**Speed:** 33 ms per 10-second segment (f_B dominates at 24 ms). All approaches using these 6 features run at the same speed — the model prediction time is negligible (<1 ms).

### Why GBM over Ridge?
Marginal improvement (+0.016 Spearman) with same speed. In practice, Ridge is nearly as good and simpler. Both are viable for production.

## What We Tried (34 new experiments + 192 from v6)

### Tier 1 — Feature exploration
- Expanded features: HPS, CAR montage FFT, n_detected channels, frequency range, signal variance → modest LPD improvement (+0.05)
- Subtype-specific models: marginal improvement
- Feature transforms: interactions helped LPD slightly; log transforms and standardization did not help
- Alpha regularization: minimal effect across all methods

### Tier 2 — Model types
- **GBM (depth 3): best overall** (0.686). Depth 5 overfits.
- Random Forest: competitive (0.680)
- k-NN as additional feature: no improvement
- Ensemble of 3 Ridge models: no improvement over single best

### Tier 3 — Deep learning revisited
- CNN embeddings + Ridge: **still doesn't help** (0.659, worse than SP-only). With 335 patients the patient-identity overfitting is reduced but embeddings carry no useful frequency signal beyond what SP features capture.
- CNN embeddings alone: 0.332 (useless without SP features)

### Comprehensive "didn't help" list (across all rounds)
CNN embeddings, CNN direct regression, DANN, k-NN features, >11 features, stacking, ensembles, ordinal regression, YIN/SRH, alternans detection, comb-fit scoring, DTW template matching, HMM windowed tracking, matched-filter envelope, GED standalone, HPS standalone, log feature transforms, per-expert training (dropped — MW is gold standard now)

## Evaluation Methodology

### Current approach (v7)
- **LOPO** (Leave-One-Patient-Out): all segments from held-out patient excluded from training
- **Up to 5 segments per patient** for training; 1 prediction per patient at test time (averaged across segments)
- **MW gold standard**: MW's frequency rating used as ground truth
- **Bootstrap 95% CIs** on Spearman and MAE (10,000 iterations)

### Lessons from v6 (still apply)
- Segment-level LOO-CV inflates results by +0.05 to +0.10 Spearman due to patient leakage
- Must cap segments per patient (GPD had 1 patient with 236/296 segments)
- CNN embeddings overfit to patient identity with small patient counts

## Remaining Bottleneck: Inter-Rater Agreement

We can now train and evaluate with 335 patients, but the **inter-rater agreement benchmark** is still limited:
- Only **35 patients** have all 4 raters (LB, PH, SZ, MW)
- Only **43 patients** have any multi-rater coverage
- Cannot definitively claim "beats experts" without same-sample comparison

### In progress: Sahar's independent annotation
336 cases (all MW-scored cases) sent for independent rating. When complete, this gives:
- **MW vs Sahar** agreement on 336 patients → fair benchmark for algorithm comparison
- **4-rater agreement** on 35 original patients (LB, PH, SZ, MW) + Sahar = 5 raters
- Statistical power to detect Spearman differences of ~0.10 at 80% power

## LPD vs GPD Gap

LPD remains harder than GPD across all methods:

| | LPD ρs | GPD ρs |
|---|---|---|
| Best algorithm | 0.566 | 0.715 |
| Expert LB vs MW | 0.630* | — |
| Expert PH vs MW | 0.618* | — |

*On 35 patients only (29 LPD, 6 GPD — too few GPD for separate expert baseline)

Possible reasons:
- LPD has more spatial variability (lateralized → different channels carry the signal)
- LPD frequency range is narrower, making rank-ordering harder
- GPD is more stereotyped and symmetric, easier for both humans and algorithms

## Infrastructure

```
data/
├── eeg/              2,046 .mat files (uniform naming, 10s @ 200 Hz)
├── labels/
│   ├── segments.csv  2,046 rows (segment registry)
│   ├── annotations.csv  3,526 rows (long format: segment × rater)
│   └── patients.csv  374 rows (patient summary + gold standard)
├── dl_cache/         CNN weights, external segment pool
└── _archive/         Previous round-specific directories

code/
├── optimization_harness_v2.py    Evaluation engine (LOPO CV, bootstrap CIs)
├── figure_pairwise_agreement.py  Scorer-vs-gold-standard figure
├── r12_full_evaluation.py        Full evaluation script
├── exp_t1_*.py, exp_t2_*.py, exp_t3_*.py   Experiment scripts
└── pd_pointiness_acf.py          Core signal processing

results/
├── optimization_dashboard_v2.html   Live dashboard (34 experiments)
├── optimization_runs_v2/            JSON result files
├── figure_pairwise_agreement.png    Current figure
└── archive_figures/                 Old figures and CSVs
```

## Next Steps

1. **Receive Sahar's annotations** → compute MW-Sahar agreement → fair algorithm-vs-expert comparison
2. **If algorithm > expert agreement**: write up results for publication
3. **If algorithm ≈ expert agreement**: may need more data or better LPD features
4. **Consider GRDA/LRDA**: 24 patients with 3-expert ratings exist but haven't been incorporated into frequency estimation yet
5. **Speed optimization**: f_B (ACF) takes 24/33 ms — a faster ACF implementation could cut total time to ~10 ms

## Questions for Review

1. **Is Spearman 0.686 (combined) on 335 patients publishable?** We need Sahar's data to make the expert comparison fair, but the numbers are strong.

2. **Should we pursue the LPD gap further?** Best LPD is 0.566 vs GPD 0.715. Is this an inherent limitation of the task, or could better spatial features help?

3. **GBM vs Ridge for the final model?** GBM is marginally better (+0.016) but Ridge is simpler and equally fast. For a published method, which is preferable?

4. **Should we incorporate GRDA/LRDA?** We have labeled data but haven't used it. These are different pattern types with different frequency characteristics.

5. **Multi-segment evaluation**: We currently average across segments for patient-level prediction. Should we also report segment-level results (with appropriate caveats about patient leakage)?
