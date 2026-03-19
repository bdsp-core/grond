# Frequency Estimation for Periodic EEG Patterns: Review v5

## Problem & Data
Estimating frequency (Hz) of periodic discharges (LPD, GPD) in 10-second, 18-channel bipolar EEG at 200 Hz. Pattern type is known. 556 annotated segments (260 LPD, 296 GPD) from **43 patients**, 3 expert raters. Critical patient imbalance: GPD has 1 patient (abn2147) with 236/296 segments (80%).

Additionally, ~10,000 PD segments with known class but no frequency labels exist (308 patients, external drive).

## Expert-Expert Agreement (target)

| | LPD Spearman | GPD Spearman | Mean | LPD MAE | GPD MAE |
|---|---|---|---|---|---|
| All 3 pairs pooled | **0.525** | **0.476** | **0.50** | 0.325 | 0.191 |

## Current Best Results — HONEST (patient-level 5-fold CV)

| Method | LPD rs | GPD rs | Mean rs | LPD MAE | GPD MAE | Evaluation |
|--------|--------|--------|---------|---------|---------|-----------|
| Expert-Expert | **0.525** | 0.476 | 0.50 | 0.325 | 0.191 | — |
| **SP features, patient-CV** | 0.385 | **0.516** | **0.428** | 0.381 | **0.196** | **Proper** |
| SP + CNN embeddings, patient-CV | 0.384 | 0.463 | 0.397 | 0.384 | 0.187 | Proper |
| Method A baseline | 0.234 | -0.145 | 0.045 | 0.537 | 0.274 | — |

**GPD: Beaten** (rs=0.516 vs 0.476) even with proper patient-level CV.
**LPD: 73% of expert level** (rs=0.385 vs 0.525). This is our main challenge.

## IMPORTANT: Evaluation Methodology Lesson

We discovered that **LOO-CV (leave-one-segment-out) dramatically overestimates performance** when multiple segments come from the same patient:

| Method | LOO-CV (inflated) | Patient-CV (honest) | Inflation |
|--------|-------------------|---------------------|-----------|
| SP features | 0.476 | 0.428 | +0.048 |
| SP + CNN embeddings | 0.492 | 0.397 | **+0.095** |

The CNN embeddings were the worst offender — they learned **patient identity**, not generalizable discharge features. When patients were properly separated, embeddings actually **hurt** performance (0.397 vs 0.428 without them).

**Root cause:** GPD patient abn2147 has 236/296 segments. In LOO-CV, the model trains on 235 of that patient's segments to predict the 236th — trivial memorization. In patient-CV, all 236 are held out together.

## Best Model Architecture (honest evaluation)

**Per-expert ridge regression on log(freq)**, patient-level 5-fold GroupKFold CV, alpha=1.0.

### Features (8 signal-processing, no CNN):

| Feature | Description |
|---------|-------------|
| f_B | Pointiness → smooth → ACF first peak (thr=0.10) |
| f_peaks | Peak-count: pointiness peaks → (n-1)/time_span |
| f_fft | FFT of pointiness → peak in [0.3, 3.5] Hz |
| f_tkeo | TKEO \|x²(n)-x(n-1)x(n+1)\| → FFT peak |
| f_coh | Cross-channel spectral coherence peak |
| is_gpd | Pattern type indicator |
| n_ch | Number of ACF-detected channels |
| placeholder | (envelope feature, currently zero) |

Per-expert training: train 3 ridge models (one targeting log(LB_freq), log(PH_freq), log(SZ_freq)), average the 3 predictions.

**Speed:** 48ms per 10-second segment (209× real-time). No GPU needed.

## What We Tried (8 rounds, 183 experiments)

### Rounds 1-5: Signal Processing (hit ceiling at ~0.42 LOO-CV)
- ACF, FFT, peak-count, HPS, matched-filter envelope, TKEO, spectral coherence
- Ridge on log-freq was the breakthrough (R3)
- TKEO and spectral coherence broke through the ceiling (R6)
- Per-expert training added +0.02 (R7)
- 25 additional features, stacking, ordinal regression, etc. — all hit the same ceiling
- **Conclusion: bottleneck is discharge recognition, not frequency estimation**

### Round 6: New Feature Traces (TKEO, HPS, phase coherence)
- TKEO was the most valuable new feature
- Combined Spearman ~0.45 (LOO-CV)

### Round 7: Multi-montage, per-expert training, GED, HMM
- Per-expert training was the key win
- Multi-montage (CAR, Laplacian) added marginal value
- Event-vs-background GED: poor standalone, marginal as ridge feature
- HMM comb-fit: too slow, poor results
- DTW template matching: too slow, couldn't complete

### Round 8: Deep Learning
- Pretrained CNN backbone on 3,816 LPD/GPD classification segments (308 patients) → **93.6% accuracy**
- Fine-tuned for frequency with eventness + frequency heads → **failed** (Spearman 0.017 end-to-end)
- CNN backbone embeddings (128-dim → 20 PCA) as ridge features → **appeared to work brilliantly** (LOO-CV Spearman 0.492)
- **BUT: patient-level CV revealed this was overfitting** — embeddings learned patient identity, not discharge features
- With proper patient-CV, embeddings hurt (0.397 vs 0.428 without them)

### What Didn't Help (comprehensive list)
- More features beyond ~8 (overfitting with 43 patients)
- RF, GBM, stacking (ridge is optimal for small data)
- Separate LPD/GPD models (smaller training sets)
- Period prediction (1/f), sqrt-freq, ordinal regression
- 25 morphological/temporal/spatial features
- More templates (8→50, PCA, from 10K segments)
- YIN/CMNDF, SRH, alternans detection, comb-fit
- GED spatial filtering, multi-montage FFT
- CNN end-to-end frequency regression (556 samples too few)
- CNN embeddings as features (patient leakage)
- Parameter grid search (432 combos — model > parameter mismatch)

## The Remaining LPD Gap

**LPD: rs=0.385 vs expert 0.525 (73% of expert level) with proper patient-CV**

### Why LPD is hard:
1. **Only 43 patients total, ~35 LPD patients** — too few for learning generalizable features
2. **Polyphasic/triphasic LPD complexes** — our feature traces (pointiness, TKEO, FFT) count multiple peaks per discharge complex
3. **Amplitude alternation** — strong/weak/strong pattern causes ACF/FFT to lock onto subharmonics
4. **Irregular intervals** — 10 seconds may have only 5-10 cycles with significant jitter
5. **Patient-specific morphology** — discharge shapes vary across patients, making cross-patient generalization hard

### What we haven't been able to try properly:
1. **Optimized DTW template matching** (timed out in R7 — needs efficient implementation)
2. **Learned eventness filter** (CNN approach failed due to insufficient labeled data)
3. **Self-supervised pretraining** on 10K segments → fine-tune (tried classification pretraining, but embeddings overfit to patient identity)
4. **Larger labeled dataset** — the 43-patient evaluation set is the fundamental bottleneck

## Available Untapped Resources

| Data | Count | Patients | Has freq? |
|------|-------|----------|-----------|
| Frequency-annotated | 556 segments | **43** | Yes (3 experts) |
| Class-labeled (external) | ~10,000 segments | **308** | No |
| Large template bank | 50+30 templates | Built from 308 patients | N/A |
| Pretrained CNN backbone | 93.6% accuracy | Trained on 308 patients | N/A |

**Key constraint:** The 43-patient evaluation set limits what we can learn from the data. Even our best signal-processing features can only be combined so well when the ridge model sees ~35 LPD patients.

## Questions for Your Advice

1. **Is the patient-level evaluation showing the true picture?** With only 43 patients (and 1 GPD patient dominating), the 5-fold patient-CV has high variance. Should we do leave-one-patient-out instead? Or bootstrap CIs?

2. **How do we get more LPD patients annotated efficiently?** We have 308 patients in the external dataset. Active learning (annotate the 50 hardest cases) vs. random sampling? What's the minimum number of new patients that would meaningfully help?

3. **Is there a way to use the 10K unlabeled segments that doesn't overfit to patient identity?** Our CNN pretraining learned patient-specific features. Would domain-adversarial training (maximize classification accuracy while minimizing patient-identification accuracy) help?

4. **Should we try a fundamentally different model?** E.g., instead of ridge regression on extracted features, use a nearest-neighbor approach: for a new segment, find the most similar segments in the training set (by embedding distance) from DIFFERENT patients, and use their frequencies.

5. **Is the GPD result (rs=0.516 vs expert 0.476) trustworthy?** With 80% of GPD from one patient, even patient-CV may be unstable. Should we exclude abn2147 and report GPD results on the remaining 60 segments from 5 patients?

6. **Given that GPD is beaten and LPD is at 73% — what's the right framing?** Is this publishable? What would make the remaining LPD gap convincing rather than a weakness?

## Technical Summary
```
Model: Per-expert ridge on log(freq), patient-level 5-fold GroupKFold, alpha=1.0
Features: 8 signal-processing (ACF, peak-count, FFT, TKEO, coherence, type, n_ch)
Training: 556 segments from 43 patients (260 LPD, 296 GPD)
Speed: 48ms per segment (209× real-time, CPU only)

Results (PROPER patient-level CV):
  GPD: Spearman=0.516 (vs expert 0.476) — BEATEN
  LPD: Spearman=0.385 (vs expert 0.525) — 73%
  Combined: 0.428 (vs expert 0.50) — 86%
```
