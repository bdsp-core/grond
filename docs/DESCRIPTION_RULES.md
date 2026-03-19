# Verbal Description Rules for EEG Pattern Labels

These rules define how to generate a concise verbal description from the model outputs
for LRDA, GRDA, LPD, and GPD patterns.

**Templates:**
```
LRDA / LPD:   {type} at {X} Hz, {laterality}; maximal in {top regions}.
GRDA / GPD:   {type} at {X} Hz, {regional predominance}.
```

---

## Rule 1 — Event type

Use the acronym exactly as output by the model. Do not spell it out.

| Model output | Use in description |
|--------------|---------------------|
| LRDA | LRDA |
| GRDA | GRDA |
| LPD  | LPD  |
| GPD  | GPD  |

The model type label is **never overridden** by other outputs (e.g., a high laterality
index does not change GRDA to LRDA).

---

## Rule 2 — Frequency

Report the median detected frequency rounded to **1 decimal place**. No band name.

Example: `at 2.1 Hz`

---

## Rule 3 — Laterality (LRDA / LPD only)

Derived from `laterality_index = (right_mean_score − left_mean_score) / (right_mean_score + left_mean_score)`.

The left/right mean scores are computed as **equal-weighted means over the 4 anatomical
regions per side** (LF, LT, LCP, LO for left; RF, RT, RCP, RO for right), so that
frontal regions (4 channels) do not dominate over occipital/centro-parietal (1 channel).

Ranges from −1 (fully left) to +1 (fully right).

**For LRDA / LPD** (per ACNS 2021 nomenclature, L = Lateralized):

| Laterality index | Label |
|------------------|-------|
| < −0.15          | unilateral left |
| −0.15 to −0.10   | bilateral asymmetric, left-predominant |
| −0.10 to +0.10   | bilateral/symmetric |
| +0.10 to +0.15   | bilateral asymmetric, right-predominant |
| > +0.15          | unilateral right |

ACNS 2021 definitions (qualitative, Section C1b, p.9–10):
- **Unilateral**: pattern seen in only one hemisphere
- **Bilateral asymmetric**: seen bilaterally but clearly and consistently (>80% of the time) higher amplitude in one hemisphere
- No quantitative amplitude ratio or LI threshold is given; the ±0.15 cutoff is calibrated to clinical judgment.

**For GRDA / GPD:** Laterality is not reported. See Rule 5.

---

## Rule 4 — Spatial extent

Not included in the verbal description. Spatial extent is available as a numeric field
but is not reported in the verbal summary.

---

## Rule 5 — Regional description

**For LRDA / LPD:**

Report top 1–2 active regions on the **dominant side only** (side already stated in
the laterality term). Use bare names without the side prefix.

- Threshold for "active": mean region score > **2.0**
- Use bare names: "frontal", "temporal", "centro-parietal", "occipital"
- If no qualifying region: "no region clearly dominant"

Example: `maximal in the centro-parietal and temporal regions`

**For GRDA / GPD** (ACNS 2021, Section C1b):

Report regional predominance using one of three ACNS-specified categories:

| Predominance group | Regions used |
|--------------------|--------------|
| frontally predominant | mean(LF, RF) |
| occipitally predominant | mean(LO, RO) |
| midline predominant | mean(Fz-Cz, Cz-Pz channel scores) |

Pick the group with the highest mean score. If that score is ≤ 2.0 (not above the
active threshold), use **"no regional predominance"** instead.

---

## Rule 6 — Modifiers (+F, +S, etc.)

Deferred. No method currently exists to detect fast activity or sharp wave morphology.
Do not add modifiers until a detection method is implemented.

---

## Examples

```
LRDA at 2.1 Hz, unilateral left; maximal in the centro-parietal and temporal regions.

LRDA at 2.1 Hz, bilateral asymmetric, left-predominant; maximal in the centro-parietal regions.

LPD at 1.2 Hz, unilateral right; maximal in the temporal region.

GRDA at 1.8 Hz, frontally predominant.

GRDA at 2.3 Hz, no regional predominance.

GPD at 1.5 Hz, occipitally predominant.
```
