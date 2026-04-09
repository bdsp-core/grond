# Independent expert annotation tasks

This folder defines four independent annotation tasks designed to address
**Reviewer Note #1** (annotation circularity) and **Reviewer Note #2**
(PD frequency gap) in the manuscript: a set of expert raters who were
**not** involved in algorithm development each score a held-out subset of
segments, so that we can compute expert--expert and expert--algorithm
agreement *without* using any MW labels as ground truth.

## Layout

```
independent_expert_tasks/
├── README.md                  ← this file (orientation for MW)
├── select_cases.py            ← case-selection driver (one script for all 4 tasks)
├── lpd/
│   ├── README.md              ← share this with the LPD rater
│   └── manifest.csv           ← 200 selected LPD segments
├── gpd/
│   ├── README.md              ← share this with the GPD rater
│   └── manifest.csv
├── lrda/
│   ├── README.md              ← share this with the LRDA rater
│   └── manifest.csv
└── grda/
    ├── README.md              ← share this with the GRDA rater
    └── manifest.csv
```

## What each task asks the rater to provide

| Subtype | Lateralization | Frequency | Discharge timing |
|---|:---:|:---:|:---:|
| LPD  | ✅ left vs right          | ✅ | ✅ |
| GPD  | (bilateral by definition) | ✅ | ✅ |
| LRDA | ✅ left vs right          | ✅ | — |
| GRDA | (generalized by definition) | ✅ | — |

## Recommended viewers

Two existing viewers in `code/generators/labeling/` cover all four tasks:

| Task | Viewer entry-point | Captures |
|---|---|---|
| LPD  | [generate_pd_freq_timing_labeler.py](../../code/generators/labeling/generate_pd_freq_timing_labeler.py) `--subtype lpd`  | laterality + freq + timing |
| GPD  | [generate_pd_freq_timing_labeler.py](../../code/generators/labeling/generate_pd_freq_timing_labeler.py) `--subtype gpd`  | freq + timing (laterality fixed = bilateral) |
| LRDA | [generate_rda_freq_labeler.py](../../code/generators/labeling/generate_rda_freq_labeler.py) `--subtype lrda` | laterality + freq |
| GRDA | [generate_rda_freq_labeler.py](../../code/generators/labeling/generate_rda_freq_labeler.py) `--subtype grda` | freq (laterality is generalized by definition) |

Both viewers were extended to accept a `--manifest` flag (consuming our
pre-curated 200-segment list) and an `--output` flag (writing the HTML
to a specific path). The RDA viewer was also extended with an
optional laterality (left/right) input that activates only in
`--subtype lrda` mode.

> **Why not `generate_lrda_labeler.py`?** That viewer was originally
> built for an LRDA wave-morphology labeling pass and references a
> `segment_id` column that no longer exists in `data/labels/segments.csv`
> after a schema cleanup, so it currently cannot run. Rather than
> resurrecting it, we extended `generate_rda_freq_labeler.py` (which is
> on the canonical schema) with the small left/right input we need for
> LRDA. The wave-triplet UI from the old viewer is not part of this
> task — it would have been extra work for the colleagues that we did
> not need.

The viewers each generate a **self-contained HTML file** with the EEG data,
the algorithm's pre-filled defaults, and the labeling UI all inlined. The rater
opens the HTML in any modern browser, clicks through the cases, and clicks
"Export" to download a JSON file with all their labels.

## Case selection

Run [`select_cases.py`](select_cases.py) to (re)generate the four manifests:

```bash
conda run -n morgoth python paper_materials/independent_expert_tasks/select_cases.py
```

Selection rules:

- 200 segments per subtype, one per unique patient.
- Stratified into 0.25-Hz bins across [0.5, 3.0) Hz.
  - PDs use MW's `expert_freq_hz` for binning (the only PD-frequency column we have).
  - RDAs use `algo_freq_hz` from the RDA-Profiler.
  - The bin assignment is **only** for stratification; the viewer's pre-filled
    default at run time still comes from the PDProfiler / RDA-Profiler at run
    time, so the new raters never see MW's labels.
- Quality preference: cases with valid `.mat` files on disk, then highest IIIC
  plurality fraction, then most IIIC votes. Deterministic seed = 42.

Distribution after running with defaults:

| Subtype | 0.5–0.75 | 0.75–1.0 | 1.0–1.25 | 1.25–1.5 | 1.5–1.75 | 1.75–2.0 | 2.0–2.25 | 2.25–2.5 | 2.5–2.75 | 2.75–3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LPD  | 27 | 27 | 27 | 27 | 27 | 26 | 26 | 10 | 2 | 1 |
| GPD  | 22 | 23 | 23 | 23 | 23 | 23 | 22 | 22 | 15 | 4 |
| LRDA |  2 | 38 | 39 | 39 | 38 | 38 |  5 |  1 | 0 | 0 |
| GRDA | 10 | 32 | 33 | 33 | 32 | 32 | 22 |  5 | 1 | 0 |

The PD distributions are close to uniform across the full range; the RDA
distributions are concentrated in 0.75–2.0 Hz because that is where the
underlying biology lives — there are essentially no >2 Hz LRDA segments
in the dataset, so the upper bins cannot be filled.

## Generating the per-task HTML files

Both viewers now accept `--manifest`, `--output`, and `--no-open`. From
the repo root, with the `morgoth` conda env:

```bash
# LPD
conda run -n morgoth python code/generators/labeling/generate_pd_freq_timing_labeler.py \
    --subtype lpd \
    --manifest paper_materials/independent_expert_tasks/lpd/manifest.csv \
    --output  paper_materials/independent_expert_tasks/lpd/lpd_task.html \
    --no-open

# GPD
conda run -n morgoth python code/generators/labeling/generate_pd_freq_timing_labeler.py \
    --subtype gpd \
    --manifest paper_materials/independent_expert_tasks/gpd/manifest.csv \
    --output  paper_materials/independent_expert_tasks/gpd/gpd_task.html \
    --no-open

# LRDA  (with left/right laterality buttons enabled)
conda run -n morgoth python code/generators/labeling/generate_rda_freq_labeler.py \
    --subtype lrda \
    --manifest paper_materials/independent_expert_tasks/lrda/manifest.csv \
    --output  paper_materials/independent_expert_tasks/lrda/lrda_task.html \
    --no-open

# GRDA  (laterality input hidden; freq only)
conda run -n morgoth python code/generators/labeling/generate_rda_freq_labeler.py \
    --subtype grda \
    --manifest paper_materials/independent_expert_tasks/grda/manifest.csv \
    --output  paper_materials/independent_expert_tasks/grda/grda_task.html \
    --no-open
```

Note the output HTMLs are gitignored (added to `.gitignore`) because
they are tens to hundreds of MB each — see the next section.

## Distribution to collaborators

The generated HTML files are tens to hundreds of MB each (200 segments
worth of inlined EEG + per-frequency precomputes). Measured sizes for
the 200-case generation:

| Task | size for 200 cases |
|---|---|
| LPD  | 142 MB |
| GPD  | 142 MB |
| LRDA |  82 MB |
| GRDA |  82 MB |
| **Total** | **448 MB** |

Too large to commit to git directly. Distribution plan:

1. Generate the four HTMLs locally with the commands above.
2. Test them in a browser end-to-end yourself (open each, click through a
   few cases, verify Export produces a sane JSON). **Do this before any
   upload.**
3. Upload the four HTMLs to a private S3 bucket.
4. Generate pre-signed URLs (e.g., 30-day expiry) for each file:

   ```bash
   aws s3 presign s3://<your-bucket>/independent_expert_tasks/lpd_task.html  --expires-in 2592000
   aws s3 presign s3://<your-bucket>/independent_expert_tasks/gpd_task.html  --expires-in 2592000
   aws s3 presign s3://<your-bucket>/independent_expert_tasks/lrda_task.html --expires-in 2592000
   aws s3 presign s3://<your-bucket>/independent_expert_tasks/grda_task.html --expires-in 2592000
   ```

5. Paste each pre-signed URL into the corresponding per-task `README.md`
   in this folder, replacing the `<DOWNLOAD_URL>` placeholder.
6. Send each rater the GitHub URL of their per-task README. They click
   the download link in the README, label the cases, export JSON, and
   email it back.

## What the collaborator returns

A single JSON file per task, named e.g. `lpd_results_<rater_initials>.json`.
Schema (for PD tasks):

```json
{
  "<patient_id>": {
    "subtype": "lpd",
    "laterality": "left" | "right" | "bilateral",
    "frequency": 1.5,
    "times": [0.43, 1.10, 1.74, ...],
    "rejected": false
  },
  ...
}
```

For RDA tasks the `times` field is omitted.

## After results come back

Add the four JSONs to `data/labels/independent_expert_v1/` (gitignored if
they contain anything PHI-sensitive; otherwise commit) and write an analysis
script that computes:

- Expert--expert ICC across the new raters (and against MW where overlap exists).
- Expert--algorithm ICC for each rater.
- Comparison of these two against the existing values reported in the manuscript.

This is the analysis Reviewer Note #1 promises.
