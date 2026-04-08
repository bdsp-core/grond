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

| Subtype | Lateralization | Frequency | Discharge timing | Notes |
|---|:---:|:---:|:---:|---|
| LPD  | ✅ left vs right         | ✅ | ✅ | full PD profile |
| GPD  | (bilateral by definition) | ✅ | ✅ | freq + timing only |
| LRDA | ✅ left vs right         | ✅ | — | wave triplets in viewer should be **skipped** |
| GRDA | (generalized by definition) | ✅ | — | freq only |

## Recommended viewers

These existing viewers in `code/generators/labeling/` are reused as-is:

| Task | Viewer entry-point | Captures |
|---|---|---|
| LPD  | [generate_pd_freq_timing_labeler.py](../../code/generators/labeling/generate_pd_freq_timing_labeler.py) `--subtype lpd` | laterality + freq + timing |
| GPD  | [generate_pd_freq_timing_labeler.py](../../code/generators/labeling/generate_pd_freq_timing_labeler.py) `--subtype gpd` | freq + timing (laterality fixed = bilateral) |
| LRDA | [generate_lrda_labeler.py](../../code/generators/labeling/generate_lrda_labeler.py) | laterality + freq (+ wave triplets — **skip**) |
| GRDA | [generate_rda_freq_labeler.py](../../code/generators/labeling/generate_rda_freq_labeler.py) | freq (laterality is generalized by definition) |

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

Once the manifests look good, generate the HTML files. The existing viewers
auto-select cases from `segment_labels.csv`, so they need a small `--manifest`
flag (or a wrapper) to consume our pre-curated 200-segment list. **This step
is not yet wired up** — see the open question in the per-task READMEs.

The intended workflow once that is in place:

```bash
# (after wiring --manifest into each viewer)
conda run -n morgoth python code/generators/labeling/generate_pd_freq_timing_labeler.py \
    --subtype lpd \
    --manifest paper_materials/independent_expert_tasks/lpd/manifest.csv \
    --output paper_materials/independent_expert_tasks/lpd/lpd_task.html
```

## Distribution to collaborators

The generated HTML files will likely be tens to hundreds of MB each (200
segments worth of inlined EEG + per-frequency HPP precomputes). That is too
large for direct commit to git. Recommended distribution path:

1. Generate the four HTML files locally.
2. Create a GitHub release `independent-expert-tasks-v1` and upload the four
   HTMLs as release assets (releases support files up to 2 GB).
3. The per-task `README.md` (already in this folder) contains a placeholder
   `<DOWNLOAD_URL>` that should be filled in with the release-asset URL
   before sharing with the collaborator.
4. Send the GitHub URL of the per-task README to each collaborator. They click
   the download link, label the cases, and send back the exported JSON.

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
