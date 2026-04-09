# Email template — independent expert annotation request

> **Before sending:** paste each colleague's first name into the
> `<NAME>` slot, paste the four pre-signed URLs into the
> `<DOWNLOAD_URL_*>` slots (these expire 7 days after generation —
> see `.presigned_urls.txt`), and fill in the **deadline** (the
> placeholder is `<DEADLINE>`). The body below is intended for
> Peter, Laura, Aaron, Sahar, Jenn, Tianyu, and Matt.

---

**Subject**: Quick favor — independent expert scoring of EEG patterns for our characterization paper

Hi <NAME>,

I'm writing to ask a favor for the manuscript I'm finishing on **automated characterization of periodic discharges and rhythmic delta activity**. A reviewer (rightly) pointed out that I have served as both the primary annotator and the algorithm developer, so we need a small set of independent expert labels — from people who were not involved in building the algorithm — so we can report expert-vs-algorithm agreement that does not rely on any of my own labels.

I'd love your help. Here is what's involved.

## What you'd be doing

Four short labeling tasks, one per pattern subtype (LPD, GPD, LRDA, GRDA). Each task is **200 ten-second EEG segments** with the algorithm's prediction pre-filled. For most cases you can either accept the default with one keystroke or override with another keystroke — I built the viewers around minimizing clicks. Per task the breakdown is:

- **LPD** (200 segments): laterality (left/right), frequency, individual discharge timing
- **GPD** (200 segments): frequency, individual discharge timing
- **LRDA** (200 segments): laterality (left/right), frequency
- **GRDA** (200 segments): frequency only

You can do all four, or whichever subset you have time for — please don't feel obligated to do all four. The most useful contribution is **at least 2 raters per task** so we can compute expert--expert reliability against expert--algorithm reliability. If you can only do one, any of them helps.

Estimated time per task: **45 minutes to 2 hours** depending on subtype (LPD/GPD are the longer ones because of the discharge-timing step). You can take breaks — your work is auto-saved to the browser's local storage as you go.

## How to do it

Each task is a self-contained HTML file (no install, no clinical-data download — everything is inlined). Open it in any modern browser, click through the cases, and click **Export** at the end to download a JSON of your labels.

**Step 1 — Download the four task files** (each is 80–150 MB; 7-day pre-signed S3 links; valid until `<EXPIRY_DATE>`). From a terminal:

```bash
curl -O '<DOWNLOAD_URL_LPD>'
curl -O '<DOWNLOAD_URL_GPD>'
curl -O '<DOWNLOAD_URL_LRDA>'
curl -O '<DOWNLOAD_URL_GRDA>'
```

If `curl` is not your thing, paste the URLs into your browser address bar and they will download.

**Step 2 — Open the HTML file in any modern browser** (Chrome, Safari, Firefox, Edge):

```bash
open lpd_task.html      # macOS
xdg-open lpd_task.html  # Linux
start lpd_task.html     # Windows
```

…or just double-click the file in Finder/Explorer.

**Step 3 — Score each segment.** The viewer pre-fills the algorithm's best guess. Use these keys (the viewer also has on-screen buttons for everyone):

| Key | Action |
|---|---|
| `Enter` | Accept the algorithm's defaults and advance |
| `1` / `2` | Set laterality = Left / Right (LPD and LRDA only) |
| `↑` / `↓` | Increase / decrease the selected frequency |
| `←` / `→` | Previous / next case (no commit) |
| `A` / `D` | Add / delete discharge marker (LPD and GPD only) |
| `X` | Reject the case (e.g., the pattern looks misclassified) |
| `E` | Export all labels as JSON |

A more detailed instructions document with decision rules per task is here:

> https://github.com/bdsp-core/grond/blob/main/paper_materials/independent_expert_tasks/instructions.md

**Step 4 — Email me the results.** When you're done with each task (or any session), press `E` to export. Your browser will download a JSON file with a name like `lpd_freq_timing_batch1_results.json`. **Please rename it** to include your initials, e.g. `lpd_results_<your initials>.json`, and email the JSON file(s) to me at:

> **mb.westover@gmail.com**

You can send results as you finish each task — no need to wait until all four are done.

## Deadline

I would love to have results by **`<DEADLINE>`** if at all possible.

## Questions

If anything is broken (the viewer won't open, the keys don't work, the export doesn't download a JSON, etc.) just reply to this email and I'll fix it. Same if any of the cases look weird and you're not sure how to score them.

Thank you so much — this materially strengthens the paper.

— Brandon

M Brandon Westover, MD, PhD
mb.westover@gmail.com
