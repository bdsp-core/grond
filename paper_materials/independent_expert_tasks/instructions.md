# Independent expert annotation instructions

> **Audience:** colleagues helping us validate the GROND PD/RDA characterization system. You should have received an email from M Brandon Westover (mb.westover@gmail.com) with the four download links. This document explains how to use the labeling tools and what each task asks of you.

## Why this exists

For our characterization paper, the algorithm and most of the training labels were produced by the same person (MW). To meet a reviewer's request for an independent validation, we need a small set of expert labels from people who were *not* involved in algorithm development. Your labels will let us compute expert--expert agreement (across the new raters) and expert--algorithm agreement against the new raters, neither of which uses any of MW's labels.

## What you'll be doing

Four labeling tasks, one per pattern subtype, **200 ten-second segments each**. The viewer pre-fills the algorithm's prediction; in most cases you can either accept it with `Enter` or override with one or two keystrokes.

| Task | Lateralization | Frequency | Discharge timing | Expected time |
|---|:---:|:---:|:---:|---|
| **LPD**  | ✅ left/right        | ✅ | ✅ | ~1.5–2 hours |
| **GPD**  | (bilateral by definition) | ✅ | ✅ | ~1.5–2 hours |
| **LRDA** | ✅ left/right        | ✅ | — | ~45–90 min |
| **GRDA** | (generalized by definition) | ✅ | — | ~45–60 min |

You don't have to do all four — please do **whichever you have time for**. The most useful contribution is at least 2 raters per task, so even a single task is genuinely helpful.

## How the viewers work

Each task is a single self-contained HTML file (no install, no clinical-data download — everything is inlined in the file).

1. **Download** the HTML for the task you're doing using the `curl` command in the email.
2. **Open** it by double-clicking, or `open task.html` (macOS) / `xdg-open task.html` (Linux) / `start task.html` (Windows).
3. **Score each segment.** The algorithm's prediction is highlighted as the default. Either accept (one keystroke) or override (one or two keystrokes).
4. **Export** when done — press `E` and your browser will download a JSON file.
5. **Email** the JSON to `mb.westover@gmail.com` (rename it to include your initials, e.g. `lpd_results_AB.json`).

Your work is **auto-saved** to the browser's local storage as you go. You can close the tab and come back to the same task later — it will pick up where you left off. **Do not clear the browser cache** for that page until you have exported the final JSON.

## Universal keyboard shortcuts

These work in all four viewers.

| Key | Action |
|---|---|
| `Enter` | Accept the algorithm's defaults and advance to the next case |
| `←` / `→` | Previous / next case (does **not** commit a label) |
| `↑` / `↓` | Increase / decrease the selected frequency |
| `X` | Reject this segment (with a reason if asked) |
| `E` | Export all labels as JSON to download |

The viewers also have on-screen buttons for everything if you'd rather click than press keys.

---

## Per-task details

### LPD task (200 segments)

You provide: **laterality**, **frequency**, **individual discharge timing**.

| Key | Action (LPD-specific) |
|---|---|
| `1` | Set laterality = **Left** |
| `2` | Set laterality = **Right** |
| `A` | Add-discharge mode (click on the EEG to add a discharge time) |
| `D` | Delete-discharge mode (click on a marker to remove it) |

**Decision rules:**

- **Lateralization**: pick the hemisphere where the discharges are clearly dominant. If discharges look bilateral and *independent* (BIPD), reject the segment with "looks like BIPD".
- **Frequency**: rate at which the discharges repeat. Round to the nearest 0.25 Hz button (or type a custom value).
- **Discharge timing**: the algorithm's red dashed lines should sit on each discharge peak. Add markers it missed, delete spurious ones, and drag any that are visibly misaligned. **Don't sweat sub-50-ms jitter** — only correct markers that are on the wrong discharge entirely or that were missed/spurious.

If you can't decide, **reject** rather than guessing.

---

### GPD task (200 segments)

You provide: **frequency**, **individual discharge timing**. (GPD is bilateral by definition — no laterality input.)

| Key | Action (GPD-specific) |
|---|---|
| `A` | Add-discharge mode |
| `D` | Delete-discharge mode |

**Decision rules:**

- **Confirm GPD**: discharges should appear bilaterally synchronously. If they're clearly more prominent on one side, reject as "looks like LPD". If the two sides are independent, reject as "looks like BIPD".
- **Frequency**: same as LPD — repetition rate, rounded to 0.25 Hz.
- **Discharge timing**: same as LPD — only correct major errors.

---

### LRDA task (200 segments)

You provide: **laterality** and **frequency**.

| Key | Action (LRDA-specific) |
|---|---|
| `1` | Set laterality = **Left** |
| `2` | Set laterality = **Right** |
| `N` | Toggle the green narrowband overlay on/off |

**Decision rules:**

- **Laterality**: pick the hemisphere where the rhythmic delta is clearly dominant. The narrowband overlay (green trace, only drawn on the selected side) should "lock onto" the rhythm — that's your visual confirmation. If the activity looks symmetrically generalized (i.e., would be better classified as GRDA), **reject** the segment.
- **Frequency**: the rate at which the delta waves repeat. The narrowband overlay should track the rhythm cleanly when you have the right frequency selected.

There is no discharge-timing step for LRDA — `A` and `D` are inert.

---

### GRDA task (200 segments)

You provide: **frequency only**. (GRDA is generalized by definition — no laterality input.)

| Key | Action (GRDA-specific) |
|---|---|
| `N` | Toggle the green narrowband overlay on/off |

**Decision rules:**

- **Confirm GRDA**: rhythmic delta should appear bilaterally and symmetrically. If it's clearly more prominent on one side, **reject** as "looks like LRDA".
- **Frequency**: same as LRDA — narrowband overlay should track the rhythm.

There is no laterality input and no discharge-timing step for GRDA — `1`/`2`/`A`/`D` are inert.

---

## Exporting and sending results

When you finish a task (or even just one session — partial results are useful):

1. Press `E` (or click **Export Labels** in the toolbar).
2. Your browser will download a JSON file. **Rename it** to include your initials, e.g. `lpd_results_AB.json`.
3. Email it to **mb.westover@gmail.com**.

You can send results as you finish each task — no need to wait until all four are done.

## Common questions

**Q: I closed the tab. Did I lose my work?**
No — the viewer saves to your browser's local storage as you go. Reopen the same HTML file in the same browser and you'll be where you left off. (Don't clear that page's cache until you've exported.)

**Q: A segment looks misclassified — what do I do?**
Reject it with the most appropriate reason (e.g., "looks like LPD", "looks like GRDA", "artifact", "background"). Rejected segments are exactly the kind of feedback we want.

**Q: The algorithm's pre-filled defaults are systematically off by a tiny amount — should I correct every one?**
No. Only correct material errors. Sub-50-ms timing jitter, sub-0.25-Hz frequency offsets, and similar small things are below the noise floor of the analysis.

**Q: Can I use a tablet / iPad?**
The viewers are tested on desktop browsers with mouse + keyboard. They should work on a touchscreen for the click-based interactions but the keyboard shortcuts won't be available — desktop is recommended.

**Q: Do I need to send results in any particular format?**
No — just press `E`, rename the file, and send the JSON as-is. The viewer's export format is exactly what we need.

**Q: Who do I contact with questions or bugs?**
M Brandon Westover · `mb.westover@gmail.com`

Thank you for helping with this — it materially strengthens the paper.
