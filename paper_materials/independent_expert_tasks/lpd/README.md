# LPD annotation task

Thank you for agreeing to score these EEG segments. Your independent
labels will let us measure how well the algorithm agrees with experts
who were **not** involved in its development — an important
robustness check for the manuscript.

## What you will do

You will review **200 ten-second EEG segments** containing
**lateralized periodic discharges (LPDs)** and provide three labels per
segment:

1. **Lateralization** — left vs. right hemisphere.
2. **Frequency** — repetition rate of the discharges, in Hz.
3. **Discharge timing** — the precise time (in seconds, within the 10-s
   window) of each individual discharge.

The viewer **pre-fills the algorithm's prediction** for all three. If
you agree, you can accept it with a single keystroke. If you disagree,
you can override it.

Plan to spend roughly **5–15 seconds** per segment if the algorithm's
default looks correct, and longer for difficult cases. Total expected
time: about **1.5–2 hours**, broken into as many sessions as you like.

## Step 1 — Download the labeling tool

Download `lpd_task.html` from:

> **`<DOWNLOAD_URL>`** *(filled in by MW before sending)*

The file is a single self-contained HTML document. There is **nothing
to install** — no Python, no Anaconda, no clinical data download. All
the EEG snippets and the algorithm pre-fills are inlined into the file.

## Step 2 — Open it in a browser

Double-click `lpd_task.html` to open it in any modern browser
(Chrome, Firefox, Safari, Edge). Your work is auto-saved to the
browser's local storage as you go, so you can close the tab and come
back to the same case later.

## Step 3 — Score each segment

For each case the viewer shows:

- **Left**: the 19-channel EEG in standard longitudinal-bipolar
  montage, 10 seconds wide. The hemisphere the algorithm thinks is
  involved is highlighted.
- **Right**: a frequency selector (buttons or arrow keys) and a
  per-channel evidence trace. Red dashed lines mark the algorithm's
  pre-detected discharge times.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Accept the algorithm's defaults and advance to the next case |
| `1` | Set lateralization = **Left** |
| `2` | Set lateralization = **Right** |
| `↑` / `↓` | Increase / decrease the selected frequency button |
| `←` / `→` | Previous / next case |
| `A` | Add-discharge mode (click on the EEG to add a discharge time) |
| `D` | Delete-discharge mode (click on a marker to remove it) |
| `X` | Reject this segment (e.g., not actually LPD) |
| `E` | Export all labels as JSON |

If you want to use the mouse instead of the keyboard, every action also
has a button in the top toolbar.

### Decision rules to follow

- **Lateralization**: pick the hemisphere where the discharges are
  clearly dominant. If discharges look bilateral and independent (BIPD),
  mark as **rejected** with a note in the rejection menu.
- **Frequency**: the rate at which the discharges repeat. Round to the
  nearest 0.25 Hz button (or type a custom value).
- **Discharge timing**: the algorithm's red dashed lines should sit on
  the peak of each discharge. Add markers for any discharges it missed,
  delete markers it added in error, and drag markers if they are
  visibly misaligned.

If you really cannot decide, click **Reject** rather than guessing — we
would much rather know which cases are ambiguous than have noisy labels.

## Step 4 — Send your results back

When you are done (or at the end of any session — it is fine to send
partial results):

1. Press `E` (or click **Export Labels** in the toolbar).
2. Your browser will download a file called something like
   `lpd_freq_timing_batch1_results.json`. **Rename it** to include
   your initials, e.g. `lpd_results_AB.json`.
3. Email or Slack the JSON file back to MW.

That is it — you do not need to commit anything to GitHub or run any
scripts. The JSON file is everything we need.

## Common questions

**Q: I see a segment that does not look like LPD at all. What do I do?**
Press `X` (reject). Add a brief reason in the dropdown if you can
(e.g., "looks like GPD", "looks like LRDA", "artifact", "background").

**Q: The algorithm's discharge markers are systematically off by a few
ms. Should I drag every single one?**
No — small offsets are fine. Only correct markers that are on the
wrong discharge entirely, or that were missed/spurious. Sub-50-ms
jitter is below the noise floor of the analysis.

**Q: Can I take breaks?**
Yes. Your work is saved to the browser's local storage automatically.
Just close the tab and reopen the same `lpd_task.html` file later — it
will pick up where you left off. **Do not clear the browser cache** for
that page until you have exported the final JSON.

**Q: How long do I have?**
*(filled in by MW: e.g., "Two weeks from receipt")*

**Q: Who should I contact with questions?**
M Brandon Westover · `mb.westover@gmail.com`
