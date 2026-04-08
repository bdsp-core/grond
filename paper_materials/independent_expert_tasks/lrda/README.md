# LRDA annotation task

Thank you for agreeing to score these EEG segments. Your independent
labels will let us measure how well the algorithm agrees with experts
who were **not** involved in its development.

## What you will do

You will review **200 ten-second EEG segments** containing **lateralized
rhythmic delta activity (LRDA)** and provide two labels per segment:

1. **Lateralization** — left vs. right hemisphere.
2. **Frequency** — repetition rate of the rhythmic delta waves, in Hz.

The viewer **pre-fills the algorithm's prediction** for both. If you
agree, you can accept it with a single keystroke. If you disagree, you
can override it.

> **Important — wave triplets:** the LRDA viewer also has a "wave
> triplet" mode (`W` key) for marking onset → peak → offset of every
> rhythmic delta wave. **Please skip this step.** It is left over from
> a different study and we do not need it here. Just label
> lateralization and frequency, then export.

Plan to spend roughly **5–15 seconds** per segment if the algorithm's
default looks correct, and longer for difficult cases. Total expected
time: about **1–1.5 hours**, broken into as many sessions as you like.

## Step 1 — Download the labeling tool

Download `lrda_task.html` from:

> **`<DOWNLOAD_URL>`** *(filled in by MW before sending)*

The file is a single self-contained HTML document. There is **nothing
to install** — no Python, no Anaconda, no clinical data download. All
the EEG snippets and the algorithm pre-fills are inlined into the file.

## Step 2 — Open it in a browser

Double-click `lrda_task.html` to open it in any modern browser
(Chrome, Firefox, Safari, Edge). Your work is auto-saved to the
browser's local storage as you go.

## Step 3 — Score each segment

For each case the viewer shows:

- **Left**: the 19-channel EEG in standard longitudinal-bipolar
  montage, 10 seconds wide. The hemisphere the algorithm thinks is
  involved is highlighted.
- **Right**: a frequency selector (buttons or arrow keys) and a
  per-channel narrowband-amplitude heatmap. A bandpass-filtered
  overlay tracks the rhythmic delta waveform at the selected frequency.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Accept the algorithm's defaults and advance to the next case |
| `1` | Set lateralization = **Left** |
| `2` | Set lateralization = **Right** |
| `3` | Set lateralization = **Bilateral** (use only if the segment looks generalized; we will reclassify it as GRDA in analysis) |
| `↑` / `↓` | Increase / decrease the selected frequency button |
| `←` / `→` | Previous / next case |
| `X` | Reject this segment (e.g., not actually LRDA) |
| `E` | Export all labels as JSON |
| ~~`W`~~ | ~~Wave-mark mode~~ — **skip this; we do not need wave triplets** |

If you want to use the mouse instead of the keyboard, every action
(except wave triplets, which you should ignore) also has a button in
the top toolbar.

### Decision rules to follow

- **Lateralization**: pick the hemisphere where the rhythmic delta
  is clearly dominant. If the activity looks symmetrically generalized,
  set lateralization to **Bilateral** — we will treat that as a
  reclassification to GRDA in the analysis.
- **Frequency**: the rate at which the delta waves repeat. Round to
  the nearest 0.25 Hz button (or type a custom value). The narrowband
  overlay should "lock onto" the rhythm at the correct frequency.
- **Wave triplets**: **do not mark them.** The wave-mark mode (`W`
  key) exists for a different study; ignore it.

If you really cannot decide, click **Reject** rather than guessing.

## Step 4 — Send your results back

When you are done (or at the end of any session — partial results are
fine):

1. Press `E` (or click **Export Labels** in the toolbar).
2. Your browser will download a JSON file. **Rename it** to include
   your initials, e.g. `lrda_results_AB.json`.
3. Email or Slack the JSON file back to MW.

You do not need to commit anything to GitHub or run any scripts. The
JSON file will contain a `triplets` array per segment — that is from
the wave-triplet UI that you skipped, and will be empty. That is
expected and correct.

## Common questions

**Q: I accidentally pressed `W` and now I am in wave-mark mode. What do
I do?**
Press `W` again or `Esc` to exit wave-mark mode. Any triplets you
accidentally placed can be left as-is — we will ignore them in
analysis.

**Q: The narrowband overlay does not seem to lock onto the rhythm at
any frequency. What do I do?**
That probably means the segment is not actually LRDA. Reject it ("not
LRDA") and move on.

**Q: Can I take breaks?**
Yes. Your work is auto-saved to the browser's local storage. Just close
the tab and reopen the same `lrda_task.html` file later — it will pick
up where you left off. **Do not clear the browser cache** for that page
until you have exported the final JSON.

**Q: How long do I have?**
*(filled in by MW: e.g., "Two weeks from receipt")*

**Q: Who should I contact with questions?**
M Brandon Westover · `mb.westover@gmail.com`
