# GRDA annotation task

Thank you for agreeing to score these EEG segments. Your independent
labels will let us measure how well the algorithm agrees with experts
who were **not** involved in its development.

## What you will do

You will review **200 ten-second EEG segments** containing **generalized
rhythmic delta activity (GRDA)** and provide one label per segment:

1. **Frequency** — repetition rate of the rhythmic delta waves, in Hz.

GRDA is bilaterally generalized by definition, so there is no
left-vs-right lateralization decision. If a segment looks clearly
**lateralized** (i.e., LRDA), please **reject** it instead of labeling.

The viewer **pre-fills the algorithm's prediction** for the frequency.
If you agree, you can accept it with a single keystroke. If you
disagree, you can override it.

Plan to spend roughly **5–15 seconds** per segment if the algorithm's
default looks correct, and longer for difficult cases. Total expected
time: about **45 minutes – 1 hour**, broken into as many sessions as
you like.

## Step 1 — Download the labeling tool

Download `grda_task.html` from:

> **`<DOWNLOAD_URL>`** *(filled in by MW before sending)*

The file is a single self-contained HTML document. There is **nothing
to install** — no Python, no Anaconda, no clinical data download. All
the EEG snippets and the algorithm pre-fills are inlined into the file.

## Step 2 — Open it in a browser

Double-click `grda_task.html` to open it in any modern browser
(Chrome, Firefox, Safari, Edge). Your work is auto-saved to the
browser's local storage as you go.

## Step 3 — Score each segment

For each case the viewer shows:

- **Top**: a row of frequency buttons in 0.25 Hz steps — the
  algorithm's prediction is highlighted, use ↑/↓ or click to change.
- **Main canvas**: the 19-channel EEG in standard longitudinal-bipolar
  montage, 10 seconds wide, with a bandpass-filtered overlay that
  tracks the rhythmic delta waveform at the selected frequency.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Accept the currently-selected frequency and advance |
| `↑` / `↓` | Increase / decrease the selected frequency button |
| `←` / `→` | Previous / next case |
| `N` | Toggle the narrowband overlay on/off |
| `X` | Reject this segment (e.g., not actually GRDA, or looks lateralized) |
| `E` | Export all labels as JSON |

If you want to use the mouse instead of the keyboard, every action
also has a button in the top toolbar.

### Decision rules to follow

- **Confirm GRDA**: the rhythmic delta should appear bilaterally and
  symmetrically. If activity is clearly more prominent on one side,
  reject the segment ("looks like LRDA") rather than labeling.
- **Frequency**: the rate at which the delta waves repeat. Round to
  the nearest 0.25 Hz button (or type a custom value). The narrowband
  overlay should "lock onto" the rhythm at the correct frequency.

If you really cannot decide, click **Reject** rather than guessing.

## Step 4 — Send your results back

When you are done (or at the end of any session — partial results are
fine):

1. Press `E` (or click **Export Labels** in the toolbar).
2. Your browser will download a JSON file. **Rename it** to include
   your initials, e.g. `grda_results_AB.json`.
3. Email or Slack the JSON file back to MW.

You do not need to commit anything to GitHub or run any scripts.

## Common questions

**Q: The narrowband overlay does not seem to lock onto the rhythm at
any frequency. What do I do?**
That probably means the segment is not actually rhythmic delta. Reject
it and move on.

**Q: I see clear left-right asymmetry. Reject?**
Yes — if the activity looks dominantly on one side, that is LRDA, not
GRDA. Reject with "looks like LRDA".

**Q: Can I take breaks?**
Yes. Your work is auto-saved to the browser's local storage. Just close
the tab and reopen the same `grda_task.html` file later — it will pick
up where you left off. **Do not clear the browser cache** for that page
until you have exported the final JSON.

**Q: How long do I have?**
*(filled in by MW: e.g., "Two weeks from receipt")*

**Q: Who should I contact with questions?**
M Brandon Westover · `mb.westover@gmail.com`
