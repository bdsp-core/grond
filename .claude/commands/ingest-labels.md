Ingest new labels into the label system. The user will provide or point to new label data (a JSON file, CSV, or describe what they labeled).

Follow these steps IN ORDER. Do not skip any step.

## Step 1: Understand what's being ingested
Ask the user (if not already clear): what label type (laterality, frequency, spatial, discharge timing, wave timing), and where is the data?

## Step 2: Backup
Run: `conda run -n morgoth python code/data_management/backup_labels.py`
This creates a timestamped backup in `data/labels/backup_YYYYMMDD_HHMMSS/`.

## Step 3: Capture pre-ingest state
Run: `conda run -n morgoth python code/data_management/label_status_report.py --save-snapshot pre`
This saves a JSON snapshot of all label counts for later comparison.

## Step 4: Ingest the labels
Place the new label file in the appropriate location:
- Laterality batch reviews → `data/labels/archive_labels/{subtype}_laterality_batch{N}.json`
- Frequency annotations → add rows to `data/labels/annotations.csv`
- Spatial annotations → add rows to `data/labels/annotations.csv`
- Discharge timing → update `data/labels/discharge_times.json`
- Wave timing → update `data/labels/rda_wave_labels.json`

Then rebuild: `conda run -n morgoth python code/data_management/build_segment_labels.py`

**CRITICAL: Never edit segment_labels.csv directly. It is a generated file.**

## Step 5: Verify
Run: `conda run -n morgoth python code/data_management/label_status_report.py --verify-against pre`
This compares post-ingest counts against the pre-ingest snapshot and reports any unexpected changes.

Check that:
- Total row count is unchanged (13,556 or whatever it was)
- No label counts DECREASED (unless exclusions were added)
- The specific label type being ingested increased by the expected amount
- All other label types are unchanged

If verification fails, tell the user and offer to revert from the backup.

## Step 6: Report
Show the user the before/after comparison and confirm the ingest was successful.
