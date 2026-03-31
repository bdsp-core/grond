Run the label status report script and present the results.

Execute: `conda run -n morgoth python code/data_management/label_status_report.py`

If the script doesn't exist, tell the user it needs to be created first.

The script is the source of truth — do not recompute or second-guess the numbers. After running the script, reformat the output as clean markdown tables (one per subtype) with columns: Label, Total, %, ≥1 expert, ≥5 experts, ≥10 experts. Include the header line showing total active/excluded segments.
