#!/usr/bin/env bash
# Refresh every cached inference column + every downstream eval/figure/table
# using the current model checkpoints. Intended to keep the manuscript's
# tables, figures, and inline numbers internally consistent after any change
# to the trained-model checkpoints in data/pd_channel_cache/, data/hemi_cache/,
# data/models/, etc.
#
# Total expected wall-clock on Apple Silicon MPS: 1-3 hours.
# Logs land in results/inference_refresh_v1/{NN_step}.log; a results/.../COMPLETE
# marker appears when the chain finishes.

set -u

REPO=/Users/mwestover/GithubRepos/pd-rda-profiler
RUNDIR=$REPO/results/inference_refresh_v1
mkdir -p "$RUNDIR"

export PYTHONHASHSEED=0
cd "$REPO" || exit 1

started=$(date '+%Y-%m-%d %H:%M:%S')
echo "$started  Inference-refresh launcher start (PID $$)" > "$RUNDIR/launcher.log"

run_step () {
    local label="$1"
    local cmd="$2"
    local log="$RUNDIR/${label}.log"
    local t0
    t0=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${t0}] BEGIN $label" | tee -a "$RUNDIR/launcher.log"
    echo "[${t0}] CMD: $cmd" >> "$log"
    bash -c "$cmd" >> "$log" 2>&1
    local rc=$?
    local t1
    t1=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${t1}] END   $label (rc=$rc)" | tee -a "$RUNDIR/launcher.log"
}

# ---- Inference column refreshes ----
# The --force flag overwrites every existing value, ensuring the column
# represents the CURRENT model output rather than a mix of old and new.

run_step 01_refresh_pdchar_freq \
    "conda run -n morgoth python code/evaluation/refresh_pdchar_freq.py --force"

run_step 02_refresh_algo_freq_rda \
    "conda run -n morgoth python code/evaluation/refresh_algo_freq_rda.py --force"

# ---- Independent-expert prediction files ----

run_step 03_generate_v12_predictions \
    "conda run -n morgoth python code/evaluation/generate_v12_predictions.py"

run_step 04_generate_v14_predictions \
    "conda run -n morgoth python code/evaluation/generate_v14_predictions.py"

# ---- Re-run downstream classifier evals on refreshed inputs ----

run_step 05_eval_subtype_classification \
    "conda run -n morgoth python code/evaluation/eval_subtype_classification.py"

run_step 06_eval_3way_classification \
    "conda run -n morgoth python code/evaluation/eval_3way_classification.py"

run_step 07_bipd_detector \
    "conda run -n morgoth python code/bipd_detector.py"

run_step 08_train_lrda_laterality \
    "conda run -n morgoth python code/evaluation/train_lrda_laterality_classifier.py"

# ---- Independent-expert per-pair analysis (drives Fig 5 + Table 5) ----

run_step 09_analyze_independent_expert \
    "conda run -n morgoth python code/evaluation/analyze_independent_expert_v1.py"

# ---- Phase 2: regenerate figures + tables on refreshed columns ----

run_step 10_regen_figures \
    "conda run -n morgoth python paper_materials/generate_all_figures.py"

run_step 11_regen_tables \
    "conda run -n morgoth python paper_materials/generate_all_tables.py"

run_step 12_freq_table_cis \
    "conda run -n morgoth python code/evaluation/freq_table_cis.py"

# ---- Phase 3: comparison harness ----

run_step 13_compare \
    "conda run -n morgoth python code/evaluation/compare_retrain_to_published.py --out $RUNDIR/comparison_report.md"

ended=$(date '+%Y-%m-%d %H:%M:%S')
echo "$ended  Inference-refresh launcher end" >> "$RUNDIR/launcher.log"
date '+%Y-%m-%d %H:%M:%S' > "$RUNDIR/COMPLETE"
