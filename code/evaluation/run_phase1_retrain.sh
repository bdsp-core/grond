#!/usr/bin/env bash
# Phase 1 from-scratch retrain launcher.
#
# Runs every model training sequentially under conda env morgoth, writing per-step
# logs into results/retrain_v1/. Use `tail -f` on individual logs to monitor;
# when the chain completes, $RUNDIR/COMPLETE will exist with the final exit code.
#
# Total expected wall-clock on Apple Silicon MPS: 12-24 hours.
# Pass --start-at "HH:MM" to defer launch until that local time tonight.

set -u  # don't `set -e` -- we want subsequent steps to attempt even if an earlier one fails

REPO=/Users/mwestover/GithubRepos/pd-rda-profiler
RUNDIR=$REPO/results/retrain_v1
mkdir -p "$RUNDIR"

# Optional --start-at HH:MM: sleep until the named time tonight before training.
START_AT=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --start-at) START_AT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [ -n "$START_AT" ]; then
    target_epoch=$(date -j -f '%Y-%m-%d %H:%M:%S' \
        "$(date '+%Y-%m-%d') ${START_AT}:00" '+%s' 2>/dev/null) || target_epoch=""
    if [ -n "$target_epoch" ]; then
        now_epoch=$(date '+%s')
        sleep_secs=$((target_epoch - now_epoch))
        if [ "$sleep_secs" -gt 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S')  sleeping ${sleep_secs}s until ${START_AT}" \
                > "$RUNDIR/launcher.log"
            # Use caffeinate so macOS doesn't suspend the process while sleeping
            command -v caffeinate >/dev/null 2>&1 \
                && caffeinate -is sleep "$sleep_secs" \
                || sleep "$sleep_secs"
        fi
    fi
fi

# Ensure deterministic Python hashing across runs
export PYTHONHASHSEED=0

cd "$REPO" || exit 1

started=$(date '+%Y-%m-%d %H:%M:%S')
echo "$started  Phase 1 launcher start (PID $$)" >> "$RUNDIR/launcher.log"

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
    return $rc
}

# ---- Step 0: pre-flight ----
run_step 00_verify_data \
    "conda run -n morgoth python code/evaluation/verify_local_data.py"

# ---- Step 1: ChannelPD-Net (5-fold CNN) ----
# NB: code/hemi_detector/train.py is an early HemiNet design experiment, NOT
# the production ChannelPD-Net trainer (its outputs go to data/hemi_cache/exp1_1/
# and it does not write cnn_attn_fold*.pt). The real production trainer for
# ChannelPDNetAttention is below; verified by the cnn_attn_fold*.pt save site.
run_step 01_channelpdnet \
    "conda run -n morgoth python code/pd_channel_detector/train_cnn_attention.py"

# ---- Step 2: HemiCET-UNet (5-fold UNet) ----
run_step 02_hemicet \
    "conda run -n morgoth python code/hemi_detector/train_hemi_cet.py"

# ---- Step 3: LPD-vs-GPD RF (300 trees) ----
run_step 03_lpd_vs_gpd_rf \
    "conda run -n morgoth python code/evaluation/eval_subtype_classification.py"

# ---- Step 4: 3-way LPD/GPD/BIPD RF ----
run_step 04_three_way_rf \
    "conda run -n morgoth python code/evaluation/eval_3way_classification.py"

# ---- Step 5: BIPD GBT ----
run_step 05_bipd_gbt \
    "conda run -n morgoth python code/bipd_detector.py"

# ---- Step 6: LRDA laterality classifier ----
run_step 06_lrda_laterality \
    "conda run -n morgoth python code/evaluation/train_lrda_laterality_classifier.py"

# ---- Step 7: NB-Hilbert V12 grid search ----
run_step 07_nb_hilbert_grid \
    "conda run -n morgoth python code/evaluation/lrda_freq_hyperparam_sweep.py"

# ---- Phase 2: re-run figure + table generators on retrained checkpoints ----
run_step 08_regen_figures \
    "conda run -n morgoth python paper_materials/generate_all_figures.py"

run_step 09_regen_tables \
    "conda run -n morgoth python paper_materials/generate_all_tables.py"

run_step 10_freq_table_cis \
    "conda run -n morgoth python code/evaluation/freq_table_cis.py"

# ---- Phase 3: comparison harness ----
# (The harness compares the freshly-regenerated JSON results vs the published
# values stored in data/evaluation_results/ and the manuscript's Tables 3-6.)
if [ -f "code/evaluation/compare_retrain_to_published.py" ]; then
    run_step 11_compare \
        "conda run -n morgoth python code/evaluation/compare_retrain_to_published.py --out $RUNDIR/comparison_report.md"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP 11_compare: compare_retrain_to_published.py not yet implemented" \
        | tee -a "$RUNDIR/launcher.log"
fi

ended=$(date '+%Y-%m-%d %H:%M:%S')
echo "$ended  Phase 1 launcher end" >> "$RUNDIR/launcher.log"
date '+%Y-%m-%d %H:%M:%S' > "$RUNDIR/COMPLETE"
