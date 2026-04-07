"""
Phase 4: Validate whether the CNN's PD probability predicts frequency estimation accuracy.

Hypothesis: patients with clearer PD patterns (higher CNN PD probability) should have
more accurate frequency estimates, because the periodic discharge signal is stronger
and easier to measure.

For each patient:
  - Compute mean/max/top-4 PD probability from CNN (out-of-fold predictions)
  - Run LOPO Ridge frequency estimation to get per-patient |error|
  - Correlate PD probability with frequency error

Run with: conda run -n foe_dl python code/pd_channel_detector/validate_quality_predictor.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_channel_detector.channel_cnn import ChannelPDNet
from pd_channel_detector.train_cnn import ChannelPDDataset, evaluate, compute_auc
from optimization_harness_v2 import (
    load_dataset, _build_segment_level_data, ridge_predict_fn,
    FEATURE_COLS, LATERALITY_FEATURE_COLS, ALL_FEATURE_COLS,
)

CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
RESULTS_DIR = PROJECT_DIR / 'results'
DEVICE = torch.device('cpu')


def get_oof_pd_probs():
    """Get out-of-fold CNN PD probabilities for every channel.

    Reconstructs the same 5-fold patient-stratified splits used during training,
    then runs each fold's saved model on its held-out channels.

    Returns:
        patient_ids: (N,) array of patient IDs per channel
        subtypes: (N,) array of subtypes per channel
        pd_probs: (N,) array of PD probabilities (out-of-fold)
    """
    print("Loading channel dataset...")
    data = np.load(str(CACHE_DIR / 'channel_dataset.npz'), allow_pickle=True)
    channels = data['channels']       # (N, 2000)
    labels = data['labels']           # (N,)
    patient_ids = data['patient_ids'] # (N,)
    subtypes = data['subtypes']       # (N,)

    unique_patients = np.unique(patient_ids)
    n_total = len(labels)

    # Reconstruct the same fold assignments (same RNG seed as train_cnn.py)
    n_folds = 5
    rng = np.random.RandomState(42)

    pid_to_subtype = {}
    for i, pid in enumerate(patient_ids):
        pid = str(pid)
        if pid not in pid_to_subtype:
            pid_to_subtype[pid] = str(subtypes[i])

    subtype_groups = {}
    for pid in unique_patients:
        st = pid_to_subtype.get(str(pid), 'unknown')
        if st not in subtype_groups:
            subtype_groups[st] = []
        subtype_groups[st].append(str(pid))

    patient_folds = {}
    for st, pids in subtype_groups.items():
        pids_shuffled = list(pids)
        rng.shuffle(pids_shuffled)
        for i, pid in enumerate(pids_shuffled):
            patient_folds[pid] = i % n_folds

    # Get out-of-fold predictions
    oof_pd_probs = np.full(n_total, np.nan, dtype=np.float32)

    # Build dummy freq_targets (we only need PD probs)
    freq_targets = np.full(n_total, np.nan, dtype=np.float32)

    for fold in range(n_folds):
        model_path = CACHE_DIR / f'cnn_fold{fold}.pt'
        print(f"  Loading fold {fold} model from {model_path}...")

        model = ChannelPDNet().to(DEVICE)
        state_dict = torch.load(str(model_path), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Get val channels for this fold
        val_mask = np.array([patient_folds.get(str(p), -1) == fold for p in patient_ids])

        val_channels = channels[val_mask]
        val_labels = labels[val_mask]
        val_freqs = freq_targets[val_mask]

        val_ds = ChannelPDDataset(val_channels, val_labels, val_freqs, augment=False)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

        val_results = evaluate(model, val_loader)
        oof_pd_probs[val_mask] = val_results['pd_probs']

        n_val = int(np.sum(val_mask))
        print(f"    Fold {fold}: {n_val} channels, mean PD prob = {val_results['pd_probs'].mean():.4f}")

    return patient_ids, subtypes, oof_pd_probs


def aggregate_patient_pd_probs(patient_ids, subtypes, pd_probs):
    """Aggregate channel-level PD probabilities to patient level.

    Returns dict: patient_id -> {subtype, mean_pd_prob, max_pd_prob, top4_pd_prob}
    """
    patient_data = {}
    unique_pids = np.unique(patient_ids)

    for pid in unique_pids:
        mask = patient_ids == pid
        probs = pd_probs[mask]
        sub = subtypes[mask][0]

        # Sort descending for top-k
        sorted_probs = np.sort(probs)[::-1]
        top4 = sorted_probs[:min(4, len(sorted_probs))]

        patient_data[str(pid)] = {
            'subtype': str(sub),
            'mean_pd_prob': float(np.mean(probs)),
            'max_pd_prob': float(np.max(probs)),
            'top4_pd_prob': float(np.mean(top4)),
            'n_channels': int(len(probs)),
        }

    return patient_data


def run_lopo_freq_estimation():
    """Run LOPO Ridge frequency estimation and return per-patient results.

    Returns dict: patient_id -> {gold_freq, pred_freq, abs_error, subtype}
    """
    print("\nLoading main dataset for frequency estimation...")
    dataset = load_dataset(verbose=True)
    df = dataset['df']
    features = dataset['features']
    segments = dataset['segments']

    # Build segment-level data
    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_pids = np.array(seg_pids)
    unique_patients = df['patient_id'].values

    predict_fn = ridge_predict_fn(alpha=1.0)

    # Use only base 6 features
    base6_indices = [ALL_FEATURE_COLS.index(c) for c in FEATURE_COLS]

    patient_freq_results = {}
    n_processed = 0

    print("Running LOPO Ridge (base 6 features, alpha=1)...")
    for pat in unique_patients:
        test_mask = seg_pids == pat
        train_mask = ~test_mask

        if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
            continue

        train_feats = seg_features[train_mask][:, base6_indices]
        test_feats = seg_features[test_mask][:, base6_indices]

        train_segs = [seg_arrays[i] for i in np.where(train_mask)[0]]
        test_segs = [seg_arrays[i] for i in np.where(test_mask)[0]]

        try:
            preds = predict_fn(train_segs, seg_labels[train_mask], train_feats,
                               test_segs, test_feats)
            preds = np.asarray(preds, dtype=float)
            valid_preds = preds[np.isfinite(preds)]
            if len(valid_preds) > 0:
                pred_freq = float(np.mean(valid_preds))
            else:
                pred_freq = np.nan
        except Exception:
            pred_freq = np.nan

        row = df[df['patient_id'] == pat].iloc[0]
        gold_freq = float(row['gold_standard_freq'])
        subtype = str(row['subtype'])

        if np.isfinite(pred_freq) and np.isfinite(gold_freq) and gold_freq > 0:
            patient_freq_results[str(pat)] = {
                'gold_freq': gold_freq,
                'pred_freq': pred_freq,
                'abs_error': abs(pred_freq - gold_freq),
                'subtype': subtype,
            }

        n_processed += 1
        if n_processed % 100 == 0:
            print(f"  Processed {n_processed}/{len(unique_patients)} patients")

    print(f"  Done: {len(patient_freq_results)} patients with valid frequency predictions")
    return patient_freq_results


def generate_html_report(merged_data, output_path):
    """Generate interactive HTML report with scatter plot and bar chart."""

    # Prepare data arrays
    mean_probs = []
    max_probs = []
    top4_probs = []
    abs_errors = []
    patient_subtypes = []
    patient_labels = []

    for pid, d in merged_data.items():
        mean_probs.append(d['mean_pd_prob'])
        max_probs.append(d['max_pd_prob'])
        top4_probs.append(d['top4_pd_prob'])
        abs_errors.append(d['abs_error'])
        patient_subtypes.append(d['subtype'])
        patient_labels.append(pid)

    mean_probs = np.array(mean_probs)
    abs_errors = np.array(abs_errors)

    # Compute Spearman for annotation
    rho_mean, p_mean = spearmanr(mean_probs, abs_errors)

    # Quartile analysis
    quartile_edges = np.percentile(mean_probs, [0, 25, 50, 75, 100])
    quartile_labels_chart = []
    quartile_means = []
    quartile_sems = []
    quartile_ns = []

    for q in range(4):
        lo, hi = quartile_edges[q], quartile_edges[q + 1]
        if q == 3:
            mask = (mean_probs >= lo) & (mean_probs <= hi)
        else:
            mask = (mean_probs >= lo) & (mean_probs < hi)
        q_errors = abs_errors[mask]
        quartile_labels_chart.append(f'Q{q+1} [{lo:.2f}-{hi:.2f}]')
        quartile_means.append(float(np.mean(q_errors)) if len(q_errors) > 0 else 0)
        quartile_sems.append(float(np.std(q_errors) / np.sqrt(len(q_errors))) if len(q_errors) > 1 else 0)
        quartile_ns.append(int(np.sum(mask)))

    # Build scatter data by subtype
    lpd_x, lpd_y, lpd_labels = [], [], []
    gpd_x, gpd_y, gpd_labels = [], [], []
    other_x, other_y, other_labels = [], [], []

    for i, st in enumerate(patient_subtypes):
        if st == 'lpd':
            lpd_x.append(mean_probs[i])
            lpd_y.append(abs_errors[i])
            lpd_labels.append(patient_labels[i])
        elif st == 'gpd':
            gpd_x.append(mean_probs[i])
            gpd_y.append(abs_errors[i])
            gpd_labels.append(patient_labels[i])
        else:
            other_x.append(mean_probs[i])
            other_y.append(abs_errors[i])
            other_labels.append(patient_labels[i])

    def to_js_array(arr):
        return '[' + ', '.join(f'{v:.4f}' for v in arr) + ']'

    def to_js_str_array(arr):
        return '[' + ', '.join(f'"{v}"' for v in arr) + ']'

    # Build scatter datasets for xy chart
    scatter_datasets = []

    if lpd_x:
        scatter_datasets.append(f'''{{
                label: 'LPD (n={len(lpd_x)})',
                data: {json.dumps([{"x": round(x, 4), "y": round(y, 4)} for x, y in zip(lpd_x, lpd_y)])},
                backgroundColor: 'rgba(44, 160, 44, 0.6)',
                borderColor: 'rgba(44, 160, 44, 1)',
                pointRadius: 5,
                pointHoverRadius: 8,
            }}''')

    if gpd_x:
        scatter_datasets.append(f'''{{
                label: 'GPD (n={len(gpd_x)})',
                data: {json.dumps([{"x": round(x, 4), "y": round(y, 4)} for x, y in zip(gpd_x, gpd_y)])},
                backgroundColor: 'rgba(31, 119, 180, 0.6)',
                borderColor: 'rgba(31, 119, 180, 1)',
                pointRadius: 5,
                pointHoverRadius: 8,
            }}''')

    if other_x:
        scatter_datasets.append(f'''{{
                label: 'Other (n={len(other_x)})',
                data: {json.dumps([{"x": round(x, 4), "y": round(y, 4)} for x, y in zip(other_x, other_y)])},
                backgroundColor: 'rgba(200, 200, 200, 0.6)',
                borderColor: 'rgba(150, 150, 150, 1)',
                pointRadius: 5,
                pointHoverRadius: 8,
            }}''')

    scatter_datasets_str = ',\n            '.join(scatter_datasets)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PD Probability vs Frequency Estimation Error</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
            color: #333;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 20px;
        }}
        .stats-box {{
            background: white;
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            font-size: 14px;
            line-height: 1.8;
        }}
        .stats-box h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #eee;
            padding: 3px 0;
        }}
        .stat-label {{
            font-weight: 500;
            color: #555;
        }}
        .stat-value {{
            font-weight: 700;
            color: #2c3e50;
        }}
        .significant {{
            color: #27ae60;
        }}
        .not-significant {{
            color: #e74c3c;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .chart-title {{
            font-size: 16px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        canvas {{
            width: 100% !important;
            height: 400px !important;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 800px) {{
            .two-col {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <h1>Phase 4: PD Probability vs Frequency Estimation Accuracy</h1>
    <p class="subtitle">Does the CNN's PD confidence predict how well we can estimate frequency?</p>

    <div id="statsContainer"></div>

    <div class="chart-container">
        <div class="chart-title">Mean PD Probability vs Absolute Frequency Error (Hz)</div>
        <canvas id="scatterChart"></canvas>
    </div>

    <div class="two-col">
        <div class="chart-container">
            <div class="chart-title">Mean |Error| by PD Probability Quartile</div>
            <canvas id="quartileChart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">Quartile Details</div>
            <canvas id="quartileNChart"></canvas>
        </div>
    </div>

    <script>
        // --- Stats ---
        const stats = __STATS_JSON__;
        let statsHtml = '<div class="stats-box"><h3>Correlation Analysis</h3>';
        for (const s of stats) {{
            const cls = (s.p < 0.05) ? 'significant' : 'not-significant';
            statsHtml += '<div class="stat-row">';
            statsHtml += '<span class="stat-label">' + s.label + '</span>';
            statsHtml += '<span class="stat-value ' + cls + '">rho = ' + s.rho.toFixed(4) + ' (p = ' + s.p.toExponential(2) + ')</span>';
            statsHtml += '</div>';
        }}
        statsHtml += '</div>';
        document.getElementById('statsContainer').innerHTML = statsHtml;

        // --- Scatter plot ---
        new Chart(document.getElementById('scatterChart').getContext('2d'), {{
            type: 'scatter',
            data: {{
                datasets: [
            {scatter_datasets_str}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Mean PD Probability (CNN out-of-fold)' }},
                        min: 0, max: 1,
                    }},
                    y: {{
                        title: {{ display: true, text: 'Absolute Frequency Error (Hz)' }},
                        min: 0,
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Spearman rho = {rho_mean:.4f} (p = {p_mean:.2e})',
                        font: {{ size: 14 }},
                    }},
                    legend: {{ position: 'top' }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                return ctx.dataset.label + ': PD prob=' + ctx.parsed.x.toFixed(3) + ', Error=' + ctx.parsed.y.toFixed(3) + ' Hz';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // --- Quartile bar chart ---
        new Chart(document.getElementById('quartileChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(quartile_labels_chart)},
                datasets: [{{
                    label: 'Mean |Error| (Hz)',
                    data: {json.dumps([round(v, 4) for v in quartile_means])},
                    backgroundColor: ['rgba(231, 76, 60, 0.7)', 'rgba(241, 196, 15, 0.7)',
                                       'rgba(52, 152, 219, 0.7)', 'rgba(46, 204, 113, 0.7)'],
                    borderColor: ['rgba(231, 76, 60, 1)', 'rgba(241, 196, 15, 1)',
                                   'rgba(52, 152, 219, 1)', 'rgba(46, 204, 113, 1)'],
                    borderWidth: 1,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'PD Probability Quartile' }} }},
                    y: {{ title: {{ display: true, text: 'Mean |Error| (Hz)' }}, beginAtZero: true }}
                }},
                plugins: {{
                    legend: {{ display: false }},
                }}
            }}
        }});

        // --- Quartile N chart ---
        new Chart(document.getElementById('quartileNChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(quartile_labels_chart)},
                datasets: [{{
                    label: 'N patients',
                    data: {json.dumps(quartile_ns)},
                    backgroundColor: 'rgba(149, 165, 166, 0.7)',
                    borderColor: 'rgba(149, 165, 166, 1)',
                    borderWidth: 1,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'PD Probability Quartile' }} }},
                    y: {{ title: {{ display: true, text: 'N Patients' }}, beginAtZero: true }}
                }},
                plugins: {{
                    legend: {{ display: false }},
                }}
            }}
        }});
    </script>
</body>
</html>'''

    # Inject stats JSON
    # We'll compute all the stats to embed
    return html


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 4: Validate PD Probability as Frequency Quality Predictor")
    print("=" * 70)

    # Step 1: Get out-of-fold PD probabilities
    print("\n--- Step 1: Out-of-fold CNN PD probabilities ---")
    patient_ids, subtypes, pd_probs = get_oof_pd_probs()

    # Step 2: Aggregate to patient level
    print("\n--- Step 2: Aggregate PD probabilities per patient ---")
    patient_pd = aggregate_patient_pd_probs(patient_ids, subtypes, pd_probs)
    print(f"  {len(patient_pd)} patients with PD probabilities")

    # Step 3: Run LOPO frequency estimation
    print("\n--- Step 3: LOPO Ridge frequency estimation ---")
    patient_freq = run_lopo_freq_estimation()

    # Step 4: Merge datasets
    print("\n--- Step 4: Merge PD probabilities with frequency errors ---")
    merged = {}
    for pid in patient_pd:
        if pid in patient_freq:
            merged[pid] = {
                **patient_pd[pid],
                **patient_freq[pid],
            }

    print(f"  Merged: {len(merged)} patients with both PD probs and freq estimates")

    if len(merged) < 10:
        print("ERROR: Too few merged patients. Cannot proceed.")
        return

    # Extract arrays
    mean_probs = np.array([d['mean_pd_prob'] for d in merged.values()])
    max_probs = np.array([d['max_pd_prob'] for d in merged.values()])
    top4_probs = np.array([d['top4_pd_prob'] for d in merged.values()])
    abs_errors = np.array([d['abs_error'] for d in merged.values()])
    patient_subtypes = np.array([d['subtype'] for d in merged.values()])

    # Step 5: Correlations
    print("\n" + "=" * 70)
    print("RESULTS: PD Probability vs Frequency Error Correlations")
    print("=" * 70)

    stats_for_html = []

    def report_corr(label, probs, errors, mask=None):
        if mask is not None:
            p, e = probs[mask], errors[mask]
        else:
            p, e = probs, errors
        if len(p) < 5:
            print(f"  {label}: N={len(p)} (too few)")
            return
        rho, pval = spearmanr(p, e)
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {label:<45s}  rho={rho:+.4f}  p={pval:.2e}  N={len(p):>4d}  {sig}")
        stats_for_html.append({'label': label, 'rho': round(float(rho), 4), 'p': float(pval), 'n': int(len(p))})

    print("\nAll patients:")
    report_corr("mean_pd_prob vs |freq_error|", mean_probs, abs_errors)
    report_corr("max_pd_prob vs |freq_error|", max_probs, abs_errors)
    report_corr("top4_pd_prob vs |freq_error|", top4_probs, abs_errors)

    print("\nLPD only:")
    lpd_mask = patient_subtypes == 'lpd'
    report_corr("mean_pd_prob vs |freq_error| (LPD)", mean_probs, abs_errors, lpd_mask)
    report_corr("max_pd_prob vs |freq_error| (LPD)", max_probs, abs_errors, lpd_mask)
    report_corr("top4_pd_prob vs |freq_error| (LPD)", top4_probs, abs_errors, lpd_mask)

    print("\nGPD only:")
    gpd_mask = patient_subtypes == 'gpd'
    report_corr("mean_pd_prob vs |freq_error| (GPD)", mean_probs, abs_errors, gpd_mask)
    report_corr("max_pd_prob vs |freq_error| (GPD)", max_probs, abs_errors, gpd_mask)
    report_corr("top4_pd_prob vs |freq_error| (GPD)", top4_probs, abs_errors, gpd_mask)

    # Step 6: AUC for predicting high vs low error
    print("\n" + "-" * 70)
    print("AUC: Predicting 'high error' (above-median) vs 'low error'")
    print("-" * 70)

    median_error = np.median(abs_errors)
    high_error = (abs_errors > median_error).astype(int)
    print(f"  Median |error|: {median_error:.4f} Hz")
    print(f"  High-error patients: {int(np.sum(high_error))}, Low-error: {int(np.sum(1 - high_error))}")

    # For AUC, we want LOWER PD prob to predict HIGH error, so flip sign
    # Actually: we predict high_error using (1 - pd_prob) as score
    for label, probs in [("mean_pd_prob", mean_probs), ("max_pd_prob", max_probs), ("top4_pd_prob", top4_probs)]:
        auc = compute_auc(high_error, 1.0 - probs)  # flip: low PD prob -> high error
        print(f"  AUC ({label}): {auc:.4f}")

    # Subtype-specific AUC
    for st, mask in [("LPD", lpd_mask), ("GPD", gpd_mask)]:
        if np.sum(mask) < 10:
            continue
        med_st = np.median(abs_errors[mask])
        he_st = (abs_errors[mask] > med_st).astype(int)
        auc_st = compute_auc(he_st, 1.0 - mean_probs[mask])
        print(f"  AUC (mean_pd_prob, {st}): {auc_st:.4f}  (median |error| = {med_st:.4f} Hz)")

    # Step 7: Bin analysis (quartiles)
    print("\n" + "-" * 70)
    print("Quartile Analysis: Mean |Error| by PD Probability Quartile")
    print("-" * 70)

    quartile_edges = np.percentile(mean_probs, [0, 25, 50, 75, 100])
    print(f"  {'Quartile':<20s}  {'Range':<20s}  {'N':>5s}  {'Mean |Error|':>14s}  {'Median |Error|':>14s}")

    for q in range(4):
        lo, hi = quartile_edges[q], quartile_edges[q + 1]
        if q == 3:
            mask = (mean_probs >= lo) & (mean_probs <= hi)
        else:
            mask = (mean_probs >= lo) & (mean_probs < hi)
        q_errors = abs_errors[mask]
        label = f"Q{q+1}"
        range_str = f"[{lo:.3f}, {hi:.3f}]"
        mean_err = float(np.mean(q_errors)) if len(q_errors) > 0 else float('nan')
        med_err = float(np.median(q_errors)) if len(q_errors) > 0 else float('nan')
        print(f"  {label:<20s}  {range_str:<20s}  {int(np.sum(mask)):>5d}  {mean_err:>14.4f}  {med_err:>14.4f}")

    # Step 8: Generate HTML report
    print("\n--- Generating HTML report ---")
    html = generate_html_report(merged, RESULTS_DIR / 'pd_prob_vs_freq_error.html')

    # Inject stats JSON into HTML
    stats_json_str = json.dumps(stats_for_html)
    html = html.replace('__STATS_JSON__', stats_json_str)

    output_path = RESULTS_DIR / 'pd_prob_vs_freq_error.html'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), 'w') as f:
        f.write(html)
    print(f"  HTML report saved to {output_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
