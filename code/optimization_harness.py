"""
Shared evaluation harness for frequency optimization experiments.

Each experimenter agent imports this module, runs its approach on the annotated
dataset, and writes results to results/optimization_runs/<experiment_name>.json.

The dashboard HTML auto-reads these JSON files to show live results.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import copy
import hdf5storage
import scipy.io
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

DATA_DIR = CODE_DIR.parent / 'data' / '_archive' / 'dataset_eeg'
ANN_DIR = CODE_DIR.parent / 'data' / '_archive' / 'annotations'
RESULTS_DIR = CODE_DIR.parent / 'results'
RUNS_DIR = RESULTS_DIR / 'optimization_runs'
RUNS_DIR.mkdir(exist_ok=True)

# Baselines for comparison
BASELINES = {
    'expert_expert_LPD_MAE': 0.594,  # mean of pairwise expert MAEs
    'expert_expert_GPD_MAE': 0.315,
    'method_A_LPD_MAE': 0.537,
    'method_A_GPD_MAE': 0.274,
}


def load_dataset():
    """Load all annotated PD segments with per-expert ratings.

    Returns list of dicts with keys:
        subdir, mat_name, mat_path,
        expert_LB_freq, expert_PH_freq, expert_SZ_freq,
        expert_consensus_freq, expert_consensus_spatial,
        data (np array), fs
    """
    # Load annotations
    records = {}
    for pattern, subdir in [('LPDS', 'lpd'), ('GPDS', 'gpd')]:
        for expert_file in sorted(ANN_DIR.glob(f'{pattern}_*')):
            expert = expert_file.stem.split('_')[1]
            df = pd.read_csv(expert_file)
            for _, row in df.iterrows():
                mat_name = Path(row['files']).stem.replace('_score', '') + '.mat'
                key = (subdir, mat_name)
                if key not in records:
                    records[key] = {'subdir': subdir, 'mat_name': mat_name, 'experts': {}}
                try:
                    freq = float(row['frequency'])
                except (ValueError, TypeError):
                    freq = np.nan
                try:
                    spatial = float(row['spatial'])
                except (ValueError, TypeError):
                    spatial = np.nan
                records[key]['experts'][expert] = {'frequency': freq, 'spatial': spatial}

    # Build dataset
    dataset = []
    for (subdir, mat_name), rec in records.items():
        mat_path = DATA_DIR / subdir / mat_name
        if not mat_path.exists():
            continue

        freqs = [rec['experts'][e]['frequency'] for e in ['LB', 'PH', 'SZ']
                 if e in rec['experts'] and np.isfinite(rec['experts'][e]['frequency'])
                 and rec['experts'][e]['frequency'] > 0]
        spatials = [rec['experts'][e]['spatial'] for e in ['LB', 'PH', 'SZ']
                    if e in rec['experts'] and np.isfinite(rec['experts'][e].get('spatial', np.nan))
                    and rec['experts'][e]['spatial'] > 0]

        if not freqs:
            continue

        entry = {
            'subdir': subdir,
            'mat_name': mat_name,
            'mat_path': str(mat_path),
            'expert_LB_freq': rec['experts'].get('LB', {}).get('frequency', np.nan),
            'expert_PH_freq': rec['experts'].get('PH', {}).get('frequency', np.nan),
            'expert_SZ_freq': rec['experts'].get('SZ', {}).get('frequency', np.nan),
            'expert_consensus_freq': float(np.median(freqs)),
            'expert_consensus_spatial': float(np.median(spatials)) if spatials else np.nan,
        }
        dataset.append(entry)

    return dataset


def load_eeg_data(entry):
    """Load EEG data for a dataset entry. Returns (data, fs) or (None, None)."""
    mat_path = entry['mat_path']
    try:
        try:
            mat = scipy.io.loadmat(mat_path)
        except NotImplementedError:
            mat = hdf5storage.loadmat(mat_path)
        data = mat.get('data_50sec', mat.get('data'))
        if data is None:
            return None, None
        if data.shape[0] > data.shape[1]:
            data = data.T
        return data, 200
    except Exception:
        return None, None


def evaluate_predictions(dataset, predictions, experiment_name):
    """Evaluate frequency predictions against expert consensus.

    Args:
        dataset: list of dicts from load_dataset()
        predictions: dict mapping mat_name -> predicted_frequency
        experiment_name: string name for this experiment

    Returns: results dict and writes JSON to optimization_runs/
    """
    results_by_type = {'lpd': {'expert': [], 'pred': [], 'mat_names': [],
                               'expert_LB': [], 'expert_PH': [], 'expert_SZ': []},
                       'gpd': {'expert': [], 'pred': [], 'mat_names': [],
                               'expert_LB': [], 'expert_PH': [], 'expert_SZ': []}}

    for entry in dataset:
        mat_name = entry['mat_name']
        if mat_name not in predictions:
            continue
        pred = predictions[mat_name]
        if not np.isfinite(pred):
            continue
        expert = entry['expert_consensus_freq']
        if not np.isfinite(expert):
            continue
        sd = entry['subdir']
        results_by_type[sd]['expert'].append(expert)
        results_by_type[sd]['pred'].append(pred)
        results_by_type[sd]['mat_names'].append(mat_name)
        results_by_type[sd]['expert_LB'].append(entry.get('expert_LB_freq', np.nan))
        results_by_type[sd]['expert_PH'].append(entry.get('expert_PH_freq', np.nan))
        results_by_type[sd]['expert_SZ'].append(entry.get('expert_SZ_freq', np.nan))

    metrics = {'experiment': experiment_name, 'timestamp': time.time()}

    for ptype in ['lpd', 'gpd']:
        expert = np.array(results_by_type[ptype]['expert'])
        pred = np.array(results_by_type[ptype]['pred'])
        n = len(expert)

        if n < 3:
            metrics[f'{ptype}_n'] = n
            metrics[f'{ptype}_mae'] = np.nan
            metrics[f'{ptype}_pearson_r'] = np.nan
            metrics[f'{ptype}_icc'] = np.nan
            metrics[f'{ptype}_pa'] = np.nan
            continue

        mae = float(np.mean(np.abs(pred - expert)))
        try:
            r, _ = pearsonr(pred, expert)
        except:
            r = np.nan
        try:
            rs, _ = spearmanr(pred, expert)
        except:
            rs = np.nan

        # ICC(3,1)
        ratings = np.column_stack([expert, pred])
        grand_mean = np.mean(ratings)
        row_means = np.mean(ratings, axis=1)
        SSR = 2 * np.sum((row_means - grand_mean) ** 2)
        SST = np.sum((ratings - grand_mean) ** 2)
        col_means = np.mean(ratings, axis=0)
        SSC = n * np.sum((col_means - grand_mean) ** 2)
        SSE = SST - SSR - SSC
        MSR = SSR / (n - 1) if n > 1 else 0
        MSE = SSE / (n - 1) if n > 1 else 1
        icc = (MSR - MSE) / (MSR + MSE) if (MSR + MSE) > 0 else 0

        # PA (frequency bins: <1, 1-1.5, 1.5-2, 2-2.5, 2.5-3, >3)
        def freq_bin(f):
            if f < 1: return 0
            elif f <= 1.5: return 1
            elif f <= 2: return 2
            elif f <= 2.5: return 3
            elif f <= 3: return 4
            else: return 5
        pa = float(np.mean([freq_bin(e) == freq_bin(p) for e, p in zip(expert, pred)])) * 100

        # Pooled algorithm-vs-each-expert Spearman (fairer comparison to expert-expert)
        algo_all_x, algo_all_y = [], []
        expert_LB = np.array(results_by_type[ptype]['expert_LB'])
        expert_PH = np.array(results_by_type[ptype]['expert_PH'])
        expert_SZ = np.array(results_by_type[ptype]['expert_SZ'])
        for expert_arr in [expert_LB, expert_PH, expert_SZ]:
            for i in range(len(pred)):
                ev = expert_arr[i]
                if np.isfinite(ev) and ev > 0:
                    algo_all_x.append(ev)
                    algo_all_y.append(pred[i])
        try:
            rs_pooled, _ = spearmanr(algo_all_x, algo_all_y)
        except:
            rs_pooled = np.nan

        metrics[f'{ptype}_n'] = int(n)
        metrics[f'{ptype}_mae'] = round(mae, 4)
        metrics[f'{ptype}_pearson_r'] = round(float(r), 4)
        metrics[f'{ptype}_spearman_r'] = round(float(rs), 4)
        metrics[f'{ptype}_spearman_pooled'] = round(float(rs_pooled), 4)
        metrics[f'{ptype}_n_pooled'] = len(algo_all_x)
        metrics[f'{ptype}_icc'] = round(float(icc), 4)
        metrics[f'{ptype}_pa'] = round(pa, 1)

        # Store raw data for scatter plots
        metrics[f'{ptype}_expert_vals'] = [round(v, 3) for v in expert.tolist()]
        metrics[f'{ptype}_pred_vals'] = [round(v, 3) for v in pred.tolist()]
        # Store per-expert values for scatter overlay
        metrics[f'{ptype}_expert_LB'] = [round(v, 3) if np.isfinite(v) else None for v in expert_LB.tolist()]
        metrics[f'{ptype}_expert_PH'] = [round(v, 3) if np.isfinite(v) else None for v in expert_PH.tolist()]
        metrics[f'{ptype}_expert_SZ'] = [round(v, 3) if np.isfinite(v) else None for v in expert_SZ.tolist()]

    # Combined scores (use pooled Spearman as primary metric)
    lpd_mae = metrics.get('lpd_mae', np.nan)
    gpd_mae = metrics.get('gpd_mae', np.nan)
    lpd_rs = metrics.get('lpd_spearman_pooled', metrics.get('lpd_spearman_r', np.nan))
    gpd_rs = metrics.get('gpd_spearman_pooled', metrics.get('gpd_spearman_r', np.nan))
    if np.isfinite(lpd_mae) and np.isfinite(gpd_mae):
        metrics['combined_mae'] = round((lpd_mae + gpd_mae) / 2, 4)
    else:
        metrics['combined_mae'] = np.nan
    if np.isfinite(lpd_rs) and np.isfinite(gpd_rs):
        metrics['combined_spearman'] = round((lpd_rs + gpd_rs) / 2, 4)
    else:
        metrics['combined_spearman'] = np.nan

    # Write to file
    out_path = RUNS_DIR / f'{experiment_name}.json'
    with open(str(out_path), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print summary
    print(f'\n{"="*60}')
    print(f'RESULTS: {experiment_name}')
    print(f'{"="*60}')
    print(f'  {"":>15s} {"LPD":>8s} {"GPD":>8s}')
    print(f'  {"N":>15s} {metrics.get("lpd_n","?"):>8} {metrics.get("gpd_n","?"):>8}')
    print(f'  {"MAE (Hz)":>15s} {metrics.get("lpd_mae","?"):>8} {metrics.get("gpd_mae","?"):>8}')
    print(f'  {"Spearman r":>15s} {metrics.get("lpd_spearman_r","?"):>8} {metrics.get("gpd_spearman_r","?"):>8}')
    print(f'  {"Pearson r":>15s} {metrics.get("lpd_pearson_r","?"):>8} {metrics.get("gpd_pearson_r","?"):>8}')
    print(f'  {"PA (%)":>15s} {metrics.get("lpd_pa","?"):>8} {metrics.get("gpd_pa","?"):>8}')
    print(f'  Combined Spearman: {metrics.get("combined_spearman", "?")}')
    print(f'  Combined MAE: {metrics.get("combined_mae", "?")}')

    return metrics
