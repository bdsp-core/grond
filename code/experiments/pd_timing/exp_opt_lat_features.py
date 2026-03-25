"""
Experiment: New laterality feature engineering for laterality classification.

Computes 4 new asymmetry features from raw EEG segments:
  - ve_asymmetry:             Variance-explained asymmetry (narrowband 0.5-3 Hz / total)
  - peak_amp_asymmetry:       Peak amplitude asymmetry (lowpassed signal)
  - coherence_asymmetry:      Within-hemisphere coherence asymmetry
  - spectral_power_asymmetry: FFT power asymmetry in 0.5-3 Hz band (log ratio)

Experiments (all LOPO laterality classification on LPD left/right):
  lat_ve_asym        - lat_idx + ve_asymmetry, Ridge alpha=1
  lat_peak_asym      - lat_idx + peak_amp_asymmetry, Ridge alpha=1
  lat_all_asym       - all 4 new + 3 original lat features, Ridge alpha=1
  lat_all_asym_a5    - same features, Ridge alpha=5
  lat_all_asym_gbm   - same features, GBM

Usage:
    conda run -n foe python code/exp_opt_lat_features.py
"""

import sys
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks, coherence as scipy_coherence

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    load_dataset, ALL_FEATURE_COLS, LATERALITY_FEATURE_COLS,
    _build_segment_level_data, LEFT_INDICES, RIGHT_INDICES, FS,
)

RESULTS_DIR = PROJECT_DIR / 'results' / 'optimization_runs_v2'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def update_dashboard():
    """Update the dashboard after each experiment."""
    subprocess.run(['python', 'code/update_dashboard_v2.py'], cwd=str(PROJECT_DIR))


# ── New feature computation ──────────────────────────────────────────

def compute_ve_asymmetry(seg):
    """Variance-explained asymmetry: narrowband (0.5-3 Hz) variance / total variance.

    Sum across left channels vs right channels, return (R-L)/(R+L).
    """
    fs = FS
    n_ch = seg.shape[0]

    # Bandpass 0.5-3 Hz
    try:
        b, a = butter(3, [0.5 / (fs / 2), 3.0 / (fs / 2)], btype='band')
    except ValueError:
        return 0.0

    left_ratio, right_ratio = 0.0, 0.0
    for ch in LEFT_INDICES:
        if ch >= n_ch:
            continue
        total_var = np.var(seg[ch])
        if total_var < 1e-12:
            continue
        try:
            narrow = filtfilt(b, a, seg[ch])
        except ValueError:
            continue
        left_ratio += np.var(narrow) / total_var

    for ch in RIGHT_INDICES:
        if ch >= n_ch:
            continue
        total_var = np.var(seg[ch])
        if total_var < 1e-12:
            continue
        try:
            narrow = filtfilt(b, a, seg[ch])
        except ValueError:
            continue
        right_ratio += np.var(narrow) / total_var

    denom = right_ratio + left_ratio
    if denom < 1e-12:
        return 0.0
    return float((right_ratio - left_ratio) / denom)


def compute_peak_amp_asymmetry(seg):
    """Peak amplitude asymmetry from lowpassed signal.

    Find peaks in each channel, compare mean peak amplitude left vs right: (R-L)/(R+L).
    """
    fs = FS
    n_ch = seg.shape[0]

    # Lowpass at 15 Hz
    try:
        b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    except ValueError:
        return 0.0

    left_amps, right_amps = [], []

    for ch_list, amp_list in [(LEFT_INDICES, left_amps), (RIGHT_INDICES, right_amps)]:
        for ch in ch_list:
            if ch >= n_ch:
                continue
            try:
                sig_lp = filtfilt(b_lp, a_lp, seg[ch])
            except ValueError:
                continue
            # Find peaks in absolute value of lowpassed signal
            abs_sig = np.abs(sig_lp)
            pks, props = find_peaks(abs_sig, height=np.max(abs_sig) * 0.3,
                                     distance=int(0.2 * fs))
            if len(pks) >= 2:
                amp_list.append(np.mean(props['peak_heights']))

    left_mean = np.mean(left_amps) if left_amps else 0.0
    right_mean = np.mean(right_amps) if right_amps else 0.0
    denom = right_mean + left_mean
    if denom < 1e-12:
        return 0.0
    return float((right_mean - left_mean) / denom)


def compute_coherence_asymmetry(seg):
    """Within-hemisphere coherence asymmetry.

    Mean coherence among left channels vs right channels (in 0.5-3 Hz). (R-L)/(R+L).
    """
    fs = FS
    n_ch = seg.shape[0]
    nperseg = min(256, seg.shape[1])

    def _mean_within_coherence(ch_indices):
        coh_vals = []
        for i in range(len(ch_indices)):
            for j in range(i + 1, len(ch_indices)):
                ch_a, ch_b = ch_indices[i], ch_indices[j]
                if ch_a >= n_ch or ch_b >= n_ch:
                    continue
                try:
                    f_coh, Cxy = scipy_coherence(seg[ch_a], seg[ch_b],
                                                  fs=fs, nperseg=nperseg)
                    mask = (f_coh >= 0.5) & (f_coh <= 3.0)
                    if np.any(mask):
                        coh_vals.append(np.mean(Cxy[mask]))
                except Exception:
                    continue
        return np.mean(coh_vals) if coh_vals else 0.0

    left_coh = _mean_within_coherence(LEFT_INDICES)
    right_coh = _mean_within_coherence(RIGHT_INDICES)
    denom = right_coh + left_coh
    if denom < 1e-12:
        return 0.0
    return float((right_coh - left_coh) / denom)


def compute_spectral_power_asymmetry(seg):
    """FFT power in 0.5-3 Hz, summed left vs right. Log ratio."""
    fs = FS
    n_ch = seg.shape[0]
    n_samples = seg.shape[1]

    left_power, right_power = 0.0, 0.0
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (freqs >= 0.5) & (freqs <= 3.0)

    if not np.any(mask):
        return 0.0

    for ch in LEFT_INDICES:
        if ch >= n_ch:
            continue
        fft_vals = np.abs(np.fft.rfft(seg[ch] - np.mean(seg[ch]))) ** 2
        left_power += np.sum(fft_vals[mask])

    for ch in RIGHT_INDICES:
        if ch >= n_ch:
            continue
        fft_vals = np.abs(np.fft.rfft(seg[ch] - np.mean(seg[ch]))) ** 2
        right_power += np.sum(fft_vals[mask])

    if left_power < 1e-12 or right_power < 1e-12:
        return 0.0
    return float(np.log(right_power / left_power))


def compute_new_features_for_segment(seg):
    """Compute all 4 new asymmetry features for a single segment."""
    return {
        've_asymmetry': compute_ve_asymmetry(seg),
        'peak_amp_asymmetry': compute_peak_amp_asymmetry(seg),
        'coherence_asymmetry': compute_coherence_asymmetry(seg),
        'spectral_power_asymmetry': compute_spectral_power_asymmetry(seg),
    }


NEW_FEATURE_NAMES = ['ve_asymmetry', 'peak_amp_asymmetry',
                     'coherence_asymmetry', 'spectral_power_asymmetry']


# ── LOPO laterality classification ──────────────────────────────────

def run_laterality_experiment(dataset, experiment_name, feature_names,
                              new_features_by_pid, alpha=1.0, use_gbm=False):
    """LOPO laterality classification with specified features.

    Args:
        dataset: dict from load_dataset()
        experiment_name: string identifier
        feature_names: list of feature names to use (from ALL_FEATURE_COLS + new features)
        new_features_by_pid: dict mapping pid -> list of dicts with new feature values
        alpha: Ridge regularization (ignored for GBM)
        use_gbm: if True, use GBM instead of Ridge logistic
    """
    t0 = time.time()
    print(f"\nRunning laterality experiment: {experiment_name}")

    df = dataset['df']
    features = dataset['features']
    segments = dataset['segments']

    # Filter to LPD with left/right laterality
    lat_map = {'left': 0, 'right': 1}
    eligible = df[df['laterality'].isin(['left', 'right'])].copy()

    if len(eligible) < 10:
        print(f"  Only {len(eligible)} eligible patients — skipping.")
        return {}

    eligible_pids = set(eligible['patient_id'].values)
    pid_to_lat = dict(zip(eligible['patient_id'], eligible['laterality'].map(lat_map)))

    # Build segment-level feature matrix (original + new)
    seg_pids_list = []
    seg_lat_list = []
    seg_feat_rows = []

    for _, row in eligible.iterrows():
        pid = row['patient_id']
        pat_feats = features.get(pid, [])
        pat_new_feats = new_features_by_pid.get(pid, [])
        label = pid_to_lat[pid]

        for i, feat_dict in enumerate(pat_feats):
            row_vals = []
            for fname in feature_names:
                if fname in ALL_FEATURE_COLS:
                    row_vals.append(feat_dict.get(fname, np.nan))
                elif fname in NEW_FEATURE_NAMES:
                    if i < len(pat_new_feats):
                        row_vals.append(pat_new_feats[i].get(fname, 0.0))
                    else:
                        row_vals.append(0.0)
                else:
                    row_vals.append(0.0)
            seg_pids_list.append(pid)
            seg_lat_list.append(label)
            seg_feat_rows.append(row_vals)

    if len(seg_feat_rows) == 0:
        print("  No segments found — skipping.")
        return {}

    seg_pids = np.array(seg_pids_list)
    seg_lat = np.array(seg_lat_list)
    seg_X = np.array(seg_feat_rows, dtype=float)

    unique_patients = eligible['patient_id'].values
    patient_preds = {}

    for pat in unique_patients:
        test_mask = seg_pids == pat
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
            continue

        X_train = seg_X[train_mask].copy()
        y_train = seg_lat[train_mask]
        X_test = seg_X[test_mask].copy()

        # Impute NaN with training median
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            X_test[~np.isfinite(X_test[:, j]), j] = med

        if use_gbm:
            # GBM via sklearn
            from sklearn.ensemble import GradientBoostingClassifier
            gbm = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            gbm.fit(X_train, y_train)
            test_probs = gbm.predict_proba(X_test)[:, 1]
            patient_preds[pat] = float(np.mean(test_probs))
        else:
            # Ridge logistic regression (IRLS)
            X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
            X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])

            w = np.zeros(X_train_b.shape[1])
            for _ in range(5):
                logits = X_train_b @ w
                logits = np.clip(logits, -10, 10)
                p = 1.0 / (1.0 + np.exp(-logits))
                p = np.clip(p, 1e-6, 1 - 1e-6)
                W_diag = p * (1 - p)
                z = logits + (y_train - p) / W_diag
                W_X = X_train_b * W_diag[:, None]
                try:
                    w = np.linalg.solve(
                        W_X.T @ X_train_b + alpha * np.eye(X_train_b.shape[1]),
                        W_X.T @ z)
                except np.linalg.LinAlgError:
                    break

            test_logits = X_test_b @ w
            test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -10, 10)))
            patient_preds[pat] = float(np.mean(test_probs))

    # Aggregate patient-level predictions
    y_true, y_prob = [], []
    for _, row in eligible.iterrows():
        pid = row['patient_id']
        if pid not in patient_preds:
            continue
        y_true.append(pid_to_lat[pid])
        y_prob.append(patient_preds[pid])

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    n = len(y_true)
    accuracy = float(np.mean(y_true == y_pred))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))  # right correct
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))  # left correct
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))  # left predicted right
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))  # right predicted left

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity for right
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity (= left accuracy)
    bal_acc = (sens + spec) / 2

    # AUC
    sorted_idx = np.argsort(-y_prob)
    y_sorted = y_true[sorted_idx]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos > 0 and n_neg > 0:
        tpr_list, fpr_list = [0.0], [0.0]
        tp_cum, fp_cum = 0, 0
        for i in range(len(y_sorted)):
            if y_sorted[i] == 1:
                tp_cum += 1
            else:
                fp_cum += 1
            tpr_list.append(tp_cum / n_pos)
            fpr_list.append(fp_cum / n_neg)
        auc = float(np.trapz(tpr_list, fpr_list))
    else:
        auc = np.nan

    metrics = {
        'experiment': experiment_name,
        'task': 'laterality_classification',
        'timestamp': time.time(),
        'n_patients': n,
        'n_left': int(np.sum(y_true == 0)),
        'n_right': int(np.sum(y_true == 1)),
        'features_used': feature_names,
        'model': 'gbm' if use_gbm else f'ridge_logistic_alpha{alpha}',
        'accuracy': round(accuracy, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'sensitivity_right': round(sens, 4),
        'specificity_left': round(spec, 4),
        'auc': round(float(auc), 4) if np.isfinite(auc) else None,
        'confusion_matrix': {'tp_right': tp, 'tn_left': tn, 'fp': fp, 'fn': fn},
        'pred_probs': [round(v, 4) for v in y_prob.tolist()],
        'true_labels': y_true.tolist(),
    }

    out_path = RESULTS_DIR / f'{experiment_name}.json'
    with open(str(out_path), 'w') as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"LATERALITY CLASSIFICATION: {experiment_name}  ({elapsed:.1f}s)")
    print(f"{'='*72}")
    print(f"  N={n} (left={metrics['n_left']}, right={metrics['n_right']})")
    print(f"  Features: {feature_names}")
    print(f"  Model: {metrics['model']}")
    print(f"  Accuracy:          {accuracy:.3f}")
    print(f"  Balanced accuracy: {bal_acc:.3f}")
    print(f"  Sens (right):      {sens:.3f}")
    print(f"  Spec (left):       {spec:.3f}")
    if np.isfinite(auc):
        print(f"  AUC:               {auc:.3f}")
    else:
        print(f"  AUC:               N/A")
    print(f"  Confusion: TP(R)={tp} TN(L)={tn} FP={fp} FN={fn}")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*72}")

    return metrics


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 72)
    print("Laterality Feature Engineering Experiments")
    print("=" * 72)

    # Load dataset once
    dataset = load_dataset(verbose=True)

    # Compute new features for all patients
    print("\nComputing new asymmetry features for all segments...")
    t0 = time.time()
    new_features_by_pid = {}
    n_segs_computed = 0

    for pid, seg_list in dataset['segments'].items():
        pid_new_feats = []
        for seg in seg_list:
            try:
                nf = compute_new_features_for_segment(seg)
            except Exception as e:
                nf = {k: 0.0 for k in NEW_FEATURE_NAMES}
            pid_new_feats.append(nf)
            n_segs_computed += 1
        new_features_by_pid[pid] = pid_new_feats

        if n_segs_computed % 200 == 0:
            print(f"  Computed {n_segs_computed} segments ({time.time() - t0:.0f}s)")

    print(f"  Done: {n_segs_computed} segments in {time.time() - t0:.1f}s")

    # ── Experiment 1: lat_ve_asym ────────────────────────────────────
    run_laterality_experiment(
        dataset, 'lat_ve_asym',
        feature_names=['lat_idx', 've_asymmetry'],
        new_features_by_pid=new_features_by_pid,
        alpha=1.0,
    )
    update_dashboard()

    # ── Experiment 2: lat_peak_asym ──────────────────────────────────
    run_laterality_experiment(
        dataset, 'lat_peak_asym',
        feature_names=['lat_idx', 'peak_amp_asymmetry'],
        new_features_by_pid=new_features_by_pid,
        alpha=1.0,
    )
    update_dashboard()

    # ── Experiment 3: lat_all_asym (all new + original lat, alpha=1) ─
    run_laterality_experiment(
        dataset, 'lat_all_asym',
        feature_names=list(LATERALITY_FEATURE_COLS) + NEW_FEATURE_NAMES,
        new_features_by_pid=new_features_by_pid,
        alpha=1.0,
    )
    update_dashboard()

    # ── Experiment 4: lat_all_asym_a5 (same features, alpha=5) ──────
    run_laterality_experiment(
        dataset, 'lat_all_asym_a5',
        feature_names=list(LATERALITY_FEATURE_COLS) + NEW_FEATURE_NAMES,
        new_features_by_pid=new_features_by_pid,
        alpha=5.0,
    )
    update_dashboard()

    # ── Experiment 5: lat_all_asym_gbm ──────────────────────────────
    run_laterality_experiment(
        dataset, 'lat_all_asym_gbm',
        feature_names=list(LATERALITY_FEATURE_COLS) + NEW_FEATURE_NAMES,
        new_features_by_pid=new_features_by_pid,
        use_gbm=True,
    )
    update_dashboard()

    print("\n\nAll laterality feature experiments complete.")
