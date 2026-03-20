"""
Generate misclassification review viewer for subtype, laterality, and frequency.

Runs LOPO classification/regression, identifies errors, generates EEG images,
and builds an HTML viewer where MW can review and update labels.

Must run with: conda run -n foe python code/generate_misclass_reviewer.py
"""

import sys, json, base64, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import detrend, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
BASE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    load_dataset, _build_segment_level_data, ALL_FEATURE_COLS,
    FEATURE_COLS, LATERALITY_FEATURE_COLS, LEFT_INDICES, RIGHT_INDICES,
)

DATA = BASE / 'data'
EEG_DIR = DATA / 'eeg'
OUT_DIR = BASE / 'results'

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]


# ── LOPO prediction helpers ──────────────────────────────────────────

def _ridge_logistic(X_train, y_train, X_test, alpha=1.0, n_iter=5):
    """Ridge logistic regression. Returns predicted probabilities for test set."""
    X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])
    w = np.zeros(X_train_b.shape[1])
    for _ in range(n_iter):
        logits = np.clip(X_train_b @ w, -10, 10)
        p = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-6, 1 - 1e-6)
        W_diag = p * (1 - p)
        z = logits + (y_train - p) / W_diag
        W_X = X_train_b * W_diag[:, None]
        try:
            w = np.linalg.solve(W_X.T @ X_train_b + alpha * np.eye(X_train_b.shape[1]),
                                W_X.T @ z)
        except np.linalg.LinAlgError:
            break
    test_logits = np.clip(X_test_b @ w, -10, 10)
    return 1.0 / (1.0 + np.exp(-test_logits))


def _impute_features(X_train, X_test):
    """Impute NaN with training median, in-place."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        finite = np.isfinite(col)
        med = np.median(col[finite]) if np.any(finite) else 0.0
        X_train[~finite, j] = med
        X_test[~np.isfinite(X_test[:, j]), j] = med
    return X_train, X_test


def get_subtype_predictions(dataset):
    """Run LOPO subtype classification using RandomForest. Returns dict: patient_id -> {prob, pred, true}."""
    df = dataset['df']
    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_pids = np.array(seg_pids)
    pid_to_subtype = dict(zip(df['patient_id'], df['subtype']))
    seg_subtypes = np.array([1 if pid_to_subtype.get(p) == 'gpd' else 0 for p in seg_pids])

    is_gpd_idx = ALL_FEATURE_COLS.index('is_gpd')
    feat_mask = [i for i in range(len(ALL_FEATURE_COLS)) if i != is_gpd_idx]

    results = {}
    for pat in df['patient_id'].values:
        test_mask = seg_pids == pat
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
            continue
        X_train, X_test = _impute_features(
            seg_features[train_mask][:, feat_mask],
            seg_features[test_mask][:, feat_mask])
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
        clf.fit(X_train, seg_subtypes[train_mask])
        probs = clf.predict_proba(X_test)[:, 1]
        avg_prob = float(np.mean(probs))
        true_label = 'gpd' if pid_to_subtype[pat] == 'gpd' else 'lpd'
        pred_label = 'gpd' if avg_prob >= 0.5 else 'lpd'
        results[pat] = {
            'prob_gpd': round(avg_prob, 4),
            'pred': pred_label,
            'true': true_label,
            'correct': pred_label == true_label,
        }
    return results


def get_laterality_predictions(dataset):
    """Run LOPO laterality classification using GradientBoosting. Returns dict: patient_id -> {prob, pred, true}."""
    df = dataset['df']
    lat_map = {'left': 0, 'right': 1}
    eligible = df[df['laterality'].isin(['left', 'right'])].copy()
    if len(eligible) < 10:
        return {}

    eligible_pids = set(eligible['patient_id'].values)
    pid_to_lat = dict(zip(eligible['patient_id'], eligible['laterality']))

    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_pids = np.array(seg_pids)

    lat_feat_indices = [ALL_FEATURE_COLS.index(c) for c in LATERALITY_FEATURE_COLS]
    freq_feat_indices = [ALL_FEATURE_COLS.index(c) for c in ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']]
    feat_indices = lat_feat_indices + freq_feat_indices

    eligible_mask = np.array([p in eligible_pids for p in seg_pids])
    seg_pids_e = seg_pids[eligible_mask]
    seg_features_e = seg_features[eligible_mask]
    seg_lat = np.array([lat_map.get(pid_to_lat.get(p, ''), -1) for p in seg_pids_e])

    results = {}
    for pat in eligible['patient_id'].values:
        test_mask = seg_pids_e == pat
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
            continue
        X_train, X_test = _impute_features(
            seg_features_e[train_mask][:, feat_indices],
            seg_features_e[test_mask][:, feat_indices])
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        clf.fit(X_train, seg_lat[train_mask])
        probs = clf.predict_proba(X_test)[:, 1]
        avg_prob = float(np.mean(probs))
        true_label = pid_to_lat[pat]
        pred_label = 'right' if avg_prob >= 0.5 else 'left'
        results[pat] = {
            'prob_right': round(avg_prob, 4),
            'pred': pred_label,
            'true': true_label,
            'correct': pred_label == true_label,
        }
    return results


def get_frequency_predictions(dataset):
    """Run LOPO frequency regression using Ridge (alpha=1.0) on all 9 features.

    Returns dict: patient_id -> {pred_freq, gold_freq, error, subtype}
    Marks as 'error' any case where abs(pred - gold) > 0.5 Hz.
    """
    df = dataset['df']
    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_pids = np.array(seg_pids)
    # seg_labels is gold_standard_freq per segment

    pid_to_subtype = dict(zip(df['patient_id'], df['subtype']))

    results = {}
    for pat in df['patient_id'].values:
        test_mask = seg_pids == pat
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
            continue

        X_train, X_test = _impute_features(
            seg_features[train_mask], seg_features[test_mask])
        y_train = seg_labels[train_mask]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Average segment-level predictions to patient-level
        pred_freq = float(np.mean(preds))
        gold_freq = float(seg_labels[test_mask][0])
        error = pred_freq - gold_freq

        results[pat] = {
            'pred_freq': round(pred_freq, 3),
            'gold_freq': round(gold_freq, 3),
            'error': round(error, 3),
            'subtype': pid_to_subtype.get(pat, '?'),
            'is_error': abs(error) > 0.5,
        }
    return results


# ── EEG image generation ─────────────────────────────────────────────

def generate_eeg_jpeg(seg_bi, fs, patient_id, title_extra=''):
    """Generate a clean EEG JPEG image using morgoth-viewer style rendering.

    Follows the morgoth_viewer.m / viewer_widget.py approach:
    - Fixed µV scaling: z_scale = 0.01 (100 µV = 1 channel unit)
    - Clip at ±300 µV before scaling
    - Uniform channel spacing (offset = channel position index)
    - Black traces on white, with L/R hemisphere coloring
    - 1-second vertical gridlines
    """
    seg_bi = seg_bi.astype(np.float64)
    if seg_bi.shape[0] > seg_bi.shape[1]:
        seg_bi = seg_bi.T
    seg_bi = np.nan_to_num(seg_bi, nan=0.0, posinf=0.0, neginf=0.0)
    n_channels, n_samples = seg_bi.shape
    time_vec = np.linspace(0, n_samples / fs, n_samples)

    # Lowpass at 20 Hz
    nyq = fs / 2.0
    if nyq > 20:
        b, a = butter(4, 20.0 / nyq, btype='low')
        for i in range(n_channels):
            try:
                seg_bi[i, :] = filtfilt(b, a, seg_bi[i, :])
            except ValueError:
                pass

    # Detrend
    for i in range(n_channels):
        seg_bi[i, :] = detrend(seg_bi[i, :], type='linear')

    # Fixed scaling (matching morgoth-viewer)
    z_scale = 0.01     # 100 µV = 1 unit of vertical space
    clip_uv = 300.0    # clip at ±300 µV

    # Build display list with blank spacer channels between groups
    # Groups: temporal L [0:4], temporal R [4:8], parasagittal L [8:12],
    #         parasagittal R [12:16], midline [16:18]
    GROUP_BREAKS = {4, 8, 12, 16}  # insert spacer before these indices
    display_channels = []  # list of (channel_index_or_None, channel_name)
    for i in range(n_channels):
        if i in GROUP_BREAKS:
            display_channels.append((None, ''))  # spacer
        display_channels.append((i, BIPOLAR_CHANNELS[i]))
    n_display = len(display_channels)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw each channel as an offset trace
    yticks = []
    ytick_labels = []
    for di in range(n_display):
        ch_idx, ch_name = display_channels[di]
        # Position: top channel at n_display, bottom at 1
        offset = float(n_display - di)
        yticks.append(offset)
        ytick_labels.append(ch_name)

        if ch_idx is None:
            continue  # spacer — no trace drawn

        # Clip then scale (morgoth style)
        clipped = np.clip(seg_bi[ch_idx, :], -clip_uv, clip_uv)
        scaled = z_scale * clipped + offset
        ax.plot(time_vec, scaled, color='black', linewidth=0.6, clip_on=True)

    # Y-axis: channel labels
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=7.5, fontfamily='monospace')
    ax.tick_params(axis='y', length=0, pad=4)

    # Fixed Y range
    ax.set_ylim(0, n_display + 1)

    # X-axis
    ax.set_xlim(0, n_samples / fs)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel('Time (seconds)', fontsize=9)
    ax.tick_params(axis='x', labelsize=7)

    # 1-second vertical gridlines (dashed, like morgoth)
    ax.grid(True, axis='x', alpha=0.25, linewidth=0.5, linestyle='--')
    ax.grid(False, axis='y')

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['left'].set_color('#999')
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['bottom'].set_color('#999')

    # Title
    title = f'{patient_id}'
    if title_extra:
        title += f'  {title_extra}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.045)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100, pil_kwargs={'quality': 70})
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── HTML viewer builder ──────────────────────────────────────────────

def build_reviewer(subtype_errors, lat_errors, all_subtype, all_lat, image_data, all_freq=None):
    """Build the misclassification review HTML."""
    if all_freq is None:
        all_freq = []

    html = """<!DOCTYPE html>
<html>
<head>
<title>Misclassification Reviewer</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #1a1a1a; color: #eee; font-family: 'Consolas', 'Monaco', monospace; }

  #header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #222; flex-wrap: wrap; gap: 8px;
  }
  #header-left { display: flex; align-items: center; gap: 12px; }
  #header-right { display: flex; align-items: center; gap: 12px; font-size: 13px; }

  .tab-bar { display: flex; gap: 0; background: #222; border-bottom: 2px solid #444; }
  .tab-btn {
    padding: 10px 24px; cursor: pointer; font-family: monospace; font-size: 14px;
    font-weight: bold; border: none; background: #2a2a2a; color: #888;
    border-bottom: 3px solid transparent; transition: all 0.15s;
  }
  .tab-btn:hover { color: #ccc; background: #333; }
  .tab-btn.active { color: #44cc88; border-bottom-color: #44cc88; background: #1a2a1a; }
  .tab-btn .badge {
    background: #ff4444; color: #fff; border-radius: 10px; padding: 1px 7px;
    font-size: 11px; margin-left: 6px;
  }

  .key { background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; }

  #info-panel {
    background: #2a2a2a; padding: 12px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333;
  }
  .info-badge {
    padding: 6px 16px; border-radius: 6px; font-size: 14px; font-weight: bold;
  }
  .badge-lpd { background: #5a2020; color: #ff8888; }
  .badge-gpd { background: #20205a; color: #8888ff; }
  .badge-left { background: #5a2020; color: #ff8888; }
  .badge-right { background: #20205a; color: #8888ff; }
  .badge-error { background: #5a2020; color: #ff6644; }
  .badge-correct { background: #1a3a1a; color: #44cc88; }
  .info-item { font-size: 14px; color: #bbb; }
  .info-item strong { color: #eee; }

  #verdict-banner {
    padding: 10px 16px; font-size: 16px; font-weight: bold; text-align: center;
    border-bottom: 2px solid #444;
  }
  .verdict-error { background: #3a1515; color: #ff6644; }
  .verdict-correct { background: #153a15; color: #44cc88; }

  #annotation-panel {
    background: #2a2a2a; padding: 14px 16px;
    display: flex; align-items: center; justify-content: center;
    gap: 12px; flex-wrap: wrap; border-bottom: 2px solid #444;
  }

  .anno-btn {
    padding: 14px 32px; border: 3px solid #555; border-radius: 10px;
    background: #444; color: #eee; cursor: pointer;
    font-family: monospace; font-size: 18px; font-weight: bold;
    min-width: 120px; text-align: center; transition: all 0.15s;
  }
  .anno-btn:hover { filter: brightness(1.2); }
  .anno-btn.selected { box-shadow: 0 0 15px; }

  .btn-lpd { background: #5a2020; border-color: #cc3333; color: #ff8888; }
  .btn-lpd.selected { background: #8a2020; border-color: #ff4444; box-shadow: 0 0 15px #ff4444; }
  .btn-gpd { background: #20205a; border-color: #3333cc; color: #8888ff; }
  .btn-gpd.selected { background: #20208a; border-color: #4444ff; box-shadow: 0 0 15px #4444ff; }
  .btn-neither { background: #3a3a3a; border-color: #888; color: #ccc; }
  .btn-neither.selected { background: #555; border-color: #aaa; box-shadow: 0 0 15px #888; }

  .btn-left { background: #5a2020; border-color: #cc3333; color: #ff8888; }
  .btn-left.selected { background: #8a2020; border-color: #ff4444; box-shadow: 0 0 15px #ff4444; }
  .btn-right { background: #20205a; border-color: #3333cc; color: #8888ff; }
  .btn-right.selected { background: #20208a; border-color: #4444ff; box-shadow: 0 0 15px #4444ff; }
  .btn-bilateral { background: #3a3a20; border-color: #aaaa33; color: #dddd66; }
  .btn-bilateral.selected { background: #5a5a20; border-color: #dddd44; box-shadow: 0 0 15px #dddd44; }
  .btn-ok { background: #1a3a1a; border-color: #44cc88; color: #44cc88; }
  .btn-ok.selected { background: #2a5a2a; border-color: #66ff88; box-shadow: 0 0 15px #44cc88; }
  .btn-skip { background: #3a3a3a; border-color: #888; color: #ccc; }
  .btn-skip.selected { background: #555; border-color: #aaa; box-shadow: 0 0 15px #888; }

  .freq-input-wrap {
    display: flex; align-items: center; gap: 6px;
  }
  .freq-input {
    width: 80px; padding: 10px 8px; font-size: 18px; font-family: monospace;
    font-weight: bold; text-align: center; border: 3px solid #cc8833;
    border-radius: 10px; background: #3a2a1a; color: #ffaa44;
  }
  .freq-input:focus { outline: none; border-color: #ffaa44; box-shadow: 0 0 10px #cc8833; }
  .freq-submit {
    padding: 10px 16px; border: 3px solid #cc8833; border-radius: 10px;
    background: #3a2a1a; color: #ffaa44; cursor: pointer;
    font-family: monospace; font-size: 16px; font-weight: bold;
  }
  .freq-submit:hover { filter: brightness(1.2); }

  #img-container { text-align: center; padding: 8px; }
  #img-container img { max-width: 100%; max-height: calc(100vh - 360px); }

  #save-status { color: #44cc44; font-size: 13px; }

  #shortcuts {
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333;
  }

  .export-btn {
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }
  .export-btn:hover { background: #3a4a3a; }

  select { font-size: 13px; padding: 3px 6px; background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; }
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff9800;">Misclassification Reviewer</span>
    <select id="filter-mode" onchange="filterChanged()">
      <option value="errors">Errors only</option>
      <option value="all">All cases</option>
      <option value="reviewed">Reviewed</option>
      <option value="unreviewed">Unreviewed errors</option>
    </select>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="exportCSV()">Export CSV</button>
    <span id="save-status"></span>
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" id="tab-subtype" onclick="switchTab('subtype')">
    Subtype (LPD/GPD) <span class="badge" id="subtype-err-count">0</span>
  </button>
  <button class="tab-btn" id="tab-lat" onclick="switchTab('laterality')">
    Laterality (L/R) <span class="badge" id="lat-err-count">0</span>
  </button>
  <button class="tab-btn" id="tab-freq" onclick="switchTab('frequency')">
    Frequency (&gt;0.5 Hz off) <span class="badge" id="freq-err-count">0</span>
  </button>
</div>

<div id="info-panel">
  <span class="info-badge" id="info-true-badge">--</span>
  <span class="info-item">Patient: <strong id="patient-id">--</strong></span>
  <span class="info-item">True: <strong id="true-label">--</strong></span>
  <span class="info-item">Predicted: <strong id="pred-label">--</strong></span>
  <span class="info-item">Confidence: <strong id="confidence">--</strong></span>
</div>

<div id="verdict-banner" class="verdict-error">--</div>

<div id="annotation-panel"></div>

<div id="img-container">
  <img id="viewer" src="" alt="Loading..." />
</div>

<div id="shortcuts">
  <span class="key">&larr;</span> / <span class="key">&rarr;</span> navigate &nbsp;&nbsp;
  <span class="key">O</span> Confirm label OK &nbsp;&nbsp;
  Subtype: <span class="key">2</span> LPD <span class="key">3</span> GPD <span class="key">0</span> Neither &nbsp;&nbsp;
  Laterality: <span class="key">1</span> Left <span class="key">2</span> Right <span class="key">3</span> Bilateral &nbsp;&nbsp;
  Frequency: <span class="key">O</span> OK <span class="key">S</span> Skip &nbsp;&nbsp;
  <span class="key">E</span> Export CSV
</div>

<script>
const SUBTYPE_DATA = SUBTYPE_PLACEHOLDER;
const LAT_DATA = LAT_PLACEHOLDER;
const FREQ_DATA = FREQ_PLACEHOLDER;
const IMAGE_DATA = IMAGE_PLACEHOLDER;

let currentTab = 'subtype';
let corrections = {};
let filteredItems = [];
let idx = 0;

// Load saved corrections
try {
  corrections = JSON.parse(localStorage.getItem('misclass_corrections') || '{}');
} catch(e) { corrections = {}; }

function saveCorrections() {
  localStorage.setItem('misclass_corrections', JSON.stringify(corrections));
}

function getCurrentData() {
  if (currentTab === 'subtype') return SUBTYPE_DATA;
  if (currentTab === 'laterality') return LAT_DATA;
  return FREQ_DATA;
}

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tab-subtype').classList.toggle('active', tab === 'subtype');
  document.getElementById('tab-lat').classList.toggle('active', tab === 'laterality');
  document.getElementById('tab-freq').classList.toggle('active', tab === 'frequency');
  filterChanged();
}

function filterChanged() {
  const mode = document.getElementById('filter-mode').value;
  const data = getCurrentData();
  filteredItems = data.filter(item => {
    if (mode === 'errors') return !item.correct;
    if (mode === 'reviewed') return corrections[currentTab + '_' + item.patient_id] != null;
    if (mode === 'unreviewed') return !item.correct && corrections[currentTab + '_' + item.patient_id] == null;
    return true;
  });
  // Sort errors first, then by confidence/error magnitude
  filteredItems.sort((a, b) => {
    if (a.correct !== b.correct) return a.correct ? 1 : -1;
    if (currentTab === 'frequency') {
      // Largest errors first
      return b.confidence - a.confidence;
    }
    return Math.abs(a.confidence - 0.5) - Math.abs(b.confidence - 0.5);
  });
  idx = 0;
  show();
}

function show() {
  if (filteredItems.length === 0) {
    document.getElementById('viewer').src = '';
    document.getElementById('counter').textContent = '0 / 0';
    document.getElementById('patient-id').textContent = '--';
    document.getElementById('verdict-banner').textContent = 'No cases to show';
    document.getElementById('verdict-banner').className = 'verdict-correct';
    document.getElementById('annotation-panel').innerHTML = '';
    return;
  }
  idx = Math.max(0, Math.min(idx, filteredItems.length - 1));
  const item = filteredItems[idx];

  // Image
  const b64 = IMAGE_DATA[item.patient_id];
  if (b64) {
    document.getElementById('viewer').src = 'data:image/jpeg;base64,' + b64;
  } else {
    document.getElementById('viewer').src = '';
  }

  // Info
  document.getElementById('patient-id').textContent = item.patient_id;
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;

  const infoPanel = document.getElementById('info-panel');
  const badge = document.getElementById('info-true-badge');

  if (currentTab === 'frequency') {
    document.getElementById('true-label').textContent = item.gold_freq.toFixed(2) + ' Hz';
    document.getElementById('pred-label').textContent = item.pred_freq.toFixed(2) + ' Hz';
    document.getElementById('confidence').textContent = item.error.toFixed(3) + ' Hz error';
    badge.textContent = item.subtype.toUpperCase();
    badge.className = 'info-badge badge-' + item.subtype.toLowerCase();
  } else {
    document.getElementById('true-label').textContent = item.true_label.toUpperCase();
    document.getElementById('pred-label').textContent = item.pred_label.toUpperCase();
    document.getElementById('confidence').textContent = item.confidence.toFixed(3);
    badge.textContent = item.true_label.toUpperCase();
    badge.className = 'info-badge badge-' + item.true_label.toLowerCase();
  }

  // Verdict banner
  const banner = document.getElementById('verdict-banner');
  const corrKey = currentTab + '_' + item.patient_id;
  const correction = corrections[corrKey];

  if (correction) {
    if (correction === 'ok' || correction === 'skip') {
      banner.textContent = 'REVIEWED: ' + (correction === 'ok' ? 'Gold standard confirmed OK' : 'SKIPPED');
      banner.className = correction === 'ok' ? 'verdict-correct' : 'verdict-error';
    } else if (currentTab === 'frequency') {
      banner.textContent = 'REVIEWED: Frequency corrected to ' + correction + ' Hz';
      banner.className = 'verdict-error';
    } else {
      banner.textContent = 'REVIEWED: Label changed to ' + correction.toUpperCase();
      banner.className = 'verdict-error';
    }
  } else if (item.correct) {
    banner.textContent = currentTab === 'frequency'
      ? 'OK: Prediction within 0.5 Hz (error=' + item.error.toFixed(2) + ' Hz)'
      : 'CORRECT: Algorithm agrees with label';
    banner.className = 'verdict-correct';
  } else {
    if (currentTab === 'frequency') {
      banner.textContent = 'ERROR: Gold=' + item.gold_freq.toFixed(2) + ' Hz, Predicted=' + item.pred_freq.toFixed(2) + ' Hz (off by ' + Math.abs(item.error).toFixed(2) + ' Hz)';
    } else {
      banner.textContent = 'ERROR: Label=' + item.true_label.toUpperCase() + ' but predicted=' + item.pred_label.toUpperCase();
    }
    banner.className = 'verdict-error';
  }

  // Annotation buttons
  buildButtons(item, correction);
}

function buildButtons(item, correction) {
  const panel = document.getElementById('annotation-panel');

  if (currentTab === 'frequency') {
    const okSel = correction === 'ok' ? ' selected' : '';
    const skipSel = correction === 'skip' ? ' selected' : '';
    const isCustom = correction && correction !== 'ok' && correction !== 'skip';
    let html = '';
    html += `<button class="anno-btn btn-ok${okSel}" onclick="annotate('ok')">OK (keep ${item.gold_freq.toFixed(2)} Hz)<br><span class="key">O</span></button>`;
    html += `<div class="freq-input-wrap">`;
    html += `<input type="text" class="freq-input" id="freq-correction" placeholder="Hz" value="${isCustom ? correction : ''}" />`;
    html += `<button class="freq-submit" onclick="submitFreqCorrection()">Set<br>freq</button>`;
    html += `</div>`;
    html += `<button class="anno-btn btn-skip${skipSel}" onclick="annotate('skip')">SKIP<br><span class="key">S</span></button>`;
    panel.innerHTML = html;
    return;
  }

  let options;
  if (currentTab === 'subtype') {
    options = [
      { key: 'ok', label: 'OK (keep ' + item.true_label.toUpperCase() + ')', cls: 'btn-ok', shortcut: 'O' },
      { key: 'lpd', label: 'LPD', cls: 'btn-lpd', shortcut: '2' },
      { key: 'gpd', label: 'GPD', cls: 'btn-gpd', shortcut: '3' },
      { key: 'neither', label: 'NEITHER', cls: 'btn-neither', shortcut: '0' },
    ];
  } else {
    options = [
      { key: 'ok', label: 'OK (keep ' + item.true_label.toUpperCase() + ')', cls: 'btn-ok', shortcut: 'O' },
      { key: 'left', label: 'LEFT', cls: 'btn-left', shortcut: '1' },
      { key: 'right', label: 'RIGHT', cls: 'btn-right', shortcut: '2' },
      { key: 'bilateral', label: 'BILATERAL', cls: 'btn-bilateral', shortcut: '3' },
      { key: 'neither', label: 'NEITHER', cls: 'btn-neither', shortcut: '0' },
    ];
  }

  let html = '';
  for (const opt of options) {
    const sel = correction === opt.key ? ' selected' : '';
    html += `<button class="anno-btn ${opt.cls}${sel}" onclick="annotate('${opt.key}')">${opt.label}<br><span class="key">${opt.shortcut}</span></button>`;
  }
  panel.innerHTML = html;
}

function submitFreqCorrection() {
  const val = document.getElementById('freq-correction').value.trim();
  if (!val || isNaN(parseFloat(val))) {
    document.getElementById('save-status').textContent = 'Invalid frequency!';
    setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 1500);
    return;
  }
  annotate(val);
}

function annotate(value) {
  if (filteredItems.length === 0) return;
  const item = filteredItems[idx];
  const corrKey = currentTab + '_' + item.patient_id;
  corrections[corrKey] = value;
  saveCorrections();

  document.getElementById('save-status').textContent = 'Saved: ' + value.toUpperCase();
  setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 1000);

  show();

  // Auto-advance after brief delay
  if (idx < filteredItems.length - 1) {
    setTimeout(() => { idx++; show(); }, 300);
  }
}

function exportCSV() {
  // Export corrections for subtype, laterality, and frequency
  const rows = ['task,patient_id,original_label,new_label'];
  for (const [key, val] of Object.entries(corrections)) {
    if (val === 'ok' || val === 'skip') continue;  // No change
    const parts = key.split('_');
    const task = parts[0];
    const pid = parts.slice(1).join('_');
    // Find original label
    let data;
    if (task === 'subtype') data = SUBTYPE_DATA;
    else if (task === 'laterality') data = LAT_DATA;
    else data = FREQ_DATA;
    const item = data.find(d => d.patient_id === pid);
    const origLabel = item ? item.true_label : '?';
    rows.push([task, pid, origLabel, val].join(','));
  }
  const blob = new Blob([rows.join('\\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'label_corrections.csv';
  a.click();
}

document.addEventListener('keydown', e => {
  if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight') { idx = Math.min(idx + 1, filteredItems.length - 1); show(); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { idx = Math.max(idx - 1, 0); show(); e.preventDefault(); }
  else if (e.key === 'o' || e.key === 'O') { annotate('ok'); e.preventDefault(); }
  else if ((e.key === 's' || e.key === 'S') && currentTab === 'frequency') { annotate('skip'); e.preventDefault(); }
  else if (e.key === '0' && currentTab !== 'frequency') {
    annotate('neither'); e.preventDefault();
  }
  else if (e.key === '1' && currentTab !== 'frequency') {
    annotate(currentTab === 'subtype' ? 'ok' : 'left'); e.preventDefault();
  }
  else if (e.key === '2' && currentTab !== 'frequency') {
    annotate(currentTab === 'subtype' ? 'lpd' : 'right'); e.preventDefault();
  }
  else if (e.key === '3' && currentTab !== 'frequency') {
    annotate(currentTab === 'subtype' ? 'gpd' : 'bilateral'); e.preventDefault();
  }
  else if (e.key === 'e' || e.key === 'E') { exportCSV(); e.preventDefault(); }
  else if (e.key === 'Enter' && currentTab === 'frequency') { submitFreqCorrection(); e.preventDefault(); }
});

// Init
document.getElementById('subtype-err-count').textContent = SUBTYPE_DATA.filter(d => !d.correct).length;
document.getElementById('lat-err-count').textContent = LAT_DATA.filter(d => !d.correct).length;
document.getElementById('freq-err-count').textContent = FREQ_DATA.filter(d => !d.correct).length;
filterChanged();
</script>
</body>
</html>"""

    return html


def main():
    print("=" * 60)
    print("Misclassification Reviewer Generator")
    print("=" * 60)

    # Step 1: Load dataset and run predictions
    print("\n--- Step 1: Loading dataset ---")
    dataset = load_dataset(verbose=True)
    df = dataset['df']
    segments = dataset['segments']

    print("\n--- Step 2: Running LOPO subtype classification ---")
    subtype_preds = get_subtype_predictions(dataset)
    n_subtype_err = sum(1 for v in subtype_preds.values() if not v['correct'])
    print(f"  Subtype: {len(subtype_preds)} patients, {n_subtype_err} errors")

    print("\n--- Step 3: Running LOPO laterality classification ---")
    lat_preds = get_laterality_predictions(dataset)
    n_lat_err = sum(1 for v in lat_preds.values() if not v['correct'])
    print(f"  Laterality: {len(lat_preds)} patients, {n_lat_err} errors")

    print("\n--- Step 3b: Running LOPO frequency regression ---")
    freq_preds = get_frequency_predictions(dataset)
    n_freq_err = sum(1 for v in freq_preds.values() if v['is_error'])
    print(f"  Frequency: {len(freq_preds)} patients, {n_freq_err} errors (>0.5 Hz off)")

    # Step 4: Generate images for all error cases (and nearby borderline correct ones)
    print("\n--- Step 4: Generating EEG images ---")

    # Collect all patients we need images for
    need_images = set()
    for pid, v in subtype_preds.items():
        need_images.add(pid)
    for pid, v in lat_preds.items():
        need_images.add(pid)
    for pid, v in freq_preds.items():
        need_images.add(pid)

    from pd_pointiness_acf import fcn_getBanana

    image_data = {}
    n_generated = 0
    for pid in sorted(need_images):
        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue
        seg = pat_segs[0]  # Use first segment for image
        try:
            jpeg_bytes = generate_eeg_jpeg(seg, 200, pid)
            image_data[pid] = base64.b64encode(jpeg_bytes).decode('ascii')
            n_generated += 1
            if n_generated % 50 == 0:
                print(f"  Generated {n_generated} images...")
        except Exception as e:
            print(f"  IMG FAILED: {pid}: {e}")

    print(f"  Total images: {n_generated}")

    # Step 5: Build JSON data for viewer
    print("\n--- Step 5: Building HTML viewer ---")

    subtype_json = []
    for pid, v in sorted(subtype_preds.items()):
        subtype_json.append({
            'patient_id': pid,
            'true_label': v['true'],
            'pred_label': v['pred'],
            'confidence': v['prob_gpd'],
            'correct': v['correct'],
        })

    lat_json = []
    for pid, v in sorted(lat_preds.items()):
        lat_json.append({
            'patient_id': pid,
            'true_label': v['true'],
            'pred_label': v['pred'],
            'confidence': v['prob_right'],
            'correct': v['correct'],
        })

    freq_json = []
    for pid, v in sorted(freq_preds.items()):
        freq_json.append({
            'patient_id': pid,
            'true_label': str(v['gold_freq']),
            'pred_label': str(v['pred_freq']),
            'confidence': abs(v['error']),
            'correct': not v['is_error'],
            'gold_freq': v['gold_freq'],
            'pred_freq': v['pred_freq'],
            'error': v['error'],
            'subtype': v['subtype'],
        })

    html = build_reviewer([], [], subtype_json, lat_json, image_data, freq_json)

    # Replace placeholders
    html = html.replace('SUBTYPE_PLACEHOLDER', json.dumps(subtype_json))
    html = html.replace('LAT_PLACEHOLDER', json.dumps(lat_json))
    html = html.replace('FREQ_PLACEHOLDER', json.dumps(freq_json))
    html = html.replace('IMAGE_PLACEHOLDER', json.dumps(image_data))

    output_path = OUT_DIR / 'misclass_reviewer.html'
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Subtype errors: {n_subtype_err} / {len(subtype_preds)}")
    print(f"  Laterality errors: {n_lat_err} / {len(lat_preds)}")
    print(f"  Frequency errors (>0.5 Hz): {n_freq_err} / {len(freq_preds)}")
    print(f"  Images generated: {n_generated}")
    print(f"  Viewer: {output_path} ({size_mb:.1f} MB)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
