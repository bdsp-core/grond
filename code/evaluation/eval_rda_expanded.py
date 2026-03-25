"""Evaluate RDA methods on expanded dataset (original 3-rater + MW LRDA).

Includes Alexandra's rda1b_fft (FOOOF-based) and all newer methods.
Reports results for ALL cases, and broken down by LRDA vs GRDA.
"""
import sys, json, warnings, numpy as np, pandas as pd, scipy.io as sio
from pathlib import Path
from scipy.signal import detrend, butter, filtfilt, sosfiltfilt
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings('ignore')
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from rda_optimization_harness import (
    variance_explained_search, acf_frequency, fft_peak_frequency,
    fooof_frequency, classify_laterality,
    LEFT_CHANNELS, RIGHT_CHANNELS, FS,
)
from pd_pointiness_acf import fcn_getBanana
from mne.filter import notch_filter, filter_data

DATA_DIR = CODE_DIR.parent / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'

print("=" * 70)
print("  RDA Evaluation on Expanded Dataset")
print("  (Original 3-rater + MW LRDA, all cases)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# Load ALL labeled cases
# ══════════════════════════════════════════════════════════════════════

# --- Original 3-rater dataset (LRDA + GRDA with expert freq labels) ---
df_ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
df_seg = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
df_seg['patient_id'] = df_seg['patient_id'].astype(str)
rda_seg = df_seg[df_seg['subtype'].isin(['grda', 'lrda'])].copy()

core_ann = df_ann[
    (df_ann['segment_id'].isin(set(rda_seg['segment_id']))) &
    (df_ann['skipped'] == False) &
    (df_ann['rater'].isin(['LB', 'PH', 'SZ']))
]
rc = core_ann.groupby('segment_id')['rater'].nunique()
multi_ids = set(rc[rc >= 2].index)

orig_records = []
for sid in sorted(multi_ids):
    info = rda_seg[rda_seg['segment_id'] == sid]
    if len(info) == 0:
        continue
    si = info.iloc[0]
    sa = df_ann[(df_ann['segment_id'] == sid) &
                (df_ann['rater'].isin(['LB', 'PH', 'SZ', 'MW']))]
    fv = [r['frequency_hz'] for _, r in sa.iterrows()
          if pd.notna(r['frequency_hz']) and r['frequency_hz'] > 0]
    gf = float(np.median(fv)) if fv else np.nan
    if pd.isna(gf) or gf <= 0:
        continue
    orig_records.append({
        'patient_id': str(si['patient_id']),
        'segment_id': sid,
        'subtype': si['subtype'],
        'gold_freq': gf,
        'mat_file': si['mat_file'],
        'source': 'original_3rater',
    })

# --- MW LRDA labels ---
with open(str(LABELS_DIR / 'lrda_labels_mw.json')) as f:
    mw_lrda = json.load(f)
mw_records = []
for pid, e in mw_lrda.items():
    if e.get('rejected', False):
        continue
    freq = e.get('selected_freq')
    if freq is None or freq <= 0:
        continue
    sid = e.get('segment_id', f'{pid}_seg000')
    mw_records.append({
        'patient_id': str(pid),
        'segment_id': sid,
        'subtype': 'lrda',
        'gold_freq': float(freq),
        'gold_laterality': e.get('laterality'),
        'mat_file': f'{sid}.mat',
        'source': 'mw_lrda',
    })

orig_pids = set(r['patient_id'] for r in orig_records)
mw_pids = set(r['patient_id'] for r in mw_records)

print(f"\nOriginal 3-rater: {len(orig_records)} segments, {len(orig_pids)} patients")
n_orig_lrda = sum(1 for r in orig_records if r['subtype'] == 'lrda')
n_orig_grda = sum(1 for r in orig_records if r['subtype'] == 'grda')
print(f"  LRDA: {n_orig_lrda}, GRDA: {n_orig_grda}")
print(f"MW LRDA: {len(mw_records)} cases, {len(mw_pids)} patients")
print(f"Overlap: {len(orig_pids & mw_pids)} patients")

# Combine ALL: keep all original records AND all MW records
# For overlapping patients, keep BOTH (original may be different segments)
# but deduplicate by segment_id
all_records = orig_records + mw_records
seen_sids = set()
deduped = []
for r in all_records:
    if r['segment_id'] not in seen_sids:
        deduped.append(r)
        seen_sids.add(r['segment_id'])
    elif r['source'] == 'mw_lrda':
        # MW label takes precedence for same segment
        deduped = [x for x in deduped if x['segment_id'] != r['segment_id']]
        deduped.append(r)

df = pd.DataFrame(deduped)
print(f"\nCombined (all): {len(df)} segments, {df['patient_id'].nunique()} patients")
print(f"  GRDA: {(df['subtype']=='grda').sum()}, LRDA: {(df['subtype']=='lrda').sum()}")

# ══════════════════════════════════════════════════════════════════════
# Load EEG segments
# ══════════════════════════════════════════════════════════════════════
print("\nLoading EEG segments...")
segs_raw = []   # monopolar/raw for Alexandra's method
segs_bi = []    # preprocessed bipolar for newer methods
valid_idx = []

for idx, row in df.iterrows():
    mp = EEG_DIR / row['mat_file']
    if not mp.exists():
        mp = EEG_DIR / f"{row['patient_id']}_seg000.mat"
        if not mp.exists():
            continue
    try:
        mat = sio.loadmat(str(mp))
        dk = [k for k in mat if not k.startswith('_')][0]
        s = mat[dk].astype(np.float64)
        if s.shape[0] > s.shape[1]:
            s = s.T

        # Keep raw for Alexandra's method
        raw = s[:, :2000].copy()

        # Preprocess for newer methods
        if s.shape[0] >= 20:
            sb = np.array(fcn_getBanana(s[:20, :2000]), dtype=np.float64)
        elif s.shape[0] == 18:
            sb = s[:18, :2000].copy()
        else:
            continue
        sb = notch_filter(sb, FS, 60, n_jobs=1, verbose='ERROR')
        sb = filter_data(sb, FS, 0.5, 40, n_jobs=1, verbose='ERROR')
        for ch in range(sb.shape[0]):
            sb[ch] = detrend(sb[ch], type='linear')

        segs_raw.append(raw)
        segs_bi.append(sb)
        valid_idx.append(idx)
    except Exception:
        continue

df = df.loc[valid_idx].reset_index(drop=True)
gold = df['gold_freq'].values
subtypes = df['subtype'].values
sources = df['source'].values

print(f"Loaded: {len(df)} segments")
print(f"  GRDA: {(subtypes=='grda').sum()} ({df[subtypes=='grda']['patient_id'].nunique()} patients)")
print(f"  LRDA: {(subtypes=='lrda').sum()} ({df[subtypes=='lrda']['patient_id'].nunique()} patients)")

# ══════════════════════════════════════════════════════════════════════
# Run frequency estimation methods
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FREQUENCY ESTIMATION")
print("=" * 70)


def eval_method(name, preds):
    """Compute metrics for a set of predictions."""
    preds = np.array(preds, dtype=float)
    v = np.isfinite(preds) & np.isfinite(gold)
    if v.sum() < 5:
        return None
    rho, _ = spearmanr(gold[v], preds[v])
    mae = float(np.mean(np.abs(gold[v] - preds[v])))
    return {'name': name, 'rho': rho, 'mae': mae, 'n': int(v.sum()), 'preds': preds}


# --- Alexandra's rda1b_fft (FOOOF-based) ---
print("\nRunning Alexandra's rda1b_fft...")
sys.path.insert(0, str(CODE_DIR / 'rda_detector'))
from rda1b_fft import rda1b_fft

alex_preds = []
alex_types = []
for raw in segs_raw:
    try:
        # rda1b_fft expects monopolar (≥19ch) — it calls fcn_getBanana internally
        if raw.shape[0] >= 19:
            result = rda1b_fft(raw[:19, :2000], FS, 0)
        elif raw.shape[0] == 20:
            result = rda1b_fft(raw[:20, :2000], FS, 0)
        else:
            # 18-channel bipolar: can't use Alexandra's method (needs monopolar)
            alex_preds.append(np.nan)
            alex_types.append(np.nan)
            continue
        f = result.get('event_frequency', np.nan)
        alex_preds.append(float(f) if pd.notna(f) else np.nan)
        t = result.get('type_event', np.nan)
        if pd.notna(t):
            alex_types.append(1 if 'lrda' in str(t).lower() else 0)
        else:
            alex_types.append(np.nan)
    except Exception:
        alex_preds.append(np.nan)
        alex_types.append(np.nan)
alex_types = np.array(alex_types, dtype=float)
r_alex = eval_method('Alexandra rda1b', alex_preds)
if r_alex:
    print(f"  Alexandra: {r_alex['n']}/{len(segs_raw)} cases had valid predictions")

# --- VE Search ---
print("Running VE Search...")
ve_preds = []
for sb in segs_bi:
    try:
        ve_preds.append(variance_explained_search(sb)[0])
    except:
        ve_preds.append(np.nan)
r_ve = eval_method('VE Search', ve_preds)

# --- FFT Peak ---
print("Running FFT Peak...")
fft_preds = []
for sb in segs_bi:
    try:
        r = fft_peak_frequency(sb)
        fft_preds.append(r[0] if isinstance(r, tuple) else r)
    except:
        fft_preds.append(np.nan)
r_fft = eval_method('FFT Peak', fft_preds)

# --- ACF ---
print("Running ACF...")
acf_preds = []
for sb in segs_bi:
    try:
        acf_preds.append(acf_frequency(sb))
    except:
        acf_preds.append(np.nan)
r_acf = eval_method('ACF', acf_preds)

# --- NVO Bandpass (laterality-aware) ---
print("Running NVO Bandpass...")
nvo_preds = []
for sb in segs_bi:
    try:
        freqs = np.arange(0.5, 3.55, 0.05)
        tv = np.array([max(np.var(sb[ch]), 1e-12) for ch in range(18)])
        scores = np.zeros(len(freqs))
        for fi, f in enumerate(freqs):
            lo, hi = max(f - 0.3, 0.1), min(f + 0.3, FS / 2 - 0.1)
            sos = butter(4, [lo, hi], btype='bandpass', fs=FS, output='sos')
            filt = sosfiltfilt(sos, sb, axis=1)
            cv = np.array([np.var(filt[ch]) / tv[ch] for ch in range(18)])
            lv = np.sort(cv[LEFT_CHANNELS])[::-1]
            rv = np.sort(cv[RIGHT_CHANNELS])[::-1]
            scores[fi] = max(np.mean(lv[:3]), np.mean(rv[:3]))
        nvo_preds.append(freqs[np.argmax(scores)])
    except:
        nvo_preds.append(np.nan)
r_nvo = eval_method('NVO Bandpass', nvo_preds)

# --- LP8Hz + FFT ---
print("Running LP8Hz + FFT...")
lp8_preds = []
for sb in segs_bi:
    try:
        b, a = butter(4, 8.0 / (FS / 2), btype='low')
        lp = filtfilt(b, a, sb, axis=1)
        r2 = fft_peak_frequency(lp)
        lp8_preds.append(r2[0] if isinstance(r2, tuple) else r2)
    except:
        lp8_preds.append(np.nan)
r_lp8 = eval_method('LP8Hz + FFT', lp8_preds)

# --- Ensemble median ---
print("Running Ensemble...")
ens_preds = []
for i in range(len(segs_bi)):
    vals = []
    for r in [r_ve, r_fft, r_acf]:
        if r and np.isfinite(r['preds'][i]):
            vals.append(r['preds'][i])
    ens_preds.append(np.median(vals) if vals else np.nan)
r_ens = eval_method('Ensemble (VE+FFT+ACF)', ens_preds)

all_results = [r for r in [r_alex, r_fft, r_lp8, r_nvo, r_ve, r_ens, r_acf] if r]

# ── Print results: ALL, then by subtype ──
def print_table(label, mask):
    g = gold[mask]
    n_lrda = (subtypes[mask] == 'lrda').sum()
    n_grda = (subtypes[mask] == 'grda').sum()
    print(f"\n{label} (N={mask.sum()}, {n_lrda} LRDA + {n_grda} GRDA):")
    print(f"  {'Method':<25} {'Spearman':>9} {'MAE':>7} {'N':>5}")
    print(f"  {'-' * 50}")
    for r in all_results:
        p = r['preds'][mask]
        v = np.isfinite(p) & np.isfinite(g)
        if v.sum() >= 3:
            rho, _ = spearmanr(g[v], p[v])
            mae = np.mean(np.abs(g[v] - p[v]))
            print(f"  {r['name']:<25} {rho:>9.4f} {mae:>7.3f} {v.sum():>5}")

all_mask = np.ones(len(df), dtype=bool)
lrda_mask = subtypes == 'lrda'
grda_mask = subtypes == 'grda'
orig_mask = sources == 'original_3rater'
mw_mask = sources == 'mw_lrda'

print_table("ALL CASES", all_mask)
print_table("LRDA only", lrda_mask)
print_table("GRDA only", grda_mask)
print_table("Original 3-rater only", orig_mask)
print_table("MW LRDA only", mw_mask)

# ══════════════════════════════════════════════════════════════════════
# LRDA vs GRDA Classification
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  LRDA vs GRDA CLASSIFICATION")
print("=" * 70)

gold_bin = (subtypes == 'lrda').astype(int)
n_lrda_total = gold_bin.sum()
n_grda_total = (1 - gold_bin).sum()
print(f"\nN={len(gold_bin)} ({n_lrda_total} LRDA, {n_grda_total} GRDA)")

# Alexandra's method classification (computed during freq estimation above)
v = np.isfinite(alex_types)
if v.sum() > 0:
    p = alex_types[v].astype(int)
    g = gold_bin[v]
    acc = accuracy_score(g, p)
    f1m = f1_score(g, p, average='macro')
    f1l = f1_score(g, p, pos_label=1)
    f1g = f1_score(g, p, pos_label=0)
    sens = np.mean(p[g == 1]) if (g == 1).sum() > 0 else 0
    spec = np.mean(1 - p[g == 0]) if (g == 0).sum() > 0 else 0
    print(f"\nAlexandra rda1b_fft classification:")
    print(f"  Acc={acc:.3f}  F1-macro={f1m:.3f}  F1-LRDA={f1l:.3f}  F1-GRDA={f1g:.3f}  Sens={sens:.3f}  Spec={spec:.3f}  (n={v.sum()})")

# LI-based classification at various thresholds using NVO freq
for method_name, pred_r in [('NVO freq', r_nvo), ('FFT freq', r_fft), ('VE freq', r_ve)]:
    if pred_r is None:
        continue
    li = []
    for i, sb in enumerate(segs_bi):
        try:
            f = pred_r['preds'][i]
            if np.isnan(f):
                f = 1.5
            lo, hi = max(f - 0.3, 0.1), min(f + 0.3, FS / 2 - 0.1)
            sos = butter(4, [lo, hi], btype='bandpass', fs=FS, output='sos')
            filt = sosfiltfilt(sos, sb, axis=1)
            cv = np.array([np.var(filt[ch]) / max(np.var(sb[ch]), 1e-12) for ch in range(18)])
            ls, rs = np.sum(cv[LEFT_CHANNELS]), np.sum(cv[RIGHT_CHANNELS])
            li.append((rs - ls) / max(ls + rs, 1e-12))
        except:
            li.append(0.0)
    li = np.array(li)

    print(f"\n{method_name} → LI threshold:")
    print(f"  {'Thresh':<8} {'Acc':>7} {'F1-mac':>7} {'F1-LRDA':>8} {'F1-GRDA':>8} {'Sens':>6} {'Spec':>6}")
    print(f"  {'-' * 55}")
    for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        p = (np.abs(li) > t).astype(int)
        acc = accuracy_score(gold_bin, p)
        f1m = f1_score(gold_bin, p, average='macro')
        f1l = f1_score(gold_bin, p, pos_label=1)
        f1g = f1_score(gold_bin, p, pos_label=0)
        sens = np.mean(p[gold_bin == 1]) if gold_bin.sum() > 0 else 0
        spec = np.mean(1 - p[gold_bin == 0]) if (1 - gold_bin).sum() > 0 else 0
        print(f"  {t:<8.2f} {acc:>7.3f} {f1m:>7.3f} {f1l:>8.3f} {f1g:>8.3f} {sens:>6.3f} {spec:>6.3f}")

print("\n" + "=" * 70)
