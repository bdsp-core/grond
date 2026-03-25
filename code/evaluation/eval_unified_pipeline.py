"""Evaluate the unified PDCharacterizer pipeline on all 4 tasks.

Task 1: Laterality (LPD only) — AUC
Task 2: Spatial localization — Composite (MacF1, Jaccard, AUC)
Task 3: Discharge timing — F1
Task 4: Frequency estimation — Spearman ρ
"""
import sys, time, json, warnings, numpy as np, pandas as pd, scipy.io as sio
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pd_characterizer import PDCharacterizer

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'

REGIONS = ['LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO']


def load_segment(pid, seg_df):
    """Find and load an EEG segment for a patient."""
    mat_file = None
    for _, sr in seg_df[seg_df['patient_id'] == pid].iterrows():
        if (EEG_DIR / sr['mat_file']).exists():
            mat_file = sr['mat_file']
            break
    if not mat_file:
        for sx in ['_seg000.mat', '.mat']:
            if (EEG_DIR / f'{pid}{sx}').exists():
                mat_file = f'{pid}{sx}'
                break
    if not mat_file:
        return None
    try:
        mat = sio.loadmat(str(EEG_DIR / mat_file))
        dk = [k for k in mat if not k.startswith('_')][0]
        seg = mat[dk].astype(np.float64)
        if seg.shape[0] > seg.shape[1]:
            seg = seg.T
        if seg.shape[0] < 18:
            return None
        return seg[:18, :2000]
    except:
        return None


def compute_timing_f1(pred_times, gold_times, tol=0.1):
    if len(pred_times) == 0 and len(gold_times) == 0:
        return 1.0
    if len(pred_times) == 0 or len(gold_times) == 0:
        return 0.0
    pred = np.array(pred_times)
    gold = np.array(gold_times)
    # Precision
    matched = 0
    used = set()
    for p in pred:
        d = np.abs(gold - p)
        best = np.argmin(d)
        if d[best] <= tol and best not in used:
            matched += 1
            used.add(best)
    prec = matched / len(pred)
    # Recall
    matched2 = 0
    used2 = set()
    for g in gold:
        d = np.abs(pred - g)
        best = np.argmin(d)
        if d[best] <= tol and best not in used2:
            matched2 += 1
            used2.add(best)
    rec = matched2 / len(gold)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def main():
    print("=" * 70)
    print("  Unified PDCharacterizer — Full Pipeline Evaluation")
    print("=" * 70)

    charzer = PDCharacterizer()

    pat = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    pat['patient_id'] = pat['patient_id'].astype(str)
    pat['gold_standard_freq'] = pd.to_numeric(pat['gold_standard_freq'], errors='coerce')

    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)

    ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))

    with open(str(LABELS_DIR / 'discharge_times.json')) as f:
        dt_gt = json.load(f)

    # ── Task 1: Laterality ──
    print("\n--- Task 1: Laterality (LPD only) ---")
    lpd = pat[(pat['subtype'] == 'lpd') & (pat['excluded'] != True)].copy()
    lpd['laterality_clean'] = lpd['laterality'].apply(
        lambda x: x if x in ['left', 'right'] else None)
    lpd_labeled = lpd[lpd['laterality_clean'].notna()]
    print(f"  LPD with laterality labels: {len(lpd_labeled)}")

    lat_true, lat_scores = [], []
    n_lat = 0
    for _, row in lpd_labeled.iterrows():
        pid = row['patient_id']
        seg = load_segment(pid, seg_df)
        if seg is None:
            continue
        result = charzer.characterize(seg, subtype='lpd')
        true_lat = 1 if row['laterality_clean'] == 'right' else 0
        # Use confidence as score (positive = right)
        probs = np.array(result['channel_probs'])
        from pd_characterizer import LEFT_INDICES, RIGHT_INDICES
        score = np.mean(probs[RIGHT_INDICES]) - np.mean(probs[LEFT_INDICES])
        lat_true.append(true_lat)
        lat_scores.append(score)
        n_lat += 1
        if n_lat % 50 == 0:
            print(f"    {n_lat}/{len(lpd_labeled)}...")

    lat_auc = roc_auc_score(lat_true, lat_scores) if len(set(lat_true)) > 1 else float('nan')
    print(f"  Laterality AUC: {lat_auc:.4f} (n={n_lat})")

    # ── Task 2: Spatial Localization ──
    print("\n--- Task 2: Spatial Localization ---")
    # Load spatial annotations
    spatial_ann = ann[ann['spatial_channels'].notna() & (ann['spatial_channels'] != '')]
    if len(spatial_ann) > 0:
        # Group by segment, get majority vote regions
        spatial_segs = {}
        for _, row in spatial_ann.iterrows():
            sid = str(row['segment_id'])
            pid = str(row['patient_id'])
            regions = set(str(row['spatial_channels']).split())
            if sid not in spatial_segs:
                spatial_segs[sid] = {'pid': pid, 'votes': {}}
            for r in REGIONS:
                if r not in spatial_segs[sid]['votes']:
                    spatial_segs[sid]['votes'][r] = 0
                if r in regions:
                    spatial_segs[sid]['votes'][r] += 1

        # Majority vote
        gold_spatial = {}
        for sid, info in spatial_segs.items():
            gold_regions = set()
            for r, count in info['votes'].items():
                if count >= 2:
                    gold_regions.add(r)
            if gold_regions:
                gold_spatial[sid] = {'pid': info['pid'], 'regions': gold_regions}

        print(f"  Spatial gold standard: {len(gold_spatial)} segments")

        all_gold, all_pred, all_scores = [], [], []
        n_sp = 0
        for sid, info in gold_spatial.items():
            pid = info['pid']
            subtype_row = pat[pat['patient_id'] == pid]
            if len(subtype_row) == 0:
                continue
            subtype = subtype_row.iloc[0]['subtype']
            if subtype not in ['lpd', 'gpd']:
                continue
            seg = load_segment(pid, seg_df)
            if seg is None:
                continue
            result = charzer.characterize(seg, subtype=subtype)
            gold_vec = [1 if r in info['regions'] else 0 for r in REGIONS]
            pred_vec = [1 if r in result['regions'] else 0 for r in REGIONS]
            score_vec = [result['region_scores'].get(r, 0) for r in REGIONS]
            all_gold.append(gold_vec)
            all_pred.append(pred_vec)
            all_scores.append(score_vec)
            n_sp += 1
            if n_sp % 50 == 0:
                print(f"    {n_sp}/{len(gold_spatial)}...")

        if all_gold:
            gold_flat = np.array(all_gold).ravel()
            pred_flat = np.array(all_pred).ravel()
            score_flat = np.array(all_scores).ravel()
            macro_f1 = f1_score(gold_flat, pred_flat, average='macro')
            micro_f1 = f1_score(gold_flat, pred_flat, average='micro')
            # Jaccard
            jaccards = []
            for g, p in zip(all_gold, all_pred):
                g_set = set(i for i, v in enumerate(g) if v)
                p_set = set(i for i, v in enumerate(p) if v)
                if g_set or p_set:
                    jaccards.append(len(g_set & p_set) / len(g_set | p_set))
            jaccard = np.mean(jaccards) if jaccards else 0
            mean_auc = roc_auc_score(gold_flat, score_flat)
            composite = 0.30 * macro_f1 + 0.25 * jaccard + 0.25 * mean_auc + 0.20 * micro_f1
            print(f"  MacroF1={macro_f1:.3f} MicroF1={micro_f1:.3f} Jaccard={jaccard:.3f} AUC={mean_auc:.3f} Composite={composite:.3f} (n={n_sp})")
    else:
        print("  No spatial annotations found")

    # ── Task 3: Discharge Timing ──
    print("\n--- Task 3: Discharge Timing ---")
    f1_scores = []
    n_dt = 0
    for pid, gt_entry in dt_gt.items():
        gold_times = gt_entry.get('times', gt_entry.get('global_times', []))
        if len(gold_times) < 2:
            continue
        row = pat[pat['patient_id'] == pid]
        if len(row) == 0:
            continue
        subtype = row.iloc[0]['subtype']
        if subtype not in ['lpd', 'gpd']:
            continue
        seg = load_segment(pid, seg_df)
        if seg is None:
            continue
        result = charzer.characterize(seg, subtype=subtype)
        f1 = compute_timing_f1(result['discharge_times'], gold_times)
        f1_scores.append(f1)
        n_dt += 1
        if n_dt % 100 == 0:
            print(f"    {n_dt} patients, mean F1={np.mean(f1_scores):.4f}")

    print(f"  Discharge Timing F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f} (n={n_dt})")

    # ── Task 4: Frequency ──
    print("\n--- Task 4: Frequency Estimation ---")
    pred_freqs, gold_freqs = [], []
    n_fr = 0
    pd_with_freq = pat[
        (pat['subtype'].isin(['lpd', 'gpd'])) &
        (pat['excluded'] != True) &
        (pat['gold_standard_freq'].notna()) &
        (pat['gold_standard_freq'] > 0)
    ]
    for _, row in pd_with_freq.iterrows():
        pid = row['patient_id']
        seg = load_segment(pid, seg_df)
        if seg is None:
            continue
        result = charzer.characterize(seg, subtype=row['subtype'])
        if np.isfinite(result['frequency']) and result['frequency'] > 0:
            pred_freqs.append(result['frequency'])
            gold_freqs.append(row['gold_standard_freq'])
            n_fr += 1
        if n_fr % 100 == 0 and n_fr > 0:
            rho, _ = spearmanr(pred_freqs, gold_freqs)
            print(f"    {n_fr} patients, Spearman rho={rho:.4f}")
        if n_fr >= 500:  # Cap for speed
            break

    rho_freq, _ = spearmanr(pred_freqs, gold_freqs)
    mae = np.mean(np.abs(np.array(pred_freqs) - np.array(gold_freqs)))
    print(f"  Frequency Spearman rho: {rho_freq:.4f}, MAE: {mae:.3f} Hz (n={n_fr})")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  UNIFIED PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Task 1 — Laterality:    AUC = {lat_auc:.4f} (n={n_lat})")
    if all_gold:
        print(f"  Task 2 — Spatial:       Composite = {composite:.3f} (n={n_sp})")
    print(f"  Task 3 — Timing:        F1 = {np.mean(f1_scores):.4f} (n={n_dt})")
    print(f"  Task 4 — Frequency:     rho = {rho_freq:.4f} (n={n_fr})")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
