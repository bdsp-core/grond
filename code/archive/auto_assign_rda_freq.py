"""Auto-assign RDA frequencies using M3_HilbertCV for all unlabeled cases."""
import sys, time, numpy as np, pandas as pd, scipy.io as sio, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, hilbert, detrend

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pd_pointiness_acf import fcn_getBanana
from mne.filter import notch_filter, filter_data

FS = 200
LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])
PROJECT = Path(__file__).resolve().parent.parent
EEG_DIR = PROJECT / 'data' / 'eeg'


def hilbert_freq(seg_bi):
    sos1 = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos1, seg_bi, axis=1)
    sos2 = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
    seg_nb = sosfiltfilt(sos2, seg_f, axis=1)
    dp = np.var(seg_nb, axis=1)
    lt = LEFT_CHS[np.argsort(dp[LEFT_CHS])[::-1][:3]]
    rt = RIGHT_CHS[np.argsort(dp[RIGHT_CHS])[::-1][:3]]
    top = lt if np.mean(dp[lt]) > np.mean(dp[rt]) else rt
    freqs = []
    for ch in top:
        s = seg_nb[ch]
        if np.std(s) < 1e-10:
            continue
        ip = np.unwrap(np.angle(hilbert(s)))
        iff = np.diff(ip) * FS / (2 * np.pi)
        v = iff[(iff > 0.3) & (iff < 4.0)]
        if len(v) >= 20:
            freqs.append(float(np.median(v)))
    return float(np.median(freqs)) if freqs else np.nan


def load_and_preprocess(mat_file):
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    dk = [k for k in mat if not k.startswith('_')][0]
    s = mat[dk].astype(np.float64)
    if s.shape[0] > s.shape[1]:
        s = s.T
    nc = s.shape[0]
    if nc >= 20:
        sb = np.array(fcn_getBanana(s[:20, :2000]), dtype=np.float64)
    elif nc == 19:
        sb = np.array(fcn_getBanana(s[:19, :2000]), dtype=np.float64)
    elif nc == 18:
        sb = s[:18, :2000].astype(np.float64)
    else:
        return None
    sb = notch_filter(sb, FS, 60, n_jobs=1, verbose='ERROR')
    sb = filter_data(sb, FS, 0.5, 40, n_jobs=1, verbose='ERROR')
    for c in range(sb.shape[0]):
        sb[c] = detrend(sb[c], type='linear')
    return sb


def main():
    pat = pd.read_csv(str(PROJECT / 'data/labels/patients.csv'))
    pat['patient_id'] = pat['patient_id'].astype(str)
    pat['gold_standard_freq'] = pd.to_numeric(pat['gold_standard_freq'], errors='coerce')

    seg_df = pd.read_csv(str(PROJECT / 'data/labels/segments.csv'))
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)

    rda = pat[(pat['subtype'].isin(['lrda', 'grda'])) & (pat['excluded'] != True)]
    no_freq = rda[~(rda['gold_standard_freq'].notna() & (rda['gold_standard_freq'] > 0))]
    print(f"To process: {len(no_freq)} RDA cases without frequency")

    t0 = time.time()
    n_done, n_fail = 0, 0

    for i, (idx, row) in enumerate(no_freq.iterrows()):
        pid = str(row['patient_id'])

        # Find mat file
        mf = None
        for _, sr in seg_df[seg_df['patient_id'] == pid].iterrows():
            if (EEG_DIR / sr['mat_file']).exists():
                mf = sr['mat_file']
                break
        if not mf:
            for sx in ['_seg000.mat', '.mat']:
                if (EEG_DIR / f'{pid}{sx}').exists():
                    mf = f'{pid}{sx}'
                    break
        if not mf:
            n_fail += 1
            continue

        try:
            sb = load_and_preprocess(mf)
            if sb is None or sb.shape[0] != 18:
                n_fail += 1
                continue
            f = hilbert_freq(sb)
            if np.isfinite(f) and 0.25 <= f <= 4.0:
                pat.loc[idx, 'gold_standard_freq'] = round(f, 2)
                n_done += 1
            else:
                n_fail += 1
        except Exception:
            n_fail += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(no_freq)} done={n_done} fail={n_fail} ({time.time()-t0:.0f}s)")

    pat.to_csv(str(PROJECT / 'data/labels/patients.csv'), index=False)
    print(f"\nDone ({time.time()-t0:.0f}s): {n_done} estimated, {n_fail} failed")

    # Verify
    p2 = pd.read_csv(str(PROJECT / 'data/labels/patients.csv'))
    p2['patient_id'] = p2['patient_id'].astype(str)
    p2['gold_standard_freq'] = pd.to_numeric(p2['gold_standard_freq'], errors='coerce')
    r2 = p2[p2['subtype'].isin(['lrda', 'grda']) & (p2['excluded'] != True)]
    h = r2['gold_standard_freq'].notna() & (r2['gold_standard_freq'] > 0)
    lrda_h = (r2[h]['subtype'] == 'lrda').sum()
    grda_h = (r2[h]['subtype'] == 'grda').sum()
    lrda_t = (r2['subtype'] == 'lrda').sum()
    grda_t = (r2['subtype'] == 'grda').sum()
    print(f"Final: {h.sum()}/{len(r2)} (LRDA: {lrda_h}/{lrda_t}, GRDA: {grda_h}/{grda_t})")


if __name__ == '__main__':
    main()
