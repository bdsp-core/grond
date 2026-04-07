"""V3 Feature Extraction — comprehensive feature families for LRDA vs GRDA.

Feature families:
  1. Focality / spatial concentration
  2. Homologous channel pairs (PLV, coherence, power ratio)
  3. Connectivity / synchrony
  4. Rhythmicity / morphology (inc. Hilbert frequency estimation)
  5. Waveform shape
  6. Time-frequency stability
  7. Propagation / lag
  8. Per-channel features
  9. Previous round cached scores
"""
import json
import numpy as np
from scipy.signal import (welch, hilbert, butter, sosfiltfilt, coherence as sig_coherence,
                          find_peaks)
from scipy.stats import kurtosis, skew, pearsonr
from numpy.fft import fft, ifft
from pathlib import Path

FS = 200
LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])
MIDLINE_CHS = np.array([16, 17])

# Homologous channel pairs (left_idx, right_idx)
HOMOLOGOUS_PAIRS = [
    (0, 4),   # Fp1-F7 ↔ Fp2-F8  (frontal-temporal anterior)
    (1, 5),   # F7-T3 ↔ F8-T4    (anterior temporal)
    (2, 6),   # T3-T5 ↔ T4-T6    (mid temporal)
    (3, 7),   # T5-O1 ↔ T6-O2    (posterior temporal-occipital)
    (8, 12),  # Fp1-F3 ↔ Fp2-F4  (frontal parasagittal)
    (9, 13),  # F3-C3 ↔ F4-C4    (fronto-central)
    (10, 14), # C3-P3 ↔ C4-P4    (central-parietal)
    (11, 15), # P3-O1 ↔ P4-O2    (parietal-occipital)
]

# Left and right chains (anterior → posterior)
LEFT_CHAIN = [0, 1, 2, 3]   # temporal chain
RIGHT_CHAIN = [4, 5, 6, 7]
LEFT_PARA = [8, 9, 10, 11]  # parasagittal chain
RIGHT_PARA = [12, 13, 14, 15]

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'lateralization_contest_v2' / '_cache'


def _bp(seg, lo=0.5, hi=4.0):
    sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
    return sosfiltfilt(sos, seg, axis=1)


def _gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def _entropy(x):
    x = np.abs(x)
    s = x.sum()
    if s == 0:
        return 0.0
    p = x / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _plv(sig1, sig2):
    """Phase-locking value between two signals."""
    p1 = np.angle(hilbert(sig1))
    p2 = np.angle(hilbert(sig2))
    return float(np.abs(np.mean(np.exp(1j * (p1 - p2)))))


def _hilbert_freq_cv(sig):
    """Hilbert instantaneous frequency: returns (median_freq, cv, q_score)."""
    if np.std(sig) < 1e-10:
        return 0.0, 1.0, 0.0
    analytic = hilbert(sig)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) * FS / (2.0 * np.pi)
    mask = (inst_freq > 0.3) & (inst_freq < 4.0)
    inst_freq_valid = inst_freq[mask]
    if len(inst_freq_valid) < 20:
        return 0.0, 1.0, 0.0
    med_f = float(np.median(inst_freq_valid))
    cv = float(np.std(inst_freq_valid) / max(med_f, 1e-6))
    q = max(0.0, 1.0 - 2.0 * cv)
    return med_f, cv, q


# ═══════════════════════════════════════════════════════════════
# Feature Family 1: Focality / Spatial Concentration
# ═══════════════════════════════════════════════════════════════

def feat_focality(seg):
    seg_f = _bp(seg)
    ch_power = np.array([np.var(seg_f[ch]) for ch in range(18)])
    total = ch_power.sum()
    if total < 1e-12:
        return {f'foc_{k}': 0.0 for k in ['entropy', 'gini', 'max_frac', 'top2_frac',
                'top3_frac', 'spread', 'n_active', 'left_frac', 'right_frac',
                'hemi_imbalance', 'within_left_var', 'within_right_var',
                'within_var_ratio']}

    p = ch_power / total
    feats = {}
    feats['foc_entropy'] = _entropy(ch_power)
    feats['foc_gini'] = _gini(ch_power)
    feats['foc_max_frac'] = float(np.max(p))
    sorted_p = np.sort(p)[::-1]
    feats['foc_top2_frac'] = float(sorted_p[:2].sum())
    feats['foc_top3_frac'] = float(sorted_p[:3].sum())
    feats['foc_n_active'] = float(np.sum(ch_power > 0.1 * np.max(ch_power)))

    # Center-of-mass and spread (using channel index as proxy for position)
    max_ch = np.argmax(ch_power)
    feats['foc_spread'] = float(np.sqrt(np.average((np.arange(18) - max_ch) ** 2, weights=p)))

    # Hemispheric fractions
    left_total = ch_power[LEFT_CHS].sum()
    right_total = ch_power[RIGHT_CHS].sum()
    feats['foc_left_frac'] = float(left_total / total)
    feats['foc_right_frac'] = float(right_total / total)
    feats['foc_hemi_imbalance'] = abs(left_total - right_total) / (left_total + right_total + 1e-12)

    # Within-hemisphere variance of channel power
    feats['foc_within_left_var'] = float(np.var(ch_power[LEFT_CHS]))
    feats['foc_within_right_var'] = float(np.var(ch_power[RIGHT_CHS]))
    feats['foc_within_var_ratio'] = abs(feats['foc_within_left_var'] - feats['foc_within_right_var']) / (
        feats['foc_within_left_var'] + feats['foc_within_right_var'] + 1e-12)

    return feats


# ═══════════════════════════════════════════════════════════════
# Feature Family 2: Homologous Channel Pairs
# ═══════════════════════════════════════════════════════════════

def feat_homologous(seg):
    seg_f = _bp(seg)
    feats = {}

    corrs, plvs, power_ratios, freq_diffs = [], [], [], []

    for i, (l_ch, r_ch) in enumerate(HOMOLOGOUS_PAIRS):
        l_sig, r_sig = seg_f[l_ch], seg_f[r_ch]

        # Correlation
        if np.std(l_sig) > 1e-12 and np.std(r_sig) > 1e-12:
            c = float(pearsonr(l_sig, r_sig)[0])
        else:
            c = 0.0
        corrs.append(c)
        feats[f'hom_corr_{i}'] = c

        # PLV
        plv = _plv(l_sig, r_sig)
        plvs.append(plv)
        feats[f'hom_plv_{i}'] = plv

        # Power ratio
        lp, rp = np.var(l_sig), np.var(r_sig)
        pr = abs(lp - rp) / (lp + rp + 1e-12)
        power_ratios.append(pr)
        feats[f'hom_power_ratio_{i}'] = pr

        # Delta peak frequency difference
        for sig, side in [(l_sig, 'l'), (r_sig, 'r')]:
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= 0.5) & (f <= 3.5)
            if delta.any() and pxx[delta].sum() > 0:
                feats[f'hom_peakf_{side}_{i}'] = float(f[delta][np.argmax(pxx[delta])])
            else:
                feats[f'hom_peakf_{side}_{i}'] = 0.0
        freq_diffs.append(abs(feats[f'hom_peakf_l_{i}'] - feats[f'hom_peakf_r_{i}']))

    # Summary stats across pairs
    feats['hom_corr_mean'] = float(np.mean(corrs))
    feats['hom_corr_min'] = float(np.min(corrs))
    feats['hom_corr_std'] = float(np.std(corrs))
    feats['hom_plv_mean'] = float(np.mean(plvs))
    feats['hom_plv_min'] = float(np.min(plvs))
    feats['hom_plv_std'] = float(np.std(plvs))
    feats['hom_power_ratio_mean'] = float(np.mean(power_ratios))
    feats['hom_power_ratio_max'] = float(np.max(power_ratios))
    feats['hom_power_ratio_std'] = float(np.std(power_ratios))
    feats['hom_freq_diff_mean'] = float(np.mean(freq_diffs))
    feats['hom_freq_diff_max'] = float(np.max(freq_diffs))
    feats['hom_n_asym_pairs'] = float(np.sum(np.array(power_ratios) > 0.3))

    return feats


# ═══════════════════════════════════════════════════════════════
# Feature Family 3: Connectivity / Synchrony
# ═══════════════════════════════════════════════════════════════

def feat_connectivity(seg):
    seg_f = _bp(seg)
    feats = {}

    # Intra-hemisphere PLV
    for side, chs in [('L', LEFT_CHS), ('R', RIGHT_CHS)]:
        plvs = []
        phases = np.array([np.angle(hilbert(seg_f[ch])) for ch in chs])
        for i in range(len(chs)):
            for j in range(i + 1, len(chs)):
                plv = float(np.abs(np.mean(np.exp(1j * (phases[i] - phases[j])))))
                plvs.append(plv)
        feats[f'conn_{side}_plv_mean'] = float(np.mean(plvs))
        feats[f'conn_{side}_plv_std'] = float(np.std(plvs))

    feats['conn_plv_hemi_diff'] = abs(feats['conn_L_plv_mean'] - feats['conn_R_plv_mean'])

    # Inter-hemisphere PLV (left-right pairs mean)
    inter_plvs = []
    for l_ch in LEFT_CHS[:4]:
        for r_ch in RIGHT_CHS[:4]:
            inter_plvs.append(_plv(seg_f[l_ch], seg_f[r_ch]))
    feats['conn_inter_plv_mean'] = float(np.mean(inter_plvs))
    feats['conn_inter_plv_std'] = float(np.std(inter_plvs))

    # Intra vs inter ratio
    avg_intra = (feats['conn_L_plv_mean'] + feats['conn_R_plv_mean']) / 2
    feats['conn_intra_inter_ratio'] = avg_intra / max(feats['conn_inter_plv_mean'], 1e-12)

    # Correlation matrix summary
    corr_matrix = np.corrcoef(seg_f[:16])
    corr_matrix = np.nan_to_num(corr_matrix)
    left_left = corr_matrix[np.ix_(LEFT_CHS, LEFT_CHS)]
    right_right = corr_matrix[np.ix_(RIGHT_CHS, RIGHT_CHS)]
    left_right = corr_matrix[np.ix_(LEFT_CHS, RIGHT_CHS)]
    feats['conn_ll_corr_mean'] = float(np.mean(np.abs(left_left[np.triu_indices(8, k=1)])))
    feats['conn_rr_corr_mean'] = float(np.mean(np.abs(right_right[np.triu_indices(8, k=1)])))
    feats['conn_lr_corr_mean'] = float(np.mean(np.abs(left_right)))
    feats['conn_corr_asym'] = abs(feats['conn_ll_corr_mean'] - feats['conn_rr_corr_mean'])

    # SVD of full correlation matrix
    try:
        _, s, _ = np.linalg.svd(corr_matrix[:16, :16])
        feats['conn_svd_dom'] = float(s[0] / s.sum())
        feats['conn_svd_ratio12'] = float(s[0] / max(s[1], 1e-12))
    except:
        feats['conn_svd_dom'] = 0.0
        feats['conn_svd_ratio12'] = 0.0

    return feats


# ═══════════════════════════════════════════════════════════════
# Feature Family 4: Rhythmicity (inc. M3_HilbertCV frequency)
# ═══════════════════════════════════════════════════════════════

def feat_rhythmicity(seg):
    seg_f = _bp(seg)
    feats = {}

    for side, chs in [('L', LEFT_CHS), ('R', RIGHT_CHS)]:
        # Top-4 channel average
        powers = np.array([np.var(seg_f[ch]) for ch in chs])
        top_idx = chs[np.argsort(powers)[::-1][:4]]
        sig = np.mean(seg_f[top_idx], axis=0)

        # Hilbert frequency estimation (our best method - M3_HilbertCV)
        freq, cv, q = _hilbert_freq_cv(sig)
        feats[f'rhy_{side}_hilbert_freq'] = freq
        feats[f'rhy_{side}_hilbert_cv'] = cv
        feats[f'rhy_{side}_hilbert_q'] = q

        # ACF peak
        x = sig - np.mean(sig)
        n = len(x)
        acf = np.real(ifft(np.abs(fft(x, 2 * n)) ** 2))[:n]
        acf = acf / max(acf[0], 1e-12)
        min_lag, max_lag = int(FS / 3.5), min(int(FS / 0.5), n - 1)
        acf_seg = acf[min_lag:max_lag]
        feats[f'rhy_{side}_acf_peak'] = float(np.max(acf_seg)) if len(acf_seg) > 0 else 0.0

        # Spectral peak prominence / Q-factor
        f, pxx = welch(sig, fs=FS, nperseg=400)
        delta = (f >= 0.5) & (f <= 3.5)
        if delta.any() and pxx[delta].sum() > 0:
            peak_idx = np.argmax(pxx[delta])
            peak_power = pxx[delta][peak_idx]
            peak_f = f[delta][peak_idx]
            feats[f'rhy_{side}_peak_prominence'] = float(peak_power / np.mean(pxx[delta]))

            # Q-factor: peak_freq / bandwidth at -3dB
            half_power = peak_power / 2
            above = pxx[delta] >= half_power
            bw = float(np.sum(above) * (f[1] - f[0]))
            feats[f'rhy_{side}_q_factor'] = float(peak_f / max(bw, 0.1))
        else:
            feats[f'rhy_{side}_peak_prominence'] = 0.0
            feats[f'rhy_{side}_q_factor'] = 0.0

        # IPI regularity
        peaks, _ = find_peaks(sig, distance=int(FS / 3.5))
        if len(peaks) >= 3:
            ipis = np.diff(peaks) / FS
            feats[f'rhy_{side}_ipi_cv'] = float(np.std(ipis) / max(np.mean(ipis), 1e-6))
            feats[f'rhy_{side}_ipi_mean'] = float(np.mean(ipis))
        else:
            feats[f'rhy_{side}_ipi_cv'] = 1.0
            feats[f'rhy_{side}_ipi_mean'] = 0.0

    # Cross-hemisphere rhythmicity comparisons
    feats['rhy_freq_diff'] = abs(feats['rhy_L_hilbert_freq'] - feats['rhy_R_hilbert_freq'])
    feats['rhy_cv_diff'] = abs(feats['rhy_L_hilbert_cv'] - feats['rhy_R_hilbert_cv'])
    feats['rhy_q_diff'] = abs(feats['rhy_L_hilbert_q'] - feats['rhy_R_hilbert_q'])
    feats['rhy_acf_diff'] = abs(feats['rhy_L_acf_peak'] - feats['rhy_R_acf_peak'])

    # Delta peak frequency uniformity across ALL channels
    ch_freqs = []
    for ch in range(16):
        f, pxx = welch(seg_f[ch], fs=FS, nperseg=400)
        delta = (f >= 0.5) & (f <= 3.5)
        if delta.any() and pxx[delta].sum() > 0:
            ch_freqs.append(f[delta][np.argmax(pxx[delta])])
    if ch_freqs:
        feats['rhy_freq_variance'] = float(np.var(ch_freqs))
        feats['rhy_freq_range'] = float(np.max(ch_freqs) - np.min(ch_freqs))
    else:
        feats['rhy_freq_variance'] = 0.0
        feats['rhy_freq_range'] = 0.0

    return feats


# ═══════════════════════════════════════════════════════════════
# Feature Family 5: Waveform Shape
# ═══════════════════════════════════════════════════════════════

def feat_waveform(seg):
    seg_f = _bp(seg)
    feats = {}

    for side, chs in [('L', LEFT_CHS), ('R', RIGHT_CHS)]:
        powers = np.array([np.var(seg_f[ch]) for ch in chs])
        top_idx = chs[np.argsort(powers)[::-1][:4]]
        sig = np.mean(seg_f[top_idx], axis=0)

        feats[f'wf_{side}_kurtosis'] = float(kurtosis(sig))
        feats[f'wf_{side}_skewness'] = float(skew(sig))
        feats[f'wf_{side}_line_length'] = float(np.sum(np.abs(np.diff(sig))) / len(sig))
        feats[f'wf_{side}_crest_factor'] = float(np.max(np.abs(sig)) / max(np.sqrt(np.mean(sig**2)), 1e-12))

        # Hjorth parameters
        d1 = np.diff(sig)
        d2 = np.diff(d1)
        activity = float(np.var(sig))
        mobility = float(np.sqrt(np.var(d1) / max(activity, 1e-12)))
        complexity = float(np.sqrt(np.var(d2) / max(np.var(d1), 1e-12)) / max(mobility, 1e-12))
        feats[f'wf_{side}_hjorth_activity'] = activity
        feats[f'wf_{side}_hjorth_mobility'] = mobility
        feats[f'wf_{side}_hjorth_complexity'] = complexity

        # Smoothness
        sig_e = np.mean(sig ** 2)
        grad_e = np.mean(np.diff(sig) ** 2)
        feats[f'wf_{side}_smoothness'] = float(sig_e / (sig_e + grad_e + 1e-12))

        # Sample entropy approximation (permutation entropy)
        feats[f'wf_{side}_perm_entropy'] = _perm_entropy(sig, order=3, delay=int(FS/4))

    # Asymmetries
    for k in ['kurtosis', 'skewness', 'smoothness', 'hjorth_mobility', 'hjorth_complexity', 'perm_entropy']:
        lv = feats.get(f'wf_L_{k}', 0)
        rv = feats.get(f'wf_R_{k}', 0)
        feats[f'wf_{k}_asym'] = abs(rv - lv) / (abs(rv) + abs(lv) + 1e-12)

    return feats


def _perm_entropy(x, order=3, delay=1):
    """Permutation entropy."""
    n = len(x)
    from itertools import permutations
    perms = list(permutations(range(order)))
    counts = {p: 0 for p in perms}
    for i in range(n - (order - 1) * delay):
        pattern = tuple(np.argsort([x[i + j * delay] for j in range(order)]))
        if pattern in counts:
            counts[pattern] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values() if c > 0])
    return float(-np.sum(probs * np.log(probs)))


# ═══════════════════════════════════════════════════════════════
# Feature Family 6: Time-Frequency Stability
# ═══════════════════════════════════════════════════════════════

def feat_timefreq(seg):
    seg_f = _bp(seg)
    feats = {}
    win_len = int(2 * FS)  # 2-second windows
    n_wins = len(seg_f[0]) // win_len

    if n_wins < 2:
        for k in ['L_power_cv', 'R_power_cv', 'power_cv_diff', 'topo_stability',
                   'dom_ch_persistence_L', 'dom_ch_persistence_R', 'dom_persist_diff']:
            feats[f'tf_{k}'] = 0.0
        return feats

    # Per-window channel powers
    win_powers = []
    for w in range(n_wins):
        s, e = w * win_len, (w + 1) * win_len
        ch_p = np.array([np.var(seg_f[ch, s:e]) for ch in range(16)])
        win_powers.append(ch_p)
    win_powers = np.array(win_powers)  # (n_wins, 16)

    # Temporal stability of power per hemisphere
    for side, chs in [('L', LEFT_CHS), ('R', RIGHT_CHS)]:
        side_power = win_powers[:, chs].mean(axis=1)
        feats[f'tf_{side}_power_cv'] = float(np.std(side_power) / max(np.mean(side_power), 1e-12))

    feats['tf_power_cv_diff'] = abs(feats['tf_L_power_cv'] - feats['tf_R_power_cv'])

    # Topographic map stability (correlation between windows)
    if n_wins >= 3:
        topo_corrs = []
        for w in range(n_wins - 1):
            if np.std(win_powers[w]) > 1e-12 and np.std(win_powers[w+1]) > 1e-12:
                topo_corrs.append(float(np.corrcoef(win_powers[w], win_powers[w+1])[0, 1]))
        feats['tf_topo_stability'] = float(np.mean(topo_corrs)) if topo_corrs else 0.0
    else:
        feats['tf_topo_stability'] = 0.0

    # Dominant channel persistence
    dom_chs = np.argmax(win_powers, axis=1)
    for side, chs in [('L', LEFT_CHS), ('R', RIGHT_CHS)]:
        side_dom = np.array([np.argmax(win_powers[w, chs]) for w in range(n_wins)])
        feats[f'tf_dom_ch_persistence_{side}'] = float(np.max(np.bincount(side_dom)) / n_wins)

    feats['tf_dom_persist_diff'] = abs(feats['tf_dom_ch_persistence_L'] - feats['tf_dom_ch_persistence_R'])

    return feats


# ═══════════════════════════════════════════════════════════════
# Feature Family 7: Propagation / Lag
# ═══════════════════════════════════════════════════════════════

def feat_propagation(seg):
    seg_f = _bp(seg)
    feats = {}

    for side_name, chain in [('L_temp', LEFT_CHAIN), ('R_temp', RIGHT_CHAIN),
                              ('L_para', LEFT_PARA), ('R_para', RIGHT_PARA)]:
        lags = []
        for i in range(len(chain) - 1):
            # Cross-correlation lag
            s1, s2 = seg_f[chain[i]], seg_f[chain[i + 1]]
            xc = np.correlate(s1 - s1.mean(), s2 - s2.mean(), mode='full')
            max_lag_samp = int(FS * 0.1)  # max 100ms lag
            center = len(xc) // 2
            xc_seg = xc[center - max_lag_samp:center + max_lag_samp + 1]
            if len(xc_seg) > 0:
                lag = (np.argmax(xc_seg) - max_lag_samp) / FS * 1000  # ms
                lags.append(lag)

        if lags:
            feats[f'prop_{side_name}_mean_lag'] = float(np.mean(lags))
            feats[f'prop_{side_name}_consistency'] = float(1.0 - np.std(np.sign(lags)))  # direction consistency
        else:
            feats[f'prop_{side_name}_mean_lag'] = 0.0
            feats[f'prop_{side_name}_consistency'] = 0.0

    # Propagation asymmetry
    feats['prop_lag_asym_temp'] = abs(feats['prop_L_temp_mean_lag'] - feats['prop_R_temp_mean_lag'])
    feats['prop_lag_asym_para'] = abs(feats['prop_L_para_mean_lag'] - feats['prop_R_para_mean_lag'])

    return feats


# ═══════════════════════════════════════════════════════════════
# Feature Family 8: Per-Channel (compact)
# ═══════════════════════════════════════════════════════════════

def feat_perchannel(seg):
    seg_f = _bp(seg)
    feats = {}
    for ch in range(18):
        sig = seg_f[ch]
        feats[f'ch{ch:02d}_power'] = float(np.var(sig))

        # Spectral peak ratio
        f, pxx = welch(sig, fs=FS, nperseg=400)
        delta = (f >= 0.5) & (f <= 3.5)
        if delta.any() and pxx[delta].mean() > 0:
            feats[f'ch{ch:02d}_peak_ratio'] = float(np.max(pxx[delta]) / np.mean(pxx[delta]))
        else:
            feats[f'ch{ch:02d}_peak_ratio'] = 0.0

        # Hilbert q-score per channel
        _, _, q = _hilbert_freq_cv(sig)
        feats[f'ch{ch:02d}_hilbert_q'] = q

    return feats


# ═══════════════════════════════════════════════════════════════
# Feature Family 9: Homologous Background Subtraction
# ═══════════════════════════════════════════════════════════════

def feat_background_subtraction(seg):
    """Subtract homologous channel pairs and analyze residual."""
    seg_f = _bp(seg)
    feats = {}

    residual_powers = []
    residual_rhythmicities = []

    for i, (l_ch, r_ch) in enumerate(HOMOLOGOUS_PAIRS):
        l_sig = seg_f[l_ch]
        r_sig = seg_f[r_ch]

        # Normalize then subtract
        l_norm = l_sig / max(np.std(l_sig), 1e-12)
        r_norm = r_sig / max(np.std(r_sig), 1e-12)
        residual = l_norm - r_norm

        res_power = float(np.var(residual))
        residual_powers.append(res_power)

        # Rhythmicity of residual
        _, _, q = _hilbert_freq_cv(residual)
        residual_rhythmicities.append(q)

    feats['bgsub_residual_power_mean'] = float(np.mean(residual_powers))
    feats['bgsub_residual_power_max'] = float(np.max(residual_powers))
    feats['bgsub_residual_power_std'] = float(np.std(residual_powers))
    feats['bgsub_residual_rhythm_mean'] = float(np.mean(residual_rhythmicities))
    feats['bgsub_residual_rhythm_max'] = float(np.max(residual_rhythmicities))

    return feats


# ═══════════════════════════════════════════════════════════════
# Combined extraction
# ═══════════════════════════════════════════════════════════════

FEATURE_FAMILIES = {
    'focality': feat_focality,
    'homologous': feat_homologous,
    'connectivity': feat_connectivity,
    'rhythmicity': feat_rhythmicity,
    'waveform': feat_waveform,
    'timefreq': feat_timefreq,
    'propagation': feat_propagation,
    'perchannel': feat_perchannel,
    'bgsub': feat_background_subtraction,
}


def extract_all(seg, families=None):
    """Extract all or selected feature families."""
    if families is None:
        families = list(FEATURE_FAMILIES.keys())
    feats = {}
    for name in families:
        feats.update(FEATURE_FAMILIES[name](seg))
    return feats


def extract_cached(pid):
    """Load Round 2 cached scores."""
    feats = {}
    for path in sorted(CACHE_DIR.glob('*_scores.json')):
        method_name = path.stem.replace('_scores', '')
        with open(path) as f:
            scores = json.load(f)
        if pid in scores:
            s = scores[pid]
            feats[f'{method_name}_asym'] = s['asymmetry']
            feats[f'{method_name}_lat'] = s['laterality_index']
    return feats
