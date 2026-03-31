"""V4 Unified Methods Round 2 — 25 methods targeting AUC>0.814 AND Freq ρ>0.667.

All methods process hemispheres independently, outputting:
    left_score, right_score — for lateralization + classification
    extras.freq — frequency estimate from dominant hemisphere
"""
import numpy as np
from scipy.signal import welch, hilbert, butter, sosfiltfilt, find_peaks
from scipy.stats import pearsonr
from numpy.fft import fft, ifft

from .base import LateralMethod, FS, LEFT_CHS, RIGHT_CHS


def _hemi_top_signal(seg, chs, top_k=4):
    powers = np.array([np.var(seg[ch]) for ch in chs])
    top_idx = chs[np.argsort(powers)[::-1][:top_k]]
    return np.mean(seg[top_idx], axis=0), top_idx


def _hilbert_freq_cv(sig):
    if np.std(sig) < 1e-10:
        return np.nan, 1.0, 0.0
    analytic = hilbert(sig)
    inst_freq = np.diff(np.unwrap(np.angle(analytic))) * FS / (2 * np.pi)
    mask = (inst_freq > 0.3) & (inst_freq < 4.0)
    valid = inst_freq[mask]
    if len(valid) < 20:
        return np.nan, 1.0, 0.0
    return float(np.median(valid)), float(np.std(valid) / max(np.median(valid), 1e-6)), max(0, 1 - 2 * np.std(valid) / max(np.median(valid), 1e-6))


def _acf_freq(sig):
    x = sig - np.mean(sig)
    n = len(x)
    acf = np.real(ifft(np.abs(fft(x, 2 * n)) ** 2))[:n]
    acf = acf / max(acf[0], 1e-12)
    min_lag, max_lag = int(FS / 3.5), min(int(FS / 0.5), n - 1)
    seg_acf = acf[min_lag:max_lag]
    if len(seg_acf) == 0:
        return np.nan, 0.0
    peak_idx = np.argmax(seg_acf)
    return FS / (min_lag + peak_idx), float(seg_acf[peak_idx])


def _spectral_peak(sig):
    f, pxx = welch(sig, fs=FS, nperseg=400)
    delta = (f >= 0.5) & (f <= 3.5)
    if not delta.any() or pxx[delta].sum() == 0:
        return np.nan, 0.0
    return float(f[delta][np.argmax(pxx[delta])]), float(np.max(pxx[delta]) / np.mean(pxx[delta]))


def _ve_freq(sig, grid=np.arange(0.5, 3.55, 0.1)):
    t = np.arange(len(sig)) / FS
    best_ve, best_f = 0.0, np.nan
    tv = np.var(sig)
    if tv < 1e-12:
        return np.nan, 0.0
    for f in grid:
        basis = np.column_stack([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t), np.ones(len(t))])
        try:
            c, _, _, _ = np.linalg.lstsq(basis, sig, rcond=None)
            ve = max(0, 1 - np.var(sig - basis @ c) / tv)
            if ve > best_ve:
                best_ve, best_f = ve, f
        except:
            pass
    return float(best_f), float(best_ve)


def _normalize_pair(lv, rv):
    mx = max(lv, rv, 1e-12)
    return lv / mx, rv / mx


def _make_result(ls, rs, lf, rf):
    ls_n, rs_n = _normalize_pair(ls, rs)
    freq = lf if ls_n >= rs_n and np.isfinite(lf) else rf if np.isfinite(rf) else lf
    return {'left_score': ls_n, 'right_score': rs_n,
            'extras': {'left_freq': lf, 'right_freq': rf,
                       'freq': float(freq) if np.isfinite(freq) else None}}


# ═══════════════════════════════════════════════════════════════
# Strategy A: Better channel selection for frequency
# ═══════════════════════════════════════════════════════════════

class V01_DomHemi_Top3Hilbert(LateralMethod):
    """All-channel envelope for lateralization, top-3 Hilbert on dominant hemi for freq."""
    name = "V01_DomHemi_Top3Hilbert"
    description = "Envelope amp (all ch) + Hilbert freq (top-3, dominant hemi)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        powers = np.array([np.var(seg_f[ch]) for ch in dom_chs])
        top3 = dom_chs[np.argsort(powers)[::-1][:3]]
        freqs = [_hilbert_freq_cv(seg_f[ch])[0] for ch in top3]
        freqs = [f for f in freqs if np.isfinite(f)]
        freq = float(np.median(freqs)) if freqs else np.nan
        # Also get per-hemi freq for completeness
        lf_list = [_hilbert_freq_cv(seg_f[ch])[0] for ch in LEFT_CHS[:3]]
        rf_list = [_hilbert_freq_cv(seg_f[ch])[0] for ch in RIGHT_CHS[:3]]
        lf = float(np.median([f for f in lf_list if np.isfinite(f)])) if any(np.isfinite(f) for f in lf_list) else np.nan
        rf = float(np.median([f for f in rf_list if np.isfinite(f)])) if any(np.isfinite(f) for f in rf_list) else np.nan
        ls_n, rs_n = _normalize_pair(ls, rs)
        return {'left_score': ls_n, 'right_score': rs_n,
                'extras': {'left_freq': lf, 'right_freq': rf, 'freq': freq}}


class V02_PowerWeightedHilbert(LateralMethod):
    """Per-channel Hilbert freq weighted by channel power."""
    name = "V02_PowerWeightedHilbert"
    description = "Power-weighted Hilbert freq + envelope amp per hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        def score_hemi(chs):
            envs, freqs, weights = [], [], []
            for ch in chs:
                env = np.abs(hilbert(seg_f[ch]))
                envs.append(np.mean(env))
                power = np.var(seg_f[ch])
                f, _, q = _hilbert_freq_cv(seg_f[ch])
                if np.isfinite(f) and q > 0.05:
                    freqs.append(f)
                    weights.append(power)
            score = float(np.mean(envs))
            if freqs and weights:
                freq = float(np.average(freqs, weights=weights))
            else:
                freq = np.nan
            return score, freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V03_ConsistencySelected(LateralMethod):
    """Select channels with consistent frequency, use their distribution for lateralization."""
    name = "V03_ConsistencySelected"
    description = "Channels with agreeing freq → spatial distribution = lateralization"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        ch_freqs = {}
        ch_envs = {}
        for ch in range(16):
            f, _, q = _hilbert_freq_cv(seg_f[ch])
            ch_freqs[ch] = f
            ch_envs[ch] = float(np.mean(np.abs(hilbert(seg_f[ch]))))
        # Find consensus frequency
        valid = [(ch, f) for ch, f in ch_freqs.items() if np.isfinite(f)]
        if len(valid) < 3:
            return _make_result(0, 0, np.nan, np.nan)
        all_f = [f for _, f in valid]
        median_f = np.median(all_f)
        # Channels within ±0.3 Hz of median
        consistent = [ch for ch, f in valid if abs(f - median_f) < 0.3]
        left_score = sum(ch_envs[ch] for ch in consistent if ch in LEFT_CHS)
        right_score = sum(ch_envs[ch] for ch in consistent if ch in RIGHT_CHS)
        freq = float(np.median([ch_freqs[ch] for ch in consistent]))
        lf = float(np.median([ch_freqs[ch] for ch in consistent if ch in LEFT_CHS])) if any(ch in LEFT_CHS for ch in consistent) else np.nan
        rf = float(np.median([ch_freqs[ch] for ch in consistent if ch in RIGHT_CHS])) if any(ch in RIGHT_CHS for ch in consistent) else np.nan
        return _make_result(left_score, right_score, lf, rf)


class V04_PLVSelected(LateralMethod):
    """PLV-coherent channels for both score and frequency."""
    name = "V04_PLVSelected"
    description = "High-PLV channels → envelope for score, Hilbert for freq"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)
        def score_hemi(chs):
            phases = np.array([np.angle(hilbert(seg_f[ch])) for ch in chs])
            # Mean PLV per channel with its neighbors
            ch_plvs = []
            for i in range(len(chs)):
                plvs = []
                for j in range(len(chs)):
                    if i != j:
                        plvs.append(float(np.abs(np.mean(np.exp(1j * (phases[i] - phases[j]))))))
                ch_plvs.append(np.mean(plvs))
            ch_plvs = np.array(ch_plvs)
            # Weight envelope by PLV
            envs = np.array([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in chs])
            score = float(np.sum(envs * ch_plvs))
            # Freq from top-PLV channels
            top2 = np.argsort(ch_plvs)[::-1][:2]
            freqs = [_hilbert_freq_cv(seg_f[chs[i]])[0] for i in top2]
            freqs = [f for f in freqs if np.isfinite(f)]
            freq = float(np.median(freqs)) if freqs else np.nan
            return score, freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V05_AdaptiveTopK(LateralMethod):
    """Adaptively select channels with stable rhythm (low Hilbert CV)."""
    name = "V05_AdaptiveTopK"
    description = "Channels with Hilbert CV < threshold → count for score, freq for freq"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        def score_hemi(chs):
            good_freqs, good_envs = [], []
            for ch in chs:
                f, cv, q = _hilbert_freq_cv(seg_f[ch])
                if np.isfinite(f) and cv < 0.5:
                    good_freqs.append(f)
                    good_envs.append(np.mean(np.abs(hilbert(seg_f[ch]))))
            score = float(np.sum(good_envs)) if good_envs else 0.0
            freq = float(np.median(good_freqs)) if good_freqs else np.nan
            return score, freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


# ═══════════════════════════════════════════════════════════════
# Strategy B: Better frequency estimation
# ═══════════════════════════════════════════════════════════════

class V06_MultiMethodFreq(LateralMethod):
    """Envelope amplitude for score + median of Hilbert/ACF/spectral for freq."""
    name = "V06_MultiMethodFreq"
    description = "Envelope amp + median(Hilbert, ACF, spectral) freq per hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        def score_hemi(chs):
            sig, _ = _hemi_top_signal(seg_f, chs)
            score = float(np.mean(np.abs(hilbert(sig))))
            f1, _, _ = _hilbert_freq_cv(sig)
            f2, _ = _acf_freq(sig)
            f3, _ = _spectral_peak(sig)
            freqs = [f for f in [f1, f2, f3] if np.isfinite(f)]
            freq = float(np.median(freqs)) if freqs else np.nan
            return score, freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V07_NarrowbandSweep(LateralMethod):
    """Sweep narrowband filters, find freq with max envelope amplitude."""
    name = "V07_NarrowbandSweep"
    description = "Narrowband envelope sweep → peak freq + peak amplitude per hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        def score_hemi(chs):
            sig, _ = _hemi_top_signal(seg_f, chs)
            best_env, best_freq = 0.0, np.nan
            for f in np.arange(0.5, 3.6, 0.25):
                lo, hi = max(f - 0.3, 0.1), min(f + 0.3, FS/2 - 0.1)
                if lo >= hi:
                    continue
                sos = butter(4, [lo/(FS/2), hi/(FS/2)], btype='bandpass', output='sos')
                nb = sosfiltfilt(sos, sig)
                env_mean = float(np.mean(np.abs(hilbert(nb))))
                if env_mean > best_env:
                    best_env, best_freq = env_mean, f
            return best_env, best_freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V08_TemplateBank(LateralMethod):
    """Sinusoidal template bank — best match gives freq + correlation gives score."""
    name = "V08_TemplateBank"
    description = "Sin template bank correlation per hemi → freq + score"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        t = np.arange(2000) / FS
        def score_hemi(chs):
            sig, _ = _hemi_top_signal(seg_f, chs)
            best_r2, best_freq = 0.0, np.nan
            tv = np.var(sig)
            if tv < 1e-12:
                return 0.0, np.nan
            for f in np.arange(0.5, 3.55, 0.1):
                basis = np.column_stack([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t), np.ones(2000)])
                try:
                    c, _, _, _ = np.linalg.lstsq(basis, sig, rcond=None)
                    r2 = max(0, 1 - np.var(sig - basis @ c) / tv)
                    if r2 > best_r2:
                        best_r2, best_freq = r2, f
                except:
                    pass
            return best_r2, best_freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V09_VEGrid_EnvScore(LateralMethod):
    """NVO-style VE grid for freq + envelope amplitude for score."""
    name = "V09_VEGrid_EnvScore"
    description = "VE frequency grid + envelope amplitude per hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        def score_hemi(chs):
            sig, _ = _hemi_top_signal(seg_f, chs)
            score = float(np.mean(np.abs(hilbert(sig))))
            freq, _ = _ve_freq(sig)
            return score, freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V10_CepstralFreq(LateralMethod):
    """Cepstral peak for frequency + envelope amplitude for score."""
    name = "V10_CepstralFreq"
    description = "Cepstral peak quefrency → freq + envelope amp → score per hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        def score_hemi(chs):
            sig, _ = _hemi_top_signal(seg_f, chs)
            score = float(np.mean(np.abs(hilbert(sig))))
            spectrum = np.fft.fft(sig)
            cepstrum = np.real(np.fft.ifft(np.log(np.abs(spectrum) + 1e-12)))
            min_q, max_q = int(FS / 3.5), min(int(FS / 0.5), len(cepstrum) // 2)
            if min_q >= max_q:
                return score, np.nan
            seg_c = cepstrum[min_q:max_q]
            freq = FS / (min_q + np.argmax(seg_c))
            return score, float(freq)
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


# ═══════════════════════════════════════════════════════════════
# Strategy C: Joint optimization
# ═══════════════════════════════════════════════════════════════

class V11_NarrowbandAtPeak(LateralMethod):
    """Estimate freq first, then lateralize in narrowband at that freq."""
    name = "V11_NarrowbandAtPeak"
    description = "Spectral peak freq → narrowband → envelope lateralization"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        # Estimate freq from full signal
        full_sig = np.mean(seg_f[:16], axis=0)
        peak_freq, _ = _spectral_peak(full_sig)
        if not np.isfinite(peak_freq):
            peak_freq = 1.5
        # Narrowband at peak freq
        lo, hi = max(peak_freq - 0.4, 0.1), min(peak_freq + 0.4, FS/2 - 0.1)
        sos = butter(4, [lo/(FS/2), hi/(FS/2)], btype='bandpass', output='sos')
        seg_nb = sosfiltfilt(sos, seg_f, axis=1)
        # Lateralize on narrowband
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in RIGHT_CHS]))
        # Freq from dominant hemisphere (Hilbert on narrowband)
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        dom_sig, _ = _hemi_top_signal(seg_nb, dom_chs, top_k=3)
        lf, _, _ = _hilbert_freq_cv(_hemi_top_signal(seg_nb, LEFT_CHS, 3)[0])
        rf, _, _ = _hilbert_freq_cv(_hemi_top_signal(seg_nb, RIGHT_CHS, 3)[0])
        return _make_result(ls, rs, lf, rf)


class V12_IterativeRefine(LateralMethod):
    """Two-pass: coarse lateralization → freq from dominant → narrowband → refined lateralization."""
    name = "V12_IterativeRefine"
    description = "Coarse lat → freq → narrowband → refined lat + freq"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        # Pass 1: coarse lateralization
        ls1 = float(np.mean([np.var(seg_f[ch]) for ch in LEFT_CHS]))
        rs1 = float(np.mean([np.var(seg_f[ch]) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls1 >= rs1 else RIGHT_CHS
        # Estimate freq from dominant hemisphere
        dom_sig, _ = _hemi_top_signal(seg_f, dom_chs, 3)
        est_freq, _, _ = _hilbert_freq_cv(dom_sig)
        if not np.isfinite(est_freq):
            est_freq, _ = _spectral_peak(dom_sig)
        if not np.isfinite(est_freq):
            est_freq = 1.5
        # Pass 2: narrowband at estimated freq
        lo, hi = max(est_freq - 0.4, 0.1), min(est_freq + 0.4, FS/2 - 0.1)
        sos = butter(4, [lo/(FS/2), hi/(FS/2)], btype='bandpass', output='sos')
        seg_nb = sosfiltfilt(sos, seg_f, axis=1)
        ls2 = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in LEFT_CHS]))
        rs2 = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in RIGHT_CHS]))
        # Final freq from dominant of refined lateralization
        dom_chs2 = LEFT_CHS if ls2 >= rs2 else RIGHT_CHS
        dom_sig2, _ = _hemi_top_signal(seg_nb, dom_chs2, 3)
        freq2, _, _ = _hilbert_freq_cv(dom_sig2)
        lf, _, _ = _hilbert_freq_cv(_hemi_top_signal(seg_nb, LEFT_CHS, 3)[0])
        rf, _, _ = _hilbert_freq_cv(_hemi_top_signal(seg_nb, RIGHT_CHS, 3)[0])
        return _make_result(ls2, rs2, lf, rf)


class V13_MatchedFilterLat(LateralMethod):
    """For each freq, compute template match per hemisphere. Best asymmetry freq wins."""
    name = "V13_MatchedFilterLat"
    description = "Template match asymmetry across freq grid → best freq + lateralization"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        t = np.arange(2000) / FS
        l_sig, _ = _hemi_top_signal(seg_f, LEFT_CHS)
        r_sig, _ = _hemi_top_signal(seg_f, RIGHT_CHS)
        best_asym, best_freq = 0.0, 1.5
        best_ls, best_rs = 0.0, 0.0
        tv_l, tv_r = np.var(l_sig), np.var(r_sig)
        if tv_l < 1e-12 and tv_r < 1e-12:
            return _make_result(0, 0, np.nan, np.nan)
        for f in np.arange(0.5, 3.55, 0.15):
            basis = np.column_stack([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t), np.ones(2000)])
            try:
                cl, _, _, _ = np.linalg.lstsq(basis, l_sig, rcond=None)
                cr, _, _, _ = np.linalg.lstsq(basis, r_sig, rcond=None)
                ve_l = max(0, 1 - np.var(l_sig - basis @ cl) / max(tv_l, 1e-12))
                ve_r = max(0, 1 - np.var(r_sig - basis @ cr) / max(tv_r, 1e-12))
                asym = abs(ve_l - ve_r)
                if asym > best_asym:
                    best_asym, best_freq = asym, f
                    best_ls, best_rs = ve_l, ve_r
            except:
                pass
        lf, _ = _ve_freq(l_sig, np.arange(max(0.5, best_freq-0.5), min(3.5, best_freq+0.5), 0.05))
        rf, _ = _ve_freq(r_sig, np.arange(max(0.5, best_freq-0.5), min(3.5, best_freq+0.5), 0.05))
        return _make_result(best_ls, best_rs, lf, rf)


class V14_FreqSpecificPowerRatio(LateralMethod):
    """Power asymmetry at each frequency — peak asymmetry gives freq + lateralization."""
    name = "V14_FreqSpecificPowerRatio"
    description = "L/R power ratio per freq bin → peak asymmetry freq"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        l_sig, _ = _hemi_top_signal(seg_f, LEFT_CHS)
        r_sig, _ = _hemi_top_signal(seg_f, RIGHT_CHS)
        fl, pxx_l = welch(l_sig, fs=FS, nperseg=400)
        fr, pxx_r = welch(r_sig, fs=FS, nperseg=400)
        delta = (fl >= 0.5) & (fl <= 3.5)
        if not delta.any():
            return _make_result(0, 0, np.nan, np.nan)
        asym = np.abs(pxx_l[delta] - pxx_r[delta]) / (pxx_l[delta] + pxx_r[delta] + 1e-12)
        peak_idx = np.argmax(asym)
        freq = float(fl[delta][peak_idx])
        ls = float(np.sum(pxx_l[delta]))
        rs = float(np.sum(pxx_r[delta]))
        lf, _ = _spectral_peak(l_sig)
        rf, _ = _spectral_peak(r_sig)
        return _make_result(ls, rs, lf, rf)


class V15_CrossFreqProfile(LateralMethod):
    """Lateralization profile across frequencies — peaked = LRDA, flat = GRDA."""
    name = "V15_CrossFreqProfile"
    description = "Lateralization index per freq → peak height + peak location"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        freqs = np.arange(0.5, 3.6, 0.25)
        lat_profile = []
        l_scores, r_scores = [], []
        for f in freqs:
            lo, hi = max(f - 0.2, 0.1), min(f + 0.2, FS/2 - 0.1)
            if lo >= hi:
                lat_profile.append(0)
                l_scores.append(0)
                r_scores.append(0)
                continue
            sos = butter(3, [lo/(FS/2), hi/(FS/2)], btype='bandpass', output='sos')
            nb = sosfiltfilt(sos, seg_f, axis=1)
            lp = float(np.mean([np.var(nb[ch]) for ch in LEFT_CHS]))
            rp = float(np.mean([np.var(nb[ch]) for ch in RIGHT_CHS]))
            lat = abs(lp - rp) / (lp + rp + 1e-12)
            lat_profile.append(lat)
            l_scores.append(lp)
            r_scores.append(rp)
        lat_profile = np.array(lat_profile)
        peak_idx = np.argmax(lat_profile)
        freq = float(freqs[peak_idx])
        ls = float(np.sum(l_scores))
        rs = float(np.sum(r_scores))
        lf, rf = freq, freq  # same freq estimate for both
        return _make_result(ls, rs, lf, rf)


# ═══════════════════════════════════════════════════════════════
# Strategy D: Waveform-based
# ═══════════════════════════════════════════════════════════════

class V16_EnvPeriodicity(LateralMethod):
    """ACF of envelope for freq, envelope amplitude for score."""
    name = "V16_EnvPeriodicity"
    description = "Envelope ACF → freq, envelope mean → score per hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)
        def score_hemi(chs):
            sig, _ = _hemi_top_signal(seg_f, chs)
            env = np.abs(hilbert(sig))
            score = float(np.mean(env))
            freq, _ = _acf_freq(env)
            return score, freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V17_PeakCountFreq(LateralMethod):
    """Count peaks / 10 sec = frequency, envelope amplitude = score."""
    name = "V17_PeakCountFreq"
    description = "Peak count → freq, envelope amp → score per hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)
        def score_hemi(chs):
            sig, _ = _hemi_top_signal(seg_f, chs)
            score = float(np.mean(np.abs(hilbert(sig))))
            peaks, _ = find_peaks(sig, distance=int(FS/3.5))
            freq = len(peaks) / 10.0 if len(peaks) >= 2 else np.nan
            return score, freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V18_ZeroCrossFreq(LateralMethod):
    """Zero-crossing rate / 2 = frequency, RMS = score."""
    name = "V18_ZeroCrossFreq"
    description = "Zero-crossing rate → freq, RMS → score per hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)
        def score_hemi(chs):
            sig, _ = _hemi_top_signal(seg_f, chs)
            score = float(np.sqrt(np.mean(sig**2)))
            crossings = np.where(np.diff(np.sign(sig)))[0]
            freq = len(crossings) / (2 * 10.0)  # half-cycles per second
            return score, freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


# ═══════════════════════════════════════════════════════════════
# Strategy E: Multi-channel spatial
# ═══════════════════════════════════════════════════════════════

class V19_SpatialCoherenceFreq(LateralMethod):
    """Mean PLV across channel pairs at each freq → peak freq + peak PLV."""
    name = "V19_SpatialCoherenceFreq"
    description = "Intra-hemi PLV per freq band → peak freq + PLV score"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        def score_hemi(chs):
            best_plv, best_freq = 0.0, np.nan
            for f in np.arange(0.5, 3.6, 0.5):
                lo, hi = max(f - 0.3, 0.1), min(f + 0.3, FS/2 - 0.1)
                if lo >= hi:
                    continue
                sos = butter(3, [lo/(FS/2), hi/(FS/2)], btype='bandpass', output='sos')
                nb = sosfiltfilt(sos, seg_f[chs], axis=1)
                phases = np.array([np.angle(hilbert(nb[i])) for i in range(len(chs))])
                plvs = []
                for i in range(min(4, len(chs))):
                    for j in range(i+1, min(i+3, len(chs))):
                        plvs.append(float(np.abs(np.mean(np.exp(1j*(phases[i]-phases[j]))))))
                mean_plv = np.mean(plvs) if plvs else 0.0
                if mean_plv > best_plv:
                    best_plv, best_freq = mean_plv, f
            return best_plv, best_freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V20_SVDFreq(LateralMethod):
    """SVD of hemisphere channels → first component signal → Hilbert freq."""
    name = "V20_SVDFreq"
    description = "SVD first component per hemi → singular value for score, Hilbert for freq"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)
        def score_hemi(chs):
            try:
                U, s, Vt = np.linalg.svd(seg_f[chs], full_matrices=False)
                score = float(s[0] / s.sum()) if s.sum() > 1e-12 else 0.0
                # Project onto first component
                pc1 = Vt[0, :]  # first right singular vector (time course)
                freq, _, _ = _hilbert_freq_cv(pc1)
                return score, freq
            except:
                return 0.0, np.nan
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V21_ChannelFreqMatrix(LateralMethod):
    """Per-channel spectral peak → find consensus freq, spatial extent gives score."""
    name = "V21_ChannelFreqMatrix"
    description = "Channel consensus freq + spatial extent per hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        def score_hemi(chs):
            ch_freqs, ch_powers = [], []
            for ch in chs:
                f, prom = _spectral_peak(seg_f[ch])
                ch_freqs.append(f)
                ch_powers.append(np.var(seg_f[ch]))
            valid = [(f, p) for f, p in zip(ch_freqs, ch_powers) if np.isfinite(f)]
            if not valid:
                return 0.0, np.nan
            # Weighted frequency by power
            freqs, powers = zip(*valid)
            freq = float(np.average(freqs, weights=powers))
            score = float(np.sum(powers))
            return score, freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


# ═══════════════════════════════════════════════════════════════
# Strategy F: Hybrid / cherry-pick
# ═══════════════════════════════════════════════════════════════

class V22_EnvAmp_DomHilbert(LateralMethod):
    """U10 score + U11 freq from dominant hemisphere only."""
    name = "V22_EnvAmp_DomHilbert"
    description = "All-ch envelope amp + top-3 Hilbert freq (dominant hemi only)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_narrow = sosfiltfilt(sos, seg_f, axis=1)
        # Score: envelope amplitude per hemisphere (all channels)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        # Freq: Hilbert CV on top-3 of EACH hemisphere (like U11)
        def hemi_freq(chs):
            powers = np.array([np.var(seg_narrow[ch]) for ch in chs])
            top3 = chs[np.argsort(powers)[::-1][:3]]
            freqs = []
            for ch in top3:
                f, cv, q = _hilbert_freq_cv(seg_narrow[ch])
                if np.isfinite(f):
                    freqs.append(f)
            return float(np.median(freqs)) if freqs else np.nan
        lf = hemi_freq(LEFT_CHS)
        rf = hemi_freq(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V23_CherryPick(LateralMethod):
    """Use L24's method for score, U11's method for freq."""
    name = "V23_CherryPick"
    description = "L24 envelope amp score + U11 Hilbert CV freq (independent)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        # Score: mean envelope per channel per hemisphere (L24 style)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        # Freq: narrowband Hilbert CV top-3 per hemisphere (U11 style)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        def hemi_freq(chs):
            powers = np.array([np.var(seg_n[ch]) for ch in chs])
            top3 = chs[np.argsort(powers)[::-1][:3]]
            ch_freqs, ch_cvs = [], []
            for ch in top3:
                f, cv, q = _hilbert_freq_cv(seg_n[ch])
                if np.isfinite(f):
                    ch_freqs.append(f)
                    ch_cvs.append(cv)
            return float(np.median(ch_freqs)) if ch_freqs else np.nan
        lf = hemi_freq(LEFT_CHS)
        rf = hemi_freq(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V24_SoftChannelWeight(LateralMethod):
    """Weight Hilbert freq by envelope amplitude per channel."""
    name = "V24_SoftChannelWeight"
    description = "Per-channel envelope-weighted Hilbert freq + total envelope score"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        def score_hemi(chs):
            envs, freqs = [], []
            for ch in chs:
                env = np.mean(np.abs(hilbert(seg_f[ch])))
                envs.append(env)
                f, _, q = _hilbert_freq_cv(seg_f[ch])
                if np.isfinite(f) and q > 0.05:
                    freqs.append((f, env))
            score = float(np.mean(envs))
            if freqs:
                fs, ws = zip(*freqs)
                freq = float(np.average(fs, weights=ws))
            else:
                freq = np.nan
            return score, freq
        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        return _make_result(ls, rs, lf, rf)


class V25_FreqBandEnvRatio(LateralMethod):
    """Find freq that maximizes L/R envelope ratio — freq + lateralization in one."""
    name = "V25_FreqBandEnvRatio"
    description = "Freq with max hemisphere envelope ratio → freq + lateralization"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_ratio, best_freq = 0.0, 1.5
        best_ls, best_rs = 0.0, 0.0
        for f in np.arange(0.5, 3.6, 0.2):
            lo, hi = max(f - 0.3, 0.1), min(f + 0.3, FS/2 - 0.1)
            if lo >= hi:
                continue
            sos = butter(3, [lo/(FS/2), hi/(FS/2)], btype='bandpass', output='sos')
            nb = sosfiltfilt(sos, seg_f, axis=1)
            lp = float(np.mean([np.mean(np.abs(hilbert(nb[ch]))) for ch in LEFT_CHS]))
            rp = float(np.mean([np.mean(np.abs(hilbert(nb[ch]))) for ch in RIGHT_CHS]))
            ratio = max(lp, rp) / (min(lp, rp) + 1e-12)
            if ratio > best_ratio:
                best_ratio, best_freq = ratio, f
                best_ls, best_rs = lp, rp
        # Refine freq with Hilbert on dominant hemisphere at best band
        lo, hi = max(best_freq - 0.4, 0.1), min(best_freq + 0.4, FS/2 - 0.1)
        sos = butter(3, [lo/(FS/2), hi/(FS/2)], btype='bandpass', output='sos')
        nb = sosfiltfilt(sos, seg_f, axis=1)
        dom_chs = LEFT_CHS if best_ls >= best_rs else RIGHT_CHS
        dom_sig, _ = _hemi_top_signal(nb, dom_chs, 3)
        refined_freq, _, _ = _hilbert_freq_cv(dom_sig)
        freq = refined_freq if np.isfinite(refined_freq) else best_freq
        lf, _, _ = _hilbert_freq_cv(_hemi_top_signal(nb, LEFT_CHS, 3)[0])
        rf, _, _ = _hilbert_freq_cv(_hemi_top_signal(nb, RIGHT_CHS, 3)[0])
        return _make_result(best_ls, best_rs, lf, rf)


# ═══════════════════════════════════════════════════════════════
# Strategy G: Dominant-side-only frequency + auto channel selection
# ═══════════════════════════════════════════════════════════════

class W01_DomOnly_StrictHilbert(LateralMethod):
    """Envelope lateralization; frequency ONLY from predicted-dominant hemisphere.

    Key difference from V22: frequency is estimated exclusively from the
    dominant side (not both), and the non-dominant frequency is set to NaN.
    This prevents contamination from the non-LRDA hemisphere.
    """
    name = "W01_DomOnly_StrictHilbert"
    description = "Envelope amp lat + Hilbert freq strictly from dominant hemi only"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        powers = np.array([np.var(seg_n[ch]) for ch in dom_chs])
        top3 = dom_chs[np.argsort(powers)[::-1][:3]]
        freqs = []
        for ch in top3:
            f, cv, q = _hilbert_freq_cv(seg_n[ch])
            if np.isfinite(f):
                freqs.append(f)
        dom_freq = float(np.median(freqs)) if freqs else np.nan
        # Only report dominant side frequency
        if ls >= rs:
            return _make_result(ls, rs, dom_freq, np.nan)
        else:
            return _make_result(ls, rs, np.nan, dom_freq)


class W02_DomOnly_AutoK(LateralMethod):
    """Auto-select number of channels for frequency based on agreement.

    Instead of fixed top-3, keeps adding channels while they agree on
    frequency (CV of per-channel estimates < threshold).
    """
    name = "W02_DomOnly_AutoK"
    description = "Envelope lat + auto-K channel selection for dom-side freq"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        # Rank channels by power
        powers = np.array([np.var(seg_n[ch]) for ch in dom_chs])
        ranked = dom_chs[np.argsort(powers)[::-1]]
        # Add channels one at a time while frequency estimates agree
        ch_freqs = []
        for ch in ranked:
            f, cv, q = _hilbert_freq_cv(seg_n[ch])
            if np.isfinite(f):
                ch_freqs.append(f)
                if len(ch_freqs) >= 2:
                    freq_cv = np.std(ch_freqs) / max(np.mean(ch_freqs), 1e-6)
                    if freq_cv > 0.3:
                        ch_freqs.pop()  # remove disagreeing channel
                        break
        dom_freq = float(np.median(ch_freqs)) if ch_freqs else np.nan
        if ls >= rs:
            return _make_result(ls, rs, dom_freq, np.nan)
        else:
            return _make_result(ls, rs, np.nan, dom_freq)


class W03_DomOnly_QualityWeighted(LateralMethod):
    """Frequency from dominant hemi, weighted by Hilbert CV quality score.

    Channels with more stable instantaneous frequency (lower CV) get more weight.
    """
    name = "W03_DomOnly_QualityWeighted"
    description = "Envelope lat + quality-weighted Hilbert freq from dom hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        freqs, weights = [], []
        for ch in dom_chs:
            f, cv, q = _hilbert_freq_cv(seg_n[ch])
            if np.isfinite(f) and q > 0.0:
                freqs.append(f)
                weights.append(q)
        if freqs:
            dom_freq = float(np.average(freqs, weights=weights))
        else:
            dom_freq = np.nan
        if ls >= rs:
            return _make_result(ls, rs, dom_freq, np.nan)
        else:
            return _make_result(ls, rs, np.nan, dom_freq)


class W04_DomOnly_MultiMethod(LateralMethod):
    """Dominant-side freq using consensus of Hilbert, ACF, and spectral peak.

    Takes median of multiple frequency estimators on the dominant hemisphere.
    """
    name = "W04_DomOnly_MultiMethod"
    description = "Envelope lat + multi-method consensus freq from dom hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        dom_sig, _ = _hemi_top_signal(seg_n, dom_chs, 3)
        estimates = []
        f_h, _, _ = _hilbert_freq_cv(dom_sig)
        if np.isfinite(f_h):
            estimates.append(f_h)
        f_a, _ = _acf_freq(dom_sig)
        if np.isfinite(f_a):
            estimates.append(f_a)
        f_s, _ = _spectral_peak(dom_sig)
        if np.isfinite(f_s):
            estimates.append(f_s)
        dom_freq = float(np.median(estimates)) if estimates else np.nan
        if ls >= rs:
            return _make_result(ls, rs, dom_freq, np.nan)
        else:
            return _make_result(ls, rs, np.nan, dom_freq)


class W05_DomOnly_IterRefine(LateralMethod):
    """Iterative refinement like V12, but frequency strictly from dominant side.

    Two-pass: coarse lateralize → estimate dom freq → narrowband → refined lateralize.
    Frequency output only from the dominant hemisphere.
    """
    name = "W05_DomOnly_IterRefine"
    description = "Iterative narrowband refinement + strict dom-side freq"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        # Pass 1: coarse lateralization
        ls1 = float(np.mean([np.var(seg_n[ch]) for ch in LEFT_CHS]))
        rs1 = float(np.mean([np.var(seg_n[ch]) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls1 >= rs1 else RIGHT_CHS
        # Estimate frequency from dominant hemisphere
        dom_sig, _ = _hemi_top_signal(seg_n, dom_chs, 3)
        est_freq, _, _ = _hilbert_freq_cv(dom_sig)
        if not np.isfinite(est_freq):
            est_freq, _ = _spectral_peak(dom_sig)
        if not np.isfinite(est_freq):
            est_freq = 1.5
        # Pass 2: narrowband at estimated freq
        bw = 0.4
        lo = max(est_freq - bw, 0.1)
        hi = min(est_freq + bw, FS/2 - 0.1)
        if lo < hi:
            sos2 = butter(3, [lo/(FS/2), hi/(FS/2)], btype='bandpass', output='sos')
            seg_nb = sosfiltfilt(sos2, seg_f, axis=1)
        else:
            seg_nb = seg_n
        # Refined lateralization via envelope amplitude
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in RIGHT_CHS]))
        # Refined frequency from dominant side in narrowband
        dom_chs2 = LEFT_CHS if ls >= rs else RIGHT_CHS
        dom_sig2, _ = _hemi_top_signal(seg_nb, dom_chs2, 3)
        refined_freq, _, _ = _hilbert_freq_cv(dom_sig2)
        dom_freq = refined_freq if np.isfinite(refined_freq) else est_freq
        if ls >= rs:
            return _make_result(ls, rs, dom_freq, np.nan)
        else:
            return _make_result(ls, rs, np.nan, dom_freq)


class W06_AutoChannel_EnvThreshold(LateralMethod):
    """Auto-select channels with envelope amplitude above threshold.

    Instead of top-K by power, selects all channels on the dominant side
    whose envelope amplitude exceeds 50% of the max channel's amplitude.
    """
    name = "W06_AutoChannel_EnvThreshold"
    description = "Auto-select channels by envelope threshold + dom freq"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        # Compute per-channel envelope amplitude
        envs = {ch: np.mean(np.abs(hilbert(seg_n[ch]))) for ch in dom_chs}
        max_env = max(envs.values())
        if max_env < 1e-10:
            if ls >= rs:
                return _make_result(ls, rs, np.nan, np.nan)
            else:
                return _make_result(ls, rs, np.nan, np.nan)
        threshold = 0.5 * max_env
        selected = [ch for ch, env in envs.items() if env >= threshold]
        if not selected:
            selected = [max(envs, key=envs.get)]
        # Frequency from selected channels
        freqs, weights = [], []
        for ch in selected:
            f, cv, q = _hilbert_freq_cv(seg_n[ch])
            if np.isfinite(f):
                freqs.append(f)
                weights.append(envs[ch])
        if freqs:
            dom_freq = float(np.average(freqs, weights=weights))
        else:
            dom_freq = np.nan
        if ls >= rs:
            return _make_result(ls, rs, dom_freq, np.nan)
        else:
            return _make_result(ls, rs, np.nan, dom_freq)


class W07_AutoChannel_FreqAgreement(LateralMethod):
    """Select channels on dominant side that agree on frequency.

    Computes per-channel Hilbert freq on all dominant channels, then
    keeps only those within 1 MAD of the median — robust outlier rejection.
    """
    name = "W07_AutoChannel_FreqAgreement"
    description = "Auto-select channels by freq agreement + dom freq"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        # Get all channel frequencies
        ch_freqs = {}
        for ch in dom_chs:
            f, cv, q = _hilbert_freq_cv(seg_n[ch])
            if np.isfinite(f):
                ch_freqs[ch] = f
        if not ch_freqs:
            if ls >= rs:
                return _make_result(ls, rs, np.nan, np.nan)
            else:
                return _make_result(ls, rs, np.nan, np.nan)
        all_f = np.array(list(ch_freqs.values()))
        med = np.median(all_f)
        mad = np.median(np.abs(all_f - med))
        if mad < 0.05:
            mad = 0.05  # floor to avoid over-filtering
        # Keep channels within 1.5 MAD of median
        agreed = {ch: f for ch, f in ch_freqs.items() if abs(f - med) <= 1.5 * mad}
        if not agreed:
            agreed = ch_freqs
        dom_freq = float(np.median(list(agreed.values())))
        if ls >= rs:
            return _make_result(ls, rs, dom_freq, np.nan)
        else:
            return _make_result(ls, rs, np.nan, dom_freq)


class W08_DomOnly_VEFreq(LateralMethod):
    """Variance-explained frequency estimation on dominant side only.

    Uses sine/cosine fitting across a frequency grid, picking the frequency
    that best explains variance in the dominant hemisphere signal.
    """
    name = "W08_DomOnly_VEFreq"
    description = "Envelope lat + variance-explained freq from dom hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        dom_sig, _ = _hemi_top_signal(seg_n, dom_chs, 3)
        dom_freq, ve = _ve_freq(dom_sig)
        if ls >= rs:
            return _make_result(ls, rs, dom_freq, np.nan)
        else:
            return _make_result(ls, rs, np.nan, dom_freq)


class W09_DomOnly_IPIFreq(LateralMethod):
    """Inter-peak interval frequency on dominant side.

    Detects peaks in the dominant hemisphere signal and estimates frequency
    from the regularity of inter-peak intervals.
    """
    name = "W09_DomOnly_IPIFreq"
    description = "Envelope lat + IPI freq from dom hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        dom_sig, _ = _hemi_top_signal(seg_n, dom_chs, 3)
        # IPI frequency
        peaks, _ = find_peaks(dom_sig, distance=int(FS / 3.5))
        if len(peaks) >= 3:
            ipis = np.diff(peaks) / FS
            dom_freq = float(1.0 / np.median(ipis))
        else:
            dom_freq = np.nan
        if ls >= rs:
            return _make_result(ls, rs, dom_freq, np.nan)
        else:
            return _make_result(ls, rs, np.nan, dom_freq)


class W10_DomOnly_EnvPeakFreq(LateralMethod):
    """Frequency from envelope periodicity on dominant side.

    Computes Hilbert envelope on dominant channels, then finds the
    periodicity of the envelope itself via ACF.
    """
    name = "W10_DomOnly_EnvPeakFreq"
    description = "Envelope lat + envelope-periodicity freq from dom hemi"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5/(FS/2), 3.5/(FS/2)], btype='bandpass', output='sos')
        seg_n = sosfiltfilt(sos, seg_f, axis=1)
        ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
        rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
        dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
        # Compute mean envelope across top-3 dom channels
        dom_sig, top_idx = _hemi_top_signal(seg_n, dom_chs, 3)
        env = np.abs(hilbert(dom_sig))
        # ACF of envelope to find periodicity
        dom_freq, acf_val = _acf_freq(env)
        if not np.isfinite(dom_freq):
            # Fallback to Hilbert on raw signal
            dom_freq, _, _ = _hilbert_freq_cv(dom_sig)
        if ls >= rs:
            return _make_result(ls, rs, dom_freq, np.nan)
        else:
            return _make_result(ls, rs, np.nan, dom_freq)
