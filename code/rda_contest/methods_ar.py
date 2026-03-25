"""AR model-based RDA analysis methods."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from .base import RDAMethod, FS, LEFT_CHS, RIGHT_CHS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yule_walker_ar2(x: np.ndarray):
    """Fit AR(2) via Yule-Walker equations.

    Parameters
    ----------
    x : 1-d signal (already filtered / zero-mean recommended)

    Returns
    -------
    a : (2,) AR coefficients  [a1, a2]  for  x[n] = a1*x[n-1] + a2*x[n-2] + e[n]
    sigma2 : residual variance estimate
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    N = len(x)
    if N < 10:
        return np.array([0.0, 0.0]), np.var(x) if np.var(x) > 0 else 1.0

    # Biased autocorrelation
    r = np.correlate(x, x, mode='full')[N - 1:]  # lags 0, 1, 2, ...
    r = r / N  # biased
    r0, r1, r2 = r[0], r[1], r[2]

    if r0 < 1e-15:
        return np.array([0.0, 0.0]), 1e-15

    # Yule-Walker for order 2:
    #   [r0  r1] [a1]   [r1]
    #   [r1  r0] [a2] = [r2]
    denom = r0 * r0 - r1 * r1
    if abs(denom) < 1e-30:
        return np.array([0.0, 0.0]), r0
    a1 = (r1 * r0 - r2 * r1) / denom
    a2 = (r2 * r0 - r1 * r1) / denom
    sigma2 = r0 - a1 * r1 - a2 * r2
    sigma2 = max(sigma2, 1e-15)
    return np.array([a1, a2]), sigma2


def _ar2_poles(a1: float, a2: float):
    """Roots of z^2 - a1*z - a2 = 0.

    Returns (poles, freqs_hz, radii) — arrays of length 2.
    """
    coeffs = np.array([1.0, -a1, -a2])
    poles = np.roots(coeffs)
    radii = np.abs(poles)
    angles = np.angle(poles)
    freqs = np.abs(angles) * FS / (2 * np.pi)
    return poles, freqs, radii


def _ar_residual_variance(x: np.ndarray, a: np.ndarray):
    """Compute residual variance for AR model with coefficients a.

    x[n] = a[0]*x[n-1] + a[1]*x[n-2] + ... + e[n]
    """
    p = len(a)
    N = len(x)
    if N <= p:
        return np.var(x) if np.var(x) > 0 else 1.0
    residuals = np.empty(N - p)
    for n in range(p, N):
        pred = 0.0
        for k in range(p):
            pred += a[k] * x[n - 1 - k]
        residuals[n - p] = x[n] - pred
    return np.var(residuals) if len(residuals) > 0 else 1.0


def _yule_walker(x: np.ndarray, order: int):
    """General Yule-Walker AR(p) fit.

    Returns (a, sigma2) where a has length `order`.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    N = len(x)
    if N < 2 * order:
        return np.zeros(order), np.var(x) if np.var(x) > 0 else 1.0

    r = np.correlate(x, x, mode='full')[N - 1:]
    r = r / N
    if r[0] < 1e-15:
        return np.zeros(order), 1e-15

    # Build Toeplitz matrix
    R = np.empty((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = r[abs(i - j)]
    rhs = r[1:order + 1]

    try:
        a = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        return np.zeros(order), r[0]

    sigma2 = r[0] - np.dot(a, rhs)
    sigma2 = max(sigma2, 1e-15)
    return a, sigma2


def _select_top_channels(values: np.ndarray, n_per_hemi: int = 3):
    """Select top-n channels per hemisphere by `values`, return indices.

    values should have length >= 16 (first 16 channels used for selection).
    """
    left_idx = LEFT_CHS
    right_idx = RIGHT_CHS
    left_vals = values[left_idx]
    right_vals = values[right_idx]

    left_top = left_idx[np.argsort(left_vals)[::-1][:n_per_hemi]]
    right_top = right_idx[np.argsort(right_vals)[::-1][:n_per_hemi]]
    return np.concatenate([left_top, right_top])


# ---------------------------------------------------------------------------
# M1: AR(2) Oscillator — pole-based frequency and rhythmicity
# ---------------------------------------------------------------------------

class M1_AR2Oscillator(RDAMethod):
    name = "M1_AR2Oscillator"
    description = "AR(2) pole analysis: frequency from pole angle, rhythmicity from pole radius"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        seg = self.prefilter(seg_bi)
        n_ch = seg.shape[0]

        pole_radii = np.zeros(n_ch)
        pole_freqs = np.full(n_ch, np.nan)
        in_delta = np.zeros(n_ch, dtype=bool)

        for ch in range(n_ch):
            a, _ = _yule_walker_ar2(seg[ch])
            poles, freqs, radii = _ar2_poles(a[0], a[1])

            # Look for complex conjugate poles in delta band
            for i in range(2):
                if np.iscomplex(poles[i]) and 0.5 <= freqs[i] <= 3.5:
                    if radii[i] > pole_radii[ch]:
                        pole_radii[ch] = radii[i]
                        pole_freqs[ch] = freqs[i]
                        in_delta[ch] = True

        # Laterality-aware selection: top-3 per hemisphere by pole radius
        top_chs = _select_top_channels(pole_radii, n_per_hemi=3)
        # Filter to those actually in delta band
        valid = top_chs[in_delta[top_chs]]

        if len(valid) == 0:
            return {'freq': np.nan, 'q_score': 0.0, 'extras': {
                'pole_radii': pole_radii, 'pole_freqs': pole_freqs
            }}

        freq = float(np.median(pole_freqs[valid]))
        q_score = float(np.median(pole_radii[valid]))

        return {
            'freq': freq,
            'q_score': q_score,
            'extras': {
                'pole_radii': pole_radii,
                'pole_freqs': pole_freqs,
                'valid_channels': valid,
            },
        }


# ---------------------------------------------------------------------------
# M2: AR(2) Likelihood Ratio — constrained delta AR(2) vs broadband AR(6)
# ---------------------------------------------------------------------------

def _constrained_ar2_grid(x: np.ndarray):
    """Find best AR(2) coefficients with poles constrained to delta band.

    Poles must be complex conjugate with angle in delta band (0.5-3.5 Hz)
    and radius > 0.7.

    Parameterise AR(2) by (r, f):
        poles = r * exp(±j*2π*f/fs)
        a1 = 2*r*cos(2π*f/fs)
        a2 = -r^2
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()

    best_var = np.inf
    best_a = np.array([0.0, 0.0])
    best_f = np.nan
    best_r = 0.0

    freq_grid = np.arange(0.5, 3.55, 0.1)
    r_grid = np.arange(0.70, 0.995, 0.02)

    for f in freq_grid:
        theta = 2.0 * np.pi * f / FS
        cos_theta = np.cos(theta)
        for r in r_grid:
            a1 = 2.0 * r * cos_theta
            a2 = -(r * r)
            a = np.array([a1, a2])
            rv = _ar_residual_variance(x, a)
            if rv < best_var:
                best_var = rv
                best_a = a.copy()
                best_f = f
                best_r = r

    return best_a, best_var, best_f, best_r


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        return ex / (1.0 + ex)


class M2_AR2LikelihoodRatio(RDAMethod):
    name = "M2_AR2LikelihoodRatio"
    description = "Log-likelihood ratio: constrained delta AR(2) vs broadband AR(6)"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        seg = self.prefilter(seg_bi)
        n_ch = seg.shape[0]

        lr_values = np.full(n_ch, np.nan)
        ar2_freqs = np.full(n_ch, np.nan)
        ar2_radii = np.zeros(n_ch)

        for ch in range(n_ch):
            signal = seg[ch]

            # --- Fit AR(2) ---
            a2, sigma2_ar2 = _yule_walker_ar2(signal)
            poles, freqs, radii = _ar2_poles(a2[0], a2[1])

            # Check if AR(2) poles are in delta band with r > 0.7
            use_grid = True
            for i in range(2):
                if (np.iscomplex(poles[i]) and 0.5 <= freqs[i] <= 3.5
                        and radii[i] > 0.7):
                    ar2_freqs[ch] = freqs[i]
                    ar2_radii[ch] = radii[i]
                    sigma2_ar2 = _ar_residual_variance(signal, a2)
                    use_grid = False
                    break

            if use_grid:
                a2_c, sigma2_ar2, f_c, r_c = _constrained_ar2_grid(signal)
                ar2_freqs[ch] = f_c
                ar2_radii[ch] = r_c

            # --- Fit AR(6) ---
            a6, _ = _yule_walker(signal, order=6)
            sigma2_ar6 = _ar_residual_variance(signal, a6)

            # Log-likelihood ratio: positive means delta AR(2) fits better
            if sigma2_ar2 > 1e-15 and sigma2_ar6 > 1e-15:
                lr_values[ch] = np.log(sigma2_ar6 / sigma2_ar2)

        # Laterality-aware: top-3 per hemisphere by likelihood ratio
        lr_for_select = np.where(np.isfinite(lr_values), lr_values, -np.inf)
        top_chs = _select_top_channels(lr_for_select, n_per_hemi=3)
        valid_lr = lr_values[top_chs]
        valid_lr = valid_lr[np.isfinite(valid_lr)]

        if len(valid_lr) == 0:
            return {'freq': np.nan, 'q_score': 0.0, 'extras': {}}

        mean_lr = float(np.mean(valid_lr))
        q_score = _sigmoid(mean_lr)

        # Freq from AR(2) poles of top channels
        valid_freqs = ar2_freqs[top_chs]
        valid_freqs = valid_freqs[np.isfinite(valid_freqs)]
        freq = float(np.median(valid_freqs)) if len(valid_freqs) > 0 else np.nan

        return {
            'freq': freq,
            'q_score': q_score,
            'extras': {
                'lr_values': lr_values,
                'ar2_freqs': ar2_freqs,
                'ar2_radii': ar2_radii,
                'mean_lr': mean_lr,
            },
        }


# ---------------------------------------------------------------------------
# M3: Hilbert CV — instantaneous frequency regularity
# ---------------------------------------------------------------------------

class M3_HilbertCV(RDAMethod):
    name = "M3_HilbertCV"
    description = "Instantaneous frequency CV via Hilbert transform; low CV = regular RDA"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        seg = self.prefilter(seg_bi)

        # Additional narrowband filter for Hilbert analysis
        sos = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)],
                     btype='bandpass', output='sos')
        seg_narrow = sosfiltfilt(sos, seg, axis=1)

        n_ch = seg_narrow.shape[0]
        delta_power = np.var(seg_narrow, axis=1)

        # Select top-3 per hemisphere by delta power
        top_chs = _select_top_channels(delta_power, n_per_hemi=3)

        ch_freqs = []
        ch_cvs = []

        for ch in top_chs:
            signal = seg_narrow[ch]
            if np.std(signal) < 1e-10:
                continue

            analytic = hilbert(signal)
            inst_phase = np.unwrap(np.angle(analytic))
            # Instantaneous frequency = d(phase)/dt / (2*pi)
            inst_freq = np.diff(inst_phase) * FS / (2.0 * np.pi)

            # Keep only reasonable values
            mask = (inst_freq > 0.3) & (inst_freq < 4.0)
            inst_freq_valid = inst_freq[mask]

            if len(inst_freq_valid) < 20:
                continue

            med_f = np.median(inst_freq_valid)
            std_f = np.std(inst_freq_valid)
            cv = std_f / med_f if med_f > 1e-6 else 1.0

            ch_freqs.append(med_f)
            ch_cvs.append(cv)

        if len(ch_freqs) == 0:
            return {'freq': np.nan, 'q_score': 0.0, 'extras': {}}

        freq = float(np.median(ch_freqs))
        cv_median = float(np.median(ch_cvs))
        q_score = max(0.0, 1.0 - 2.0 * cv_median)

        return {
            'freq': freq,
            'q_score': q_score,
            'extras': {
                'cv_median': cv_median,
                'ch_cvs': ch_cvs,
                'ch_freqs': ch_freqs,
                'top_channels': top_chs,
            },
        }
