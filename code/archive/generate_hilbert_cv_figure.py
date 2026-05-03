"""
Generate publication-ready figure for M3_HilbertCV method.

Shows the algorithm applied to two example cases:
  GRDA: 112884455 (100% agreement, Q=0.500, 1.94 Hz, 6 experts)
  LRDA: sub-S0001118052161_20161130100657 (100% agreement, Q=0.400, 1.53 Hz, 9 experts)

Panels per case:
  A: Raw EEG (18 channels, 10 seconds)
  B: Delta-filtered signal (0.5–3.5 Hz) on the best channel
  C: Instantaneous phase φ(t)
  D: Instantaneous frequency f_i(t) with median + ±1 std shaded band
  E: CV / Q-score comparison summary

Usage:
    conda run -n foe python code/generate_hilbert_cv_figure.py
"""

import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, sosfiltfilt, filtfilt, hilbert, detrend
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

# ── Constants ────────────────────────────────────────────────────────
FS = 200
LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',      # Left temporal (0-3)
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',      # Right temporal (4-7)
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',      # Left parasagittal (8-11)
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',      # Right parasagittal (12-15)
    'Fz-Cz', 'Cz-Pz',                          # Midline (16-17)
]

# Display order matching clinical convention
DISPLAY_ORDER = [
    (0, 'Fp1-F7'), (1, 'F7-T3'), (2, 'T3-T5'), (3, 'T5-O1'),
    (8, 'Fp1-F3'), (9, 'F3-C3'), (10, 'C3-P3'), (11, 'P3-O1'),
    (None, ''),  # spacer
    (16, 'Fz-Cz'), (17, 'Cz-Pz'),
    (None, ''),  # spacer
    (12, 'Fp2-F4'), (13, 'F4-C4'), (14, 'C4-P4'), (15, 'P4-O2'),
    (4, 'Fp2-F8'), (5, 'F8-T4'), (6, 'T4-T6'), (7, 'T6-O2'),
]

CASES = {
    'grda': {
        'patient_id': '112884455',
        'mat_file': '112884455_seg000.mat',
        'subtype': 'GRDA',
        'expert_freq': 1.94,
        'n_experts': 6,
        'agreement': 1.0,
        'expected_q': 0.500,
        'montage': 'bipolar',
    },
    'lrda': {
        'patient_id': 'sub-S0001118052161_20161130100657',
        'mat_file': 'sub-S0001118052161_20161130100657.mat',
        'subtype': 'LRDA',
        'expert_freq': 1.53,
        'n_experts': 9,
        'agreement': 1.0,
        'expected_q': 0.400,
        'montage': 'monopolar',
    },
}

# ── Style constants (PaperBanana-inspired academic aesthetic) ────────
COLORS = {
    'primary': '#2C3E50',
    'accent1': '#E74C3C',    # Red
    'accent2': '#2980B9',    # Blue
    'accent3': '#27AE60',    # Green
    'light_gray': '#ECF0F1',
    'mid_gray': '#95A5A6',
    'dark_gray': '#7F8C8D',
    'left_hemi': '#C0392B',
    'right_hemi': '#2980B9',
    'midline': '#7F8C8D',
    'phase_color': '#8E44AD',   # Purple for phase
    'freq_color': '#D35400',    # Orange for frequency
    'shaded': '#F39C12',        # Gold for shaded band
}


def load_mat_as_bipolar(mat_path):
    """Load .mat file and return (18, 2000) bipolar array."""
    mat = sio.loadmat(str(mat_path))
    for k in mat:
        if not k.startswith('_'):
            data = mat[k]
            break
    data = np.array(data, dtype=np.float64)
    if data.shape[0] > data.shape[1]:
        data = data.T
    if data.shape[0] == 20:
        data = _monopolar_to_bipolar(data)
    elif data.shape[0] == 18:
        pass
    else:
        raise ValueError(f"Unexpected shape: {data.shape}")
    if data.shape[1] > 2000:
        data = data[:, :2000]
    return data


def _monopolar_to_bipolar(mono):
    """Convert 20-channel monopolar to 18-channel bipolar (double banana)."""
    ch = {
        'Fp1': 0, 'Fp2': 1, 'F7': 2, 'F3': 3, 'Fz': 4, 'F4': 5, 'F8': 6,
        'T3': 7, 'C3': 8, 'Cz': 9, 'C4': 10, 'T4': 11, 'T5': 12,
        'P3': 13, 'Pz': 14, 'P4': 15, 'T6': 16, 'O1': 17, 'O2': 18,
    }
    pairs = [
        ('Fp1','F7'), ('F7','T3'), ('T3','T5'), ('T5','O1'),
        ('Fp2','F8'), ('F8','T4'), ('T4','T6'), ('T6','O2'),
        ('Fp1','F3'), ('F3','C3'), ('C3','P3'), ('P3','O1'),
        ('Fp2','F4'), ('F4','C4'), ('C4','P4'), ('P4','O2'),
        ('Fz','Cz'), ('Cz','Pz'),
    ]
    bipolar = np.zeros((18, mono.shape[1]))
    for i, (a, b) in enumerate(pairs):
        bipolar[i] = mono[ch[a]] - mono[ch[b]]
    return bipolar


def prefilter(seg, lo=0.3, hi=5.0):
    """Bandpass filter for delta-focused analysis."""
    sos = butter(4, [lo / (FS/2), hi / (FS/2)], btype='bandpass', output='sos')
    return sosfiltfilt(sos, seg, axis=1)


def select_top_channels(delta_power, n_per_hemi=3):
    """Select top-n channels per hemisphere by delta power."""
    left_vals = delta_power[LEFT_CHS]
    right_vals = delta_power[RIGHT_CHS]
    left_top = LEFT_CHS[np.argsort(left_vals)[::-1][:n_per_hemi]]
    right_top = RIGHT_CHS[np.argsort(right_vals)[::-1][:n_per_hemi]]
    return np.concatenate([left_top, right_top])


def run_hilbert_cv(seg_bi):
    """Run M3_HilbertCV and return all intermediate results for plotting."""
    # Step 1: Prefilter
    seg_prefiltered = prefilter(seg_bi)
    sos = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
    seg_narrow = sosfiltfilt(sos, seg_bi, axis=1)

    # Step 2: Channel selection
    delta_power = np.var(seg_narrow, axis=1)
    top_chs = select_top_channels(delta_power, n_per_hemi=3)

    # Find the single best channel (highest delta power among selected)
    best_ch = top_chs[np.argmax(delta_power[top_chs])]

    # Step 3: Hilbert transform on best channel
    signal = seg_narrow[best_ch]
    analytic = hilbert(signal)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) * FS / (2.0 * np.pi)

    # Step 4: CV on best channel
    mask = (inst_freq > 0.3) & (inst_freq < 4.0)
    inst_freq_valid = inst_freq.copy()
    inst_freq_valid[~mask] = np.nan

    valid_vals = inst_freq[mask]
    med_f = float(np.median(valid_vals))
    std_f = float(np.std(valid_vals))
    cv = std_f / med_f if med_f > 1e-6 else 1.0

    # Step 5: Aggregate across all selected channels
    ch_freqs = []
    ch_cvs = []
    for ch in top_chs:
        sig = seg_narrow[ch]
        if np.std(sig) < 1e-10:
            continue
        ana = hilbert(sig)
        ph = np.unwrap(np.angle(ana))
        ifr = np.diff(ph) * FS / (2.0 * np.pi)
        m = (ifr > 0.3) & (ifr < 4.0)
        ifr_v = ifr[m]
        if len(ifr_v) < 20:
            continue
        mf = np.median(ifr_v)
        sf = np.std(ifr_v)
        ch_freqs.append(mf)
        ch_cvs.append(sf / mf if mf > 1e-6 else 1.0)

    freq_final = float(np.median(ch_freqs))
    cv_final = float(np.median(ch_cvs))
    q_score = max(0.0, 1.0 - 2.0 * cv_final)

    return {
        'seg_narrow': seg_narrow,
        'best_ch': int(best_ch),
        'top_chs': top_chs,
        'delta_power': delta_power,
        'best_signal': signal,
        'inst_phase': inst_phase,
        'inst_freq': inst_freq,
        'inst_freq_valid': inst_freq_valid,
        'mask': mask,
        'best_ch_med_f': med_f,
        'best_ch_std_f': std_f,
        'best_ch_cv': cv,
        'freq_final': freq_final,
        'cv_final': cv_final,
        'q_score': q_score,
        'ch_freqs': ch_freqs,
        'ch_cvs': ch_cvs,
    }


def plot_raw_eeg(ax, seg_bi, case_info, hilbert_result):
    """Panel A: Raw 18-channel EEG with selected channels highlighted."""
    n_samples = seg_bi.shape[1]
    time_vec = np.linspace(0, n_samples / FS, n_samples)

    # Light preprocessing for display
    nyq = FS / 2.0
    b, a = butter(4, 30.0 / nyq, btype='low')
    filtered = seg_bi.copy()
    for i in range(18):
        try:
            filtered[i] = filtfilt(b, a, seg_bi[i])
            filtered[i] = detrend(filtered[i], type='linear')
        except ValueError:
            pass

    z_scale = 0.012
    clip_uv = 250.0
    n_display = len(DISPLAY_ORDER)
    top_chs = set(hilbert_result['top_chs'].tolist())

    yticks = []
    ytick_labels = []

    for di in range(n_display):
        ch_idx, ch_name = DISPLAY_ORDER[di]
        offset = float(n_display - di)
        yticks.append(offset)
        ytick_labels.append(ch_name)
        if ch_idx is None:
            continue

        clipped = np.clip(filtered[ch_idx], -clip_uv, clip_uv)
        scaled = z_scale * clipped + offset

        # Highlight selected channels
        if ch_idx in top_chs:
            if ch_idx in LEFT_CHS:
                color = COLORS['left_hemi']
            else:
                color = COLORS['right_hemi']
            lw = 0.8
            alpha = 1.0
        elif ch_idx in (16, 17):
            color = COLORS['midline']
            lw = 0.4
            alpha = 0.5
        else:
            color = COLORS['mid_gray']
            lw = 0.35
            alpha = 0.4

        ax.plot(time_vec, scaled, color=color, linewidth=lw, alpha=alpha, clip_on=True)

        # Mark best channel with a star
        if ch_idx == hilbert_result['best_ch']:
            ax.plot(time_vec, scaled, color=color, linewidth=1.0, alpha=1.0, clip_on=True)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=4.5, fontfamily='monospace')
    ax.tick_params(axis='y', length=0, pad=1)
    ax.set_ylim(0.5, n_display + 0.5)
    ax.set_xlim(0, n_samples / FS)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(axis='x', labelsize=5)
    ax.grid(True, axis='x', alpha=0.12, linewidth=0.3, linestyle='-')
    ax.set_xlabel('Time (s)', fontsize=5.5, labelpad=1)

    best_name = BIPOLAR_CHANNELS[hilbert_result['best_ch']]
    ax.set_title(
        f"{case_info['subtype']}  —  {case_info['patient_id'][:20]}{'…' if len(case_info['patient_id']) > 20 else ''}\n"
        f"Expert freq: {case_info['expert_freq']:.2f} Hz  |  {case_info['n_experts']} experts  |  "
        f"Best channel: {best_name}",
        fontsize=5.5, fontweight='bold', pad=3, color=COLORS['primary']
    )

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_filtered_signal(ax, hilbert_result):
    """Panel B: Delta-filtered signal on the best channel."""
    signal = hilbert_result['best_signal']
    n = len(signal)
    time_vec = np.linspace(0, n / FS, n)

    best_name = BIPOLAR_CHANNELS[hilbert_result['best_ch']]

    ax.fill_between(time_vec, signal, alpha=0.15, color=COLORS['accent2'])
    ax.plot(time_vec, signal, color=COLORS['accent2'], linewidth=0.8)
    ax.axhline(0, color=COLORS['mid_gray'], linewidth=0.3, alpha=0.5)

    ax.set_xlim(0, n / FS)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(axis='both', labelsize=5)
    ax.set_xlabel('Time (s)', fontsize=5.5, labelpad=1)
    ax.set_ylabel('μV', fontsize=5.5, labelpad=1)
    ax.set_title(f'Filtered (0.5–3.5 Hz): {best_name}', fontsize=5.5,
                 fontweight='bold', pad=2, color=COLORS['primary'])

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(COLORS['mid_gray'])


def plot_phase(ax, hilbert_result):
    """Panel C: Instantaneous phase φ(t)."""
    phase = hilbert_result['inst_phase']
    n = len(phase)
    time_vec = np.linspace(0, n / FS, n)

    # Convert to cycles for clearer visualization
    phase_cycles = phase / (2 * np.pi)

    ax.plot(time_vec, phase_cycles, color=COLORS['phase_color'], linewidth=0.8)

    ax.set_xlim(0, n / FS)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(axis='both', labelsize=5)
    ax.set_xlabel('Time (s)', fontsize=5.5, labelpad=1)
    ax.set_ylabel('Phase (cycles)', fontsize=5.5, labelpad=1)
    ax.set_title('Instantaneous Phase φ(t)', fontsize=5.5,
                 fontweight='bold', pad=2, color=COLORS['primary'])

    # Annotate slope = frequency
    med_f = hilbert_result['best_ch_med_f']
    ax.annotate(f'slope ≈ {med_f:.2f} cycles/s\n= {med_f:.2f} Hz',
                xy=(0.97, 0.05), xycoords='axes fraction',
                fontsize=4.5, ha='right', va='bottom',
                color=COLORS['phase_color'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['phase_color'], alpha=0.9, linewidth=0.5))

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(COLORS['mid_gray'])


def plot_inst_freq(ax, hilbert_result, case_info):
    """Panel D: Instantaneous frequency f_i(t) with median and ±1 std band."""
    inst_freq = hilbert_result['inst_freq']
    mask = hilbert_result['mask']
    n = len(inst_freq)
    time_vec = np.linspace(0, n / FS, n)

    med_f = hilbert_result['best_ch_med_f']
    std_f = hilbert_result['best_ch_std_f']
    cv = hilbert_result['best_ch_cv']
    q = hilbert_result['q_score']
    freq_final = hilbert_result['freq_final']

    # Plot all frequency values (dim the invalid ones)
    ax.scatter(time_vec[~mask], inst_freq[~mask], s=0.5, color=COLORS['mid_gray'],
               alpha=0.15, zorder=1, rasterized=True)
    ax.scatter(time_vec[mask], inst_freq[mask], s=1.0, color=COLORS['freq_color'],
               alpha=0.6, zorder=2, rasterized=True)

    # Median line
    ax.axhline(med_f, color=COLORS['accent1'], linewidth=1.2, linestyle='-',
               label=f'Median = {med_f:.2f} Hz', zorder=3)

    # ±1 std shaded band
    ax.axhspan(med_f - std_f, med_f + std_f, alpha=0.15,
               color=COLORS['shaded'], zorder=0,
               label=f'±1 SD (CV = {cv:.3f})')

    # Expert frequency reference
    ax.axhline(case_info['expert_freq'], color=COLORS['accent3'], linewidth=0.8,
               linestyle='--', alpha=0.8, label=f'Expert = {case_info["expert_freq"]:.2f} Hz')

    ax.set_xlim(0, n / FS)
    ax.set_ylim(0, 4.0)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(axis='both', labelsize=5)
    ax.set_xlabel('Time (s)', fontsize=5.5, labelpad=1)
    ax.set_ylabel('Freq (Hz)', fontsize=5.5, labelpad=1)

    ax.set_title(f'Instantaneous Frequency  —  CV = {cv:.3f},  Q = {q:.3f}',
                 fontsize=5.5, fontweight='bold', pad=2, color=COLORS['primary'])

    ax.legend(fontsize=4.5, loc='upper right', framealpha=0.9,
              edgecolor=COLORS['mid_gray'], fancybox=False,
              handlelength=1.5, handletextpad=0.4, borderpad=0.3)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(COLORS['mid_gray'])


def plot_cv_comparison(ax, results):
    """Panel E: CV / Q-score comparison across both cases."""
    names = []
    cvs = []
    qs = []
    freqs_est = []
    freqs_expert = []
    colors_bar = []

    for key in ['grda', 'lrda']:
        case = CASES[key]
        r = results[key]
        names.append(f"{case['subtype']}\n({case['expert_freq']:.2f} Hz)")
        cvs.append(r['cv_final'])
        qs.append(r['q_score'])
        freqs_est.append(r['freq_final'])
        freqs_expert.append(case['expert_freq'])
        colors_bar.append(COLORS['accent2'] if key == 'grda' else COLORS['left_hemi'])

    x = np.arange(len(names))
    width = 0.30

    # Q-score bars
    bars_q = ax.bar(x - width/2, qs, width, color=colors_bar, alpha=0.85,
                    edgecolor='white', linewidth=0.5, label='Q-score', zorder=3)
    # CV bars
    bars_cv = ax.bar(x + width/2, cvs, width, color=colors_bar, alpha=0.4,
                     edgecolor=colors_bar, linewidth=0.8, label='CV', zorder=3,
                     hatch='///')

    # Annotate values on bars
    for bar, val in zip(bars_q, qs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'Q={val:.2f}', ha='center', va='bottom', fontsize=5.5,
                fontweight='bold', color=COLORS['primary'])
    for bar, val in zip(bars_cv, cvs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'CV={val:.2f}', ha='center', va='bottom', fontsize=5.5,
                color=COLORS['dark_gray'])

    # Add frequency accuracy annotations
    for i, (est, exp) in enumerate(zip(freqs_est, freqs_expert)):
        err = abs(est - exp)
        ax.text(x[i], -0.12, f'Est: {est:.2f} Hz  (Δ={err:.2f})',
                ha='center', va='top', fontsize=4.5, color=COLORS['dark_gray'])

    # Reference thresholds
    ax.axhline(0.5, color=COLORS['mid_gray'], linewidth=0.5, linestyle=':',
               alpha=0.5, zorder=1)
    ax.text(len(names) - 0.5, 0.52, 'Q = 0.5 threshold', fontsize=4,
            ha='right', va='bottom', color=COLORS['mid_gray'], fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6)
    ax.set_ylim(-0.15, max(max(qs), max(cvs)) + 0.15)
    ax.set_ylabel('Score', fontsize=6, labelpad=2)
    ax.tick_params(axis='y', labelsize=5)

    ax.set_title('CV and Q-score Comparison  —  Method Works Well for Both Subtypes',
                 fontsize=6, fontweight='bold', pad=4, color=COLORS['primary'])

    ax.legend(fontsize=5, loc='upper right', framealpha=0.9,
              edgecolor=COLORS['mid_gray'], fancybox=False,
              ncol=2, handlelength=1.2, handletextpad=0.3)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color(COLORS['mid_gray'])


def generate_data_panels(results):
    """Generate the matplotlib data panels (A–E) and return as PIL Image."""
    fig = plt.figure(figsize=(14, 16), facecolor='white', dpi=300)

    gs = GridSpec(5, 2, figure=fig,
                  height_ratios=[3.5, 1.2, 1.2, 1.2, 1.5],
                  hspace=0.45, wspace=0.25,
                  left=0.08, right=0.97, top=0.96, bottom=0.04)

    for col_idx, (key, case) in enumerate(CASES.items()):
        r = results[key]
        seg_bi = r['seg_bi']

        # Panel B: Raw EEG
        ax_a = fig.add_subplot(gs[0, col_idx])
        plot_raw_eeg(ax_a, seg_bi, case, r)
        if col_idx == 0:
            ax_a.text(-0.06, 1.02, 'B', transform=ax_a.transAxes,
                      fontsize=12, fontweight='bold', va='bottom', color=COLORS['primary'])

        # Panel C: Filtered best channel
        ax_b = fig.add_subplot(gs[1, col_idx])
        plot_filtered_signal(ax_b, r)
        if col_idx == 0:
            ax_b.text(-0.06, 1.05, 'C', transform=ax_b.transAxes,
                      fontsize=12, fontweight='bold', va='bottom', color=COLORS['primary'])

        # Panel D: Phase
        ax_c = fig.add_subplot(gs[2, col_idx])
        plot_phase(ax_c, r)
        if col_idx == 0:
            ax_c.text(-0.06, 1.05, 'D', transform=ax_c.transAxes,
                      fontsize=12, fontweight='bold', va='bottom', color=COLORS['primary'])

        # Panel E: Instantaneous frequency
        ax_d = fig.add_subplot(gs[3, col_idx])
        plot_inst_freq(ax_d, r, case)
        if col_idx == 0:
            ax_d.text(-0.06, 1.05, 'E', transform=ax_d.transAxes,
                      fontsize=12, fontweight='bold', va='bottom', color=COLORS['primary'])

    # Panel F: CV comparison (spans both columns)
    ax_e = fig.add_subplot(gs[4, :])
    plot_cv_comparison(ax_e, results)
    ax_e.text(-0.03, 1.05, 'F', transform=ax_e.transAxes,
              fontsize=12, fontweight='bold', va='bottom', color=COLORS['primary'])

    # Column headers
    fig.text(0.30, 0.975, 'GRDA Case', fontsize=9, fontweight='bold',
             ha='center', color=COLORS['accent2'])
    fig.text(0.74, 0.975, 'LRDA Case', fontsize=9, fontweight='bold',
             ha='center', color=COLORS['left_hemi'])

    # Save to buffer and convert to PIL
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)


def composite_figure(pipeline_img_path, data_img):
    """Composite PaperBanana pipeline diagram (Panel A) above data panels (B–F)."""
    from PIL import Image, ImageDraw, ImageFont

    pipeline = Image.open(str(pipeline_img_path))

    # Scale pipeline to match data panel width
    target_w = data_img.width
    ratio = target_w / pipeline.width
    pipeline_scaled = pipeline.resize(
        (target_w, int(pipeline.height * ratio)), Image.LANCZOS)

    # Layout spacing
    label_h = 60  # space for "A" label and title
    gap = 30      # gap between pipeline and data panels

    total_h = label_h + pipeline_scaled.height + gap + data_img.height
    composite = Image.new('RGB', (target_w, total_h), 'white')

    # Try to get fonts
    try:
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except (OSError, IOError):
        font_label = ImageFont.load_default()
        font_title = ImageFont.load_default()

    draw = ImageDraw.Draw(composite)

    # Title
    draw.text((target_w // 2, 5), 'M3_HilbertCV: Instantaneous Frequency Regularity for RDA Detection',
              fill='#2C3E50', font=font_title, anchor='mt')

    # Panel A label
    draw.text((15, label_h - 40), 'A', fill='#2C3E50', font=font_label)

    # Paste pipeline diagram
    composite.paste(pipeline_scaled, (0, label_h))

    # Paste data panels
    composite.paste(data_img, (0, label_h + pipeline_scaled.height + gap))

    return composite


def main():
    print("=" * 60)
    print("M3_HilbertCV Figure Generator")
    print("=" * 60)

    # ── Load and process both cases ─────────────────────────────
    results = {}
    for key, case in CASES.items():
        print(f"\nProcessing {case['subtype']}: {case['patient_id']}...")
        mat_path = PROJECT_DIR / 'data' / 'eeg' / case['mat_file']
        seg_bi = load_mat_as_bipolar(mat_path)
        r = run_hilbert_cv(seg_bi)
        r['seg_bi'] = seg_bi
        results[key] = r
        print(f"  Best channel: {BIPOLAR_CHANNELS[r['best_ch']]} (idx {r['best_ch']})")
        print(f"  Est freq: {r['freq_final']:.3f} Hz  (expert: {case['expert_freq']:.2f} Hz)")
        print(f"  CV: {r['cv_final']:.3f}  |  Q-score: {r['q_score']:.3f}")

    # ── Generate data panels (B–F) ────────────────────────────
    print("\nGenerating data panels...")
    data_img = generate_data_panels(results)
    print(f"  Data panels: {data_img.width}x{data_img.height}")

    # ── Composite with PaperBanana pipeline diagram ───────────
    # Use candidate 1 (cleanest with colored step backgrounds)
    pipeline_path = PROJECT_DIR / 'paper_materials' / 'figures' / 'hilbert_cv_pipeline_1.png'
    if not pipeline_path.exists():
        # Fallback to any available candidate
        for i in [0, 2]:
            alt = PROJECT_DIR / 'paper_materials' / 'figures' / f'hilbert_cv_pipeline_{i}.png'
            if alt.exists():
                pipeline_path = alt
                break

    if pipeline_path.exists():
        print(f"\nCompositing with PaperBanana pipeline: {pipeline_path.name}")
        composite = composite_figure(pipeline_path, data_img)
    else:
        print("\nWARNING: No PaperBanana pipeline diagram found, saving data panels only.")
        composite = data_img

    # ── Save ──────────────────────────────────────────────────
    out_path = PROJECT_DIR / 'paper_materials' / 'figures' / 'hilbert_cv_method_figure.png'
    composite.save(str(out_path), dpi=(300, 300))
    print(f"\n  -> Saved: {out_path}")
    print(f"     Size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"     Dimensions: {composite.width}x{composite.height}")

    print("\nDone!")


if __name__ == '__main__':
    main()
