"""CNN-based RDA contest methods using the UnifiedPDModel.

The UnifiedPDModel is a multi-task CNN trained on IIIC data with four heads:
  1. 4-class subtype classification (LPD=0, GPD=1, LRDA=2, GRDA=3)
  2. Frequency estimation (log Hz)
  3. Per-channel PD detection
  4. Per-channel RDA detection

Methods:
  CNN1_SubtypeSoftmax: Q = softmax(LRDA) + softmax(GRDA), freq from M3_HilbertCV
  CNN2_RDAChannelMax:  Q = max per-channel RDA prob, freq from M3_HilbertCV
  CNN3_Unified:        Q = softmax RDA prob, freq = CNN freq head (exp of log pred)
  CNN4_SubtypeXChannel: Q = subtype_rda * channel_rda (product), freq from Hilbert
"""

import sys
import numpy as np
import torch
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, hilbert

from .base import RDAMethod, FS

# Paths
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
UNIFIED_CACHE = PROJECT_DIR / 'data' / 'unified_model_cache'

# Lazy-loaded model ensemble
_model_ensemble = None
_device = None


def _load_models():
    """Load the 5-fold UnifiedPDModel ensemble (lazy, cached)."""
    global _model_ensemble, _device

    if _model_ensemble is not None:
        return _model_ensemble, _device

    sys.path.insert(0, str(CODE_DIR))
    from unified_model.model import UnifiedPDModel

    if torch.backends.mps.is_available():
        _device = torch.device('mps')
    elif torch.cuda.is_available():
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    print(f"  CNN device: {_device}")
    _model_ensemble = []

    for fold in range(5):
        weight_path = UNIFIED_CACHE / f'unified_fold{fold}.pt'
        if not weight_path.exists():
            print(f"  Warning: missing {weight_path}")
            continue

        model = UnifiedPDModel()
        state_dict = torch.load(str(weight_path), map_location=_device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        _model_ensemble.append(model)

    if not _model_ensemble:
        raise RuntimeError(f"No unified model weights found in {UNIFIED_CACHE}")

    print(f"  Loaded {len(_model_ensemble)} UnifiedPDModel folds")
    return _model_ensemble, _device


def _run_ensemble(seg_bi: np.ndarray):
    """Run the model ensemble on a (18, 2000) segment.

    Returns averaged predictions:
        subtype_probs: (4,) softmax probabilities [LPD, GPD, LRDA, GRDA]
        freq_log: float, log(Hz) frequency prediction
        rda_channel_probs: (18,) per-channel RDA probability
        pd_channel_probs: (18,) per-channel PD probability
    """
    models, device = _load_models()

    # Prepare input tensor: (1, 18, 2000)
    x = torch.tensor(seg_bi, dtype=torch.float32).unsqueeze(0).to(device)

    all_subtype = []
    all_freq = []
    all_rda_ch = []
    all_pd_ch = []

    with torch.no_grad():
        for model in models:
            subtype_logits, freq_pred, pd_ch, rda_ch = model(x)

            subtype_probs = torch.softmax(subtype_logits, dim=-1).squeeze(0).cpu().numpy()
            all_subtype.append(subtype_probs)
            all_freq.append(freq_pred.squeeze().cpu().item())
            all_rda_ch.append(rda_ch.squeeze(0).cpu().numpy())
            all_pd_ch.append(pd_ch.squeeze(0).cpu().numpy())

    return {
        'subtype_probs': np.mean(all_subtype, axis=0),   # (4,)
        'freq_log': np.mean(all_freq),                     # scalar
        'rda_channel_probs': np.mean(all_rda_ch, axis=0),  # (18,)
        'pd_channel_probs': np.mean(all_pd_ch, axis=0),    # (18,)
    }


# ─── Hilbert frequency helper (from M3_HilbertCV) ────────────────────────

LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])


def _hilbert_freq(seg_bi: np.ndarray) -> float:
    """Estimate frequency via Hilbert instantaneous frequency on best hemisphere.

    Mirrors M3_HilbertCV logic.
    """
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg = sosfiltfilt(sos_pre, seg_bi, axis=1)

    sos_narrow = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
    seg_narrow = sosfiltfilt(sos_narrow, seg, axis=1)

    delta_power = np.var(seg_narrow, axis=1)

    # Top-3 per hemisphere
    left_idx = LEFT_CHS[np.argsort(delta_power[LEFT_CHS])[::-1][:3]]
    right_idx = RIGHT_CHS[np.argsort(delta_power[RIGHT_CHS])[::-1][:3]]
    if np.mean(delta_power[left_idx]) > np.mean(delta_power[right_idx]):
        top_chs = left_idx
    else:
        top_chs = right_idx

    ch_freqs = []
    for ch in top_chs:
        signal = seg_narrow[ch]
        if np.std(signal) < 1e-10:
            continue
        analytic = hilbert(signal)
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(inst_phase) * FS / (2.0 * np.pi)
        mask = (inst_freq > 0.3) & (inst_freq < 4.0)
        inst_freq_valid = inst_freq[mask]
        if len(inst_freq_valid) < 20:
            continue
        ch_freqs.append(float(np.median(inst_freq_valid)))

    if not ch_freqs:
        return 1.0  # fallback
    return float(np.median(ch_freqs))


# ─── Contest Methods ──────────────────────────────────────────────────────

class CNN1_SubtypeSoftmax(RDAMethod):
    """Q = P(LRDA) + P(GRDA) from CNN subtype softmax. Freq from Hilbert."""
    name = "CNN1_SubtypeSoftmax"
    description = "CNN subtype softmax: P(LRDA)+P(GRDA) as RDA quality, Hilbert freq"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        preds = _run_ensemble(seg_bi)
        # subtype_probs: [LPD=0, GPD=1, LRDA=2, GRDA=3]
        q_score = float(preds['subtype_probs'][2] + preds['subtype_probs'][3])
        freq = _hilbert_freq(seg_bi)
        return {
            'freq': freq,
            'q_score': q_score,
            'extras': {
                'subtype_probs': preds['subtype_probs'].tolist(),
            },
        }


class CNN2_RDAChannelMax(RDAMethod):
    """Q = max per-channel RDA probability. Freq from Hilbert."""
    name = "CNN2_RDAChannelMax"
    description = "CNN per-channel RDA head: max P(RDA) across 18 channels, Hilbert freq"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        preds = _run_ensemble(seg_bi)
        q_score = float(np.max(preds['rda_channel_probs']))
        freq = _hilbert_freq(seg_bi)
        return {
            'freq': freq,
            'q_score': q_score,
            'extras': {
                'rda_channel_probs': preds['rda_channel_probs'].tolist(),
            },
        }


class CNN3_Unified(RDAMethod):
    """Q = subtype RDA prob, freq = CNN frequency head prediction."""
    name = "CNN3_Unified"
    description = "CNN subtype softmax for Q, CNN freq head for frequency (fully CNN)"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        preds = _run_ensemble(seg_bi)
        q_score = float(preds['subtype_probs'][2] + preds['subtype_probs'][3])

        # Frequency: CNN predicts log(Hz), convert back
        freq_hz = float(np.exp(preds['freq_log']))
        # Clamp to reasonable RDA range
        freq_hz = float(np.clip(freq_hz, 0.25, 3.5))

        return {
            'freq': freq_hz,
            'q_score': q_score,
            'extras': {
                'freq_log': preds['freq_log'],
                'subtype_probs': preds['subtype_probs'].tolist(),
            },
        }


class CNN4_SubtypeXChannel(RDAMethod):
    """Q = subtype_rda * mean_top3_channel_rda (product of both heads)."""
    name = "CNN4_SubtypeXChannel"
    description = "Product of subtype P(RDA) and mean top-3 channel P(RDA), Hilbert freq"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        preds = _run_ensemble(seg_bi)

        subtype_rda = float(preds['subtype_probs'][2] + preds['subtype_probs'][3])

        # Mean of top-3 channel RDA probabilities
        ch_rda = preds['rda_channel_probs']
        top3_rda = float(np.mean(np.sort(ch_rda)[::-1][:3]))

        q_score = subtype_rda * top3_rda
        freq = _hilbert_freq(seg_bi)

        return {
            'freq': freq,
            'q_score': q_score,
            'extras': {
                'subtype_rda': subtype_rda,
                'top3_channel_rda': top3_rda,
            },
        }
