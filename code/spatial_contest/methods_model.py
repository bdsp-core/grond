"""Model-based spatial localization methods (M1-M4) using existing CNN models."""
import numpy as np
from pathlib import Path
from .base import SpatialMethod, FS, REGIONS, REGION_TO_CHANNELS, LEFT_CHS, RIGHT_CHS

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
MODEL_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'


def _load_cnn_models():
    """Try to load the trained ChannelPDNetAttention models."""
    import torch
    import sys
    sys.path.insert(0, str(CODE_DIR))
    from pd_channel_detector.channel_cnn import ChannelPDNetAttention

    models = []
    for fold in range(5):
        # Try several naming patterns
        for pattern in [
            MODEL_DIR / f'cnn_attn_fold{fold}.pt',
            MODEL_DIR / f'channel_pd_attention_fold{fold}.pt',
            MODEL_DIR / f'channel_pdnet_attention_fold{fold}.pt',
            MODEL_DIR / f'cnn_attention_fold{fold}.pt',
        ]:
            if pattern.exists():
                model = ChannelPDNetAttention()
                model.load_state_dict(torch.load(str(pattern), map_location='cpu', weights_only=True))
                model.eval()
                models.append(model)
                break
    return models


class M1_CNNChannelProbs(SpatialMethod):
    """Use trained ChannelPDNetAttention per-channel PD probabilities."""
    name = "M1_CNNChannelProbs"
    description = "Pretrained CNN per-channel PD probability"

    def __init__(self):
        self._models = None

    def _get_models(self):
        if self._models is None:
            self._models = _load_cnn_models()
        return self._models

    def _analyze(self, seg_bi, subtype):
        import torch
        models = self._get_models()
        n_ch = min(18, seg_bi.shape[0])

        if not models:
            # Fallback: use signal power
            scores = np.array([np.var(seg_bi[ch]) for ch in range(n_ch)])
            mx = scores.max()
            if mx > 0:
                scores = scores / mx
            return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.4}

        probs = np.zeros(n_ch)
        with torch.no_grad():
            for ch in range(n_ch):
                x = seg_bi[ch, :2000].copy()
                mu, sigma = np.mean(x), np.std(x)
                if sigma > 1e-8:
                    x = (x - mu) / sigma
                else:
                    x = x - mu
                tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0)  # (1, 1, 2000)
                fold_probs = []
                for model in models:
                    pd_prob, _, _ = model(tensor)
                    fold_probs.append(float(pd_prob.item()))
                probs[ch] = np.mean(fold_probs)

        mx = probs.max()
        if mx > 0:
            probs = probs / mx
        return {'region_scores': self.channel_scores_to_regions(probs), 'threshold': 0.4}


class M2_CNNAttentionWeighted(SpatialMethod):
    """CNN PD probability weighted by attention temporal concentration."""
    name = "M2_CNNAttentionWeighted"
    description = "CNN prob * attention entropy (focused attention = more certain)"

    def __init__(self):
        self._models = None

    def _get_models(self):
        if self._models is None:
            self._models = _load_cnn_models()
        return self._models

    def _analyze(self, seg_bi, subtype):
        import torch
        models = self._get_models()
        n_ch = min(18, seg_bi.shape[0])

        if not models:
            scores = np.array([np.var(seg_bi[ch]) for ch in range(n_ch)])
            mx = scores.max()
            if mx > 0:
                scores = scores / mx
            return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.4}

        scores = np.zeros(n_ch)
        with torch.no_grad():
            for ch in range(n_ch):
                x = seg_bi[ch, :2000].copy()
                mu, sigma = np.mean(x), np.std(x)
                if sigma > 1e-8:
                    x = (x - mu) / sigma
                else:
                    x = x - mu
                tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0)
                fold_probs = []
                fold_attn_focus = []
                for model in models:
                    pd_prob, _, attn_w = model(tensor)
                    p = float(pd_prob.item())
                    fold_probs.append(p)
                    # Attention focus: negative entropy (higher = more focused)
                    w = attn_w.numpy().flatten()
                    w = w + 1e-12
                    entropy = -np.sum(w * np.log(w))
                    max_entropy = np.log(len(w))
                    focus = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
                    fold_attn_focus.append(focus)
                prob = np.mean(fold_probs)
                focus = np.mean(fold_attn_focus)
                scores[ch] = prob * (0.5 + 0.5 * focus)

        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.4}


class M3_CNNPlusPointiness(SpatialMethod):
    """Blend CNN probability with pointiness for robustness."""
    name = "M3_CNNPlusPointiness"
    description = "0.6*CNN_prob + 0.4*pointiness blend"

    def __init__(self):
        self._models = None

    def _get_models(self):
        if self._models is None:
            self._models = _load_cnn_models()
        return self._models

    def _analyze(self, seg_bi, subtype):
        import torch
        from pd_pointiness_acf import compute_pointiness_trace
        from scipy.ndimage import gaussian_filter1d

        models = self._get_models()
        n_ch = min(18, seg_bi.shape[0])

        # Pointiness
        pt_scores = np.zeros(n_ch)
        for ch in range(n_ch):
            pt = compute_pointiness_trace(seg_bi[ch])
            pt_scores[ch] = np.max(gaussian_filter1d(pt, sigma=4))
        pt_mx = pt_scores.max()
        if pt_mx > 0:
            pt_scores = pt_scores / pt_mx

        if not models:
            return {'region_scores': self.channel_scores_to_regions(pt_scores), 'threshold': 0.35}

        # CNN
        cnn_scores = np.zeros(n_ch)
        with torch.no_grad():
            for ch in range(n_ch):
                x = seg_bi[ch, :2000].copy()
                mu, sigma = np.mean(x), np.std(x)
                if sigma > 1e-8:
                    x = (x - mu) / sigma
                else:
                    x = x - mu
                tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0)
                fold_probs = []
                for model in models:
                    pd_prob, _, _ = model(tensor)
                    fold_probs.append(float(pd_prob.item()))
                cnn_scores[ch] = np.mean(fold_probs)
        cnn_mx = cnn_scores.max()
        if cnn_mx > 0:
            cnn_scores = cnn_scores / cnn_mx

        blend = 0.6 * cnn_scores + 0.4 * pt_scores
        return {'region_scores': self.channel_scores_to_regions(blend), 'threshold': 0.35}


class M4_CNNSymmetryGPD(SpatialMethod):
    """CNN + GPD symmetry enforcement + LPD laterality focus."""
    name = "M4_CNNSymmetryGPD"
    description = "CNN with GPD bilateral symmetry + LPD ipsilateral boost"

    def __init__(self):
        self._models = None

    def _get_models(self):
        if self._models is None:
            self._models = _load_cnn_models()
        return self._models

    def _analyze(self, seg_bi, subtype):
        import torch
        models = self._get_models()
        n_ch = min(18, seg_bi.shape[0])

        if not models:
            scores = np.array([np.var(seg_bi[ch]) for ch in range(n_ch)])
            mx = scores.max()
            if mx > 0:
                scores = scores / mx
            return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.4}

        probs = np.zeros(n_ch)
        with torch.no_grad():
            for ch in range(n_ch):
                x = seg_bi[ch, :2000].copy()
                mu, sigma = np.mean(x), np.std(x)
                if sigma > 1e-8:
                    x = (x - mu) / sigma
                else:
                    x = x - mu
                tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0)
                fold_probs = [float(model(tensor)[0].item()) for model in models]
                probs[ch] = np.mean(fold_probs)

        region_scores = self.channel_scores_to_regions(probs)

        if subtype == 'gpd':
            pairs = [('LF', 'RF'), ('LT', 'RT'), ('LCP', 'RCP'), ('LO', 'RO')]
            for r1, r2 in pairs:
                avg = (region_scores[r1] + region_scores[r2]) / 2.0
                region_scores[r1] = avg
                region_scores[r2] = avg
            threshold = 0.25
        else:
            # LPD: boost ipsilateral side
            left_mean = np.mean([region_scores[r] for r in ['LF', 'LT', 'LCP', 'LO']])
            right_mean = np.mean([region_scores[r] for r in ['RF', 'RT', 'RCP', 'RO']])
            if left_mean > right_mean:
                for r in ['LF', 'LT', 'LCP', 'LO']:
                    region_scores[r] *= 1.2
                for r in ['RF', 'RT', 'RCP', 'RO']:
                    region_scores[r] *= 0.8
            else:
                for r in ['RF', 'RT', 'RCP', 'RO']:
                    region_scores[r] *= 1.2
                for r in ['LF', 'LT', 'LCP', 'LO']:
                    region_scores[r] *= 0.8
            threshold = 0.35

        # Re-normalize
        mx = max(region_scores.values())
        if mx > 0:
            region_scores = {r: min(1.0, v / mx) for r, v in region_scores.items()}

        return {'region_scores': region_scores, 'threshold': threshold}
