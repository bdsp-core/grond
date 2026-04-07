"""Compare PLV vs CNN vs Hybrid for spatial PD localization."""
import sys, time, warnings, numpy as np
warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spatial_contest.harness import load_spatial_data, run_method, evaluate, save_result
from spatial_contest.base import SpatialMethod, FS
from spatial_contest.methods_crosschannel import X5_PhaseCoherence, X2_CrossCorrPeak
from spatial_contest.methods_model import M1_CNNChannelProbs, _load_cnn_models
from scipy.signal import butter, sosfiltfilt, hilbert


class Hybrid_CNN_PLV(SpatialMethod):
    """CNN identifies best channels, PLV finds which others are phase-locked."""
    name = "Hybrid_CNN_PLV"
    description = "CNN picks reference channels, PLV identifies connected channels"

    def __init__(self):
        self._models = None

    def _analyze(self, seg_bi, subtype):
        import torch
        if self._models is None:
            self._models = _load_cnn_models()

        n_ch = min(18, seg_bi.shape[0])

        # Step 1: CNN per-channel PD probabilities
        if self._models:
            probs = np.zeros(n_ch)
            for ch in range(n_ch):
                ch_sig = seg_bi[ch:ch+1, :].astype(np.float32)
                std = np.std(ch_sig)
                if std > 1e-8:
                    ch_sig = (ch_sig - np.mean(ch_sig)) / std
                x = torch.tensor(ch_sig[np.newaxis, :, :], dtype=torch.float32)
                fold_probs = []
                with torch.no_grad():
                    for m in self._models:
                        out = m(x)
                        if isinstance(out, tuple):
                            p = torch.sigmoid(out[0]).item()
                        else:
                            p = torch.sigmoid(out).item()
                        fold_probs.append(p)
                probs[ch] = np.mean(fold_probs)
        else:
            probs = np.array([np.var(seg_bi[ch]) for ch in range(n_ch)])
            mx = probs.max()
            if mx > 0:
                probs = probs / mx

        # Step 2: Use top CNN channels as reference for PLV
        top_chs = np.argsort(probs)[::-1][:3]

        sos = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
        seg_f = sosfiltfilt(sos, seg_bi[:n_ch], axis=1)

        phases = np.zeros_like(seg_f)
        for ch in range(n_ch):
            phases[ch] = np.angle(hilbert(seg_f[ch]))

        ref_phase = np.mean(np.exp(1j * phases[top_chs]), axis=0)
        ref_phase = np.angle(ref_phase)

        scores = np.zeros(n_ch)
        for ch in range(n_ch):
            phase_diff = phases[ch] - ref_phase
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            scores[ch] = probs[ch] * 0.5 + plv * 0.5

        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.4}


def main():
    print('Loading spatial data...')
    data = load_spatial_data()

    methods = [
        X5_PhaseCoherence(),
        X2_CrossCorrPeak(),
        M1_CNNChannelProbs(),
        Hybrid_CNN_PLV(),
    ]

    print(f"\n{'Method':<25} {'MacF1':>6} {'MicF1':>6} {'Jaccard':>8} {'AUC':>6} {'Composite':>10}")
    print("-" * 65)

    for m in methods:
        t0 = time.time()
        results = run_method(m, data, verbose=False)
        metrics = evaluate(results, data)
        elapsed = time.time() - t0
        mac = metrics.get('macro_f1', 0) or 0
        mic = metrics.get('micro_f1', 0) or 0
        jac = metrics.get('jaccard', 0) or 0
        auc = metrics.get('mean_auc', 0) or 0
        comp = metrics.get('composite', 0) or 0
        print(f"{m.name:<25} {mac:>6.3f} {mic:>6.3f} {jac:>8.3f} {auc:>6.3f} {comp:>10.3f}  ({elapsed:.0f}s)")
        save_result(m.name, metrics)

    print("\nDone!")


if __name__ == '__main__':
    main()
