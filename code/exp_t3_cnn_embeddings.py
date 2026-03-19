"""
Experiment T3: CNN backbone embeddings + SP features for Ridge regression.

Extracts 128-dim embeddings from pretrained CNN backbone, reduces with PCA,
combines with 6 SP features, and trains Ridge on log(freq) with LOPO CV.

PCA is fit ONLY on training data within each LOPO fold (no data leakage).

Run: conda run -n foe_dl python code/exp_t3_cnn_embeddings.py
"""

import sys
import time
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'dl'))

import torch
from model import EEGBackbone
from optimization_harness_v2 import (
    load_dataset, evaluate_experiment, FEATURE_COLS, CACHE_DIR
)
from sklearn.decomposition import PCA

DEVICE = 'cpu'


# ── Load pretrained backbone ──────────────────────────────────────────

def load_backbone():
    """Load backbone weights from best available pretrained model."""
    freq_path = CACHE_DIR / 'frequency_model_best.pt'
    class_path = CACHE_DIR / 'classifier_best.pt'

    backbone = EEGBackbone(in_channels=18, dropout=0.0)

    for path, label in [(freq_path, 'frequency model'), (class_path, 'classifier')]:
        if path.exists():
            print(f"Loading backbone from {label}: {path}")
            state = torch.load(str(path), map_location='cpu', weights_only=False)
            backbone_dict = {}
            for k, v in state.items():
                if k.startswith('backbone.'):
                    backbone_dict[k.replace('backbone.', '')] = v
            if backbone_dict:
                backbone.load_state_dict(backbone_dict, strict=True)
                print(f"  Loaded {len(backbone_dict)} backbone parameters")
                return backbone

    raise FileNotFoundError("No pretrained model found in dl_cache/")


# ── Precompute embeddings ─────────────────────────────────────────────

def precompute_embeddings(dataset, backbone):
    """Precompute 128-dim CNN embeddings for every segment.

    Returns a dict mapping id(segment_array) -> (128,) embedding.
    This allows the predict_fn to look up embeddings by segment identity.
    """
    print("\nPrecomputing CNN embeddings for all segments...")
    t0 = time.time()
    backbone.eval()

    df = dataset['df']
    segments = dataset['segments']

    # Collect all unique segment arrays
    all_segs = []
    seg_ids = []
    for _, row in df.iterrows():
        pid = row['patient_id']
        for seg in segments.get(pid, []):
            if seg is not None:
                all_segs.append(seg)
                seg_ids.append(id(seg))

    n_segs = len(all_segs)
    print(f"  Total segments to embed: {n_segs}")

    # Batch inference
    emb_map = {}
    batch_size = 64
    for i in range(0, n_segs, batch_size):
        batch = all_segs[i:i+batch_size]
        batch_tensor = torch.tensor(
            np.stack([s.astype(np.float32) for s in batch]),
            dtype=torch.float32
        )
        with torch.no_grad():
            features = backbone(batch_tensor)  # (B, 128, 125)
            embs = features.mean(dim=-1)       # (B, 128)
        embs_np = embs.numpy()
        for j, sid in enumerate(seg_ids[i:i+batch_size]):
            emb_map[sid] = embs_np[j]

    elapsed = time.time() - t0
    print(f"  Extracted {len(emb_map)} embeddings in {elapsed:.1f}s")
    return emb_map


# ── Predict function factory ──────────────────────────────────────────

def make_predict_fn(emb_map, n_pca_components, alpha, use_sp_features=True):
    """Create a predict_fn using precomputed CNN embeddings + PCA + Ridge.

    PCA is fit ONLY on training embeddings within each LOPO fold.
    Embeddings are looked up from emb_map using id(segment_array).
    """

    def _lookup_embeddings(segment_list):
        """Look up precomputed embeddings for a list of segment arrays."""
        embs = np.zeros((len(segment_list), 128))
        for i, seg in enumerate(segment_list):
            if seg is not None:
                sid = id(seg)
                if sid in emb_map:
                    embs[i] = emb_map[sid]
        return embs

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        n_train = len(train_segments)
        n_test = len(test_segments)

        # Look up precomputed embeddings
        train_embs = _lookup_embeddings(train_segments)
        test_embs = _lookup_embeddings(test_segments)

        # Fit PCA on training embeddings ONLY (no data leakage)
        n_comp = min(n_pca_components, n_train, 128)
        pca = PCA(n_components=n_comp, random_state=42)
        train_pca = pca.fit_transform(train_embs)
        test_pca = pca.transform(test_embs)

        # Build feature matrix
        if use_sp_features:
            X_train = np.column_stack([train_features, train_pca])
            X_test = np.column_stack([test_features, test_pca])
        else:
            X_train = train_pca
            X_test = test_pca

        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        # Impute NaN with training median
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            test_col = X_test[:, j]
            X_test[~np.isfinite(test_col), j] = med

        # Standardize (fit on train only)
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1.0
        X_train = (X_train - mu) / std
        X_test = (X_test - mu) / std

        # Add intercept
        X_train_b = np.column_stack([X_train, np.ones(n_train)])
        X_test_b = np.column_stack([X_test, np.ones(n_test)])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0  # Don't regularize intercept

        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            pred_log = X_test_b @ w
            pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
            return np.exp(pred_log)
        except np.linalg.LinAlgError:
            return np.full(n_test, np.nan)

    return _predict


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 72)
    print("Experiment T3: CNN Backbone Embeddings + SP Features")
    print("=" * 72)

    # Load dataset
    dataset = load_dataset(verbose=True)

    # Load backbone
    backbone = load_backbone()
    backbone.eval()

    # Precompute all embeddings once
    emb_map = precompute_embeddings(dataset, backbone)

    # Define experiments
    experiments = [
        ("t3_cnn_emb_pca10_ridge_a1", 10, 1.0, True),
        ("t3_cnn_emb_pca20_ridge_a1", 20, 1.0, True),
        ("t3_cnn_emb_pca20_ridge_a5", 20, 5.0, True),
        ("t3_cnn_emb_pca30_ridge_a5", 30, 5.0, True),
        ("t3_cnn_emb_only_pca20_a1", 20, 1.0, False),
    ]

    results = {}
    for exp_name, n_pca, alpha, use_sp in experiments:
        sp_str = "SP + " if use_sp else ""
        print(f"\n{'='*72}")
        print(f"Running: {exp_name} ({sp_str}PCA-{n_pca}, alpha={alpha})")
        print(f"{'='*72}")

        predict_fn = make_predict_fn(
            emb_map=emb_map,
            n_pca_components=n_pca,
            alpha=alpha,
            use_sp_features=use_sp,
        )

        metrics = evaluate_experiment(
            dataset,
            experiment_name=exp_name,
            predict_fn=predict_fn,
            eval_type='patient_lopo',
        )
        results[exp_name] = metrics

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("T3 EXPERIMENT SUMMARY")
    print("=" * 72)
    header = f"  {'Experiment':>35s} {'Spearman':>10s} {'95% CI':>22s} {'MAE':>8s}"
    print(header)
    print(f"  {'-'*75}")

    for exp_name, n_pca, alpha, use_sp in experiments:
        m = results[exp_name]
        rs = m.get('combined_spearman', float('nan'))
        ci = m.get('combined_spearman_ci', [float('nan'), float('nan')])
        mae = m.get('combined_mae', float('nan'))
        rs_str = f"{rs:.4f}" if np.isfinite(rs) else "N/A"
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if np.isfinite(ci[0]) else "N/A"
        mae_str = f"{mae:.4f}" if np.isfinite(mae) else "N/A"
        print(f"  {exp_name:>35s} {rs_str:>10s} {ci_str:>22s} {mae_str:>8s}")

    print(f"\n{'='*72}")
    print("Done! Run 'conda run -n foe python code/update_dashboard_v2.py' to update dashboard.")
