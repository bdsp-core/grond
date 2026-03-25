"""
Visualize CNN backbone embeddings: t-SNE, PCA variance, embedding-frequency correlation.
"""
import sys
sys.path.insert(0, 'code/')
sys.path.insert(0, 'code/dl/')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine as cosine_dist
from model import EEGClassifier
from data_loader import normalize_segment
import os

os.makedirs('results', exist_ok=True)

# ── Load model ──────────────────────────────────────────────────────────────
device = torch.device('cpu')
model = EEGClassifier(in_channels=18, dropout=0.0)
state_dict = torch.load('data/dl_cache/classifier_best.pt', map_location='cpu')
model.load_state_dict(state_dict, strict=True)
model.eval()
backbone = model.backbone

# ── Load data ───────────────────────────────────────────────────────────────
data = np.load('data/dl_cache/annotated_pd_data.npz', allow_pickle=True)
segments = data['segments']       # (556, 18, 2000)
expert_freqs = data['expert_freqs']  # (556, 3)
subtypes = data['subtypes']       # (556,) 'lpd' or 'gpd'

# Consensus frequency: mean of non-zero experts
consensus = np.full(len(segments), np.nan)
for i in range(len(segments)):
    vals = expert_freqs[i]
    valid = vals[(vals > 0) & np.isfinite(vals)]
    if len(valid) > 0:
        consensus[i] = np.mean(valid)

# ── Normalize and extract embeddings ────────────────────────────────────────
print("Extracting embeddings...")
normed = np.stack([normalize_segment(seg) for seg in segments])  # (556, 18, 2000)
batch = torch.from_numpy(normed)  # (556, 18, 2000)

with torch.no_grad():
    features = backbone(batch)  # (556, 128, 125)
    embeddings = features.mean(dim=2).numpy()  # (556, 128) — global average pool

print(f"Embeddings shape: {embeddings.shape}")

# Masks
is_lpd = subtypes == 'lpd'
is_gpd = subtypes == 'gpd'
has_freq = np.isfinite(consensus)

# ── t-SNE ───────────────────────────────────────────────────────────────────
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
coords = tsne.fit_transform(embeddings)

# ── Figure A: t-SNE colored by frequency ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, mask, label in [(axes[0], is_lpd, 'LPD'), (axes[1], is_gpd, 'GPD')]:
    m = mask & has_freq
    sc = ax.scatter(coords[m, 0], coords[m, 1], c=consensus[m],
                    cmap='viridis', s=20, alpha=0.7, edgecolors='none')
    # Plot segments without freq in gray
    m_no = mask & ~has_freq
    if m_no.sum() > 0:
        ax.scatter(coords[m_no, 0], coords[m_no, 1], c='lightgray', s=10, alpha=0.4)
    ax.set_title(f'{label} (n={mask.sum()})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(sc, ax=ax, label='Expert consensus freq (Hz)')
fig.suptitle('CNN Backbone Embeddings: t-SNE colored by expert frequency', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig('results/figure_tsne_frequency.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved figure_tsne_frequency.png")

# ── Figure B: t-SNE colored by type ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(coords[is_lpd, 0], coords[is_lpd, 1], c='green', s=20, alpha=0.6, label='LPD')
ax.scatter(coords[is_gpd, 0], coords[is_gpd, 1], c='blue', s=20, alpha=0.6, label='GPD')
ax.set_title('CNN Backbone Embeddings: t-SNE colored by LPD/GPD type', fontsize=13)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig('results/figure_tsne_type.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved figure_tsne_type.png")

# ── Figure C: PCA explained variance ───────────────────────────────────────
pca = PCA(n_components=128)
pca.fit(embeddings)
cumvar = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 129), cumvar, 'b-', linewidth=2)
ax.axvline(x=20, color='red', linestyle='--', linewidth=1.5, label='20-component cutoff')
# Mark specific points
for n in [5, 10, 20, 50]:
    val = cumvar[n-1]
    ax.plot(n, val, 'ro', markersize=8)
    ax.annotate(f'{n}PC: {val:.1%}', (n, val), textcoords='offset points',
                xytext=(10, -5 if n != 20 else 10), fontsize=10)
ax.set_xlabel('Number of PCA components')
ax.set_ylabel('Cumulative explained variance')
ax.set_title('PCA Explained Variance of 128-dim CNN Embeddings')
ax.set_xlim(0, 130)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('results/figure_pca_variance.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved figure_pca_variance.png")

# ── Figure D: Embedding similarity vs frequency similarity ──────────────────
from sklearn.metrics.pairwise import cosine_similarity

# Only use segments with valid consensus frequency
valid_idx = np.where(has_freq)[0]
valid_emb = embeddings[valid_idx]
valid_freq = consensus[valid_idx]
valid_types = subtypes[valid_idx]

# Cosine similarity matrix (only upper triangle)
cos_sim = cosine_similarity(valid_emb)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, type_label in [(axes[0], 'lpd'), (axes[1], 'gpd')]:
    type_mask = valid_types == type_label
    type_idx = np.where(type_mask)[0]
    n_type = len(type_idx)

    # Generate all pairs (upper triangle)
    ii, jj = np.triu_indices(n_type, k=1)
    n_pairs = len(ii)

    # Subsample to ~5000 pairs
    if n_pairs > 5000:
        rng = np.random.RandomState(42)
        sel = rng.choice(n_pairs, 5000, replace=False)
        ii, jj = ii[sel], jj[sel]

    sims = cos_sim[np.ix_(type_idx, type_idx)][ii, jj]
    freq_diffs = np.abs(valid_freq[type_idx[ii]] - valid_freq[type_idx[jj]])

    ax.scatter(sims, freq_diffs, s=3, alpha=0.3, c='steelblue')

    # Trend line
    m, b = np.polyfit(sims, freq_diffs, 1)
    x_range = np.linspace(sims.min(), sims.max(), 100)
    ax.plot(x_range, m * x_range + b, 'r-', linewidth=2)

    r, p = pearsonr(sims, freq_diffs)
    ax.set_xlabel('Embedding cosine similarity')
    ax.set_ylabel('|Frequency difference| (Hz)')
    ax.set_title(f'{type_label.upper()} (n_pairs={len(ii)}, Pearson r={r:.3f}, p={p:.2e})')
    ax.grid(True, alpha=0.3)

fig.suptitle('Embedding Similarity vs Frequency Similarity', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig('results/figure_embedding_freq_correlation.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved figure_embedding_freq_correlation.png")

print("\nDone! All figures saved to results/")
