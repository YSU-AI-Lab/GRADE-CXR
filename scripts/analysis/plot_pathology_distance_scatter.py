#!/usr/bin/env python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]
EVAL_IDX = [i for i in range(14) if i != 13]
SHORT = {
    "Enlarged Cardiomediastinum": "Enl. Card.",
    "Cardiomegaly": "Card.",
    "Lung Opacity": "Opacity",
    "Lung Lesion": "Lesion",
    "Edema": "Edema",
    "Consolidation": "Consol.",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumo.",
    "Pleural Effusion": "Effusion",
    "Pleural Other": "Pl. Other",
    "Fracture": "Fracture",
    "Support Devices": "Support",
}

def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def load_npz(cache_dir, prefix):
    files = sorted(Path(cache_dir).glob(f"{prefix}_n*_topk4_a0.5_seed42.npz"))
    if not files:
        raise FileNotFoundError(f"No cache for {prefix} in {cache_dir}")
    # Prefer the most recent/small balanced table if multiple exist. Current formal run is n6000.
    files = sorted(files, key=lambda p: ("n6000" not in p.name, -p.stat().st_mtime))
    p = files[0]
    d = np.load(p, allow_pickle=True)
    print(f"[INFO] {prefix}: {p.name}, X={d['features'].shape}")
    return l2(d['features'].astype(np.float32)), (d['labels'] > 0.5).astype(int), d['source'].astype(int)

def cross_source_dist_by_class(X, Y, S, max_pairs_per_pair=20000, seed=42):
    rng = np.random.default_rng(seed)
    rows = {}
    sources = sorted(np.unique(S).tolist())
    for c in EVAL_IDX:
        pair_means = []
        for i, s1 in enumerate(sources):
            idx1 = np.where((S == s1) & (Y[:, c] == 1))[0]
            for s2 in sources[i+1:]:
                idx2 = np.where((S == s2) & (Y[:, c] == 1))[0]
                if len(idx1) == 0 or len(idx2) == 0:
                    continue
                n = min(max_pairs_per_pair, len(idx1) * len(idx2))
                a = rng.choice(idx1, size=n, replace=True)
                b = rng.choice(idx2, size=n, replace=True)
                dist = 1.0 - np.sum(X[a] * X[b], axis=1)
                pair_means.append(float(np.mean(dist)))
        rows[LABELS[c]] = float(np.mean(pair_means)) if pair_means else np.nan
    return rows

def draw_panel(ax, x, y, labels, title, ylabel, highlight_color):
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]; y = y[finite]
    labs = [lab for lab, keep in zip(labels, finite) if keep]
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    pad = max(0.01, (hi - lo) * 0.08)
    lo -= pad; hi += pad
    ax.scatter(x, y, s=44, color=highlight_color, alpha=0.75, edgecolor='white', linewidth=0.7, zorder=3)
    ax.plot([lo, hi], [lo, hi], linestyle='--', color='#9a9a9a', linewidth=1.0, zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='#d9d9d9', linewidth=0.55, alpha=0.65)
    ax.set_title(title, fontsize=10.5, fontweight='semibold', pad=7)
    ax.set_xlabel('CLIP cross-source distance', fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')
    ax.tick_params(labelsize=8.5, colors='#333333')
    delta = y - x
    # annotate 3 best improvements and 2 worst/non-improvements
    order = np.argsort(delta)
    chosen = list(order[:3]) + [i for i in order[-2:] if i not in order[:3]]
    for j in chosen:
        dx = 5 if delta[j] <= 0 else -34
        dy = 5 if j % 2 == 0 else -10
        ax.annotate(SHORT.get(labs[j], labs[j]), (x[j], y[j]), xytext=(dx, dy), textcoords='offset points',
                    fontsize=7.8, color='#222222', arrowprops=dict(arrowstyle='-', color='#b0b0b0', lw=0.5, alpha=0.7))

def main():
    out = Path('/home/hyr/clip/biomedclip_encoder_generalization/motivation_analysis/representation_analysis_table')
    cache = out / 'feature_cache'
    Xc, Yc, Sc = load_npz(cache, 'CLIP')
    Xb, Yb, Sb = load_npz(cache, 'GRADE-CXR_base')
    Xr, Yr, Sr = load_npz(cache, 'GRADE-CXR_ref')
    # sanity: labels/source should match because caches were extracted from same rows.
    if not (np.array_equal(Yc, Yb) and np.array_equal(Yc, Yr) and np.array_equal(Sc, Sb) and np.array_equal(Sc, Sr)):
        print('[WARN] cache label/source arrays differ; distances are still computed per model, but sample order may differ.')
    clip = cross_source_dist_by_class(Xc, Yc, Sc)
    base = cross_source_dist_by_class(Xb, Yb, Sb)
    ref = cross_source_dist_by_class(Xr, Yr, Sr)
    rows = []
    for c in EVAL_IDX:
        p = LABELS[c]
        rows.append({
            'pathology': p,
            'clip_cross_src_dist': clip[p],
            'grade_base_cross_src_dist': base[p],
            'grade_ref_cross_src_dist': ref[p],
            'delta_base': base[p] - clip[p],
            'delta_ref': ref[p] - clip[p],
        })
    df = pd.DataFrame(rows)
    csv_path = out / 'pathology_cross_source_distance_by_class.csv'
    df.to_csv(csv_path, index=False)
    labels = df['pathology'].tolist()
    x = df['clip_cross_src_dist'].to_numpy(float)
    yb = df['grade_base_cross_src_dist'].to_numpy(float)
    yr = df['grade_ref_cross_src_dist'].to_numpy(float)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    draw_panel(axes[0], x, yb, labels, '(a) CLIP vs GRADE-CXR_base', 'GRADE-CXR_base cross-source distance', '#4C78A8')
    draw_panel(axes[1], x, yr, labels, '(b) CLIP vs GRADE-CXR_ref', 'GRADE-CXR_ref cross-source distance', '#B24A4A')
    fig.suptitle('Pathology-level cross-source distance', fontsize=11.5, fontweight='semibold')
    png = out / 'pathology_distance_scatter.png'
    pdf = out / 'pathology_distance_scatter.pdf'
    fig.savefig(png, dpi=600, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    n_base = int(np.sum(df['delta_base'] < 0))
    n_ref = int(np.sum(df['delta_ref'] < 0))
    summary = [
        '# Pathology Distance Scatter Summary',
        '',
        f'- Output figure: `{png}` / `{pdf}`',
        f'- Per-class CSV: `{csv_path}`',
        f'- GRADE-CXR_base below y=x: {n_base}/{len(df)} pathologies; mean delta = {df["delta_base"].mean():.4f}',
        f'- GRADE-CXR_ref below y=x: {n_ref}/{len(df)} pathologies; mean delta = {df["delta_ref"].mean():.4f}',
        '',
        'Negative delta means lower cross-source same-pathology distance than CLIP.',
    ]
    (out / 'pathology_distance_scatter_summary.md').write_text('\n'.join(summary) + '\n')
    print('\n'.join(summary))

if __name__ == '__main__':
    main()
