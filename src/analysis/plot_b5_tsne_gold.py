"""B5 단일 모델 t-SNE — column 임베딩을 gold label(y) 기준으로 색칠.

기존 `plot_tsne_per_layer` (table 기준) 과 대비용.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from torch_geometric.loader import DataLoader

from analysis.gat_bottleneck_analysis_v2 import (
    _build_dataset,
    _build_model_v2,
    extract_layerwise_embeddings_v2,
    _layer_names,
)


def plot_tsne_by_gold(sample_embeddings, y, layer_names, output_path: Path, query_idx: int):
    from sklearn.manifold import TSNE

    n_layers = len(sample_embeddings)
    labels = y.cpu().numpy().astype(int)
    n_gold = int(labels.sum())
    n_total = len(labels)

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4.3))
    if n_layers == 1:
        axes = [axes]

    for embs, name, ax in zip(sample_embeddings, layer_names, axes):
        embs_np = embs.cpu().numpy()
        if embs_np.shape[0] < 4:
            ax.set_title(f"{name} (n<4)")
            continue
        perplexity = min(30, max(2, embs_np.shape[0] // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
        coords = tsne.fit_transform(embs_np)
        # non-gold (회색) 먼저, gold (빨강) 나중에 겹쳐 그리기
        non_gold_mask = labels == 0
        gold_mask = labels == 1
        ax.scatter(coords[non_gold_mask, 0], coords[non_gold_mask, 1],
                   c="lightgray", s=20, alpha=0.6, edgecolors="none", label=f"non-gold ({n_total - n_gold})")
        ax.scatter(coords[gold_mask, 0], coords[gold_mask, 1],
                   c="crimson", s=40, alpha=0.9, edgecolors="black", linewidths=0.5,
                   label=f"gold ({n_gold})")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1].legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle(f"B5 Column Embedding t-SNE per Layer (query_idx={query_idx}, colored by gold label)")
    fig.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation/s06_a01_06_b5_dual_stream.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="/SSL_NAS/peoples/khj/thesis/checkpoints/s06_gat_bottleneck_fix/best_gat_s06_a01_06_b5.pt")
    parser.add_argument("--output", type=str,
                        default="outputs/analysis/s06_bottleneck/B5/tsne_query0_by_gold.png")
    parser.add_argument("--query_idx", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    dataset = _build_dataset(cfg)
    model = _build_model_v2(cfg, Path(args.checkpoint), device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = None
    for idx, b in enumerate(loader):
        if idx == args.query_idx:
            batch = b.to(device)
            break
    assert batch is not None, f"query_idx={args.query_idx} out of range"

    q_emb = batch["query"] if "query" in batch else None
    with torch.no_grad():
        layer_embs = extract_layerwise_embeddings_v2(model, batch, q_emb)

    col_embs = [d["column"] for d in layer_embs if "column" in d]
    y = batch["column"].y.float()
    layer_names = _layer_names(model.num_layers)[: len(col_embs)]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_tsne_by_gold(col_embs, y, layer_names, out_path, args.query_idx)
    print(f"saved → {out_path}")
    print(f"n_columns={len(y)}, n_gold={int(y.sum().item())}")


if __name__ == "__main__":
    main()
