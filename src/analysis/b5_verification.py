"""B5 L_out 가설 검증 — 3가지 정량/시각 분석.

1. Pairwise cosine: gold-gold / gold-nongold / nongold-nongold (column 노드 한정)
2. Classifier logit scatter: ClassifierHead 통과 후 gold vs non-gold logit 분포 + AUC
3. Linear probe: L_out 임베딩 → Logistic Regression → AUC (5-fold CV)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch_geometric.loader import DataLoader

from models.direct_classifier import DirectClassifierHead
from analysis.gat_bottleneck_analysis_v2 import (
    _build_dataset,
    _build_model_v2,
    extract_layerwise_embeddings_v2,
)


def pairwise_group_cosines(emb: torch.Tensor, y: torch.Tensor) -> dict:
    """
    emb: [N, D], y: [N] (0 or 1)
    returns: {'gg': list, 'gn': list, 'nn': list}  (upper triangular pairs)
    """
    out = {"gg": [], "gn": [], "nn": []}
    if emb.size(0) < 2:
        return out
    normed = torch.nn.functional.normalize(emb, dim=-1)
    sim = normed @ normed.T  # [N, N]
    y_np = y.detach().cpu().numpy().astype(int)
    n = emb.size(0)
    iu = np.triu_indices(n, k=1)
    for i, j in zip(iu[0], iu[1]):
        s = sim[i, j].item()
        if y_np[i] == 1 and y_np[j] == 1:
            out["gg"].append(s)
        elif y_np[i] == 0 and y_np[j] == 0:
            out["nn"].append(s)
        else:
            out["gn"].append(s)
    return out


def plot_cosine_distributions(agg: dict, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(-0.5, 1.0, 60)
    for key, color, label in [("nn", "tab:gray", "non-gold ↔ non-gold"),
                               ("gn", "tab:orange", "gold ↔ non-gold"),
                               ("gg", "tab:red", "gold ↔ gold")]:
        vals = np.array(agg[key])
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, density=True, alpha=0.55, color=color,
                label=f"{label} (μ={vals.mean():.3f}, n={len(vals)})")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("L_out pairwise cosine (column nodes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[1]
    data = [agg["nn"], agg["gn"], agg["gg"]]
    ax.boxplot(data, labels=["non↔non", "gold↔non", "gold↔gold"], showfliers=False,
               patch_artist=True, boxprops=dict(facecolor="lightblue"))
    means = [np.mean(d) if d else np.nan for d in data]
    ax.plot([1, 2, 3], means, "rx", markersize=10, label="mean")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("L_out cosine by group")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_logit_distribution(logits: np.ndarray, labels: np.ndarray,
                             auc_classifier: float, output_path: Path):
    gold_logits = logits[labels == 1]
    nongold_logits = logits[labels == 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = np.linspace(np.min(logits), np.max(logits), 80)
    ax.hist(nongold_logits, bins=bins, density=True, alpha=0.55,
            color="tab:gray", label=f"non-gold (μ={nongold_logits.mean():.3f}, n={len(nongold_logits)})")
    ax.hist(gold_logits, bins=bins, density=True, alpha=0.6,
            color="tab:red", label=f"gold (μ={gold_logits.mean():.3f}, n={len(gold_logits)})")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.7, label="logit=0")
    ax.set_xlabel("Classifier logit (pre-sigmoid)")
    ax.set_ylabel("Density")
    ax.set_title(f"B5 DirectClassifierHead output — column\nAUC = {auc_classifier:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box
    ax = axes[1]
    ax.boxplot([nongold_logits, gold_logits], labels=["non-gold", "gold"],
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_ylabel("Classifier logit")
    ax.set_title("Logit by label")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def linear_probe_auc(embeddings: np.ndarray, labels: np.ndarray, n_folds: int = 5,
                      max_iter: int = 500, seed: int = 42) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X = embeddings
    y = labels
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        lr = LogisticRegression(max_iter=max_iter, solver="liblinear", C=1.0)
        lr.fit(X[tr], y[tr])
        pred = lr.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y[te], pred)
        fold_aucs.append(auc)
    return {"mean_auc": float(np.mean(fold_aucs)),
            "std_auc": float(np.std(fold_aucs)),
            "fold_aucs": [float(a) for a in fold_aucs]}


def plot_probe_roc(embeddings: np.ndarray, labels: np.ndarray, output_path: Path) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, roc_auc_score

    X_tr, X_te, y_tr, y_te = train_test_split(embeddings, labels, test_size=0.3,
                                               random_state=42, stratify=labels)
    lr = LogisticRegression(max_iter=500, solver="liblinear")
    lr.fit(X_tr, y_tr)
    pred = lr.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, pred)
    fpr, tpr, _ = roc_curve(y_te, pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="tab:red", linewidth=2, label=f"Linear Probe (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Linear Probe on B5 L_out (hold-out 30%, n={len(y_te)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    return {"holdout_auc": float(auc), "holdout_size": int(len(y_te))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation/s06_a01_06_b5_dual_stream.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="/SSL_NAS/peoples/khj/thesis/checkpoints/s06_gat_bottleneck_fix/best_gat_s06_a01_06_b5.pt")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/analysis/s06_bottleneck/B5")
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 모델 + classifier 로드
    gat = _build_model_v2(cfg, Path(args.checkpoint), device)
    classifier_heads = nn.ModuleDict({
        nt: DirectClassifierHead(
            in_dim=cfg["model"]["out_channels"],
            hidden_dim=cfg["model"].get("classifier_hidden", 256),
            dropout=cfg["model"].get("dropout", 0.1),
        ).to(device)
        for nt in ["table", "column", "fk_node"]
    })
    ckpt = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    classifier_heads.load_state_dict(ckpt["classifier_state_dict"])
    for h in classifier_heads.values():
        h.eval()

    dataset = _build_dataset(cfg)
    n = len(dataset) if args.max_queries is None else min(args.max_queries, len(dataset))
    print(f"Analyzing {n}/{len(dataset)} queries on CPU...")

    # 누적 버퍼
    cos_agg = {"gg": [], "gn": [], "nn": []}
    logits_all = []
    labels_all = []
    emb_all = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= n:
                break
            batch = batch.to(device)
            q_emb = batch["query"] if "query" in batch else None
            layer_embs = extract_layerwise_embeddings_v2(gat, batch, q_emb)
            if "column" not in layer_embs[-1]:
                continue
            l_out_col = layer_embs[-1]["column"]  # [N_col, D]
            y = batch["column"].y.float()
            if y.size(0) != l_out_col.size(0):
                continue

            # 1. pairwise cosine
            pc = pairwise_group_cosines(l_out_col, y)
            for k in ("gg", "gn", "nn"):
                cos_agg[k].extend(pc[k])

            # 2. classifier logit
            logits = classifier_heads["column"](l_out_col).cpu().numpy()
            logits_all.append(logits)
            labels_all.append(y.cpu().numpy().astype(int))

            # 3. 임베딩 buffer (linear probe용) — 메모리 절약 위해 float32
            emb_all.append(l_out_col.cpu().numpy().astype(np.float32))

            if (idx + 1) % 300 == 0:
                print(f"  {idx+1}/{n}")

    # 취합
    logits_all = np.concatenate(logits_all)
    labels_all = np.concatenate(labels_all)
    emb_all = np.concatenate(emb_all, axis=0)
    print(f"total column nodes: {len(labels_all)}, gold: {int(labels_all.sum())} "
          f"({100*labels_all.mean():.2f}%)")

    # (1) cosine plot + summary
    print("\n[1] Pairwise cosine — computing plots and summary ...")
    cosine_summary = {}
    for k in ("gg", "gn", "nn"):
        a = np.array(cos_agg[k])
        if len(a):
            cosine_summary[k] = {"mean": float(a.mean()), "std": float(a.std()),
                                  "median": float(np.median(a)), "n": int(len(a))}
        else:
            cosine_summary[k] = {"mean": None, "std": None, "median": None, "n": 0}
    plot_cosine_distributions(cos_agg, out_dir / "verification_cosine.png")
    print(f"  gg: μ={cosine_summary['gg']['mean']:.4f}, n={cosine_summary['gg']['n']}")
    print(f"  gn: μ={cosine_summary['gn']['mean']:.4f}, n={cosine_summary['gn']['n']}")
    print(f"  nn: μ={cosine_summary['nn']['mean']:.4f}, n={cosine_summary['nn']['n']}")

    # (2) classifier logit scatter + AUC
    print("\n[2] Classifier logit distribution ...")
    from sklearn.metrics import roc_auc_score
    auc_classifier = float(roc_auc_score(labels_all, logits_all))
    plot_logit_distribution(logits_all, labels_all, auc_classifier,
                             out_dir / "verification_logit.png")
    gold_logits = logits_all[labels_all == 1]
    nongold_logits = logits_all[labels_all == 0]
    print(f"  AUC(classifier) = {auc_classifier:.4f}")
    print(f"  gold logit: μ={gold_logits.mean():.4f}, σ={gold_logits.std():.4f}")
    print(f"  nongold   : μ={nongold_logits.mean():.4f}, σ={nongold_logits.std():.4f}")

    # (3) linear probe
    print("\n[3] Linear probe (5-fold CV) ...")
    probe_cv = linear_probe_auc(emb_all, labels_all, n_folds=5)
    probe_roc = plot_probe_roc(emb_all, labels_all, out_dir / "verification_probe_roc.png")
    print(f"  CV-AUC  = {probe_cv['mean_auc']:.4f} ± {probe_cv['std_auc']:.4f}")
    print(f"  holdout = {probe_roc['holdout_auc']:.4f}")

    # JSON summary
    summary = {
        "n_columns": int(len(labels_all)),
        "n_gold": int(labels_all.sum()),
        "cosine": cosine_summary,
        "classifier": {
            "auc": auc_classifier,
            "gold_logit_mean": float(gold_logits.mean()),
            "gold_logit_std": float(gold_logits.std()),
            "nongold_logit_mean": float(nongold_logits.mean()),
            "nongold_logit_std": float(nongold_logits.std()),
        },
        "linear_probe": {
            "cv_mean_auc": probe_cv["mean_auc"],
            "cv_std_auc": probe_cv["std_auc"],
            "cv_fold_aucs": probe_cv["fold_aucs"],
            "holdout_auc": probe_roc["holdout_auc"],
            "holdout_size": probe_roc["holdout_size"],
        },
    }
    summary_path = out_dir / "verification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
