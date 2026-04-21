"""Merge B5E bottleneck summary into existing B0-B5 summary → cross-model plots.

Reads:
  outputs/analysis/s06_bottleneck/batch_summary.json            (B0..B5)
  outputs/analysis/s06_bottleneck_b5_enriched/batch_summary.json (B5E)

Writes:
  outputs/analysis/s06_bottleneck_merged/batch_summary.json
  outputs/analysis/s06_bottleneck_merged/cross_model_oversmoothing.png
  outputs/analysis/s06_bottleneck_merged/cross_model_grad_ratio.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/analysis/s06_bottleneck/batch_summary.json"
NEW = ROOT / "outputs/analysis/s06_bottleneck_b5_enriched/batch_summary.json"
OUT = ROOT / "outputs/analysis/s06_bottleneck_merged"


def merge():
    base = json.loads(BASE.read_text())
    new = json.loads(NEW.read_text())
    merged = {"step1": {}, "step2": {}, "step3": {}}
    for step in merged:
        merged[step].update(base.get(step, {}))
        merged[step].update(new.get(step, {}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "batch_summary.json").write_text(json.dumps(merged, indent=2, default=str))
    return merged


def plot_oversmoothing(results):
    names = list(results["step2"].keys())
    x = np.arange(len(names))
    l0, lfinal, lout = [], [], []
    for n in names:
        stats = results["step2"][n]["stats"]
        ln = results["step2"][n]["layer_names"]
        gats = [k for k in ln if k.startswith("L") and "GAT" in k]
        l0.append(stats["L0_PLM"]["mean"])
        lfinal.append(stats[gats[-1]]["mean"])
        lout.append(stats["L_out"]["mean"])
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.27
    ax.bar(x - w, l0, w, label="L0_PLM", color="tab:gray")
    ax.bar(x, lfinal, w, label="deepest_GAT", color="tab:purple")
    ax.bar(x + w, lout, w, label="L_out", color="tab:green")
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.5, label="critical 0.85")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Intra-Table Cosine (mean)")
    ax.set_title("Over-smoothing at Key Depths (B0-B5 + B5E)")
    ax.grid(True, alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cross_model_oversmoothing.png", dpi=120)
    plt.close(fig)


def plot_grad_ratio(results):
    names = list(results["step3"].keys())
    ratios = [results["step3"][n].get("grad_ratio") or float("nan") for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["tab:orange"] * len(names)
    if "B5E" in names:
        colors[names.index("B5E")] = "tab:red"
    ax.bar(names, ratios, color=colors, alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="balanced")
    ax.axhline(0.1, color="red", linestyle="--", alpha=0.5, label="vanish threshold")
    ax.set_ylabel("grad(last conv) / grad(first conv)")
    ax.set_title("Gradient Flow Ratio (B0-B5 + B5E)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cross_model_grad_ratio.png", dpi=120)
    plt.close(fig)


def plot_recall_summary(results):
    """Best Val R@15 by model, highlight B5E vs B5."""
    names = list(results["step1"].keys())
    best = [results["step1"][n]["best_recall"] for n in names]
    epochs = [results["step1"][n]["best_recall_epoch"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["tab:blue"] * len(names)
    if "B5E" in names:
        colors[names.index("B5E")] = "tab:red"
    if "B5" in names:
        colors[names.index("B5")] = "tab:cyan"
    bars = ax.bar(names, best, color=colors, alpha=0.85)
    for bar, e in zip(bars, epochs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"@{e}", ha="center", fontsize=8)
    ax.set_ylim(min(best) - 0.02, max(best) + 0.02)
    ax.axhline(best[names.index("B0")], color="gray", linestyle="--", alpha=0.5, label="B0 baseline")
    ax.set_ylabel("Best Val Recall@15")
    ax.set_title("Best Val Recall@15 (B0-B5 + B5E)")
    ax.grid(True, alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cross_model_best_recall.png", dpi=120)
    plt.close(fig)


def main():
    merged = merge()
    plot_oversmoothing(merged)
    plot_grad_ratio(merged)
    plot_recall_summary(merged)
    # Print concise table
    print("\nModel  | Best R@15 | @Epoch | L0→L_out  | grad_ratio")
    print("-------|-----------|--------|-----------|-----------")
    for n in merged["step1"]:
        s1 = merged["step1"][n]
        stats = merged["step2"].get(n, {}).get("stats", {})
        l0 = stats.get("L0_PLM", {}).get("mean", float("nan"))
        lout = stats.get("L_out", {}).get("mean", float("nan"))
        gr = merged["step3"].get(n, {}).get("grad_ratio")
        gr_s = f"{gr:.3f}" if gr else "  nan"
        print(f"{n:6s} | {s1['best_recall']:.4f}    | {s1['best_recall_epoch']:>4d}   "
              f"| {l0:.3f}→{lout:.3f} | {gr_s}")
    print(f"\nMerged output: {OUT}")


if __name__ == "__main__":
    main()
