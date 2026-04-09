"""Generate B-4b dashboard based on notebooks/log_analyzer.ipynb::plot_dashboard."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve


def plot_dashboard(score_json_path: str, profiling_json_path: str, exp_name: str,
                   save_path: str, max_k: int = 50):
    print(f"Loading Score data from {score_json_path}...")
    with open(score_json_path, "r", encoding="utf-8") as f:
        score_data = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(score_data)

    print(f"Loading Profiling data from {profiling_json_path}...")
    with open(profiling_json_path, "r", encoding="utf-8") as f:
        prof_data = [json.loads(line) for line in f if line.strip()]
    prof_df = pd.DataFrame(prof_data)

    time_cols = [c for c in prof_df.columns if c != "query_id"]
    avg_durations = prof_df[time_cols].mean().to_dict()
    total_avg_time = sum(avg_durations.values())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    fig.suptitle(f"Schema Linking Dashboard: {exp_name}", fontsize=24, fontweight="bold", y=1.00)

    # Panel 1 - Execution time
    if avg_durations:
        sorted_modules = sorted(avg_durations.keys())
        times = [avg_durations[m] for m in sorted_modules]
        axes[0, 0].barh(sorted_modules[::-1], times[::-1], color="coral", edgecolor="black", alpha=0.8)
        axes[0, 0].set_title("1. Execution Time per Module", fontsize=15, fontweight="bold")
        axes[0, 0].set_xlabel("Average Time (seconds)", fontsize=12)
        for i, v in enumerate(times[::-1]):
            axes[0, 0].text(v + (max(times) * 0.02 if max(times) > 0 else 0.05), i,
                            f"{v:.3f}s", va="center", fontweight="bold")

    # Panel 2 - Global score histogram
    all_scores = df["score"].values
    axes[0, 1].hist(all_scores, bins=50, color="royalblue", edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(np.mean(all_scores), color="red", linestyle="dashed", linewidth=2,
                       label=f"Mean: {np.mean(all_scores):.3f}")
    axes[0, 1].axvline(np.median(all_scores), color="green", linestyle="dashed", linewidth=2,
                       label=f"Median: {np.median(all_scores):.3f}")
    axes[0, 1].set_title("2. Global Node Score Distribution", fontsize=15, fontweight="bold")
    axes[0, 1].set_xlabel("Predicted Score", fontsize=12)
    axes[0, 1].set_ylabel("Frequency", fontsize=12)
    axes[0, 1].legend()

    # Panel 3 - KDE by relevance
    sns.kdeplot(data=df[df["is_gold"] == True], x="score", fill=True, color="#1f77b4",
                alpha=0.5, label="Gold (Positive)", ax=axes[1, 0])
    sns.kdeplot(data=df[df["is_gold"] == False], x="score", fill=True, color="#d62728",
                alpha=0.3, label="Noise (Negative)", ax=axes[1, 0])
    axes[1, 0].set_title("3. Score Density by Relevance (Separability)", fontsize=15, fontweight="bold")
    axes[1, 0].set_xlabel("Predicted Score", fontsize=12)
    axes[1, 0].set_ylabel("Density", fontsize=12)
    axes[1, 0].legend()

    # Panel 4 - Violin
    sns.violinplot(data=df, x="is_gold", y="score",
                   palette={True: "#1f77b4", False: "#d62728",
                            "True": "#1f77b4", "False": "#d62728"},
                   inner="quartile", ax=axes[1, 1])
    axes[1, 1].set_title("4. Score Distribution (Violin Plot)", fontsize=15, fontweight="bold")
    axes[1, 1].set_xlabel("Is Gold Schema?", fontsize=12)
    axes[1, 1].set_ylabel("Score", fontsize=12)
    axes[1, 1].set_xticklabels(["False (Noise)", "True (Gold)"])

    # Panel 5 - Recall@K
    recall_at_k_list = {k: [] for k in range(1, max_k + 1)}
    for _, group in df.groupby("query_id"):
        sorted_group = group.sort_values(by="score", ascending=False).reset_index(drop=True)
        total_golds = sorted_group["is_gold"].sum()
        if total_golds == 0:
            continue
        for k in range(1, max_k + 1):
            hits = sorted_group.head(k)["is_gold"].sum()
            recall_at_k_list[k].append(hits / total_golds)
    k_values = list(range(1, max_k + 1))
    mean_recalls = [np.mean(recall_at_k_list[k]) * 100 for k in k_values]

    axes[2, 0].plot(k_values, mean_recalls, marker="o", markersize=4, color="#2ca02c", linewidth=2)
    axes[2, 0].axhline(y=100, color="gray", linestyle="--", alpha=0.7)
    axes[2, 0].set_title(f"5. Average Recall@K (Up to Top-{max_k})", fontsize=15, fontweight="bold")
    axes[2, 0].set_xlabel("K (Number of Retrieved Nodes)", fontsize=12)
    axes[2, 0].set_ylabel("Recall (%)", fontsize=12)
    axes[2, 0].set_ylim(0, 105)
    for k_idx in [5, 10, 20, 50]:
        if k_idx <= max_k:
            axes[2, 0].annotate(f"{mean_recalls[k_idx - 1]:.1f}%",
                                (k_idx, mean_recalls[k_idx - 1]),
                                textcoords="offset points", xytext=(0, 10),
                                ha="center", fontsize=10)

    # Panel 6 - Threshold tuning
    y_true = df["is_gold"].astype(int).values
    y_scores = df["score"].values
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls,
                          out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    # thresholds length is len(P)-1; align
    thr_aligned = np.append(thresholds, thresholds[-1])
    best_idx = int(np.argmax(f1_scores[:-1]))
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_p = precisions[best_idx]
    best_r = recalls[best_idx]

    axes[2, 1].plot(thr_aligned, precisions, label="Precision", color="#1f77b4", linewidth=2)
    axes[2, 1].plot(thr_aligned, recalls, label="Recall", color="#ff7f0e", linewidth=2)
    axes[2, 1].plot(thr_aligned, f1_scores, label="F1-Score", color="#2ca02c", linewidth=3, linestyle="--")
    axes[2, 1].axvline(x=best_threshold, color="red", linestyle=":", linewidth=2)
    axes[2, 1].scatter([best_threshold], [best_f1], color="red", s=100, zorder=5)
    axes[2, 1].annotate(f"Best Thresh: {best_threshold:.3f}\n(F1: {best_f1:.3f})",
                        (best_threshold, best_f1), textcoords="offset points",
                        xytext=(10, -15), ha="left", fontsize=11, color="red", fontweight="bold")
    axes[2, 1].set_title("6. Threshold Tuning", fontsize=15, fontweight="bold")
    axes[2, 1].set_xlabel("Threshold", fontsize=12)
    axes[2, 1].set_ylabel("Score", fontsize=12)
    axes[2, 1].legend()

    # Panel 7 - PR curve
    axes[3, 0].plot(recalls, precisions, color="purple", linewidth=2)
    axes[3, 0].scatter([best_r], [best_p], color="red", s=100, zorder=5,
                       label=f"Best Thresh ({best_threshold:.3f})")
    axes[3, 0].set_title("7. Precision-Recall (PR) Curve", fontsize=15, fontweight="bold")
    axes[3, 0].set_xlabel("Recall", fontsize=12)
    axes[3, 0].set_ylabel("Precision", fontsize=12)
    axes[3, 0].legend()

    # Panel 8 - summary text
    axes[3, 1].axis("off")
    recall_at_20 = f"{mean_recalls[19]:.2f}%" if len(mean_recalls) >= 20 else "N/A"
    summary_text = (
        f"[ EXPERIMENT SUMMARY ]\n\n"
        f"- Total Queries Analyzed: {df['query_id'].nunique():,}\n"
        f"- Total Nodes Evaluated: {len(df):,}\n"
        f"- Avg. Execution Time / Query: {total_avg_time:.3f} sec\n\n"
        f"[ SELECTION METRICS ]\n\n"
        f"- Recall @ 20: {recall_at_20}\n"
        f"- Best Threshold: {best_threshold:.4f}\n"
        f"- Peak F1-Score: {best_f1:.4f}\n"
        f"  (Precision: {best_p:.4f} / Recall: {best_r:.4f})"
    )
    props = dict(boxstyle="round,pad=1.5", facecolor="#f8f9fa", alpha=0.9, edgecolor="gray")
    axes[3, 1].text(0.5, 0.5, summary_text, transform=axes[3, 1].transAxes,
                    fontsize=15, va="center", ha="center", bbox=props,
                    linespacing=1.8, family="monospace")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Dashboard saved to: {save_path}")


if __name__ == "__main__":
    root = Path("/home/hyeonjin/thesis_refactored")
    exp_dir = root / "outputs/experiments/experiment_b4_xiyan_filter"
    plot_dashboard(
        score_json_path=str(exp_dir / "score_analysis_b4_xiyan_filter.jsonl"),
        profiling_json_path=str(exp_dir / "profiling_b4_xiyan_filter.jsonl"),
        exp_name="B-4b (XiYanFilter)",
        save_path=str(root / "notebooks/analysis_results/b4b_dashboard.png"),
        max_k=50,
    )
