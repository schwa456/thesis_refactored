#!/usr/bin/env python3
"""Per-stage failure analysis for 8 ablation models (corrected pipeline)."""

import json
import os
import numpy as np
from collections import defaultdict

BASE = "/home/hyeonjin/thesis_refactored/outputs/experiments"

# Model definitions: (dir_name, label, scoring, pcst_type, filter_type)
MODELS = [
    ("experiment_b0_raw_pcst_baseline",   "#1 C+B+N", "Cosine",   "Basic",    "None"),
    ("experiment_abl_cos_basic_xiyan",    "#2 C+B+X", "Cosine",   "Basic",    "XiYan"),
    ("experiment_b1_adaptive_pcst",       "#3 C+A+N", "Cosine",   "Adaptive", "None"),
    ("experiment_abl_cos_adaptive_xiyan", "#4 C+A+X", "Cosine",   "Adaptive", "XiYan"),
    ("experiment_b2_ensemble",            "#5 E+B+N", "Ensemble", "Basic",    "None"),
    ("experiment_abl_ens_basic_xiyan",    "#6 E+B+X", "Ensemble", "Basic",    "XiYan"),
    ("experiment_b_combined",             "#7 E+A+N", "Ensemble", "Adaptive", "None"),
    ("experiment_b4_xiyan_filter",        "#8 E+A+X", "Ensemble", "Adaptive", "XiYan"),
]

# No-filter pairing: filtered model -> no-filter counterpart
NO_FILTER_PAIR = {
    "experiment_abl_cos_basic_xiyan":    "experiment_b0_raw_pcst_baseline",
    "experiment_abl_cos_adaptive_xiyan": "experiment_b1_adaptive_pcst",
    "experiment_abl_ens_basic_xiyan":    "experiment_b2_ensemble",
    "experiment_b4_xiyan_filter":        "experiment_b_combined",
}

# PCST config: Basic uses node_threshold=0.1, Adaptive uses P80
BASIC_THRESHOLD = 0.1


def load_predictions(exp_dir):
    """Load predictions.jsonl -> {question_id: set(final_nodes)}"""
    preds = {}
    path = os.path.join(BASE, exp_dir, "predictions.jsonl")
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            preds[d["question_id"]] = set(d["final_nodes"])
    return preds


def load_score_analysis(exp_dir):
    """Load score_analysis -> {query_id: [(node_name, score, is_gold), ...]}"""
    # Find the score_analysis file
    files = os.listdir(os.path.join(BASE, exp_dir))
    sa_file = [f for f in files if f.startswith("score_analysis_")][0]
    path = os.path.join(BASE, exp_dir, sa_file)

    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            data[d["query_id"]].append((d["node_name"], d["score"], d["is_gold"]))
    return data


def analyze_model(exp_dir, label, scoring, pcst_type, filter_type):
    """Full analysis for one model."""
    preds = load_predictions(exp_dir)
    scores = load_score_analysis(exp_dir)

    # If this is a filtered model, load the no-filter counterpart
    if filter_type != "None":
        nf_dir = NO_FILTER_PAIR[exp_dir]
        nf_preds = load_predictions(nf_dir)
    else:
        nf_preds = preds  # no filter means PCST output = final output

    # Per-query analysis
    total_gold = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pcst_miss = 0
    total_filter_miss = 0

    tp_scores = []
    fn_pcst_scores = []
    fn_filter_scores = []

    # For Table 3: PCST recovery
    pcst_miss_scores = []  # scores of PCST-missed gold nodes
    per_query_thresholds = []  # (query_id, threshold) for adaptive

    # For Table 4: top-20 analysis
    gold_in_top20 = 0
    gold_total_for_top20 = 0

    for qid in sorted(preds.keys()):
        final_nodes = preds[qid]
        pcst_nodes = nf_preds.get(qid, set())

        if qid not in scores:
            continue

        query_scores = scores[qid]
        score_map = {name: sc for name, sc, _ in query_scores}
        gold_nodes = {name for name, _, is_gold in query_scores if is_gold}
        all_nodes_sorted = sorted(query_scores, key=lambda x: x[1], reverse=True)

        # Top-20 by score
        top20_names = {x[0] for x in all_nodes_sorted[:20]}
        gold_in_top20 += len(gold_nodes & top20_names)
        gold_total_for_top20 += len(gold_nodes)

        # Compute threshold for this query (for Table 3)
        if pcst_type == "Adaptive":
            all_scores_vals = [sc for _, sc, _ in query_scores]
            threshold = np.percentile(all_scores_vals, 80)
        else:
            threshold = BASIC_THRESHOLD
        per_query_thresholds.append((qid, threshold))

        # TP, FP, FN
        tp = gold_nodes & final_nodes
        fn = gold_nodes - final_nodes
        fp = final_nodes - gold_nodes

        total_gold += len(gold_nodes)
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        # Categorize FN
        for node in fn:
            sc = score_map.get(node, 0.0)
            if node not in pcst_nodes:
                total_pcst_miss += 1
                fn_pcst_scores.append(sc)
                pcst_miss_scores.append((sc, threshold))
            else:
                total_filter_miss += 1
                fn_filter_scores.append(sc)

        # TP scores
        for node in tp:
            sc = score_map.get(node, 0.0)
            tp_scores.append(sc)

    recall = total_tp / total_gold if total_gold > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "label": label,
        "scoring": scoring,
        "pcst_type": pcst_type,
        "filter_type": filter_type,
        "total_gold": total_gold,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "pcst_miss": total_pcst_miss,
        "filter_miss": total_filter_miss,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp_scores": tp_scores,
        "fn_pcst_scores": fn_pcst_scores,
        "fn_filter_scores": fn_filter_scores,
        "pcst_miss_data": pcst_miss_scores,  # (score, threshold) pairs
        "gold_in_top20": gold_in_top20,
        "gold_total_for_top20": gold_total_for_top20,
    }


def fmt(val, decimals=4):
    return f"{val:.{decimals}f}"


def main():
    results = []
    for exp_dir, label, scoring, pcst_type, filter_type in MODELS:
        print(f"Analyzing {label} ({exp_dir})...")
        r = analyze_model(exp_dir, label, scoring, pcst_type, filter_type)
        results.append(r)

    lines = []
    lines.append("# Per-Stage Failure Analysis (Corrected Pipeline)")
    lines.append("")
    lines.append("**Pipeline stages**: Score computation -> PCST extraction (uses ALL node scores) -> Filter (optional)")
    lines.append("")
    lines.append("**FN categories**:")
    lines.append("- **PCST X**: gold node not in PCST output (determined from no-filter counterpart)")
    lines.append("- **Filter X**: gold node in PCST output but removed by XiYan filter")
    lines.append("")

    # ===================== TABLE 1 =====================
    lines.append("## Table 1 -- Per-Stage Failure Breakdown")
    lines.append("")
    lines.append("| Model | Gold | TP | FP | FN | PCST X | Filter X | Recall | Precision | F1 |")
    lines.append("|-------|-----:|---:|---:|---:|-------:|---------:|-------:|----------:|---:|")
    for r in results:
        lines.append(
            f"| {r['label']} | {r['total_gold']} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['pcst_miss']} | {r['filter_miss']} | "
            f"{fmt(r['recall'])} | {fmt(r['precision'])} | {fmt(r['f1'])} |"
        )
    lines.append("")

    # ===================== TABLE 2 =====================
    lines.append("## Table 2 -- Score Distribution by Category")
    lines.append("")
    lines.append("| Model | TP mean | TP median | FN:PCST X mean | FN:PCST X median | FN:Filter X mean | FN:Filter X median |")
    lines.append("|-------|--------:|----------:|---------------:|-----------------:|-----------------:|-------------------:|")
    for r in results:
        tp_mean = np.mean(r["tp_scores"]) if r["tp_scores"] else 0
        tp_med = np.median(r["tp_scores"]) if r["tp_scores"] else 0
        pcst_mean = np.mean(r["fn_pcst_scores"]) if r["fn_pcst_scores"] else 0
        pcst_med = np.median(r["fn_pcst_scores"]) if r["fn_pcst_scores"] else 0
        filt_mean = np.mean(r["fn_filter_scores"]) if r["fn_filter_scores"] else 0
        filt_med = np.median(r["fn_filter_scores"]) if r["fn_filter_scores"] else 0
        lines.append(
            f"| {r['label']} | {fmt(tp_mean)} | {fmt(tp_med)} | "
            f"{fmt(pcst_mean)} | {fmt(pcst_med)} | "
            f"{fmt(filt_mean)} | {fmt(filt_med)} |"
        )
    lines.append("")

    # ===================== TABLE 3 =====================
    lines.append("## Table 3 -- PCST Recovery Analysis")
    lines.append("")
    lines.append("For PCST X (false negative) nodes: what fraction have scores above various fractions of the PCST threshold?")
    lines.append("")
    lines.append("- Basic PCST: fixed node_threshold = 0.1")
    lines.append("- Adaptive PCST: per-query P80 of all node scores")
    lines.append("")

    fractions = [0.25, 0.50, 0.75, 1.0, 1.25, 1.50]
    header = "| Model | N(PCST X) | Mean threshold | " + " | ".join(f">={int(f*100)}% thr" for f in fractions) + " |"
    sep = "|-------|----------:|---------------:|" + "|".join("-" * 12 + ":" for _ in fractions) + "|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        data = r["pcst_miss_data"]  # list of (score, threshold)
        n = len(data)
        if n == 0:
            mean_thr = 0
            frac_strs = ["N/A"] * len(fractions)
        else:
            mean_thr = np.mean([t for _, t in data])
            frac_strs = []
            for f in fractions:
                count = sum(1 for sc, thr in data if sc >= f * thr)
                frac_strs.append(f"{count}/{n} ({count/n*100:.1f}%)")

        lines.append(
            f"| {r['label']} | {n} | {fmt(mean_thr)} | " + " | ".join(frac_strs) + " |"
        )
    lines.append("")

    # ===================== TABLE 4 =====================
    lines.append("## Table 4 -- Ensemble vs Cosine Scoring Quality")
    lines.append("")
    lines.append("Fraction of gold nodes ranked in top-20 by score per query (even though top-k is not used as a gate).")
    lines.append("")
    lines.append("| Model | Scoring | Gold total | Gold in top-20 | Fraction |")
    lines.append("|-------|---------|----------:|--------------:|---------:|")
    for r in results:
        frac = r["gold_in_top20"] / r["gold_total_for_top20"] if r["gold_total_for_top20"] > 0 else 0
        lines.append(
            f"| {r['label']} | {r['scoring']} | {r['gold_total_for_top20']} | {r['gold_in_top20']} | {fmt(frac)} |"
        )
    lines.append("")

    # Cosine vs Ensemble summary
    lines.append("### Summary by Scoring Method")
    lines.append("")
    cos_total = sum(r["gold_total_for_top20"] for r in results if r["scoring"] == "Cosine")
    cos_top20 = sum(r["gold_in_top20"] for r in results if r["scoring"] == "Cosine")
    ens_total = sum(r["gold_total_for_top20"] for r in results if r["scoring"] == "Ensemble")
    ens_top20 = sum(r["gold_in_top20"] for r in results if r["scoring"] == "Ensemble")

    # Since each scoring method has 4 models with same scores (different PCST/filter don't affect scores),
    # use just one representative model per scoring method
    lines.append("Note: Since PCST and Filter do not affect scoring, we also show per-scoring-method stats from one representative model each:")
    lines.append("")

    # Representative: #1 for Cosine, #5 for Ensemble
    for r in results:
        if r["label"] in ("#1 C+B+N", "#5 E+B+N"):
            frac = r["gold_in_top20"] / r["gold_total_for_top20"] if r["gold_total_for_top20"] > 0 else 0
            lines.append(f"- **{r['scoring']}** ({r['label']}): {r['gold_in_top20']}/{r['gold_total_for_top20']} = {frac*100:.1f}% of gold nodes in top-20")
    lines.append("")

    # Additional: score stats by scoring method
    lines.append("### Gold Node Score Statistics by Scoring Method")
    lines.append("")
    lines.append("| Scoring | Model | Gold mean score | Gold median score | Non-gold mean | Non-gold median |")
    lines.append("|---------|-------|----------------:|------------------:|--------------:|----------------:|")

    for exp_dir, label, scoring, pcst_type, filter_type in MODELS:
        if label not in ("#1 C+B+N", "#5 E+B+N"):
            continue
        sa = load_score_analysis(exp_dir)
        gold_scores_all = []
        nongold_scores_all = []
        for qid, nodes in sa.items():
            for name, sc, is_gold in nodes:
                if is_gold:
                    gold_scores_all.append(sc)
                else:
                    nongold_scores_all.append(sc)
        lines.append(
            f"| {scoring} | {label} | {fmt(np.mean(gold_scores_all))} | {fmt(np.median(gold_scores_all))} | "
            f"{fmt(np.mean(nongold_scores_all))} | {fmt(np.median(nongold_scores_all))} |"
        )
    lines.append("")

    # Write output
    output_path = "/home/hyeonjin/thesis_refactored/notebooks/analysis_results/per_stage_failure_analysis.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
