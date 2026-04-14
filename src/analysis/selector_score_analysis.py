"""
Selector Score Analysis: Score Discrimination, Distribution, and GAT Marginal Contribution.
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/home/hyeonjin/thesis_refactored")
OUT_DIR = BASE_DIR / "notebooks" / "analysis_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
def load_scores(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

cosine_file = BASE_DIR / "outputs/experiments/experiment_b0_raw_pcst_baseline/score_analysis_b0_raw_pcst_baseline.jsonl"
ensemble_file = BASE_DIR / "outputs/experiments/experiment_b2_ensemble/score_analysis_b2_ensemble.jsonl"

print("Loading data...")
cosine_data = load_scores(cosine_file)
ensemble_data = load_scores(ensemble_file)

# Build lookup for ensemble scores by (query_id, node_name)
ensemble_lookup = {}
for r in ensemble_data:
    ensemble_lookup[(r["query_id"], r["node_name"])] = r["score"]

# Group by query
def group_by_query(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["query_id"]].append(r)
    return groups

cosine_by_query = group_by_query(cosine_data)
ensemble_by_query = group_by_query(ensemble_data)

# --- Analysis 1: Score Discrimination ---
print("Analysis 1: Score Discrimination...")

def compute_auc_metrics(records):
    labels = np.array([int(r["is_gold"]) for r in records])
    scores = np.array([r["score"] for r in records])
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None, None
    roc = roc_auc_score(labels, scores)
    pr = average_precision_score(labels, scores)
    return roc, pr

def compute_per_query_auc(by_query):
    roc_list, pr_list = [], []
    skipped = 0
    for qid, recs in by_query.items():
        labels = [int(r["is_gold"]) for r in recs]
        if sum(labels) == 0 or sum(labels) == len(labels):
            skipped += 1
            continue
        scores = [r["score"] for r in recs]
        roc_list.append(roc_auc_score(labels, scores))
        pr_list.append(average_precision_score(labels, scores))
    return np.mean(roc_list), np.std(roc_list), np.mean(pr_list), np.std(pr_list), len(roc_list), skipped

cos_roc_global, cos_pr_global = compute_auc_metrics(cosine_data)
ens_roc_global, ens_pr_global = compute_auc_metrics(ensemble_data)

cos_roc_avg, cos_roc_std, cos_pr_avg, cos_pr_std, cos_n, cos_skip = compute_per_query_auc(cosine_by_query)
ens_roc_avg, ens_roc_std, ens_pr_avg, ens_pr_std, ens_n, ens_skip = compute_per_query_auc(ensemble_by_query)

# --- Analysis 2: Score Distribution & PCST Prize Impact ---
print("Analysis 2: Score Distribution & PCST Prize Impact...")

def percentile_summary(arr):
    return {
        "min": np.min(arr), "p10": np.percentile(arr, 10), "p25": np.percentile(arr, 25),
        "median": np.median(arr), "p75": np.percentile(arr, 75), "p80": np.percentile(arr, 80),
        "p90": np.percentile(arr, 90), "p95": np.percentile(arr, 95), "max": np.max(arr),
        "mean": np.mean(arr), "std": np.std(arr)
    }

def analyze_threshold(by_query, label=""):
    all_gold_scores = []
    all_nongold_scores = []
    gold_above_count = 0
    gold_total_count = 0
    queries_with_all_gold_above = 0
    per_query_rates = []

    for qid, recs in by_query.items():
        scores_all = [r["score"] for r in recs]
        p80 = np.percentile(scores_all, 80)

        gold_scores = [r["score"] for r in recs if r["is_gold"]]
        nongold_scores = [r["score"] for r in recs if not r["is_gold"]]

        all_gold_scores.extend(gold_scores)
        all_nongold_scores.extend(nongold_scores)

        if len(gold_scores) > 0:
            above = sum(1 for s in gold_scores if s >= p80)
            gold_above_count += above
            gold_total_count += len(gold_scores)
            rate = above / len(gold_scores)
            per_query_rates.append(rate)
            if above == len(gold_scores):
                queries_with_all_gold_above += 1

    gold_arr = np.array(all_gold_scores)
    nongold_arr = np.array(all_nongold_scores)

    return {
        "gold_percentiles": percentile_summary(gold_arr),
        "nongold_percentiles": percentile_summary(nongold_arr),
        "gold_above_p80_count": gold_above_count,
        "gold_total": gold_total_count,
        "gold_above_p80_rate": gold_above_count / gold_total_count if gold_total_count > 0 else 0,
        "per_query_rate_mean": np.mean(per_query_rates),
        "per_query_rate_std": np.std(per_query_rates),
        "per_query_rate_median": np.median(per_query_rates),
        "queries_all_gold_above": queries_with_all_gold_above,
        "total_queries": len(per_query_rates),
    }

cos_dist = analyze_threshold(cosine_by_query, "Cosine")
ens_dist = analyze_threshold(ensemble_by_query, "Ensemble")

# --- Analysis 3: GAT Marginal Contribution ---
print("Analysis 3: GAT Marginal Contribution...")

gat_rescued = []  # gold nodes: cosine < p80 but ensemble >= p80
gat_hurt = []     # gold nodes: cosine >= p80 but ensemble < p80
gat_neutral_both_above = 0
gat_neutral_both_below = 0
total_gold = 0
queries_with_rescue = 0
queries_with_hurt = 0
total_queries_analyzed = 0

per_query_rescue_counts = []

for qid in cosine_by_query:
    cos_recs = cosine_by_query[qid]
    ens_recs = ensemble_by_query.get(qid, [])

    if not ens_recs:
        continue

    total_queries_analyzed += 1

    # Compute P80 for each scoring method separately
    cos_scores_all = [r["score"] for r in cos_recs]
    ens_scores_all = [r["score"] for r in ens_recs]
    cos_p80 = np.percentile(cos_scores_all, 80)
    ens_p80 = np.percentile(ens_scores_all, 80)

    # Build ensemble lookup for this query
    ens_map = {r["node_name"]: r["score"] for r in ens_recs}

    query_rescued = 0
    query_hurt = 0

    for r in cos_recs:
        if not r["is_gold"]:
            continue
        total_gold += 1
        cos_s = r["score"]
        ens_s = ens_map.get(r["node_name"])
        if ens_s is None:
            continue

        cos_above = cos_s >= cos_p80
        ens_above = ens_s >= ens_p80

        if not cos_above and ens_above:
            gat_rescued.append({"node": r["node_name"], "qid": qid, "cos": cos_s, "ens": ens_s, "cos_p80": cos_p80, "ens_p80": ens_p80})
            query_rescued += 1
        elif cos_above and not ens_above:
            gat_hurt.append({"node": r["node_name"], "qid": qid, "cos": cos_s, "ens": ens_s, "cos_p80": cos_p80, "ens_p80": ens_p80})
            query_hurt += 1
        elif cos_above and ens_above:
            gat_neutral_both_above += 1
        else:
            gat_neutral_both_below += 1

    per_query_rescue_counts.append(query_rescued)
    if query_rescued > 0:
        queries_with_rescue += 1
    if query_hurt > 0:
        queries_with_hurt += 1

# Stats for rescued nodes
rescued_cos_scores = [r["cos"] for r in gat_rescued]
rescued_ens_scores = [r["ens"] for r in gat_rescued]
hurt_cos_scores = [r["cos"] for r in gat_hurt]
hurt_ens_scores = [r["ens"] for r in gat_hurt]

# Compute implied GAT scores: ensemble = 0.85*cos + 0.15*gat => gat = (ensemble - 0.85*cos) / 0.15
rescued_gat_implied = [(r["ens"] - 0.85 * r["cos"]) / 0.15 for r in gat_rescued]
hurt_gat_implied = [(r["ens"] - 0.85 * r["cos"]) / 0.15 for r in gat_hurt]

# --- Generate Report ---
print("Generating report...")

def fmt(v, decimals=4):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"

def fmt_pct(v, decimals=1):
    return f"{v*100:.{decimals}f}%"

def percentile_table(p, label):
    lines = []
    lines.append(f"| Stat | {label} |")
    lines.append(f"|------|--------|")
    for k in ["min", "p10", "p25", "median", "mean", "p75", "p80", "p90", "p95", "max", "std"]:
        lines.append(f"| {k} | {fmt(p[k])} |")
    return "\n".join(lines)

report = []
report.append("# Selector Score Analysis")
report.append("")
report.append("## Data Sources")
report.append(f"- **Cosine model**: `experiment_b0_raw_pcst_baseline` ({len(cosine_data):,} node-scores, {len(cosine_by_query)} queries)")
report.append(f"- **Ensemble model**: `experiment_b2_ensemble` ({len(ensemble_data):,} node-scores, {len(ensemble_by_query)} queries)")
report.append(f"- Ensemble formula: `0.85 * cosine + 0.15 * GAT`")
report.append("")

# Analysis 1
report.append("---")
report.append("## Analysis 1: Score Discrimination (ROC-AUC / PR-AUC)")
report.append("")
report.append("### Global Metrics (all nodes pooled)")
report.append("")
report.append("| Metric | Cosine | Ensemble | Delta |")
report.append("|--------|--------|----------|-------|")
report.append(f"| ROC-AUC | {fmt(cos_roc_global)} | {fmt(ens_roc_global)} | {fmt(ens_roc_global - cos_roc_global, 4)} |")
report.append(f"| PR-AUC  | {fmt(cos_pr_global)} | {fmt(ens_pr_global)} | {fmt(ens_pr_global - cos_pr_global, 4)} |")
report.append("")
report.append("### Per-Query Metrics (macro-averaged)")
report.append("")
report.append("| Metric | Cosine (mean +/- std) | Ensemble (mean +/- std) | Delta (mean) |")
report.append("|--------|-----------------------|-------------------------|--------------|")
report.append(f"| ROC-AUC | {fmt(cos_roc_avg)} +/- {fmt(cos_roc_std)} | {fmt(ens_roc_avg)} +/- {fmt(ens_roc_std)} | {fmt(ens_roc_avg - cos_roc_avg, 4)} |")
report.append(f"| PR-AUC  | {fmt(cos_pr_avg)} +/- {fmt(cos_pr_std)} | {fmt(ens_pr_avg)} +/- {fmt(ens_pr_std)} | {fmt(ens_pr_avg - cos_pr_avg, 4)} |")
report.append(f"")
report.append(f"- Queries evaluated: {cos_n} (skipped {cos_skip} with all-gold or no-gold)")
report.append("")

# Analysis 2
report.append("---")
report.append("## Analysis 2: Score Distribution & PCST Prize Impact (P80 Threshold)")
report.append("")
report.append("### Score Distribution: Gold Nodes")
report.append("")
report.append("| Stat | Cosine | Ensemble |")
report.append("|------|--------|----------|")
for k in ["min", "p10", "p25", "median", "mean", "p75", "p80", "p90", "p95", "max", "std"]:
    report.append(f"| {k} | {fmt(cos_dist['gold_percentiles'][k])} | {fmt(ens_dist['gold_percentiles'][k])} |")
report.append("")

report.append("### Score Distribution: Non-Gold Nodes")
report.append("")
report.append("| Stat | Cosine | Ensemble |")
report.append("|------|--------|----------|")
for k in ["min", "p10", "p25", "median", "mean", "p75", "p80", "p90", "p95", "max", "std"]:
    report.append(f"| {k} | {fmt(cos_dist['nongold_percentiles'][k])} | {fmt(ens_dist['nongold_percentiles'][k])} |")
report.append("")

report.append("### Gold Nodes Above P80 Threshold (Positive PCST Prize)")
report.append("")
report.append("| Metric | Cosine | Ensemble | Delta |")
report.append("|--------|--------|----------|-------|")
report.append(f"| Gold nodes above P80 (global) | {cos_dist['gold_above_p80_count']}/{cos_dist['gold_total']} ({fmt_pct(cos_dist['gold_above_p80_rate'])}) | {ens_dist['gold_above_p80_count']}/{ens_dist['gold_total']} ({fmt_pct(ens_dist['gold_above_p80_rate'])}) | {fmt_pct(ens_dist['gold_above_p80_rate'] - cos_dist['gold_above_p80_rate'])} |")
report.append(f"| Per-query rate (mean +/- std) | {fmt_pct(cos_dist['per_query_rate_mean'])} +/- {fmt_pct(cos_dist['per_query_rate_std'])} | {fmt_pct(ens_dist['per_query_rate_mean'])} +/- {fmt_pct(ens_dist['per_query_rate_std'])} | {fmt_pct(ens_dist['per_query_rate_mean'] - cos_dist['per_query_rate_mean'])} |")
report.append(f"| Per-query rate (median) | {fmt_pct(cos_dist['per_query_rate_median'])} | {fmt_pct(ens_dist['per_query_rate_median'])} | {fmt_pct(ens_dist['per_query_rate_median'] - cos_dist['per_query_rate_median'])} |")
report.append(f"| Queries where ALL gold above P80 | {cos_dist['queries_all_gold_above']}/{cos_dist['total_queries']} ({fmt_pct(cos_dist['queries_all_gold_above']/cos_dist['total_queries'])}) | {ens_dist['queries_all_gold_above']}/{ens_dist['total_queries']} ({fmt_pct(ens_dist['queries_all_gold_above']/ens_dist['total_queries'])}) | {fmt_pct((ens_dist['queries_all_gold_above'] - cos_dist['queries_all_gold_above'])/cos_dist['total_queries'])} |")
report.append("")
report.append(f"> **Interpretation**: P80 threshold means top 20% of nodes per query get positive PCST prize.")
report.append(f"> A higher 'gold above P80' rate means the scoring method better separates gold nodes from non-gold.")
report.append("")

# Analysis 3
report.append("---")
report.append("## Analysis 3: GAT Marginal Contribution (Ensemble vs Cosine at P80 Threshold)")
report.append("")
report.append(f"Using per-method P80 thresholds (cosine P80 for cosine, ensemble P80 for ensemble).")
report.append("")
report.append("### Threshold Crossing Summary")
report.append("")
report.append("| Category | Count | % of Gold |")
report.append("|----------|-------|-----------|")
report.append(f"| GAT rescued (cos < P80, ens >= P80) | {len(gat_rescued)} | {fmt_pct(len(gat_rescued)/total_gold if total_gold else 0)} |")
report.append(f"| GAT hurt (cos >= P80, ens < P80) | {len(gat_hurt)} | {fmt_pct(len(gat_hurt)/total_gold if total_gold else 0)} |")
report.append(f"| Neutral: both above P80 | {gat_neutral_both_above} | {fmt_pct(gat_neutral_both_above/total_gold if total_gold else 0)} |")
report.append(f"| Neutral: both below P80 | {gat_neutral_both_below} | {fmt_pct(gat_neutral_both_below/total_gold if total_gold else 0)} |")
report.append(f"| **Total gold nodes** | **{total_gold}** | **100%** |")
report.append("")
report.append(f"- Net rescued: {len(gat_rescued) - len(gat_hurt)} gold nodes ({fmt_pct((len(gat_rescued) - len(gat_hurt))/total_gold if total_gold else 0)})")
report.append("")

report.append("### Score Characteristics of Rescued vs Hurt Nodes")
report.append("")
report.append("| Metric | GAT Rescued | GAT Hurt |")
report.append("|--------|-------------|----------|")
if rescued_cos_scores:
    report.append(f"| Mean cosine score | {fmt(np.mean(rescued_cos_scores))} | {fmt(np.mean(hurt_cos_scores)) if hurt_cos_scores else 'N/A'} |")
    report.append(f"| Mean ensemble score | {fmt(np.mean(rescued_ens_scores))} | {fmt(np.mean(hurt_ens_scores)) if hurt_ens_scores else 'N/A'} |")
    report.append(f"| Mean implied GAT score | {fmt(np.mean(rescued_gat_implied))} | {fmt(np.mean(hurt_gat_implied)) if hurt_gat_implied else 'N/A'} |")
    report.append(f"| Median implied GAT score | {fmt(np.median(rescued_gat_implied))} | {fmt(np.median(hurt_gat_implied)) if hurt_gat_implied else 'N/A'} |")
else:
    report.append("| (no rescued nodes) | - | - |")
report.append("")

report.append("### Per-Query Breakdown")
report.append("")
report.append(f"| Metric | Value |")
report.append(f"|--------|-------|")
report.append(f"| Total queries analyzed | {total_queries_analyzed} |")
report.append(f"| Queries where GAT rescued >= 1 gold node | {queries_with_rescue} ({fmt_pct(queries_with_rescue/total_queries_analyzed if total_queries_analyzed else 0)}) |")
report.append(f"| Queries where GAT hurt >= 1 gold node | {queries_with_hurt} ({fmt_pct(queries_with_hurt/total_queries_analyzed if total_queries_analyzed else 0)}) |")
report.append(f"| Mean rescued per query (among all queries) | {fmt(np.mean(per_query_rescue_counts), 2)} |")
report.append(f"| Mean rescued per query (among queries with rescue) | {fmt(np.mean([c for c in per_query_rescue_counts if c > 0]), 2) if queries_with_rescue > 0 else 'N/A'} |")
report.append("")

# Separation analysis
report.append("### Score Separation: Gold vs Non-Gold")
report.append("")
cos_gold_mean = cos_dist['gold_percentiles']['mean']
cos_nongold_mean = cos_dist['nongold_percentiles']['mean']
ens_gold_mean = ens_dist['gold_percentiles']['mean']
ens_nongold_mean = ens_dist['nongold_percentiles']['mean']
cos_gap = cos_gold_mean - cos_nongold_mean
ens_gap = ens_gold_mean - ens_nongold_mean
report.append(f"| Metric | Cosine | Ensemble |")
report.append(f"|--------|--------|----------|")
report.append(f"| Mean gold score | {fmt(cos_gold_mean)} | {fmt(ens_gold_mean)} |")
report.append(f"| Mean non-gold score | {fmt(cos_nongold_mean)} | {fmt(ens_nongold_mean)} |")
report.append(f"| Gap (gold - non-gold) | {fmt(cos_gap)} | {fmt(ens_gap)} |")
report.append("")

out_path = OUT_DIR / "selector_analysis.md"
with open(out_path, "w") as f:
    f.write("\n".join(report))
print(f"Report saved to {out_path}")
