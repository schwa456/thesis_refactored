"""
Per-Stage Failure Analysis for 8 Ablation Models in Schema Linking Pipeline.

For each experiment, we reconstruct per-node stage decisions:
  - Seed selected: node is in top-20 by score
  - PCST extracted: node appears in final_nodes of the paired "no-filter" experiment
  - Filter kept: node appears in final_nodes of this experiment
  - Gold: node is a gold table or column

We pair filtered experiments with their no-filter counterparts:
  #2 C+B+X pairs with #1 C+B+N  (same seed+PCST, different filter)
  #4 C+A+X pairs with #3 C+A+N
  #6 E+B+X pairs with #5 E+B+N
  #8 E+A+X pairs with #7 E+A+N
"""

import json
import os
import glob
from collections import defaultdict

BASE = "/home/hyeonjin/thesis_refactored/outputs/experiments"

EXPERIMENTS = [
    ("experiment_b0_raw_pcst_baseline",    "#1 C+B+N", "b0_raw_pcst_baseline"),
    ("experiment_abl_cos_basic_xiyan",     "#2 C+B+X", "abl_cos_basic_xiyan"),
    ("experiment_b1_adaptive_pcst",        "#3 C+A+N", "b1_adaptive_pcst"),
    ("experiment_abl_cos_adaptive_xiyan",  "#4 C+A+X", "abl_cos_adaptive_xiyan"),
    ("experiment_b2_ensemble",             "#5 E+B+N", "b2_ensemble"),
    ("experiment_abl_ens_basic_xiyan",     "#6 E+B+X", "abl_ens_basic_xiyan"),
    ("experiment_b_combined",              "#7 E+A+N", "b_combined"),
    ("experiment_b4_xiyan_filter",         "#8 E+A+X", "b4_xiyan_filter"),
]

# Pairs: filtered -> no-filter counterpart (same selector + PCST)
FILTER_PAIRS = {
    "experiment_abl_cos_basic_xiyan":    "experiment_b0_raw_pcst_baseline",
    "experiment_abl_cos_adaptive_xiyan": "experiment_b1_adaptive_pcst",
    "experiment_abl_ens_basic_xiyan":    "experiment_b2_ensemble",
    "experiment_b4_xiyan_filter":        "experiment_b_combined",
}

TOP_K = 20  # seed selection top-k


def load_score_analysis(exp_dir, exp_suffix):
    """Load score_analysis file: returns {query_id: [(node_name, score, is_gold), ...]}"""
    path = os.path.join(exp_dir, f"score_analysis_{exp_suffix}.jsonl")
    data = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            data[rec['query_id']].append((rec['node_name'], rec['score'], rec['is_gold']))
    return data


def load_predictions(exp_dir):
    """Load predictions.jsonl: returns {question_id: set(final_nodes_lower)}"""
    path = os.path.join(exp_dir, "predictions.jsonl")
    data = {}
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            qid = rec.get('question_id')
            nodes = rec.get('final_nodes', [])
            data[qid] = set(n.lower() for n in nodes)
    return data


def load_output(exp_dir, exp_suffix):
    """Load output file: returns {question_id: {gold_tables, gold_cols}}"""
    path = os.path.join(exp_dir, f"output_{exp_suffix}.jsonl")
    data = {}
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            qid = rec.get('question_id')
            if qid is None:
                continue
            data[qid] = {
                'gold_tables': set(t.lower() for t in rec.get('gold_tables', [])),
                'gold_cols': set(c.lower() for c in rec.get('gold_cols', [])),
            }
    return data


def get_seed_selected(nodes_with_scores, top_k=20):
    """Given list of (node_name, score, is_gold), return set of top-k node names (lowercase)."""
    sorted_nodes = sorted(nodes_with_scores, key=lambda x: x[1], reverse=True)
    return set(n[0].lower() for n in sorted_nodes[:top_k])


def is_gold_node(node_name, gold_tables, gold_cols):
    """Check if a node is gold (table or column)."""
    name_lower = node_name.lower()
    if '.' in name_lower:
        tbl, col = name_lower.split('.', 1)
        # Check both "table.col" format and just "col" format
        if name_lower in gold_cols or col in gold_cols:
            return True
        if tbl in gold_tables and col in gold_cols:
            return True
    else:
        if name_lower in gold_tables:
            return True
    return False


def analyze_experiment(exp_dir, exp_suffix, label, pcst_final_nodes=None):
    """
    Analyze one experiment.

    pcst_final_nodes: if this is a filtered experiment, pass the final_nodes from
                      the paired no-filter experiment to distinguish PCST vs Filter failures.
    """
    scores_data = load_score_analysis(exp_dir, exp_suffix)
    predictions = load_predictions(exp_dir)
    output_data = load_output(exp_dir, exp_suffix)

    has_filter = pcst_final_nodes is not None

    # Aggregate counters
    gold_total = 0
    tp = 0
    fp = 0
    fn_seed = 0  # gold & not seed_selected
    fn_pcst = 0  # gold & seed_selected & not pcst_extracted
    fn_filter = 0  # gold & pcst_extracted & not final_selected
    fn_pcst_filter = 0  # For no-filter: gold & seed_selected & not final (= PCST failure)

    # Score tracking for Table 2
    tp_scores = []
    fn_seed_scores = []
    fn_pcst_scores = []
    fn_filter_scores = []
    fn_pcst_filter_scores = []

    # Seed recovery analysis for Table 3
    seed_min_scores = []  # minimum score among seed-selected nodes per query
    fn_seed_node_scores = []  # scores of FN:Seed nodes

    for qid in sorted(scores_data.keys()):
        nodes = scores_data[qid]
        final_nodes = predictions.get(qid, set())
        gold_info = output_data.get(qid, {'gold_tables': set(), 'gold_cols': set()})
        gold_tables = gold_info['gold_tables']
        gold_cols = gold_info['gold_cols']

        # Get seed-selected nodes (top-k by score)
        seed_selected = get_seed_selected(nodes, TOP_K)

        # Get PCST-extracted nodes (from paired no-filter experiment)
        if has_filter:
            pcst_extracted = pcst_final_nodes.get(qid, set())
        else:
            pcst_extracted = final_nodes  # no filter => final = PCST output

        # Compute minimum seed score for this query
        sorted_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
        seed_nodes_scores = [n[1] for n in sorted_nodes[:min(TOP_K, len(sorted_nodes))]]
        if seed_nodes_scores:
            seed_min_scores.append(min(seed_nodes_scores))

        for node_name, score, is_gold_flag in nodes:
            name_lower = node_name.lower()

            # Skip FK nodes
            if '->' in name_lower:
                continue

            # Determine gold status using output file's gold info
            gold = is_gold_node(node_name, gold_tables, gold_cols)

            in_seed = name_lower in seed_selected
            in_pcst = name_lower in pcst_extracted
            in_final = name_lower in final_nodes

            if gold:
                gold_total += 1
                if in_final:
                    tp += 1
                    tp_scores.append(score)
                elif not in_seed:
                    fn_seed += 1
                    fn_seed_scores.append(score)
                    fn_seed_node_scores.append(score)
                elif has_filter:
                    if not in_pcst:
                        fn_pcst += 1
                        fn_pcst_scores.append(score)
                    else:
                        fn_filter += 1
                        fn_filter_scores.append(score)
                else:
                    # No filter: seed selected but not in final => PCST failure
                    fn_pcst_filter += 1
                    fn_pcst_filter_scores.append(score)
            else:
                if in_final:
                    fp += 1

    # Compute metrics
    recall = tp / gold_total if gold_total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # For no-filter experiments, PCST failure = fn_pcst_filter
    if not has_filter:
        fn_pcst = fn_pcst_filter
        fn_pcst_scores = fn_pcst_filter_scores
        fn_filter = 0

    fn_total = fn_seed + fn_pcst + fn_filter

    result = {
        'label': label,
        'gold_total': gold_total,
        'tp': tp,
        'fp': fp,
        'fn_total': fn_total,
        'fn_seed': fn_seed,
        'fn_pcst': fn_pcst,
        'fn_filter': fn_filter,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'tp_score_mean': sum(tp_scores) / len(tp_scores) if tp_scores else 0,
        'fn_seed_score_mean': sum(fn_seed_scores) / len(fn_seed_scores) if fn_seed_scores else 0,
        'fn_pcst_score_mean': sum(fn_pcst_scores) / len(fn_pcst_scores) if fn_pcst_scores else 0,
        'fn_filter_score_mean': sum(fn_filter_scores) / len(fn_filter_scores) if fn_filter_scores else 0,
        'seed_min_scores': seed_min_scores,
        'fn_seed_node_scores': fn_seed_node_scores,
    }

    return result


def compute_seed_recovery(result):
    """Table 3: Seed Recovery Analysis."""
    seed_min_scores = result['seed_min_scores']
    fn_scores = result['fn_seed_node_scores']

    if not seed_min_scores or not fn_scores:
        return {
            'total_fn_seed': len(fn_scores),
            'threshold_median': 0,
            'pct_90': 0, 'pct_75': 0, 'pct_50': 0, 'pct_unrecoverable': 100,
            'n_90': 0, 'n_75': 0, 'n_50': 0, 'n_unrec': len(fn_scores),
        }

    # Use median of per-query min-seed-scores as the effective threshold
    sorted_mins = sorted(seed_min_scores)
    threshold = sorted_mins[len(sorted_mins) // 2]

    n_total = len(fn_scores)
    n_90 = sum(1 for s in fn_scores if s >= threshold * 0.90)
    n_75 = sum(1 for s in fn_scores if s >= threshold * 0.75)
    n_50 = sum(1 for s in fn_scores if s >= threshold * 0.50)
    n_unrec = n_total - n_50

    return {
        'total_fn_seed': n_total,
        'threshold_median': threshold,
        'pct_90': 100 * n_90 / n_total if n_total > 0 else 0,
        'pct_75': 100 * n_75 / n_total if n_total > 0 else 0,
        'pct_50': 100 * n_50 / n_total if n_total > 0 else 0,
        'pct_unrecoverable': 100 * n_unrec / n_total if n_total > 0 else 0,
        'n_90': n_90, 'n_75': n_75, 'n_50': n_50, 'n_unrec': n_unrec,
    }


def main():
    # First pass: load no-filter experiments to get PCST outputs
    no_filter_predictions = {}
    for exp_dir_name, label, exp_suffix in EXPERIMENTS:
        if exp_dir_name not in FILTER_PAIRS.values():
            continue
        exp_dir = os.path.join(BASE, exp_dir_name)
        no_filter_predictions[exp_dir_name] = load_predictions(exp_dir)

    # Analyze all experiments
    results = []
    for exp_dir_name, label, exp_suffix in EXPERIMENTS:
        exp_dir = os.path.join(BASE, exp_dir_name)

        # Determine if this is a filtered experiment
        paired = FILTER_PAIRS.get(exp_dir_name)
        pcst_final_nodes = no_filter_predictions.get(paired) if paired else None

        print(f"Analyzing {label} ({exp_dir_name})...")
        result = analyze_experiment(exp_dir, exp_suffix, label, pcst_final_nodes)
        results.append(result)

    # Generate markdown output
    lines = []
    lines.append("# Per-Stage Failure Analysis")
    lines.append("")

    # Table 1: Per-Stage Failure Breakdown
    lines.append("## Table 1: Per-Stage Failure Breakdown")
    lines.append("")
    lines.append("| Model | Gold | TP | FP | FN | Seed\\u2717 | PCST\\u2717 | Filter\\u2717 | Recall | Precision | F1 |")
    lines.append("|-------|-----:|---:|---:|---:|-------:|-------:|--------:|------:|---------:|---:|")

    for r in results:
        lines.append(
            f"| {r['label']} "
            f"| {r['gold_total']} "
            f"| {r['tp']} "
            f"| {r['fp']} "
            f"| {r['fn_total']} "
            f"| {r['fn_seed']} "
            f"| {r['fn_pcst']} "
            f"| {r['fn_filter']} "
            f"| {r['recall']:.4f} "
            f"| {r['precision']:.4f} "
            f"| {r['f1']:.4f} |"
        )

    lines.append("")
    lines.append("**Legend**: Seed\\u2717 = gold node not in top-20 seeds; PCST\\u2717 = seed-selected but lost in PCST extraction; Filter\\u2717 = PCST-extracted but removed by XiYan filter.")
    lines.append("")
    lines.append("For no-filter models (N), Filter\\u2717 is always 0; PCST\\u2717 includes all post-seed losses.")
    lines.append("")

    # Table 2: Score Distribution by Category
    lines.append("## Table 2: Mean Score by Failure Category")
    lines.append("")
    lines.append("| Model | TP (mean) | FN:Seed\\u2717 (mean) | FN:PCST\\u2717 (mean) | FN:Filter\\u2717 (mean) |")
    lines.append("|-------|----------:|-------------------:|-------------------:|--------------------:|")

    for r in results:
        lines.append(
            f"| {r['label']} "
            f"| {r['tp_score_mean']:.4f} "
            f"| {r['fn_seed_score_mean']:.4f} "
            f"| {r['fn_pcst_score_mean']:.4f} "
            f"| {r['fn_filter_score_mean']:.4f} |"
        )

    lines.append("")
    lines.append("**Interpretation**: Higher mean scores for FN categories suggest the scoring function ranked them well but a downstream stage still dropped them.")
    lines.append("")

    # Table 3: Seed Recovery Analysis
    lines.append("## Table 3: Seed Recovery Analysis (FN:Seed\\u2717 Nodes)")
    lines.append("")
    lines.append("For Seed\\u2717 FN nodes, what fraction could be recovered by lowering the seed threshold.")
    lines.append("The threshold is approximated as the median of per-query minimum seed-selected scores.")
    lines.append("")
    lines.append("| Model | FN:Seed\\u2717 | Threshold | >=90% thr | >=75% thr | >=50% thr | Unrecoverable |")
    lines.append("|-------|----------:|----------:|----------:|----------:|----------:|--------------:|")

    for r in results:
        rec = compute_seed_recovery(r)
        lines.append(
            f"| {r['label']} "
            f"| {rec['total_fn_seed']} "
            f"| {rec['threshold_median']:.4f} "
            f"| {rec['n_90']} ({rec['pct_90']:.1f}%) "
            f"| {rec['n_75']} ({rec['pct_75']:.1f}%) "
            f"| {rec['n_50']} ({rec['pct_50']:.1f}%) "
            f"| {rec['n_unrec']} ({rec['pct_unrecoverable']:.1f}%) |"
        )

    lines.append("")
    lines.append("**Interpretation**: Nodes at >=90% of threshold are near-misses recoverable with a slight threshold relaxation. Unrecoverable nodes (below 50% of threshold) have fundamentally low similarity and cannot be rescued by threshold tuning alone.")

    # Write output
    output_path = "/home/hyeonjin/thesis_refactored/notebooks/analysis_results/per_stage_failure_analysis.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\nResults saved to: {output_path}")

    # Also print to stdout
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
