"""
Threshold × Steiner Backbone sweep from pre-computed score_analysis files.
No GAT inference needed — uses saved scores + reconstructed graph edges.
"""
import json
import numpy as np
import networkx as nx
from collections import defaultdict
from modules.extractors.mst import steiner_tree_2approx


def load_query_data(score_path):
    """Load per-query node scores, gold labels, and reconstruct graph edges."""
    queries = defaultdict(lambda: {'nodes': [], 'scores': [], 'gold': [], 'names': []})

    with open(score_path) as f:
        for line in f:
            d = json.loads(line)
            qid = d['query_id']
            queries[qid]['nodes'].append(len(queries[qid]['nodes']))
            queries[qid]['scores'].append(d['score'])
            queries[qid]['gold'].append(d['is_gold'])
            queries[qid]['names'].append(d['node_name'])

    result = {}
    for qid, qd in queries.items():
        names = qd['names']
        n = len(names)

        # Build name -> index mapping
        name_to_idx = {name: i for i, name in enumerate(names)}

        # Reconstruct edges from node names
        # Tables, columns (table.col), FK nodes (table.col->table.col)
        edges = []
        for i, name in enumerate(names):
            if '->' in name:
                # FK node: src_table.src_col -> dst_table.dst_col
                parts = name.split('->')
                src_col_name = parts[0].strip()
                dst_col_name = parts[1].strip()
                src_table = src_col_name.split('.')[0]
                dst_table = dst_col_name.split('.')[0]

                # FK node connects to its source and target columns
                if src_col_name in name_to_idx:
                    edges.append((i, name_to_idx[src_col_name]))
                if dst_col_name in name_to_idx:
                    edges.append((i, name_to_idx[dst_col_name]))
            elif '.' in name:
                # Column: belongs_to table
                table_name = name.split('.')[0]
                if table_name in name_to_idx:
                    edges.append((i, name_to_idx[table_name]))

        # Gold set (by name, for metric computation)
        gold_names = set()
        for i, name in enumerate(names):
            if qd['gold'][i]:
                nl = name.lower()
                if '->' not in nl:
                    gold_names.add(nl)

        result[qid] = {
            'scores': qd['scores'],
            'names': names,
            'gold_names': gold_names,
            'edges': edges,
            'n': n,
        }

    return result


def compute_metrics_at_threshold(query_data, thresh, use_steiner=False):
    """Compute macro-avg R/P/F1 at a given threshold."""
    recalls, precisions = [], []

    for qid, qd in query_data.items():
        scores = qd['scores']
        names = qd['names']
        edges = qd['edges']
        gold = qd['gold_names']
        n = qd['n']

        # Binary selection
        seeds = [i for i in range(n) if scores[i] >= thresh]
        if not seeds:
            seeds = [int(np.argmax(scores))]

        selected = set(seeds)

        if use_steiner and len(seeds) >= 2:
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)
            try:
                st_nodes, _ = steiner_tree_2approx(G, seeds)
                selected = set(st_nodes)
            except Exception:
                pass

        # Convert to names for metric computation
        pred_names = set()
        for idx in selected:
            name = names[idx].lower()
            if '->' in name:
                continue
            pred_names.add(name)

        tp = len(pred_names & gold)
        r = tp / len(gold) if gold else 1.0
        p = tp / len(pred_names) if pred_names else 1.0
        recalls.append(r)
        precisions.append(p)

    avg_r = np.mean(recalls)
    avg_p = np.mean(precisions)
    avg_f1 = 2 * avg_r * avg_p / (avg_r + avg_p) if (avg_r + avg_p) > 0 else 0
    return avg_r, avg_p, avg_f1


def run_sweep(score_path, variant_name, thresholds):
    print(f"\nLoading scores from {score_path}...")
    query_data = load_query_data(score_path)
    print(f"Loaded {len(query_data)} queries.")

    print(f"\n{'='*95}")
    print(f"  {variant_name}: Threshold × Steiner Backbone Sweep")
    print(f"{'='*95}")
    print(f"\n{'Thresh':>7} | {'Sel R':>8} {'Sel P':>8} {'Sel F1':>8} | "
          f"{'+ Steiner R':>11} {'+ Steiner P':>11} {'+ Steiner F1':>12} | "
          f"{'R delta':>8}")
    print('-' * 95)

    for thresh in thresholds:
        sr, sp, sf = compute_metrics_at_threshold(query_data, thresh, use_steiner=False)
        tr, tp_, tf = compute_metrics_at_threshold(query_data, thresh, use_steiner=True)
        delta = tr - sr
        marker = " ***" if tr >= 0.75 else ""
        print(f"  {thresh:>5.2f} | {sr:>8.4f} {sp:>8.4f} {sf:>8.4f} | "
              f"{tr:>11.4f} {tp_:>11.4f} {tf:>12.4f} | "
              f"{delta:>+8.4f}{marker}")


if __name__ == "__main__":
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    run_sweep(
        "outputs/experiments/ablation_qcond_direct_selector_only/"
        "score_analysis_ablation_qcond_direct_selector_only.jsonl",
        "QCond Concat Direct",
        thresholds,
    )

    run_sweep(
        "outputs/experiments/ablation_supernode_direct_selector_only/"
        "score_analysis_ablation_supernode_direct_selector_only.jsonl",
        "SuperNode Direct",
        thresholds,
    )
