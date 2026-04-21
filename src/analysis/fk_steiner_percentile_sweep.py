#!/usr/bin/env python
"""
Offline sweep: FKBackboneSteinerExtractor column_recovery_percentile.

4 scopes × 21 percentiles (0..100 step 5) + abs θ_r=0.8 anchor.
Selector outputs are reused (no re-encoding, no GAT forward): reads
per-query scores from a previous a10_09 inference's score_analysis jsonl,
rebuilds metadata from the graph cache, and re-runs only extract().
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


def build_metadata(hetero_data, score_rows: List[Dict[str, Any]]):
    """Reconstruct the metadata dict FKBackboneSteinerExtractor.extract() expects.

    The order in score_rows matches flat_id order (table, column, fk).
    """
    table_rows, col_rows, fk_rows = [], [], []
    for r in score_rows:
        name = r['node_name']
        if '->' in name:
            fk_rows.append(r)
        elif '.' in name:
            col_rows.append(r)
        else:
            table_rows.append(r)

    num_t = len(table_rows)
    num_c = len(col_rows)

    # HeteroData shape sanity
    ht = hetero_data['table'].x.shape[0]
    hc = hetero_data['column'].x.shape[0]
    hf = hetero_data['fk_node'].x.shape[0]
    assert num_t == ht and num_c == hc and len(fk_rows) == hf, (
        f"shape mismatch: scores (T={num_t},C={num_c},F={len(fk_rows)}) vs "
        f"cache (T={ht},C={hc},F={hf})")

    table_to_id = {r['node_name']: i for i, r in enumerate(table_rows)}
    col_to_id = {r['node_name']: i for i, r in enumerate(col_rows)}
    fk_to_id = {r['node_name']: i for i, r in enumerate(fk_rows)}

    node_metadata = {}
    for name, i in table_to_id.items():
        node_metadata[i] = name
    for name, i in col_to_id.items():
        node_metadata[i + num_t] = name
    for name, i in fk_to_id.items():
        node_metadata[i + num_t + num_c] = name

    edges: List[Tuple[int, int]] = []
    edge_types: List[str] = []

    def add_edges(et_key, et_label, s_offset, d_offset):
        if et_key in hetero_data.edge_types:
            ei = hetero_data[et_key].edge_index
            for i in range(ei.shape[1]):
                edges.append((int(ei[0, i]) + s_offset, int(ei[1, i]) + d_offset))
                edge_types.append(et_label)

    # (table, has_column, column) -> (t, c+num_t) labeled 'belongs_to' (builder convention)
    add_edges(('table', 'has_column', 'column'), 'belongs_to', 0, num_t)
    # (column, is_source_of, fk_node) -> (c+num_t, f+num_t+num_c)
    add_edges(('column', 'is_source_of', 'fk_node'), 'is_source_of', num_t, num_t + num_c)
    # (fk_node, points_to, column) -> (f+num_t+num_c, c+num_t)
    add_edges(('fk_node', 'points_to', 'column'), 'points_to', num_t + num_c, num_t)
    # (table, table_to_table, table) -> (t, t)
    add_edges(('table', 'table_to_table', 'table'), 'table_to_table', 0, 0)

    metadata = {
        'table_to_id': table_to_id,
        'col_to_id': col_to_id,
        'fk_to_id': fk_to_id,
        'node_metadata': node_metadata,
        'edges': edges,
        'edge_types': edge_types,
    }
    scores = [float(r['score']) for r in table_rows] + \
             [float(r['score']) for r in col_rows] + \
             [float(r['score']) for r in fk_rows]
    return metadata, scores


def _build_subgraph_dict(selected_ids, node_metadata):
    """Replicate schema_linking.py:196-206 subgraph_dict construction."""
    subgraph_dict: Dict[str, List[str]] = {}
    for flat_id in selected_ids:
        name = node_metadata.get(int(flat_id), str(flat_id))
        if '.' in name and '->' not in name:
            tbl, col = name.split('.', 1)
            subgraph_dict.setdefault(tbl, []).append(col)
        elif '->' not in name:
            subgraph_dict.setdefault(name, [])
    return subgraph_dict


def _apply_auto_join_keys(subgraph_dict, node_metadata):
    """Replicate schema_linking.py:208-233 auto_join_keys post-processing.

    Adds FK columns to subgraph_dict for FK names (\"t1.c1->t2.c2\") whose both
    parent tables are already present, even when the column nodes are absent
    from the graph (the in-extractor force_fk_columns only covers columns that
    exist as nodes).
    """
    if len(subgraph_dict) < 2:
        return
    for idx, name in node_metadata.items():
        s = str(name)
        if '->' not in s:
            continue
        parts = s.split('->')
        if len(parts) != 2:
            continue
        src, dst = parts[0].strip(), parts[1].strip()
        src_tbl = src.split('.')[0] if '.' in src else src
        dst_tbl = dst.split('.')[0] if '.' in dst else dst
        src_col = src.split('.', 1)[1] if '.' in src else None
        dst_col = dst.split('.', 1)[1] if '.' in dst else None
        if src_tbl in subgraph_dict and dst_tbl in subgraph_dict:
            if src_col and src_col not in subgraph_dict[src_tbl]:
                subgraph_dict[src_tbl].append(src_col)
            if dst_col and dst_col not in subgraph_dict[dst_tbl]:
                subgraph_dict[dst_tbl].append(dst_col)


def _subgraph_to_pred(subgraph_dict, gold_cols):
    """Replicate NoneFilter flatten + main.py:99-123 pred collapsing."""
    pred_tables, pred_cols = set(), set()
    for tbl, cols in subgraph_dict.items():
        if not cols:
            # Table with no columns → still counts as a predicted table
            pred_tables.add(tbl.lower())
            continue
        for col in cols:
            node = f"{tbl}.{col}"
            pred_tables.add(tbl.lower())
            col_lower = col.lower()
            tbl_col_lower = f"{tbl.lower()}.{col_lower}"
            if tbl_col_lower in gold_cols:
                pred_cols.add(tbl_col_lower)
            else:
                pred_cols.add(col_lower)
    return pred_tables, pred_cols


def flat_ids_to_pred(selected_ids, node_metadata, gold_cols, auto_join_keys=True):
    """Replicate main pipeline's path: extractor output -> subgraph_dict ->
    (optional) auto_join_keys -> NoneFilter flatten -> main.py collapsing.
    """
    subgraph_dict = _build_subgraph_dict(selected_ids, node_metadata)
    if auto_join_keys:
        _apply_auto_join_keys(subgraph_dict, node_metadata)
    return _subgraph_to_pred(subgraph_dict, gold_cols)


def compute_r_p(gold_cols: set, pred_cols: set) -> Tuple[float, float]:
    if not gold_cols and not pred_cols:
        return 1.0, 1.0
    inter = gold_cols & pred_cols
    r = len(inter) / len(gold_cols) if gold_cols else 0.0
    p = len(inter) / len(pred_cols) if pred_cols else 0.0
    return r, p


def run_config(ExtractorCls, extractor_kwargs, per_qid_state, desc):
    extractor = ExtractorCls(**extractor_kwargs)
    rs, ps = [], []
    for metadata, scores, gold_cols in tqdm(per_qid_state, desc=desc, leave=False):
        selected, _ = extractor.extract(metadata, scores)
        _, pred_cols = flat_ids_to_pred(selected, metadata['node_metadata'], gold_cols)
        r, p = compute_r_p(gold_cols, pred_cols)
        rs.append(r); ps.append(p)
    R = float(np.mean(rs)); P = float(np.mean(ps))
    F1 = 2 * R * P / (R + P) if (R + P) > 0 else 0.0
    return R, P, F1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph-cache',
                    default='data/processed/dev_plm_graphs.pt',
                    help='HeteroData graph cache for dev set')
    ap.add_argument('--score-analysis',
                    default='outputs/experiments/s03_gat_ensemble/a10_fk_steiner/'
                            's03_a10_09_fk_steiner_r08/'
                            'score_analysis_s03_a10_09_fk_steiner_r08.jsonl',
                    help='Per-query node scores (reused Selector outputs)')
    ap.add_argument('--output-jsonl',
                    default='outputs/experiments/s03_gat_ensemble/a10_fk_steiner/'
                            's03_a10_09_fk_steiner_r08/'
                            'output_s03_a10_09_fk_steiner_r08.jsonl',
                    help='Previous run outputs (for gold_cols)')
    ap.add_argument('--out-csv',
                    default='notebooks/analysis_results/'
                            'fk_steiner_percentile_sweep.csv')
    ap.add_argument('--percentile-step', type=int, default=5,
                    help='Sweep grid step in percentile (default 5 → 21 points)')
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'src'))
    # Silence extractor init/debug logs (84 configs × 1534 queries = very chatty)
    logging.getLogger('modules.extractors.pcst').setLevel(logging.WARNING)

    from modules.extractors.pcst import FKBackboneSteinerExtractor

    graph_cache_path = root / args.graph_cache
    score_path = root / args.score_analysis
    output_path = root / args.output_jsonl
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = root / out_csv

    print(f"Loading graph cache: {graph_cache_path}")
    graph_cache = torch.load(graph_cache_path, weights_only=False)

    print(f"Loading score_analysis: {score_path}")
    scores_by_qid: Dict[int, List[Dict[str, Any]]] = {}
    with open(score_path) as f:
        for line in f:
            r = json.loads(line)
            scores_by_qid.setdefault(r['query_id'], []).append(r)

    print(f"Loading gold (output jsonl): {output_path}")
    gold_by_qid: Dict[int, set] = {}
    with open(output_path) as f:
        for line in f:
            r = json.loads(line)
            gold_by_qid[r['question_id']] = set(r['gold_cols'])

    n_queries = len(graph_cache)
    assert n_queries == len(scores_by_qid) == len(gold_by_qid), (
        f"count mismatch: graphs={n_queries}, scores={len(scores_by_qid)}, "
        f"gold={len(gold_by_qid)}")

    print(f"Building per-query metadata ({n_queries} queries)...")
    per_qid_state = []
    for qid in tqdm(range(n_queries), desc='build'):
        metadata, scores = build_metadata(graph_cache[qid], scores_by_qid[qid])
        per_qid_state.append((metadata, scores, gold_by_qid[qid]))

    scopes = ('global', 'all_cols', 'closed_cols', 'per_table')
    percentiles = list(range(0, 101, args.percentile_step))

    results = []

    # 1) Absolute anchor (a10_09 config: θ_r=0.8)
    R, P, F1 = run_config(
        FKBackboneSteinerExtractor,
        {'column_recovery_threshold': 0.8},
        per_qid_state,
        desc='abs=0.8')
    results.append({'scope': 'abs_anchor', 'percentile': 0.8, 'R': R, 'P': P, 'F1': F1})
    print(f"  [abs_anchor θ_r=0.8] R={R:.4f} P={P:.4f} F1={F1:.4f}")

    # 2) Percentile sweep
    for scope in scopes:
        for p_val in percentiles:
            R, P, F1 = run_config(
                FKBackboneSteinerExtractor,
                {'column_recovery_percentile': float(p_val),
                 'column_recovery_percentile_scope': scope},
                per_qid_state,
                desc=f'{scope} p={p_val}')
            results.append({'scope': scope, 'percentile': float(p_val),
                            'R': R, 'P': P, 'F1': F1})
            print(f"  [{scope:12s} p={p_val:3d}] R={R:.4f} P={P:.4f} F1={F1:.4f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scope', 'percentile', 'R', 'P', 'F1'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'scope': r['scope'],
                'percentile': r['percentile'],
                'R': f"{r['R']:.4f}",
                'P': f"{r['P']:.4f}",
                'F1': f"{r['F1']:.4f}",
            })
    print(f"\nWrote {out_csv} ({len(results)} rows)")


if __name__ == '__main__':
    main()
