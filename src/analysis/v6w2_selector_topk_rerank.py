"""V6-W2 selector-only top-K rerank R/P/F1 계산.

score_analysis_*.jsonl 위 per-query score 활용 top-K 노드 선택 후 R/P/F1 측정.
4 cells (v6w2_p2_{standalone, phase1, standalone_no_selfloop, sum}) × top-K ∈ {20} (default).

Usage:
    python src/analysis/v6w2_selector_topk_rerank.py [--top_k 20]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

CELLS = [
    "v6w2_p2_standalone",
    "v6w2_p2_phase1",
    "v6w2_p2_standalone_no_selfloop",
    "v6w2_p2_sum",
]
SCORE_DIR = Path("outputs/experiments/s06_gat_bottleneck_fix/w2_edge_type_split")


def compute_topk_metrics(score_file: Path, top_k: int):
    """per-query top-K selection → micro R/P/F1."""
    per_query = defaultdict(list)
    if not score_file.exists():
        return None
    with score_file.open() as f:
        for line in f:
            rec = json.loads(line)
            per_query[rec["query_id"]].append((rec["score"], rec.get("is_gold", False)))

    if not per_query:
        return None

    tp_sum = fp_sum = fn_sum = 0
    total_queries = len(per_query)
    for qid, items in per_query.items():
        items.sort(key=lambda x: x[0], reverse=True)
        topk = items[:top_k]
        gold_total = sum(1 for _, g in items if g)
        tp = sum(1 for _, g in topk if g)
        fp = top_k - tp if len(topk) >= top_k else len(topk) - tp
        fn = gold_total - tp
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    return {
        "queries": total_queries,
        "tp": tp_sum,
        "fp": fp_sum,
        "fn": fn_sum,
        "recall": round(micro_r, 4),
        "precision": round(micro_p, 4),
        "f1": round(micro_f1, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    print(f"V6-W2 selector top-{args.top_k} rerank R/P/F1 (micro)")
    print(f"{'Cell':<34} {'Queries':>8} {'R':>8} {'P':>8} {'F1':>8}")
    print("-" * 70)
    for cell in CELLS:
        sf = SCORE_DIR / cell / f"score_analysis_{cell}.jsonl"
        m = compute_topk_metrics(sf, args.top_k)
        if m is None:
            print(f"{cell:<34} {'N/A':>8}")
            continue
        print(f"{cell:<34} {m['queries']:>8} {m['recall']:>8.4f} {m['precision']:>8.4f} {m['f1']:>8.4f}")


if __name__ == "__main__":
    main()
