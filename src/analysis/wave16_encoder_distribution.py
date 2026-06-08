"""Wave 16 — Encoder Backbone Score Distribution 분석 (post-measurement analyzer).

DECISIONS 2026-05-21 (Wave 16) §7.3 + §7.5 정합. Qwen3-Embedding-0.6B (Wave 16) vs
all-MiniLM-L6-v2 (Wave 6 P2 M4 anchor) 의 selector score distribution 비교 —
TP-TN spread + histogram + per-DB + per-difficulty + score scale 정량.

Note: score_analysis_*.jsonl 의 'score' = Ensemble Selector α=0.5 의 ensemble score
(`α * cos + (1-α) * GAT_sigmoid`) — Wave 16 + Wave 6 anchor 모두 α=0.5 동일 retain
위 apples-to-apples 비교 가능. 본 script 는 ensemble score distribution 위 PLM
backbone 영향 정량 — "embedding quality (cosine 위 GAT 기여 후) 의 TP-TN 분리도".

Spec:
  - TP-TN spread = mean(gold scores) − mean(non-gold scores), per query + overall
  - Histogram bins=20, range=[0, 1]
  - Per-DB / per-difficulty 분해 (BIRD-Dev dev.json 의 difficulty field)
  - Score scale: overall mean / std / quantiles, gold mean/std, non-gold mean/std

산출:
  - notebooks/analysis_results/wave16_encoder_distribution_2026-05-22.md
  - outputs/analysis/wave16_encoder_distribution_2026-05-22.csv (per-DB + overall + per-diff rows)
  - outputs/analysis/wave16_encoder_distribution_2026-05-22.json (full summary)
  - outputs/analysis/wave16_encoder_distribution_histograms_2026-05-22.json (bins=20, both encoders)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path("/home/hyeonjin/thesis_refactored")

CELLS = [
    ("wave16_qwen3",
     ROOT / "outputs/experiments/abl/wave16_encoder_backbone/m16_qwen3_0.6b_m4/score_analysis_m16_qwen3_0.6b_m4.jsonl",
     "Qwen3-Embedding-0.6B (1024-dim, 600M params)"),
    ("wave6_minilm",
     ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional/score_analysis_w6_p2_m4_bidirectional.jsonl",
     "all-MiniLM-L6-v2 (384-dim, 22M params)"),
]

HIST_BINS = 20
RANGE_LO, RANGE_HI = 0.0, 1.0


def load_gold_lookup() -> Dict[int, Dict]:
    with (ROOT / "data/raw/BIRD_dev/dev.json").open() as f:
        items = json.load(f)
    return {
        int(it["question_id"]): {
            "db_id": it.get("db_id"),
            "difficulty": it.get("difficulty", "unknown"),
        }
        for it in items
    }


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    s = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    return m, s


def _histogram(vals: List[float], bins: int = HIST_BINS, lo: float = RANGE_LO, hi: float = RANGE_HI) -> List[int]:
    counts = [0] * bins
    width = (hi - lo) / bins
    for v in vals:
        if v < lo or v > hi:
            continue
        idx = int((v - lo) / width)
        if idx == bins:
            idx = bins - 1
        counts[idx] += 1
    return counts


def aggregate_cell(score_path: Path, gold_lookup: Dict[int, Dict]) -> Dict:
    gold_scores: List[float] = []
    nongold_scores: List[float] = []
    per_q_spread: List[Tuple[int, float]] = []
    per_db_buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"gold": [], "nongold": [], "per_q_spreads": []})
    per_diff_buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"gold": [], "nongold": [], "per_q_spreads": []})

    cur_qid = None
    cur_gold: List[float] = []
    cur_nongold: List[float] = []

    def _flush(qid: int):
        if not cur_gold and not cur_nongold:
            return
        meta = gold_lookup.get(int(qid)) or {}
        db_id = meta.get("db_id", "unknown")
        difficulty = meta.get("difficulty", "unknown")

        # per-query spread = mean(gold) - mean(nongold) (skip if no gold or no nongold)
        if cur_gold and cur_nongold:
            spread = (sum(cur_gold) / len(cur_gold)) - (sum(cur_nongold) / len(cur_nongold))
            per_q_spread.append((qid, spread))
            per_db_buckets[db_id]["per_q_spreads"].append(spread)
            per_diff_buckets[difficulty]["per_q_spreads"].append(spread)

        gold_scores.extend(cur_gold)
        nongold_scores.extend(cur_nongold)
        per_db_buckets[db_id]["gold"].extend(cur_gold)
        per_db_buckets[db_id]["nongold"].extend(cur_nongold)
        per_diff_buckets[difficulty]["gold"].extend(cur_gold)
        per_diff_buckets[difficulty]["nongold"].extend(cur_nongold)

    with score_path.open() as f:
        for line in f:
            d = json.loads(line)
            qid = int(d["query_id"])
            if cur_qid is None:
                cur_qid = qid
            if qid != cur_qid:
                _flush(cur_qid)
                cur_qid = qid
                cur_gold = []
                cur_nongold = []
            if d["is_gold"]:
                cur_gold.append(d["score"])
            else:
                cur_nongold.append(d["score"])
        if cur_qid is not None:
            _flush(cur_qid)

    all_scores = gold_scores + nongold_scores
    all_sorted = sorted(all_scores)
    gold_sorted = sorted(gold_scores)
    nongold_sorted = sorted(nongold_scores)

    g_mean, g_std = _mean_std(gold_scores)
    n_mean, n_std = _mean_std(nongold_scores)
    overall_mean, overall_std = _mean_std(all_scores)

    # Aggregate per-query spread
    spreads = [s for _q, s in per_q_spread]
    spread_mean, spread_std = _mean_std(spreads)

    # per-DB / per-diff summaries
    per_db_summary: Dict[str, Dict] = {}
    for db_id, b in per_db_buckets.items():
        gm, _ = _mean_std(b["gold"])
        nm, _ = _mean_std(b["nongold"])
        sm, _ = _mean_std(b["per_q_spreads"])
        per_db_summary[db_id] = {
            "n_gold": len(b["gold"]),
            "n_nongold": len(b["nongold"]),
            "n_queries": len(b["per_q_spreads"]),
            "gold_mean": gm,
            "nongold_mean": nm,
            "overall_spread": (gm - nm) if (b["gold"] and b["nongold"]) else float("nan"),
            "per_q_spread_mean": sm,
        }

    per_diff_summary: Dict[str, Dict] = {}
    for diff, b in per_diff_buckets.items():
        gm, _ = _mean_std(b["gold"])
        nm, _ = _mean_std(b["nongold"])
        sm, _ = _mean_std(b["per_q_spreads"])
        per_diff_summary[diff] = {
            "n_gold": len(b["gold"]),
            "n_nongold": len(b["nongold"]),
            "n_queries": len(b["per_q_spreads"]),
            "gold_mean": gm,
            "nongold_mean": nm,
            "overall_spread": (gm - nm) if (b["gold"] and b["nongold"]) else float("nan"),
            "per_q_spread_mean": sm,
        }

    return {
        "n_total_scores": len(all_scores),
        "n_gold_scores": len(gold_scores),
        "n_nongold_scores": len(nongold_scores),
        "score_overall": {"mean": overall_mean, "std": overall_std,
                          "q25": _quantile(all_sorted, 0.25),
                          "q50": _quantile(all_sorted, 0.50),
                          "q75": _quantile(all_sorted, 0.75),
                          "q95": _quantile(all_sorted, 0.95)},
        "score_gold": {"mean": g_mean, "std": g_std,
                       "q25": _quantile(gold_sorted, 0.25),
                       "q50": _quantile(gold_sorted, 0.50),
                       "q75": _quantile(gold_sorted, 0.75),
                       "q95": _quantile(gold_sorted, 0.95)},
        "score_nongold": {"mean": n_mean, "std": n_std,
                          "q25": _quantile(nongold_sorted, 0.25),
                          "q50": _quantile(nongold_sorted, 0.50),
                          "q75": _quantile(nongold_sorted, 0.75),
                          "q95": _quantile(nongold_sorted, 0.95)},
        "tp_tn_spread_overall": g_mean - n_mean,
        "tp_tn_spread_per_query": {"mean": spread_mean, "std": spread_std, "n_q": len(spreads)},
        "histogram": {
            "bins": HIST_BINS, "range": [RANGE_LO, RANGE_HI],
            "gold_counts":    _histogram(gold_scores),
            "nongold_counts": _histogram(nongold_scores),
            "all_counts":     _histogram(all_scores),
        },
        "per_db": per_db_summary,
        "per_difficulty": per_diff_summary,
    }


def main():
    gold_lookup = load_gold_lookup()
    print(f"Loaded {len(gold_lookup)} gold records from dev.json")
    print()

    summaries: Dict[str, Dict] = {}
    for cell_tag, score_path, label in CELLS:
        print(f"=== {cell_tag} === ({label})")
        summary = aggregate_cell(score_path, gold_lookup)
        summary["cell_tag"] = cell_tag
        summary["label"] = label
        summaries[cell_tag] = summary

        sg = summary["score_gold"]; sn = summary["score_nongold"]; ov = summary["score_overall"]
        print(f"  scores: n_total={summary['n_total_scores']}  n_gold={summary['n_gold_scores']}  n_nongold={summary['n_nongold_scores']}")
        print(f"  overall: mean={ov['mean']:.4f}  std={ov['std']:.4f}  q25/50/75/95={ov['q25']:.4f}/{ov['q50']:.4f}/{ov['q75']:.4f}/{ov['q95']:.4f}")
        print(f"  gold:    mean={sg['mean']:.4f}  std={sg['std']:.4f}  q25/50/75/95={sg['q25']:.4f}/{sg['q50']:.4f}/{sg['q75']:.4f}/{sg['q95']:.4f}")
        print(f"  nongold: mean={sn['mean']:.4f}  std={sn['std']:.4f}  q25/50/75/95={sn['q25']:.4f}/{sn['q50']:.4f}/{sn['q75']:.4f}/{sn['q95']:.4f}")
        print(f"  TP-TN spread overall: {summary['tp_tn_spread_overall']:.4f}")
        pq = summary['tp_tn_spread_per_query']
        print(f"  TP-TN spread per-query: mean={pq['mean']:.4f}  std={pq['std']:.4f}  n_q={pq['n_q']}")
        print()
        print(f"  per-difficulty TP-TN spread (overall = gold_mean - nongold_mean):")
        for d in ("simple", "moderate", "challenging"):
            pd = summary["per_difficulty"].get(d)
            if pd:
                print(f"    {d:12s}  n_q={pd['n_queries']:4d}  gold_mean={pd['gold_mean']:.4f}  nongold_mean={pd['nongold_mean']:.4f}  spread={pd['overall_spread']:+.4f}  per_q_spread_mean={pd['per_q_spread_mean']:+.4f}")
        print()

    # Δ TP-TN spread comparison
    print("=" * 100)
    print("Δ (Qwen3 − MiniLM)")
    print("=" * 100)
    q = summaries["wave16_qwen3"]; m = summaries["wave6_minilm"]
    print(f"  TP-TN spread overall:     Δ = {q['tp_tn_spread_overall'] - m['tp_tn_spread_overall']:+.4f}  (Qwen3={q['tp_tn_spread_overall']:.4f}, MiniLM={m['tp_tn_spread_overall']:.4f})")
    print(f"  TP-TN spread per-query:   Δ = {q['tp_tn_spread_per_query']['mean'] - m['tp_tn_spread_per_query']['mean']:+.4f}  (Qwen3={q['tp_tn_spread_per_query']['mean']:.4f}, MiniLM={m['tp_tn_spread_per_query']['mean']:.4f})")
    print(f"  Overall score mean:       Δ = {q['score_overall']['mean'] - m['score_overall']['mean']:+.4f}  (Qwen3={q['score_overall']['mean']:.4f}, MiniLM={m['score_overall']['mean']:.4f})")
    print(f"  Gold score mean:          Δ = {q['score_gold']['mean'] - m['score_gold']['mean']:+.4f}  (Qwen3={q['score_gold']['mean']:.4f}, MiniLM={m['score_gold']['mean']:.4f})")
    print(f"  Nongold score mean:       Δ = {q['score_nongold']['mean'] - m['score_nongold']['mean']:+.4f}  (Qwen3={q['score_nongold']['mean']:.4f}, MiniLM={m['score_nongold']['mean']:.4f})")
    print()

    # CSV
    csv_path = ROOT / "outputs/analysis/wave16_encoder_distribution_2026-05-22.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_tag", "scope", "key", "n", "gold_mean", "nongold_mean", "overall_spread", "per_q_spread_mean", "overall_mean", "overall_std"])
        for cell_tag, s in summaries.items():
            w.writerow([cell_tag, "overall", "(all)",
                        s["n_total_scores"],
                        round(s["score_gold"]["mean"], 4),
                        round(s["score_nongold"]["mean"], 4),
                        round(s["tp_tn_spread_overall"], 4),
                        round(s["tp_tn_spread_per_query"]["mean"], 4),
                        round(s["score_overall"]["mean"], 4),
                        round(s["score_overall"]["std"], 4)])
            for d in ("simple", "moderate", "challenging"):
                pd = s["per_difficulty"].get(d)
                if not pd:
                    continue
                w.writerow([cell_tag, "difficulty", d,
                            pd["n_queries"],
                            round(pd["gold_mean"], 4),
                            round(pd["nongold_mean"], 4),
                            round(pd["overall_spread"], 4),
                            round(pd["per_q_spread_mean"], 4),
                            "", ""])
            for db_id, pd in sorted(s["per_db"].items()):
                w.writerow([cell_tag, "db", db_id,
                            pd["n_queries"],
                            round(pd["gold_mean"], 4),
                            round(pd["nongold_mean"], 4),
                            round(pd["overall_spread"], 4),
                            round(pd["per_q_spread_mean"], 4),
                            "", ""])
    print(f"→ csv:  {csv_path}")

    # JSON
    json_path = ROOT / "outputs/analysis/wave16_encoder_distribution_2026-05-22.json"
    with json_path.open("w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"→ json: {json_path}")

    # Histogram-only file (smaller for plotting)
    hist_path = ROOT / "outputs/analysis/wave16_encoder_distribution_histograms_2026-05-22.json"
    hists = {cell_tag: {"label": s["label"], "histogram": s["histogram"]} for cell_tag, s in summaries.items()}
    with hist_path.open("w") as f:
        json.dump(hists, f, indent=2, ensure_ascii=False)
    print(f"→ hist: {hist_path}")


if __name__ == "__main__":
    main()
