#!/usr/bin/env python3
"""V6-W2 selector-only top-K 종합 분석 — Val R@15 ↔ top-K F1 disconnect 정량.

score_analysis_*.jsonl (per-query node score + is_gold) 위:
  1. top-K sensitivity matrix (K in {5,10,15,20,25,30,50,100}) micro R/P/F1
  2. gold vs nongold score distribution (mean/std/percentile/gap)
  3. ROC-AUC + PR-AUC (per-query macro avg + pooled) — top-K 무관 global ranking quality
  4. per-query 분해 (best cell per query × difficulty/gold-count/node-count)

산출: outputs/analysis/v6w2_selector_topk_*.{csv,json} + stdout.
"""
import json, os, csv, math, statistics
from collections import defaultdict, Counter

ROOT = "/home/hyeonjin/thesis_refactored"
SCORE_DIR = os.path.join(ROOT, "outputs/experiments/s06_gat_bottleneck_fix/w2_edge_type_split")
OUTDIR = os.path.join(ROOT, "outputs/analysis")
os.makedirs(OUTDIR, exist_ok=True)
DEV = os.path.join(ROOT, "data/raw/BIRD_dev/dev.json")

CELLS = ["v6w2_p2_sum", "v6w2_p2_standalone", "v6w2_p2_phase1", "v6w2_p2_standalone_no_selfloop"]
SHORT = {"v6w2_p2_sum": "sum", "v6w2_p2_standalone": "standalone",
         "v6w2_p2_phase1": "phase1", "v6w2_p2_standalone_no_selfloop": "no_selfloop"}
KS = [5, 10, 15, 20, 25, 30, 50, 100]
VAL_R15 = {"sum": 0.5692, "standalone": 0.5666, "phase1": 0.5697, "no_selfloop": 0.5573}


def load_cell(cell):
    pq = defaultdict(list)  # qid -> [(score, is_gold)]
    with open(os.path.join(SCORE_DIR, cell, f"score_analysis_{cell}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            pq[r["query_id"]].append((r["score"], bool(r.get("is_gold", False))))
    return pq


def topk_micro(pq, k):
    tp = fp = fn = 0
    for qid, items in pq.items():
        items_s = sorted(items, key=lambda x: x[0], reverse=True)
        topk = items_s[:k]
        gold_total = sum(1 for _, g in items if g)
        t = sum(1 for _, g in topk if g)
        tp += t; fp += (len(topk) - t); fn += (gold_total - t)
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return R, P, F


def per_query_f1(pq, k):
    out = {}
    for qid, items in pq.items():
        items_s = sorted(items, key=lambda x: x[0], reverse=True)
        topk = items_s[:k]
        gold_total = sum(1 for _, g in items if g)
        t = sum(1 for _, g in topk if g)
        P = t / len(topk) if topk else 0.0
        R = t / gold_total if gold_total else 0.0
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        out[qid] = F
    return out


def percentiles(vs, ps=(25, 50, 75, 90)):
    vs = sorted(vs)
    n = len(vs)
    out = {}
    for p in ps:
        if n == 0:
            out[p] = float("nan"); continue
        idx = min(n - 1, int(round((p / 100) * (n - 1))))
        out[p] = vs[idx]
    return out


def roc_auc_query(items):
    """Mann-Whitney rank-based ROC-AUC for one query. None if single class."""
    pos = [s for s, g in items if g]
    neg = [s for s, g in items if not g]
    if not pos or not neg:
        return None
    # rank all scores (ascending), average ties
    allv = sorted([s for s, _ in items])
    n = len(allv)
    # build rank lookup with tie-average
    rank = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and allv[j + 1] == allv[i]:
            j += 1
        avg = (i + j) / 2.0 + 1
        rank[allv[i]] = avg
        i = j + 1
    sum_ranks_pos = sum(rank[s] for s in pos)
    npos = len(pos); nneg = len(neg)
    auc = (sum_ranks_pos - npos * (npos + 1) / 2) / (npos * nneg)
    return auc


def pr_auc_query(items):
    """Average precision (PR-AUC approx) for one query. None if no positives."""
    npos = sum(1 for _, g in items if g)
    if npos == 0:
        return None
    items_s = sorted(items, key=lambda x: x[0], reverse=True)
    tp = 0; ap = 0.0
    for i, (_, g) in enumerate(items_s, 1):
        if g:
            tp += 1
            ap += tp / i  # precision at this recall point
    return ap / npos


def main():
    data = {c: load_cell(c) for c in CELLS}
    qids = sorted(next(iter(data.values())).keys())

    # difficulty map
    diff = {}
    if os.path.exists(DEV):
        dev = json.load(open(DEV))
        for i, q in enumerate(dev):
            diff[i] = q.get("difficulty", "?")

    # ---- 1. top-K matrix ----
    print("=== 1. top-K sensitivity matrix (micro F1) ===")
    print(f"{'K':>4} | " + " | ".join(f"{SHORT[c]:>11}" for c in CELLS))
    matrix = {}
    for k in KS:
        row = {}
        for c in CELLS:
            R, P, F = topk_micro(data[c], k)
            row[c] = (R, P, F)
        matrix[k] = row
        # ranking by F1
        order = sorted(CELLS, key=lambda c: -row[c][2])
        rankstr = " > ".join(SHORT[c] for c in order)
        print(f"{k:>4} | " + " | ".join(f"{row[c][2]:.4f}" for c in CELLS) + f"   [{rankstr}]")

    # ---- 2. score distribution ----
    print("\n=== 2. score distribution (gold vs nongold, pooled) ===")
    distrows = []
    for c in CELLS:
        gold = [s for items in data[c].values() for s, g in items if g]
        nong = [s for items in data[c].values() for s, g in items if not g]
        gp = percentiles(gold); npc = percentiles(nong)
        row = dict(cell=SHORT[c],
                   gold_mean=statistics.mean(gold), gold_std=statistics.pstdev(gold),
                   gold_p50=gp[50], gold_p90=gp[90],
                   nong_mean=statistics.mean(nong), nong_std=statistics.pstdev(nong),
                   nong_p50=npc[50], nong_p90=npc[90],
                   gap_mean=statistics.mean(gold) - statistics.mean(nong),
                   gap_p50=gp[50] - npc[50])
        distrows.append(row)
        print(f"  {SHORT[c]:12s} gold μ={row['gold_mean']:.4f}±{row['gold_std']:.4f} p50={row['gold_p50']:.4f} p90={row['gold_p90']:.4f}"
              f" | nong μ={row['nong_mean']:.4f}±{row['nong_std']:.4f} p50={row['nong_p50']:.4f}"
              f" | gap_μ={row['gap_mean']:+.4f} gap_p50={row['gap_p50']:+.4f}")

    # ---- 3. AUC ----
    print("\n=== 3. ROC-AUC + PR-AUC (per-query macro avg) ===")
    aucrows = []
    for c in CELLS:
        rocs = []; prs = []
        for qid, items in data[c].items():
            r = roc_auc_query(items)
            if r is not None:
                rocs.append(r)
            p = pr_auc_query(items)
            if p is not None:
                prs.append(p)
        row = dict(cell=SHORT[c], roc_auc=statistics.mean(rocs), pr_auc=statistics.mean(prs),
                   n_roc=len(rocs), val_r15=VAL_R15[SHORT[c]], top20_f1=matrix[20][c][2])
        aucrows.append(row)
        print(f"  {SHORT[c]:12s} ROC-AUC={row['roc_auc']:.4f}  PR-AUC={row['pr_auc']:.4f}  "
              f"(top20F1={row['top20_f1']:.4f}, ValR@15={row['val_r15']:.4f})")
    # ranking comparison
    print("\n  ranking comparison:")
    for metric, key in [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("top20-F1", "top20_f1"), ("ValR@15", "val_r15")]:
        order = sorted(aucrows, key=lambda r: -r[key])
        print(f"    {metric:10s}: " + " > ".join(f"{r['cell']}({r[key]:.4f})" for r in order))

    # ---- 4. per-query decomposition ----
    print("\n=== 4. per-query 분해 (best cell @ top-20 F1) ===")
    pqf1 = {c: per_query_f1(data[c], 20) for c in CELLS}
    best_counter = Counter()
    best_by_diff = defaultdict(Counter)
    # per cell: gold count + node count for queries where it's strictly best
    for qid in qids:
        f1s = {c: pqf1[c][qid] for c in CELLS}
        mx = max(f1s.values())
        winners = [c for c in CELLS if f1s[c] == mx]
        # attribute to single winner only if unique; else 'tie'
        if len(winners) == 1:
            best_counter[SHORT[winners[0]]] += 1
            best_by_diff[diff.get(qid, "?")][SHORT[winners[0]]] += 1
        else:
            best_counter["tie"] += 1
            best_by_diff[diff.get(qid, "?")]["tie"] += 1
    print("  best-cell count (unique winner):", dict(best_counter))
    print("  by difficulty:")
    for d in ["simple", "moderate", "challenging"]:
        print(f"    {d:12s}: {dict(best_by_diff[d])}")

    # characteristics of sum-best vs phase1-best query subsets
    def subset_stats(target_short):
        gold_counts = []; node_counts = []
        for qid in qids:
            f1s = {c: pqf1[c][qid] for c in CELLS}
            mx = max(f1s.values())
            winners = [SHORT[c] for c in CELLS if f1s[c] == mx]
            if winners == [target_short]:
                items = data["v6w2_p2_sum"][qid]
                gold_counts.append(sum(1 for _, g in items if g))
                node_counts.append(len(items))
        if not gold_counts:
            return None
        return dict(n=len(gold_counts), gold_mean=statistics.mean(gold_counts),
                    node_mean=statistics.mean(node_counts))
    print("\n  winner-subset characteristics (gold count + total node count):")
    for sh in ["sum", "standalone", "phase1", "no_selfloop"]:
        st = subset_stats(sh)
        if st:
            print(f"    {sh:12s} n={st['n']:>4} gold_avg={st['gold_mean']:.2f} node_avg={st['node_mean']:.2f}")

    # ---- write artifacts ----
    with open(os.path.join(OUTDIR, "v6w2_selector_topk_matrix_2026-06-05.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K"] + [f"{SHORT[c]}_{m}" for c in CELLS for m in ("R", "P", "F1")])
        for k in KS:
            row = [k]
            for c in CELLS:
                R, P, F = matrix[k][c]
                row += [round(R, 4), round(P, 4), round(F, 4)]
            w.writerow(row)
    summary = dict(
        topk_matrix={k: {SHORT[c]: dict(R=round(matrix[k][c][0], 4), P=round(matrix[k][c][1], 4),
                                        F1=round(matrix[k][c][2], 4)) for c in CELLS} for k in KS},
        score_dist=[{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()} for r in distrows],
        auc=[{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()} for r in aucrows],
        best_cell_count=dict(best_counter),
        best_by_difficulty={d: dict(best_by_diff[d]) for d in ["simple", "moderate", "challenging"]},
    )
    with open(os.path.join(OUTDIR, "v6w2_selector_topk_analysis_2026-06-05.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote outputs/analysis/v6w2_selector_topk_matrix_2026-06-05.csv")
    print("Wrote outputs/analysis/v6w2_selector_topk_analysis_2026-06-05.json")


if __name__ == "__main__":
    main()
