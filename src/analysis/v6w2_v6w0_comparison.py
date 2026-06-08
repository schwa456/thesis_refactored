#!/usr/bin/env python3
"""V6-W2 (DirectGATv2) vs V6-W0 (Projector/Ensemble) score quality 비교 + e2e ranking 반전 분석.

- V6-W0 baseline + V6-W2 4 cells: top-K matrix + ROC/PR-AUC + score distribution
- e2e: extractor threshold(0.1) 통과 노드 분석 — selector top-20 ↔ e2e EX 반전 mechanism
"""
import json, os, csv, math, statistics
from collections import defaultdict

ROOT = "/home/hyeonjin/thesis_refactored"
OUTDIR = os.path.join(ROOT, "outputs/analysis")
os.makedirs(OUTDIR, exist_ok=True)

# cell -> score_analysis path
W2DIR = os.path.join(ROOT, "outputs/experiments/s06_gat_bottleneck_fix/w2_edge_type_split")
W0FILE = os.path.join(ROOT, "outputs/experiments/s06_gat_bottleneck_fix/w0_baseline/v6w0_baseline_s11/score_analysis_v6w0_baseline_s11.jsonl")
PATHS = {
    "w0_baseline": W0FILE,
    "sum":         os.path.join(W2DIR, "v6w2_p2_sum/score_analysis_v6w2_p2_sum.jsonl"),
    "standalone":  os.path.join(W2DIR, "v6w2_p2_standalone/score_analysis_v6w2_p2_standalone.jsonl"),
    "phase1":      os.path.join(W2DIR, "v6w2_p2_phase1/score_analysis_v6w2_p2_phase1.jsonl"),
    "no_selfloop": os.path.join(W2DIR, "v6w2_p2_standalone_no_selfloop/score_analysis_v6w2_p2_standalone_no_selfloop.jsonl"),
}
ORDER = ["w0_baseline", "sum", "standalone", "phase1", "no_selfloop"]
KS = [5, 10, 15, 20, 25, 30, 50, 100]
# e2e EX (metrics.txt) + selector-only Val R@15 for context
E2E_EX = {"sum": 0.3116, "standalone": 0.2777, "phase1": 0.2653, "no_selfloop": 0.3331}
E2E_R = {"sum": 0.6714, "standalone": 0.6024, "phase1": 0.4853, "no_selfloop": 0.6344}
E2E_EXT_NODES = {"sum": 30.8990, "standalone": 26.9817, "phase1": 20.7647, "no_selfloop": 34.3325}


def load(path):
    pq = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pq[r["query_id"]].append((r["score"], bool(r.get("is_gold", False))))
    return pq


def topk_micro(pq, k):
    tp = fp = fn = 0
    for items in pq.values():
        s = sorted(items, key=lambda x: x[0], reverse=True)[:k]
        gt = sum(1 for _, g in items if g)
        t = sum(1 for _, g in s if g)
        tp += t; fp += len(s) - t; fn += gt - t
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return R, P, F


def pct(vs, p):
    vs = sorted(vs); n = len(vs)
    if n == 0:
        return float("nan")
    return vs[min(n - 1, int(round(p / 100 * (n - 1))))]


def roc_auc_q(items):
    pos = [s for s, g in items if g]; neg = [s for s, g in items if not g]
    if not pos or not neg:
        return None
    allv = sorted(s for s, _ in items); n = len(allv); rank = {}; i = 0
    while i < n:
        j = i
        while j + 1 < n and allv[j + 1] == allv[i]:
            j += 1
        rank[allv[i]] = (i + j) / 2.0 + 1; i = j + 1
    sr = sum(rank[s] for s in pos)
    return (sr - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def pr_auc_q(items):
    npos = sum(1 for _, g in items if g)
    if npos == 0:
        return None
    s = sorted(items, key=lambda x: x[0], reverse=True); tp = 0; ap = 0.0
    for i, (_, g) in enumerate(s, 1):
        if g:
            tp += 1; ap += tp / i
    return ap / npos


def thresh_stats(pq, thr=0.1):
    """extractor threshold(0.1) 통과 노드 분석 — gold/nongold 통과 수 + gold recall@thr."""
    pass_total = []; gold_recall = []; gold_pass = []; nong_pass = []
    for items in pq.values():
        gt = sum(1 for _, g in items if g)
        gp = sum(1 for sc, g in items if g and sc >= thr)
        npss = sum(1 for sc, g in items if (not g) and sc >= thr)
        pass_total.append(gp + npss); gold_pass.append(gp); nong_pass.append(npss)
        if gt:
            gold_recall.append(gp / gt)
    return dict(pass_mean=statistics.mean(pass_total), gold_pass_mean=statistics.mean(gold_pass),
                nong_pass_mean=statistics.mean(nong_pass), gold_recall_thr=statistics.mean(gold_recall))


def main():
    data = {c: load(PATHS[c]) for c in ORDER}

    # 1. top-K matrix
    print("=== 1. top-K micro F1 matrix (incl V6-W0) ===")
    print(f"{'K':>4} | " + " | ".join(f"{c:>11}" for c in ORDER))
    matrix = {}
    for k in KS:
        matrix[k] = {c: topk_micro(data[c], k) for c in ORDER}
        print(f"{k:>4} | " + " | ".join(f"{matrix[k][c][2]:.4f}     " for c in ORDER))
    print("\n  top-20 R/P/F1:")
    for c in ORDER:
        R, P, F = matrix[20][c]
        print(f"    {c:12s} R={R:.4f} P={P:.4f} F1={F:.4f}")

    # 2. score distribution
    print("\n=== 2. score distribution (gold vs nongold) ===")
    dist = {}
    for c in ORDER:
        gold = [s for it in data[c].values() for s, g in it if g]
        nong = [s for it in data[c].values() for s, g in it if not g]
        dist[c] = dict(gold_mean=statistics.mean(gold), gold_p50=pct(gold, 50), gold_p90=pct(gold, 90),
                       nong_mean=statistics.mean(nong), nong_p50=pct(nong, 50),
                       gap_mean=statistics.mean(gold) - statistics.mean(nong), gap_p50=pct(gold, 50) - pct(nong, 50))
        d = dist[c]
        print(f"  {c:12s} gold μ={d['gold_mean']:.4f} p50={d['gold_p50']:.4f} p90={d['gold_p90']:.4f}"
              f" | nong μ={d['nong_mean']:.4f} p50={d['nong_p50']:.4f} | gap_μ={d['gap_mean']:+.4f} gap_p50={d['gap_p50']:+.4f}")

    # 3. AUC
    print("\n=== 3. ROC-AUC + PR-AUC (per-query macro) ===")
    auc = {}
    for c in ORDER:
        rocs = [roc_auc_q(it) for it in data[c].values()]; rocs = [x for x in rocs if x is not None]
        prs = [pr_auc_q(it) for it in data[c].values()]; prs = [x for x in prs if x is not None]
        auc[c] = dict(roc=statistics.mean(rocs), pr=statistics.mean(prs))
        print(f"  {c:12s} ROC-AUC={auc[c]['roc']:.4f}  PR-AUC={auc[c]['pr']:.4f}  (top20F1={matrix[20][c][2]:.4f})")
    print("\n  ROC-AUC ranking: " + " > ".join(f"{c}({auc[c]['roc']:.4f})" for c in sorted(ORDER, key=lambda c: -auc[c]['roc'])))
    print("  PR-AUC  ranking: " + " > ".join(f"{c}({auc[c]['pr']:.4f})" for c in sorted(ORDER, key=lambda c: -auc[c]['pr'])))

    # 4. e2e threshold mechanism (V6-W2 only)
    print("\n=== 4. e2e ranking 반전 — extractor threshold(0.1) 통과 분석 (V6-W2 4 cells) ===")
    print(f"  {'cell':12s} {'pass@0.1':>9} {'gold_pass':>10} {'nong_pass':>10} {'goldR@0.1':>10} | {'ext_nodes':>9} {'e2e_R':>7} {'e2e_EX':>7} {'sel_top20F1':>11}")
    thr = {}
    for c in ["sum", "standalone", "phase1", "no_selfloop"]:
        t = thresh_stats(data[c], 0.1); thr[c] = t
        print(f"  {c:12s} {t['pass_mean']:>9.2f} {t['gold_pass_mean']:>10.2f} {t['nong_pass_mean']:>10.2f} "
              f"{t['gold_recall_thr']:>10.4f} | {E2E_EXT_NODES[c]:>9.2f} {E2E_R[c]:>7.4f} {E2E_EX[c]:>7.4f} {matrix[20][c][2]:>11.4f}")

    # write artifacts
    with open(os.path.join(OUTDIR, "v6w2_v6w0_topk_matrix_2026-06-05.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["K"] + [f"{c}_F1" for c in ORDER])
        for k in KS:
            w.writerow([k] + [round(matrix[k][c][2], 4) for c in ORDER])
    summary = dict(
        topk={k: {c: dict(R=round(matrix[k][c][0], 4), P=round(matrix[k][c][1], 4), F1=round(matrix[k][c][2], 4)) for c in ORDER} for k in KS},
        dist={c: {kk: round(vv, 4) for kk, vv in dist[c].items()} for c in ORDER},
        auc={c: {kk: round(vv, 4) for kk, vv in auc[c].items()} for c in ORDER},
        e2e_threshold={c: {kk: round(vv, 4) for kk, vv in thr[c].items()} for c in thr},
        e2e_ex=E2E_EX, e2e_r=E2E_R, e2e_ext_nodes=E2E_EXT_NODES,
    )
    with open(os.path.join(OUTDIR, "v6w2_v6w0_comparison_2026-06-05.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote outputs/analysis/v6w2_v6w0_{topk_matrix.csv, comparison.json}_2026-06-05")


if __name__ == "__main__":
    main()
