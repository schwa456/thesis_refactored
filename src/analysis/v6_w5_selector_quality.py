#!/usr/bin/env python3
"""V6-W5 selector-level 점수 분별력 측정 — mad 회복이 과제 분별로 전환됐는가.

W5 selector-only score_analysis (GAT-only, α=0) 위:
  (1) gold/nongold score 분리 (gold p50 vs nongold μ, table 내 column)
  (2) GAT-only top-20 F1 / ROC-AUC (vs V6-W2 sum 0.3027 / V6-W0 0.2896)
  (3) table/column GAT 기여 분해 (W5-GAT vs cosine α=1) — column net −1181 (multi-gold −1188) 가 양수로 뒤집히는가
의도: V6-W5 mad 회복 (L1 0.0136→0.28~0.35) 이 selector 점수의 gold-정렬 분별력으로 전환됐는지 판정.
"""
import json, os, statistics
from collections import defaultdict

ROOT = "/home/hyeonjin/thesis_refactored"
SO = os.path.join(ROOT, "outputs/experiments/s06_gat_bottleneck_fix/w5_self_loop_residual")
COS_F = os.path.join(ROOT, "outputs/experiments/s04_ablation/pipeline/t00_S2_alpha1/score_analysis_s04_pipeline_t00_S2_alpha1.jsonl")
OUT = os.path.join(ROOT, "outputs/analysis")
CELLS = {"a": "column self-loop", "b": "per-layer residual", "c": "self-loop + residual"}
# baselines
BASE = dict(top20_sum=0.3027, top20_w0=0.2896, top20_phase1=0.2595, roc_sum=0.7204, roc_w0=0.7408,
            m4_col_net=-1181, m4_multi_net=-1188, m4_sole_net=7, m4_table_net=106,
            sum_gold_p50=0.8195, sum_nong_mu=0.2060, phase1_gold_p50=0.0002)


def col_table(name):
    base = name.split("->")[0]
    return base.split(".", 1)[0] if "." in base else None


def load(path):
    pq = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pq[r["query_id"]][r["node_name"]] = (r["score"], bool(r.get("is_gold", False)))
    return pq


def pct(vs, p):
    vs = sorted(vs)
    return vs[min(len(vs)-1, int(round((len(vs)-1)*p/100)))] if vs else float("nan")


def percentile_interp(vals, p):
    vs = sorted(vals)
    if not vs:
        return float("inf")
    k = (len(vs)-1)*p/100; f = int(k); c = min(f+1, len(vs)-1)
    return vs[f] + (vs[c]-vs[f])*(k-f)


def topk_micro(pq, k):
    tp = fp = fn = 0
    for nodes in pq.values():
        items = [(s, g) for s, g in nodes.values()]
        s = sorted(items, key=lambda x: x[0], reverse=True)[:k]
        gt = sum(1 for _, g in items if g); t = sum(1 for _, g in s if g)
        tp += t; fp += len(s)-t; fn += gt-t
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    return R, P, (2*P*R/(P+R) if P+R else 0)


def roc_auc_q(items):
    pos = [s for s, g in items if g]; neg = [s for s, g in items if not g]
    if not pos or not neg:
        return None
    allv = sorted(s for s, _ in items); n = len(allv); rk = {}; i = 0
    while i < n:
        j = i
        while j+1 < n and allv[j+1] == allv[i]:
            j += 1
        rk[allv[i]] = (i+j)/2+1; i = j+1
    return (sum(rk[s] for s in pos)-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))


def spearman(xs, ys):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for k, i in enumerate(o): r[i] = k+1
        return r
    rx, ry = rk(xs), rk(ys); n = len(xs)
    return 1-6*sum((a-b)**2 for a, b in zip(rx, ry))/(n*(n*n-1))


def main():
    cos = load(COS_F)
    results = {}
    print("=== (1) gold/nongold score 분리 + (2) top-20 F1 / ROC-AUC ===")
    print(f"{'cell':10s} {'gold_p50':>9} {'gold_mu':>8} {'nong_mu':>8} {'gap_p50':>8} | {'top20F1':>8} {'ROC-AUC':>8}")
    for c in CELLS:
        gat = load(os.path.join(SO, f"v6w5_{c}_s11", f"score_analysis_v6w5_{c}_s11.jsonl"))
        # (1) gold/nongold (column nodes only)
        gold = [s for nodes in gat.values() for n, (s, g) in nodes.items() if g and col_table(n)]
        nong = [s for nodes in gat.values() for n, (s, g) in nodes.items() if (not g) and col_table(n)]
        gold_p50 = pct(gold, 50); gold_mu = statistics.mean(gold); nong_mu = statistics.mean(nong)
        # (2) top-K + AUC
        R20, P20, F20 = topk_micro(gat, 20)
        rocs = [roc_auc_q([(s, g) for s, g in nd.values()]) for nd in gat.values()]
        rocs = [x for x in rocs if x is not None]
        roc = statistics.mean(rocs)
        # (3) GAT(W5) vs cosine contribution split
        rescued_col = defaultdict(int); hurt_col = defaultdict(int)  # sole/multi
        tot_r = tot_h = 0; tbl_r = tbl_h = 0
        rescued_span = defaultdict(int); hurt_span = defaultdict(int)
        for q in gat:
            if q not in cos:
                continue
            gn = gat[q]; cn = cos[q]
            gthr = percentile_interp([s for s, _ in gn.values()], 80)
            cthr = percentile_interp([s for s, _ in cn.values()], 80)
            gold_cols = {n for n, (s, g) in gn.items() if g and col_table(n)}
            tbl_gold = defaultdict(set)
            for n in gold_cols:
                tbl_gold[col_table(n)].add(n)
            span = "single" if len(tbl_gold) <= 1 else "cross"
            for n in gold_cols:
                gp = gn[n][0] >= gthr; cp = cn.get(n, (0, False))[0] >= cthr
                dens = "sole" if len(tbl_gold[col_table(n)]) == 1 else "multi"
                if gp and not cp:
                    rescued_col[dens] += 1; rescued_span[span] += 1; tot_r += 1
                elif cp and not gp:
                    hurt_col[dens] += 1; hurt_span[span] += 1; tot_h += 1
            gold_tabs = {n for n, (s, g) in gn.items() if g and col_table(n) is None}
            for n in gold_tabs:
                gp = gn[n][0] >= gthr; cp = cn.get(n, (0, False))[0] >= cthr
                if gp and not cp: tbl_r += 1
                elif cp and not gp: tbl_h += 1
        results[c] = dict(label=CELLS[c], gold_p50=gold_p50, gold_mu=gold_mu, nong_mu=nong_mu,
                          gap_p50=gold_p50-pct(nong, 50), top20_R=R20, top20_P=P20, top20_F1=F20, roc_auc=roc,
                          col_net=tot_r-tot_h, col_rescued=tot_r, col_hurt=tot_h,
                          sole_net=rescued_col["sole"]-hurt_col["sole"], multi_net=rescued_col["multi"]-hurt_col["multi"],
                          single_net=rescued_span["single"]-hurt_span["single"], cross_net=rescued_span["cross"]-hurt_span["cross"],
                          table_net=tbl_r-tbl_h, table_rescued=tbl_r, table_hurt=tbl_h)
        r = results[c]
        print(f"v6w5_{c:8s} {gold_p50:>9.4f} {gold_mu:>8.4f} {nong_mu:>8.4f} {r['gap_p50']:>8.4f} | {F20:>8.4f} {roc:>8.4f}")

    print(f"\n  baseline: V6-W2 sum top20F1=0.3027/ROC0.7204 gold_p50=0.8195 | V6-W0 top20F1=0.2896/ROC0.7408 | phase1 gold_p50=0.0002")

    print("\n=== (3) table/column GAT 기여 분해 (W5-GAT vs cosine, P80) ===")
    print(f"  baseline M4-GAT(α=0): column net=−1181 (multi −1188 / sole +7) / table net=+106")
    print(f"  {'cell':10s} {'col_net':>8} {'multi_net':>10} {'sole_net':>9} {'cross_net':>10} {'table_net':>10}")
    for c in CELLS:
        r = results[c]
        print(f"  v6w5_{c:6s} {r['col_net']:>8d} {r['multi_net']:>10d} {r['sole_net']:>9d} {r['cross_net']:>10d} {r['table_net']:>10d}")

    # disconnect: top20F1/gold_p50 vs e2e EX (W5 e2e from prior)
    e2e_ex = {"a": 0.3168, "b": 0.3201, "c": 0.2934}
    l1mad = {"a": 0.2813, "b": 0.3416, "c": 0.3463}
    print("\n=== disconnect 결합 (selector 분별력 ↔ e2e EX ↔ L1 mad) ===")
    cs = list(CELLS)
    f1s = [results[c]["top20_F1"] for c in cs]; exs = [e2e_ex[c] for c in cs]; mads = [l1mad[c] for c in cs]
    colnets = [results[c]["col_net"] for c in cs]
    print(f"  Spearman(top20 F1, e2e EX) = {spearman(f1s, exs):+.4f}")
    print(f"  Spearman(L1 mad, col_net)  = {spearman(mads, colnets):+.4f}  (mad 회복 → column 분별 회복?)")
    print(f"  Spearman(col_net, e2e EX)  = {spearman(colnets, exs):+.4f}")

    out = dict(baseline=BASE, e2e_ex=e2e_ex, l1mad=l1mad, cells=results,
               spearman=dict(top20f1_ex=spearman(f1s, exs), mad_colnet=spearman(mads, colnets), colnet_ex=spearman(colnets, exs)))
    with open(os.path.join(OUT, "v6_w5_selector_quality_2026-06-07.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}/v6_w5_selector_quality_2026-06-07.json")


if __name__ == "__main__":
    main()
