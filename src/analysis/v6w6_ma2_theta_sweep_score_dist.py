#!/usr/bin/env python3
"""V6-W6 / MA-2 selector-side 분석 (non-GLM, e2e EX 보류) — DECISIONS 2026-06-08.

분석 1 (theta sweep, SECONDARY = pure score≥θ diagnostic): θ∈{0.1..0.9} × cell × column 노드 위
  {gold pass, nongold pass, precision, recall, F1}. (PRIMARY = MSTPCSTUnion-after 는 root 위임.)
분석 2 (score 분포): cell 별 gold/nongold {mean, p10~p90, max} + 분리도 {p50 gap, ROC-AUC, Cohen's d, KL} + hist viz.
핵심 질문: ma2_a_p50 (gap 0.86) 위 clean 운영점 (input↓ ∧ recall 유지) θ 존재?
caveat: selector-side/extractor-stage 만, e2e EX 미측정 (GLM 차단). gate=NA.
"""
import json, os, math, statistics
from collections import defaultdict

ROOT = "/home/hyeonjin/thesis_refactored"
B = os.path.join(ROOT, "outputs/experiments/s06_gat_bottleneck_fix")
S4 = os.path.join(ROOT, "outputs/experiments/s04_ablation/pipeline")
OUT = os.path.join(ROOT, "outputs/analysis")
FIGDIR = os.path.join(ROOT, "notebooks/analysis_results/figs")
os.makedirs(FIGDIR, exist_ok=True)
THETAS = [round(0.1*i, 1) for i in range(1, 10)]

# 신규 cells (DECISIONS 2026-06-08) + baselines
CELLS = {
 # ── 신규 V6-W6 / MA-2 ──
 "ma2_a_p50":   ("MA2 gold-margin w1.0 + p50 mon ⭐", f"{B}/w6_ma/ma2_a_p50_s11/score_analysis_ma2_a_p50_s11.jsonl"),
 "v6w6_a_p50":  ("DSN+SL + p50 mon", f"{B}/w6_ma/v6w6_a_p50_s11/score_analysis_v6w6_a_p50_s11.jsonl"),
 "ma2_a":       ("MA2 gold-margin (recall@θ mon)", f"{B}/w6_ma/ma2_a_s11/score_analysis_ma2_a_s11.jsonl"),
 "ma2_b_p50":   ("MA2 per-table norm + p50 mon", f"{B}/w6_ma/ma2_b_p50_s11/score_analysis_ma2_b_p50_s11.jsonl"),
 "ma2_b":       ("MA2 per-table norm (recall@θ mon)", f"{B}/w6_ma/ma2_b_s11/score_analysis_ma2_b_s11.jsonl"),
 "v6w6_a":      ("DSN+SL (recall@θ mon)", f"{B}/w6_directed_sn_selfloop/v6w6_a_s11/score_analysis_v6w6_a_s11.jsonl"),
 # ── baselines ──
 "M4_anchor":   ("M4 anchor (GAT α=0)", f"{S4}/t00_S1_alpha0/score_analysis_s04_pipeline_t00_S1_alpha0.jsonl"),
 "w5_b":        ("W5 b(residual)", f"{B}/w5_self_loop_residual/v6w5_b_s11/score_analysis_v6w5_b_s11.jsonl"),
 "w3_c":        ("W3 c(HubLocalVN)", f"{B}/w3_hub_reduction/v6w3_c_s11/score_analysis_v6w3_c_s11.jsonl"),
 "w2_sum":      ("W2 sum", f"{B}/w2_edge_type_split/v6w2_p2_sum/score_analysis_v6w2_p2_sum.jsonl"),
 "w2_phase1":   ("W2 phase1(PN)", f"{B}/w2_edge_type_split/v6w2_p2_phase1/score_analysis_v6w2_p2_phase1.jsonl"),
}


def is_col(name):
    return "." in name.split("->")[0]


def load_cols(path):
    """qid -> list of (score, is_gold) for COLUMN nodes."""
    pq = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if is_col(r["node_name"]):
                pq[r["query_id"]].append((r["score"], bool(r.get("is_gold", False))))
    return pq


def pct(vs, p):
    vs = sorted(vs)
    return vs[min(len(vs)-1, int(round((len(vs)-1)*p/100)))] if vs else float("nan")


def roc_q(items):
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


def kl_div(gold, nong, bins=20):
    """KL(gold || nong) over [0,1] histogram (smoothed)."""
    eps = 1e-6
    edges = [i/bins for i in range(bins+1)]
    def hist(v):
        h = [0]*bins
        for x in v:
            b = min(bins-1, int(x*bins))
            h[b] += 1
        s = sum(h)
        return [(c+eps)/(s+eps*bins) for c in h]
    pg, pn = hist(gold), hist(nong)
    return sum(pg[i]*math.log(pg[i]/pn[i]) for i in range(bins))


def main():
    res = {}
    # ── 분석 2: score 분포 + 분리도 ──
    print("=== 분석 2: gold/nongold score 분포 + 분리도 (column 노드) ===")
    print(f"{'cell':14s} {'g_mean':>7} {'g_p50':>7} {'g_p90':>7} {'n_mean':>7} {'n_p50':>7} | "
          f"{'p50gap':>7} {'ROC':>7} {'Cohen_d':>8} {'KL':>7}")
    allg = {}
    for k, (lab, p) in CELLS.items():
        pq = load_cols(p)
        gold = [s for items in pq.values() for s, g in items if g]
        nong = [s for items in pq.values() for s, g in items if not g]
        allg[k] = (gold, nong)
        rocs = [roc_q(it) for it in pq.values()]; rocs = [x for x in rocs if x is not None]
        roc = statistics.mean(rocs)
        mg, mn = statistics.mean(gold), statistics.mean(nong)
        sg, sn = statistics.pstdev(gold), statistics.pstdev(nong)
        pooled = math.sqrt((sg**2 + sn**2)/2) or 1e-9
        cohen = (mg - mn)/pooled
        kl = kl_div(gold, nong)
        res[k] = dict(label=lab, gold_mean=mg, nong_mean=mn,
                      gold_p10=pct(gold, 10), gold_p25=pct(gold, 25), gold_p50=pct(gold, 50),
                      gold_p75=pct(gold, 75), gold_p90=pct(gold, 90), gold_max=max(gold),
                      nong_p50=pct(nong, 50), nong_p90=pct(nong, 90),
                      p50_gap=pct(gold, 50)-pct(nong, 50), roc_auc=roc, cohen_d=cohen, kl=kl)
        r = res[k]
        print(f"{k:14s} {mg:>7.4f} {r['gold_p50']:>7.4f} {r['gold_p90']:>7.4f} {mn:>7.4f} {r['nong_p50']:>7.4f} | "
              f"{r['p50_gap']:>7.4f} {roc:>7.4f} {cohen:>8.4f} {kl:>7.4f}")

    # ── 분석 1 secondary: pure score≥θ sweep ──
    print("\n=== 분석 1 SECONDARY: pure score≥θ (column, precision/recall/F1) ===")
    sweep = {}
    for k, (lab, p) in CELLS.items():
        pq = load_cols(p)
        per_t = {}
        for t in THETAS:
            gp, npass, gtot = [], [], []
            P, R, F = [], [], []
            for items in pq.values():
                g_pass = sum(1 for s, g in items if g and s >= t)
                n_pass = sum(1 for s, g in items if (not g) and s >= t)
                g_tot = sum(1 for s, g in items if g)
                gp.append(g_pass); npass.append(n_pass); gtot.append(g_tot)
                prec = g_pass/(g_pass+n_pass) if (g_pass+n_pass) else 0.0
                rec = g_pass/g_tot if g_tot else 0.0
                P.append(prec); R.append(rec); F.append(2*prec*rec/(prec+rec) if (prec+rec) else 0.0)
            per_t[t] = dict(gold_pass=statistics.mean(gp), nong_pass=statistics.mean(npass),
                            precision=statistics.mean(P), recall=statistics.mean(R), f1=statistics.mean(F))
        sweep[k] = per_t
    # print key cells
    for k in ["ma2_a_p50", "v6w6_a_p50", "M4_anchor", "w3_c", "w2_sum", "ma2_b_p50"]:
        print(f"\n[{k}] gold_p50={res[k]['gold_p50']:.4f} gap={res[k]['p50_gap']:.4f}")
        print(f"  {'θ':>4} {'gold_pass':>9} {'nong_pass':>9} {'prec':>7} {'recall':>7} {'F1':>7}")
        for t in THETAS:
            s = sweep[k][t]
            print(f"  {t:>4.1f} {s['gold_pass']:>9.3f} {s['nong_pass']:>9.3f} {s['precision']:>7.4f} {s['recall']:>7.4f} {s['f1']:>7.4f}")

    # ── viz: gold vs nongold hist (key cells) ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keyc = ["ma2_a_p50", "v6w6_a_p50", "ma2_a", "M4_anchor", "w3_c", "ma2_b_p50"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, k in zip(axes.flat, keyc):
            gold, nong = allg[k]
            ax.hist(nong, bins=40, range=(0, 1), alpha=0.5, label="non-gold", color="tab:red", density=True)
            ax.hist(gold, bins=40, range=(0, 1), alpha=0.5, label="gold", color="tab:blue", density=True)
            ax.set_title(f"{k} (gap={res[k]['p50_gap']:.2f}, d={res[k]['cohen_d']:.2f})", fontsize=10)
            ax.set_xlabel("score"); ax.legend(fontsize=8)
        plt.tight_layout()
        figp = os.path.join(FIGDIR, "v6w6_ma2_score_dist_2026-06-08.png")
        plt.savefig(figp, dpi=110); plt.close()
        print(f"\nWrote fig {figp}")
    except Exception as e:
        print(f"\n[viz skip] {e}")

    with open(os.path.join(OUT, "v6w6_ma2_theta_sweep_score_dist_2026-06-08.json"), "w") as f:
        json.dump(dict(thetas=THETAS, dist=res, sweep_secondary=sweep), f, indent=2)
    print(f"Wrote {OUT}/v6w6_ma2_theta_sweep_score_dist_2026-06-08.json")


if __name__ == "__main__":
    main()
