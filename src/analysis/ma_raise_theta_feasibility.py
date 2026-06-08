#!/usr/bin/env python3
"""MA raise-θ feasibility (analysis-only, 학습 무관) — 현 score 분포 위 시뮬레이션.

목표 (DECISIONS 2026-06-07 #4 §MA): calibration 좋은 cell 에서 extractor θ 올리면
gold recall 유지 + 통과 노드(=Filter input)↓ → iso-perf 에서 input 축소 (효율). EX gain 아님.

(1) column-level 분리: cell별 {gold p50, nongold p50, gold μ, nongold μ, ROC-AUC}
(2) θ-raise 시뮬: θ∈{0.1,0.2,0.3,0.4,0.5} 위 gold-col recall@θ + pass@θ(all-node, Filter input proxy) + nongold-col 절단
(3) feasibility: gold recall 유지하며 pass↓ 운영점 — 특히 gold p50 高 cell (w3_c 0.8373)
"""
import json, os, statistics
from collections import defaultdict

ROOT = "/home/hyeonjin/thesis_refactored"
B = os.path.join(ROOT, "outputs/experiments/s06_gat_bottleneck_fix")
OUT = os.path.join(ROOT, "outputs/analysis")
THETAS = [0.1, 0.2, 0.3, 0.4, 0.5]
M4_POSTFILTER_R = 0.9376  # 참조 target (단 post-extractor/filter — selector-seed 와 직접 비교 caveat)

CELLS = {
 "M4_anchor":     ("Ensemble (M4)", f"{B}/w0_baseline/v6w0_baseline_s11/score_analysis_v6w0_baseline_s11.jsonl"),
 "w2_sum":        ("W2 sum", f"{B}/w2_edge_type_split/v6w2_p2_sum/score_analysis_v6w2_p2_sum.jsonl"),
 "w2_phase1":     ("W2 phase1(PN)", f"{B}/w2_edge_type_split/v6w2_p2_phase1/score_analysis_v6w2_p2_phase1.jsonl"),
 "w2_standalone": ("W2 standalone", f"{B}/w2_edge_type_split/v6w2_p2_standalone/score_analysis_v6w2_p2_standalone.jsonl"),
 "w2_nosl":       ("W2 no_selfloop", f"{B}/w2_edge_type_split/v6w2_p2_standalone_no_selfloop/score_analysis_v6w2_p2_standalone_no_selfloop.jsonl"),
 "w3_a":          ("W3 a(VirtualSum)", f"{B}/w3_hub_reduction/v6w3_a_s11/score_analysis_v6w3_a_s11.jsonl"),
 "w3_b":          ("W3 b(ColPool)", f"{B}/w3_hub_reduction/v6w3_b_s11/score_analysis_v6w3_b_s11.jsonl"),
 "w3_c":          ("W3 c(HubLocalVN)", f"{B}/w3_hub_reduction/v6w3_c_s11/score_analysis_v6w3_c_s11.jsonl"),
 "w5_a":          ("W5 a(self-loop)", f"{B}/w5_self_loop_residual/v6w5_a_s11/score_analysis_v6w5_a_s11.jsonl"),
 "w5_b":          ("W5 b(residual)", f"{B}/w5_self_loop_residual/v6w5_b_s11/score_analysis_v6w5_b_s11.jsonl"),
 "w5_c":          ("W5 c(both)", f"{B}/w5_self_loop_residual/v6w5_c_s11/score_analysis_v6w5_c_s11.jsonl"),
}


def is_col(name):
    return "." in name.split("->")[0]


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


def analyze(path):
    pq = defaultdict(list)  # qid -> [(score, is_gold, is_col)]
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pq[r["query_id"]].append((r["score"], bool(r.get("is_gold", False)), is_col(r["node_name"])))
    gold_c, nong_c = [], []
    rocs = []
    # per-θ accumulators
    grec = {t: [] for t in THETAS}       # gold-col recall@θ (per query)
    passn = {t: [] for t in THETAS}      # all-node pass@θ count (Filter input proxy)
    nong_pass = {t: [] for t in THETAS}  # nongold-col pass@θ count
    for items in pq.values():
        cols = [(s, g) for s, g, c in items if c]
        gold = [s for s, g in cols if g]; nong = [s for s, g in cols if not g]
        gold_c += gold; nong_c += nong
        rc = roc_q(cols)
        if rc is not None:
            rocs.append(rc)
        ng = len(gold)
        for t in THETAS:
            if ng:
                grec[t].append(sum(1 for s in gold if s >= t)/ng)
            passn[t].append(sum(1 for s, g, c in items if s >= t))
            nong_pass[t].append(sum(1 for s in nong if s >= t))
    m = lambda L: statistics.mean(L) if L else float("nan")
    return dict(
        gold_p50=pct(gold_c, 50), nong_p50=pct(nong_c, 50), gold_mu=m(gold_c), nong_mu=m(nong_c),
        roc_auc=m(rocs),
        per_theta={t: dict(gold_recall=m(grec[t]), pass_all=m(passn[t]), nong_pass=m(nong_pass[t])) for t in THETAS},
    )


def main():
    res = {}
    print("=== (1) column-level 분리 ===")
    print(f"{'cell':18s} {'gold_p50':>8} {'nong_p50':>8} {'gold_mu':>8} {'nong_mu':>8} {'gap_mu':>7} {'ROC':>7}")
    for k, (lab, p) in CELLS.items():
        r = analyze(p); res[k] = r
        print(f"{k:18s} {r['gold_p50']:>8.4f} {r['nong_p50']:>8.4f} {r['gold_mu']:>8.4f} {r['nong_mu']:>8.4f} "
              f"{r['gold_mu']-r['nong_mu']:>7.4f} {r['roc_auc']:>7.4f}")

    print("\n=== (2) θ-raise 시뮬 — gold-col recall@θ / pass@θ(all-node, Filter input) ===")
    for k in CELLS:
        r = res[k]; pt = r["per_theta"]
        base_pass = pt[0.1]["pass_all"]; base_grec = pt[0.1]["gold_recall"]
        print(f"\n[{k}] gold_p50={r['gold_p50']:.4f}")
        print(f"  {'θ':>4} {'gold_rec':>9} {'(retain%)':>9} {'pass_all':>9} {'(input%)':>9} {'nong_pass':>9}")
        for t in THETAS:
            gr = pt[t]["gold_recall"]; pa = pt[t]["pass_all"]
            ret = 100*gr/base_grec if base_grec else 0
            inp = 100*pa/base_pass if base_pass else 0
            print(f"  {t:>4.1f} {gr:>9.4f} {ret:>8.1f}% {pa:>9.2f} {inp:>8.1f}% {pt[t]['nong_pass']:>9.2f}")

    # (3) feasibility: θ where gold recall retains ≥95% of θ=0.1 while pass drops most
    print("\n=== (3) feasibility — gold recall ≥95% retain (vs θ=0.1) 조건 위 최대 pass 축소 θ ===")
    print(f"  {'cell':18s} {'gold_p50':>8} {'best θ':>7} {'gold_rec':>9} {'retain%':>8} {'pass↓%':>8} {'clean?':>7}")
    feas = {}
    for k in CELLS:
        r = res[k]; pt = r["per_theta"]
        base_grec = pt[0.1]["gold_recall"]; base_pass = pt[0.1]["pass_all"]
        best = 0.1
        for t in THETAS:
            ret = (pt[t]["gold_recall"]/base_grec) if base_grec else 0
            if ret >= 0.95:
                best = t
        bg = pt[best]["gold_recall"]; bp = pt[best]["pass_all"]
        ret = 100*bg/base_grec if base_grec else 0
        cut = 100*(1 - bp/base_pass) if base_pass else 0
        clean = "✓" if (best >= 0.3 and r["gold_p50"] >= 0.3 and cut >= 20) else ("△" if best >= 0.2 else "✗")
        feas[k] = dict(best_theta=best, gold_recall=bg, retain_pct=ret, pass_cut_pct=cut, clean=clean)
        print(f"  {k:18s} {r['gold_p50']:>8.4f} {best:>7.1f} {bg:>9.4f} {ret:>7.1f}% {cut:>7.1f}% {clean:>7}")

    out = dict(thetas=THETAS, m4_postfilter_r=M4_POSTFILTER_R, cells=res, feasibility=feas)
    with open(os.path.join(OUT, "ma_raise_theta_feasibility_2026-06-07.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}/ma_raise_theta_feasibility_2026-06-07.json")


if __name__ == "__main__":
    main()
