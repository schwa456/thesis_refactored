#!/usr/bin/env python3
"""DECISIONS 2026-06-09 analyzer 3-task:
(1) extractor frontier — STE-topk vs MSTPCSTUnion-θ iso-recall (decision sweep 57+19 cells)
(2) v6w6_a EX 0.4407 분해 — threshold-pass vs genuine gold-separation (V6-W3 cell C 식)
(3) dsn_nosl condition-comparability (값은 외부, 본 script 는 v6w6_a 분해 evidence 제공)
gate=NA (extractor-stage / selector-side, e2e EX 별도).
"""
import json, os, statistics
from collections import defaultdict

ROOT = "/home/hyeonjin/thesis_refactored"
SW = os.path.join(ROOT, "outputs/experiments/extractor_decision_2026_06_08")
B = os.path.join(ROOT, "outputs/experiments/s06_gat_bottleneck_fix")
S4 = os.path.join(ROOT, "outputs/experiments/s04_ablation/pipeline")
OUT = os.path.join(ROOT, "outputs/analysis")


def read_metrics(d):
    r = nodes = None
    mp = os.path.join(d, "metrics.txt")
    if not os.path.exists(mp):
        return None, None
    for line in open(mp):
        if line.startswith("recall:"):
            r = float(line.split()[1])
        if "extractor_selected_nodes_mean" in line:
            nodes = float(line.split()[1])
    return r, nodes


# ── TASK 1: frontier ──
CELLS_SW = ["m4_anchor", "ma2_a_p50", "w2_sum"]
KS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
THS = [round(0.1*i, 1) for i in range(1, 10)]
SW_M4 = os.path.join(ROOT, "outputs/experiments/extractor_decision_2026_06_08_m4")  # m4_anchor 완전 런
front = {}
for c in CELLS_SW:
    src = SW_M4 if c == "m4_anchor" else SW  # main m4_anchor metrics 비어있음 → _m4 dir 사용
    ste = []; mst = []
    for k in KS:
        r, n = read_metrics(os.path.join(src, f"{c}_ste_k{k:03d}"))
        if r is not None:
            ste.append((n, r, k))
    for t in THS:
        r, n = read_metrics(os.path.join(src, f"{c}_mst_t{int(t*100):03d}"))
        if r is not None:
            mst.append((n, r, t))
    front[c] = dict(ste=ste, mst=mst)

print("=== TASK 1: STE-topk vs MSTPCSTUnion-θ frontier (extractor-stage recall @ Filter-input nodes) ===")
for c in CELLS_SW:
    print(f"\n[{c}]  STE-topk:")
    for n, r, k in front[c]["ste"]:
        print(f"   k={k:>3} nodes={n:>6.2f} recall={r:.4f}")
    print(f"  MSTPCSTUnion-θ:")
    for n, r, t in front[c]["mst"]:
        print(f"   θ={t} nodes={n:>6.2f} recall={r:.4f}")

# iso-node 비교 (STE vs MST 가장 가까운 노드 budget)
print("\n=== iso-node 비교 (STE-topk − MST-θ recall, 같은 노드 budget) ===")
for c in CELLS_SW:
    print(f"[{c}]")
    for n_s, r_s, k in front[c]["ste"]:
        # nearest MST by node count
        best = min(front[c]["mst"], key=lambda x: abs(x[0]-n_s))
        n_m, r_m, t = best
        if abs(n_m - n_s) <= 6:
            print(f"  ~{n_s:.0f} nodes: STE k{k} R={r_s:.4f} vs MST θ{t}({n_m:.0f}) R={r_m:.4f}  ΔSTE={r_s-r_m:+.4f}")

# STE-topk: cell 간 비교 (rank lever — w2_sum/M4 high ROC vs ma2_a_p50 calibration)
print("\n=== STE-topk cell 비교 (rank lever 확인, 같은 k) ===")
print(f"  {'k':>4} {'m4_anchor':>10} {'ma2_a_p50':>10} {'w2_sum':>10}")
sd = {c: {k: r for n, r, k in front[c]["ste"]} for c in CELLS_SW}
for k in KS:
    print(f"  {k:>4} {sd['m4_anchor'].get(k,0):>10.4f} {sd['ma2_a_p50'].get(k,0):>10.4f} {sd['w2_sum'].get(k,0):>10.4f}")
print("  (ROC: m4 0.6963 > w2_sum 0.6784 > ma2_a_p50 0.6572)")
print("  (MST θ=0.1 nodes: m4 36.0 / w2_sum 30.9 / ma2_a_p50 47.7 ← calibration 이 nongold 올려 MST 팽창)")


# ── TASK 2: v6w6_a EX 0.4407 분해 ──
def iscol(n):
    return "." in n.split("->")[0]


def load_scores(path):
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
    k = (len(vs)-1)*p/100; f = int(k); cc = min(f+1, len(vs)-1)
    return vs[f]+(vs[cc]-vs[f])*(k-f)


COS = load_scores(os.path.join(S4, "t00_S1_alpha0/score_analysis_s04_pipeline_t00_S1_alpha0.jsonl"))  # placeholder GAT
COSINE = load_scores(os.path.join(S4, "t00_S2_alpha1/score_analysis_s04_pipeline_t00_S2_alpha1.jsonl"))

DECOMP_CELLS = {
 "v6w6_a": (f"{B}/w6_directed_sn_selfloop/v6w6_a_s11/score_analysis_v6w6_a_s11.jsonl",
            f"{B}/w6_directed_sn_selfloop_e2e/v6w6_a_e2e_s11"),
 "ma2_a_p50": (f"{B}/w6_ma/ma2_a_p50_s11/score_analysis_ma2_a_p50_s11.jsonl",
               f"{B}/w6_ma_e2e/ma2_a_p50_e2e_s11"),
}
# ref: V6-W3 cell C (threshold-pass) + w2_sum (genuine)
DECOMP_CELLS["w3_c"] = (f"{B}/w3_hub_reduction/v6w3_c_s11/score_analysis_v6w3_c_s11.jsonl", None)
DECOMP_CELLS["w2_sum"] = (f"{B}/w2_edge_type_split/v6w2_p2_sum/score_analysis_v6w2_p2_sum.jsonl", None)

print("\n\n=== TASK 2: v6w6_a EX 0.4407 분해 (threshold-pass vs genuine gold-separation) ===")
print(f"  {'cell':12s} {'gold_p50':>8} {'nong_p50':>8} {'nong_mu':>8} {'gap':>7} {'pass@0.1':>9} {'col_net':>8} {'ext_nodes':>9} {'e2e_EX':>7} {'e2e_R':>7}")
E2E = {"v6w6_a": (0.4407, 0.8702), "ma2_a_p50": (0.3501, 0.7438)}
for c, (sp, ed) in DECOMP_CELLS.items():
    gat = load_scores(sp)
    gold = [s for nd in gat.values() for n, (s, g) in nd.items() if g and iscol(n)]
    nong = [s for nd in gat.values() for n, (s, g) in nd.items() if (not g) and iscol(n)]
    passk = statistics.mean([sum(1 for s, g in nd.values() if s >= 0.1) for nd in gat.values()])
    # column net vs cosine (P80)
    tr = th = 0
    for q in gat:
        if q not in COSINE:
            continue
        gn = gat[q]; cn = COSINE[q]
        gthr = percentile_interp([s for s, _ in gn.values()], 80)
        cthr = percentile_interp([s for s, _ in cn.values()], 80)
        for n, (s, g) in gn.items():
            if not (g and iscol(n)):
                continue
            gp = s >= gthr; cp = cn.get(n, (0, False))[0] >= cthr
            if gp and not cp:
                tr += 1
            elif cp and not gp:
                th += 1
    col_net = tr - th
    ext_nodes = None
    if ed and os.path.exists(os.path.join(ed, "metrics.txt")):
        _, ext_nodes = read_metrics(ed)
    ex, r = E2E.get(c, (None, None))
    print(f"  {c:12s} {pct(gold,50):>8.4f} {pct(nong,50):>8.4f} {statistics.mean(nong):>8.4f} "
          f"{pct(gold,50)-pct(nong,50):>7.4f} {passk:>9.2f} {col_net:>8d} "
          f"{(ext_nodes if ext_nodes else 0):>9.2f} {(ex if ex else 0):>7.4f} {(r if r else 0):>7.4f}")


if __name__ == "__main__":
    pass
