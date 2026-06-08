#!/usr/bin/env python3
"""R@15 ↔ inference recall disconnect 정량 분해 — training 모니터링 지표 교체 (MA-1) + calibration (MA-2) 게이트.

11 cells (M4 anchor + V6-W2×4 + V6-W3×3 + V6-W5×3) 위:
  proxy = {Val R@15, gold p50(calibration), gold recall@θ=0.1, pass@0.1, gold recall@top20(rank)}
  target = inference recall (extractor-stage, e2e predictions extractor_selected_nodes vs gold)
  → Spearman(각 proxy, inference recall) — 어느 proxy 가 inference recall 최선 예측?
  + MA-2: phase1 gold p50=0.0002 collapse 가 calibration(절대점수) vs ranking 문제인지
    (gold recall@top20-rank vs gold recall@θ=0.1 gap).
"""
import json, os, statistics
from collections import defaultdict

ROOT = "/home/hyeonjin/thesis_refactored"
B = os.path.join(ROOT, "outputs/experiments/s06_gat_bottleneck_fix")
S4 = os.path.join(ROOT, "outputs/experiments/s04_ablation/pipeline")
OUT = os.path.join(ROOT, "outputs/analysis")

# cell: (label, val_r15, selector_score_path, e2e_pred_dir)
CELLS = {
 "M4_anchor":  ("Ensemble (M4)", 0.6061, f"{B}/w0_baseline/v6w0_baseline_s11/score_analysis_v6w0_baseline_s11.jsonl", f"{B}/w0_baseline/v6w0_baseline_s11"),
 "w2_sum":     ("W2 sum", 0.5736, f"{B}/w2_edge_type_split/v6w2_p2_sum/score_analysis_v6w2_p2_sum.jsonl", f"{B}/w2_edge_type_split_e2e/v6w2_p2_sum_e2e"),
 "w2_phase1":  ("W2 phase1(PairNorm)", 0.5736, f"{B}/w2_edge_type_split/v6w2_p2_phase1/score_analysis_v6w2_p2_phase1.jsonl", f"{B}/w2_edge_type_split_e2e/v6w2_p2_phase1_e2e"),
 "w2_standalone": ("W2 standalone", 0.5726, f"{B}/w2_edge_type_split/v6w2_p2_standalone/score_analysis_v6w2_p2_standalone.jsonl", f"{B}/w2_edge_type_split_e2e/v6w2_p2_standalone_e2e"),
 "w2_nosl":    ("W2 no_selfloop", 0.5638, f"{B}/w2_edge_type_split/v6w2_p2_standalone_no_selfloop/score_analysis_v6w2_p2_standalone_no_selfloop.jsonl", f"{B}/w2_edge_type_split_e2e/v6w2_p2_standalone_no_selfloop_e2e"),
 "w3_a":       ("W3 a(VirtualSum)", 0.5672, f"{B}/w3_hub_reduction/v6w3_a_s11/score_analysis_v6w3_a_s11.jsonl", f"{B}/w3_hub_reduction_e2e/v6w3_a_e2e_s11"),
 "w3_b":       ("W3 b(ColPool)", 0.5637, f"{B}/w3_hub_reduction/v6w3_b_s11/score_analysis_v6w3_b_s11.jsonl", f"{B}/w3_hub_reduction_e2e/v6w3_b_e2e_s11"),
 "w3_c":       ("W3 c(HubLocalVN)", 0.5633, f"{B}/w3_hub_reduction/v6w3_c_s11/score_analysis_v6w3_c_s11.jsonl", f"{B}/w3_hub_reduction_e2e/v6w3_c_e2e_s11"),
 "w5_a":       ("W5 a(self-loop)", 0.5732, f"{B}/w5_self_loop_residual/v6w5_a_s11/score_analysis_v6w5_a_s11.jsonl", f"{B}/w5_self_loop_residual_e2e/v6w5_a_e2e_s11"),
 "w5_b":       ("W5 b(residual)", 0.5715, f"{B}/w5_self_loop_residual/v6w5_b_s11/score_analysis_v6w5_b_s11.jsonl", f"{B}/w5_self_loop_residual_e2e/v6w5_b_e2e_s11"),
 "w5_c":       ("W5 c(both)", 0.5723, f"{B}/w5_self_loop_residual/v6w5_c_s11/score_analysis_v6w5_c_s11.jsonl", f"{B}/w5_self_loop_residual_e2e/v6w5_c_e2e_s11"),
}
THETA = 0.1


def is_col(name):
    base = name.split("->")[0]
    return "." in base


def norm_col(c):
    return c.split("->")[0].strip().lower()


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


def spearman(xs, ys):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for k, i in enumerate(o): r[i] = k+1
        return r
    rx, ry = rk(xs), rk(ys); n = len(xs)
    return 1-6*sum((a-b)**2 for a, b in zip(rx, ry))/(n*(n*n-1))


def selector_metrics(path):
    pq = defaultdict(list)  # qid -> [(score, is_gold, is_col)]
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pq[r["query_id"]].append((r["score"], bool(r.get("is_gold", False)), is_col(r["node_name"])))
    gold_all, nong_all = [], []
    grec_thr, grec_top20, pass_cnt, rocs = [], [], [], []
    for items in pq.values():
        cols = [(s, g) for s, g, c in items if c]
        gold = [s for s, g in cols if g]; nong = [s for s, g in cols if not g]
        gold_all += gold; nong_all += nong
        ng = len(gold)
        if ng:
            grec_thr.append(sum(1 for s in gold if s >= THETA)/ng)
            top20 = set(sorted(range(len(cols)), key=lambda i: cols[i][0], reverse=True)[:20])
            grec_top20.append(sum(1 for i in top20 if cols[i][1])/ng)
        pass_cnt.append(sum(1 for s, g, c in items if s >= THETA))
        rc = roc_q([(s, g) for s, g, c in items if c])
        if rc is not None:
            rocs.append(rc)
    m = lambda L: statistics.mean(L) if L else float("nan")
    return dict(gold_p50=pct(gold_all, 50), gold_mu=m(gold_all), nong_mu=m(nong_all),
                gold_recall_thr=m(grec_thr), gold_recall_top20=m(grec_top20),
                pass_at_thr=m(pass_cnt), roc_auc=m(rocs))


def inference_recall(e2e_dir):
    """extractor-stage recall: extractor_selected_nodes vs gold_cols (output file)."""
    pred_f = os.path.join(e2e_dir, "predictions.jsonl")
    out_f = None
    for fn in os.listdir(e2e_dir):
        if fn.startswith("output_") and fn.endswith(".jsonl"):
            out_f = os.path.join(e2e_dir, fn); break
    gold_by_q = {}
    if out_f:
        with open(out_f) as f:
            for line in f:
                o = json.loads(line)
                gold_by_q[o["question_id"]] = set(g.strip().lower() for g in o.get("gold_cols", []) if g.strip())
    recs = []
    with open(pred_f) as f:
        for line in f:
            r = json.loads(line); qid = r["question_id"]
            gold = gold_by_q.get(qid)
            if not gold:
                continue
            sel = r.get("extractor_info", {}).get("extractor_selected_nodes", {})
            pred = set()
            if isinstance(sel, dict):
                for cols in sel.values():
                    for c in cols:
                        pred.add(norm_col(c))
            inter = len(pred & gold)
            recs.append(inter/len(gold))
    return statistics.mean(recs) if recs else float("nan")


def main():
    rows = []
    print(f"{'cell':18s} {'ValR@15':>8} {'gold_p50':>8} {'gRec@.1':>8} {'gRec@top20':>10} {'pass@.1':>8} {'ROC':>7} {'infR(ext)':>9}")
    for k, (lab, vr, sp, ed) in CELLS.items():
        sm = selector_metrics(sp); ir = inference_recall(ed)
        row = dict(cell=k, label=lab, val_r15=vr, inf_recall=ir, **sm)
        rows.append(row)
        print(f"{k:18s} {vr:>8.4f} {sm['gold_p50']:>8.4f} {sm['gold_recall_thr']:>8.4f} {sm['gold_recall_top20']:>10.4f} "
              f"{sm['pass_at_thr']:>8.2f} {sm['roc_auc']:>7.4f} {ir:>9.4f}")

    # Spearman(proxy, inference recall)
    ir = [r["inf_recall"] for r in rows]
    print("\n=== Spearman(proxy, inference recall) — 11 cells ===")
    for key, lab in [("val_r15", "Val R@15 (현 monitor)"), ("gold_p50", "gold p50 (calibration)"),
                     ("gold_recall_thr", "gold recall@θ=0.1"), ("gold_recall_top20", "gold recall@top20(rank)"),
                     ("pass_at_thr", "pass@0.1 count"), ("roc_auc", "ROC-AUC")]:
        rho = spearman([r[key] for r in rows], ir)
        print(f"  {lab:28s} ρ = {rho:+.4f}")

    # MA-2 calibration: rank vs threshold gap (gold recall@top20 − gold recall@0.1)
    print("\n=== MA-2 calibration 진단 (gold recall@top20[rank] − gold recall@θ=0.1[absolute]) ===")
    print(f"  {'cell':18s} {'gRec@top20':>10} {'gRec@.1':>8} {'gap(calib deficit)':>18} {'gold_p50':>8} {'ROC':>7}")
    for r in rows:
        gap = r["gold_recall_top20"] - r["gold_recall_thr"]
        flag = " ★calib-broken" if (gap > 0.3 and r["gold_p50"] < 0.05) else ""
        print(f"  {r['cell']:18s} {r['gold_recall_top20']:>10.4f} {r['gold_recall_thr']:>8.4f} {gap:>18.4f} {r['gold_p50']:>8.4f} {r['roc_auc']:>7.4f}{flag}")

    out = dict(theta=THETA, cells=rows,
               spearman={key: spearman([r[key] for r in rows], ir) for key in
                         ["val_r15", "gold_p50", "gold_recall_thr", "gold_recall_top20", "pass_at_thr", "roc_auc"]})
    with open(os.path.join(OUT, "selector_monitor_inference_alignment_2026-06-07.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}/selector_monitor_inference_alignment_2026-06-07.json")


if __name__ == "__main__":
    main()
