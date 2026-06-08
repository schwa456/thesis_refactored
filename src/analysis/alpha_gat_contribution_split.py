#!/usr/bin/env python3
"""GAT 순기여 (+214/+2.1%) 를 hub 내부 컬럼 식별 vs cross-table 구조 연결 로 분해.

검증 가설 (v6_intra_table_collapse_origin §5.3): hub 내부는 GAT 가 collapse 시키므로
+2.1% 순기여는 cross-table (table-level 구조) 신호일 것.

데이터: α=0 (GAT, t00_S1_alpha0) + α=1 (cosine, t00_S2_alpha1) score_analysis.
방법: per-query P80 threshold 위 rescued (GAT pass ∧ cosine fail) / hurt (반대) gold column 분해.
split: (A) query gold span single-table vs cross-table, (B) rescued col 이 table 내 sole-gold vs multi-gold.
"""
import json, os, argparse
from collections import defaultdict

ROOT = "/home/hyeonjin/thesis_refactored"
PIPE = os.path.join(ROOT, "outputs/experiments/s04_ablation/pipeline")
CELL_FILE = {
    "alpha0": f"{PIPE}/t00_S1_alpha0/score_analysis_s04_pipeline_t00_S1_alpha0.jsonl",   # pure GAT
    "alpha08": f"{PIPE}/t00_alpha_08/score_analysis_s04_pipeline_t00_alpha_08.jsonl",     # 0.8cos+0.2GAT
    "alpha1": f"{PIPE}/t00_S2_alpha1/score_analysis_s04_pipeline_t00_S2_alpha1.jsonl",   # pure cosine
}
OUT = os.path.join(ROOT, "outputs/analysis")
P_PCT = 80  # AdaptivePCST per-query P80


def col_table(name):
    """column node 'table.col' / FK 'a.b->c.d' → owning table. table node (bare) → None."""
    base = name.split("->")[0]  # FK: 출발 컬럼 기준
    if "." in base:
        return base.split(".", 1)[0]
    return None  # bare = table node (not a column)


def load(path):
    pq = defaultdict(dict)  # qid -> {node_name: (score, is_gold)}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pq[r["query_id"]][r["node_name"]] = (r["score"], bool(r.get("is_gold", False)))
    return pq


def percentile(vals, p):
    vs = sorted(vals)
    if not vs:
        return float("inf")
    k = (len(vs) - 1) * p / 100
    f = int(k); c = min(f + 1, len(vs) - 1)
    return vs[f] + (vs[c] - vs[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gat_cell", default="alpha0", choices=["alpha0", "alpha08"],
                    help="alpha0=pure GAT isolation / alpha08=ensemble regime (0.2 GAT, +214 ref 근사)")
    args = ap.parse_args()
    print(f"### GAT cell = {args.gat_cell} (vs cosine alpha1) ###")
    gat = load(CELL_FILE[args.gat_cell]); cos = load(CELL_FILE["alpha1"])
    qids = sorted(set(gat) & set(cos))

    # net rescue 재현 + split
    rescued = defaultdict(int); hurt = defaultdict(int)   # key: 'single'/'cross' (query span)
    rescued_col = defaultdict(int); hurt_col = defaultdict(int)  # key: 'sole'/'multi' (table gold density)
    tot_rescued = tot_hurt = 0
    # table-level rescue (gold TABLE 식별): table node 기준
    tbl_rescued = tbl_hurt = 0

    for q in qids:
        gnodes = gat[q]; cnodes = cos[q]
        gscores = [s for s, _ in gnodes.values()]
        cscores = [s for s, _ in cnodes.values()]
        gthr = percentile(gscores, P_PCT); cthr = percentile(cscores, P_PCT)
        # gold columns + their tables (query 단위)
        gold_cols = {n for n, (s, g) in gnodes.items() if g and col_table(n) is not None}
        gold_tables_of_col = defaultdict(set)  # table -> set(gold col)
        for n in gold_cols:
            gold_tables_of_col[col_table(n)].add(n)
        n_gold_tables = len(gold_tables_of_col)
        span = "single" if n_gold_tables <= 1 else "cross"

        for n in gold_cols:
            g_pass = gnodes[n][0] >= gthr
            c_pass = cnodes.get(n, (0, False))[0] >= cthr
            tbl = col_table(n)
            density = "sole" if len(gold_tables_of_col[tbl]) == 1 else "multi"
            if g_pass and not c_pass:
                rescued[span] += 1; rescued_col[density] += 1; tot_rescued += 1
            elif c_pass and not g_pass:
                hurt[span] += 1; hurt_col[density] += 1; tot_hurt += 1

        # table-level: gold table nodes (bare name, is_gold)
        gold_tabs = {n for n, (s, g) in gnodes.items() if g and col_table(n) is None}
        for n in gold_tabs:
            gp = gnodes[n][0] >= gthr
            cp = cnodes.get(n, (0, False))[0] >= cthr
            if gp and not cp:
                tbl_rescued += 1
            elif cp and not gp:
                tbl_hurt += 1

    net = tot_rescued - tot_hurt
    print(f"=== GAT 순기여 재현 (P{P_PCT}, gold COLUMN 노드) ===")
    print(f"  rescued={tot_rescued}  hurt={tot_hurt}  net={net:+d}")
    print(f"  (ref: selector_analysis §4 rescued 544 / hurt 330 / +214)")

    print(f"\n=== SPLIT A: query gold span (single-table vs cross-table) ===")
    for k in ("single", "cross"):
        r, h = rescued[k], hurt[k]
        print(f"  {k:7s}: rescued={r:4d} hurt={h:4d} net={r-h:+4d}  ({100*(r-h)/net:+.1f}% of net)" if net else f"  {k}: r={r} h={h}")
    print(f"\n=== SPLIT B: rescued col 의 table gold-density (sole vs multi gold-col) ===")
    for k in ("sole", "multi"):
        r, h = rescued_col[k], hurt_col[k]
        print(f"  {k:5s}: rescued={r:4d} hurt={h:4d} net={r-h:+4d}  ({100*(r-h)/net:+.1f}% of net)" if net else f"  {k}: r={r} h={h}")

    print(f"\n=== table-level GAT 기여 (gold TABLE 노드 식별) ===")
    print(f"  rescued={tbl_rescued} hurt={tbl_hurt} net={tbl_rescued-tbl_hurt:+d}")

    out = dict(p_pct=P_PCT, total=dict(rescued=tot_rescued, hurt=tot_hurt, net=net),
               split_query_span={k: dict(rescued=rescued[k], hurt=hurt[k], net=rescued[k]-hurt[k]) for k in ("single", "cross")},
               split_table_density={k: dict(rescued=rescued_col[k], hurt=hurt_col[k], net=rescued_col[k]-hurt_col[k]) for k in ("sole", "multi")},
               table_level=dict(rescued=tbl_rescued, hurt=tbl_hurt, net=tbl_rescued-tbl_hurt))
    with open(os.path.join(OUT, f"alpha_gat_contribution_split_{args.gat_cell}_2026-06-07.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}/alpha_gat_contribution_split_{args.gat_cell}_2026-06-07.json")


if __name__ == "__main__":
    main()
