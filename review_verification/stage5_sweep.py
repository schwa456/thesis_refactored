"""
Stage 5 [2/2] — θ·edge cost sweep (recall-only, LLM 0). 캐시(stage5_graph_cache.pt) 순수 재계산.

각 (θ, cost 조합)에서 MSTPCSTUnion(= MSTKruskal(θ) ∪ PCST(θ, cost)) 을 재계산하여
pre-Filter recall / strict recall / avg nodes 산출. b′/stage3 정의(element-level: 테이블∪컬럼).
grid:
  θ ∈ {0.05, 0.1(기준), 0.15, 0.2, 0.3}
  cost 전체 스케일 ∈ {0.5, 1(기준), 2}  (belongs_to/fk/macro/base 전부 ×scale)
  table_to_table 만 개별 ∈ {0.5, 1, 2}   (θ=0.1, 전체 scale=1 고정, macro 만 변화)
LLM 0, seed 불필요(결정론적). pcst_fast 결정론적.

실행: PYTHONPATH=src python review_verification/stage5_sweep.py
출력: stage5_sweep_results.csv + stage5_sweep_summary.json
"""
import os, sys, json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from modules.extractors.mst_kruskal import MSTKruskalExtractor
from modules.extractors.pcst import PCSTExtractor

CACHE = torch.load(ROOT / "review_verification/stage5_graph_cache.pt", weights_only=False)
QIDS = sorted(CACHE.keys())

# 기준 cost (논문 확정): base=1.0, belongs_to=0.01, fk=0.05, macro=0.5
BASE = dict(base_cost=1.0, belongs_to_cost=0.01, fk_cost=0.05, macro_cost=0.5)

def node_to_tabcol(name):
    s = str(name)
    if "->" in s:            # fk_node 제외 (R/P/F1 관례)
        return None
    if "." in s:
        t, c = s.split(".", 1)
        return ("col", t.lower(), c.lower())
    return ("tab", s.lower(), None)

def recall_of_selection(sel_idx, rec):
    names = rec["node_names"]
    gt = set(rec["gold_tables"]); gc = set(rec["gold_cols"])
    gold_all = gt | gc
    pred = set()
    for i in sel_idx:
        r = node_to_tabcol(names[int(i)])
        if r is None: continue
        if r[0] == "tab": pred.add(r[1])
        else: pred.add(r[2])  # bare col
    if not gold_all:
        return None, None
    inter = len(gold_all & pred)
    recall = inter / len(gold_all)
    strict = 1 if inter == len(gold_all) else 0
    return recall, strict

def run_cell(theta, costs):
    mst = MSTKruskalExtractor(score_threshold=theta)
    pcst = PCSTExtractor(node_threshold=theta, **costs)
    rec_sum = strict_sum = node_sum = n = 0
    for qid in QIDS:
        c = CACHE[qid]
        gd = {"edges": c["edges"], "edge_types": c["edge_types"]}
        ns = c["node_scores"]
        try:
            m_nodes, _ = mst.extract(gd, ns)
            p_nodes, _ = pcst.extract(gd, ns)
        except Exception:
            continue
        union = set(int(x) for x in m_nodes) | set(int(x) for x in p_nodes)
        r, s = recall_of_selection(union, c)
        if r is None: continue
        rec_sum += r; strict_sum += s; node_sum += len(union); n += 1
    return round(rec_sum/n, 4), round(strict_sum/n, 4), round(node_sum/n, 2), n

rows = []
# Axis 1+2: θ × 전체 cost scale
THETAS = [0.05, 0.1, 0.15, 0.2, 0.3]
SCALES = [0.5, 1.0, 2.0]
for theta in THETAS:
    for scale in SCALES:
        costs = {k: v*scale for k, v in BASE.items()}
        r, s, nn, n = run_cell(theta, costs)
        rows.append({"axis": "theta_x_costscale", "theta": theta, "cost_scale": scale,
                     "t2t_scale": 1.0, "recall": r, "strict_recall": s, "avg_nodes": nn, "n": n})
        print(f"  θ={theta} scale×{scale}: recall={r} strict={s} nodes={nn}")

# Axis 3: table_to_table(macro) 만 개별 scale (θ=0.1, 전체 scale=1)
for t2t in [0.5, 1.0, 2.0]:
    costs = dict(BASE); costs["macro_cost"] = BASE["macro_cost"] * t2t
    r, s, nn, n = run_cell(0.1, costs)
    rows.append({"axis": "t2t_only", "theta": 0.1, "cost_scale": 1.0,
                 "t2t_scale": t2t, "recall": r, "strict_recall": s, "avg_nodes": nn, "n": n})
    print(f"  [t2t only] θ=0.1 macro×{t2t}: recall={r} strict={s} nodes={nn}")

# CSV
with open(ROOT / "review_verification/stage5_sweep_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["axis","theta","cost_scale","t2t_scale","recall","strict_recall","avg_nodes","n"])
    w.writeheader()
    for r in rows: w.writerow(r)

# 요약: 변동폭
recalls = [r["recall"] for r in rows]
stricts = [r["strict_recall"] for r in rows]
base_row = next(r for r in rows if r["axis"]=="theta_x_costscale" and r["theta"]==0.1 and r["cost_scale"]==1.0)
summary = {
    "n_cells": len(rows),
    "baseline_cell(θ=0.1,×1)": {"recall": base_row["recall"], "strict": base_row["strict_recall"],
                                "avg_nodes": base_row["avg_nodes"], "paper_recall": 0.9964,
                                "match": abs(base_row["recall"]-0.9964) < 0.005},
    "recall_min": min(recalls), "recall_max": max(recalls), "recall_range": round(max(recalls)-min(recalls), 4),
    "strict_min": min(stricts), "strict_max": max(stricts), "strict_range": round(max(stricts)-min(stricts), 4),
    "min_recall_cell": min(rows, key=lambda r: r["recall"]),
}
json.dump(summary, open(ROOT / "review_verification/stage5_sweep_summary.json", "w"), indent=2, ensure_ascii=False)
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))
