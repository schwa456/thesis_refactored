"""
b′ 후속 분석 — LLM 호출 0회. 기존 결과만 재사용 (재실험 금지).

사용 데이터 (모두 기존):
  - b′ EX (generator=filter 출력): review_verification/stage1_counterfactual_ex.jsonl  (per-query ex)
  - as-run EX (generator=extractor 출력): m4_canonical_sonnet/predictions.jsonl        (per-query ex)
  - filter 출력 스키마: m4_canonical_sonnet/predictions.jsonl                            (pred_tables/pred_cols)
  - extractor 출력 스키마: m4_ablation/m4_abl_filter_sonnet/predictions.jsonl (filter=None → final=extractor)
  - gold: data/raw/BIRD_dev/dev.json (gold SQL) → parse_sql_elements
산출: stage_bprime_per_query.csv + (요약은 stdout, 보고서에 반영)
"""
import os, sys, json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.evaluator import parse_sql_elements
from scipy.stats import binomtest

BASE = ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10"
canon = {r["question_id"]: r for r in (json.loads(l) for l in open(BASE / "m4_canonical_sonnet/predictions.jsonl"))}
extr  = {r["question_id"]: r for r in (json.loads(l) for l in open(BASE / "m4_ablation/m4_abl_filter_sonnet/predictions.jsonl"))}
bprime = {r["question_id"]: r for r in (json.loads(l) for l in open(ROOT / "review_verification/stage1_counterfactual_ex.jsonl"))}
DEV = json.load(open(ROOT / "data/raw/BIRD_dev/dev.json", encoding="utf-8"))
gold_by_id = {}
for i, d in enumerate(DEV):
    qid = d.get("question_id", i)
    gs = d.get("SQL", d.get("query", ""))
    gt, gc = parse_sql_elements(gs)
    gold_by_id[qid] = (set(t.lower() for t in gt) | set(c.lower() for c in gc))

def rp(pred_tables, pred_cols, gold_all):
    """batch script 와 동일 recall 정의 (table+col set)."""
    pred_all = set(t.lower() for t in pred_tables) | set(c.lower() for c in pred_cols)
    if not gold_all:
        return None, None
    inter = len(gold_all & pred_all)
    recall = inter / len(gold_all)
    strict = 1 if recall >= 0.99999 else 0
    return recall, strict

rows = []
for qid in canon:
    g = gold_by_id.get(qid, set())
    fr, fstrict = rp(canon[qid].get("pred_tables", []), canon[qid].get("pred_cols", []), g)   # filter 출력
    er, estrict = rp(extr.get(qid, {}).get("pred_tables", []), extr.get(qid, {}).get("pred_cols", []), g) if qid in extr else (None, None)
    rows.append({
        "qid": qid, "db_id": canon[qid]["db_id"],
        "filter_nodes": len(canon[qid].get("pred_tables", [])) + len(canon[qid].get("pred_cols", [])),
        "extractor_nodes": (len(extr[qid].get("pred_tables", [])) + len(extr[qid].get("pred_cols", []))) if qid in extr else None,
        "filter_recall": round(fr, 4) if fr is not None else None,
        "filter_strict_recall": fstrict,
        "extractor_recall": round(er, 4) if er is not None else None,
        "extractor_strict_recall": estrict,
        "ex_asrun_extractor": canon[qid].get("ex"),
        "ex_bprime_filter": bprime.get(qid, {}).get("ex"),
    })

# CSV
with open(ROOT / "review_verification/stage_bprime_per_query.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

def mean(key, sub=None):
    vs = [r[key] for r in rows if r.get(key) is not None and (sub is None or sub(r))]
    return sum(vs) / len(vs) if vs else None

n = len(rows)
# 요약
f_recall = mean("filter_recall"); f_strict = mean("filter_strict_recall")
e_recall = mean("extractor_recall"); e_strict = mean("extractor_strict_recall")
f_nodes = mean("filter_nodes"); e_nodes = mean("extractor_nodes")
ex_asrun = mean("ex_asrun_extractor"); ex_bp = mean("ex_bprime_filter")

# EX_b′ 분해: filter 출력이 gold 전체 포함(strict=1) vs 미포함(strict=0)
def group_ex(strict_val):
    sub = [r for r in rows if r["filter_strict_recall"] == strict_val and r["ex_bprime_filter"] is not None]
    ex = sum(r["ex_bprime_filter"] for r in sub) / len(sub) if sub else None
    return len(sub), ex

n_incl, ex_incl = group_ex(1)
n_excl, ex_excl = group_ex(0)

# McNemar: as-run vs b′
b = sum(1 for r in rows if r["ex_asrun_extractor"] == 1 and r["ex_bprime_filter"] == 0)  # as-run만 정답
c = sum(1 for r in rows if r["ex_asrun_extractor"] == 0 and r["ex_bprime_filter"] == 1)  # b′만 정답
both = sum(1 for r in rows if r["ex_asrun_extractor"] == 1 and r["ex_bprime_filter"] == 1)
neither = sum(1 for r in rows if r["ex_asrun_extractor"] == 0 and r["ex_bprime_filter"] == 0)
mcn_p = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue if (b + c) > 0 else 1.0

print("=== b′ 후속 분석 (재실험 없음, 기존 데이터) ===")
print(f"n={n}")
print(f"[filter 출력]    avg_nodes={f_nodes:.2f}  recall={f_recall:.4f}  strict_recall={f_strict:.4f}  EX(b′)={ex_bp:.4f}")
print(f"[extractor 출력] avg_nodes={e_nodes:.2f}  recall={e_recall:.4f}  strict_recall={e_strict:.4f}  EX(as-run)={ex_asrun:.4f}")
print(f"\nEX_b′ 분해 (filter 출력 gold 포함 여부):")
print(f"  strict_recall=1 (gold 전체 포함): n={n_incl}  EX={ex_incl:.4f}")
print(f"  strict_recall=0 (gold 일부 누락): n={n_excl}  EX={ex_excl:.4f}")
print(f"\nMcNemar (as-run vs b′): both={both} neither={neither} b(as-run only)={b} c(b′ only)={c}")
print(f"  discordant b+c={b+c}, exact p={mcn_p:.4f}")
print(f"  ΔEX = {ex_bp-ex_asrun:+.4f}")
# JSON dump for report
summary = dict(n=n, f_nodes=round(f_nodes,2), f_recall=round(f_recall,4), f_strict=round(f_strict,4),
               e_nodes=round(e_nodes,2), e_recall=round(e_recall,4), e_strict=round(e_strict,4),
               ex_bprime=round(ex_bp,4), ex_asrun=round(ex_asrun,4), delta=round(ex_bp-ex_asrun,4),
               ex_incl=round(ex_incl,4), n_incl=n_incl, ex_excl=round(ex_excl,4), n_excl=n_excl,
               mcn_b=b, mcn_c=c, mcn_both=both, mcn_neither=neither, mcn_p=round(mcn_p,4))
json.dump(summary, open(ROOT / "review_verification/stage_bprime_summary.json", "w"), indent=2)
print("\nsaved: stage_bprime_per_query.csv, stage_bprime_summary.json")
