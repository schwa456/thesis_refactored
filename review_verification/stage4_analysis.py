"""
Stage 4 분석 — Haiku Full vs Ours: McNemar + b′식 분해. LLM 0 (batch 결과 재사용).
입력: stage4_per_query.csv (full_ex, ours_ex) + m4_canonical predictions (filter 출력 strict recall).
출력: stage4_analysis_summary.json (보고서 반영).
"""
import os, sys, json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.evaluator import parse_sql_elements
from scipy.stats import binomtest

# Haiku per-query
rows = list(csv.DictReader(open(ROOT / "review_verification/stage4_per_query.csv")))
for r in rows:
    r["full_ex"] = int(r["full_ex"]); r["ours_ex"] = int(r["ours_ex"]); r["qid"] = int(r["qid"])
n = len(rows)

# filter 출력 strict recall (b′식 분해용)
DEV = json.load(open(ROOT / "data/raw/BIRD_dev/dev.json", encoding="utf-8"))
gold = {}
for i, d in enumerate(DEV):
    qid = d.get("question_id", i)
    gt, gc = parse_sql_elements(d.get("SQL", ""))
    gold[qid] = set(t.lower() for t in gt) | set(c.lower() for c in gc)
CANON = {r["question_id"]: r for r in (json.loads(l) for l in
         open(ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10/m4_canonical_sonnet/predictions.jsonl"))}
def strict_of(qid):
    r = CANON.get(qid, {}); g = gold.get(qid, set())
    pred = set(t.lower() for t in r.get("pred_tables", [])) | set(
        (c.lower().split(".", 1)[1] if "." in c else c.lower()) for c in r.get("pred_cols", []))
    if not g: return None
    return 1 if g.issubset(pred) else 0

def mean(key, sub=None):
    s = [r for r in rows if sub is None or sub(r)]
    return round(sum(r[key] for r in s) / len(s), 4) if s else None

# McNemar Full vs Ours
b = sum(1 for r in rows if r["full_ex"] == 1 and r["ours_ex"] == 0)  # full only
c = sum(1 for r in rows if r["full_ex"] == 0 and r["ours_ex"] == 1)  # ours only
p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
def stars(pv): return "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))

# b′식 분해: strict recall 보존(=1) vs 누락(=0)
for r in rows:
    r["strict"] = strict_of(r["qid"])
incl = [r for r in rows if r["strict"] == 1]
excl = [r for r in rows if r["strict"] == 0]

summary = {
    "model": "claude-haiku-4-5", "n": n, "seed": 13,
    "full_EX": mean("full_ex"), "ours_EX": mean("ours_ex"),
    "delta_ours_minus_full": round(mean("ours_ex") - mean("full_ex"), 4),
    "mcnemar": {"b_full_only": b, "c_ours_only": c, "discordant": b + c, "p": round(p, 4), "sig": stars(p)},
    "decomposition_by_strict_recall": {
        "gold_preserved(strict=1)": {"n": len(incl), "full_EX": mean("full_ex", lambda r: r["strict"] == 1),
                                     "ours_EX": mean("ours_ex", lambda r: r["strict"] == 1),
                                     "delta": round(mean("ours_ex", lambda r: r["strict"]==1) - mean("full_ex", lambda r: r["strict"]==1), 4)},
        "gold_missed(strict=0)": {"n": len(excl), "full_EX": mean("full_ex", lambda r: r["strict"] == 0),
                                  "ours_EX": mean("ours_ex", lambda r: r["strict"] == 0),
                                  "delta": round(mean("ours_ex", lambda r: r["strict"]==0) - mean("full_ex", lambda r: r["strict"]==0), 4)},
    },
    "sonnet_subsample_ref": {},  # 채움 아래
    "difficulty_breakdown": {},
}
# 난이도별
for lv in ["simple", "moderate", "challenging"]:
    sub = lambda r: r["difficulty"] == lv
    ss = [r for r in rows if r["difficulty"] == lv]
    if ss:
        summary["difficulty_breakdown"][lv] = {"n": len(ss), "full_EX": mean("full_ex", sub), "ours_EX": mean("ours_ex", sub)}

# Sonnet 서브샘플 참조 (as-run + b′)
sub_ids = set(r["qid"] for r in rows)
son_asrun = round(sum(CANON[q]["ex"] for q in sub_ids if q in CANON) / len(sub_ids), 4)
bp = {}
for row in csv.DictReader(open(ROOT / "review_verification/stage_bprime_per_query.csv")):
    v = row["ex_bprime_filter"]; bp[int(row["qid"])] = int(v) if v not in ("", "None") else None
son_bp = round(sum(bp[q] for q in sub_ids if bp.get(q) is not None) / len([q for q in sub_ids if bp.get(q) is not None]), 4)
summary["sonnet_subsample_ref"] = {"as_run_EX": son_asrun, "bprime_EX": son_bp,
                                   "note": "서브샘플 300건 Sonnet — 전체 Dev(0.6030/0.6089)와 혼용 금지"}
json.dump(summary, open(ROOT / "review_verification/stage4_analysis_summary.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
