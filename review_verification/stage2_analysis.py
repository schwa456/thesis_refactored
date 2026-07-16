"""
Stage 2 분석 — MST vs PCST 레짐. LLM 0 호출, 순수 계산.
입력(기존): stage2_extractor_outputs.jsonl (V_MST/V_union/pcst_added/gold, 로컬 재계산)
           + MST-only predictions.jsonl (ex_mst) + m4_canonical predictions.jsonl (ex_union)
           + dev.json (difficulty, gold_table_count)
출력: stage2_subset_D.csv + stage2_analysis_summary.json (보고서에 반영)
"""
import os, sys, json, csv
from pathlib import Path
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scipy.stats import binomtest

BASE = ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10"
ext = {r["question_id"]: r for r in (json.loads(l) for l in open(ROOT / "review_verification/stage2_extractor_outputs.jsonl"))}
ex_union = {r["question_id"]: r["ex"] for r in (json.loads(l) for l in open(BASE / "m4_canonical_sonnet/predictions.jsonl"))}
ex_mst = {r["question_id"]: r["ex"] for r in (json.loads(l) for l in open(BASE / "module_ablation_b/extractor/ext_mst_only_sonnet/predictions.jsonl"))}

rows = []
for qid, e in ext.items():
    if qid not in ex_union or qid not in ex_mst:
        continue
    rows.append({
        "qid": qid, "db_id": e.get("db_id"), "difficulty": e.get("difficulty"),
        "gold_table_count": e.get("gold_table_count"),
        "n_mst": e["n_mst"], "n_union": e["n_union"], "n_pcst_added": e["n_pcst_added"],
        "pcst_added_gold_count": e["pcst_added_gold_count"], "in_D": e["in_D"],
        "ex_mst": ex_mst[qid], "ex_union": ex_union[qid],
    })
n = len(rows)
D = [r for r in rows if r["in_D"]]
Dc = [r for r in rows if not r["in_D"]]

def ex_mean(sub, key):
    return sum(r[key] for r in sub) / len(sub) if sub else None
def mcnemar(sub):
    b = sum(1 for r in sub if r["ex_mst"] == 1 and r["ex_union"] == 0)  # MST만 정답
    c = sum(1 for r in sub if r["ex_mst"] == 0 and r["ex_union"] == 1)  # union만 정답
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return b, c, b + c, round(p, 4)

# 전체/D/Dc
summary = {"n": n, "n_D": len(D), "D_ratio": round(len(D)/n, 4),
           "n_Dc": len(Dc),
           "EX_mst_all": round(ex_mean(rows, "ex_mst"), 4), "EX_union_all": round(ex_mean(rows, "ex_union"), 4),
           "EX_mst_D": round(ex_mean(D, "ex_mst"), 4) if D else None,
           "EX_union_D": round(ex_mean(D, "ex_union"), 4) if D else None,
           "EX_mst_Dc": round(ex_mean(Dc, "ex_mst"), 4) if Dc else None,
           "EX_union_Dc": round(ex_mean(Dc, "ex_union"), 4) if Dc else None,
           "avg_n_mst": round(ex_mean(rows, "n_mst"), 2), "avg_n_union": round(ex_mean(rows, "n_union"), 2)}
# McNemar
b, c, disc, p = mcnemar(D); summary.update(D_mcn_b=b, D_mcn_c=c, D_discordant=disc, D_mcn_p=p, D_deltaEX=round((ex_mean(D,"ex_union")-ex_mean(D,"ex_mst")),4) if D else None)
bc_, cc_, discc_, pc_ = mcnemar(Dc); summary.update(Dc_mcn_b=bc_, Dc_mcn_c=cc_, Dc_discordant=discc_, Dc_mcn_p=pc_)
summary["Dc_discordant_rate(nondeterminism_floor)"] = round(discc_/max(len(Dc),1), 4)
summary["D_discordant_rate"] = round(disc/max(len(D),1), 4)

# PCST-added 노드 수 분포 (D 내)
hist = Counter(min(r["n_pcst_added"], 10) for r in D)   # 10+ clamp
summary["pcst_added_hist(D, node수:질의수, 10=10+)"] = dict(sorted(hist.items()))
# D 에서 PCST-added 가 gold 포함하는 질의 비율
D_gold = sum(1 for r in D if r["pcst_added_gold_count"] > 0)
summary["D_pcst_added_has_gold"] = D_gold
summary["D_pcst_added_has_gold_ratio"] = round(D_gold/max(len(D),1), 4)

# 층화: gold table 수 (1/2/3+)
def tbin(t): return "1" if t == 1 else ("2" if t == 2 else "3+")
strat_t = {}
for key in ["1", "2", "3+"]:
    sub = [r for r in rows if tbin(r["gold_table_count"]) == key]
    subD = [r for r in sub if r["in_D"]]
    strat_t[key] = {"n": len(sub), "D_ratio": round(len(subD)/max(len(sub),1),4),
                    "EX_mst_D": round(ex_mean(subD,"ex_mst"),4) if subD else None,
                    "EX_union_D": round(ex_mean(subD,"ex_union"),4) if subD else None,
                    "deltaEX_D": round((ex_mean(subD,"ex_union")-ex_mean(subD,"ex_mst")),4) if subD else None,
                    "D_added_gold_ratio": round(sum(1 for r in subD if r["pcst_added_gold_count"]>0)/max(len(subD),1),4)}
summary["strat_gold_tables"] = strat_t
# 층화: 난이도
strat_d = {}
for key in ["simple", "moderate", "challenging"]:
    sub = [r for r in rows if str(r["difficulty"]).lower() == key]
    subD = [r for r in sub if r["in_D"]]
    strat_d[key] = {"n": len(sub), "D_ratio": round(len(subD)/max(len(sub),1),4),
                    "EX_mst_D": round(ex_mean(subD,"ex_mst"),4) if subD else None,
                    "EX_union_D": round(ex_mean(subD,"ex_union"),4) if subD else None,
                    "deltaEX_D": round((ex_mean(subD,"ex_union")-ex_mean(subD,"ex_mst")),4) if subD else None}
summary["strat_difficulty"] = strat_d

# CSV
with open(ROOT / "review_verification/stage2_subset_D.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["qid","db_id","difficulty","gold_table_count","n_mst","n_union","n_pcst_added","pcst_added_gold_count","in_D","ex_mst","ex_union"])
    w.writeheader()
    for r in rows: w.writerow(r)
json.dump(summary, open(ROOT / "review_verification/stage2_analysis_summary.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
