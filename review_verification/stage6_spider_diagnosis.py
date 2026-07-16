"""
Stage 6 — Spider 2.0-Lite 붕괴 진단. LLM 0 호출. 저장 출력 우선.
seed=42 (실패 샘플링).

데이터 소스 (전부 기존):
  - Extractor 출력(테이블): outputs/.../g_s2_1_spider2/predicted_schema.jsonl (predicted_tables/columns)
  - Filter 출력 + EX: outputs/.../g_s2_2_spider2/predicted_sql.jsonl (n_filtered_tables, predicted_sql)
  - EX per-query: outputs/analysis/g_s2_2_sonnet_ex_score_2026-06-11.json (per_db, n_correct)
  - gold 테이블: data/Spider2/methods/gold-tables/spider2-lite-gold-tables.jsonl (547, 테이블만)
  - gold SQL(부분): data/Spider2/spider2-lite/evaluation_suite/gold/sql/*.sql (local subset 21건 overlap)
제약: θ-통과 후보(V̂) = Spider2 graph/score 캐시 부재 → 재계산 불가 (리포트에 명시).
      컬럼 gold 부재(gold-tables=테이블만) → 테이블 레벨 recall 만.
"""
import os, sys, json, csv, ast, random, re
from pathlib import Path
from difflib import SequenceMatcher
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SP = ROOT / "data/Spider2/spider2-lite"
G1 = ROOT / "outputs/experiments/g_s2_1_spider2/predicted_schema.jsonl"
G2 = ROOT / "outputs/experiments/g_s2_2_spider2/predicted_sql.jsonl"
GOLD_T = ROOT / "data/Spider2/methods/gold-tables/spider2-lite-gold-tables.jsonl"
GOLD_SQL_DIR = SP / "evaluation_suite/gold/sql"
EXJ = ROOT / "outputs/analysis/g_s2_2_sonnet_ex_score_2026-06-11.json"

def norm_tab(t):
    """backend-qualified table (a.b.c 또는 db.schema.table) → 마지막 segment lower."""
    return str(t).split(".")[-1].strip().strip("`\"'[]").lower()

# gold tables (547)
gold_tabs = {}
for l in open(GOLD_T):
    x = json.loads(l)
    v = x["gold_tables"]
    v = ast.literal_eval(v) if isinstance(v, str) else v
    gold_tabs[x["instance_id"]] = set(norm_tab(t) for t in v)

# extractor 출력 (predicted_schema)
ext = {x["instance_id"]: x for x in (json.loads(l) for l in open(G1))}
# filter 출력 + SQL (local EX subset)
flt = {x["instance_id"]: x for x in (json.loads(l) for l in open(G2))}
# 질문 (external_knowledge 유무)
meta = {x["instance_id"]: x for x in (json.loads(l) for l in open(SP / "spider2-lite.jsonl"))}
# EX per-query 근사: g_s2_2 analysis 의 per_db (db별 correct/total) — per-query EX 는 predicted_sql 에 없으니 db-level
exj = json.load(open(EXJ))

def sql_tables(sql):
    """gold SQL 텍스트에서 FROM/JOIN 테이블 추출 (정규식, sqlglot 실패 대비)."""
    tabs = set()
    for m in re.finditer(r"\b(?:FROM|JOIN)\s+([`\"\[]?[\w.\-]+[`\"\]]?)", sql, re.IGNORECASE):
        t = norm_tab(m.group(1))
        if t and not t.startswith("("):
            tabs.add(t)
    return tabs

def join_count(sql):
    return len(re.findall(r"\bJOIN\b", sql, re.IGNORECASE))

# ── 2. Coverage funnel (테이블 레벨, local EX 123 subset) ────────────
local_ids = [i for i in flt if i in gold_tabs]  # local EX subset ∩ gold
def recall_strict(pred_set, gold_set):
    if not gold_set:
        return None, None
    inter = len(pred_set & gold_set)
    return inter / len(gold_set), (1 if inter == len(gold_set) else 0)

funnel = {}
for stage, getter in [
    ("extractor", lambda iid: set(norm_tab(t) for t in ext.get(iid, {}).get("predicted_tables", []))),
    ("filter", lambda iid: sql_tables(flt[iid].get("predicted_sql", "")) if flt.get(iid, {}).get("predicted_sql") else set()),
]:
    rs, ss, ns, n = 0.0, 0, 0, 0
    for iid in local_ids:
        pred = getter(iid)
        r, s = recall_strict(pred, gold_tabs[iid])
        if r is None: continue
        rs += r; ss += s; ns += len(pred); n += 1
    funnel[stage] = {"recall": round(rs/n, 4), "strict": round(ss/n, 4), "avg_nodes": round(ns/n, 2), "n": n}

# 전체 468 (all backends) extractor table recall (참조)
all_ext_ids = [i for i in ext if i in gold_tabs]
rs = ss = 0.0
for iid in all_ext_ids:
    pred = set(norm_tab(t) for t in ext[iid].get("predicted_tables", []))
    r, s = recall_strict(pred, gold_tabs[iid])
    rs += r; ss += s
funnel_all = {"recall": round(rs/len(all_ext_ids), 4), "strict": round(ss/len(all_ext_ids), 4), "n": len(all_ext_ids)}

# ── 3. 실패 자동 사전분류 (seed=42, 100 샘플 또는 전체) ──────────────
# 실패 = local EX subset 에서 extractor 테이블 recall < 1 (gold 이탈) 또는 EX=0
rng = random.Random(42)
# gold SQL overlap (JOIN 분석 가능)
gold_sql_ids = set(f[:-4] for f in os.listdir(GOLD_SQL_DIR) if f.endswith(".sql"))
def load_gold_sql(iid):
    p = GOLD_SQL_DIR / f"{iid}.sql"
    return p.read_text() if p.exists() else None

# 실패 후보: extractor recall < 1 (gold 테이블 이탈)
fail_rows = []
for iid in local_ids:
    ext_pred = set(norm_tab(t) for t in ext.get(iid, {}).get("predicted_tables", []))
    g = gold_tabs[iid]
    r, _ = recall_strict(ext_pred, g)
    missing = g - ext_pred
    if r is not None and r < 0.9999:  # gold 이탈 발생
        # 이탈 단계 판정: extractor 에서 이미 빠졌으면 extractor, 아니면 filter
        gsql = load_gold_sql(iid)
        q = meta.get(iid, {}).get("question", "")
        # (a) 명명/약어: missing gold table 과 question 의 fuzzy match 최대
        def fuzzy(a, b): return SequenceMatcher(None, a, b).ratio()
        name_score = max((max(fuzzy(mt, w.lower()) for w in q.split()) if q.split() else 0.0) for mt in missing) if missing else 1.0
        # (b) 외부지식 부재: Spider2 는 evidence 없음 (external_knowledge = 파일명만) → 항상 True 맥락
        ext_knowledge = meta.get(iid, {}).get("external_knowledge", "")
        # (c) 긴 JOIN: gold SQL join >= 3
        jc = join_count(gsql) if gsql else None
        # 유형 추정
        if name_score < 0.4:
            typ = "(a) 명명/약어 불일치"
        elif jc is not None and jc >= 3:
            typ = "(c) 긴 JOIN 경로"
        elif not q or name_score < 0.55:
            typ = "(b) 외부지식/표면형 무관"
        else:
            typ = "(d) 기타/복합"
        fail_rows.append({
            "instance_id": iid, "db": ext.get(iid, {}).get("db"),
            "gold_tables": sorted(g), "missing_tables": sorted(missing),
            "ext_recall": round(r, 3), "stage_missed": "extractor" if missing else "filter+",
            "gold_sql_join_count": jc, "name_fuzzy_max": round(name_score, 3),
            "has_gold_sql": iid in gold_sql_ids, "est_type": typ,
            "question": q[:120],
        })

# CSV
with open(ROOT / "review_verification/stage6_failure_sample.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["instance_id","db","est_type","stage_missed","ext_recall",
                                      "gold_tables","missing_tables","gold_sql_join_count",
                                      "name_fuzzy_max","has_gold_sql","question"])
    w.writeheader()
    for r in fail_rows: w.writerow(r)

from collections import Counter
type_dist = Counter(r["est_type"] for r in fail_rows)
summary = {
    "n_local_ex_subset": len(local_ids),
    "funnel_local_table_level": funnel,
    "funnel_all_backends_extractor": funnel_all,
    "bird_reference_funnel": {"pre_filter_recall": 0.9964, "filter_recall": 0.9539, "strict": 0.8012, "EX": 0.6030},
    "spider_ex": exj["ex"], "spider_correct": exj["n_correct"], "spider_exec_fail": exj["n_exec_fail"],
    "n_fail_extractor_recall_lt1": len(fail_rows),
    "fail_type_distribution": dict(type_dist),
    "constraints": {
        "theta_candidate_Vhat": "재계산 불가 — Spider2 graph/score 캐시 부재 (data/processed/*spider2* 없음)",
        "column_gold": "부재 — gold-tables 는 테이블만. 컬럼 recall 측정 불가 → 테이블 레벨만",
        "gold_sql_overlap": f"local EX 123건 중 gold SQL 파일 보유 {len(set(local_ids)&gold_sql_ids)}건 (JOIN 분석 한정)",
        "per_query_EX": "predicted_sql 에 per-query EX 없음 — db-level correct/total 만 (g_s2_2 analysis)",
    },
}
json.dump(summary, open(ROOT / "review_verification/stage6_summary.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nsaved: stage6_failure_sample.csv ({len(fail_rows)} rows), stage6_summary.json")
