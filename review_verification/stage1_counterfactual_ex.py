"""
Stage 1 counterfactual — '배선 수정 시 EX': generator 가 filter 출력을 받았다면?

방법 (generator-only 재실행, GPU 불필요; Builder/Selector/Extractor/Filter 재실행 없음):
  1) 확정 실행본 predictions.jsonl 의 filter 출력(pred_tables/pred_cols)을 filter 출력 스키마로 사용.
     - pred_cols 는 batch 기록 과정에서 bare 컬럼명(소문자)로 저장됨 → dev_tables.json 으로
       테이블 귀속 + 원본 casing 복원(한 컬럼명이 여러 pred_table 에 있으면 모두 귀속 — join key).
  2) 실제 LLMSQLGenerator (sonnet, temp=0) 로 그 스키마 + 실제 evidence/question 으로 SQL 생성.
  3) evaluate_ex 로 gold SQL 대비 EX 재채점.
현재 EX 0.6030(generator 가 extractor 출력 76.9노드 수신) 대비, 배선 수정(filter 출력 7.4노드) 시 EX 확정.

실행: PYTHONPATH=src python review_verification/stage1_counterfactual_ex.py --limit 0   (0=전체 1534)
LLM 호출: query 당 generator 1회. 원본 코드/데이터/체크포인트 미수정.
"""
import os, sys, json, argparse, concurrent.futures, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)   # 0 = 전체
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

from utils.config_parser import get_args_and_config
from utils.logger import setup_logger
from modules.generators.sql_generator import LLMSQLGenerator
from utils.executor import evaluate_ex

sys.argv = ["main", "--config", "experiments/sonnet_rebaseline_2026_06_10/m4_canonical_sonnet"]
_, config = get_args_and_config()
setup_logger(log_dir="./logs/", exp_name="stage1_cf")

gp = config["sql_generator"]["params"]
gen = LLMSQLGenerator(llm_model=gp.get("llm_model", "claude-sonnet-4-6"),
                      temperature=gp.get("temperature", 0.0),
                      provider=gp.get("provider", "sonnet"))

# ── 데이터 로드 ───────────────────────────────────────────────────────
DEV = json.load(open(ROOT / "data/raw/BIRD_dev/dev.json", encoding="utf-8"))
DEV_BY_ID = {d.get("question_id", i): d for i, d in enumerate(DEV)}
DEVT = {d["db_id"]: d for d in json.load(open(ROOT / "data/raw/BIRD_dev/dev_tables.json", encoding="utf-8"))}
PRED = [json.loads(l) for l in open(ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10/m4_canonical_sonnet/predictions.jsonl")]
if args.limit and 0 < args.limit < len(PRED):
    stride = len(PRED) / args.limit
    PRED = [PRED[int(i * stride)] for i in range(args.limit)]

def db_schema(db_id):
    """{table_lower: (table_orig, {col_lower: col_orig})}"""
    d = DEVT[db_id]; tabs = d["table_names_original"]
    out = {t.lower(): (t, {}) for t in tabs}
    for ti, col in d["column_names_original"]:
        if ti < 0: continue
        out[tabs[ti].lower()][1][col.lower()] = col
    return out

def reconstruct_filter_subgraph(pred_tables, pred_cols, db_id):
    """bare pred_cols → dev_tables.json 으로 테이블 귀속 + 원본 casing 복원."""
    sch = db_schema(db_id)
    pt_lower = [t.lower() for t in pred_tables]
    sg = {}
    for tl in pt_lower:
        if tl in sch: sg[sch[tl][0]] = []
    for c in pred_cols:
        cl = c.lower().split(".", 1)[1] if "." in c else c.lower()
        for tl in pt_lower:
            if tl in sch and cl in sch[tl][1]:
                t_orig = sch[tl][0]; col_orig = sch[tl][1][cl]
                if col_orig not in sg[t_orig]:
                    sg.setdefault(t_orig, []).append(col_orig)
    return sg

# ── 생성 (threaded) ───────────────────────────────────────────────────
_lock = threading.Lock(); _done = {"n": 0}
def gen_one(r):
    qid = r["question_id"]; db_id = r["db_id"]
    item = DEV_BY_ID.get(qid, {})
    sg = reconstruct_filter_subgraph(r.get("pred_tables", []), r.get("pred_cols", []), db_id)
    n_nodes = sum(len(v) if v else 1 for v in sg.values()) if sg else 0
    try:
        sql = gen.generate(query=item.get("question", ""), subgraph=sg, evidence=item.get("evidence", ""))
        err = 0
    except Exception as e:
        sql = ""; err = 1
    with _lock:
        _done["n"] += 1
        if _done["n"] % 100 == 0: print(f"  생성 {_done['n']}/{len(PRED)}")
    return {"question_id": qid, "db_id": db_id, "filter_nodes": n_nodes, "pred_sql": sql, "err": err}

print(f"[counterfactual] {len(PRED)} queries, generator-only (filter 출력 입력)")
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
    gens = list(ex.map(gen_one, PRED))

# ── EX 재채점 (per-eval timeout) ──────────────────────────────────────
print("[counterfactual] EX 채점...")
rows = []; ex_total = ex_valid = 0
for g in gens:
    qid = g["question_id"]; db_id = g["db_id"]
    item = DEV_BY_ID.get(qid, {})
    gold_sql = item.get("SQL", item.get("query", ""))
    db_path = str(ROOT / f"data/raw/BIRD_dev/dev_databases/{db_id}/{db_id}.sqlite")
    ex = 0
    if g["pred_sql"] and gold_sql and os.path.exists(db_path):
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pex:
                ex = pex.submit(evaluate_ex, pred_sql=g["pred_sql"], gold_sql=gold_sql, db_path=db_path).result(timeout=15.0)
        except Exception:
            ex = 0
        ex_total += ex; ex_valid += 1
    rows.append({**g, "ex": ex})

# ── 저장 + 요약 ───────────────────────────────────────────────────────
out_jsonl = ROOT / "review_verification/stage1_counterfactual_ex.jsonl"
with open(out_jsonl, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

n = len(rows); ex_mean = ex_total / max(ex_valid, 1)
n_err = sum(r["err"] for r in rows)
avg_nodes = sum(r["filter_nodes"] for r in rows) / max(n, 1)
summary = {
    "n_queries": n, "ex_valid": ex_valid, "ex_count": f"{ex_total}/{ex_valid}",
    "EX_counterfactual_filter_output": round(ex_mean, 4),
    "EX_current_extractor_output": 0.6030,
    "delta_EX": round(ex_mean - 0.6030, 4),
    "avg_filter_output_nodes": round(avg_nodes, 2),
    "gen_errors": n_err,
}
with open(ROOT / "review_verification/stage1_counterfactual_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\n=== COUNTERFACTUAL RESULT ===")
for k, v in summary.items(): print(f"  {k}: {v}")
