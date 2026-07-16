"""
Stage 4 — 약한 생성기(Haiku)에서 Full Schema vs Ours. Batch API, temp=0.
300건(seed=13 서브샘플) × 2조건(Full Schema / Ours=filter출력) = 600 프롬프트.
Filter 재호출 없음(final_nodes 캐시). Builder/Selector/Extractor 재실행 없음(GPU 불요).

프롬프트 조립: Ours = AgentUtils.generate_ddl(filter 출력) + sql_generator.md (b′와 동일);
             Full = AgentUtils.generate_ddl(전체 DB 스키마) + 동일 템플릿 (stage1 검증).
gold SQL = dev.json SQL 필드. EX = utils.executor.evaluate_ex.

실행: PYTHONPATH=src python review_verification/stage4_haiku_batch.py --model claude-haiku-4-5-20251001
출력: stage4_per_query.csv + stage4_haiku_summary.json
"""
import os, sys, json, csv, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="claude-haiku-4-5-20251001")
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--poll", type=int, default=20)
args = ap.parse_args()

from prompts.prompt_manager import PromptManager
from modules.filters.agents import AgentUtils
from utils.executor import evaluate_ex
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# ── 데이터 ────────────────────────────────────────────────────────────
DEV = json.load(open(ROOT / "data/raw/BIRD_dev/dev.json", encoding="utf-8"))
DEV_BY_ID = {d.get("question_id", i): d for i, d in enumerate(DEV)}
DEVT = {d["db_id"]: d for d in json.load(open(ROOT / "data/raw/BIRD_dev/dev_tables.json", encoding="utf-8"))}
SUB = [int(r["question_id"]) for r in csv.DictReader(open(ROOT / "review_verification/stage4_subsample_ids.csv"))]
# filter 출력 (final_nodes) → predictions.jsonl 의 pred_tables/pred_cols
CANON = {r["question_id"]: r for r in (json.loads(l) for l in
         open(ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10/m4_canonical_sonnet/predictions.jsonl"))}

pm = PromptManager()

def db_schema(db_id):
    d = DEVT[db_id]; tabs = d["table_names_original"]
    out = {t.lower(): (t, {}) for t in tabs}
    for ti, col in d["column_names_original"]:
        if ti < 0: continue
        out[tabs[ti].lower()][1][col.lower()] = col
    return out

def full_subgraph(db_id):
    d = DEVT[db_id]; tabs = d["table_names_original"]
    sg = {t: [] for t in tabs}
    for ti, col in d["column_names_original"]:
        if ti < 0: continue
        sg[tabs[ti]].append(col)
    return sg

def ours_subgraph(qid, db_id):
    """filter 출력(pred_tables/pred_cols, bare) → 원본 casing 복원 (b′ 방식)."""
    r = CANON.get(qid, {})
    sch = db_schema(db_id); pt = [t.lower() for t in r.get("pred_tables", [])]
    sg = {}
    for tl in pt:
        if tl in sch: sg[sch[tl][0]] = []
    for c in r.get("pred_cols", []):
        cl = c.lower().split(".", 1)[1] if "." in c else c.lower()
        for tl in pt:
            if tl in sch and cl in sch[tl][1]:
                col_orig = sch[tl][1][cl]
                if col_orig not in sg[sch[tl][0]]: sg.setdefault(sch[tl][0], []).append(col_orig)
    return sg

def build_prompt(subgraph, item):
    ddl = AgentUtils.generate_ddl(subgraph=subgraph)
    return pm.load_prompt(file_name="sql_generator", section="sql_generator",
                          schema_str=ddl, evidence=item.get("evidence", "") or "(none)",
                          query=item.get("question", ""))

# ── 프롬프트 조립 (600) ───────────────────────────────────────────────
tasks = []  # (custom_id, condition, qid, db_id, prompt)
for qid in SUB:
    item = DEV_BY_ID[qid]; db_id = item["db_id"]
    tasks.append((f"full_{qid}", "full", qid, db_id, build_prompt(full_subgraph(db_id), item)))
    tasks.append((f"ours_{qid}", "ours", qid, db_id, build_prompt(ours_subgraph(qid, db_id), item)))
print(f"[stage4] {len(tasks)} prompts (300×2), model={args.model}")

# ── Batch 제출 ────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
reqs = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
            model=args.model, max_tokens=args.max_tokens,
            system=[{"type": "text", "text": "You are a helpful database expert."}],
            messages=[{"role": "user", "content": prompt}], temperature=0.0,
        )) for (cid, _, _, _, prompt) in tasks]
batch = client.messages.batches.create(requests=reqs)
print(f"[stage4] batch submitted: {batch.id} ({len(reqs)} reqs)")
t0 = time.time()
while True:
    b = client.messages.batches.retrieve(batch.id)
    if b.processing_status == "ended": break
    print(f"  {b.processing_status} done={b.request_counts.succeeded}+{b.request_counts.errored} ({int(time.time()-t0)}s)")
    time.sleep(args.poll)
USAGE = {"input": 0, "output": 0}
res_map = {}
for res in client.messages.batches.results(batch.id):
    if res.result.type == "succeeded":
        msg = res.result.message
        res_map[res.custom_id] = "".join(bl.text for bl in msg.content if bl.type == "text")
        USAGE["input"] += int(getattr(msg.usage, "input_tokens", 0) or 0)
        USAGE["output"] += int(getattr(msg.usage, "output_tokens", 0) or 0)
    else:
        res_map[res.custom_id] = "SELECT 'API ERROR'"
print(f"[stage4] batch done ({int(time.time()-t0)}s), usage={USAGE}")

# ── EX 채점 ───────────────────────────────────────────────────────────
def score_ex(sql, gold_sql, db_id):
    db_path = str(ROOT / f"data/raw/BIRD_dev/dev_databases/{db_id}/{db_id}.sqlite")
    if not sql or "API ERROR" in sql or not os.path.exists(db_path): return 0
    sql = sql.replace("```sql", "").replace("```", "").strip()
    try:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
            return ex.submit(evaluate_ex, pred_sql=sql, gold_sql=gold_sql, db_path=db_path).result(timeout=15.0)
    except Exception:
        return 0

rows = []
for (cid, cond, qid, db_id, _) in tasks:
    item = DEV_BY_ID[qid]; gold_sql = item.get("SQL", "")
    ex = score_ex(res_map.get(cid, ""), gold_sql, db_id)
    rows.append({"qid": qid, "db_id": db_id, "difficulty": item.get("difficulty"),
                 "condition": cond, "ex": ex})

# per-query wide: qid → full_ex, ours_ex
wide = {}
for r in rows:
    wide.setdefault(r["qid"], {"qid": r["qid"], "difficulty": r["difficulty"]})[f"{r['condition']}_ex"] = r["ex"]
with open(ROOT / "review_verification/stage4_per_query.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["qid", "difficulty", "full_ex", "ours_ex"])
    w.writeheader()
    for q in sorted(wide): w.writerow(wide[q])

summary = {"model": args.model, "n": len(wide), "seed": 13,
           "full_EX": round(sum(v.get("full_ex", 0) for v in wide.values())/len(wide), 4),
           "ours_EX": round(sum(v.get("ours_ex", 0) for v in wide.values())/len(wide), 4),
           "usage": USAGE}
summary["delta_ours_minus_full"] = round(summary["ours_EX"] - summary["full_EX"], 4)
json.dump(summary, open(ROOT / "review_verification/stage4_haiku_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print("saved: stage4_per_query.csv, stage4_haiku_summary.json")
