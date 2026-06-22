#!/usr/bin/env python
"""
Sonnet 4.6 e2e via Message Batches API (DECISIONS 2026-06-10 #5 Phase 5 / STEP 1·2).

설계 — collect-then-replay monkeypatch (모듈 union/parse 로직 그대로 재사용, high-fidelity):
  pipeline.filter.client / pipeline.generator.client 는 별개 APIClient 인스턴스 →
  각 stage generate_text 를 독립 치환.
  Pass1: filter collect (benign 반환) → 모든 filter prompt 수집 → Batch1
  Pass2: filter replay(실결과) + sql collect → 실 filtered schema 기반 sql prompt 수집 → Batch2
  Pass3: filter replay + sql replay → 실 result → 평가/기록
Batch: client.messages.batches.create([Request(custom_id, params=MessageCreateParamsNonStreaming(...))])
  50% 할인, 100k req/256MB, custom_id 매핑, 결과 29일 보존.
사용: python scripts/run_sonnet_batch_e2e.py --config experiments/.../m4_anchor_sonnet --limit 300
"""
import os, sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default="claude-sonnet-4-6")
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--poll", type=int, default=20)
ns = ap.parse_args()

sys.argv = ["main", "--config", ns.config]
from utils.config_parser import get_args_and_config
from utils.logger import setup_logger, get_logger
from utils.executor import evaluate_ex
from utils.evaluator import parse_sql_elements
from pipeline import SchemaLinkingPipeline
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

args, config = get_args_and_config()
setup_logger(log_dir=config['paths']['log_dir'], exp_name=config['experiment_name'])
logger = get_logger("batch_e2e")
exp = config['experiment_name']
out_dir = config['paths']['output_dir']
os.makedirs(out_dir, exist_ok=True)

pipeline = SchemaLinkingPipeline(config)

data_path = config['paths'].get('dev_json', 'data/raw/BIRD_dev/dev.json')
dataset = json.load(open(data_path, encoding='utf-8'))
if ns.limit and 0 < ns.limit < len(dataset):
    stride = len(dataset) / ns.limit
    dataset = [dataset[int(i * stride)] for i in range(ns.limit)]
logger.info(f"[batch_e2e] {len(dataset)} queries (limit={ns.limit or 'full'})")

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

# ── collect/replay infra ───────────────────────────────────────────
results_map = {}          # prompt -> real LLM text
prompt_to_cid, cid_to_prompt, submitted = {}, {}, set()
COLLECT = {"filter": False, "sql": False}
USAGE = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0, "calls": 0}
BENIGN_FILTER, BENIGN_SQL = "{}", "SELECT 1"

def register(prompt):
    if prompt not in prompt_to_cid:
        cid = f"r{len(prompt_to_cid)}"
        prompt_to_cid[prompt] = cid
        cid_to_prompt[cid] = prompt

def gen_factory(stage, benign):
    def gen(prompt, model=None, temperature=None):
        if prompt in results_map:
            return results_map[prompt]
        if COLLECT[stage]:
            register(prompt)
        return benign
    return gen

# filter None (ablation) 가드 — filter 모듈/클라이언트 부재 시 patch skip
if getattr(pipeline, "filter", None) is not None and getattr(pipeline.filter, "client", None) is not None:
    pipeline.filter.client.generate_text = gen_factory("filter", BENIGN_FILTER)
pipeline.generator.client.generate_text = gen_factory("sql", BENIGN_SQL)

def run_pass(tag):
    ok = 0
    for item in dataset:
        try:
            pipeline.run(db_id=item.get("db_id"), query=item.get("question"),
                         evidence=item.get("evidence", ""))
            ok += 1
        except Exception as e:
            logger.warning(f"[{tag}] qid={item.get('question_id')} run err: {e}")
    logger.info(f"[{tag}] pipeline.run ok={ok}/{len(dataset)}")

def submit_batch(tag):
    pending = [cid for cid in cid_to_prompt if cid not in submitted]
    if not pending:
        logger.info(f"[{tag}] no pending requests"); return
    reqs = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
        model=ns.model, max_tokens=ns.max_tokens,
        system=[{"type": "text", "text": "You are a helpful database expert."}],
        messages=[{"role": "user", "content": cid_to_prompt[cid]}],
        temperature=0.0,
    )) for cid in pending]
    logger.info(f"[{tag}] submitting batch: {len(reqs)} requests")
    batch = client.messages.batches.create(requests=reqs)
    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        logger.info(f"[{tag}] {b.processing_status} proc={b.request_counts.processing} "
                    f"done={b.request_counts.succeeded}+{b.request_counts.errored} "
                    f"({int(time.time()-t0)}s)")
        time.sleep(ns.poll)
    n_ok = n_err = 0
    for res in client.messages.batches.results(batch.id):
        prompt = cid_to_prompt[res.custom_id]
        if res.result.type == "succeeded":
            msg = res.result.message
            txt = "".join(bl.text for bl in msg.content if bl.type == "text")
            results_map[prompt] = txt
            u = msg.usage
            USAGE["input_tokens"] += int(getattr(u, "input_tokens", 0) or 0)
            USAGE["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)
            USAGE["cache_read"] += int(getattr(u, "cache_read_input_tokens", 0) or 0)
            USAGE["cache_creation"] += int(getattr(u, "cache_creation_input_tokens", 0) or 0)
            USAGE["calls"] += 1
            n_ok += 1
        else:
            results_map[prompt] = "SELECT 'API ERROR'"
            n_err += 1
        submitted.add(res.custom_id)
    logger.info(f"[{tag}] batch done: ok={n_ok} err={n_err} (wall {int(time.time()-t0)}s)")

# ── 3-pass ──────────────────────────────────────────────────────────
t_start = time.time()
COLLECT["filter"], COLLECT["sql"] = True, False
run_pass("pass1-collect-filter")
submit_batch("batch1-filter")

COLLECT["filter"], COLLECT["sql"] = False, True
run_pass("pass2-replay-filter+collect-sql")
submit_batch("batch2-sql")

COLLECT["filter"], COLLECT["sql"] = False, False
# Pass3: 실 result 평가 + 기록
pred_path = os.path.join(out_dir, "predictions.jsonl")
open(pred_path, "w").close()
R = P = F1 = 0.0
ex_total = ex_valid = 0
api_err = 0
n = 0
for item in dataset:
    qid = item.get("question_id"); db_id = item.get("db_id")
    gold_sql = item.get("SQL", item.get("query", ""))
    db_path = os.path.join("data/raw/BIRD_dev/dev_databases", db_id, f"{db_id}.sqlite")
    try:
        result = pipeline.run(db_id=db_id, query=item.get("question"), evidence=item.get("evidence", ""))
    except Exception as e:
        logger.warning(f"[pass3] qid={qid} err: {e}"); continue
    gold_tables, gold_cols = parse_sql_elements(gold_sql)
    gold_tables = set(t.lower() for t in gold_tables); gold_cols = set(c.lower() for c in gold_cols)
    pred_tables, pred_cols = set(), set()
    for node in result.get("final_nodes", []):
        if '->' in node: continue
        if '.' in node:
            tbl, col = node.split('.', 1); pred_tables.add(tbl.lower())
            tcl = f"{tbl.lower()}.{col.lower()}"
            pred_cols.add(tcl if tcl in gold_cols else col.lower())
        else:
            pred_tables.add(node.lower())
    pred_sql = result.get("generated_sql", "")
    if "API ERROR" in pred_sql: api_err += 1
    # R/P/F1 (table+col 합집합 set 기준 근사)
    gold_all = gold_tables | gold_cols
    pred_all = pred_tables | pred_cols
    if gold_all:
        rec = len(gold_all & pred_all) / len(gold_all)
        prec = len(gold_all & pred_all) / len(pred_all) if pred_all else 0.0
        R += rec; P += prec; F1 += (2*rec*prec/(rec+prec) if rec+prec else 0.0)
    ex = 0
    if pred_sql and gold_sql and os.path.exists(db_path):
        try:
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex_pool:
                ex = ex_pool.submit(evaluate_ex, pred_sql=pred_sql, gold_sql=gold_sql, db_path=db_path).result(timeout=15.0)
        except Exception:
            ex = 0
        ex_total += ex; ex_valid += 1
    n += 1
    with open(pred_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"question_id": qid, "db_id": db_id,
                            "pred_tables": list(pred_tables), "pred_cols": list(pred_cols),
                            "predicted_sql": pred_sql, "ex": ex}, ensure_ascii=False) + "\n")

wall = time.time() - t_start
# ── gate 리포트 ─────────────────────────────────────────────────────
cost = USAGE["input_tokens"]*3/1e6 + USAGE["output_tokens"]*15/1e6 + USAGE["cache_read"]*0.3/1e6 + USAGE["cache_creation"]*3.75/1e6
cost_batch = cost * 0.5  # Batch 50% 할인
report = {
    "experiment": exp, "n_queries": n, "wall_s": round(wall, 1),
    "per_query_s": round(wall/max(n,1), 2),
    "eta_1534_h": round(wall/max(n,1)*1534/3600, 2),
    "usage": USAGE,
    "cache_hit_rate": round(USAGE["cache_read"]/max(USAGE["input_tokens"]+USAGE["cache_read"], 1), 4),
    "cost_usd_batch50pct": round(cost_batch, 4),
    "cost_usd_full": round(cost, 4),
    "api_error_sentinel": api_err,
    "parser_fail_rate": round(api_err/max(n,1), 4),
    "R_mean": round(R/max(n,1), 4), "P_mean": round(P/max(n,1), 4), "F1_mean": round(F1/max(n,1), 4),
    "EX": round(ex_total/max(ex_valid,1), 4), "EX_count": f"{ex_total}/{ex_valid}",
}
with open(os.path.join(out_dir, "batch_gate_report.json"), "w") as f:
    json.dump(report, f, indent=2)
logger.info("=== BATCH GATE REPORT ===")
for k, v in report.items():
    logger.info(f"  {k}: {v}")
print(json.dumps(report, indent=2, ensure_ascii=False))
