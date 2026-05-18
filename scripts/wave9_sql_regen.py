"""Wave 9 Baseline Relog — SQL Gen 재실행 + EX 재측정.

DECISIONS 2026-05-18 (Wave 9 Baseline Relog Chain) §2 + §4 정합.

Stand-alone script (main.py 변경 없이). 기존 baseline 의 final_nodes 보존 + 신규 SQL Gen
prompt 정합 (Wave 5+ EX-Friendly Property + LLMSQLGenerator evidence 정합) 으로 EX 재측정.

Usage:
  python scripts/wave9_sql_regen.py --baseline g_retriever | linkalign | xiyansql
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

# .env 로드 (main.py 정합 — GLM_BASE_URL + GLM_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

# src/ 경로 sys.path 추가 (main.py 정합)
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

# project root cd (data/raw/BIRD_dev/... 경로 정합)
os.chdir(Path(__file__).parent.parent)

from modules.generators.sql_generator import LLMSQLGenerator  # noqa: E402
from utils.executor import evaluate_ex  # noqa: E402


def parse_final_nodes(fn):
    """list of "table.col" → {table: [col, col, ...]} (LLMSQLGenerator subgraph 형식)."""
    if isinstance(fn, dict):
        return {k: list(v) if isinstance(v, (list, set)) else list(v.keys()) for k, v in fn.items()}
    subgraph: dict = {}
    for item in fn or []:
        if "." in item:
            tbl, col = item.split(".", 1)
            subgraph.setdefault(tbl, []).append(col)
        else:
            subgraph.setdefault(item, [])
    return subgraph


def compute_ex_with_timeout(pred_sql, gold_sql, db_path, timeout=15.0):
    if not pred_sql or not gold_sql or not os.path.exists(db_path):
        return 0
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
            f = ex.submit(evaluate_ex, pred_sql=pred_sql, gold_sql=gold_sql, db_path=db_path)
            return f.result(timeout=timeout)
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=["g_retriever", "linkalign", "xiyansql"])
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    src_pred = f"outputs/baselines/baseline_{args.baseline}/predictions.jsonl"
    if not os.path.exists(src_pred):
        print(f"ERROR: source not found: {src_pred}")
        sys.exit(1)

    out_dir = args.output_dir or f"outputs/baselines/wave9_relog/{args.baseline}_relog"
    os.makedirs(out_dir, exist_ok=True)

    bird = {r["question_id"]: r for r in json.load(open("data/raw/BIRD_dev/dev.json"))}

    generator = LLMSQLGenerator(llm_model="zai-org/glm-4.7", temperature=0.0, provider="glm")

    pred_out = os.path.join(out_dir, "predictions.jsonl")
    metrics_out = os.path.join(out_dir, "metrics.txt")

    ex_total = 0
    by_diff = {"simple": [0, 0], "moderate": [0, 0], "challenging": [0, 0]}
    n = 0
    t_start = time.time()

    with open(src_pred) as fin, open(pred_out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            qid = r["question_id"]
            bird_r = bird.get(qid, {})
            evidence = bird_r.get("evidence", "")
            gold_sql = bird_r.get("SQL", "")
            difficulty = bird_r.get("difficulty", "unknown")
            db_id = r.get("db_id") or bird_r.get("db_id")
            db_path = f"data/raw/BIRD_dev/dev_databases/{db_id}/{db_id}.sqlite"

            subgraph = parse_final_nodes(r.get("final_nodes", []))

            t0 = time.time()
            try:
                new_sql = generator.generate(query=r["question"], subgraph=subgraph, evidence=evidence) if subgraph else ""
            except Exception as e:
                new_sql = ""
                print(f"  [SQL ERR qid={qid}] {type(e).__name__}: {e}", flush=True)
            sql_time = time.time() - t0

            ex_score = compute_ex_with_timeout(new_sql, gold_sql, db_path, timeout=15.0)

            ex_total += ex_score
            if difficulty in by_diff:
                by_diff[difficulty][0] += ex_score
                by_diff[difficulty][1] += 1

            new_rec = {
                **r,
                "generated_sql": new_sql,
                "ex_score": ex_score,
                "difficulty": difficulty,
                "sql_gen_time": sql_time,
                # 기존 generated_sql 도 보존 (prior_generated_sql)
                "prior_generated_sql": r.get("generated_sql", ""),
            }
            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1

            if n % 100 == 0:
                elapsed = time.time() - t_start
                rate = n / elapsed * 60.0
                eta_min = (1534 - n) / max(rate / 60.0, 0.01) / 60.0
                print(f"[{args.baseline}][{n}/1534 ({100*n/1534:.1f}%)] rate={rate:.1f} q/min, ex_running={ex_total/n:.4f}, eta={eta_min:.0f}min", flush=True)

    overall_ex = ex_total / max(n, 1)
    with open(metrics_out, "w") as f:
        f.write(f"ex: {overall_ex:.4f}\n")
        f.write(f"n: {n}\n")
        f.write(f"baseline: {args.baseline}\n")
        for diff in ("simple", "moderate", "challenging"):
            s, ns = by_diff[diff]
            if ns:
                f.write(f"ex_{diff}: {s/ns:.4f} ({s}/{ns})\n")
        f.write(f"wall_s: {time.time() - t_start:.1f}\n")

    print(f"\n=== Wave 9 {args.baseline} relog DONE ===", flush=True)
    print(f"overall EX: {overall_ex:.4f} (n={n})", flush=True)
    for diff in ("simple", "moderate", "challenging"):
        s, ns = by_diff[diff]
        if ns:
            print(f"  {diff}: {s/ns:.4f} ({s}/{ns})", flush=True)
    print(f"wall: {(time.time() - t_start)/60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
