"""GLM 4.7 baseline SQL gen + EX evaluation — 3 scenario (Full / Gold Table / Gold Column).

Based on notebooks/{direct_generation, gold_schema_table_test, gold_schema_column_test}.ipynb
(Llama 3.1 8B baseline). LLM backbone 만 GLM 4.7 (Elice ML API, OpenAI-compatible) 로 교체.

근거: planning/DECISIONS.md 2026-05-12 (B1'+B2'+B3' GLM 4.7 Baseline 즉시 launch)
  - B1' Full Schema (Maamari 2024 paradigm)
  - B2' Gold Table oracle
  - B3' Gold Column oracle (perfect schema linking, EX absolute upper bound)
  - B4' (Filter 모듈 확정 후 보류) — paper main F1 의 EX transfer

Output: outputs/analysis/glm_baseline/{b1_full, b2_gold_table, b3_gold_column}/predictions.jsonl + metrics.txt
"""

from __future__ import annotations

import os
import sys
import json
import sqlite3
import threading
import time
import argparse
from pathlib import Path
from typing import Set, Dict, Any, List, Optional

import sqlglot
from sqlglot.expressions import Table, Column
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 기존 XiYanFilter / SGBE 가 사용하는 동일 LLM client (OpenAI-compatible).
# provider="glm" → GLM_BASE_URL + GLM_API_KEY env 사용.
from llm_client.api_handler import APIClient
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_ROOT = ROOT / "outputs/analysis/glm_baseline"
DEFAULT_DATA_DIR = ROOT / "data/raw/BIRD_dev"

# Scenario → output 디렉토리명 매핑 (DECISIONS 2026-05-12 명세)
SCENARIO_DIRS = {
    "full":        "b1_full",
    "gold_table":  "b2_gold_table",
    "gold_column": "b3_gold_column",
}


# ──────────────────────────────────────────────────────────────
# Gold element 추출 (기존 notebook 정합)
# ──────────────────────────────────────────────────────────────

def extract_gold_tables(sql: str) -> Set[str]:
    """Gold SQL → tables (lowercase set). 파싱 실패 시 빈 set."""
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        return set(node.name.lower() for node in parsed.find_all(Table) if node.name)
    except Exception:
        return set()


def extract_gold_columns(sql: str) -> Set[str]:
    """Gold SQL → columns (lowercase set). 파싱 실패 시 빈 set."""
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        return set(node.name.lower() for node in parsed.find_all(Column) if node.name)
    except Exception:
        return set()


# ──────────────────────────────────────────────────────────────
# Schema formatting (기존 notebook 의 build_schema_dot_format 통합 버전)
# ──────────────────────────────────────────────────────────────

def build_schema_dot_format(db_schema_map: Dict[str, Any], db_id: str,
                              used_tables: Optional[Set[str]] = None,
                              used_columns: Optional[Set[str]] = None) -> str:
    """Schema → "table.column" 줄단위 format. used_tables / used_columns 로 필터링.

    - used_tables=None + used_columns=None  → Full schema (B1')
    - used_tables=set, used_columns=None    → Gold Table 필터 (B2')
    - used_tables=None, used_columns=set    → Gold Column 필터 (B3')

    column_names 구조: [[table_idx, column_name], ...] (BIRD dev_tables.json).
    table_idx=-1 (전체 '*' marker) 는 skip.
    """
    schema_info = db_schema_map.get(db_id)
    if not schema_info:
        return ""
    table_names = schema_info['table_names_original']
    column_names = schema_info['column_names_original']

    schema_lines: List[str] = []
    for table_idx, col_name in column_names:
        if table_idx < 0:
            continue
        t_name = table_names[table_idx]
        if used_tables is not None and t_name.lower() not in used_tables:
            continue
        if used_columns is not None and col_name.lower() not in used_columns:
            continue
        schema_lines.append(f"{t_name}.{col_name}")
    return "\n".join(schema_lines)


# ──────────────────────────────────────────────────────────────
# Prompt (기존 notebook 정합 — System/User 분리)
# ──────────────────────────────────────────────────────────────

def create_messages(schema_str: str, question: str, evidence: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an expert SQL developer. Your task is to write a SQLite query based on "
        "the given schema and external knowledge. "
        "IMPORTANT: If a column name contains spaces or special characters, you MUST wrap "
        "it in backticks (e.g., `Column Name` or `Percent (%)`). "
        "Output ONLY the SQL query. Do not include markdown formatting like ```sql or "
        "any explanations."
    )
    user_prompt = (
        f"### Schema (table.column):\n{schema_str}\n\n"
        f"### External Knowledge:\n{evidence}\n\n"
        f"### Question:\n{question}\n\n### SQL:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ──────────────────────────────────────────────────────────────
# GLM 4.7 chat completion (OpenAI-compatible, messages-aware)
# ──────────────────────────────────────────────────────────────

def chat_completion_glm(api_client: APIClient, model: str, messages: List[Dict[str, str]],
                         max_tokens: int = 256, temperature: float = 0.0,
                         timeout: float = 120.0, max_retries: int = 3,
                         retry_delay: float = 4.0) -> str:
    """APIClient.client (OpenAI SDK) 의 chat.completions.create 를 messages-aware 로 호출.
    APIClient.generate_text 는 prompt 단일 string + 고정 system prompt 라 본 use case 부적합 →
    동일 client 의 raw chat API 직접 호출.

    네트워크 일시 오류 시 exponential backoff 로 재시도. 최종 실패 시 'Error: <reason>' 반환
    (기존 notebook 의 generate_text fallback 패턴 정합).
    """
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            response = api_client.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            # token usage 기록 (APIClient 의 class-level aggregator 재사용)
            APIClient._record_usage(model, getattr(response, "usage", None))
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = str(e)
            if attempt + 1 < max_retries:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            logger.warning(f"GLM API call failed after {max_retries} retries: {last_err}")
            return f"-- Error: GLM API failure ({last_err})"
    return f"-- Error: GLM API failure ({last_err})"


# ──────────────────────────────────────────────────────────────
# SQL execution + EX evaluation (기존 notebook 정합)
# ──────────────────────────────────────────────────────────────

def execute_sql_on_db(db_path: str, sql_query: str, timeout_sec: float = 3.0):
    """Thread-based hard timeout SQL 실행 (Cross Join 방지). 결과는 set(fetchall()) 또는 error str."""
    result = [None]
    exception = [None]

    def target():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            result[0] = set(cursor.fetchall())
            conn.close()
        except Exception as e:
            exception[0] = str(e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_sec)

    if thread.is_alive():
        return "Error: Query Timeout (Possible Cross Join)"
    if exception[0]:
        return exception[0]
    return result[0]


def eval_ex(gold_result, pred_result) -> bool:
    """EX (Execution Accuracy) 비교 — gold result 가 정상 set 일 때만 True 가능."""
    return (gold_result == pred_result) and not isinstance(gold_result, str)


# ──────────────────────────────────────────────────────────────
# Scenario runner
# ──────────────────────────────────────────────────────────────

def run_scenario(scenario: str, dev_data: List[Dict[str, Any]],
                  db_schema_map: Dict[str, Any], db_dir: str,
                  api_client: APIClient, model: str, output_dir: Path,
                  max_queries: Optional[int] = None,
                  resume: bool = False) -> Dict[str, Any]:
    """3 scenario 중 하나 실행 — scenario ∈ {"full", "gold_table", "gold_column"}.

    resume=True 면 predictions.jsonl 에 이미 기록된 qid 는 skip (rate limit 중단 후 재개용).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.txt"

    # Resume — 기존 predictions.jsonl 의 qid 수집
    already_done: Set[int] = set()
    if resume and predictions_path.exists():
        try:
            with open(predictions_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    already_done.add(rec["qid"])
            logger.info(f"  resume: {len(already_done)} predictions already written, skipping")
        except Exception as e:
            logger.warning(f"  resume failed (parse): {e}, starting fresh")
            already_done = set()

    correct = 0
    total = 0
    error_pred = 0
    error_gold = 0

    target_data = dev_data[:max_queries] if max_queries else dev_data

    # Re-tally already-done from existing file (for resume metrics continuity)
    if already_done and predictions_path.exists():
        with open(predictions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                total += 1
                if rec.get("is_correct"):
                    correct += 1
                if rec.get("pred_sql", "").startswith("-- Error:"):
                    error_pred += 1
                if isinstance(rec.get("gold_result_kind"), str) and rec["gold_result_kind"] == "error":
                    error_gold += 1

    file_mode = "a" if (resume and already_done) else "w"

    t_start = time.time()
    with open(predictions_path, file_mode) as f_pred:
        for qid, item in enumerate(tqdm(target_data, desc=f"GLM 4.7 {scenario}")):
            if qid in already_done:
                continue
            db_id = item['db_id']
            question = item['question']
            evidence = item.get('evidence', '')
            gold_sql = item['SQL']
            db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

            # Scenario 별 schema_str 결정
            if scenario == "full":
                schema_str = build_schema_dot_format(db_schema_map, db_id)
            elif scenario == "gold_table":
                gold_tables = extract_gold_tables(gold_sql)
                schema_str = build_schema_dot_format(db_schema_map, db_id, used_tables=gold_tables)
            elif scenario == "gold_column":
                gold_columns = extract_gold_columns(gold_sql)
                schema_str = build_schema_dot_format(db_schema_map, db_id, used_columns=gold_columns)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            # GLM 4.7 SQL gen
            messages = create_messages(schema_str, question, evidence)
            pred_sql = chat_completion_glm(api_client, model, messages)

            # EX evaluation
            gold_result = execute_sql_on_db(db_path, gold_sql)
            pred_result = execute_sql_on_db(db_path, pred_sql)
            is_correct = eval_ex(gold_result, pred_result)

            if is_correct:
                correct += 1
            total += 1
            if isinstance(pred_result, str):
                error_pred += 1
            if isinstance(gold_result, str):
                error_gold += 1

            f_pred.write(json.dumps({
                "qid": qid,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "schema_size": len(schema_str.split("\n")) if schema_str else 0,
                "is_correct": bool(is_correct),
                "pred_result_kind": "error" if isinstance(pred_result, str) else "rows",
                "gold_result_kind": "error" if isinstance(gold_result, str) else "rows",
            }, ensure_ascii=False) + "\n")
            f_pred.flush()

    wall = time.time() - t_start
    ex = (correct / total * 100) if total > 0 else 0.0

    with open(metrics_path, "w") as f_metrics:
        f_metrics.write(f"Scenario: {scenario}\n")
        f_metrics.write(f"Backbone: GLM 4.7 ({model})\n")
        f_metrics.write(f"Total: {total}\n")
        f_metrics.write(f"Correct: {correct}\n")
        f_metrics.write(f"EX (Execution Accuracy): {ex:.4f}%\n")
        f_metrics.write(f"Pred Error: {error_pred}\n")
        f_metrics.write(f"Gold Error: {error_gold}\n")
        f_metrics.write(f"Wall: {wall:.1f}s\n")
        usage = APIClient.get_usage_summary()
        f_metrics.write(f"Token Usage: {json.dumps(usage)}\n")
    logger.info(f"  {scenario}: EX={ex:.4f}% ({correct}/{total}), wall={wall:.1f}s")

    return {
        "scenario": scenario,
        "total": total,
        "correct": correct,
        "ex": ex,
        "pred_error": error_pred,
        "gold_error": error_gold,
        "wall_sec": wall,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["full", "gold_table", "gold_column", "all"],
                        default="all", help="실행할 scenario")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT),
                        help="output 디렉토리 root")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                        help="BIRD_dev directory (containing dev.json / dev_tables.json / dev_databases/)")
    parser.add_argument("--model", default="zai-org/glm-4.7",
                        help="GLM model name")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="처음 N개 query 만 사용 (smoke test 용)")
    parser.add_argument("--resume", action="store_true",
                        help="기존 predictions.jsonl 의 qid 는 skip")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    data_dir = Path(args.data_dir)

    # 데이터 로드
    with open(data_dir / "dev.json") as f:
        dev_data = json.load(f)
    with open(data_dir / "dev_tables.json") as f:
        tables_data = json.load(f)
    db_schema_map = {item['db_id']: item for item in tables_data}
    db_dir = str(data_dir / "dev_databases")

    logger.info(f"Loaded dev: {len(dev_data)} queries, {len(db_schema_map)} DBs")
    if args.max_queries:
        logger.info(f"max_queries={args.max_queries} (smoke test mode)")

    # GLM 4.7 client (Elice ML API, OpenAI-compatible)
    # provider="glm" → GLM_BASE_URL + GLM_API_KEY env 사용 (XiYanFilter / SGBE 와 동일 backbone)
    api_client = APIClient(provider="glm")

    scenarios = ["full", "gold_table", "gold_column"] if args.scenario == "all" else [args.scenario]
    results = []
    for scenario in scenarios:
        output_dir = output_root / SCENARIO_DIRS[scenario]
        result = run_scenario(scenario, dev_data, db_schema_map, db_dir,
                                api_client, args.model, output_dir,
                                max_queries=args.max_queries, resume=args.resume)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print(f"GLM 4.7 Baseline EX ({args.model})")
    print("=" * 60)
    print(f"{'Scenario':<15s} {'EX':>10s} {'Correct/Total':>20s} {'Wall(s)':>10s}")
    for r in results:
        print(f"{r['scenario']:<15s} {r['ex']:>9.4f}% {r['correct']}/{r['total']:>10}  "
              f"{r['wall_sec']:>10.1f}")

    # Cross-scenario summary file
    summary_path = output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "results": results,
            "token_usage": APIClient.get_usage_summary(),
        }, f, indent=2)
    logger.info(f"summary written: {summary_path}")


if __name__ == "__main__":
    main()
