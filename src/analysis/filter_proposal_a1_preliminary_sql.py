"""Filter Proposal Phase 1 — A-1: Preliminary SQL Quality.

근거:
  - planning/DECISIONS.md 2026-05-13 (학술 Agent Phase 1 GO Response + Filter Proposal Data Spec 정정)
  - planning/filter_proposal_data_spec_2026-05-13.md §2 A-1
  - planning/filter_proposal_scholar_agent_response_phase1_2026-05-13.md (Decision Rules + truncated 권고)

목적:
  BIRD-Dev 1534 query × 2 prompts (full schema + S_fwd) 의 preliminary SQL 측정.
  GLM 4.7 backbone, evidence forward (Filter Sweep v2 fix 정합).

  Direction A (RSL-SQL Backward) 의 prerequisite — preliminary SQL 품질 + truncation 빈도.

🆕 학술 Agent 5/13 권고 보강:
  - GLM 4.7 token limit (128K) 초과 시 prompt truncation 감지
  - `is_executable_full = None`, `exec_match_full = None`, `truncated_full = True` 의 nullable 처리

Output: outputs/analysis/filter_proposal/A1_preliminary_sql_quality.jsonl (1534 records)
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_client.api_handler import APIClient
from utils.logger import get_logger

# 기존 glm_baseline_sql_eval 의 helpers 재사용 (B1' / B4' script 정합)
from analysis.glm_baseline_sql_eval import (
    build_schema_dot_format,
    create_messages,
    chat_completion_glm,
    execute_sql_on_db,
    eval_ex,
)

logger = get_logger(__name__)

# Anchor 의 XiYan filtered final_nodes (S_fwd) 출처.
# Filter Sweep v2 C0 (evidence-aware) 권장 — 직전 anchor base (sql_gen 비활성) 와 final_nodes 동등 하지만 evidence-aware 측정 정합.
DEFAULT_ANCHOR_PRED = ROOT / "outputs/experiments/s04_ablation/pipeline/filter_sweep/c0_xiyan_glm_sql/predictions.jsonl"
FALLBACK_ANCHOR_PRED = ROOT / "outputs/experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_kruskal_glm/predictions.jsonl"

DEFAULT_OUTPUT_DIR = ROOT / "outputs/analysis/filter_proposal"
DEFAULT_DATA_DIR = ROOT / "data/raw/BIRD_dev"

# GLM 4.7 token limit (128K context window).
# 학술 Agent 5/13 권고 — truncation 감지로 null 처리.
GLM_4_7_TOKEN_LIMIT = 128000


# ──────────────────────────────────────────────────────────────
# Token counting
# ──────────────────────────────────────────────────────────────

def _get_tokenizer():
    """tiktoken cl100k_base — GLM 4.7 의 정확한 tokenizer 가 아니지만 ±10% 근사.
    학술 Agent 의 truncation 감지 목적상 충분 (정확한 limit 보다 진단 빈도가 중요)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(f"tiktoken unavailable ({e}), falling back to char/3 heuristic")
        return None


def count_tokens(messages: List[Dict[str, str]], tokenizer=None) -> int:
    """messages 의 token 수 합 (system + user)."""
    if tokenizer is None:
        # Heuristic: ~3 chars per token (English mid-density)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 3
    n = 0
    for m in messages:
        c = m.get("content", "") or ""
        n += len(tokenizer.encode(c))
    return n


# ──────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────

def load_anchor_final_nodes(predictions_path: Path) -> Dict[int, List[str]]:
    """anchor 의 predictions.jsonl → {question_id: final_nodes (S_fwd)}.
    final_nodes 는 'table.column' 또는 'table' (table-level) 형식.
    """
    out: Dict[int, List[str]] = {}
    with open(predictions_path) as f:
        for line in f:
            d = json.loads(line)
            qid = d.get("question_id", d.get("qid"))
            out[qid] = d.get("final_nodes", []) or []
    logger.info(f"loaded {len(out)} anchor predictions from {predictions_path.name}")
    return out


def build_schema_from_nodes(final_nodes: List[str]) -> str:
    """anchor 의 final_nodes (table.col 또는 table) → 'table.column' 줄단위 schema 문자열.
    Filter Sweep v2 C0 의 LLMSQLGenerator 와 정합 (table-only entry 도 보존, table 단독은 keep).
    """
    lines = []
    for n in final_nodes:
        if not isinstance(n, str):
            continue
        # 'table.column' 형식 그대로, 'table' 단독은 그대로
        lines.append(n)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────

def _process_one_query(qid: int, item: Dict[str, Any],
                       db_schema_map: Dict[str, Any], db_dir: str,
                       anchor_final_nodes: Dict[int, List[str]],
                       api_client: APIClient, model: str, tokenizer) -> Dict[str, Any]:
    """단일 query 처리 — GLM API call 2회 (full + S_fwd) + sqlite execution.
    Thread-safe: api_client (requests-based), execute_sql_on_db (per-call sqlite connection).
    """
    db_id = item["db_id"]
    question = item["question"]
    evidence = item.get("evidence", "")
    gold_sql = item["SQL"]
    db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

    full_schema_str = build_schema_dot_format(db_schema_map, db_id)
    full_messages = create_messages(full_schema_str, question, evidence)
    full_token_count = count_tokens(full_messages, tokenizer)

    final_nodes = anchor_final_nodes.get(qid, [])
    fwd_schema_str = build_schema_from_nodes(final_nodes)
    fwd_messages = create_messages(fwd_schema_str, question, evidence)
    fwd_token_count = count_tokens(fwd_messages, tokenizer)

    truncated_full = full_token_count > GLM_4_7_TOKEN_LIMIT
    gold_result = execute_sql_on_db(db_path, gold_sql)

    if truncated_full:
        prelim_sql_full = None
        is_executable_full = None
        exec_match_full = None
        logger.warning(f"qid={qid} db={db_id} prompt truncated ({full_token_count} tokens > {GLM_4_7_TOKEN_LIMIT})")
    else:
        prelim_sql_full = chat_completion_glm(api_client, model, full_messages)
        pred_result_full = execute_sql_on_db(db_path, prelim_sql_full)
        is_executable_full = not isinstance(pred_result_full, str)
        exec_match_full = eval_ex(gold_result, pred_result_full)

    prelim_sql_fwd = chat_completion_glm(api_client, model, fwd_messages)
    pred_result_fwd = execute_sql_on_db(db_path, prelim_sql_fwd)
    is_executable_fwd = not isinstance(pred_result_fwd, str)
    exec_match_fwd = eval_ex(gold_result, pred_result_fwd)

    return {
        "query_id": qid,
        "db_id": db_id,
        "question": question,
        "evidence": evidence,
        "gold_sql": gold_sql,
        "prelim_sql_full": prelim_sql_full,
        "prelim_sql_fwd": prelim_sql_fwd,
        "is_executable_full": is_executable_full,
        "is_executable_fwd": is_executable_fwd,
        "exec_match_full": exec_match_full,
        "exec_match_fwd": exec_match_fwd,
        "schema_size_full": len(full_schema_str.split("\n")) if full_schema_str else 0,
        "schema_size_fwd": len(final_nodes),
        "truncated_full": truncated_full,
        "prompt_tokens_full": full_token_count,
        "prompt_tokens_fwd": fwd_token_count,
        "S_fwd_from_anchor": final_nodes,
    }


def run_a1(dev_data: List[Dict[str, Any]], db_schema_map: Dict[str, Any], db_dir: str,
            anchor_final_nodes: Dict[int, List[str]],
            api_client: APIClient, model: str,
            output_path: Path,
            max_queries: Optional[int] = None,
            resume: bool = False,
            workers: int = 1) -> Dict[str, Any]:
    """A-1: 1534 query × 2 prompts (full + S_fwd) preliminary SQL.

    workers > 1 시 ThreadPoolExecutor 로 GLM API call concurrent 처리.
    Main thread 만 f_out.write — 결과 순서는 as_completed (qid order 보장 X, query_id 필드로 식별).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = _get_tokenizer()

    # Resume support — 기존 jsonl 의 qid 는 skip
    already_done: set = set()
    if resume and output_path.exists():
        try:
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    already_done.add(rec["query_id"])
            logger.info(f"resume: {len(already_done)} records already written, skipping")
        except Exception as e:
            logger.warning(f"resume failed (parse): {e}, starting fresh")
            already_done = set()

    target = dev_data[:max_queries] if max_queries else dev_data

    n_truncated = 0
    n_exec_full = 0
    n_exec_fwd = 0
    n_match_full = 0
    n_match_fwd = 0
    n_total = 0

    # Re-tally existing records (resume continuity)
    if already_done and output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("truncated_full"):
                    n_truncated += 1
                if rec.get("is_executable_full"):
                    n_exec_full += 1
                if rec.get("is_executable_fwd"):
                    n_exec_fwd += 1
                if rec.get("exec_match_full"):
                    n_match_full += 1
                if rec.get("exec_match_fwd"):
                    n_match_fwd += 1
                n_total += 1

    file_mode = "a" if (resume and already_done) else "w"
    t_start = time.time()

    # Pending queries (qid, item) — already_done 제외
    pending = [(qid, item) for qid, item in enumerate(target) if qid not in already_done]

    def _tally_and_write(rec: Dict[str, Any], f_out):
        nonlocal n_total, n_truncated, n_exec_full, n_exec_fwd, n_match_full, n_match_fwd
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f_out.flush()
        n_total += 1
        if rec.get("truncated_full"):
            n_truncated += 1
        if rec.get("is_executable_full"):
            n_exec_full += 1
        if rec.get("is_executable_fwd"):
            n_exec_fwd += 1
        if rec.get("exec_match_full"):
            n_match_full += 1
        if rec.get("exec_match_fwd"):
            n_match_fwd += 1

    with open(output_path, file_mode) as f_out:
        if workers <= 1:
            # Sequential (backward compat)
            for qid, item in tqdm(pending, desc="A-1 preliminary SQL"):
                rec = _process_one_query(qid, item, db_schema_map, db_dir,
                                         anchor_final_nodes, api_client, model, tokenizer)
                _tally_and_write(rec, f_out)
        else:
            # Parallel — ThreadPoolExecutor (GLM API call concurrent)
            logger.info(f"A-1 parallel mode: workers={workers}, pending={len(pending)} queries")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_qid = {
                    executor.submit(_process_one_query, qid, item, db_schema_map, db_dir,
                                    anchor_final_nodes, api_client, model, tokenizer): qid
                    for qid, item in pending
                }
                with tqdm(total=len(pending), desc=f"A-1 preliminary SQL (workers={workers})") as pbar:
                    for future in as_completed(future_to_qid):
                        qid = future_to_qid[future]
                        try:
                            rec = future.result()
                            _tally_and_write(rec, f_out)
                        except Exception as e:
                            logger.error(f"qid={qid} failed: {e}")
                        pbar.update(1)

    wall = time.time() - t_start
    summary = {
        "n_total": n_total,
        "n_truncated_full": n_truncated,
        "truncation_rate": (n_truncated / n_total) if n_total else 0.0,
        "is_executable_full_rate": (n_exec_full / max(1, n_total - n_truncated)),  # truncated 제외
        "is_executable_fwd_rate": (n_exec_fwd / n_total) if n_total else 0.0,
        "exec_match_full_rate": (n_match_full / max(1, n_total - n_truncated)),
        "exec_match_fwd_rate": (n_match_fwd / n_total) if n_total else 0.0,
        "wall_sec": wall,
        "token_usage": APIClient.get_usage_summary(),
        "model": model,
    }
    logger.info(f"A-1 summary: {json.dumps(summary, indent=2)}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_pred", default=str(DEFAULT_ANCHOR_PRED),
                        help="Anchor predictions.jsonl (S_fwd 출처)")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--model", default="zai-org/glm-4.7")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="처음 N개 query 만 (smoke test 용)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="ThreadPoolExecutor max_workers (1=sequential, >1=concurrent GLM API call)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    # 데이터 로드
    with open(data_dir / "dev.json") as f:
        dev_data = json.load(f)
    with open(data_dir / "dev_tables.json") as f:
        tables_data = json.load(f)
    db_schema_map = {item["db_id"]: item for item in tables_data}
    db_dir = str(data_dir / "dev_databases")

    # Anchor S_fwd 로드 (fallback 체크)
    anchor_path = Path(args.anchor_pred)
    if not anchor_path.exists():
        logger.warning(f"primary anchor not found: {anchor_path}, trying fallback")
        anchor_path = FALLBACK_ANCHOR_PRED
    anchor_final_nodes = load_anchor_final_nodes(anchor_path)

    # Verify anchor coverage
    missing = [i for i in range(min(len(dev_data), args.max_queries or len(dev_data))) if i not in anchor_final_nodes]
    if missing:
        logger.warning(f"{len(missing)} dev qids not in anchor (treating as empty S_fwd)")

    # GLM 4.7 client
    api_client = APIClient(provider="glm")

    output_path = output_dir / "A1_preliminary_sql_quality.jsonl"
    summary = run_a1(
        dev_data=dev_data,
        db_schema_map=db_schema_map,
        db_dir=db_dir,
        anchor_final_nodes=anchor_final_nodes,
        api_client=api_client,
        model=args.model,
        output_path=output_path,
        max_queries=args.max_queries,
        resume=args.resume,
        workers=args.workers,
    )

    # Summary dump
    summary_path = output_dir / "A1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"A-1 summary written: {summary_path}")

    print("\n" + "=" * 60)
    print("A-1 Preliminary SQL Quality")
    print("=" * 60)
    print(f"Total: {summary['n_total']}")
    print(f"Truncated (full schema): {summary['n_truncated_full']} ({summary['truncation_rate']*100:.2f}%)")
    print(f"is_executable_full: {summary['is_executable_full_rate']*100:.2f}% (truncated 제외)")
    print(f"is_executable_fwd: {summary['is_executable_fwd_rate']*100:.2f}%")
    print(f"exec_match_full (= B1' Full EX): {summary['exec_match_full_rate']*100:.2f}%")
    print(f"exec_match_fwd (≈ anchor EX): {summary['exec_match_fwd_rate']*100:.2f}%")
    print(f"Wall: {summary['wall_sec']:.1f}s")


if __name__ == "__main__":
    main()
