import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import datetime
import traceback
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from typing import Dict, Any

from utils.config_parser import get_args_and_config
from utils.logger import setup_logger, get_logger
from utils.executor import evaluate_ex
from utils.evaluator import parse_sql_elements, calculate_schema_metrics
from pipeline import SchemaLinkingPipeline


def main():
    # 1. Config 로드 + 로거 세팅 (config_parser 가 logs/outputs 디렉토리를 생성).
    args, config = get_args_and_config()

    log_dir = config['paths']['log_dir']
    output_dir = config['paths']['output_dir']
    exp_name = config['experiment_name']

    setup_logger(log_dir=log_dir, exp_name=exp_name)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info(f"🔥 Starting Evaluation for Experiment: [{exp_name}]")
    logger.info("=" * 60)

    # 2. 파이프라인 조립
    pipeline = SchemaLinkingPipeline(config)

    # 3. 평가 데이터셋 로드 (BIRD dev)
    data_path = config['paths'].get('dev_json', 'data/raw/BIRD_dev/dev.json')
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} queries from {data_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 4. 추론 루프
    pred_save_path = os.path.join(output_dir, "predictions.jsonl")
    score_save_path = os.path.join(output_dir, f"score_analysis_{exp_name}.jsonl")
    profiling_path = os.path.join(output_dir, f"profiling_{exp_name}.jsonl")
    output_path = os.path.join(output_dir, f"output_{exp_name}.jsonl")

    # RESUME=1 이면 이미 처리한 question_id 를 건너뛴다.
    resume = os.environ.get("RESUME", "0") == "1"
    processed_ids = set()
    if resume and os.path.exists(pred_save_path):
        with open(pred_save_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line).get("question_id"))
                except Exception:
                    continue
        logger.info(f"[RESUME] {len(processed_ids)} already-processed question_ids found; skipping them")
    else:
        for path in [pred_save_path, score_save_path, profiling_path, output_path]:
            if os.path.exists(path):
                os.remove(path)

    total_ex_score = 0
    valid_ex_count = 0

    for item in tqdm(dataset, desc="Running Pipeline"):
        if resume and item.get("question_id") in processed_ids:
            continue
        db_id = item.get("db_id")
        question = item.get("question")
        question_id = item.get("question_id")
        gold_sql = item.get("SQL", item.get("query", ""))
        db_path = os.path.join("data/raw/BIRD_dev/dev_databases", db_id, f"{db_id}.sqlite")

        try:
            # evidence: BIRD-dev `external_knowledge` 필드 → SQL gen prompt 에 삽입.
            result = pipeline.run(db_id=db_id, query=question, evidence=item.get("evidence", ""))

            gold_tables, gold_cols = parse_sql_elements(gold_sql)
            gold_tables = set(t.lower() for t in gold_tables)
            gold_cols = set(c.lower() for c in gold_cols)

            pred_tables = set()
            pred_cols = set()
            pred_sql = result.get("generated_sql", "")

            for node in result.get("final_nodes", []):
                if '->' in node:
                    continue  # 매크로(FK) 엣지 제외
                if '.' in node:
                    tbl, col = node.split('.', 1)
                    pred_tables.add(tbl.lower())
                    col_lower = col.lower()
                    tbl_col_lower = f"{tbl.lower()}.{col_lower}"
                    pred_cols.add(tbl_col_lower if tbl_col_lower in gold_cols else col_lower)
                else:
                    pred_tables.add(node.lower())

            pred_tables = list(pred_tables)
            pred_cols = list(pred_cols)

            # per-stage 처리시간 기록
            if "execution_time" in result:
                profiling_record = {"query_id": question_id}
                profiling_record.update(result.get("execution_time", {}))
                with open(profiling_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(profiling_record, ensure_ascii=False) + '\n')

            # Execution Accuracy (별도 프로세스 + 15초 타임아웃으로 cartesian/OOM 방어)
            ex_score = 0
            if pred_sql and gold_sql and os.path.exists(db_path):
                try:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(evaluate_ex, pred_sql=pred_sql, gold_sql=gold_sql, db_path=db_path)
                        ex_score = future.result(timeout=15.0)
                except concurrent.futures.TimeoutError:
                    logger.warning("🚨 SQL Execution Timeout (15s). Setting EX = 0")
                    ex_score = 0
                except concurrent.futures.process.BrokenProcessPool:
                    logger.warning("🚨 SQL Execution caused OOM (SIGKILL). Setting EX = 0")
                    ex_score = 0
                except Exception as e:
                    logger.warning(f"🚨 SQL Execution Error: {e}")
                    ex_score = 0
                total_ex_score += ex_score
                valid_ex_count += 1
            elif not os.path.exists(db_path):
                logger.warning(f"DB file not found for EX evaluation: {db_path}")

            # 노드별 score + gold 여부 기록 (분석용)
            node_names = result.get("node_names", [])
            raw_scores = result.get("raw_scores", [])
            with open(score_save_path, 'a', encoding='utf-8') as f:
                for name, score in zip(node_names, raw_scores):
                    name_lower = name.lower()
                    is_gold = False
                    if '.' in name_lower:
                        tbl, col = name_lower.split('.', 1)
                        if tbl in gold_tables and col in gold_cols:
                            is_gold = True
                    elif name_lower in gold_tables:
                        is_gold = True
                    f.write(json.dumps({
                        "query_id": question_id, "node_name": name,
                        "score": float(score), "is_gold": is_gold,
                    }, ensure_ascii=False) + '\n')

            recall, precision, missing_cols, extra_cols = calculate_schema_metrics(set(pred_cols), set(gold_cols))
            _, _, missing_tables, extra_tables = calculate_schema_metrics(set(pred_tables), set(gold_tables))

            pred_record = {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "status": result.get("status"),
                "final_nodes": result.get("final_nodes", []),
                "reasoning": result.get("reasoning", ""),
                "generated_sql": pred_sql,
                "ex_score": ex_score,
            }
            for k in ("builder_info", "selector_info", "extractor_info", "filter_info"):
                if k in result:
                    pred_record[k] = result[k]
            with open(pred_save_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(pred_record, ensure_ascii=False) + '\n')

            output_record = {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "gold_tables": list(gold_tables),
                "gold_cols": list(gold_cols),
                "pred_tables": list(pred_tables),
                "pred_cols": list(pred_cols),
                "missing_tables": missing_tables,
                "missing_cols": missing_cols,
                "extra_tables": extra_tables,
                "extra_cols": extra_cols,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "ex": ex_score,
            }
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(f"🚨 Pipeline failed on Question ID {question_id}: {e}")
            logger.debug(f"[Traceback] Question ID {question_id}:\n{traceback.format_exc()}")
            with open(pred_save_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"question_id": question_id, "status": "Error"}, ensure_ascii=False) + '\n')
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"question_id": question_id, "status": "Error"}, ensure_ascii=False) + '\n')
        finally:
            if 'result' in locals():
                del result
            gc.collect()

    logger.info("🎉 Inference loop finished. Calculating final metrics...")

    # 5. 디스크 기록을 다시 읽어 평균 계산
    try:
        df_output = pd.read_json(output_path, lines=True, orient='records')
        if 'recall' in df_output.columns:
            overall_recall = df_output['recall'].mean()
            overall_precision = df_output['precision'].mean()
            overall_ex = df_output['ex'].mean()
        else:
            logger.warning("No valid predictions found. Setting metrics to 0.0")
            overall_recall = overall_precision = overall_ex = 0.0
    except Exception as e:
        logger.error(f"Output JSONL Parsing Failed: {e}")
        overall_recall = overall_precision = overall_ex = 0.0

    # F1 (조화평균) — REPRODUCIBILITY.md 의 headline 지표
    overall_f1 = (
        2 * overall_recall * overall_precision / (overall_recall + overall_precision)
        if (overall_recall + overall_precision) else 0.0
    )

    # 6. summary_all.csv 누적
    hparams_str = f"Filter: {config.get('filter', {})} | Extractor: {config.get('connectivity_extractor', {})}"
    summary_record = pd.DataFrame([{
        "method": exp_name,
        "recall": overall_recall,
        "precision": overall_precision,
        "f1": overall_f1,
        "ex": overall_ex,
        "hparams": hparams_str,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])
    base_outputs_dir = os.path.abspath(os.path.join(output_dir, ".."))
    summary_path = os.path.join(base_outputs_dir, "summary_all.csv")
    if os.path.exists(summary_path):
        summary_record.to_csv(summary_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        summary_record.to_csv(summary_path, index=False, encoding='utf-8')
    logger.info(f"📈 Summary appended to: {summary_path}")

    # 7. 최종 메트릭 로깅
    logger.info("=" * 60)
    logger.info("📊 Final Evaluation Metrics")
    logger.info("=" * 60)
    logger.info(f"🎯 Recall:             {overall_recall:.4f}")
    logger.info(f"🎯 Precision:          {overall_precision:.4f}")
    logger.info(f"🎯 F1 (harmonic):      {overall_f1:.4f}")
    logger.info(f"🎯 Execution Accuracy: {overall_ex:.4f} ({total_ex_score}/{valid_ex_count})")
    logger.info("=" * 60)

    # Filter 단계 처리시간 집계 (profiling 에서)
    filter_timing: Dict[str, Any] = {}
    try:
        if os.path.exists(profiling_path):
            times = []
            with open(profiling_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        t = json.loads(line).get("filtering")
                        if isinstance(t, (int, float)):
                            times.append(float(t))
                    except Exception:
                        continue
            if times:
                ts = sorted(times)
                n = len(ts)
                filter_timing = {
                    'filter_time_mean_s': sum(ts) / n,
                    'filter_time_median_s': ts[n // 2],
                    'filter_time_p95_s': ts[min(n - 1, int(0.95 * n))],
                    'filter_time_max_s': ts[-1],
                    'filter_time_total_s': sum(ts),
                    'filter_samples': n,
                }
                logger.info(
                    f"⏱️  Filter timing — mean={filter_timing['filter_time_mean_s']:.2f}s | "
                    f"median={filter_timing['filter_time_median_s']:.2f}s | n={n}"
                )
    except Exception as e:
        logger.warning(f"Filter timing aggregation failed: {e}")

    # LLM 토큰 사용량 (GLM 비용 추적)
    token_usage_summary: Dict[str, Any] = {}
    try:
        from llm_client.api_handler import APIClient
        token_usage_summary = APIClient.get_usage_summary()
        total = token_usage_summary.get("total", {})
        if total.get("calls", 0) > 0:
            fresh_in = total["input_tokens"] - total["cached_input_tokens"]
            logger.info(
                f"🪙 LLM token usage — calls={total['calls']} | "
                f"input={total['input_tokens']} (fresh={fresh_in}, cached={total['cached_input_tokens']}) | "
                f"output={total['output_tokens']}"
            )
            with open(os.path.join(output_dir, "token_usage.json"), 'w', encoding='utf-8') as f:
                json.dump(token_usage_summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Token usage logging failed: {e}")

    # 8. metrics.txt 작성
    metric_save_path = os.path.join(output_dir, "metrics.txt")
    with open(metric_save_path, 'w', encoding='utf-8') as f:
        f.write(f"recall: {overall_recall:.4f}\n")
        f.write(f"precision: {overall_precision:.4f}\n")
        f.write(f"f1: {overall_f1:.4f}\n")
        f.write(f"ex: {overall_ex:.4f}\n")
        for k, v in filter_timing.items():
            f.write(f"{k}: {int(v)}\n" if k == 'filter_samples' else f"{k}: {v:.4f}\n")
        total = token_usage_summary.get("total", {}) if token_usage_summary else {}
        if total.get("calls", 0) > 0:
            f.write(f"llm_calls: {total['calls']}\n")
            f.write(f"llm_input_tokens: {total['input_tokens']}\n")
            f.write(f"llm_cached_input_tokens: {total['cached_input_tokens']}\n")
            f.write(f"llm_output_tokens: {total['output_tokens']}\n")

    logger.info("✅ All tasks completed. Forcing process termination.")
    os._exit(0)


if __name__ == "__main__":
    main()
