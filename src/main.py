import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import csv
import json
import datetime
import traceback
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Any

from utils.config_parser import get_args_and_config
from utils.logger import setup_logger, get_logger
from utils.executor import evaluate_ex
from utils.evaluator import parse_sql_elements, calculate_schema_metrics
from pipeline import SchemaLinkingPipeline

def main():
    # 1. Config 로드 및 전역 로거 세팅 (config_parser가 폴더들을 다 만들어줍니다!)
    args, config = get_args_and_config()

    log_dir = config['paths']['log_dir']
    output_dir = config['paths']['output_dir']
    exp_name = config['experiment_name']
    
    setup_logger(log_dir=log_dir, exp_name=config['experiment_name'])
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info(f"🔥 Starting Evaluation for Experiment: [{config['experiment_name']}]")
    logger.info("=" * 60)

    # 2. 파이프라인 객체 생성
    pipeline = SchemaLinkingPipeline(config)

    # 3. 평가 데이터셋 로드 (dev.json 경로)
    data_path = config['paths'].get('dev_json', 'data/raw/BIRD_dev/dev.json')
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        # For Testing
        # dataset = dataset[:5]
        logger.info(f"Loaded {len(dataset)} queries from {data_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 4. 추론 루프 실행
    pred_save_path = os.path.join(output_dir, "predictions.jsonl")
    score_save_path = os.path.join(output_dir, f"score_analysis_{exp_name}.jsonl")
    profiling_path = os.path.join(output_dir, f"profiling_{exp_name}.jsonl")
    output_path = os.path.join(output_dir, f"output_{exp_name}.jsonl")
    reasoning_path = os.path.join(output_dir, f"reasoning_detail_{exp_name}.jsonl")

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
        for path in [pred_save_path, score_save_path, profiling_path, output_path, reasoning_path]:
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

        logger.debug(f"Question {question_id}: {question}")
        
        try:
            # evidence: BIRD-dev `external_knowledge` 필드 forward (LLMSQLGenerator 의 SQL gen prompt 에 삽입).
            # 2026-05-13 fix — Baseline EX 비교 시 anchor EX 21.91%p gap 의 dominant 원인 회수.
            result = pipeline.run(db_id=db_id, query=question, evidence=item.get("evidence", ""))
            
            gold_tables, gold_cols = parse_sql_elements(gold_sql)
            gold_tables = set(t.lower() for t in gold_tables)
            gold_cols = set(c.lower() for c in gold_cols)

            pred_tables = set()
            pred_cols = set()
            pred_sql = result.get("generated_sql", "")

            final_nodes = result.get("final_nodes", [])
            for node in final_nodes:
                if '->' in node:
                    continue  # 매크로 엣지 제외
                
                if '.' in node:
                    tbl, col = node.split('.', 1)
                    pred_tables.add(tbl.lower())
                    
                    col_lower = col.lower()
                    tbl_col_lower = f"{tbl.lower()}.{col_lower}"
                    
                    if tbl_col_lower in gold_cols:
                        pred_cols.add(tbl_col_lower)
                    else:
                        pred_cols.add(col_lower)
                else:
                    pred_tables.add(node.lower())

            pred_tables = list(pred_tables)
            pred_cols = list(pred_cols)

            if "execution_time" in result:
                profiling_record = {"query_id": question_id}
                profiling_record.update(result.get("execution_time", {}))
                with open(profiling_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(profiling_record, ensure_ascii=False) + '\n')
                logger.debug(f"[Execution Time]: \n{profiling_record}")

            ex_score = 0
            if pred_sql and gold_sql and os.path.exists(db_path):
                try:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(evaluate_ex, pred_sql=pred_sql, gold_sql=gold_sql, db_path=db_path)
                        ex_score = future.result(timeout=15.0)
                
                except concurrent.futures.TimeoutError:
                    logger.warning(f"🚨 SQL Execution Timeout (15s). Cartesian Product detected. Setting EX = 0")
                    ex_score = 0
                except concurrent.futures.process.BrokenProcessPool:
                    logger.warning(f"🚨 SQL Execution caused OOM (SIGKILL). Main process protected. Setting EX = 0")
                    ex_score = 0
                except Exception as e:
                    logger.warning(f"🚨 SQL Execution Error: {e}")
                    ex_score = 0

                total_ex_score += ex_score
                valid_ex_count += 1
            elif not os.path.exists(db_path):
                logger.warning(f"DB file not found for EX evaluation: {db_path}")

            #gold_tables, gold_cols = parse_sql_elements(gold_sql)
            #pred_tables, pred_cols = parse_sql_elements(pred_sql)

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
                    else:
                        if name_lower in gold_tables:
                            is_gold = True
                
                    score_record = {
                        "query_id": question_id,
                        "node_name": name,
                        "score": float(score),
                        "is_gold": is_gold
                    }
                    f.write(json.dumps(score_record, ensure_ascii=False) + '\n')

            recall, precision, missing_cols, extra_cols = calculate_schema_metrics(set(pred_cols), set(gold_cols))
            _, _, missing_tables, extra_tables = calculate_schema_metrics(set(pred_tables), set(gold_tables))
            
            pred_record = {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "status": result.get("status"),
                "uncertainty": result.get("uncertainty", 0.0),
                "final_nodes": result.get("final_nodes", []),
                "reasoning": result.get("reasoning", ""),
                "generated_sql": pred_sql,
                "ex_score": ex_score,
            }
            for k in (
                "adaptive_route", "adaptive_confidence",
                "tier1_dropped_count", "tier2_pool_count",
                "restored_count", "promoted_count",
                "retry_attempts",
                "connectivity_valid", "connectivity_issues",
                "repair_added_tables", "repair_added_fk_cols",
                "builder_info", "selector_info", "extractor_info", "filter_info",
            ):
                if k in result:
                    pred_record[k] = result[k]
            with open(pred_save_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(pred_record, ensure_ascii=False) + '\n')

            if result.get("trace"):
                reasoning_record = {
                    "question_id": question_id,
                    "db_id": db_id,
                    "question": question,
                    "reasoning_summary": result.get("reasoning", ""),
                    "trace": result["trace"],
                }
                with open(reasoning_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(reasoning_record, ensure_ascii=False) + '\n')
            
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
                "ex": ex_score
            }

            
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            
        except Exception as e:
            logger.error(f"🚨 Pipeline failed on Question ID {question_id}: {e}")
            logger.debug(f"[Traceback] Question ID {question_id}:\n{traceback.format_exc()}")
            
            # 에러 발생 시 빈 값으로 파일에 기록
            with open(pred_save_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"question_id": question_id, "status": "Error"}, ensure_ascii=False) + '\n')
            
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"question_id": question_id, "status": "Error"}, ensure_ascii=False) + '\n')

        finally:
            if 'result' in locals():
                del result
            gc.collect()

    logger.info("🎉 Inference loop finished. Calculating final metrics...")
    
    # 디스크에 기록해둔 CSV를 다시 읽어와서 평균을 냅니다.
    
    try:
        df_output = pd.read_json(
            output_path, 
            lines=True,
            orient='records'
        )

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
    
    hparams_str = f"Filter: {config.get('filter', {})} | Extractor: {config.get('connectivity_extractor', {})}"
    
    summary_record = pd.DataFrame([{
        "method": exp_name,
        "recall": overall_recall,
        "precision": overall_precision,
        "ex": overall_ex,
        "hparams": hparams_str,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    base_outputs_dir = os.path.abspath(os.path.join(output_dir, ".."))
    summary_path = os.path.join(base_outputs_dir, "summary_all.csv")
    
    if os.path.exists(summary_path):
        summary_record.to_csv(summary_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        summary_record.to_csv(summary_path, index=False, encoding='utf-8')
    logger.info(f"📈 Summary appended to: {summary_path}")

    # ==========================================
    # 💡 4. 최종 메트릭 로깅 및 종료
    # ==========================================
    logger.info("=" * 60)
    logger.info("📊 Final Evaluation Metrics (Parsed from SQL)")
    logger.info("=" * 60)
    logger.info(f"🎯 Average Precision:  {overall_precision:.4f}")
    logger.info(f"🎯 Average Recall:     {overall_recall:.4f}")
    logger.info(f"🎯 Execution Accuracy: {overall_ex:.4f} ({total_ex_score}/{valid_ex_count})")
    logger.info("=" * 60)

    metrics = {
        'precision': overall_precision,
        'recall': overall_recall,
        'ex': overall_ex
    }

    filter_timing = {}
    try:
        if os.path.exists(profiling_path):
            times = []
            with open(profiling_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        t = rec.get("filtering")
                        if isinstance(t, (int, float)):
                            times.append(float(t))
                    except Exception:
                        continue
            if times:
                times_sorted = sorted(times)
                n = len(times_sorted)
                filter_timing = {
                    'filter_time_mean_s': sum(times_sorted) / n,
                    'filter_time_median_s': times_sorted[n // 2],
                    'filter_time_p95_s': times_sorted[min(n - 1, int(0.95 * n))],
                    'filter_time_max_s': times_sorted[-1],
                    'filter_time_total_s': sum(times_sorted),
                    'filter_samples': n,
                }
                logger.info(
                    f"⏱️  Filter timing — mean={filter_timing['filter_time_mean_s']:.2f}s | "
                    f"median={filter_timing['filter_time_median_s']:.2f}s | "
                    f"p95={filter_timing['filter_time_p95_s']:.2f}s | "
                    f"max={filter_timing['filter_time_max_s']:.2f}s | n={n}"
                )
    except Exception as e:
        logger.warning(f"Filter timing aggregation failed: {e}")

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
            for m, u in token_usage_summary.get("by_model", {}).items():
                logger.info(
                    f"    └ [{m}] calls={u['calls']} "
                    f"input={u['input_tokens']} cached={u['cached_input_tokens']} output={u['output_tokens']}"
                )
            token_usage_path = os.path.join(output_dir, "token_usage.json")
            with open(token_usage_path, 'w', encoding='utf-8') as f:
                json.dump(token_usage_summary, f, ensure_ascii=False, indent=2)
            logger.info(f"🪙 Token usage saved to {token_usage_path}")
    except Exception as e:
        logger.warning(f"Token usage logging failed: {e}")

    # Aggregate per-stage telemetry from pred_save_path (*_info fields)
    stage_aggregates: Dict[str, Any] = _aggregate_stage_telemetry(pred_save_path, logger)

    metric_save_path = os.path.join(output_dir, "metrics.txt")
    with open(metric_save_path, 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        for k, v in filter_timing.items():
            if k == 'filter_samples':
                f.write(f"{k}: {int(v)}\n")
            else:
                f.write(f"{k}: {v:.4f}\n")
        total = token_usage_summary.get("total", {}) if token_usage_summary else {}
        if total.get("calls", 0) > 0:
            f.write(f"llm_calls: {total['calls']}\n")
            f.write(f"llm_input_tokens: {total['input_tokens']}\n")
            f.write(f"llm_cached_input_tokens: {total['cached_input_tokens']}\n")
            f.write(f"llm_output_tokens: {total['output_tokens']}\n")
        if stage_aggregates:
            f.write("\n# --- Stage telemetry aggregates ---\n")
            for k, v in stage_aggregates.items():
                if isinstance(v, float):
                    f.write(f"{k}: {v:.4f}\n")
                elif isinstance(v, int):
                    f.write(f"{k}: {v}\n")
                else:
                    f.write(f"{k}: {v}\n")
            stage_agg_path = os.path.join(output_dir, "stage_aggregates.json")
            with open(stage_agg_path, 'w', encoding='utf-8') as sf:
                json.dump(stage_aggregates, sf, ensure_ascii=False, indent=2)
            logger.info(f"📊 Stage aggregates saved to {stage_agg_path}")

    logger.info("✅ All tasks completed. Forcing process termination.")
    os._exit(0)


def _aggregate_stage_telemetry(pred_path: str, logger) -> Dict[str, Any]:
    """Aggregate builder/selector/extractor/filter *_info fields from predictions.jsonl.

    Returns flat dict of means/counters keyed with stage prefix, useful for
    at-a-glance per-experiment telemetry in metrics.txt / stage_aggregates.json.
    """
    if not os.path.exists(pred_path):
        return {}

    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    builder_time: List[float] = []
    builder_tables: List[int] = []
    builder_columns: List[int] = []
    builder_edges: List[int] = []
    builder_types: Dict[str, int] = {}

    selector_time: List[float] = []
    selector_types: Dict[str, int] = {}

    extractor_time: List[float] = []
    extractor_types: Dict[str, int] = {}
    extractor_selected_nodes: List[int] = []
    extractor_prize_nonzero: List[int] = []
    extractor_thresholds: List[float] = []

    filter_time: List[float] = []
    filter_types: Dict[str, int] = {}
    filter_llm_calls: List[int] = []
    filter_tokens_in: List[int] = []
    filter_tokens_out: List[int] = []
    filter_cached: List[int] = []
    filter_route_dist: Dict[str, int] = {}
    filter_retry_counts: List[int] = []
    filter_repair_attempts: int = 0
    filter_repair_success: int = 0

    n_rows = 0
    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                n_rows += 1

                bi = rec.get("builder_info") or {}
                if bi:
                    bt = bi.get("builder_type")
                    if bt:
                        builder_types[bt] = builder_types.get(bt, 0) + 1
                    t = (bi.get("builder_timings") or {}).get("total_s")
                    if isinstance(t, (int, float)):
                        builder_time.append(float(t))
                    if isinstance(bi.get("num_tables"), int):
                        builder_tables.append(bi["num_tables"])
                    if isinstance(bi.get("num_columns"), int):
                        builder_columns.append(bi["num_columns"])
                    if isinstance(bi.get("num_edges"), int):
                        builder_edges.append(bi["num_edges"])

                si = rec.get("selector_info") or {}
                if si:
                    st = si.get("selector_type") or si.get("sel_type")
                    if st:
                        selector_types[st] = selector_types.get(st, 0) + 1
                    for key in ("selector_time_s", "sel_time_s", "select_time_s"):
                        if isinstance(si.get(key), (int, float)):
                            selector_time.append(float(si[key]))
                            break

                ei = rec.get("extractor_info") or {}
                if ei:
                    et = ei.get("extractor_type")
                    if et:
                        extractor_types[et] = extractor_types.get(et, 0) + 1
                    if isinstance(ei.get("extractor_time_s"), (int, float)):
                        extractor_time.append(float(ei["extractor_time_s"]))
                    if isinstance(ei.get("extractor_num_selected_nodes"), int):
                        extractor_selected_nodes.append(ei["extractor_num_selected_nodes"])
                    if isinstance(ei.get("extractor_prize_nonzero"), int):
                        extractor_prize_nonzero.append(ei["extractor_prize_nonzero"])
                    if isinstance(ei.get("extractor_threshold"), (int, float)):
                        extractor_thresholds.append(float(ei["extractor_threshold"]))

                fi = rec.get("filter_info") or {}
                if fi:
                    ft = fi.get("filter_type")
                    if ft:
                        filter_types[ft] = filter_types.get(ft, 0) + 1
                    if isinstance(fi.get("filter_time_s"), (int, float)):
                        filter_time.append(float(fi["filter_time_s"]))
                    if isinstance(fi.get("filter_llm_calls"), int):
                        filter_llm_calls.append(fi["filter_llm_calls"])
                    if isinstance(fi.get("filter_tokens_in"), int):
                        filter_tokens_in.append(fi["filter_tokens_in"])
                    if isinstance(fi.get("filter_tokens_out"), int):
                        filter_tokens_out.append(fi["filter_tokens_out"])
                    if isinstance(fi.get("filter_cached_tokens"), int):
                        filter_cached.append(fi["filter_cached_tokens"])
                    route = fi.get("filter_route") or fi.get("route")
                    if isinstance(route, str):
                        filter_route_dist[route] = filter_route_dist.get(route, 0) + 1
                    ra = fi.get("filter_pipeline_retry_attempts")
                    if ra is None:
                        ra = fi.get("pipeline_retry_attempts")
                    if isinstance(ra, int):
                        filter_retry_counts.append(ra)
                    repair_flag = fi.get("filter_repair_attempted")
                    if repair_flag is None:
                        repair_flag = fi.get("repair_attempted")
                    if repair_flag:
                        filter_repair_attempts += 1
                        conn_valid = fi.get("filter_connectivity_valid")
                        if conn_valid is None:
                            conn_valid = fi.get("connectivity_valid")
                        if conn_valid:
                            filter_repair_success += 1
    except Exception as e:
        logger.warning(f"Stage telemetry aggregation failed: {e}")
        return {}

    agg: Dict[str, Any] = {"n_rows": n_rows}

    if builder_time:
        agg["builder_time_mean_s"] = _mean(builder_time)
    if builder_tables:
        agg["builder_num_tables_mean"] = _mean(builder_tables)
    if builder_columns:
        agg["builder_num_columns_mean"] = _mean(builder_columns)
    if builder_edges:
        agg["builder_num_edges_mean"] = _mean(builder_edges)
    if builder_types:
        agg["builder_type_distribution"] = dict(builder_types)

    if selector_time:
        agg["selector_time_mean_s"] = _mean(selector_time)
    if selector_types:
        agg["selector_type_distribution"] = dict(selector_types)

    if extractor_time:
        agg["extractor_time_mean_s"] = _mean(extractor_time)
    if extractor_selected_nodes:
        agg["extractor_selected_nodes_mean"] = _mean(extractor_selected_nodes)
    if extractor_prize_nonzero:
        agg["extractor_prize_nonzero_mean"] = _mean(extractor_prize_nonzero)
    if extractor_thresholds:
        agg["extractor_threshold_mean"] = _mean(extractor_thresholds)
    if extractor_types:
        agg["extractor_type_distribution"] = dict(extractor_types)

    if filter_time:
        agg["filter_stage_time_mean_s"] = _mean(filter_time)
    if filter_llm_calls:
        agg["filter_llm_calls_mean"] = _mean(filter_llm_calls)
        agg["filter_llm_calls_total"] = int(sum(filter_llm_calls))
    if filter_tokens_in:
        agg["filter_tokens_in_total"] = int(sum(filter_tokens_in))
    if filter_tokens_out:
        agg["filter_tokens_out_total"] = int(sum(filter_tokens_out))
    if filter_cached:
        agg["filter_cached_tokens_total"] = int(sum(filter_cached))
    if filter_types:
        agg["filter_type_distribution"] = dict(filter_types)
    if filter_route_dist:
        agg["filter_route_distribution"] = dict(filter_route_dist)
    if filter_retry_counts:
        agg["pipeline_retry_attempts_mean"] = _mean(filter_retry_counts)
        agg["pipeline_retry_attempts_max"] = int(max(filter_retry_counts))
    if filter_repair_attempts:
        agg["filter_repair_attempts"] = int(filter_repair_attempts)
        agg["filter_repair_successes"] = int(filter_repair_success)

    return agg

if __name__ == "__main__":
    main()