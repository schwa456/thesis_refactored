import os
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

    for path in [pred_save_path, score_save_path, profiling_path, output_path]:
        if os.path.exists(path):
            os.remove(path)

    total_ex_score = 0
    valid_ex_count = 0
    
    for item in tqdm(dataset, desc="Running Pipeline"):
        db_id = item.get("db_id")
        question = item.get("question")
        question_id = item.get("question_id")
        gold_sql = item.get("SQL", item.get("query", ""))
        db_path = os.path.join("data/raw/BIRD_dev/dev_databases", db_id, f"{db_id}.sqlite")

        logger.debug(f"Question {question_id}: {question}")
        
        try:
            result = pipeline.run(db_id=db_id, query=question)
            pred_sql = result.get("generated_sql", "")

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

            gold_tables, gold_cols = parse_sql_elements(gold_sql)
            pred_tables, pred_cols = parse_sql_elements(pred_sql)

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

            recall, precision, missing_cols, extra_cols = calculate_schema_metrics(pred_cols, gold_cols)
            _, _, missing_tables, extra_tables = calculate_schema_metrics(pred_tables, gold_tables)
            
            pred_record = {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "status": result.get("status"),
                "uncertainty": result.get("uncertainty", 0.0),
                "final_nodes": result.get("final_nodes", []),
                "reasoning": result.get("reasoning", ""),
                "generated_sql": pred_sql,
                "ex_score": ex_score
            }
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
    metric_save_path = os.path.join(output_dir, "metrics.txt")
    with open(metric_save_path, 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    logger.info("✅ All tasks completed. Forcing process termination.")
    os._exit(0)

if __name__ == "__main__":
    main()