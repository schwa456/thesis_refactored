import os
import json
import time
import datetime
import pandas as pd
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
    
    setup_logger(log_dir=log_dir, exp_name=config['experiment_name'])
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info(f"🔥 Starting Evaluation for Experiment: [{config['experiment_name']}]")
    logger.info("=" * 60)

    # 2. 파이프라인 객체 생성
    pipeline = SchemaLinkingPipeline(config)

    # 3. 평가 데이터셋 로드 (dev.json 경로)
    data_path = config['paths'].get('dev_json', '../data/raw/BIRD_dev/dev.json')
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} queries from {data_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 4. 추론 루프 실행
    predictions = []
    csv_records = []
    start_time = time.time()

    total_ex_score = 0
    valid_ex_count = 0
    
    for item in tqdm(dataset, desc="Running Pipeline"):
        db_id = item.get("db_id")
        question = item.get("question")
        question_id = item.get("question_id", len(predictions))
        gold_sql = item.get("SQL", item.get("query", ""))
        db_path = os.path.join("./data/raw/BIRD_dev/dev_databases", db_id, f"{db_id}.sqlite")
        
        try:
            result = pipeline.run(db_id=db_id, query=question)
            pred_sql = result.get("generated_sql", "")

            ex_score = 0
            if pred_sql and gold_sql and os.path.exists(db_path):
                ex_score = evaluate_ex(pred_sql=pred_sql, gold_sql=gold_sql, db_path=db_path)
                total_ex_score += ex_score
                valid_ex_count += 1
            elif not os.path.exists(db_path):
                logger.warning(f"DB file not found for EX evaluation: {db_path}")

            gold_tables, gold_cols = parse_sql_elements(gold_sql)
            pred_tables, pred_cols = parse_sql_elements(pred_sql)

            recall, precision, missing_cols, extra_cols = calculate_schema_metrics(pred_cols, gold_cols)
            _, _, missing_tables, extra_tables = calculate_schema_metrics(pred_tables, gold_tables)
            
            # 결과를 저장용 리스트에 보관
            predictions.append({
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "status": result.get("status"),
                "uncertainty": result.get("uncertainty", 0.0),
                "final_nodes": result.get("final_nodes", []),
                "reasoning": result.get("reasoning", ""),
                "generated_sql": pred_sql,
                "ex_score": ex_score
            })
            
            csv_records.append({
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
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
            })
            
        except Exception as e:
            logger.error(f"🚨 Pipeline failed on Question ID {question_id}: {e}")
            # 에러 발생 시에도 CSV 행 수가 틀어지지 않도록 빈 값 삽입
            predictions.append({"question_id": question_id, "status": "Error"})
            csv_records.append({
                "question_id": question_id, "db_id": db_id, "question": question,
                "gold_sql": gold_sql, "pred_sql": "", "gold_tables": [], "gold_cols": [],
                "pred_tables": [], "pred_cols": [], "missing_tables": [], "missing_cols": [],
                "extra_tables": [], "extra_cols": [], "recall": 0.0, "precision": 0.0, "ex": 0
            })

    # 5. Output 저장 로직
    # 5-1. predictions.json 저장
    pred_save_path = os.path.join(output_dir, "predictions.json")
    with open(pred_save_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    # 5-2. 상세 Output CSV 저장
    df_output = pd.DataFrame(csv_records)
    output_csv_path = os.path.join(output_dir, f"output_{config['experiment_name']}.csv")
    df_output.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"💾 Detailed output saved to: {output_csv_path}")

    # 5-3. 글로벌 Summary CSV 로직 (Append)
    overall_recall = df_output['recall'].mean()
    overall_precision = df_output['precision'].mean()
    overall_ex = df_output['ex'].mean()
    
    # 설정된 Hyperparameters 문자열로 압축 저장
    hparams_str = f"Filter: {config.get('filter', {})} | Extractor: {config.get('connectivity_extractor', {})}"
    
    summary_record = pd.DataFrame([{
        "method": config['experiment_name'],
        "recall": overall_recall,
        "precision": overall_precision,
        "ex": overall_ex,
        "hparams": hparams_str,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    
    # 모든 실험 결과가 모이는 최상위 outputs 폴더에 summary 저장
    base_outputs_dir = os.path.abspath(os.path.join(output_dir, ".."))
    summary_path = os.path.join(base_outputs_dir, "summary_all.csv")
    
    if os.path.exists(summary_path):
        summary_record.to_csv(summary_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        summary_record.to_csv(summary_path, index=False, encoding='utf-8-sig')
    logger.info(f"📈 Summary appended to: {summary_path}")

    # 6. 최종 메트릭 로깅
    logger.info("=" * 60)
    logger.info("📊 Final Evaluation Metrics (Parsed from SQL)")
    logger.info("=" * 60)
    logger.info(f"🎯 Average Precision:  {overall_precision:.4f}")
    logger.info(f"🎯 Average Recall:     {overall_recall:.4f}")
    logger.info(f"🎯 Execution Accuracy: {overall_ex:.4f} ({total_ex_score}/{valid_ex_count})")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()