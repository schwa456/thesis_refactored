import os
import json
import time
from tqdm import tqdm
from typing import List, Dict, Any

from utils.config_parser import get_args_and_config
from utils.logger import setup_logger, get_logger
from utils.executor import evaluate_ex
from pipeline import SchemaLinkingPipeline

def calculate_metrics(predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, float]:
    """
    예측된 스키마와 정답(Gold) 스키마를 비교하여 Precision, Recall, F1을 계산합니다.
    (데이터셋의 정답 포맷에 따라 'gold_schema' 추출 부분은 수정이 필요할 수 있습니다)
    """
    total_tp = 0
    total_pred = 0
    total_gold = 0

    for pred, truth in zip(predictions, ground_truths):
        # 예측된 노드 리스트 (테이블명.컬럼명 형식의 1차원 리스트로 평탄화 가정)
        pred_nodes = set(pred.get('final_nodes', []))
        
        # 정답 노드 리스트 (데이터셋 구조에 맞게 수정 필요)
        # 예시: BIRD 데이터셋에서 사용자가 미리 추출해둔 정답 컬럼들
        gold_nodes = set(truth.get('gold_columns', [])) 
        
        tp = len(pred_nodes.intersection(gold_nodes))
        
        total_tp += tp
        total_pred += len(pred_nodes)
        total_gold += len(gold_nodes)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gold if total_gold > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"Precision": precision, "Recall": recall, "F1": f1}

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

    # 2. 파이프라인 객체 생성 (Registry의 마법이 여기서 일어납니다)
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
    start_time = time.time()

    total_ex_score = 0
    valid_ex_count = 0
    
    # tqdm으로 진행 바 표시
    for item in tqdm(dataset, desc="Running Pipeline"):
        db_id = item.get("db_id")
        question = item.get("question")
        question_id = item.get("question_id", len(predictions))

        gold_sql = item.get("SQL", item.get("query", ""))
        db_path = os.path.join("./data/raw/BIRD_dev/dev_databases", db_id, f"{db_id}.sqlite")
        
        try:
            # 💡 단 한 줄로 7단계 파이프라인 관통!
            result = pipeline.run(db_id=db_id, query=question)

            pred_sql = result.get("generated_sql", "")
            ex_score = 0

            if pred_sql and gold_sql and os.path.exists(db_path):
                ex_score = evaluate_ex(pred_sql=pred_sql, gold_sql=gold_sql, db_path=db_path)
                total_ex_score += ex_score
                valid_ex_count += 1
            elif not os.path.exists(db_path):
                logger.warning(f"DB file not found for EX evaluation: {db_path}")
            
            # 결과를 저장용 리스트에 보관
            pred_record = {
                "question_id": question_id,
                "db_id": db_id,
                "question": question,
                "status": result.get("status"),
                "uncertainty": result.get("uncertainty", 0.0),
                "final_nodes": result.get("final_nodes", []),
                "reasoning": result.get("reasoning", ""),
                "generated_sql": pred_sql,  # 생성된 SQL도 저장
                "ex_score": ex_score        # 개별 EX 점수 저장 (분석용)
            }
            predictions.append(pred_record)
            
        except Exception as e:
            logger.error(f"🚨 Pipeline failed on Question ID {question_id}: {e}")
            # 실패하더라도 전체 루프가 죽지 않도록 방어
            predictions.append({
                "question_id": question_id,
                "db_id": db_id,
                "status": "Error",
                "final_nodes": [],
                "generated_sql": "",
                "ex_score": 0
            })

    # 5. 결과 저장 및 지표 계산
    elapsed_time = time.time() - start_time
    logger.info(f"✅ Inference completed in {elapsed_time:.2f} seconds.")

    # 예측 결과 JSON 저장 (outputs/exp_name/predictions.json)
    pred_save_path = os.path.join(output_dir, "predictions.json")
    with open(pred_save_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    logger.info(f"💾 Predictions saved to: {pred_save_path}")

    # 평가 지표 계산 (정답 데이터가 dataset 안에 있다고 가정)
    metrics = calculate_metrics(predictions, dataset)

    ex_accuracy = total_ex_score / valid_ex_count if valid_ex_count > 0 else 0.0
    metrics["Execution_Accuracy"] = ex_accuracy
    
    logger.info("=" * 60)
    logger.info("📊 Final Evaluation Metrics")
    logger.info("=" * 60)
    logger.info(f"🎯 Precision:          {metrics['Precision']:.4f}")
    logger.info(f"🎯 Recall:             {metrics['Recall']:.4f}")
    logger.info(f"🎯 F1 Score:           {metrics['F1']:.4f}")
    logger.info(f"🎯 Execution Accuracy: {metrics['Execution_Accuracy']:.4f} ({total_ex_score}/{valid_ex_count})")
    logger.info("=" * 60)
    
    # 지표도 텍스트 파일로 저장
    metric_save_path = os.path.join(output_dir, "metrics.txt")
    with open(metric_save_path, 'w', encoding='utf-8') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

if __name__ == "__main__":
    main()