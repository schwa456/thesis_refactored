import sqlite3
from typing import Tuple, Any, Set
from utils.logger import get_logger

logger = get_logger(__name__)

def execute_sql(db_path: str, sql: str) -> Tuple[bool, Any]:
    """해당 경로의 SQLite DB에 연결하여 SQL을 실행하고 결과를 반환합니다."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        conn.close()
        return True, res
    except Exception as e:
        return False, str(e)

def evaluate_ex(pred_sql: str, gold_sql: str, db_path: str) -> int:
    """
    EX (Execution Accuracy)를 측정합니다.
    결과 집합(Set)이 완전히 동일하면 1, 아니면 0을 반환합니다.
    """
    pred_success, pred_res = execute_sql(db_path, pred_sql)
    gold_success, gold_res = execute_sql(db_path, gold_sql)

    if not pred_success:
        logger.debug(f"[EX Failed] Prediction Execution Error: {pred_res}")
        return 0

    if not gold_success:
        logger.warning(f"[Warning] Gold SQL also failed to execute: {gold_res}")
        return 0

    # 결과값 순서 무관 비교를 위해 Set으로 변환
    # (주의: ORDER BY가 포함된 쿼리라면 리스트 자체로 비교해야 엄밀하지만, 일반적인 EX는 Set 단위 비교를 허용합니다)
    try:
        if set(pred_res) == set(gold_res):
            logger.debug("[EX Succeded] EX Evaluation Success")
            return 1
    except TypeError:
        # 결과에 unhashable type이 있는 경우 (예: dict) 문자열 처리 후 비교
        if str(pred_res) == str(gold_res):
            return 1

    return 0