import sqlglot
from sqlglot.expressions import Table, Column
from typing import Set, Tuple, List
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

def parse_sql_elements(sql: str) -> Tuple[Set[str], Set[str]]:
    """SQL을 파싱하여 명시된 Table 명과 Column 명 집합을 반환"""

    if not sql:
        return set(), set()
    
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        tables = set(node.name.lower() for node in parsed.find_all(Table) if node.name)
        columns = set(node.name.lower() for node in parsed.find_all(Column) if node.name)
        return tables, columns

    except Exception as e:
        logger.debug(f"SQL Parsing Error: {e} | SQL: {sql}")
        return set(), set()

def calculate_schema_metrics(pred_elements: Set[str], gold_elements: Set[str]) -> Tuple[float, float, List[str], List[str]]:
    """Recall, Precision 및 Missing, Extra 요소를 계산합니다."""
    if not gold_elements and not pred_elements:
        return 1.0, 1.0, [], []
    
    intersection = pred_elements.intersection(gold_elements)

    recall = len(intersection) / len(gold_elements) if gold_elements else 0.0
    precision = len(intersection) / len(pred_elements) if pred_elements else 0.0
    
    missing = list(gold_elements - pred_elements)
    extra = list(pred_elements - gold_elements)
    
    return recall, precision, missing, extra