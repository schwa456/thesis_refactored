import sqlglot
from sqlglot.expressions import Alias, Column, Table
from typing import Set, Tuple, List
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

def parse_sql_elements(sql: str) -> Tuple[Set[str], Set[str]]:
    """SQL을 파싱하여 명시된 Table 명과 Column 명 집합을 반환.

    Wave 13 patch (DECISIONS 2026-05-20 §2): SQL alias (derived/computed column)
    는 schema linking target 의 real DB column 아님 — sqlglot 의 find_all(Alias)
    로 alias name 식별 후 columns set 에서 제외. alias 의 inner expression 의
    real columns 는 find_all(Column) 으로 동시 추출 정합 위 retain.
    """

    if not sql:
        return set(), set()

    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        tables = set(node.name.lower() for node in parsed.find_all(Table) if node.name)

        # alias names 식별 (sqlglot Alias AST nodes)
        alias_names = set()
        for alias_node in parsed.find_all(Alias):
            if alias_node.alias:
                alias_names.add(alias_node.alias.lower())

        # columns 추출 시 alias 제외 (alias name = derived/computed column, NOT real DB column)
        columns = set(
            node.name.lower() for node in parsed.find_all(Column)
            if node.name and node.name.lower() not in alias_names
        )
        return tables, columns

    except Exception as e:
        logger.debug(f"SQL Parsing Error: {e} | SQL: {sql}")
        return set(), set()

def parse_sql_join_tables(sql: str) -> Set[str]:
    """SQL에서 JOIN clause에 등장하는 테이블만 추출합니다.
    Bridge table 식별에 사용: FROM/JOIN에 등장하지만 SELECT/WHERE에서
    직접 참조되지 않는 테이블이 bridge table일 가능성이 높습니다."""
    if not sql:
        return set()
    try:
        from sqlglot.expressions import Join
        parsed = sqlglot.parse_one(sql, read="sqlite")
        join_tables = set()
        for join_node in parsed.find_all(Join):
            for table_node in join_node.find_all(Table):
                if table_node.name:
                    join_tables.add(table_node.name.lower())
        return join_tables
    except Exception as e:
        logger.debug(f"JOIN Parsing Error: {e} | SQL: {sql}")
        return set()


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