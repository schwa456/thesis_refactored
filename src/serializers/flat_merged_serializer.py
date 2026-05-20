"""C-v3 Flat Merged Representation (Wave 11, 학술 agent §3.4 + §5.5 정합).

JOIN simplification — table.col flat list 로 schema 평탄화. SQL Generator 의
JOIN 추론 부담 감소 + multi-table EX 효과 측정.

Variants:
  - C-v3a : fk_relations 포함 (FK hint 추가, JOIN path 명시)
  - C-v3b : fk_relations 없음 (table.col flat only, 가장 minimal)

**Immutability contract (Wave 11 Debug 2026-05-20)**:
본 함수는 m4_output / fk_relations 입력을 read-only 사용. dict/list mutation 없음
(string concat 만). 즉 caller 의 final_nodes / final_result 를 절대 변경 안 함.
Wave 11 Phase B 의 c_v3a/v3b Schema Content Invariance violation 의 root cause
는 LLM stochastic 이지 본 serializer 의 implementation bug 아님 (regression
test src/serializers/tests/test_wave11_serializers.py 로 검증).
"""
from typing import Dict, List, Optional, Sequence, Tuple


def format_flat_schema(
    m4_output: Dict[str, List[str]],
    fk_relations: Optional[Sequence[Tuple[str, str, str, str]]] = None,
) -> str:
    """학술 agent §5.5 정합. table.col flat string + (옵션) FK relations.

    Args:
        m4_output: {table: [col, ...]} — M4 union output
        fk_relations: [(parent_table, parent_col, child_table, child_col), ...]
                      C-v3a 에만 전달. None / [] 이면 C-v3b (FK 없음).

    Returns:
        Flat schema string. Filter 된 column 만 포함 (Schema Content Invariance retain).
    """
    all_columns: List[str] = []
    for table, cols in (m4_output or {}).items():
        for col in (cols or []):
            all_columns.append(f"{table}.{col}")

    schema_str = "[Available Columns]\n" + ", ".join(all_columns)

    # C-v3a: FK hint 추가 — Filter 된 column 중 endpoint 가 union 안에 있는 FK 만 포함
    if fk_relations:
        union_set = set(all_columns)
        fk_lines: List[str] = []
        for fk in fk_relations:
            if not isinstance(fk, (tuple, list)) or len(fk) != 4:
                continue
            pt, pc, ct, cc = (str(x) for x in fk)
            parent_full = f"{pt}.{pc}"
            child_full = f"{ct}.{cc}"
            # 학술 agent §5.5: "Filter된 컬럼 중 해당 FK가 포함된 경우만"
            # 양쪽 endpoint 중 하나라도 union 안에 있으면 hint 로 노출
            if parent_full in union_set or child_full in union_set:
                fk_lines.append(f"  {parent_full} -> {child_full}")
        if fk_lines:
            schema_str += "\n\n[Foreign Key Relations]\n" + "\n".join(fk_lines)

    return schema_str
