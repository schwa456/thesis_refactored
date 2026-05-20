"""C-v1 Source-Tagged Schema Serialization (Wave 11, 학술 agent §3.2 + §5.2 정합).

M4 Bidirectional Filter 의 Forward / Backward 결과를 각 column 에 태그로 부착해
SQL Generator 에 confidence signal 을 전달.

Tag legend (학술 agent §3.2):
  [F+B]  : Forward + Backward 모두 select (high confidence — anchor selected)
  [F]    : Forward 만 select (question-matched, semantic confidence)
  [B]    : Backward 만 select (structural confidence, SQL-aware)

Schema Content Invariance retain: 같은 M4 union → 같은 column 집합 (R/P/F1
identical, ΔR/ΔP/ΔF1 ±0.0001 허용). EX 만 직렬화 방식 차이로 변화 측정.

**Immutability contract (Wave 11 Debug 2026-05-20)**:
본 함수는 m4_output / forward_set / backward_set 입력을 read-only 사용. dict/set
mutation 없음 (tag_columns 의 union_set 도 새 dict 반환, _normalize_set 도 새 set
반환). 즉 caller 의 final_nodes / filter_info 를 절대 변경 안 함. Wave 11 Phase B
의 c_v1 invariance OK (R/P/F1 sub-noise) 와 정합.
"""
from typing import Dict, List, Set, Tuple


def tag_columns(
    forward_set: Set[Tuple[str, str]],
    backward_set: Set[Tuple[str, str]],
    union_set: Set[Tuple[str, str]],
) -> Dict[Tuple[str, str], str]:
    """학술 agent §5.2 그대로.

    Args:
        forward_set: {(table, col)} — M4 Forward output (sanitized)
        backward_set: {(table, col)} — M4 Backward output (sanitized)
        union_set: {(table, col)} — 최종 M4 union output (직렬화 대상)

    Returns:
        {(table, col): "[F+B]" | "[F]" | "[B]"}
    """
    tagged: Dict[Tuple[str, str], str] = {}
    for (table, col) in union_set:
        in_f = (table, col) in forward_set
        in_b = (table, col) in backward_set
        if in_f and in_b:
            tagged[(table, col)] = "[F+B]"
        elif in_f:
            tagged[(table, col)] = "[F]"
        elif in_b:
            tagged[(table, col)] = "[B]"
        else:
            # neither side selected (predict: not in M4 union) — defensive
            tagged[(table, col)] = "[?]"
    return tagged


def _normalize_set(items) -> Set[Tuple[str, str]]:
    """List[str like "table.col"] or List[Tuple] or Set → Set[(table, col)].

    Bidirectional filter stats["forward_set"] 는 sorted list of "table.col" strings.
    """
    out: Set[Tuple[str, str]] = set()
    for it in items or []:
        if isinstance(it, tuple) and len(it) == 2:
            out.add((str(it[0]), str(it[1])))
        elif isinstance(it, str) and "." in it:
            t, c = it.split(".", 1)
            out.add((t, c))
    return out


def format_tagged_schema(
    m4_output: Dict[str, List[str]],
    forward_set,
    backward_set,
) -> str:
    """M4 union schema 의 column 별 source tag 부착 — 학술 agent §5.2 정합.

    Args:
        m4_output: {table: [col, ...]} — M4 union (Filter.refine 의 subgraph form)
        forward_set: List[str "table.col"] or Set[Tuple] — M4 Forward set
        backward_set: List[str "table.col"] or Set[Tuple] — M4 Backward set

    Returns:
        Formatted schema text with per-column tags. Legend prefix 포함.
    """
    fwd_norm = _normalize_set(forward_set)
    bwd_norm = _normalize_set(backward_set)
    union_norm: Set[Tuple[str, str]] = set()
    for table, cols in (m4_output or {}).items():
        for col in (cols or []):
            union_norm.add((table, col))
    tags = tag_columns(fwd_norm, bwd_norm, union_norm)

    lines: List[str] = [
        "[Database Schema with Selection Confidence]",
        "Legend: [F+B]=High confidence | [F]=Question-matched | [B]=Structural",
        "",
    ]
    for table, cols in (m4_output or {}).items():
        lines.append(f"Table: {table}")
        for col in (cols or []):
            tag = tags.get((table, col), "[?]")
            lines.append(f"  - {col} {tag}")
        lines.append("")
    return "\n".join(lines)
