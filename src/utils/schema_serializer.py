"""V7-W1 (RFP #3, FKH) — FK hint schema serializer.

학술 agent RFP #3 spec (`planning/extractor/scholar_agent_extractor_rfp_2026-06-04.md`
§3.8) 정합. 본 모듈은 LLM filter prompt 의 schema 부분을 FK hint 첨가 후 직렬화
하는 helper 를 제공.

본 framework 의 실제 graph_data interface (`Dict[str, List[str]]` subgraph +
`metadata['fk_to_id']` keys `"src_t.src_c->dst_t.dst_c"`) 정합 정정:
- RFP §3.8 의 `graph_data.node_types`, `graph_data.edges(etype)`,
  `graph_data.node_name(nid)` 등 method 호출 — 본 framework 의 실측 dict 구조와
  다름. 본 모듈은 (extracted_subgraph: Dict[str, List[str]], metadata: dict)
  signature 로 정정 — FK 정보는 metadata['fk_to_id'] 또는 'fk_descriptions'
  keys 로 부터 추출.

핵심 contract: 본 함수는 extracted_nodes / extracted_edges 자체를 변경하지
않음. 단순 string format 만. R/P/F1 동일 보장.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# FK relation extraction
# ---------------------------------------------------------------------------
def _parse_fk_key(fk_key: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse 'src_t.src_c->dst_t.dst_c' → (src_t, src_c, dst_t, dst_c).

    Returns None on malformed key. Builder convention: fk_to_id key format =
    `_generate_fk_descriptions()` 에서 정의된 `"{from_table}.{from_column}->{to_table}.{to_column}"`.
    """
    if not isinstance(fk_key, str) or "->" not in fk_key:
        return None
    left, right = fk_key.split("->", 1)
    left, right = left.strip(), right.strip()
    if "." not in left or "." not in right:
        return None
    src_t, src_c = left.split(".", 1)
    dst_t, dst_c = right.split(".", 1)
    return src_t.strip(), src_c.strip(), dst_t.strip(), dst_c.strip()


def _extract_fk_relations(
    metadata: Optional[Dict[str, Any]],
) -> List[Tuple[str, str, str, str]]:
    """metadata 로부터 FK relation list 추출.

    우선순위:
    1) metadata['fk_to_id'] (Dict[str, int]) 의 keys
    2) metadata['fk_descriptions'] (Dict[str, str]) 의 keys
    3) metadata['node_metadata'] (Dict[int, str]) 내 '->' 포함 entries

    Returns:
        List of (src_table, src_col, dst_table, dst_col) tuples. 중복 제거됨.
    """
    if not metadata or not isinstance(metadata, dict):
        return []
    candidates: List[str] = []
    fk_to_id = metadata.get("fk_to_id")
    if isinstance(fk_to_id, dict) and fk_to_id:
        candidates.extend([k for k in fk_to_id.keys() if isinstance(k, str)])
    if not candidates:
        fk_desc = metadata.get("fk_descriptions")
        if isinstance(fk_desc, dict) and fk_desc:
            candidates.extend([k for k in fk_desc.keys() if isinstance(k, str)])
    if not candidates:
        node_meta = metadata.get("node_metadata")
        if isinstance(node_meta, dict) and node_meta:
            for v in node_meta.values():
                if isinstance(v, str) and "->" in v:
                    candidates.append(v)

    out: List[Tuple[str, str, str, str]] = []
    seen: set = set()
    for k in candidates:
        parsed = _parse_fk_key(k)
        if parsed is None:
            continue
        if parsed in seen:
            continue
        seen.add(parsed)
        out.append(parsed)
    return out


def _filter_fk_by_extracted(
    fk_relations: List[Tuple[str, str, str, str]],
    extracted_subgraph: Dict[str, List[str]],
) -> List[Tuple[str, str, str, str]]:
    """extracted_subgraph 내 두 column 이 모두 포함된 FK 만 retain.

    LLM context window 보호 (RFP §3.10 prompt 길이 risk 정합) + relevance.
    """
    if not fk_relations or not extracted_subgraph:
        return []
    table_cols: Dict[str, set] = {
        t: set(cols or []) for t, cols in extracted_subgraph.items()
    }
    out: List[Tuple[str, str, str, str]] = []
    for (st, sc, dt, dc) in fk_relations:
        if st not in table_cols or dt not in table_cols:
            continue
        if sc not in table_cols[st]:
            continue
        if dc not in table_cols[dt]:
            continue
        out.append((st, sc, dt, dc))
    return out


# ---------------------------------------------------------------------------
# Hint string formatting
# ---------------------------------------------------------------------------
def _format_fk_hints(
    fk_relations: List[Tuple[str, str, str, str]],
    fk_hint_format: str,
) -> str:
    """FK relation list → hint block string. format ∈ {explicit, compact}.

    explicit: "FK: src_t.src_c -> dst_t.dst_c"  (RFP §3.7 예시 정합)
    compact:  "[FK src_t.src_c=dst_t.dst_c]"     (sweep 위 본 작업 spec 정합)
    """
    if not fk_relations:
        return ""
    lines: List[str] = []
    for (st, sc, dt, dc) in fk_relations:
        if fk_hint_format == "explicit":
            lines.append(f"FK: {st}.{sc} -> {dt}.{dc}")
        elif fk_hint_format == "compact":
            lines.append(f"[FK {st}.{sc}={dt}.{dc}]")
        else:
            # 안전 fallback (validated upstream — 도달 안 함)
            lines.append(f"FK: {st}.{sc} -> {dt}.{dc}")
    header = "# Foreign Key Relationships"
    return header + "\n" + "\n".join(lines)


def _per_table_fk_hints(
    fk_relations: List[Tuple[str, str, str, str]],
    fk_hint_format: str,
) -> Dict[str, List[str]]:
    """Per-table dict for inline injection.

    Returns: {table_name: [hint_line, ...]} — table_name 은 src_table 기준.
    한 FK 는 src table block 만 inline 추가 (중복 방지).
    """
    by_table: Dict[str, List[str]] = {}
    for (st, sc, dt, dc) in fk_relations:
        if fk_hint_format == "explicit":
            line = f"FK: {st}.{sc} -> {dt}.{dc}"
        elif fk_hint_format == "compact":
            line = f"[FK {st}.{sc}={dt}.{dc}]"
        else:
            line = f"FK: {st}.{sc} -> {dt}.{dc}"
        by_table.setdefault(st, []).append(line)
    return by_table


def _insert_fk_hints_inline(
    base_schema: str,
    per_table_hints: Dict[str, List[str]],
) -> str:
    """base schema 의 각 '# Table: <name>' block 직후 (table 의 Columns 줄 뒤) 위 hint 삽입.

    base_schema format (XiYanFilter._build_mschema_with_values 정합):
        # Table: <tbl>
          Columns: c1 (Examples: ...) | c2 ...
        # Table: <tbl2>
          Columns: ...
    """
    if not per_table_hints:
        return base_schema
    lines = base_schema.split("\n")
    out_lines: List[str] = []
    current_table: Optional[str] = None
    pending_for_current: List[str] = []

    def flush_pending() -> None:
        nonlocal pending_for_current
        if pending_for_current:
            out_lines.extend(pending_for_current)
            pending_for_current = []

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("# Table:"):
            # 새 table block 진입 — 이전 table 의 pending hints flush
            flush_pending()
            current_table = stripped[len("# Table:"):].strip()
            out_lines.append(line)
            # 이 table 의 hint 가 있으면 columns 라인 뒤 추가 위 보관
            hints = per_table_hints.get(current_table, [])
            pending_for_current = []  # reset; injection 는 Columns 줄 뒤에서
            # marker 로 hint pending — Columns 줄 발견 시 hint 삽입
            # 구현 단순화: Columns 줄 처리 path 위 self-injection 표시 위해 별도
            # 변수 hint_to_inject 사용
            setattr(flush_pending, "_pending_hint", hints)
        elif stripped.startswith("Columns:"):
            out_lines.append(line)
            hints = getattr(flush_pending, "_pending_hint", []) or []
            for h in hints:
                out_lines.append("  " + h)
            setattr(flush_pending, "_pending_hint", [])
        else:
            out_lines.append(line)
    flush_pending()
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
VALID_FK_HINT_FORMATS: Tuple[str, ...] = ("none", "explicit", "compact")
VALID_FK_HINT_POSITIONS: Tuple[str, ...] = ("inline", "prefix", "suffix")


def serialize_schema_with_fk_hints(
    base_schema: str,
    extracted_subgraph: Dict[str, List[str]],
    metadata: Optional[Dict[str, Any]] = None,
    fk_hint_format: str = "none",
    fk_hint_position: str = "inline",
) -> str:
    """LLM filter prompt 위 전달될 schema 직렬화 + 선택적 FK 힌트.

    Args:
        base_schema: 기존 직렬화 결과 (XiYanFilter._build_mschema_with_values
                     output). FK hint 추가 base. 본 함수가 base 를 새로 만들지
                     않음 — 외부 호출자가 미리 만들어 전달.
        extracted_subgraph: {table_name: [col_name, ...]} — extractor output.
                            metadata['fk_to_id'] 내 FK 중 본 subgraph 위 양쪽
                            column 모두 존재하는 FK 만 hint 위 포함.
        metadata: builder metadata dict (fk_to_id / fk_descriptions /
                  node_metadata 중 한 곳에서 FK 추출).
        fk_hint_format: "none" | "explicit" | "compact".
            - none: base_schema 그대로 반환 (baseline 호환).
            - explicit: "FK: src_t.src_c -> dst_t.dst_c"
            - compact: "[FK src_t.src_c=dst_t.dst_c]"
        fk_hint_position: "inline" | "prefix" | "suffix".
            - inline: 각 table 의 Columns 줄 다음 hint 삽입 (src_table 기준)
            - prefix: base_schema 앞 FK block
            - suffix: base_schema 뒤 FK block

    Returns:
        Hint 가 추가된 schema string. fk_hint_format=='none' 이면 base_schema
        identical. extracted_subgraph 자체는 mutate 되지 않음.
    """
    if fk_hint_format not in VALID_FK_HINT_FORMATS:
        raise ValueError(
            f"fk_hint_format='{fk_hint_format}' invalid. "
            f"Expected one of {VALID_FK_HINT_FORMATS}."
        )
    if fk_hint_position not in VALID_FK_HINT_POSITIONS:
        raise ValueError(
            f"fk_hint_position='{fk_hint_position}' invalid. "
            f"Expected one of {VALID_FK_HINT_POSITIONS}."
        )
    if fk_hint_format == "none":
        return base_schema

    fk_all = _extract_fk_relations(metadata)
    fk_in_scope = _filter_fk_by_extracted(fk_all, extracted_subgraph)
    if not fk_in_scope:
        return base_schema

    if fk_hint_position == "prefix":
        hint_block = _format_fk_hints(fk_in_scope, fk_hint_format)
        return f"{hint_block}\n\n{base_schema}"
    elif fk_hint_position == "suffix":
        hint_block = _format_fk_hints(fk_in_scope, fk_hint_format)
        return f"{base_schema}\n\n{hint_block}"
    else:  # inline
        per_table = _per_table_fk_hints(fk_in_scope, fk_hint_format)
        return _insert_fk_hints_inline(base_schema, per_table)
