"""Direction C — `inferred_fk` 예측 (GPT-4.1-mini).

근거:
  - planning/DECISIONS.md 2026-05-14 (Direction C inferred_fk Analyzer 핸드오프 정식 launch)
  - planning/filter_proposal_data_spec_2026-05-13.md §C-1 inferred_fk_count 필드 정의
  - planning/filter_proposal_by_scholar_agent_phase2_2026-05-13.md §2 (학술 Agent Phase 3 권고)
  - src/modules/filters/grast_fd_filter.py commit e90d91a (inferred_fk 인자 schema)

목적:
  debit_card_specializing + card_games 의 declared FK 의 sparse 한 영역에서
  GPT-4.1-mini 기반으로 missing FK 후보 예측. GRASTFDFilter sweep launch 의 prerequisite.

LLM spec:
  - Provider: openai (env OPENAI_API_KEY)
  - Model: gpt-4.1-mini
  - Temperature: 0.0 (deterministic)
  - APIClient 재사용 (token usage tracking 포함)

Inferred FK 정합성 검증 (per-candidate):
  (a) LLM confidence (high/medium/low, prompt 출력)
  (b) Column type 일치 (dev_tables.json column_types)
  (c) Naming convention (suffix 'Id'/'ID'/'_id' + 의미적 유사도)
  (d) Schema graph plausibility (declared FK 와의 graph 정합)

Output:
  - notebooks/analysis_results/direction_c_inferred_fk.md (analyzer 보고서)
  - outputs/analysis/direction_c_inferred_fk.yaml (GRASTFDFilter ingestion form)
  - outputs/analysis/direction_c_infer_fk_raw.json (LLM raw output + per-candidate validation)

Usage:
  PYTHONPATH=src OPENAI_API_KEY=... conda run -n base python src/analysis/direction_c_infer_fk.py
  PYTHONPATH=src OPENAI_API_KEY=... conda run -n base python src/analysis/direction_c_infer_fk.py --smoke  # dry-run 1 DB
"""

from __future__ import annotations

import os
import re
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Auto-load .env (project convention, src/main.py 와 정합)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from llm_client.api_handler import APIClient
from utils.logger import get_logger

logger = get_logger(__name__)

DEV_TABLES = ROOT / "data/raw/BIRD_dev/dev_tables.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/analysis"
DEFAULT_REPORT_PATH = ROOT / "notebooks/analysis_results/direction_c_inferred_fk.md"

TARGET_DBS = ["debit_card_specializing", "card_games"]
MODEL_NAME = "gpt-4.1-mini"


# ──────────────────────────────────────────────────────────────
# Schema extraction
# ──────────────────────────────────────────────────────────────

def load_db_schema(dev_tables_path: Path, db_id: str) -> Dict[str, Any]:
    """dev_tables.json 에서 단일 DB 의 schema 추출."""
    with open(dev_tables_path) as f:
        data = json.load(f)
    for d in data:
        if d["db_id"] == db_id:
            return d
    raise ValueError(f"db_id {db_id} not found in {dev_tables_path}")


def schema_to_text(db_info: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str, str, bool]]]:
    """DB schema → human-readable text + per-col list (table.col, type, is_pk)."""
    cn = db_info["column_names_original"]
    ct = db_info["column_types"]
    tnames = db_info["table_names_original"]
    pks_raw = db_info.get("primary_keys", []) or []
    fks = db_info.get("foreign_keys", []) or []

    pk_set: Set[int] = set()
    for pk in pks_raw:
        if isinstance(pk, list):
            for p in pk:
                pk_set.add(p)
        else:
            pk_set.add(pk)

    lines = []
    cols_list: List[Tuple[str, str, str, bool]] = []  # (table, col, type, is_pk)

    # Group columns by table
    by_table: Dict[str, List[Tuple[str, str, bool]]] = {t: [] for t in tnames}
    for i, (ti, c) in enumerate(cn):
        if ti < 0:
            continue
        tn = tnames[ti]
        ty = ct[i] if i < len(ct) else "?"
        ispk = i in pk_set
        by_table[tn].append((c, ty, ispk))
        cols_list.append((tn, c, ty, ispk))

    for tn in tnames:
        lines.append(f"Table: {tn}")
        for c, ty, ispk in by_table[tn]:
            pk_tag = " [PK]" if ispk else ""
            lines.append(f"  {c}: {ty}{pk_tag}")
        lines.append("")

    # Declared FK
    lines.append("Declared foreign keys:")
    if fks:
        for src, dst in fks:
            tsrc, csrc = cn[src]
            tdst, cdst = cn[dst]
            lines.append(f"  {tnames[tsrc]}.{csrc} -> {tnames[tdst]}.{cdst}")
    else:
        lines.append("  (none)")

    return "\n".join(lines), cols_list


def declared_fk_keys(db_info: Dict[str, Any]) -> Set[str]:
    """선언된 FK 를 'src_tbl.src_col->dst_tbl.dst_col' set 으로 반환."""
    cn = db_info["column_names_original"]
    tnames = db_info["table_names_original"]
    out: Set[str] = set()
    for src, dst in db_info.get("foreign_keys", []) or []:
        tsrc, csrc = cn[src]
        tdst, cdst = cn[dst]
        out.add(f"{tnames[tsrc]}.{csrc}->{tnames[tdst]}.{cdst}")
    return out


def col_type_lookup(db_info: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    """(table_lower, col_lower) → type."""
    cn = db_info["column_names_original"]
    ct = db_info["column_types"]
    tnames = db_info["table_names_original"]
    out: Dict[Tuple[str, str], str] = {}
    for i, (ti, c) in enumerate(cn):
        if ti < 0:
            continue
        out[(tnames[ti].lower(), c.lower())] = ct[i] if i < len(ct) else "?"
    return out


# ──────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────

def build_prompt(db_id: str, schema_text: str) -> List[Dict[str, str]]:
    """GPT-4.1-mini 호출용 messages list 구성.
    System + user 분리, few-shot 1 example (confidence 등급 정합).
    """
    system = (
        "You are an expert database schema designer. "
        "Given a SQLite database schema with explicit primary keys (PK) and a partial list of "
        "declared foreign keys (FK), your task is to identify MISSING (implicit) foreign-key "
        "relationships that the original schema author likely intended but forgot to declare. "
        "Use ONLY the column names, types, primary-key markers, and table groupings provided. "
        "Do NOT invent columns that don't exist. "
        "For each candidate inferred FK, output exactly ONE line in this format:\n"
        "  src_table.src_column -> dst_table.dst_column [confidence: high/medium/low] reason: <short reason>\n"
        "Conventions:\n"
        "  - src side = the 'foreign' (referencing) column\n"
        "  - dst side = the 'primary' (referenced) column (usually a PK of another table)\n"
        "  - src_column type should match dst_column type\n"
        "  - common naming hints: 'ID', 'Id', '_id' suffix indicates a candidate FK\n"
        "  - high confidence: type+PK+naming all match, semantically obvious\n"
        "  - medium: 2 of 3 match\n"
        "  - low: 1 of 3 match or naming ambiguity\n"
        "Skip relationships that are already declared (shown below). "
        "If a column refers to multiple candidates, output all and rank by confidence. "
        "Output ONLY the FK lines, no preamble, no markdown."
    )

    # Few-shot example (small schema example for grounding)
    fewshot_example = (
        "Example input (an unrelated minimal schema):\n"
        "Table: users\n  user_id: integer [PK]\n  name: text\n\n"
        "Table: orders\n  order_id: integer [PK]\n  user_id: integer\n  product_id: integer\n\n"
        "Table: products\n  product_id: integer [PK]\n  name: text\n\n"
        "Declared foreign keys:\n  (none)\n\n"
        "Example output:\n"
        "  orders.user_id -> users.user_id [confidence: high] reason: PK match, naming 'user_id' identical, type integer match\n"
        "  orders.product_id -> products.product_id [confidence: high] reason: PK match, naming identical, type integer match\n"
    )

    user = (
        f"{fewshot_example}\n"
        f"---\n\n"
        f"Now infer missing FKs for DB '{db_id}':\n\n"
        f"{schema_text}\n\n"
        f"Output ONLY inferred FK lines (no preamble, no markdown)."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ──────────────────────────────────────────────────────────────
# LLM call
# ──────────────────────────────────────────────────────────────

def call_gpt4_mini(api_client: APIClient, messages: List[Dict[str, str]],
                    model: str = MODEL_NAME, temperature: float = 0.0,
                    max_tokens: int = 2048, timeout: float = 120.0) -> str:
    """OpenAI API call via APIClient (provider='openai')."""
    response = api_client.client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    APIClient._record_usage(model, getattr(response, "usage", None))
    return (response.choices[0].message.content or "").strip()


# ──────────────────────────────────────────────────────────────
# Output parsing
# ──────────────────────────────────────────────────────────────

FK_LINE_RE = re.compile(
    r"^\s*[-•*]?\s*"
    r"([A-Za-z_][\w]*)\s*\.\s*([A-Za-z_][\w]*)"
    r"\s*->\s*"
    r"([A-Za-z_][\w]*)\s*\.\s*([A-Za-z_][\w]*)"
    r"\s*\[?\s*confidence\s*:\s*(high|medium|low)\s*\]?"
    r"\s*(?:reason\s*:\s*(.+?))?$",
    re.IGNORECASE,
)


def parse_fk_lines(raw: str) -> List[Dict[str, Any]]:
    """LLM raw output → list of {src_tbl, src_col, dst_tbl, dst_col, confidence, reason}."""
    out: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = FK_LINE_RE.match(line)
        if not m:
            continue
        src_tbl, src_col, dst_tbl, dst_col, conf, reason = m.groups()
        out.append({
            "src_tbl": src_tbl,
            "src_col": src_col,
            "dst_tbl": dst_tbl,
            "dst_col": dst_col,
            "confidence": conf.lower(),
            "reason": (reason or "").strip(),
            "key": f"{src_tbl}.{src_col}->{dst_tbl}.{dst_col}",
        })
    return out


# ──────────────────────────────────────────────────────────────
# Per-candidate validation
# ──────────────────────────────────────────────────────────────

NAMING_SUFFIX_HINTS = ("id", "ID", "Id", "_id", "_ID", "Id$", "ID$")


def col_exists(cols_list: List[Tuple[str, str, str, bool]], tbl: str, col: str) -> bool:
    """schema 에 (tbl, col) 존재 여부 (case-insensitive)."""
    return any(t.lower() == tbl.lower() and c.lower() == col.lower()
                for t, c, _, _ in cols_list)


def col_type(cols_list: List[Tuple[str, str, str, bool]], tbl: str, col: str) -> Optional[str]:
    for t, c, ty, _ in cols_list:
        if t.lower() == tbl.lower() and c.lower() == col.lower():
            return ty
    return None


def col_is_pk(cols_list: List[Tuple[str, str, str, bool]], tbl: str, col: str) -> bool:
    for t, c, _, ispk in cols_list:
        if t.lower() == tbl.lower() and c.lower() == col.lower():
            return ispk
    return False


def naming_match_score(src_col: str, dst_col: str, dst_tbl: str) -> Tuple[bool, str]:
    """Naming convention 정합 — suffix 패턴 + 의미적 유사도.

    True 케이스:
      - 동일 col name (e.g., user_id -> user_id)
      - src 가 dst_tbl + Id suffix (e.g., transactions.CustomerID -> customers.CustomerID)
      - src 에 dst_col 포함 (substring containment)
    """
    s = src_col.lower()
    d = dst_col.lower()
    dt = dst_tbl.lower()

    if s == d:
        return True, "identical column name"
    if any(suf in s for suf in ("id", "_id")) and any(suf in d for suf in ("id", "_id")):
        if d in s or s.endswith(d):
            return True, f"suffix match: '{src_col}' contains '{dst_col}'"
    # dst_tbl + 'ID' pattern (e.g., dst_tbl='customers', src_col='CustomerID')
    dt_singular = dt[:-1] if dt.endswith("s") else dt
    if dt_singular in s.lower():
        return True, f"src_col '{src_col}' contains dst_tbl base '{dt_singular}'"
    return False, "no naming pattern match"


def validate_candidate(
    cand: Dict[str, Any],
    cols_list: List[Tuple[str, str, str, bool]],
    declared: Set[str],
) -> Dict[str, Any]:
    """4 axis validation → retained / filtered + reasoning."""
    src_tbl, src_col = cand["src_tbl"], cand["src_col"]
    dst_tbl, dst_col = cand["dst_tbl"], cand["dst_col"]
    key = cand["key"]
    reverse_key = f"{dst_tbl}.{dst_col}->{src_tbl}.{src_col}"

    checks: Dict[str, Any] = {
        "key": key,
        "llm_confidence": cand["confidence"],
        "llm_reason": cand["reason"],
    }

    # (0) Schema existence — column 이 실제로 schema 에 있는가
    src_exists = col_exists(cols_list, src_tbl, src_col)
    dst_exists = col_exists(cols_list, dst_tbl, dst_col)
    checks["src_exists"] = src_exists
    checks["dst_exists"] = dst_exists
    if not src_exists or not dst_exists:
        checks["retained"] = False
        checks["filter_reason"] = "schema mismatch (column not in dev_tables.json)"
        return checks

    # (a) Type match
    src_ty = col_type(cols_list, src_tbl, src_col)
    dst_ty = col_type(cols_list, dst_tbl, dst_col)
    type_match = src_ty == dst_ty
    checks["src_type"] = src_ty
    checks["dst_type"] = dst_ty
    checks["type_match"] = type_match

    # (b) PK match (dst should be PK)
    dst_pk = col_is_pk(cols_list, dst_tbl, dst_col)
    checks["dst_is_pk"] = dst_pk

    # (c) Naming convention
    name_ok, name_reason = naming_match_score(src_col, dst_col, dst_tbl)
    checks["naming_match"] = name_ok
    checks["naming_reason"] = name_reason

    # (d) Declared duplicate (skip already-declared)
    declared_dup = key in declared or reverse_key in declared
    checks["declared_duplicate"] = declared_dup

    # (e) Self-reference (src_tbl == dst_tbl, may be valid 단 caveat 필요)
    self_ref = src_tbl.lower() == dst_tbl.lower()
    checks["self_reference"] = self_ref

    # Retention criteria (4-axis validation, 학술 Agent Phase 3 권고 정합):
    # - Reject: declared duplicate, type mismatch
    # - Reject: LLM confidence=low AND naming_match=False (LLM 자체 uncertainty + no naming signal)
    # - Accept: dst_is_pk AND (type_match OR (LLM high + naming match))
    if declared_dup:
        checks["retained"] = False
        checks["filter_reason"] = "duplicate of declared FK"
    elif not type_match:
        checks["retained"] = False
        checks["filter_reason"] = f"type mismatch ({src_ty} != {dst_ty})"
    elif cand["confidence"] == "low" and not name_ok:
        # LLM 자체 confidence low + naming 신호 없음 → 신뢰도 부족
        checks["retained"] = False
        checks["filter_reason"] = "LLM low-confidence + no naming match (likely false positive)"
    elif not dst_pk:
        # type 일치 단 dst 가 PK 아님 — composite or non-PK reference 가능
        # LLM confidence high + naming match 시 유지
        if cand["confidence"] == "high" and name_ok:
            checks["retained"] = True
            checks["filter_reason"] = "non-PK dst but high confidence + naming match"
        else:
            checks["retained"] = False
            checks["filter_reason"] = "dst is not PK + low/medium confidence"
    else:
        # 모두 OK
        checks["retained"] = True
        checks["filter_reason"] = "all checks passed"

    # Composite validation score (정합성 강도 측정)
    score = 0
    if checks.get("type_match"): score += 1
    if checks.get("dst_is_pk"): score += 1
    if checks.get("naming_match"): score += 1
    score_map = {"high": 1, "medium": 0.5, "low": 0}
    score += score_map.get(cand["confidence"], 0)
    checks["validation_score"] = round(score, 2)  # max 4.0
    return checks


# ──────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────

def run_for_db(db_id: str, db_info: Dict[str, Any], api_client: APIClient,
                model: str, output_dir: Path, smoke: bool = False) -> Dict[str, Any]:
    """단일 DB 의 inferred FK 예측 + validation."""
    schema_text, cols_list = schema_to_text(db_info)
    declared = declared_fk_keys(db_info)

    messages = build_prompt(db_id, schema_text)
    logger.info(f"[{db_id}] declared FK = {len(declared)}, schema text {len(schema_text)} chars")

    if smoke:
        # Dry-run: print prompt only
        print(f"\n=== SMOKE: {db_id} prompt ===")
        print(messages[1]["content"][:2000])
        print("...")
        print("=== END SMOKE prompt ===\n")
        return {"db_id": db_id, "smoke": True}

    raw_output = call_gpt4_mini(api_client, messages, model=model)
    logger.info(f"[{db_id}] raw output {len(raw_output)} chars")

    candidates = parse_fk_lines(raw_output)
    logger.info(f"[{db_id}] parsed {len(candidates)} candidate FKs")

    # Validate each
    validated = []
    retained = []
    filtered = []
    for cand in candidates:
        v = validate_candidate(cand, cols_list, declared)
        validated.append(v)
        if v["retained"]:
            retained.append(v["key"])
        else:
            filtered.append((v["key"], v["filter_reason"]))

    return {
        "db_id": db_id,
        "declared_fk_count": len(declared),
        "declared_fk": sorted(declared),
        "raw_output": raw_output,
        "parsed_candidates": candidates,
        "validated": validated,
        "retained_fk": sorted(set(retained)),
        "filtered_fk": filtered,
        "messages_user_excerpt": messages[1]["content"][:500],
    }


def write_yaml_output(results: Dict[str, Dict], output_path: Path) -> None:
    """GRASTFDFilter ingestion format — Dict[db_id, List[str]]."""
    yaml_data = {
        db_id: results[db_id]["retained_fk"]
        for db_id in results
        if not results[db_id].get("smoke")
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Comment header
        f.write("# Direction C — Inferred FK (GPT-4.1-mini, 2026-05-14)\n")
        f.write("# Schema: GRASTFDFilter.params.inferred_fk (commit e90d91a)\n")
        f.write("# Format: Dict[db_id, List[\"src_tbl.src_col->dst_tbl.dst_col\"]]\n")
        f.write("# Source: src/analysis/direction_c_infer_fk.py\n")
        f.write("\n")
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=True, allow_unicode=True)


def write_raw_json(results: Dict[str, Dict], output_path: Path) -> None:
    """Raw output + validation details (analyzer 보고서 base)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbs", nargs="+", default=TARGET_DBS,
                        help="대상 DB list (default: debit_card + card_games)")
    parser.add_argument("--smoke", action="store_true",
                        help="Dry-run: prompt + LLM call 없음 (1 DB)")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Validate env
    if not args.smoke and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY env var 필수 (smoke 모드 아니면)")

    # APIClient (openai provider)
    api_client = None
    if not args.smoke:
        api_client = APIClient(provider="openai")
        logger.info(f"Initialized OpenAI APIClient, model={args.model}")

    # Run per DB
    results: Dict[str, Dict] = {}
    target_dbs = args.dbs[:1] if args.smoke else args.dbs
    for db_id in target_dbs:
        db_info = load_db_schema(DEV_TABLES, db_id)
        r = run_for_db(db_id, db_info, api_client, args.model, output_dir, smoke=args.smoke)
        results[db_id] = r

    if args.smoke:
        print("\n=== SMOKE done. To run full: omit --smoke ===")
        return

    # Write outputs
    yaml_path = output_dir / "direction_c_inferred_fk.yaml"
    write_yaml_output(results, yaml_path)
    logger.info(f"YAML output: {yaml_path}")

    raw_path = output_dir / "direction_c_infer_fk_raw.json"
    write_raw_json(results, raw_path)
    logger.info(f"Raw + validation: {raw_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Direction C Inferred FK Summary")
    print("=" * 60)
    for db_id, r in results.items():
        print(f"\n=== {db_id} ===")
        print(f"  Declared FK: {r['declared_fk_count']}")
        print(f"  Candidates parsed: {len(r['parsed_candidates'])}")
        print(f"  Retained (passed validation): {len(r['retained_fk'])}")
        print(f"  Filtered: {len(r['filtered_fk'])}")
        for fk in r["retained_fk"]:
            print(f"    + {fk}")
        for k, reason in r["filtered_fk"]:
            print(f"    - {k}  ({reason})")
    print(f"\nToken usage: {json.dumps(APIClient.get_usage_summary(), indent=2)}")


if __name__ == "__main__":
    main()
