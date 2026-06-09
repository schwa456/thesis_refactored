"""Spider 2.0-Lite DDL.csv ingest 어댑터 — G-S2-1 generalization wave.

DECISIONS 2026-06-09 #3 §G-S2-0 (2026-06-10 data-check 완료) + G-S2-1 spec 정합.

`data/Spider2/spider2-lite/resource/databases/{bigquery,snowflake,sqlite}/<db>/DDL.csv`
전 backend uniform 포맷 (CREATE TABLE 문) → 테이블/컬럼/타입/FK 추출 →
HeteroData + metadata (BIRD `EnrichedHeteroGraphBuilder` 인터페이스 정합).

핵심 정합:
- BIRD HeteroGraphBuilder 의 (HeteroData, metadata_dict) 계약 보존 — table_to_id /
  col_to_id / fk_to_id / node_metadata / edges / edge_types / fk_reachability /
  schema_diameter 모두 동일 키. downstream selector/extractor/filter 무변경 호환.
- FK 는 sqlite/snowflake 일부만 추출 가능 (bigquery 위 FK 없음 — column-table
  edge 위주). 없으면 has_column / belongs_to + table_to_table 비활성.
- per-table `<table>.json` (sample_rows) 선택 활용 — 현 v1 위 skip (스키마 다양,
  v2 위 column-level enrich 검토).

instance_id → backend 매핑 (spider2-lite.jsonl 의 instance_id prefix):
- `bq*` / `ga*`  → bigquery (multi-inner: `<db>/<inner_1>/DDL.csv`, ...)
- `sf*` / `sf_bq*` → snowflake (`<db>/<inner>/DDL.csv`)
- `local*` → sqlite (`<db>/DDL.csv`, no inner)

★ Enterprise-scale (bq/sf 수백~수천 컬럼) 핸들링:
- `max_columns` 초과 위 RuntimeError 발생 (caller 가 try/except 위 skip + log)
- PLM batch_size 명시 (default 256 — 대량 column 위 메모리 spike 방지)
- DDL 파싱 실패 위 sqlglot → regex fallback → graceful skip + parse_errors metadata
- 0 column DB 위 RuntimeError (empty graph 방지)

cache 분리: `extra_cache_suffix = "_spider2"`.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

import sqlglot
from sqlglot import expressions as sqlexp

from modules.registry import register
from modules.builders.graph_builder import HeteroGraphBuilder
from utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_SPIDER2_ROOT = Path("data/Spider2/spider2-lite/resource/databases")
DEFAULT_MAX_COLUMNS = 5000      # enterprise-scale 위 skip 임계 (~수천 컬럼)
DEFAULT_PLM_BATCH_SIZE = 256

# spider2-lite.jsonl 의 instance_id prefix → backend directory 매핑
# 사례 (DECISIONS 2026-06-09 #3 §G-S2-0 + 실측):
#   bq001~bq134 → bigquery
#   ga001~ga014 → bigquery (Google Analytics 데이터)
#   sf001~sf084 → snowflake
#   sf_bq001~  → snowflake (BigQuery 원본 → Snowflake 이관)
#   local001~  → sqlite (Spider 2.0 localdb 데이터)
INSTANCE_PREFIX_TO_BACKEND: Dict[str, str] = {
    "sf_bq": "snowflake",   # 더 긴 prefix 우선 (sf_bq 가 sf 보다 먼저 매칭)
    "bq":    "bigquery",
    "ga":    "bigquery",
    "sf":    "snowflake",
    "local": "sqlite",
}


# --------------------------------------------------------------------------- #
# DDL.csv parser
# --------------------------------------------------------------------------- #

def _parse_ddl_csv(csv_path: Path, dialect: str) -> Dict[str, Any]:
    """단일 DDL.csv 파일 위 CREATE TABLE 문 추출.

    Returns:
        {
            "tables": List[str] (short name 위),
            "columns": Dict[short_name, List[{"name", "type", "samples"}]],
            "foreign_keys": List[{"from_table", "from_column", "to_table", "to_column"}],
            "parse_errors": List[{"table", "error"}],
        }

    CSV 헤더 분기:
        bigquery: `table_name,ddl` (lowercase ddl)
        snowflake: `table_name,description,DDL` (description 컬럼 추가)
        sqlite: `table_name,DDL`
    """
    tables: List[str] = []
    columns_dict: Dict[str, List[Dict[str, Any]]] = {}
    fks: List[Dict[str, str]] = []
    parse_errors: List[Dict[str, str]] = []

    try:
        f = open(csv_path, encoding="utf-8-sig", errors="replace", newline="")
    except OSError as e:
        return {
            "tables": [], "columns": {}, "foreign_keys": [],
            "parse_errors": [{"table": "<file>", "error": f"open: {e}"}],
        }

    with f:
        try:
            reader = csv.DictReader(f)
        except csv.Error as e:
            return {
                "tables": [], "columns": {}, "foreign_keys": [],
                "parse_errors": [{"table": "<file>", "error": f"csv reader: {e}"}],
            }

        for row in reader:
            # 헤더 분기 — 대/소문자 모두 시도
            ddl_str = (
                row.get("ddl") or row.get("DDL") or row.get("Ddl") or ""
            ).strip()
            tbl_name = (row.get("table_name") or "").strip()
            if not tbl_name or not ddl_str:
                # snowflake 위 description 만 있고 DDL 비어있는 row 흔함 (정상 skip)
                continue

            # full namespace 위 마지막 segment 가 짧은 table 명
            # 예: "spider2-public-data.1000_genomes.sample_info" → "sample_info"
            #     "FINANCE__ECONOMICS.CYBERSYN.AIRCRAFT_INDEX" → "AIRCRAFT_INDEX"
            short_name = tbl_name.split(".")[-1]

            cols, table_fks, err = _parse_create_table(
                ddl_str, short_name, dialect=dialect
            )
            if err:
                parse_errors.append({"table": short_name, "error": err})
                # regex fallback 시도
                cols, table_fks = _regex_extract_columns(ddl_str)

            if not cols:
                # 둘 다 실패 시 빈 column → 다음 row
                continue

            tables.append(short_name)
            columns_dict[short_name] = cols
            fks.extend(table_fks)

    return {
        "tables": tables,
        "columns": columns_dict,
        "foreign_keys": fks,
        "parse_errors": parse_errors,
    }


def _parse_create_table(
    ddl: str, table_name: str, dialect: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], Optional[str]]:
    """sqlglot 위 CREATE TABLE → (column list, FK list, error or None).

    sqlglot 위 dialect 별 parse (bigquery / snowflake / sqlite). ARRAY<...> /
    STRUCT<...> 같은 BigQuery nested type 도 ColumnDef 로 인식.
    """
    try:
        tree = sqlglot.parse_one(ddl, dialect=dialect)
    except Exception as e:
        return [], [], f"sqlglot parse error: {type(e).__name__}"

    if tree is None:
        return [], [], "sqlglot returned None"

    # top-level Schema.expressions 만 추출 — nested STRUCT/ARRAY 안 ColumnDef 무시
    # (BigQuery 위 STRUCT<DP STRING, ...> 위 DP 가 top-level DP 와 중복되는 사례 방지).
    schema_obj = tree.find(sqlexp.Schema)
    if schema_obj is None:
        # CREATE TABLE 위 Schema 없는 경우 (rare, e.g., CREATE TABLE AS SELECT)
        return [], [], "no Schema in CREATE TABLE"

    cols: List[Dict[str, Any]] = []
    seen_col_names: set = set()
    for cdef in schema_obj.expressions:
        if not isinstance(cdef, sqlexp.ColumnDef):
            continue  # FK / PK / Constraint 등 별도 처리
        cname = cdef.name
        if cname in seen_col_names:
            continue  # 중복 column 명 silently skip (downstream dict 무결성)
        seen_col_names.add(cname)
        ctype = ""
        ckind = cdef.args.get("kind")
        if ckind is not None:
            try:
                ctype = ckind.sql(dialect=dialect)
            except Exception:
                ctype = str(ckind)
            # 너무 긴 nested type 위 truncate (PLM input 보호)
            if len(ctype) > 200:
                ctype = ctype[:200] + "..."
        cols.append({"name": cname, "type": ctype, "samples": []})

    # FK 도 top-level Schema.expressions 위 ForeignKey 만 (nested 위 일반 없음)
    fks: List[Dict[str, str]] = []
    for fk in schema_obj.expressions:
        if not isinstance(fk, sqlexp.ForeignKey):
            continue
        # from_columns = FK 의 expressions (current table 위 컬럼 list)
        from_cols = []
        for e in fk.args.get("expressions", []) or []:
            if hasattr(e, "name") and e.name:
                from_cols.append(e.name)
        if not from_cols:
            continue
        # REFERENCES clause
        ref = fk.find(sqlexp.Reference)
        if ref is None:
            continue
        # parse "REFERENCES <tbl>(<col>, ...)" structure
        try:
            ref_sql = ref.sql(dialect=dialect)
        except Exception:
            continue
        # REFERENCES <tbl>(<col>) — to_col 위 greedy 매칭 (`id` 같은 짧은 이름
        # 위 char class 위 끝까지 모두 잡도록). `["`]?` 옵셔널 매칭 의 lazy
        # 매칭이 1글자만 잡는 회귀 방지.
        m = re.search(
            r'REFERENCES\s+["`]?([^\s(]+?)["`]?\s*\(\s*["`]?([^"`\s),]+)["`]?',
            ref_sql, re.IGNORECASE,
        )
        if not m:
            continue
        to_tbl = m.group(1).split(".")[-1]   # short name
        to_col = m.group(2)
        for fc in from_cols:
            fks.append({
                "from_table": table_name,
                "from_column": fc,
                "to_table": to_tbl,
                "to_column": to_col,
            })

    return cols, fks, None


def _regex_extract_columns(
    ddl: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """sqlglot 실패 시 단순 regex fallback (column 명 + type only, FK 미추출).

    CREATE TABLE 본문 (괄호 사이) 위 콤마 분리 후 column 명/타입 추정. nested
    paren / generics 위 depth 추적.
    """
    # 마지막 닫는 괄호 + ; 까지를 본문으로
    m = re.search(r"\((.+)\)\s*;?\s*$", ddl, re.DOTALL)
    if not m:
        return [], []
    body = m.group(1)

    # depth-aware comma split
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in body:
        if ch in "(<":
            depth += 1; cur.append(ch)
        elif ch in ")>":
            depth -= 1; cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())

    cols: List[Dict[str, Any]] = []
    SKIP_KEYWORDS = re.compile(
        r"^\s*(FOREIGN\s+KEY|PRIMARY\s+KEY|CONSTRAINT|UNIQUE|CHECK|INDEX)\b",
        re.IGNORECASE,
    )
    for p in parts:
        if SKIP_KEYWORDS.match(p):
            continue
        # column name + type
        m = re.match(
            r'\s*[`"]?([A-Za-z_][\w]*)[`"]?\s+(.+?)$',
            p, re.DOTALL,
        )
        if not m:
            continue
        ctype = m.group(2).strip()
        # 후행 column constraint 제거
        ctype = re.sub(
            r"\s+(NOT\s+NULL|PRIMARY\s+KEY|DEFAULT\s+.+|UNIQUE|AUTOINCREMENT|COLLATE\s+\S+).*$",
            "", ctype, flags=re.IGNORECASE | re.DOTALL,
        )
        if len(ctype) > 200:
            ctype = ctype[:200] + "..."
        cols.append({"name": m.group(1), "type": ctype.strip(), "samples": []})

    return cols, []


# --------------------------------------------------------------------------- #
# DB → DDL path resolver
# --------------------------------------------------------------------------- #

def infer_backend(db_id: str) -> Optional[str]:
    """instance_id (bq*/ga*/sf*/sf_bq*/local*) 위 prefix → backend.

    더 긴 prefix (sf_bq) 우선 매칭. 매칭 실패 시 None.
    """
    iid = db_id.lower()
    # 긴 prefix 우선
    for prefix in sorted(INSTANCE_PREFIX_TO_BACKEND.keys(), key=len, reverse=True):
        if iid.startswith(prefix):
            return INSTANCE_PREFIX_TO_BACKEND[prefix]
    return None


def resolve_spider2_db_paths(
    db_id: str,
    db_field: str,
    spider2_root: Path = DEFAULT_SPIDER2_ROOT,
) -> Tuple[str, List[Path]]:
    """instance_id + db_field → backend + DDL.csv path list.

    bigquery: <root>/bigquery/<db>/<inner_1>/DDL.csv, ..., <inner_n>/DDL.csv
              (multi-inner, austin 위 5 dataset 등)
    snowflake: <root>/snowflake/<db>/<inner>/DDL.csv (보통 single inner)
    sqlite:    <root>/sqlite/<db>/DDL.csv (no inner)

    Raises FileNotFoundError if backend / db dir / DDL.csv 미존재.
    """
    backend = infer_backend(db_id)
    if backend is None:
        raise ValueError(
            f"Cannot infer backend from instance_id '{db_id}' — "
            f"expected prefix in {sorted(INSTANCE_PREFIX_TO_BACKEND.keys())}"
        )

    db_dir = spider2_root / backend / db_field
    if not db_dir.exists():
        raise FileNotFoundError(
            f"Spider2 DB dir not found: {db_dir} "
            f"(backend={backend}, db_field={db_field!r})"
        )

    if backend == "sqlite":
        ddl = db_dir / "DDL.csv"
        if not ddl.exists():
            raise FileNotFoundError(f"DDL.csv not found: {ddl}")
        return backend, [ddl]

    # bigquery / snowflake — recursive search
    ddl_paths = sorted(db_dir.rglob("DDL.csv"))
    if not ddl_paths:
        raise FileNotFoundError(f"No DDL.csv found under {db_dir}")
    return backend, ddl_paths


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

@register("builder", "Spider2GraphBuilder")
class Spider2GraphBuilder(HeteroGraphBuilder):
    """Spider 2.0-Lite DDL.csv → HeteroData adapter.

    BIRD `EnrichedHeteroGraphBuilder` 와 동일 (HeteroData, metadata_dict) 인터페이스.
    Selector / Extractor / Filter 의 graph_data 소비 코드 무수정.

    Backend uniform: bigquery / snowflake / sqlite — 단일 어댑터 위 DDL.csv 파싱
    (sqlglot dialect 분기). FK 는 sqlite/snowflake 일부만 (bigquery 위 없음).

    Args:
        plm_model_name: PLM 모델 (default MiniLM-L6).
        max_columns: 한 DB 위 column 총합 초과 시 build() 가 RuntimeError raise
            (caller 위 try/except 위 skip).
        plm_batch_size: PLM encode batch size (메모리 제어).
        use_sample_json: per-table .json (sample data) 활용 여부 (v1 위 metadata
            노출만, column-level enrich 미적용).
        spider2_db_field_map: 외부 caller 위 instance_id → spider2-lite.jsonl 의
            db 필드 매핑을 미리 제공할 때 사용. None 시 build(spider2_db_field=...)
            인자 위 명시 또는 db_id 자체를 db_field 위 fallback.
    """

    extra_cache_suffix = "_spider2"

    def __init__(
        self,
        plm_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_columns: int = DEFAULT_MAX_COLUMNS,
        plm_batch_size: int = DEFAULT_PLM_BATCH_SIZE,
        use_sample_json: bool = True,
        spider2_db_field_map: Optional[Dict[str, str]] = None,
        add_t2t_edges: bool = True,
        **kwargs,
    ):
        super().__init__(
            plm_model_name=plm_model_name,
            add_t2t_edges=add_t2t_edges,
            **kwargs,
        )
        self.max_columns = int(max_columns)
        self.plm_batch_size = int(plm_batch_size)
        self.use_sample_json = bool(use_sample_json)
        self.spider2_db_field_map = dict(spider2_db_field_map or {})
        logger.info(
            f"Spider2GraphBuilder — max_columns={max_columns}, "
            f"plm_batch={plm_batch_size}, use_sample_json={use_sample_json}"
        )

    # build() 인자 호환: BIRD 와 같은 (db_id, db_dir). spider2_db_field 는 optional
    # — None 시 self.spider2_db_field_map 또는 db_id fallback.
    def build(
        self,
        db_id: str,
        db_dir: str,
        spider2_db_field: Optional[str] = None,
    ) -> Tuple[HeteroData, Dict]:
        t_total = time.perf_counter()

        # db_field 결정
        if spider2_db_field is None:
            spider2_db_field = self.spider2_db_field_map.get(db_id, db_id)

        spider2_root = Path(db_dir)
        if not spider2_root.exists():
            raise FileNotFoundError(
                f"Spider2 root not found: {spider2_root} "
                f"(expected {DEFAULT_SPIDER2_ROOT})"
            )

        # 1. DDL.csv path(s) 결정
        backend, ddl_paths = resolve_spider2_db_paths(
            db_id, spider2_db_field, spider2_root)
        dialect = backend  # sqlglot dialect identical to backend name

        # 2. 모든 DDL.csv 파싱 + 병합 (table 중복 시 inner__table 위 unique)
        merged_tables: List[str] = []
        merged_columns: Dict[str, List[Dict]] = {}
        merged_fks: List[Dict] = []
        parse_errors_all: List[Dict] = []
        inner_datasets: List[str] = []
        # short_name → list of (qualified_name) — 중복 검출용
        seen_short_names: Dict[str, int] = {}

        for ddl_path in ddl_paths:
            inner = ddl_path.parent.name
            inner_datasets.append(inner)
            try:
                parsed = _parse_ddl_csv(ddl_path, dialect=dialect)
            except Exception as e:
                logger.warning(f"DDL.csv parse failed ({ddl_path}): {e}")
                parse_errors_all.append({"ddl": str(ddl_path), "error": str(e)})
                continue

            for tbl in parsed["tables"]:
                if tbl in seen_short_names:
                    # 같은 short_name 위 inner suffix 위 unique
                    unique_name = f"{inner}__{tbl}"
                    seen_short_names[tbl] += 1
                else:
                    unique_name = tbl
                    seen_short_names[tbl] = 1
                merged_tables.append(unique_name)
                merged_columns[unique_name] = parsed["columns"].get(tbl, [])

            # FK 위 short name 만 — 같은 inner 안 위 reference 가정 (cross-inner
            # FK 는 Spider2 위 거의 없음, 있어도 short name 매핑 시 ambiguity)
            for fk in parsed["foreign_keys"]:
                merged_fks.append(fk)
            for err in parsed["parse_errors"]:
                err["ddl"] = str(ddl_path)
                parse_errors_all.append(err)

        # 3. enterprise-scale check
        total_cols = sum(len(c) for c in merged_columns.values())
        if total_cols == 0:
            raise RuntimeError(
                f"Spider2 DB '{spider2_db_field}' (backend={backend}) — parsed "
                f"0 columns from {len(ddl_paths)} DDL.csv file(s) "
                f"(parse errors: {len(parse_errors_all)})"
            )
        if total_cols > self.max_columns:
            raise RuntimeError(
                f"Spider2 DB '{spider2_db_field}' (backend={backend}) has "
                f"{total_cols} columns (> max_columns={self.max_columns}) — "
                f"skip per enterprise-scale policy"
            )

        # 4. per-table .json (sample) 로드 — v1 위 metadata 노출만
        samples_per_db: Dict[str, Dict] = {}
        if self.use_sample_json:
            for ddl_path in ddl_paths:
                try:
                    samples_per_db.update(_load_table_samples(ddl_path.parent))
                except Exception:
                    pass

        # 5. schema_info dict (BIRD format 정합)
        schema_info = {
            "tables": merged_tables,
            "columns": merged_columns,
            "foreign_keys": _filter_valid_fks(merged_fks, merged_columns),
        }

        # 6. build (HeteroData + metadata)
        return self._build_from_schema(
            db_id=spider2_db_field, schema_info=schema_info,
            backend=backend, inner_datasets=inner_datasets,
            parse_errors=parse_errors_all,
            samples_per_db=samples_per_db,
            t_total_start=t_total,
        )

    def _build_from_schema(
        self,
        db_id: str,
        schema_info: Dict[str, Any],
        backend: str,
        inner_datasets: List[str],
        parse_errors: List[Dict],
        samples_per_db: Dict[str, Dict],
        t_total_start: float,
    ) -> Tuple[HeteroData, Dict]:
        """BIRD HeteroGraphBuilder.build() Step 2~ 와 같은 logic 직접 구현."""
        fk_descriptions = self._generate_fk_descriptions(schema_info["foreign_keys"])

        data = HeteroData()
        table_to_id: Dict[str, int] = {}
        col_to_id: Dict[str, int] = {}
        fk_to_id: Dict[str, int] = {}
        table_texts: List[str] = []
        col_texts: List[str] = []
        fk_texts: List[str] = []

        # Step 1: Nodes
        for idx, t in enumerate(schema_info["tables"]):
            table_to_id[t] = idx
            table_texts.append(f"Table: {t}")

        c_idx = 0
        for table, cols in schema_info["columns"].items():
            for col in cols:
                full_name = f"{table}.{col['name']}"
                col_to_id[full_name] = c_idx
                ctype = col.get("type") or "UNKNOWN"
                col_texts.append(
                    f"Column: {col['name']} in table {table}, type {ctype}."
                )
                c_idx += 1

        for idx, (edge_id, desc) in enumerate(fk_descriptions.items()):
            fk_to_id[edge_id] = idx
            fk_texts.append(desc)

        # Step 2: PLM encoding (batch_size 명시 위 메모리 제어)
        t = time.perf_counter()
        data["table"].x = self.encoder.encode(
            table_texts, convert_to_tensor=True,
            batch_size=self.plm_batch_size, show_progress_bar=False,
        ).cpu()
        data["column"].x = self.encoder.encode(
            col_texts, convert_to_tensor=True,
            batch_size=self.plm_batch_size, show_progress_bar=False,
        ).cpu()
        if fk_texts:
            data["fk_node"].x = self.encoder.encode(
                fk_texts, convert_to_tensor=True,
                batch_size=self.plm_batch_size, show_progress_bar=False,
            ).cpu()
        else:
            data["fk_node"].x = torch.empty(
                (0, self.encoder.get_sentence_embedding_dimension())
            ).cpu()
        t_encode = time.perf_counter() - t

        # Step 3: Edges
        h_src, h_dst = [], []
        f_src, f_dst = [], []
        r_src, r_dst = [], []
        t_fk_src, t_fk_dst = [], []

        for table, cols in schema_info["columns"].items():
            t_id = table_to_id[table]
            for col in cols:
                c_id = col_to_id[f"{table}.{col['name']}"]
                h_src.append(t_id); h_dst.append(c_id)

        for fk in schema_info["foreign_keys"]:
            f_col = f"{fk['from_table']}.{fk['from_column']}"
            t_col = f"{fk['to_table']}.{fk['to_column']}"
            edge_id = f"{f_col}->{t_col}"
            if edge_id in fk_to_id and f_col in col_to_id and t_col in col_to_id:
                fid, cid1, cid2 = fk_to_id[edge_id], col_to_id[f_col], col_to_id[t_col]
                f_src.append(cid1); f_dst.append(fid)
                r_src.append(fid); r_dst.append(cid2)
            f_t = fk["from_table"]
            t_t = fk["to_table"]
            if self.add_t2t_edges and f_t in table_to_id and t_t in table_to_id:
                t_fk_src.extend([table_to_id[f_t], table_to_id[t_t]])
                t_fk_dst.extend([table_to_id[t_t], table_to_id[f_t]])

        data["table", "has_column", "column"].edge_index = torch.tensor(
            [h_src, h_dst], dtype=torch.long)
        data["column", "belongs_to", "table"].edge_index = torch.tensor(
            [h_dst, h_src], dtype=torch.long)
        if f_src:
            data["column", "is_source_of", "fk_node"].edge_index = torch.tensor(
                [f_src, f_dst], dtype=torch.long)
            data["fk_node", "points_to", "column"].edge_index = torch.tensor(
                [r_src, r_dst], dtype=torch.long)
        if t_fk_src:
            data["table", "table_to_table", "table"].edge_index = torch.tensor(
                [t_fk_src, t_fk_dst], dtype=torch.long)

        # Step 4: Metadata (PCST flat 인덱싱 + downstream contract)
        num_t, num_c = len(table_to_id), len(col_to_id)
        node_meta: Dict[int, str] = {}
        for k, v in table_to_id.items(): node_meta[v] = k
        for k, v in col_to_id.items():   node_meta[v + num_t] = k
        for k, v in fk_to_id.items():    node_meta[v + num_t + num_c] = k

        pcst_edges = (
            [(s, d + num_t) for s, d in zip(h_src, h_dst)] +
            [(s + num_t, d + num_t + num_c) for s, d in zip(f_src, f_dst)] +
            [(s + num_t + num_c, d + num_t) for s, d in zip(r_src, r_dst)] +
            [(s, d) for s, d in zip(t_fk_src, t_fk_dst)]
        )
        pcst_edge_types = (
            ["belongs_to"] * len(h_src) +
            ["is_source_of"] * len(f_src) +
            ["points_to"] * len(r_src) +
            ["table_to_table"] * len(t_fk_src)
        )

        metadata: Dict[str, Any] = {
            "table_to_id": table_to_id,
            "col_to_id": col_to_id,
            "fk_to_id": fk_to_id,
            "node_metadata": node_meta,
            "edges": pcst_edges,
            "edge_types": pcst_edge_types,
            "add_t2t_edges": bool(self.add_t2t_edges),
            # Spider2-specific (downstream optional consumption)
            "spider2_backend": backend,
            "spider2_db": db_id,
            "spider2_inner_datasets": inner_datasets,
            "spider2_total_columns": num_c,
            "spider2_parse_errors": parse_errors,
            "spider2_table_samples_loaded": len(samples_per_db),
        }
        t = time.perf_counter()
        metadata.update(self._compute_fk_reachability(schema_info, table_to_id))
        t_reach = time.perf_counter() - t
        t = time.perf_counter()
        metadata.update(self._compute_schema_diameter(
            table_to_id, col_to_id, fk_to_id, pcst_edges))
        t_diam = time.perf_counter() - t

        self.last_info = self._build_builder_info(
            builder_type="Spider2GraphBuilder",
            db_id=db_id,
            schema_info=schema_info,
            table_to_id=table_to_id, col_to_id=col_to_id, fk_to_id=fk_to_id,
            table_texts=table_texts, col_texts=col_texts, fk_texts=fk_texts,
            pcst_edges=pcst_edges, pcst_edge_types=pcst_edge_types,
            metadata=metadata,
            timings={
                "encode_s": float(t_encode),
                "reachability_s": float(t_reach),
                "diameter_s": float(t_diam),
                "total_s": float(time.perf_counter() - t_total_start),
            },
            extra={
                "spider2_backend": backend,
                "spider2_inner_datasets_count": len(inner_datasets),
                "spider2_parse_errors_count": len(parse_errors),
                "spider2_table_samples_loaded": len(samples_per_db),
                "add_t2t_edges": bool(self.add_t2t_edges),
                "schema_diameter": int(metadata.get("schema_diameter", 0) or 0),
            },
        )
        metadata["builder_info"] = dict(self.last_info)
        return data, metadata


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _filter_valid_fks(
    fks: List[Dict[str, str]],
    columns: Dict[str, List[Dict]],
) -> List[Dict[str, str]]:
    """FK 위 from/to table+column 이 schema 위 존재하는지 검증. invalid skip.

    Spider2 위 short_name 위 ambiguity (multi-inner 위 같은 table 명) 있을 수
    있어 FK 가 가리키는 table 이 schema 위 없을 가능성 — 그 경우 silently skip.
    """
    valid = []
    col_lookup: Dict[Tuple[str, str], bool] = {}
    for tbl, cols in columns.items():
        for c in cols:
            col_lookup[(tbl, c["name"])] = True
    for fk in fks:
        ft, fc = fk.get("from_table"), fk.get("from_column")
        tt, tc = fk.get("to_table"), fk.get("to_column")
        if (ft, fc) in col_lookup and (tt, tc) in col_lookup:
            valid.append(fk)
    return valid


def _load_table_samples(ddl_dir: Path) -> Dict[str, Any]:
    """ddl_dir 위 *.json (per-table sample) → {table_name: dict}.

    Spider2 per-table .json 구조 예시 (sample_info.json):
        {"table_name": "sample_info", "table_fullname": "...",
         "column_names": [...], "sample_data": [...]}
    """
    out: Dict[str, Any] = {}
    for jpath in ddl_dir.glob("*.json"):
        try:
            with open(jpath, encoding="utf-8") as f:
                j = json.load(f)
            tn = j.get("table_name") or jpath.stem
            out[tn] = j
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_spider2_lite_jsonl(
    jsonl_path: Path = Path("data/Spider2/spider2-lite/spider2-lite.jsonl"),
) -> Dict[str, Dict[str, Any]]:
    """spider2-lite.jsonl 위 instance_id → {db, question, external_knowledge} 매핑.

    G-S2-1 inference 위 root 가 사용 — instance_id 위 graph build 호출 시
    db_field 결정 위 활용.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"spider2-lite.jsonl not found: {jsonl_path}")
    out: Dict[str, Dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = d.get("instance_id")
            if iid:
                out[iid] = d
    return out
