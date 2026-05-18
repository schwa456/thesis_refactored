"""DB FK/PK metadata extractor — Wave 8 Direction 2 (FK/PK Connectivity Steiner Closure) 선결 utility.

학술 agent improving_m4_plan_scholar_agent_2026-05-18.md §2.1 + DECISIONS 2026-05-18 Wave 8 §2 D2 정합.

LLM 호출 0×, BIRD-Dev 11 DBs 1회성 추출 (idempotent if cache present).

Output JSON 형식:
  {
    db_id: {
      "tables": {
        table_name: {
          "fk_cols": [col, ...],
          "pk_cols": [col, ...],
          "referenced": {ref_tbl: [[local_col, ref_col], ...]}
        }
      },
      "fk_constraints": [
        {"from_table": tbl, "from_col": col, "to_table": tbl2, "to_col": col2},
        ...
      ]
    }
  }

- `tables[t].fk_cols` / `pk_cols` / `referenced`: 사용자 spec (DECISIONS §2 D2 Step 1)
- `fk_constraints`: 학술 agent §2.1 build_local_fk_graph 가 직접 소비하는 flat list

Steiner Closure (filters 모듈 D2Filter) 가 `load_db_fk_metadata()` 로 전체 dict 를 로드 후
`db_fk_metadata[db_id]` 로 per-DB 메타데이터 사용.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_DB_ROOT = Path("data/raw/BIRD_dev/dev_databases")
DEFAULT_OUT = Path("data/processed/db_fk_metadata.json")


def _safe_quote(name: str) -> str:
    """SQL identifier 내 single quote escape (PRAGMA 인자용)."""
    return name.replace("'", "''")


def _extract_db_fk_metadata(sqlite_path: Path) -> Dict[str, Any]:
    """단일 SQLite DB 에서 FK/PK 메타데이터 추출.

    SQLite PRAGMA:
      - foreign_key_list(table): (id, seq, ref_table, from_col, to_col, on_update, on_delete, match)
      - table_info(table): (cid, name, type, notnull, dflt_value, pk)
    """
    con = sqlite3.connect(str(sqlite_path))
    cur = con.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    all_tables = [r[0] for r in cur.fetchall() if not r[0].startswith("sqlite_")]

    tables_meta: Dict[str, Dict[str, Any]] = {}
    fk_constraints: List[Dict[str, str]] = []

    for tbl in all_tables:
        safe_tbl = _safe_quote(tbl)

        pk_cols: List[str] = []
        try:
            cur.execute(f"PRAGMA table_info('{safe_tbl}')")
            for row in cur.fetchall():
                if row[5] and int(row[5]) > 0:
                    pk_cols.append(row[1])
        except sqlite3.Error:
            pass

        fk_cols_seen: List[str] = []
        referenced: Dict[str, List[List[str]]] = {}
        try:
            cur.execute(f"PRAGMA foreign_key_list('{safe_tbl}')")
            for row in cur.fetchall():
                ref_tbl = row[2]
                from_col = row[3]
                to_col = row[4]
                if not from_col or not ref_tbl:
                    continue
                fk_cols_seen.append(from_col)
                referenced.setdefault(ref_tbl, []).append([from_col, to_col])
                fk_constraints.append({
                    "from_table": tbl,
                    "from_col": from_col,
                    "to_table": ref_tbl,
                    "to_col": to_col,
                })
        except sqlite3.Error:
            pass

        tables_meta[tbl] = {
            "fk_cols": sorted(set(fk_cols_seen)),
            "pk_cols": pk_cols,
            "referenced": referenced,
        }

    con.close()
    return {"tables": tables_meta, "fk_constraints": fk_constraints}


def extract_all_dbs(db_root: Path) -> Dict[str, Dict[str, Any]]:
    """db_root 아래 각 DB 폴더의 <db>.sqlite 에서 FK/PK metadata 추출."""
    if not db_root.exists():
        raise FileNotFoundError(f"DB root not found: {db_root}")

    out: Dict[str, Dict[str, Any]] = {}
    for db_dir in sorted(db_root.iterdir()):
        if not db_dir.is_dir():
            continue
        sqlite_path = db_dir / f"{db_dir.name}.sqlite"
        if not sqlite_path.exists():
            continue
        out[db_dir.name] = _extract_db_fk_metadata(sqlite_path)
    return out


def load_db_fk_metadata(path: str | Path = DEFAULT_OUT) -> Dict[str, Dict[str, Any]]:
    """filters 모듈 D2Filter 가 사용하는 loader. 캐시 미존재 시 RuntimeError."""
    p = Path(path)
    if not p.exists():
        raise RuntimeError(
            f"db_fk_metadata cache not found at {p}. "
            f"Run: python -m modules.builders.db_fk_extractor --out {p}"
        )
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db-root",
        default=str(DEFAULT_DB_ROOT),
        help="Root directory containing per-DB folders with <db>.sqlite",
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output JSON path",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if output already exists (idempotent default).",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"[db_fk_metadata] exists at {out_path} — skip (use --force to rebuild)")
        return

    db_root = Path(args.db_root)
    payload = extract_all_dbs(db_root)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    n_dbs = len(payload)
    n_fks = sum(len(p["fk_constraints"]) for p in payload.values())
    n_tbls = sum(len(p["tables"]) for p in payload.values())
    print(
        f"[db_fk_metadata] wrote {n_dbs} DBs / {n_tbls} tables / "
        f"{n_fks} FK constraints → {out_path}"
    )

    # Per-DB summary
    for db_id, p in payload.items():
        print(
            f"  {db_id}: tables={len(p['tables'])}, "
            f"fk_constraints={len(p['fk_constraints'])}"
        )


if __name__ == "__main__":
    main()
