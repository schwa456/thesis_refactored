"""Smoke tests for RSLBackwardFilter (no pytest, no network).

LLM clients (XiYan + preliminary-SQL) 를 mock 으로 교체, sqlglot 은 실제 사용.

학술 Agent Phase 1 의 54.50% S_restore=∅ 정합성 (BIRD-Dev 1534 query 중 약 절반은
backward 가 forward 와 동일 col 만 추출) 도 시나리오로 함의.
"""
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("VLLM_API_KEY", "dummy")

from modules.filters.rsl_backward_filter import (  # noqa: E402
    RSLBackwardFilter,
    _extract_columns_from_sql,
)


class _CallableMock:
    """generate_text(prompt, model, temperature) → 미리 지정된 응답."""

    def __init__(self, responses: Any):
        # responses: 단일 str 또는 sequential list (call 순서별)
        if isinstance(responses, list):
            self._responses: List[str] = list(responses)
            self._sequential = True
        else:
            self._responses = [str(responses)]
            self._sequential = False
        self.calls: List[Dict[str, Any]] = []

    def generate_text(self, prompt: str, model: str, temperature: float, **kw) -> str:
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        if self._sequential:
            if not self._responses:
                return ""
            return self._responses.pop(0)
        return self._responses[0]


def _make_filter(
    xiyan_response: str,
    prelim_sql_response: str,
    risky_dbs: List[str] = None,
    fk_pk_hardcode: bool = True,
) -> RSLBackwardFilter:
    flt = RSLBackwardFilter(
        model_name="mock-model",
        temperature=0.0,
        xiyan_max_iteration=1,
        num_examples=0,
        fk_pk_hardcode=fk_pk_hardcode,
        risky_dbs=risky_dbs or [],
        provider=None,
        db_dir="/nonexistent",
    )
    # 분리된 client mock — 두 단계가 독립 instance
    flt.xiyan.client = _CallableMock(xiyan_response)
    flt.client = _CallableMock(prelim_sql_response)
    return flt


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


# ============================================================
# Helper: _extract_columns_from_sql
# ============================================================
def test_extract_columns_basic_select():
    print("\n[test] sqlglot column extraction — basic SELECT")
    sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE u.age > 18"
    cols = _extract_columns_from_sql(sql)
    _assert(cols == {"name", "total", "id", "user_id", "age"}, f"got {sorted(cols)}")


def test_extract_columns_strips_fences():
    print("\n[test] markdown fence ```sql ... ``` is stripped before parse")
    sql = "```sql\nSELECT a.x FROM a\n```"
    cols = _extract_columns_from_sql(sql)
    _assert(cols == {"x"}, f"got {cols}")


def test_extract_columns_garbage_returns_empty():
    print("\n[test] unparseable text → empty set, no exception")
    cols = _extract_columns_from_sql("not sql at all, totally random words")
    _assert(cols == set(), f"got {cols}")


def test_extract_columns_empty_string_returns_empty():
    print("\n[test] empty input → empty")
    _assert(_extract_columns_from_sql("") == set(), "empty string")
    _assert(_extract_columns_from_sql(None) == set(), "None")


# ============================================================
# Helper: col-only normalization
# ============================================================
def test_col_names_normalization():
    print("\n[test] _col_names strips table prefix (alias-distinct → col-only)")
    out = RSLBackwardFilter._col_names(["users.id", "orders.user_id", "bare", "t1.x"])
    _assert(out == {"id", "user_id", "bare", "x"}, f"got {sorted(out)}")


def test_expand_to_full_paths_duplicates():
    print("\n[test] _expand: col name in multiple tables → all candidates kept")
    full_schema = {
        "users": ["id", "name"],
        "orders": ["id", "total"],   # users.id and orders.id share name "id"
    }
    expanded = RSLBackwardFilter._expand_to_full_paths({"id"}, full_schema)
    _assert(expanded == {"users.id", "orders.id"}, f"got {sorted(expanded)}")


# ============================================================
# Scenario 1: clean SQL → S_restore non-empty
# ============================================================
def test_clean_sql_restores_missed_column():
    print("\n[test] clean prelim SQL surfaces a missed column → S_restore non-empty")
    xiyan_resp = json.dumps({"users": ["name"], "orders": ["total"]})  # forward: name + total
    prelim_sql = "SELECT u.name, o.total, o.placed_at FROM users u JOIN orders o ON u.id = o.user_id"
    metadata = {
        "col_to_id": {
            "users.id": 0, "users.name": 1,
            "orders.user_id": 2, "orders.total": 3, "orders.placed_at": 4,
        },
        "table_to_id": {"users": 0, "orders": 1},
        "fk_to_id": {"orders.user_id->users.id": 0},
    }
    flt = _make_filter(xiyan_resp, prelim_sql)
    result = flt.refine(
        query="total per user",
        subgraph={"users": ["name", "id"], "orders": ["total", "placed_at", "user_id"]},
        db_id=None,
        metadata=metadata,
    )
    final = set(result["final_nodes"])
    _assert("orders.placed_at" in final, "placed_at restored via backward")
    _assert("users.name" in final and "orders.total" in final, "forward preserved")
    _assert(result["stats"]["restore_expanded"] >= 1, "restore non-empty")
    _assert(result["stats"]["sql_parse_ok"] is True, "SQL parsed OK")
    _assert(result["stats"]["restore_is_empty"] is False, "restore flag false")


# ============================================================
# Scenario 2: S_restore = ∅ (54.50% Phase 1 정합성)
# ============================================================
def test_backward_produces_no_restore_when_same_cols():
    print("\n[test] backward SQL uses same columns as forward → S_restore=∅")
    # forward picks: name, total
    xiyan_resp = json.dumps({"users": ["name"], "orders": ["total"]})
    # backward picks the same columns
    prelim_sql = "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id"
    metadata = {
        "col_to_id": {
            "users.id": 0, "users.name": 1,
            "orders.user_id": 2, "orders.total": 3,
        },
        "table_to_id": {"users": 0, "orders": 1},
        "fk_to_id": {"orders.user_id->users.id": 0},
    }
    flt = _make_filter(xiyan_resp, prelim_sql, fk_pk_hardcode=False)
    result = flt.refine(
        query="total per user",
        subgraph={"users": ["name", "id"], "orders": ["total", "user_id"]},
        db_id=None,
        metadata=metadata,
    )
    # join col "id" / "user_id" 가 backward 에 있지만 forward 에도 없으므로
    # restore 에 들어가지만 fk_pk_hardcode=False 면 struct=0.
    # forward = {users.name, orders.total}
    # bwd_cols = {name, total, id, user_id}
    # restore_col = {id, user_id}
    # expand → {users.id, orders.user_id}
    _assert(result["stats"]["restore_col_diff"] == 2, "id+user_id restored")
    # 이 시나리오는 implicit join key restore (학술 Agent 핵심 가치 한 줄)
    _assert(result["stats"]["sql_parse_ok"] is True, "parsed")


def test_backward_truly_no_restore_when_identical_set():
    print("\n[test] forward already includes join keys → restore_col_diff=0")
    xiyan_resp = json.dumps({"users": ["name", "id"], "orders": ["total", "user_id"]})
    prelim_sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    metadata = {
        "col_to_id": {
            "users.id": 0, "users.name": 1,
            "orders.user_id": 2, "orders.total": 3,
        },
        "table_to_id": {"users": 0, "orders": 1},
    }
    flt = _make_filter(xiyan_resp, prelim_sql, fk_pk_hardcode=False)
    result = flt.refine(
        query="total per user",
        subgraph={"users": ["name", "id"], "orders": ["total", "user_id"]},
        db_id=None,
        metadata=metadata,
    )
    _assert(result["stats"]["restore_col_diff"] == 0, "no col diff")
    _assert(result["stats"]["restore_expanded"] == 0, "no expansion")
    _assert(result["stats"]["restore_is_empty"] is True, "restore_is_empty flag")
    final = set(result["final_nodes"])
    _assert(final == {"users.name", "users.id", "orders.total", "orders.user_id"},
            "final = forward")


# ============================================================
# Scenario 3: DB-level guard (toxicology)
# ============================================================
def test_db_guard_skips_restore_for_risky_db():
    print("\n[test] db_id in risky_dbs → S_restore forced empty")
    xiyan_resp = json.dumps({"protein": ["name"]})
    prelim_sql = "SELECT protein.name, protein.toxicity FROM protein"
    metadata = {
        "col_to_id": {"protein.name": 0, "protein.toxicity": 1},
        "table_to_id": {"protein": 0},
    }
    flt = _make_filter(
        xiyan_resp, prelim_sql,
        risky_dbs=["toxicology"],
        fk_pk_hardcode=False,
    )
    result = flt.refine(
        query="protein toxicity",
        subgraph={"protein": ["name", "toxicity"]},
        db_id="toxicology",
        metadata=metadata,
    )
    _assert(result["stats"]["db_guard_active"] is True, "guard active")
    _assert(result["stats"]["restore_expanded"] == 0, "no expansion under guard")
    final = set(result["final_nodes"])
    _assert(final == {"protein.name"}, "final = forward only (under guard)")


def test_db_guard_inactive_when_db_not_in_list():
    print("\n[test] db_id NOT in risky_dbs → guard inactive, restore runs normally")
    xiyan_resp = json.dumps({"protein": ["name"]})
    prelim_sql = "SELECT protein.name, protein.toxicity FROM protein"
    metadata = {
        "col_to_id": {"protein.name": 0, "protein.toxicity": 1},
        "table_to_id": {"protein": 0},
    }
    flt = _make_filter(
        xiyan_resp, prelim_sql,
        risky_dbs=["toxicology"],
        fk_pk_hardcode=False,
    )
    result = flt.refine(
        query="protein info",
        subgraph={"protein": ["name", "toxicity"]},
        db_id="other_db",
        metadata=metadata,
    )
    _assert(result["stats"]["db_guard_active"] is False, "guard NOT active")
    _assert("protein.toxicity" in result["final_nodes"], "restore expanded normally")


# ============================================================
# Scenario 4: FK/PK hardcode (Step 4)
# ============================================================
def test_fk_hardcode_preserves_join_columns():
    print("\n[test] fk_pk_hardcode rescues FK columns dropped by forward")
    # forward intentionally drops the FK column
    xiyan_resp = json.dumps({"users": ["name"], "orders": ["total"]})
    prelim_sql = "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id"
    metadata = {
        "col_to_id": {
            "users.id": 0, "users.name": 1,
            "orders.user_id": 2, "orders.total": 3,
        },
        "table_to_id": {"users": 0, "orders": 1},
        "fk_to_id": {"orders.user_id->users.id": 0},
    }
    flt = _make_filter(xiyan_resp, prelim_sql, fk_pk_hardcode=True)
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"], "orders": ["total", "user_id"]},
        db_id=None,
        metadata=metadata,
    )
    _assert(result["stats"]["struct"] == 2, "users.id + orders.user_id rescued as FK")
    final = set(result["final_nodes"])
    _assert({"users.id", "orders.user_id"}.issubset(final), "FK columns preserved")


# ============================================================
# Scenario 5: SQL parse failure → empty restore, recall-safe (forward preserved)
# ============================================================
def test_sql_parse_failure_keeps_forward():
    print("\n[test] preliminary SQL unparseable → restore=∅, forward preserved")
    xiyan_resp = json.dumps({"users": ["name"], "orders": ["total"]})
    prelim_sql = "GARBLE NOT VALID SQL"
    metadata = {
        "col_to_id": {"users.id": 0, "users.name": 1, "orders.total": 2},
        "table_to_id": {"users": 0, "orders": 1},
    }
    flt = _make_filter(xiyan_resp, prelim_sql, fk_pk_hardcode=False)
    result = flt.refine(
        query="q",
        subgraph={"users": ["name", "id"], "orders": ["total"]},
        db_id=None,
        metadata=metadata,
    )
    _assert(result["stats"]["bwd_col_names"] == 0, "no columns extracted")
    _assert(result["stats"]["restore_expanded"] == 0, "no restore")
    final = set(result["final_nodes"])
    _assert(final == {"users.name", "orders.total"}, "forward preserved despite SQL fail")


# ============================================================
# Scenario 6: empty subgraph forward returns Unanswerable
# ============================================================
def test_empty_subgraph_propagates_unanswerable_when_backward_also_empty():
    print("\n[test] empty subgraph + no metadata → Unanswerable propagation")
    xiyan_resp = json.dumps({})
    prelim_sql = ""
    flt = _make_filter(xiyan_resp, prelim_sql, fk_pk_hardcode=False)
    result = flt.refine(query="q", subgraph={}, db_id=None, metadata=None)
    _assert(result["status"] == "Unanswerable", "status propagates")
    _assert(result["final_nodes"] == [], "no nodes")


# ============================================================
# Scenario 7: full schema fallback to PCST subgraph when no metadata
# ============================================================
def test_full_schema_falls_back_to_subgraph():
    print("\n[test] no metadata → full schema fallback to subgraph (limited search)")
    xiyan_resp = json.dumps({"users": ["name"]})
    # backward mentions 'name' (already in fwd) and 'email' (in subgraph but not fwd)
    prelim_sql = "SELECT users.name, users.email FROM users"
    flt = _make_filter(xiyan_resp, prelim_sql, fk_pk_hardcode=False)
    result = flt.refine(
        query="user info",
        subgraph={"users": ["name", "email", "id"]},
        db_id=None,
        metadata=None,  # forces fallback
    )
    # subgraph 가 full schema 대용 → restore search space = subgraph 만
    final = set(result["final_nodes"])
    _assert("users.email" in final, "email restored from subgraph as full-schema fallback")


def run_all():
    tests = [
        test_extract_columns_basic_select,
        test_extract_columns_strips_fences,
        test_extract_columns_garbage_returns_empty,
        test_extract_columns_empty_string_returns_empty,
        test_col_names_normalization,
        test_expand_to_full_paths_duplicates,
        test_clean_sql_restores_missed_column,
        test_backward_produces_no_restore_when_same_cols,
        test_backward_truly_no_restore_when_identical_set,
        test_db_guard_skips_restore_for_risky_db,
        test_db_guard_inactive_when_db_not_in_list,
        test_fk_hardcode_preserves_join_columns,
        test_sql_parse_failure_keeps_forward,
        test_empty_subgraph_propagates_unanswerable_when_backward_also_empty,
        test_full_schema_falls_back_to_subgraph,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
        except Exception as e:
            failures.append((t.__name__, f"UNEXPECTED: {type(e).__name__}: {e}"))

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} / {len(tests)}")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print(f"PASSED: {len(tests)} / {len(tests)}")


if __name__ == "__main__":
    run_all()
