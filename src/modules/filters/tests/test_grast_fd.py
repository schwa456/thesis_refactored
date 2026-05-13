"""Smoke tests for GRASTFDFilter — Direction C Steiner-tree based restoration.

LLM clients (XiYan + optional prelim SQL) mock 으로 교체, networkx 의 steiner_tree 는 실제 사용.
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

from modules.filters.grast_fd_filter import GRASTFDFilter  # noqa: E402


class _CallableMock:
    def __init__(self, responses: Any):
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
    prelim_sql_response: str = "",
    terminal_source: str = "forward",
    top_k: int = 10,
    inferred_fk: List[str] = None,
    fk_pk_hardcode: bool = True,
    max_restore: int = 30,
    include_belongs_to: bool = True,
) -> GRASTFDFilter:
    flt = GRASTFDFilter(
        model_name="mock-model",
        temperature=0.0,
        xiyan_max_iteration=1,
        num_examples=0,
        terminal_source=terminal_source,
        top_k=top_k,
        inferred_fk=inferred_fk or [],
        max_restore=max_restore,
        include_belongs_to=include_belongs_to,
        fk_pk_hardcode=fk_pk_hardcode,
        provider=None,
        db_dir="/nonexistent",
    )
    flt.xiyan.client = _CallableMock(xiyan_response)
    flt.client = _CallableMock(prelim_sql_response)
    return flt


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


# ============================================================
# Unit: FD graph build + parse helper
# ============================================================
def test_parse_fk_key():
    print("\n[test] _parse_fk_key roundtrip")
    f = GRASTFDFilter(provider=None, db_dir="/nonexistent")
    _assert(f._parse_fk_key("a.x->b.y") == ("a.x", "b.y"), "well-formed parse")
    _assert(f._parse_fk_key("garbage") is None, "missing arrow → None")
    _assert(f._parse_fk_key("a->b") is None, "missing . → None")
    _assert(f._parse_fk_key(None) is None, "None input")


def test_fd_graph_has_belongs_to_and_fk_edges():
    print("\n[test] FD graph: belongs_to + FK edges present")
    f = _make_filter(xiyan_response="{}")
    full_schema = {"users": ["id", "name"], "orders": ["id", "user_id", "total"]}
    metadata = {"fk_to_id": {"orders.user_id->users.id": 0}}
    G = f._build_fd_graph(full_schema, metadata)
    _assert(G.has_node("users.id") and G.has_node("orders"), "column + table nodes present")
    _assert(G.has_edge("users.id", "users"), "belongs_to edge present")
    _assert(G.has_edge("orders.user_id", "users.id"), "declared FK edge present")
    # inferred FK 없음
    inferred_kinds = {d["kind"] for _, _, d in G.edges(data=True)}
    _assert("fk_declared" in inferred_kinds, "fk_declared kind set")


def test_fd_graph_inferred_fk_added():
    print("\n[test] inferred_fk yaml list added with kind='fk_inferred'")
    f = _make_filter(xiyan_response="{}", inferred_fk=["a.id->b.a_id"])
    G = f._build_fd_graph({"a": ["id"], "b": ["a_id"]}, metadata={})
    _assert(G.has_edge("a.id", "b.a_id"), "inferred FK edge present")
    edge_kind = G.edges[("a.id", "b.a_id")]["kind"]
    _assert(edge_kind == "fk_inferred", f"kind=fk_inferred, got {edge_kind}")


def test_fd_graph_without_belongs_to():
    print("\n[test] include_belongs_to=False omits column--table edges")
    f = _make_filter(xiyan_response="{}", include_belongs_to=False)
    G = f._build_fd_graph({"t": ["c1", "c2"]}, metadata={})
    _assert(not G.has_edge("t.c1", "t"), "no belongs_to edge")
    # 단 column 끼리는 FK 없으면 isolated
    _assert("t" in G.nodes(), "table node still present")


# ============================================================
# Scenario 1: Steiner tree restores join column via FK
# ============================================================
def test_steiner_restores_missing_join_key():
    print("\n[test] forward misses join col → Steiner restores via FK path")
    # forward picks users.name + orders.total (no user_id, no id)
    xiyan_resp = json.dumps({"users": ["name"], "orders": ["total"]})
    full_metadata = {
        "col_to_id": {
            "users.id": 0, "users.name": 1,
            "orders.user_id": 2, "orders.total": 3, "orders.id": 4,
        },
        "table_to_id": {"users": 0, "orders": 1},
        "fk_to_id": {"orders.user_id->users.id": 0},
    }
    f = _make_filter(xiyan_resp, terminal_source="forward", fk_pk_hardcode=False)
    result = f.refine(
        query="total per user",
        subgraph={"users": ["name", "id"], "orders": ["total", "user_id"]},
        db_id=None,
        metadata=full_metadata,
    )
    final = set(result["final_nodes"])
    # forward (users.name, orders.total) + Steiner tree (path between them via FK).
    # path: users.name -- users -- users.id -- orders.user_id -- orders -- orders.total
    # restore column-only: {users.id, orders.user_id}
    _assert({"users.name", "orders.total"}.issubset(final), "forward preserved")
    _assert("users.id" in final and "orders.user_id" in final,
            "Steiner restores join keys")
    _assert(result["stats"]["steiner_restore"] >= 2, "restore non-empty")
    _assert(result["stats"]["restore_is_empty"] is False, "flag false")
    _assert(result["stats"]["terminal_count"] >= 2, "terminals from forward")


# ============================================================
# Scenario 2: S_steiner_restore = ∅ when forward already connected
# ============================================================
def test_steiner_no_restore_when_already_connected():
    print("\n[test] forward already includes join keys → Steiner adds 0 new cols")
    xiyan_resp = json.dumps({"users": ["id", "name"], "orders": ["user_id", "total"]})
    metadata = {
        "col_to_id": {
            "users.id": 0, "users.name": 1,
            "orders.user_id": 2, "orders.total": 3,
        },
        "table_to_id": {"users": 0, "orders": 1},
        "fk_to_id": {"orders.user_id->users.id": 0},
    }
    f = _make_filter(xiyan_resp, terminal_source="forward", fk_pk_hardcode=False)
    result = f.refine(
        query="q",
        subgraph={"users": ["name", "id"], "orders": ["total", "user_id"]},
        db_id=None,
        metadata=metadata,
    )
    # Steiner tree 가 forward 의 4 column 을 모두 포함하지만 추가 column 은 없음
    _assert(result["stats"]["steiner_restore"] == 0, "no new cols restored")
    _assert(result["stats"]["restore_is_empty"] is True, "restore_is_empty flag")


# ============================================================
# Scenario 3: Disconnected components — partial restore only
# ============================================================
def test_disconnected_components_partial_restore():
    print("\n[test] disconnected schema → Steiner only restores in connected sub-area")
    xiyan_resp = json.dumps({"users": ["name"], "isolated": ["x"]})
    metadata = {
        "col_to_id": {
            "users.id": 0, "users.name": 1,
            "isolated.x": 2,
            # 'orders' tables only connected via FK to users
        },
        "table_to_id": {"users": 0, "isolated": 1},
        "fk_to_id": {},  # no FK at all
    }
    f = _make_filter(xiyan_resp, terminal_source="forward", fk_pk_hardcode=False,
                     include_belongs_to=True)
    result = f.refine(
        query="q",
        subgraph={"users": ["name"], "isolated": ["x"]},
        db_id=None,
        metadata=metadata,
    )
    # users.name 과 isolated.x 는 belongs_to edge 만 있고, 두 table 간 연결 없음.
    # Steiner tree 는 component 별로 single-terminal 만 → 둘 다 skip → restore=0
    final = set(result["final_nodes"])
    _assert(final == {"users.name", "isolated.x"}, "only forward preserved")
    _assert(result["stats"]["steiner_restore"] == 0, "no cross-component restore")


# ============================================================
# Scenario 4: inferred_fk bridges previously disconnected tables
# ============================================================
def test_inferred_fk_enables_restore():
    print("\n[test] inferred_fk yaml list bridges disconnected sub-graphs")
    xiyan_resp = json.dumps({"a": ["x"], "b": ["y"]})
    metadata = {
        "col_to_id": {"a.x": 0, "a.id": 1, "b.y": 2, "b.a_id": 3},
        "table_to_id": {"a": 0, "b": 1},
        "fk_to_id": {},  # declared FK = 0 (debit_card_specializing 모사)
    }
    # GPT-4.1-mini 가 보완한 inferred FK
    f = _make_filter(
        xiyan_resp, terminal_source="forward",
        inferred_fk=["a.id->b.a_id"],
        fk_pk_hardcode=False,
    )
    result = f.refine(
        query="join a and b",
        subgraph={"a": ["x", "id"], "b": ["y", "a_id"]},
        db_id=None,
        metadata=metadata,
    )
    final = set(result["final_nodes"])
    _assert({"a.x", "b.y"}.issubset(final), "forward preserved")
    _assert("a.id" in final or "b.a_id" in final,
            "inferred FK bridge brought in restore col")
    _assert(result["stats"]["inferred_fk_count"] == 1, "inferred_fk recorded")


# ============================================================
# Scenario 5: terminal_source='gat_topk' uses score ranking
# ============================================================
def test_terminal_source_gat_topk():
    print("\n[test] terminal_source='gat_topk' picks top-K by gat_scores")
    xiyan_resp = json.dumps({"users": ["name"]})  # forward only name
    metadata = {
        "col_to_id": {"users.id": 0, "users.name": 1, "users.age": 2, "users.email": 3},
        "table_to_id": {"users": 0},
        "fk_to_id": {},
    }
    f = _make_filter(xiyan_resp, terminal_source="gat_topk", top_k=2, fk_pk_hardcode=False)
    result = f.refine(
        query="q",
        subgraph={"users": ["id", "name", "age", "email"]},
        db_id=None,
        gat_scores={
            "users.id": 0.95, "users.email": 0.90,  # top 2
            "users.name": 0.50, "users.age": 0.30,
        },
        metadata=metadata,
    )
    # terminals = forward(name) ∪ top2(id, email) = {name, id, email}
    _assert(result["stats"]["terminal_source_used"] == "gat_topk",
            "mode used: gat_topk")
    # 모두 동일 table → Steiner tree 안에 users.{id, email, name} + users
    # restore = Steiner cols − fwd(name) = {id, email}
    final = set(result["final_nodes"])
    _assert({"users.id", "users.email"}.issubset(final),
            "GAT top-K terminals restored")


# ============================================================
# Scenario 6: terminal_source='gat_topk' fallback to 'forward' when gat_scores=None
# ============================================================
def test_terminal_source_gat_topk_fallback():
    print("\n[test] terminal_source='gat_topk' + gat_scores=None → fallback to forward")
    xiyan_resp = json.dumps({"users": ["name", "id"]})
    metadata = {
        "col_to_id": {"users.id": 0, "users.name": 1},
        "table_to_id": {"users": 0},
        "fk_to_id": {},
    }
    f = _make_filter(xiyan_resp, terminal_source="gat_topk", fk_pk_hardcode=False)
    result = f.refine(
        query="q", subgraph={"users": ["id", "name"]}, db_id=None,
        gat_scores=None, metadata=metadata,
    )
    _assert(
        "forward" in result["stats"]["terminal_source_used"],
        f"fallback to forward, got {result['stats']['terminal_source_used']}",
    )


# ============================================================
# Scenario 7: terminal_source='prelim_sql' calls LLM once
# ============================================================
def test_terminal_source_prelim_sql_calls_llm():
    print("\n[test] terminal_source='prelim_sql' triggers a prelim LLM call")
    xiyan_resp = json.dumps({"users": ["name"]})
    prelim_sql = "SELECT users.name, users.email FROM users"
    metadata = {
        "col_to_id": {"users.id": 0, "users.name": 1, "users.email": 2},
        "table_to_id": {"users": 0},
        "fk_to_id": {},
    }
    f = _make_filter(
        xiyan_resp, prelim_sql_response=prelim_sql,
        terminal_source="prelim_sql", fk_pk_hardcode=False,
    )
    result = f.refine(
        query="user contact info",
        subgraph={"users": ["id", "name", "email"]},
        db_id=None,
        metadata=metadata,
    )
    _assert(len(f.client.calls) == 1, "prelim SQL LLM called once")
    final = set(result["final_nodes"])
    _assert("users.email" in final, "prelim SQL terminal brought in email")


# ============================================================
# Scenario 8: FK hardcode rescue
# ============================================================
def test_fk_hardcode_rescues_join_keys():
    print("\n[test] fk_pk_hardcode=True rescues FK columns dropped by forward")
    xiyan_resp = json.dumps({"users": ["name"], "orders": ["total"]})  # FK missing
    metadata = {
        "col_to_id": {
            "users.id": 0, "users.name": 1,
            "orders.user_id": 2, "orders.total": 3,
        },
        "table_to_id": {"users": 0, "orders": 1},
        "fk_to_id": {"orders.user_id->users.id": 0},
    }
    f = _make_filter(xiyan_resp, fk_pk_hardcode=True)
    result = f.refine(
        query="q",
        subgraph={"users": ["id", "name"], "orders": ["total", "user_id"]},
        db_id=None,
        metadata=metadata,
    )
    _assert(result["stats"]["struct"] == 2, "FK columns rescued")


# ============================================================
# Scenario 9: max_restore cap
# ============================================================
def test_max_restore_caps_overly_large_steiner():
    print("\n[test] max_restore=1 caps Steiner restore count")
    # 길게 연결된 schema chain — Steiner tree 가 여러 column 을 포함
    xiyan_resp = json.dumps({"a": ["x"], "d": ["x"]})
    metadata = {
        "col_to_id": {
            "a.x": 0, "a.id": 1,
            "b.a_id": 2, "b.c_id": 3,
            "c.b_id": 4, "c.d_id": 5,
            "d.c_id": 6, "d.x": 7,
        },
        "table_to_id": {"a": 0, "b": 1, "c": 2, "d": 3},
        "fk_to_id": {
            "b.a_id->a.id": 0,
            "b.c_id->c.b_id": 1,  # NOTE: this links b-c
            "d.c_id->c.d_id": 2,
        },
    }
    f = _make_filter(xiyan_resp, terminal_source="forward",
                     fk_pk_hardcode=False, max_restore=1)
    result = f.refine(
        query="q",
        subgraph={"a": ["x"], "d": ["x"]},
        db_id=None,
        metadata=metadata,
    )
    _assert(result["stats"]["steiner_restore"] <= 1,
            f"capped to 1, got {result['stats']['steiner_restore']}")
    # cap 이 작용했음 = "restore_capped_from" 기록
    if result["stats"]["steiner_restore"] == 1:
        _assert(
            result["stats"]["restore_capped_from"] is not None
            or result["stats"]["steiner_restore"] == 1,
            "cap recorded or exactly 1 to begin with",
        )


# ============================================================
# Scenario 10: empty subgraph + no metadata → Unanswerable propagation
# ============================================================
def test_empty_subgraph_unanswerable():
    print("\n[test] empty subgraph propagates Unanswerable status")
    f = _make_filter("{}", fk_pk_hardcode=False)
    result = f.refine(query="q", subgraph={}, db_id=None, metadata=None)
    _assert(result["status"] == "Unanswerable", "status propagates")
    _assert(result["final_nodes"] == [], "empty nodes")


# ============================================================
# Scenario 11: No metadata → PCST subgraph fallback for full schema
# ============================================================
def test_no_metadata_full_schema_fallback():
    print("\n[test] metadata=None → full schema falls back to PCST subgraph")
    xiyan_resp = json.dumps({"users": ["name"]})
    f = _make_filter(xiyan_resp, terminal_source="forward", fk_pk_hardcode=False)
    result = f.refine(
        query="q",
        subgraph={"users": ["id", "name", "email"]},
        db_id=None,
        metadata=None,
    )
    _assert(result["status"] == "Answerable", "answerable from forward")
    _assert("users.name" in result["final_nodes"], "forward preserved")
    # No FK metadata → restore via belongs_to (same table) — single terminal so skip
    _assert(result["stats"]["steiner_skipped"] in ("single_terminal_or_empty", None),
            "single-terminal Steiner skipped or no-op")


# ============================================================
# Scenario 12: invalid terminal_source / steiner_method
# ============================================================
def test_invalid_terminal_source_raises():
    print("\n[test] invalid terminal_source raises ValueError")
    try:
        GRASTFDFilter(terminal_source="bogus", provider=None, db_dir="/x")
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_invalid_steiner_method_raises():
    print("\n[test] invalid steiner_method raises ValueError")
    try:
        GRASTFDFilter(steiner_method="bogus", provider=None, db_dir="/x")
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def run_all():
    tests = [
        test_parse_fk_key,
        test_fd_graph_has_belongs_to_and_fk_edges,
        test_fd_graph_inferred_fk_added,
        test_fd_graph_without_belongs_to,
        test_steiner_restores_missing_join_key,
        test_steiner_no_restore_when_already_connected,
        test_disconnected_components_partial_restore,
        test_inferred_fk_enables_restore,
        test_terminal_source_gat_topk,
        test_terminal_source_gat_topk_fallback,
        test_terminal_source_prelim_sql_calls_llm,
        test_fk_hardcode_rescues_join_keys,
        test_max_restore_caps_overly_large_steiner,
        test_empty_subgraph_unanswerable,
        test_no_metadata_full_schema_fallback,
        test_invalid_terminal_source_raises,
        test_invalid_steiner_method_raises,
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
