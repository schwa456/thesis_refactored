"""Standalone tests for SymbolicVerifierFilter (no pytest dep).

Run from project root:
    conda run -n base python src/modules/filters/tests/test_symbolic_verifier.py
"""
import os
import sys
from typing import Dict, List, Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from modules.registry import register
from modules.base import BaseFilter


@register("filter", "_MockBaseFilter")
class _MockBaseFilter(BaseFilter):
    """Returns a prescribed final_nodes list for deterministic testing."""

    def __init__(self, nodes: List[str], status: str = "Answerable", **kwargs):
        self._nodes = list(nodes)
        self._status = status

    def refine(self, query, subgraph, db_id=None, **kwargs):
        return {
            "status": self._status,
            "final_nodes": list(self._nodes),
            "reasoning": "mock",
        }


from modules.filters.symbolic_verifier_filter import SymbolicVerifierFilter  # noqa: E402


def _make_metadata(fks: List[str]) -> Dict[str, Any]:
    return {"fk_to_id": {fk: i for i, fk in enumerate(fks)}}


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


def test_single_table_is_trivially_connected():
    print("\n[test] single table trivially connected")
    f = SymbolicVerifierFilter(
        base_filter={"name": "_MockBaseFilter", "params": {"nodes": ["t1.a", "t1.b"]}},
        auto_repair=True,
    )
    result = f.refine(
        query="q", subgraph={"t1": ["a", "b"]}, metadata=_make_metadata([]),
    )
    _assert(result["connectivity_valid"] is True, "valid=True for single table")
    _assert(result["final_nodes"] == ["t1.a", "t1.b"], "final_nodes unchanged")


def test_two_tables_direct_fk_are_connected():
    print("\n[test] two tables connected via direct FK")
    f = SymbolicVerifierFilter(
        base_filter={"name": "_MockBaseFilter", "params": {"nodes": ["t1.a", "t2.x"]}},
        auto_repair=True,
    )
    meta = _make_metadata(["t1.a->t2.x"])
    result = f.refine(query="q", subgraph={"t1": ["a"], "t2": ["x"]}, metadata=meta)
    _assert(result["connectivity_valid"] is True, "valid=True for direct FK")
    _assert(sorted(result["final_nodes"]) == ["t1.a", "t2.x"], "no repair applied")


def test_disconnected_no_fk_at_all():
    print("\n[test] two tables truly disconnected (no FK)")
    f = SymbolicVerifierFilter(
        base_filter={"name": "_MockBaseFilter", "params": {"nodes": ["t1.a", "t2.x"]}},
        auto_repair=True,
    )
    meta = _make_metadata([])
    result = f.refine(query="q", subgraph={"t1": ["a"], "t2": ["x"]}, metadata=meta)
    _assert(result["connectivity_valid"] is False, "valid=False when no FK exists")
    _assert(
        any("no_fk_path" in issue for issue in result.get("connectivity_issues", [])),
        "emits no_fk_path issue",
    )
    _assert(
        set(result["final_nodes"]) == {"t1.a", "t2.x"},
        "no phantom repair when unreachable",
    )


def test_repair_inserts_bridge_table():
    print("\n[test] induced-disconnected but connected via bridge")
    f = SymbolicVerifierFilter(
        base_filter={"name": "_MockBaseFilter", "params": {"nodes": ["A.id", "C.id"]}},
        auto_repair=True,
        add_fk_columns=True,
    )
    meta = _make_metadata(["A.id->B.a_id", "B.c_id->C.id"])
    result = f.refine(
        query="q", subgraph={"A": ["id"], "C": ["id"]}, metadata=meta,
    )
    _assert(result["connectivity_valid"] is False, "valid=False before repair")
    _assert(result["status"] == "repaired", "status=repaired after bridge inserted")
    nodes_set = set(result["final_nodes"])
    _assert(
        {"A.id", "C.id"}.issubset(nodes_set),
        "base selection preserved after repair",
    )
    _assert(
        "B.a_id" in nodes_set and "B.c_id" in nodes_set,
        "FK join columns of bridge added",
    )
    _assert(
        "B" not in result["repair_added_tables"]
        or "B.a_id" in nodes_set,
        "bridge table covered by its FK columns",
    )


def test_detect_only_does_not_modify_nodes():
    print("\n[test] detect-only preserves final_nodes on disconnected input")
    f = SymbolicVerifierFilter(
        base_filter={"name": "_MockBaseFilter", "params": {"nodes": ["A.id", "C.id"]}},
        auto_repair=False,
    )
    meta = _make_metadata(["A.id->B.a_id", "B.c_id->C.id"])
    result = f.refine(
        query="q", subgraph={"A": ["id"], "C": ["id"]}, metadata=meta,
    )
    _assert(result["connectivity_valid"] is False, "flagged as invalid")
    _assert(
        sorted(result["final_nodes"]) == ["A.id", "C.id"],
        "detect-only mode leaves final_nodes untouched",
    )
    _assert(result.get("status") != "repaired", "status not set to repaired in detect-only")


def test_three_disjoint_components_partial_repair():
    print("\n[test] three components, two reachable one not")
    f = SymbolicVerifierFilter(
        base_filter={
            "name": "_MockBaseFilter",
            "params": {"nodes": ["A.id", "C.id", "Z.id"]},
        },
        auto_repair=True,
        max_bridges=5,
    )
    meta = _make_metadata(["A.id->B.a_id", "B.c_id->C.id"])
    result = f.refine(
        query="q", subgraph={"A": ["id"], "C": ["id"], "Z": ["id"]}, metadata=meta,
    )
    _assert(result["connectivity_valid"] is False, "valid=False")
    _assert(
        any("no_fk_path" in s for s in result.get("connectivity_issues", [])),
        "Z flagged as unreachable",
    )
    nodes_set = set(result["final_nodes"])
    _assert(
        "B.a_id" in nodes_set or "B.c_id" in nodes_set,
        "partial repair between A-C still applied",
    )


def test_max_bridges_budget():
    print("\n[test] max_bridges=0 suppresses repair additions")
    f = SymbolicVerifierFilter(
        base_filter={"name": "_MockBaseFilter", "params": {"nodes": ["A.id", "C.id"]}},
        auto_repair=True,
        max_bridges=0,
    )
    meta = _make_metadata(["A.id->B.a_id", "B.c_id->C.id"])
    result = f.refine(
        query="q", subgraph={"A": ["id"], "C": ["id"]}, metadata=meta,
    )
    _assert(result["connectivity_valid"] is False, "valid=False")
    _assert(
        sorted(result["final_nodes"]) == ["A.id", "C.id"],
        "no additions when budget=0",
    )


def run_all():
    tests = [
        test_single_table_is_trivially_connected,
        test_two_tables_direct_fk_are_connected,
        test_disconnected_no_fk_at_all,
        test_repair_inserts_bridge_table,
        test_detect_only_does_not_modify_nodes,
        test_three_disjoint_components_partial_repair,
        test_max_bridges_budget,
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
