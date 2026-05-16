"""Smoke tests for ConditionalFilterWrapper (Phase 4.2, TCR-gated voluntary skip).

Inner Filter 는 mock (호출 횟수 측정) 으로 교체.
"""
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("VLLM_API_KEY", "dummy")

from modules.registry import register, build  # noqa: E402
from modules.base import BaseFilter  # noqa: E402
from modules.filters.conditional_filter_wrapper import (  # noqa: E402
    ConditionalFilterWrapper,
)


@register("filter", "_MockCountingFilter")
class _MockCountingFilter(BaseFilter):
    """Returns a prescribed final_nodes; counts refine() invocations."""

    _instances: Dict[str, "_MockCountingFilter"] = {}

    def __init__(
        self,
        nodes: List[str],
        status: str = "Answerable",
        tag: str = "default",
        **kwargs,
    ):
        self._nodes = list(nodes)
        self._status = status
        self.calls = 0
        self.tag = tag
        type(self)._instances[tag] = self

    def refine(self, query, subgraph, db_id=None, **kwargs):
        self.calls += 1
        return {
            "status": self._status,
            "final_nodes": list(self._nodes),
            "reasoning": f"mock {self.tag} call #{self.calls}",
            "filter_info": {"filter_type": "_MockCountingFilter",
                            "mock_tag": self.tag,
                            "mock_calls_so_far": self.calls},
        }


def _make_wrapper(call_mode: str = "conditional", tcr_threshold: float = 0.5,
                  inner_nodes=None, inner_tag: str = "default",
                  inner_status: str = "Answerable") -> ConditionalFilterWrapper:
    return ConditionalFilterWrapper(
        inner_filter={
            "name": "_MockCountingFilter",
            "params": {
                "nodes": inner_nodes or ["users.id", "users.name"],
                "status": inner_status,
                "tag": inner_tag,
            },
        },
        call_mode=call_mode,
        tcr_threshold=tcr_threshold,
    )


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


# ============================================================
# Static helpers
# ============================================================
def test_count_subgraph_columns():
    print("\n[test] _count_subgraph_columns counts table.col pairs and table-only entries")
    n = ConditionalFilterWrapper._count_subgraph_columns(
        {"users": ["id", "name"], "orders": ["total"], "loner": []}
    )
    _assert(n == 4, f"expected 4 (2 + 1 + 1 table-only), got {n}")


def test_count_full_schema_columns():
    print("\n[test] _count_full_schema_columns reads metadata['col_to_id']")
    n = ConditionalFilterWrapper._count_full_schema_columns(
        {"col_to_id": {"users.id": 0, "users.name": 1, "orders.total": 2, "skip_no_dot": 3}}
    )
    _assert(n == 3, f"expected 3 (only 'table.col' keys), got {n}")
    n2 = ConditionalFilterWrapper._count_full_schema_columns(None)
    _assert(n2 == 0, f"None metadata → 0, got {n2}")


def test_compute_tcr_uses_override_first():
    print("\n[test] tcr_override (root pre-computed) takes priority over metadata calc")
    f = _make_wrapper()
    # subgraph 4 / schema 10 → calc 0.4, override 0.9 → 0.9 used
    out = f._compute_tcr(
        subgraph={"a": ["x", "y"], "b": ["z", "w"]},
        metadata={"col_to_id": {f"t{i}.c": i for i in range(10)}},
        tcr_override=0.9,
    )
    _assert(out == 0.9, f"override used, got {out}")


def test_compute_tcr_falls_back_to_calc():
    print("\n[test] no override → compute from metadata col_to_id")
    f = _make_wrapper()
    out = f._compute_tcr(
        subgraph={"a": ["x", "y"], "b": ["z"]},  # 3 cols
        metadata={"col_to_id": {f"t{i}.c": i for i in range(10)}},  # 10 cols
        tcr_override=None,
    )
    _assert(abs(out - 0.3) < 1e-9, f"3/10=0.3, got {out}")


def test_compute_tcr_clamps_to_one():
    print("\n[test] n_sub > n_full clamps to 1.0")
    f = _make_wrapper()
    out = f._compute_tcr(
        subgraph={"a": ["x"] * 100},
        metadata={"col_to_id": {"t.c": 0}},
        tcr_override=None,
    )
    _assert(out == 1.0, f"clamped, got {out}")


def test_compute_tcr_invalid_override_falls_back():
    print("\n[test] invalid override (non-numeric / out-of-range) → fall back to calc")
    f = _make_wrapper()
    out = f._compute_tcr(
        subgraph={"a": ["x", "y"]},
        metadata={"col_to_id": {f"t{i}.c": i for i in range(4)}},
        tcr_override="garbage",
    )
    _assert(abs(out - 0.5) < 1e-9, f"calc used, got {out}")


def test_compute_tcr_no_metadata_no_override_returns_none():
    print("\n[test] no override + no metadata → None (caller decides safe path)")
    f = _make_wrapper()
    out = f._compute_tcr(subgraph={"a": ["x"]}, metadata=None)
    _assert(out is None, f"None, got {out!r}")


# ============================================================
# Skip path
# ============================================================
def test_voluntary_skip_when_tcr_below_threshold():
    print("\n[test] TCR < threshold → skip inner, return subgraph as-is")
    f = _make_wrapper(call_mode="conditional", tcr_threshold=0.5, inner_tag="skip_test")
    result = f.refine(
        query="q",
        subgraph={"users": ["id", "name"]},  # 2 cols
        db_id=None,
        metadata={"col_to_id": {f"t{i}.c": i for i in range(10)}},  # 10 → tcr=0.2
    )
    inner = _MockCountingFilter._instances["skip_test"]
    _assert(inner.calls == 0, f"inner NOT called, got {inner.calls}")
    _assert(result["stats"]["voluntary_skipped"] is True, "voluntary_skipped flag")
    _assert(result["stats"]["inner_called"] is False, "inner_called=False")
    _assert(
        set(result["final_nodes"]) == {"users.id", "users.name"},
        f"subgraph preserved, got {result['final_nodes']}",
    )
    _assert(result["status"] == "Answerable", "status answerable from extractor output")
    _assert(result["stats"]["tcr_value"] == 0.2, f"tcr=0.2, got {result['stats']['tcr_value']}")


def test_inner_called_when_tcr_above_threshold():
    print("\n[test] TCR ≥ threshold → inner called normally")
    f = _make_wrapper(
        call_mode="conditional", tcr_threshold=0.5,
        inner_nodes=["users.id"],  # inner returns just this
        inner_tag="call_test",
    )
    result = f.refine(
        query="q",
        subgraph={"users": ["id", "name", "age"], "orders": ["total"]},  # 4 cols
        db_id=None,
        metadata={"col_to_id": {f"t{i}.c": i for i in range(5)}},  # 5 → tcr=0.8
    )
    inner = _MockCountingFilter._instances["call_test"]
    _assert(inner.calls == 1, f"inner called once, got {inner.calls}")
    _assert(result["stats"]["voluntary_skipped"] is False, "not voluntary skipped")
    _assert(result["stats"]["inner_called"] is True, "inner_called=True")
    _assert(result["final_nodes"] == ["users.id"], "inner result used")


def test_call_mode_always_never_skips():
    print("\n[test] call_mode='always' ignores TCR and always calls inner")
    f = _make_wrapper(
        call_mode="always", tcr_threshold=0.99,
        inner_nodes=["users.id"], inner_tag="always_test",
    )
    result = f.refine(
        query="q",
        subgraph={"users": ["id"]},  # 1 col
        db_id=None,
        metadata={"col_to_id": {f"t{i}.c": i for i in range(100)}},  # tcr=0.01 < 0.99
    )
    inner = _MockCountingFilter._instances["always_test"]
    _assert(inner.calls == 1, f"inner called despite low TCR, got {inner.calls}")
    _assert(result["stats"]["voluntary_skipped"] is False, "no skip in always mode")


def test_tcr_override_drives_skip_decision():
    print("\n[test] kwargs tcr override drives skip even without metadata")
    f = _make_wrapper(
        call_mode="conditional", tcr_threshold=0.5,
        inner_nodes=["x"], inner_tag="override_test",
    )
    result = f.refine(
        query="q",
        subgraph={"users": ["id", "name"]},
        db_id=None,
        metadata=None,  # no schema info — would otherwise default to call-inner
        tcr=0.2,        # override → triggers skip
    )
    inner = _MockCountingFilter._instances["override_test"]
    _assert(inner.calls == 0, "override drove skip")
    _assert(result["stats"]["voluntary_skipped"] is True, "voluntary_skipped via override")
    _assert(result["filter_info"]["filter_tcr_source"] == "override",
            "tcr_source recorded as override")


def test_no_metadata_no_override_calls_inner_safely():
    print("\n[test] TCR unknown (no metadata, no override) → call inner (safe path)")
    f = _make_wrapper(
        call_mode="conditional", tcr_threshold=0.5,
        inner_nodes=["x"], inner_tag="unknown_tcr",
    )
    result = f.refine(
        query="q", subgraph={"users": ["id"]}, db_id=None, metadata=None,
    )
    inner = _MockCountingFilter._instances["unknown_tcr"]
    _assert(inner.calls == 1, "safe path → inner called")
    _assert(result["filter_info"]["filter_tcr_source"] == "unavailable",
            "tcr_source=unavailable recorded")


def test_skip_preserves_empty_subgraph_unanswerable():
    print("\n[test] skip path with empty subgraph → Unanswerable status")
    f = _make_wrapper(call_mode="conditional", tcr_threshold=0.5, inner_tag="empty_test")
    result = f.refine(
        query="q",
        subgraph={},
        db_id=None,
        metadata={"col_to_id": {f"t{i}.c": i for i in range(10)}},  # 0/10 → tcr=0
    )
    _assert(result["stats"]["voluntary_skipped"] is True, "skipped (tcr=0)")
    _assert(result["status"] == "Unanswerable", "empty → Unanswerable")
    _assert(result["final_nodes"] == [], "no nodes")


# ============================================================
# Config / validation
# ============================================================
def test_invalid_call_mode_raises():
    print("\n[test] invalid call_mode → ValueError")
    try:
        ConditionalFilterWrapper(
            inner_filter={"name": "_MockCountingFilter", "params": {"nodes": []}},
            call_mode="bogus", tcr_threshold=0.5,
        )
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_invalid_threshold_raises():
    print("\n[test] tcr_threshold out of [0,1] → ValueError")
    try:
        ConditionalFilterWrapper(
            inner_filter={"name": "_MockCountingFilter", "params": {"nodes": []}},
            call_mode="conditional", tcr_threshold=1.5,
        )
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_yaml_style_build_works():
    print("\n[test] registry.build constructs wrapper from yaml-style config")
    cfg = {
        "name": "ConditionalFilterWrapper",
        "params": {
            "call_mode": "conditional",
            "tcr_threshold": 0.7,
            "inner_filter": {
                "name": "_MockCountingFilter",
                "params": {"nodes": ["users.id"], "tag": "yaml_build"},
            },
        },
    }
    inst = build("filter", cfg)
    _assert(isinstance(inst, ConditionalFilterWrapper), "wrapper instance")
    _assert(inst.tcr_threshold == 0.7, f"threshold persisted: {inst.tcr_threshold}")
    _assert(inst.inner_filter_name == "_MockCountingFilter", "inner name recorded")


def run_all():
    tests = [
        test_count_subgraph_columns,
        test_count_full_schema_columns,
        test_compute_tcr_uses_override_first,
        test_compute_tcr_falls_back_to_calc,
        test_compute_tcr_clamps_to_one,
        test_compute_tcr_invalid_override_falls_back,
        test_compute_tcr_no_metadata_no_override_returns_none,
        test_voluntary_skip_when_tcr_below_threshold,
        test_inner_called_when_tcr_above_threshold,
        test_call_mode_always_never_skips,
        test_tcr_override_drives_skip_decision,
        test_no_metadata_no_override_calls_inner_safely,
        test_skip_preserves_empty_subgraph_unanswerable,
        test_invalid_call_mode_raises,
        test_invalid_threshold_raises,
        test_yaml_style_build_works,
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
