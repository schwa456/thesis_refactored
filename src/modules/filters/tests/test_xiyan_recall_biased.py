"""Smoke tests for XiYanFilter Wave 6 Phase 1 — Recall-Biased Prompt 3 variants.

학술 agent filter improve plan §3 (planning/filter/0516_scholar_filter_improve_plan.md):
  - prompt_mode ∈ {default, recall_biased_mild, recall_biased_strong,
                    recall_biased_exclusion_rule}
  - sanitize_filter_output() hallucination 방지 후처리
  - 측정 logging: prompt_mode / prune_pct / hallucination_removed_count
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

from modules.filters.xiyan_filter import (  # noqa: E402
    XiYanFilter,
    _PROMPT_SECTION_BY_MODE,
)


class _CapturingMock:
    """LLM client mock — capture prompt + return prescribed response."""

    def __init__(self, response: str):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def generate_text(self, prompt: str, model: str, temperature: float, **kw) -> str:
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        return self.response


def _make_filter(
    prompt_mode: str = "default",
    sanitize: bool = True,
    mock_response: str = "{}",
) -> XiYanFilter:
    flt = XiYanFilter(
        model_name="mock-model",
        max_iteration=1,
        temperature=0.0,
        db_dir="/nonexistent",
        num_examples=0,
        provider=None,
        prompt_mode=prompt_mode,
        sanitize_output=sanitize,
    )
    flt.client = _CapturingMock(mock_response)
    return flt


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


# ============================================================
# sanitize_filter_output unit tests
# ============================================================
def test_sanitize_removes_hallucinated_table():
    print("\n[test] sanitize removes whole-table hallucinations")
    raw = {"users": ["id", "name"], "ghost": ["x"]}
    extractor = {"users": ["id", "name"], "orders": ["total"]}
    out, removed = XiYanFilter.sanitize_filter_output(raw, extractor)
    _assert(out == {"users": ["id", "name"]}, f"ghost table removed, got {out}")
    _assert(removed == 1, f"1 hallucinated entry (ghost.x), got {removed}")


def test_sanitize_removes_hallucinated_column():
    print("\n[test] sanitize removes per-column hallucinations")
    raw = {"users": ["id", "fake_col", "name"]}
    extractor = {"users": ["id", "name", "age"]}
    out, removed = XiYanFilter.sanitize_filter_output(raw, extractor)
    _assert(out == {"users": ["id", "name"]}, f"only valid cols kept, got {out}")
    _assert(removed == 1, f"1 hallucinated col (fake_col), got {removed}")


def test_sanitize_drops_empty_tables():
    print("\n[test] sanitize drops tables that lose all columns")
    raw = {"users": ["fake1", "fake2"]}
    extractor = {"users": ["id"], "orders": ["total"]}
    out, removed = XiYanFilter.sanitize_filter_output(raw, extractor)
    _assert(out == {}, f"empty table dropped, got {out}")
    _assert(removed == 2, "both hallucinated cols counted")


def test_sanitize_handles_dedup():
    print("\n[test] sanitize de-duplicates within table")
    raw = {"users": ["id", "id", "name"]}
    extractor = {"users": ["id", "name"]}
    out, removed = XiYanFilter.sanitize_filter_output(raw, extractor)
    _assert(out == {"users": ["id", "name"]}, "duplicates collapsed")
    # 두 번째 id 는 dedup 으로 무시 (hallucination 아님) — removed 변하지 않음
    _assert(removed == 0, f"no hallucinations to count, got {removed}")


def test_sanitize_handles_non_list_value():
    print("\n[test] sanitize handles malformed value (non-list)")
    raw = {"users": {"id": True}, "orders": ["total"]}
    extractor = {"users": ["id"], "orders": ["total"]}
    out, removed = XiYanFilter.sanitize_filter_output(raw, extractor)
    _assert(out == {"orders": ["total"]}, "dict value rejected, valid table kept")
    _assert(removed == 1, "dict value counted as 1 removal")


def test_sanitize_handles_non_dict_input():
    print("\n[test] sanitize returns empty when raw is not dict")
    out, removed = XiYanFilter.sanitize_filter_output(["users.id"], {"users": ["id"]})
    _assert(out == {}, "non-dict raw → empty dict")
    _assert(removed == 0, "no removal count meaningful")


# ============================================================
# prompt_mode validation
# ============================================================
def test_prompt_mode_map_has_4_modes():
    print("\n[test] _PROMPT_SECTION_BY_MODE has exactly 4 modes")
    expected = {"default", "recall_biased_mild", "recall_biased_strong",
                "recall_biased_exclusion_rule"}
    _assert(set(_PROMPT_SECTION_BY_MODE.keys()) == expected, f"keys: {sorted(_PROMPT_SECTION_BY_MODE.keys())}")


def test_invalid_prompt_mode_raises():
    print("\n[test] invalid prompt_mode → ValueError")
    try:
        XiYanFilter(model_name="x", prompt_mode="bogus", provider=None)
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


# ============================================================
# Prompt mode → correct section loaded (verified via prompt text content)
# ============================================================
def test_mode_default_loads_xiyan_section():
    print("\n[test] mode='default' loads 'xiyan_filter' section")
    flt = _make_filter(prompt_mode="default", mock_response=json.dumps({"users": ["id"]}))
    flt.refine(query="q", subgraph={"users": ["id"]}, db_id=None)
    prompt = flt.client.calls[0]["prompt"]
    _assert("strictly formatted Database Schema Filtering Agent" in prompt or
            "filter the provided schema to include ONLY" in prompt,
            "default section signature present")


def test_mode_mild_loads_recall_biased_mild_section():
    print("\n[test] mode='recall_biased_mild' loads M1-A section")
    flt = _make_filter(prompt_mode="recall_biased_mild",
                       mock_response=json.dumps({"users": ["id"]}))
    flt.refine(query="q", subgraph={"users": ["id"]}, db_id=None)
    prompt = flt.client.calls[0]["prompt"]
    _assert("WHEN IN DOUBT, INCLUDE THE COLUMN" in prompt, "M1-A signature")
    _assert("RELEVANT or POTENTIALLY RELEVANT" in prompt, "M1-A inclusion-bias phrasing")


def test_mode_strong_loads_recall_biased_strong_section():
    print("\n[test] mode='recall_biased_strong' loads M1-B section")
    flt = _make_filter(prompt_mode="recall_biased_strong",
                       mock_response=json.dumps({"users": ["id"]}))
    flt.refine(query="q", subgraph={"users": ["id"]}, db_id=None)
    prompt = flt.client.calls[0]["prompt"]
    _assert("Your default decision is INCLUDE" in prompt, "M1-B signature")
    _assert("HIGHLY CONFIDENT it has ZERO relevance" in prompt, "M1-B exclusion language")


def test_mode_exclusion_rule_loads_section():
    print("\n[test] mode='recall_biased_exclusion_rule' loads M1-C section")
    flt = _make_filter(prompt_mode="recall_biased_exclusion_rule",
                       mock_response=json.dumps({"users": ["id"]}))
    flt.refine(query="q", subgraph={"users": ["id"]}, db_id=None)
    prompt = flt.client.calls[0]["prompt"]
    _assert("4-rule" in prompt.lower() or "Rule 1:" in prompt, "M1-C 4-rule structure")
    _assert("If you are UNSURE about any of the four rules" in prompt,
            "M1-C UNSURE → KEEP rule")


# ============================================================
# Sanitize integration in refine()
# ============================================================
def test_refine_sanitizes_llm_hallucinations():
    print("\n[test] refine() removes hallucinated cols from LLM output")
    # LLM 이 fake_table 과 fake_col 을 hallucinate
    mock = json.dumps({
        "users": ["id", "name", "fake_col"],
        "fake_table": ["x", "y"],
    })
    flt = _make_filter(prompt_mode="recall_biased_mild", sanitize=True, mock_response=mock)
    result = flt.refine(query="q",
                       subgraph={"users": ["id", "name"], "orders": ["total"]},
                       db_id=None)
    final = set(result["final_nodes"])
    _assert(final == {"users.id", "users.name"}, f"hallucinations removed, got {sorted(final)}")
    _assert(
        result["filter_info"]["filter_hallucination_removed_count"] == 3,
        f"3 removed (fake_col + fake_table×2), "
        f"got {result['filter_info']['filter_hallucination_removed_count']}",
    )


def test_refine_disabled_sanitize_keeps_hallucinations():
    print("\n[test] sanitize_output=False bypasses removal (backward-compat path)")
    mock = json.dumps({"users": ["id"], "fake_table": ["x"]})
    flt = _make_filter(prompt_mode="default", sanitize=False, mock_response=mock)
    result = flt.refine(query="q", subgraph={"users": ["id"]}, db_id=None)
    final = set(result["final_nodes"])
    _assert("fake_table.x" in final, "fake_table.x survives without sanitize")
    _assert(
        result["filter_info"]["filter_hallucination_removed_count"] == 0,
        "no removal when disabled",
    )
    _assert(
        result["filter_info"]["filter_sanitize_output"] is False,
        "sanitize_output flag recorded",
    )


# ============================================================
# Measurement metadata
# ============================================================
def test_metadata_records_prompt_mode_and_prune_pct():
    print("\n[test] filter_info exposes prompt_mode, prune_pct, input/output counts")
    mock = json.dumps({"users": ["id"]})  # 4 in → 1 out
    flt = _make_filter(prompt_mode="recall_biased_strong", mock_response=mock)
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name", "age", "email"]},
        db_id=None,
    )
    info = result["filter_info"]
    _assert(info["filter_prompt_mode"] == "recall_biased_strong", "mode recorded")
    _assert(info["filter_input_node_count"] == 4, "input count = 4")
    _assert(info["filter_output_node_count"] == 1, "output count = 1")
    _assert(abs(info["filter_prune_pct"] - 0.75) < 1e-9,
            f"prune_pct = 3/4 = 0.75, got {info['filter_prune_pct']}")


def test_metadata_records_zero_hallucination_for_clean_output():
    print("\n[test] clean LLM output → hallucination_removed_count=0")
    mock = json.dumps({"users": ["id", "name"]})
    flt = _make_filter(prompt_mode="recall_biased_mild", mock_response=mock)
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name", "age"]},
        db_id=None,
    )
    _assert(result["filter_info"]["filter_hallucination_removed_count"] == 0,
            "no hallucinations")


def test_empty_subgraph_metadata_includes_mode():
    print("\n[test] empty subgraph short-circuit retains mode metadata")
    flt = _make_filter(prompt_mode="recall_biased_exclusion_rule")
    result = flt.refine(query="q", subgraph={}, db_id=None)
    _assert(result["status"] == "Unanswerable", "Unanswerable status")
    _assert(result["filter_info"]["filter_prompt_mode"] == "recall_biased_exclusion_rule",
            "mode recorded even on short-circuit")
    _assert(result["filter_info"]["filter_prune_pct"] == 0.0, "prune_pct=0 on empty")


# ============================================================
# Default backward-compat
# ============================================================
def test_default_mode_backward_compat():
    print("\n[test] default mode preserves existing behavior + records new metadata")
    mock = json.dumps({"users": ["id"]})
    flt = _make_filter(prompt_mode="default", mock_response=mock)
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"]},
        db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id"}, "selection unchanged")
    _assert(result["filter_info"]["filter_prompt_mode"] == "default", "mode=default")
    _assert(result["filter_info"]["filter_sanitize_output"] is True, "sanitize default-on")


def run_all():
    tests = [
        test_sanitize_removes_hallucinated_table,
        test_sanitize_removes_hallucinated_column,
        test_sanitize_drops_empty_tables,
        test_sanitize_handles_dedup,
        test_sanitize_handles_non_list_value,
        test_sanitize_handles_non_dict_input,
        test_prompt_mode_map_has_4_modes,
        test_invalid_prompt_mode_raises,
        test_mode_default_loads_xiyan_section,
        test_mode_mild_loads_recall_biased_mild_section,
        test_mode_strong_loads_recall_biased_strong_section,
        test_mode_exclusion_rule_loads_section,
        test_refine_sanitizes_llm_hallucinations,
        test_refine_disabled_sanitize_keeps_hallucinations,
        test_metadata_records_prompt_mode_and_prune_pct,
        test_metadata_records_zero_hallucination_for_clean_output,
        test_empty_subgraph_metadata_includes_mode,
        test_default_mode_backward_compat,
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
