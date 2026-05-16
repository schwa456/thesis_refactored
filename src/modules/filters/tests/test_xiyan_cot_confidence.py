"""Smoke tests for XiYanFilter Wave 6 Phase 2 (a) — CoT + Confidence-Gated.

학술 agent filter improve plan §4 + DECISIONS 2026-05-16 Phase 2 (a) Spec.
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
    _COT_PROMPT_SECTION_BY_MODE,
    _threshold_to_gate_level,
)


class _CapturingMock:
    def __init__(self, response: str):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def generate_text(self, prompt: str, model: str, temperature: float, **kw) -> str:
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        return self.response


def _make_filter(
    prompt_mode: str = "recall_biased_strong",
    cot_reasoning: bool = True,
    confidence_gated: bool = True,
    confidence_threshold: float = 0.5,
    gate_level=None,
    sanitize: bool = True,
    mock_response: str = "",
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
        cot_reasoning=cot_reasoning,
        confidence_gated=confidence_gated,
        confidence_threshold=confidence_threshold,
        gate_level=gate_level,
    )
    flt.client = _CapturingMock(mock_response)
    return flt


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


# ============================================================
# threshold ↔ gate_level mapping
# ============================================================
def test_threshold_to_gate_level_mapping():
    print("\n[test] _threshold_to_gate_level continuous → categorical mapping")
    _assert(_threshold_to_gate_level(0.0) == "none", "0.0 → none")
    _assert(_threshold_to_gate_level(-1) == "none", "<= 0 → none")
    _assert(_threshold_to_gate_level(0.2) == "high", "(0, 0.3) → high (override all)")
    _assert(_threshold_to_gate_level(0.5) == "medium",
            "DECISIONS spec 0.5 → medium (low+medium)")
    _assert(_threshold_to_gate_level(0.7) == "medium", "[0.3, 0.8) → medium")
    _assert(_threshold_to_gate_level(0.8) == "low", ">= 0.8 → low (low only)")
    _assert(_threshold_to_gate_level(0.95) == "low", "near 1 → low")


def test_invalid_gate_level_raises():
    print("\n[test] explicit invalid gate_level → ValueError")
    try:
        XiYanFilter(model_name="x", provider=None, gate_level="bogus")
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_invalid_cot_mode_pairing_raises():
    print("\n[test] cot_reasoning=True with unsupported prompt_mode → ValueError")
    try:
        XiYanFilter(
            model_name="x", provider=None,
            prompt_mode="recall_biased_mild",  # no CoT pairing yet
            cot_reasoning=True,
        )
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised for unsupported pairing")


def test_cot_prompt_section_map():
    print("\n[test] _COT_PROMPT_SECTION_BY_MODE has 'default' + 'recall_biased_strong'")
    _assert(set(_COT_PROMPT_SECTION_BY_MODE.keys()) == {"default", "recall_biased_strong"},
            f"keys: {sorted(_COT_PROMPT_SECTION_BY_MODE.keys())}")


# ============================================================
# parse_cot_output
# ============================================================
def test_parse_cot_split_and_parse():
    print("\n[test] parse_cot_output splits ---JSON--- and parses Section 2")
    raw = (
        "I reasoned about the question.\n"
        "It needs users.id and orders.total.\n"
        "---JSON---\n"
        '{"users": {"id": {"include": true, "confidence": "high"}}, '
        '"orders": {"total": {"include": true, "confidence": "medium"}}}'
    )
    reasoning, parsed = XiYanFilter.parse_cot_output(raw)
    _assert("reasoned about the question" in reasoning, "reasoning captured")
    _assert("users" in parsed and "orders" in parsed, "both tables parsed")
    _assert(parsed["users"]["id"]["confidence"] == "high", "confidence preserved")


def test_parse_cot_handles_case_insensitive_separator():
    print("\n[test] parse_cot_output handles ---json--- (case-insensitive)")
    raw = 'reasoning\n---json---\n{"t": {"c": {"include": true, "confidence": "low"}}}'
    _, parsed = XiYanFilter.parse_cot_output(raw)
    _assert("t" in parsed, "lower-case separator handled")


def test_parse_cot_handles_markdown_fence():
    print("\n[test] parse_cot_output strips ```json fence after separator")
    raw = (
        "reasoning text\n"
        "---JSON---\n"
        '```json\n{"t": {"c": {"include": true, "confidence": "high"}}}\n```'
    )
    _, parsed = XiYanFilter.parse_cot_output(raw)
    _assert("t" in parsed, "fence stripped + parsed")


def test_parse_cot_empty_input_returns_empty():
    print("\n[test] empty / None raw returns ('', {})")
    r, p = XiYanFilter.parse_cot_output("")
    _assert(r == "" and p == {}, "empty string")
    r, p = XiYanFilter.parse_cot_output(None)
    _assert(r == "" and p == {}, "None")


def test_parse_cot_malformed_json_returns_empty():
    print("\n[test] malformed JSON after separator → ({reasoning}, {})")
    raw = "reasoning\n---JSON---\n{not json}"
    r, p = XiYanFilter.parse_cot_output(raw)
    _assert("reasoning" in r, "reasoning still captured")
    _assert(p == {}, "JSON empty on parse failure")


# ============================================================
# apply_confidence_gating
# ============================================================
def test_gating_low_overrides_only_low_confidence_excludes():
    print("\n[test] gate_level='low' overrides include=False+low only")
    cot = {
        "users": {
            "id": {"include": True, "confidence": "high"},
            "name": {"include": False, "confidence": "low"},   # override → include
            "age": {"include": False, "confidence": "medium"}, # NOT overridden
            "email": {"include": False, "confidence": "high"}, # NOT overridden
        }
    }
    final, n_over, dist = XiYanFilter.apply_confidence_gating(cot, gate_level="low")
    _assert(set(final["users"]) == {"id", "name"}, f"id + name (overridden), got {final}")
    _assert(n_over == 1, f"1 override, got {n_over}")
    _assert(dist == {"high": 2, "medium": 1, "low": 1}, f"distribution: {dist}")


def test_gating_medium_overrides_low_and_medium():
    print("\n[test] gate_level='medium' overrides low+medium include=False")
    cot = {
        "t": {
            "a": {"include": False, "confidence": "low"},
            "b": {"include": False, "confidence": "medium"},
            "c": {"include": False, "confidence": "high"},
            "d": {"include": True, "confidence": "high"},
        }
    }
    final, n_over, _ = XiYanFilter.apply_confidence_gating(cot, gate_level="medium")
    _assert(set(final["t"]) == {"a", "b", "d"}, f"a + b + d, got {final}")
    _assert(n_over == 2, f"2 overrides (low + medium), got {n_over}")


def test_gating_none_uses_include_only():
    print("\n[test] gate_level='none' → only include=True kept (no override)")
    cot = {"t": {
        "a": {"include": True, "confidence": "low"},
        "b": {"include": False, "confidence": "low"},
    }}
    final, n_over, _ = XiYanFilter.apply_confidence_gating(cot, gate_level="none")
    _assert(set(final["t"]) == {"a"}, "only include=True kept")
    _assert(n_over == 0, "no overrides applied")


def test_gating_high_overrides_all_excludes():
    print("\n[test] gate_level='high' overrides all confidence levels")
    cot = {"t": {
        "a": {"include": False, "confidence": "low"},
        "b": {"include": False, "confidence": "medium"},
        "c": {"include": False, "confidence": "high"},
    }}
    final, n_over, _ = XiYanFilter.apply_confidence_gating(cot, gate_level="high")
    _assert(set(final["t"]) == {"a", "b", "c"}, "all overridden")
    _assert(n_over == 3, "3 overrides")


def test_gating_handles_malformed_entries():
    print("\n[test] gating handles malformed entries gracefully")
    cot = {
        "t": {
            "a": {"include": True, "confidence": "weird_level"},  # → "high" fallback
            "b": "not a dict",                                     # skip
            123: {"include": True, "confidence": "high"},          # skip non-str col
        },
        "other": "not a dict",  # skip
    }
    final, n_over, dist = XiYanFilter.apply_confidence_gating(cot, gate_level="low")
    _assert(final == {"t": ["a"]}, f"only valid 'a' kept, got {final}")
    _assert(dist["high"] >= 1, "malformed confidence → 'high' fallback distribution")


# ============================================================
# Integration: refine() with CoT + Confidence-Gated
# ============================================================
def test_refine_cot_with_gating_overrides_low_conf_exclusions():
    print("\n[test] refine() CoT + gated overrides low-conf excludes (Phase 2 (a))")
    mock = (
        "Reasoning: total per user.\n"
        "---JSON---\n"
        '{"users": {"id": {"include": true, "confidence": "high"}, '
        '"name": {"include": false, "confidence": "low"}}, '
        '"orders": {"total": {"include": true, "confidence": "medium"}}}'
    )
    flt = _make_filter(
        prompt_mode="recall_biased_strong",
        cot_reasoning=True, confidence_gated=True,
        confidence_threshold=0.8,  # → gate_level="low" → override low only
        mock_response=mock,
    )
    result = flt.refine(
        query="total per user",
        subgraph={"users": ["id", "name"], "orders": ["total"]},
        db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id", "users.name", "orders.total"},
            f"low-conf override applied, got {sorted(final)}")
    info = result["filter_info"]
    _assert(info["filter_cot_reasoning"] is True, "cot flag")
    _assert(info["filter_confidence_gated"] is True, "gated flag")
    _assert(info["filter_gate_level"] == "low", f"gate_level=low, got {info['filter_gate_level']}")
    _assert(info["filter_gated_override_count"] == 1, "1 override")
    _assert(info["filter_confidence_distribution"] == {"high": 1, "medium": 1, "low": 1},
            f"distribution: {info['filter_confidence_distribution']}")
    _assert(info["filter_raw_filter_count"] == 3, "raw=3 columns considered")
    _assert(info["filter_final_filter_count"] == 3, "final=3 after gate")


def test_refine_cot_without_gating_keeps_include_only():
    print("\n[test] refine() CoT + gating=False → include=True only")
    mock = (
        "reasoning\n---JSON---\n"
        '{"users": {"id": {"include": true, "confidence": "high"}, '
        '"name": {"include": false, "confidence": "low"}}}'
    )
    flt = _make_filter(
        prompt_mode="default", cot_reasoning=True, confidence_gated=False,
        mock_response=mock,
    )
    result = flt.refine(
        query="q", subgraph={"users": ["id", "name"]}, db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id"}, f"no override; include=True only, got {sorted(final)}")
    info = result["filter_info"]
    _assert(info["filter_gated_override_count"] == 0, "no overrides when gating off")
    _assert(info["filter_confidence_distribution"]["high"] == 1, "distribution still measured")
    _assert(info["filter_confidence_distribution"]["low"] == 1, "low counted")


def test_refine_cot_uses_recall_biased_strong_section():
    print("\n[test] CoT + recall_biased_strong loads 'cot_recall_biased_strong' section")
    mock = "x\n---JSON---\n{}"
    flt = _make_filter(
        prompt_mode="recall_biased_strong",
        cot_reasoning=True, confidence_gated=True,
        mock_response=mock,
    )
    flt.refine(query="q", subgraph={"users": ["id"]}, db_id=None)
    prompt = flt.client.calls[0]["prompt"]
    _assert("Your default decision is INCLUDE" in prompt, "M1-B inclusion rule present")
    _assert("Reasoning Steps" in prompt, "CoT reasoning structure present")
    _assert("---JSON---" in prompt, "section separator present")


def test_refine_cot_uses_default_section_when_prompt_mode_default():
    print("\n[test] CoT + default mode uses 'cot_default' section")
    mock = "x\n---JSON---\n{}"
    flt = _make_filter(
        prompt_mode="default", cot_reasoning=True, confidence_gated=True,
        mock_response=mock,
    )
    flt.refine(query="q", subgraph={"users": ["id"]}, db_id=None)
    prompt = flt.client.calls[0]["prompt"]
    _assert("Reasoning Steps" in prompt, "cot_default present")
    _assert("default decision is INCLUDE" not in prompt,
            "no M1-B inclusion rule in cot_default")


def test_refine_cot_sanitizes_hallucinated_columns():
    print("\n[test] CoT + sanitize removes LLM-hallucinated cols not in subgraph")
    mock = (
        "x\n---JSON---\n"
        '{"users": {"id": {"include": true, "confidence": "high"}, '
        '"fake_col": {"include": true, "confidence": "high"}}, '
        '"fake_table": {"x": {"include": true, "confidence": "high"}}}'
    )
    flt = _make_filter(
        prompt_mode="default",
        cot_reasoning=True, confidence_gated=True,
        sanitize=True, mock_response=mock,
    )
    result = flt.refine(
        query="q", subgraph={"users": ["id"]}, db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id"}, f"hallucinations removed, got {sorted(final)}")
    _assert(result["filter_info"]["filter_hallucination_removed_count"] == 2,
            f"2 hallucinated (fake_col + fake_table.x), got "
            f"{result['filter_info']['filter_hallucination_removed_count']}")


def test_refine_cot_unparseable_response_records_parse_error():
    print("\n[test] CoT response without ---JSON--- and no valid JSON → parse_errors+=1")
    mock = "just plain reasoning, no JSON"
    flt = _make_filter(
        prompt_mode="default", cot_reasoning=True, confidence_gated=True,
        mock_response=mock,
    )
    result = flt.refine(
        query="q", subgraph={"users": ["id"]}, db_id=None,
    )
    _assert(result["filter_info"]["filter_parse_errors"] >= 1,
            f"parse_errors recorded, got {result['filter_info']['filter_parse_errors']}")
    # current_schema 가 변하지 않았으므로 subgraph 그대로 보존 — recall-safe
    _assert("users.id" in result["final_nodes"], "subgraph preserved on parse fail")


def test_refine_non_cot_path_still_works():
    print("\n[test] cot_reasoning=False uses legacy JSON-only path (backward compat)")
    mock = json.dumps({"users": ["id"]})
    flt = _make_filter(
        prompt_mode="recall_biased_strong",
        cot_reasoning=False, confidence_gated=False,
        mock_response=mock,
    )
    result = flt.refine(
        query="q", subgraph={"users": ["id", "name"]}, db_id=None,
    )
    _assert(set(result["final_nodes"]) == {"users.id"}, "legacy path unchanged")
    info = result["filter_info"]
    _assert(info["filter_cot_reasoning"] is False, "cot flag false")
    _assert(info["filter_confidence_distribution"] == {"high": 0, "medium": 0, "low": 0},
            "no CoT distribution recorded in non-CoT path")


def test_refine_phase2a_spec_exact():
    print("\n[test] Phase 2 (a) full spec: recall_biased_strong + cot + gated + thr=0.5")
    mock = (
        "reasoning\n---JSON---\n"
        '{"users": {"id": {"include": true, "confidence": "high"}, '
        '"flag": {"include": false, "confidence": "medium"}}}'
    )
    flt = _make_filter(
        prompt_mode="recall_biased_strong",
        cot_reasoning=True, confidence_gated=True,
        confidence_threshold=0.5,  # DECISIONS Phase 2 (a) spec → "medium"
        mock_response=mock,
    )
    result = flt.refine(
        query="q", subgraph={"users": ["id", "flag"]}, db_id=None,
    )
    info = result["filter_info"]
    _assert(info["filter_gate_level"] == "medium",
            "threshold=0.5 → 'medium' (DECISIONS spec)")
    # medium override → flag included
    _assert(set(result["final_nodes"]) == {"users.id", "users.flag"},
            f"medium override include applied, got {sorted(result['final_nodes'])}")
    _assert(info["filter_gated_override_count"] == 1, "1 override (flag)")


def run_all():
    tests = [
        test_threshold_to_gate_level_mapping,
        test_invalid_gate_level_raises,
        test_invalid_cot_mode_pairing_raises,
        test_cot_prompt_section_map,
        test_parse_cot_split_and_parse,
        test_parse_cot_handles_case_insensitive_separator,
        test_parse_cot_handles_markdown_fence,
        test_parse_cot_empty_input_returns_empty,
        test_parse_cot_malformed_json_returns_empty,
        test_gating_low_overrides_only_low_confidence_excludes,
        test_gating_medium_overrides_low_and_medium,
        test_gating_none_uses_include_only,
        test_gating_high_overrides_all_excludes,
        test_gating_handles_malformed_entries,
        test_refine_cot_with_gating_overrides_low_conf_exclusions,
        test_refine_cot_without_gating_keeps_include_only,
        test_refine_cot_uses_recall_biased_strong_section,
        test_refine_cot_uses_default_section_when_prompt_mode_default,
        test_refine_cot_sanitizes_hallucinated_columns,
        test_refine_cot_unparseable_response_records_parse_error,
        test_refine_non_cot_path_still_works,
        test_refine_phase2a_spec_exact,
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
