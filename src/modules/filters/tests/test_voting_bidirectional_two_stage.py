"""Smoke tests for Wave 6 Phase 2 (a+aggressive) M3 + M4 + M5.

학술 agent §5/§6/§7 + DECISIONS 2026-05-16 (a+aggressive) launch §2.
Inner LLM call 은 mock — sequential 응답 list 로 prompt 순서 검증 가능.
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

from modules.filters.multi_prompt_voting_filter import (  # noqa: E402
    MultiPromptVotingFilter,
    _VALID_VOTING_STRATEGIES,
)
from modules.filters.bidirectional_filter import BidirectionalFilter  # noqa: E402
from modules.filters.two_stage_filter import TwoStageFilter  # noqa: E402


class _SequentialMock:
    """generate_text(...) — 미리 지정된 sequence 의 응답을 순서대로 반환."""

    def __init__(self, responses: List[str]):
        self._queue = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate_text(self, prompt: str, model: str, temperature: float, **kw) -> str:
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        if not self._queue:
            return ""
        return self._queue.pop(0)


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


# ============================================================
# M3 voting helper unit
# ============================================================
def test_voting_or_keeps_any_inclusion():
    print("\n[test] OR voting keeps a col if ≥1 prompt includes it")
    results = {
        "A": {"users": ["id", "name"]},
        "B": {"users": ["id"]},
        "C": {"users": ["email"]},
    }
    out = MultiPromptVotingFilter.multi_prompt_voting(results, strategy="OR")
    _assert(set(out["users"]) == {"id", "name", "email"}, f"got {out}")


def test_voting_majority_requires_two():
    print("\n[test] MAJORITY voting requires ≥2 prompts to include")
    results = {
        "A": {"users": ["id", "name"]},
        "B": {"users": ["id", "email"]},
        "C": {"users": ["name"]},
    }
    out = MultiPromptVotingFilter.multi_prompt_voting(results, strategy="MAJORITY")
    _assert(set(out["users"]) == {"id", "name"}, f"got {out}")


def test_voting_and_requires_all_three():
    print("\n[test] AND voting requires all 3 prompts")
    results = {
        "A": {"users": ["id", "name"]},
        "B": {"users": ["id", "name"]},
        "C": {"users": ["id"]},
    }
    out = MultiPromptVotingFilter.multi_prompt_voting(results, strategy="AND")
    _assert(set(out["users"]) == {"id"}, f"AND keeps only 'id', got {out}")


def test_voting_drops_empty_tables():
    print("\n[test] voting drops tables with 0 votes")
    results = {
        "A": {"users": ["id"], "orphan": ["x"]},
        "B": {"users": ["id"]},
        "C": {"users": ["name"]},
    }
    out = MultiPromptVotingFilter.multi_prompt_voting(results, strategy="AND")
    # AND: 'id' is in A+B only (votes=2 < 3); 'name' votes=1; 'x' votes=1 → no table kept
    _assert(out == {}, f"AND with insufficient votes drops everything, got {out}")


def test_voting_invalid_strategy_raises():
    print("\n[test] invalid voting strategy raises")
    try:
        MultiPromptVotingFilter.multi_prompt_voting({}, strategy="bogus")
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


# ============================================================
# M3 integration
# ============================================================
def _make_m3(responses_A_B_C: List[str], strategies=None, default="OR") -> MultiPromptVotingFilter:
    flt = MultiPromptVotingFilter(
        model_name="mock-model", provider=None, db_dir="/nonexistent",
        num_examples=0, sanitize_output=True,
        voting_strategies=strategies or list(_VALID_VOTING_STRATEGIES),
        default_voting_strategy=default,
    )
    flt.client = _SequentialMock(responses_A_B_C)
    return flt


def test_m3_refine_runs_3_prompts_and_records_all_strategies():
    print("\n[test] M3 refine calls LLM 3 times + records all voting variants in info")
    resp_A = json.dumps({"users": ["id", "name"]})
    resp_B = json.dumps({"users": ["id", "name", "email"]})
    resp_C = json.dumps({"users": ["id"]})
    flt = _make_m3([resp_A, resp_B, resp_C], default="OR")
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name", "email", "age"]},
        db_id=None,
    )
    _assert(len(flt.client.calls) == 3, f"3 LLM calls, got {len(flt.client.calls)}")
    info = result["filter_info"]
    _assert(set(info["filter_voting_strategies"]) == {"OR", "MAJORITY", "AND"},
            "3 strategies recorded")
    voted_lists = info["filter_voted_nodes"]
    _assert(set(voted_lists["OR"]) == {"users.id", "users.name", "users.email"},
            f"OR = union, got {sorted(voted_lists['OR'])}")
    _assert(set(voted_lists["MAJORITY"]) == {"users.id", "users.name"},
            f"MAJORITY ≥2, got {sorted(voted_lists['MAJORITY'])}")
    _assert(set(voted_lists["AND"]) == {"users.id"},
            f"AND ==3, got {sorted(voted_lists['AND'])}")
    _assert(set(result["final_nodes"]) == {"users.id", "users.name", "users.email"},
            "default=OR used as final_nodes")


def test_m3_sanitizes_hallucinations_per_prompt():
    print("\n[test] M3 sanitize removes hallucinated cols from each raw output")
    resp_A = json.dumps({"users": ["id", "fake_col"], "ghost": ["x"]})
    resp_B = json.dumps({"users": ["name"]})
    resp_C = json.dumps({"users": ["id"]})
    flt = _make_m3([resp_A, resp_B, resp_C], default="OR")
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"]},
        db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id", "users.name"},
            f"hallucinations removed before voting, got {sorted(final)}")
    info = result["filter_info"]
    _assert(info["filter_hallucination_removed"]["A"] == 2,
            f"A removed fake_col + ghost.x, got {info['filter_hallucination_removed']}")


def test_m3_default_must_be_in_strategies():
    print("\n[test] default_voting_strategy not in voting_strategies → ValueError")
    try:
        MultiPromptVotingFilter(
            model_name="x", provider=None,
            voting_strategies=["OR"], default_voting_strategy="AND",
        )
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_m3_empty_subgraph_short_circuit():
    print("\n[test] M3 empty subgraph short-circuits to Unanswerable, no LLM call")
    flt = _make_m3(["", "", ""])
    result = flt.refine(query="q", subgraph={}, db_id=None)
    _assert(result["status"] == "Unanswerable", "Unanswerable")
    _assert(len(flt.client.calls) == 0, "no LLM call on empty")


# ============================================================
# M4 unit + integration
# ============================================================
def test_m4_union_helper():
    print("\n[test] M4 _union_filter_outputs unions forward + backward")
    fwd = {"users": ["id"]}
    bwd = {"users": ["name"], "orders": ["total"]}
    out = BidirectionalFilter._union_filter_outputs(fwd, bwd)
    _assert(set(out["users"]) == {"id", "name"}, "users union")
    _assert(out["orders"] == ["total"], "backward-only table kept")


def test_m4_analyze_backward_contribution_with_gold():
    print("\n[test] M4 backward_contribution computes recovery + precision with gold")
    fwd = {"users": ["id"]}
    bwd = {"users": ["id", "name"], "orders": ["total"]}
    gold = {"users": ["id", "name"], "orders": ["total"]}
    stats = BidirectionalFilter.analyze_backward_contribution(fwd, bwd, gold)
    _assert(stats["backward_added"] == 2, "users.name + orders.total")
    _assert(stats["backward_gold_recovered"] == 2, "both recovered are gold")
    _assert(stats["backward_precision"] == 1.0, "precision = 1.0")


def test_m4_analyze_backward_contribution_without_gold():
    print("\n[test] M4 backward_contribution without gold → counts only, no precision")
    fwd = {"users": ["id"]}
    bwd = {"users": ["id", "noise"]}
    stats = BidirectionalFilter.analyze_backward_contribution(fwd, bwd, gold=None)
    _assert(stats["backward_added"] == 1, "1 backward-only")
    _assert("backward_precision" not in stats, "precision not reported without gold")


def _make_m4(responses_fwd_bwd: List[str]) -> BidirectionalFilter:
    flt = BidirectionalFilter(
        model_name="mock-model", provider=None, db_dir="/nonexistent",
        num_examples=0, sanitize_output=True,
    )
    flt.client = _SequentialMock(responses_fwd_bwd)
    return flt


def test_m4_refine_2_calls_and_union():
    print("\n[test] M4 refine 2 LLM calls (forward + backward) + union")
    fwd_resp = json.dumps({"users": ["id"]})
    bwd_resp = json.dumps({"users": ["id", "name"], "orders": ["total"]})
    flt = _make_m4([fwd_resp, bwd_resp])
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"], "orders": ["total"]},
        db_id=None,
    )
    _assert(len(flt.client.calls) == 2, "2 LLM calls (forward + backward)")
    final = set(result["final_nodes"])
    _assert(final == {"users.id", "users.name", "orders.total"},
            f"union of forward+backward, got {sorted(final)}")
    info = result["filter_info"]
    _assert(info["filter_backward_added"] == 2,
            f"2 backward-only cols, got {info['filter_backward_added']}")


def test_m4_with_gold_kwarg_records_precision():
    print("\n[test] M4 gold kwarg → backward_precision recorded")
    fwd_resp = json.dumps({"users": ["id"]})
    bwd_resp = json.dumps({"users": ["id", "name"]})
    flt = _make_m4([fwd_resp, bwd_resp])
    result = flt.refine(
        query="q", subgraph={"users": ["id", "name"]}, db_id=None,
        gold={"users": ["id", "name"]},
    )
    info = result["filter_info"]
    _assert(info["filter_backward_gold_recovered"] == 1, "1 gold recovered")
    _assert(info["filter_backward_precision"] == 1.0, "precision=1.0")


def test_m4_sanitizes_both_sides():
    print("\n[test] M4 sanitize removes hallucinations from forward AND backward")
    fwd_resp = json.dumps({"users": ["id", "fake_fwd"]})
    bwd_resp = json.dumps({"users": ["id", "fake_bwd"], "ghost": ["x"]})
    flt = _make_m4([fwd_resp, bwd_resp])
    result = flt.refine(
        query="q", subgraph={"users": ["id"]}, db_id=None,
    )
    info = result["filter_info"]
    _assert(info["filter_hallucination_removed_forward"] >= 1, "forward halluc removed")
    _assert(info["filter_hallucination_removed_backward"] >= 1, "backward halluc removed")
    _assert(set(result["final_nodes"]) == {"users.id"}, "final = clean union")


def test_m4_empty_subgraph_short_circuit():
    print("\n[test] M4 empty subgraph → Unanswerable, no LLM call")
    flt = _make_m4(["", ""])
    result = flt.refine(query="q", subgraph={}, db_id=None)
    _assert(result["status"] == "Unanswerable", "Unanswerable")
    _assert(len(flt.client.calls) == 0, "no LLM call")


# ============================================================
# M4 Wave 6 Phase 4 — bidirectional_forward_prompt_mode (Top 2 C1)
# ============================================================
def test_m4_forward_mode_strong_overrides_default():
    print("\n[test] bidirectional_forward_prompt_mode='recall_biased_strong' (Top 2 C1)")
    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
        sanitize_output=True,
        bidirectional_forward_prompt_mode="recall_biased_strong",
    )
    fwd_resp = json.dumps({"users": ["id"]})
    bwd_resp = json.dumps({"users": ["name"]})
    flt.client = _SequentialMock([fwd_resp, bwd_resp])
    result = flt.refine(query="q", subgraph={"users": ["id", "name"]}, db_id=None)
    _assert(flt.forward_section == "recall_biased_strong",
            f"forward_section resolved to strong, got {flt.forward_section}")
    _assert(flt.bidirectional_forward_prompt_mode == "recall_biased_strong",
            "mode persisted")
    # 첫 LLM call 의 prompt 가 M1-B strong signature 포함 확인
    fwd_prompt = flt.client.calls[0]["prompt"]
    _assert("Your default decision is INCLUDE" in fwd_prompt,
            "M1-B strong signature present in forward prompt")
    info = result["filter_info"]
    _assert(info["filter_forward_prompt_mode"] == "recall_biased_strong",
            "mode recorded in filter_info")


def test_m4_forward_mode_mild_explicit():
    print("\n[test] explicit 'recall_biased_mild' loads M1-A signature")
    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
        bidirectional_forward_prompt_mode="recall_biased_mild",
    )
    flt.client = _SequentialMock([json.dumps({"users": ["id"]}), json.dumps({})])
    flt.refine(query="q", subgraph={"users": ["id"]}, db_id=None)
    fwd_prompt = flt.client.calls[0]["prompt"]
    _assert("WHEN IN DOUBT, INCLUDE THE COLUMN" in fwd_prompt,
            "M1-A mild signature present")


def test_m4_forward_mode_exclusion_rule():
    print("\n[test] explicit 'recall_biased_exclusion_rule' loads M1-C 4-rule")
    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
        bidirectional_forward_prompt_mode="recall_biased_exclusion_rule",
    )
    flt.client = _SequentialMock([json.dumps({"users": ["id"]}), json.dumps({})])
    flt.refine(query="q", subgraph={"users": ["id"]}, db_id=None)
    fwd_prompt = flt.client.calls[0]["prompt"]
    _assert("If you are UNSURE about any of the four rules" in fwd_prompt,
            "M1-C 4-rule UNSURE → KEEP present")


def test_m4_forward_mode_invalid_raises():
    print("\n[test] invalid bidirectional_forward_prompt_mode → ValueError")
    try:
        BidirectionalFilter(
            model_name="mock", provider=None, db_dir="/x", num_examples=0,
            bidirectional_forward_prompt_mode="bogus",
        )
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_m4_forward_mode_none_keeps_legacy_forward_section():
    print("\n[test] mode=None falls back to legacy forward_section param (backward compat)")
    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
        forward_section="recall_biased_strong",   # legacy path
        # bidirectional_forward_prompt_mode omitted
    )
    _assert(flt.forward_section == "recall_biased_strong",
            "legacy forward_section retained")
    _assert(flt.bidirectional_forward_prompt_mode is None,
            "mode flag stays None")


def test_m4_forward_mode_takes_priority_over_forward_section():
    print("\n[test] mode arg overrides legacy forward_section (priority)")
    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
        forward_section="recall_biased_mild",
        bidirectional_forward_prompt_mode="recall_biased_strong",
    )
    _assert(flt.forward_section == "recall_biased_strong",
            "mode arg wins over forward_section legacy")


def test_m4_top_2_c1_spec_yaml_build():
    print("\n[test] Top 2 C1 yaml-style build (DECISIONS 2026-05-17 §4 spec exact)")
    from modules.registry import build
    inst = build("filter", {
        "name": "BidirectionalFilter",
        "params": {
            "model_name": "zai-org/glm-4.7",
            "provider": None,
            "db_dir": "/x",
            "num_examples": 0,
            "bidirectional_forward_prompt_mode": "recall_biased_strong",
            "backward_section": "bidirectional_backward",
            "sanitize_output": True,
        },
    })
    _assert(isinstance(inst, BidirectionalFilter), "instance type")
    _assert(inst.forward_section == "recall_biased_strong",
            "C1 forward resolved to strong")
    _assert(inst.backward_section == "bidirectional_backward",
            "Backward unchanged (DECISIONS §4 retain)")
    _assert(inst.bidirectional_forward_prompt_mode == "recall_biased_strong",
            "C1 mode flag recorded")


# ============================================================
# Wave 6 Phase 5 (Top 2 C2) — voting_multi_prompt Forward mode
# ============================================================
def test_m4_voting_mode_calls_3_forward_prompts_plus_1_backward():
    print("\n[test] voting_multi_prompt Forward → 3 LLM call + 1 Backward = 4 total")
    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
        bidirectional_forward_prompt_mode="voting_multi_prompt",
        bidirectional_forward_voting_strategy="MAJORITY",
    )
    # 3 forward (A/B/C) + 1 backward = 4 sequential responses
    resp_fwd_A = json.dumps({"users": ["id", "name"]})
    resp_fwd_B = json.dumps({"users": ["id", "name", "email"]})
    resp_fwd_C = json.dumps({"users": ["id"]})
    resp_bwd = json.dumps({"orders": ["total"]})
    flt.client = _SequentialMock([resp_fwd_A, resp_fwd_B, resp_fwd_C, resp_bwd])
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name", "email"], "orders": ["total"]},
        db_id=None,
    )
    _assert(len(flt.client.calls) == 4, f"4 LLM calls (3 voting + 1 backward), got {len(flt.client.calls)}")
    info = result["filter_info"]
    _assert(info["filter_forward_llm_calls"] == 3, "forward_llm_calls=3")
    _assert(info["filter_forward_voting_strategy"] == "MAJORITY", "MAJORITY recorded")
    # MAJORITY: id (3 votes), name (2 votes) keep; email (1 vote) drop
    # Forward voted = {users: [id, name]}, Backward = {orders: [total]} → union
    final = set(result["final_nodes"])
    _assert(
        {"users.id", "users.name", "orders.total"}.issubset(final),
        f"voting+union, got {sorted(final)}",
    )
    _assert("users.email" not in final, "email below MAJORITY threshold dropped")


def test_m4_voting_or_strategy_keeps_any_vote():
    print("\n[test] voting_multi_prompt + OR strategy keeps any column with ≥1 vote")
    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
        bidirectional_forward_prompt_mode="voting_multi_prompt",
        bidirectional_forward_voting_strategy="OR",
    )
    resp_A = json.dumps({"users": ["id"]})
    resp_B = json.dumps({"users": ["name"]})
    resp_C = json.dumps({"users": ["email"]})
    resp_bwd = json.dumps({})
    flt.client = _SequentialMock([resp_A, resp_B, resp_C, resp_bwd])
    result = flt.refine(
        query="q", subgraph={"users": ["id", "name", "email"]}, db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id", "users.name", "users.email"},
            f"OR union all 3, got {sorted(final)}")


def test_m4_voting_and_strategy_requires_unanimous():
    print("\n[test] voting_multi_prompt + AND strategy requires all 3 prompts to agree")
    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
        bidirectional_forward_prompt_mode="voting_multi_prompt",
        bidirectional_forward_voting_strategy="AND",
    )
    resp_A = json.dumps({"users": ["id", "name"]})
    resp_B = json.dumps({"users": ["id", "name"]})
    resp_C = json.dumps({"users": ["id"]})
    resp_bwd = json.dumps({})
    flt.client = _SequentialMock([resp_A, resp_B, resp_C, resp_bwd])
    result = flt.refine(
        query="q", subgraph={"users": ["id", "name"]}, db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id"}, f"AND only unanimous 'id', got {sorted(final)}")


def test_m4_voting_invalid_strategy_raises():
    print("\n[test] invalid voting strategy → ValueError")
    try:
        BidirectionalFilter(
            model_name="mock", provider=None, db_dir="/x", num_examples=0,
            bidirectional_forward_prompt_mode="voting_multi_prompt",
            bidirectional_forward_voting_strategy="bogus",
        )
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_m4_voting_diagnostics_in_filter_info():
    print("\n[test] voting mode records raw_counts + voted_counts + per-prompt halluc")
    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
        bidirectional_forward_prompt_mode="voting_multi_prompt",
        bidirectional_forward_voting_strategy="MAJORITY",
    )
    resp_A = json.dumps({"users": ["id"], "ghost": ["x"]})  # halluc table
    resp_B = json.dumps({"users": ["id", "name"]})
    resp_C = json.dumps({"users": ["id", "fake"]})           # halluc col
    resp_bwd = json.dumps({})
    flt.client = _SequentialMock([resp_A, resp_B, resp_C, resp_bwd])
    result = flt.refine(
        query="q", subgraph={"users": ["id", "name"]}, db_id=None,
    )
    info = result["filter_info"]
    _assert(info["filter_forward_raw_counts"] == {"A": 1, "B": 2, "C": 1},
            f"raw counts after sanitize per-prompt, got {info['filter_forward_raw_counts']}")
    _assert(
        info["filter_forward_voted_counts"]["MAJORITY"] == 1,
        f"MAJORITY voted=1 (id has 3 votes, name has 1), got "
        f"{info['filter_forward_voted_counts']}",
    )
    # forward halluc total = ghost.x + users.fake = 2
    _assert(
        info["filter_hallucination_removed_forward"] == 2,
        f"forward halluc summed across 3 prompts, got "
        f"{info['filter_hallucination_removed_forward']}",
    )


def test_m4_non_voting_mode_has_no_voting_diag():
    print("\n[test] non-voting mode (e.g., strong) → voting diag fields are None")
    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
        bidirectional_forward_prompt_mode="recall_biased_strong",
    )
    flt.client = _SequentialMock([json.dumps({"users": ["id"]}), json.dumps({})])
    result = flt.refine(query="q", subgraph={"users": ["id"]}, db_id=None)
    info = result["filter_info"]
    _assert(info["filter_forward_voting_strategy"] is None,
            "voting_strategy None in non-voting mode")
    _assert(info["filter_forward_llm_calls"] == 1, "1 LLM call (single forward)")
    _assert(info["filter_forward_raw_counts"] is None, "raw_counts None")


def test_m4_c2_spec_yaml_build():
    print("\n[test] Top 2 C2 yaml-style build (DECISIONS 2026-05-17 §6 spec exact)")
    from modules.registry import build
    inst = build("filter", {
        "name": "BidirectionalFilter",
        "params": {
            "model_name": "zai-org/glm-4.7",
            "provider": None,
            "db_dir": "/x",
            "num_examples": 0,
            "bidirectional_forward_prompt_mode": "voting_multi_prompt",
            "bidirectional_forward_voting_strategy": "MAJORITY",
            "backward_section": "bidirectional_backward",
            "sanitize_output": True,
        },
    })
    _assert(isinstance(inst, BidirectionalFilter), "instance type")
    _assert(inst.bidirectional_forward_prompt_mode == "voting_multi_prompt",
            "C2 mode persisted")
    _assert(inst.bidirectional_forward_voting_strategy == "MAJORITY",
            "C2 voting strategy = MAJORITY")
    _assert(inst._forward_is_voting is True, "voting flag True")
    _assert(inst.backward_section == "bidirectional_backward",
            "Backward retain (DECISIONS §6)")


# ============================================================
# M5 integration
# ============================================================
def _make_m5(responses_s1_s2: List[str]) -> TwoStageFilter:
    flt = TwoStageFilter(
        model_name="mock-model", provider=None, db_dir="/nonexistent",
        num_examples=0, sanitize_output=True,
    )
    flt.client = _SequentialMock(responses_s1_s2)
    return flt


def test_m5_refine_sequential_stages():
    print("\n[test] M5 refine 2 sequential LLM calls (stage1 → stage2)")
    s1 = json.dumps({"users": ["id", "name", "age"], "orders": ["total"]})
    s2 = json.dumps({"users": ["id", "name"], "orders": ["total"]})
    flt = _make_m5([s1, s2])
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name", "age", "email"], "orders": ["total"]},
        db_id=None,
    )
    _assert(len(flt.client.calls) == 2, "2 stage calls")
    final = set(result["final_nodes"])
    _assert(final == {"users.id", "users.name", "orders.total"}, f"got {sorted(final)}")
    info = result["filter_info"]
    _assert(info["filter_stage1_count"] == 4, "stage1=4")
    _assert(info["filter_stage2_count"] == 3, "stage2=3")
    _assert(info["filter_stage2_removed_count"] == 1, "stage2 removed 1 (age)")


def test_m5_stage2_sanitize_relative_to_stage1():
    print("\n[test] M5 stage2 sanitize uses stage1_clean (not subgraph)")
    s1 = json.dumps({"users": ["id"]})
    # stage 2 mentions a column NOT in stage1 — should be removed by stage2 sanitize
    s2 = json.dumps({"users": ["id", "name"]})  # name not in stage1
    flt = _make_m5([s1, s2])
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name", "age"]},
        db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id"}, "stage2 cannot introduce 'name' (not in stage1)")
    _assert(result["filter_info"]["filter_hallucination_removed_stage2"] >= 1,
            "stage2 halluc removed")


def test_m5_stage1_empty_skips_stage2():
    print("\n[test] M5 stage1 empty → stage2 skipped, only 1 LLM call")
    s1 = json.dumps({})  # nothing survives
    flt = _make_m5([s1])
    result = flt.refine(
        query="q", subgraph={"users": ["id"]}, db_id=None,
    )
    _assert(len(flt.client.calls) == 1, "only stage1 called")
    _assert(result["status"] == "Unanswerable", "no nodes after stage2 skip")


def test_m5_format_stage1_for_stage2_lists_kept_cols():
    print("\n[test] _format_stage1_for_stage2 lists tables + kept cols (no DB)")
    flt = TwoStageFilter(model_name="x", provider=None, db_dir="/nonexistent",
                         num_examples=0)
    text = flt._format_stage1_for_stage2(
        {"users": ["id", "name"], "orders": ["total"]}, db_id=None,
    )
    _assert("Table: users" in text and "Column: id" in text, "users.id present")
    _assert("Table: orders" in text and "Column: total" in text, "orders.total present")


def test_m5_empty_subgraph_short_circuit():
    print("\n[test] M5 empty subgraph → Unanswerable, no LLM call")
    flt = _make_m5(["", ""])
    result = flt.refine(query="q", subgraph={}, db_id=None)
    _assert(result["status"] == "Unanswerable", "Unanswerable")
    _assert(len(flt.client.calls) == 0, "no LLM call")


# ============================================================
# YAML-style build smoke
# ============================================================
def test_yaml_style_build_all_three():
    print("\n[test] registry.build instantiates M3/M4/M5 with yaml-style config")
    from modules.registry import build
    m3 = build("filter", {
        "name": "MultiPromptVotingFilter",
        "params": {
            "model_name": "mock", "provider": None, "db_dir": "/x",
            "num_examples": 0, "default_voting_strategy": "MAJORITY",
        },
    })
    m4 = build("filter", {
        "name": "BidirectionalFilter",
        "params": {"model_name": "mock", "provider": None, "db_dir": "/x", "num_examples": 0},
    })
    m5 = build("filter", {
        "name": "TwoStageFilter",
        "params": {"model_name": "mock", "provider": None, "db_dir": "/x", "num_examples": 0},
    })
    _assert(m3.default_voting_strategy == "MAJORITY", "M3 default persisted")
    _assert(m4.forward_section == "recall_biased_mild", "M4 forward default")
    _assert(m5.stage1_section == "two_stage_stage1", "M5 stage1 default")


def run_all():
    tests = [
        test_voting_or_keeps_any_inclusion,
        test_voting_majority_requires_two,
        test_voting_and_requires_all_three,
        test_voting_drops_empty_tables,
        test_voting_invalid_strategy_raises,
        test_m3_refine_runs_3_prompts_and_records_all_strategies,
        test_m3_sanitizes_hallucinations_per_prompt,
        test_m3_default_must_be_in_strategies,
        test_m3_empty_subgraph_short_circuit,
        test_m4_union_helper,
        test_m4_analyze_backward_contribution_with_gold,
        test_m4_analyze_backward_contribution_without_gold,
        test_m4_refine_2_calls_and_union,
        test_m4_with_gold_kwarg_records_precision,
        test_m4_sanitizes_both_sides,
        test_m4_empty_subgraph_short_circuit,
        # Wave 6 Phase 4 (Top 2 C1) — bidirectional_forward_prompt_mode
        test_m4_forward_mode_strong_overrides_default,
        test_m4_forward_mode_mild_explicit,
        test_m4_forward_mode_exclusion_rule,
        test_m4_forward_mode_invalid_raises,
        test_m4_forward_mode_none_keeps_legacy_forward_section,
        test_m4_forward_mode_takes_priority_over_forward_section,
        test_m4_top_2_c1_spec_yaml_build,
        # Wave 6 Phase 5 (Top 2 C2) — voting_multi_prompt Forward mode
        test_m4_voting_mode_calls_3_forward_prompts_plus_1_backward,
        test_m4_voting_or_strategy_keeps_any_vote,
        test_m4_voting_and_strategy_requires_unanimous,
        test_m4_voting_invalid_strategy_raises,
        test_m4_voting_diagnostics_in_filter_info,
        test_m4_non_voting_mode_has_no_voting_diag,
        test_m4_c2_spec_yaml_build,
        test_m5_refine_sequential_stages,
        test_m5_stage2_sanitize_relative_to_stage1,
        test_m5_stage1_empty_skips_stage2,
        test_m5_format_stage1_for_stage2_lists_kept_cols,
        test_m5_empty_subgraph_short_circuit,
        test_yaml_style_build_all_three,
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
