"""Standalone smoke tests for ScoreGatedBatchExtractiveFilter (no pytest dep).

Run from project root:
    conda run -n base python src/modules/filters/tests/test_sgbe.py

LLM client 는 _MockLLMClient 로 교체하여 네트워크 의존 없이 실행한다.
"""
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# 환경 변수: vllm/openai/glm 모두 사용하지 않음 (LLM 호출 mock).
# APIClient init 시 base_url None 이라도 OK (실제 호출 직전 mock 교체).
os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("VLLM_API_KEY", "dummy")

from modules.filters.score_gated_batch_extractive_filter import (  # noqa: E402
    ScoreGatedBatchExtractiveFilter,
)


class _MockLLMClient:
    """generate_text 호출을 captures + 미리 지정된 응답 반환."""

    def __init__(self, response: str = "[]"):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def generate_text(self, prompt: str, model: str, temperature: float, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        return self.response


def _make_filter(
    mock_response: str = "[]",
    theta_keep: float = 0.65,
    theta_drop: float = 0.40,
    fk_pk_hardcode: bool = True,
    step_mode: str = "step_0+1+2",
    # 기존 boundary 시나리오 (작은 score 표본) 가 collapse 감지에 잡히지 않도록
    # helper default 는 None (비활성화). collapse 시나리오는 explicit 0.05.
    score_collapse_threshold=None,
) -> ScoreGatedBatchExtractiveFilter:
    """SGBE 인스턴스를 생성한 뒤 client 를 mock 으로 교체."""
    flt = ScoreGatedBatchExtractiveFilter(
        model_name="mock-model",
        theta_keep=theta_keep,
        theta_drop=theta_drop,
        temperature=0.0,
        num_examples=0,  # DB 미접근 — value retrieval skip
        fk_pk_hardcode=fk_pk_hardcode,
        step_mode=step_mode,
        score_collapse_threshold=score_collapse_threshold,
        provider=None,
        db_dir="/nonexistent",  # PK PRAGMA 도 skip
    )
    flt.client = _MockLLMClient(mock_response)
    return flt


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


# ============================================================
# Scenario 1: 빈 subgraph
# ============================================================
def test_empty_subgraph_short_circuits():
    print("\n[test] empty subgraph short-circuits to Unanswerable")
    flt = _make_filter()
    result = flt.refine(query="q", subgraph={}, db_id=None, gat_scores={})
    _assert(result["status"] == "Unanswerable", "status=Unanswerable")
    _assert(result["final_nodes"] == [], "final_nodes empty")
    _assert(len(flt.client.calls) == 0, "no LLM call")
    _assert(result["stats"]["uncertain"] == 0, "stats keep_hard=0")


# ============================================================
# Scenario 2: gat_scores=None → 전부 S_uncertain → LLM 호출 후 응답대로 keep
# ============================================================
def test_no_scores_routes_all_to_llm():
    print("\n[test] gat_scores=None → fallback all-uncertain → LLM-driven")
    mock = json.dumps([
        {"column": "users.id", "keep": True, "reason": "subject"},
        {"column": "users.name", "keep": False, "reason": "not needed"},
        {"column": "orders.total", "keep": True, "reason": "aggregate"},
    ])
    flt = _make_filter(mock_response=mock)
    result = flt.refine(
        query="total per user",
        subgraph={"users": ["id", "name"], "orders": ["total"]},
        db_id=None,
        gat_scores=None,
    )
    _assert(len(flt.client.calls) == 1, "exactly one LLM call")
    _assert(result["stats"]["uncertain"] == 3, "all 3 columns routed to uncertain")
    _assert(result["stats"]["keep_hard"] == 0, "no keep_hard without scores")
    _assert(result["stats"]["drop_hard"] == 0, "no drop_hard without scores")
    _assert(result["stats"]["lm_keep"] == 2, "LLM picked 2 keepers")
    _assert(set(result["final_nodes"]) == {"users.id", "orders.total"}, "final = LM keepers")
    _assert(result["status"] == "Answerable", "status=Answerable")


# ============================================================
# Scenario 3: partial scores — 일부 column 만 score 보유
# ============================================================
def test_partial_scores_route_missing_to_uncertain():
    print("\n[test] partial gat_scores: missing scores → uncertain bucket")
    mock = json.dumps([{"column": "users.name", "keep": True, "reason": "ok"}])
    flt = _make_filter(mock_response=mock)
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"], "orders": ["total"]},
        db_id=None,
        gat_scores={"users.id": 0.90, "orders.total": 0.20},  # users.name 누락
    )
    _assert(result["stats"]["keep_hard"] == 1, "users.id keep_hard (0.90≥0.65)")
    _assert(result["stats"]["drop_hard"] == 1, "orders.total drop_hard (0.20<0.40)")
    _assert(result["stats"]["uncertain"] == 1, "users.name routed to uncertain (no score)")
    _assert(result["stats"]["lm_keep"] == 1, "LLM kept users.name")
    _assert(
        {"users.id", "users.name"}.issubset(set(result["final_nodes"])),
        "final = keep_hard ∪ lm_keep",
    )
    _assert("orders.total" not in result["final_nodes"], "drop_hard excluded")


# ============================================================
# Scenario 4: 모든 score ≥ θ_keep → S_uncertain 빈 set, LLM call 0
# ============================================================
def test_all_keep_hard_skips_llm():
    print("\n[test] all scores ≥ θ_keep → no LLM call")
    flt = _make_filter()
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"]},
        db_id=None,
        gat_scores={"users.id": 0.95, "users.name": 0.88},
    )
    _assert(len(flt.client.calls) == 0, "LLM not called")
    _assert(result["stats"]["keep_hard"] == 2, "both columns keep_hard")
    _assert(result["stats"]["uncertain"] == 0, "no uncertain")
    _assert(set(result["final_nodes"]) == {"users.id", "users.name"}, "all preserved")


# ============================================================
# Scenario 5: 모든 score < θ_drop → 전부 drop, FK/PK 없으면 final empty
# ============================================================
def test_all_drop_hard_skips_llm_and_returns_empty():
    print("\n[test] all scores < θ_drop, no FK/PK → final empty + LLM skipped")
    flt = _make_filter(fk_pk_hardcode=False)  # struct 보호 끄기
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"]},
        db_id=None,
        gat_scores={"users.id": 0.10, "users.name": 0.20},
    )
    _assert(len(flt.client.calls) == 0, "LLM not called")
    _assert(result["stats"]["drop_hard"] == 2, "both columns drop_hard")
    _assert(result["stats"]["uncertain"] == 0, "no uncertain")
    _assert(result["final_nodes"] == [], "no surviving nodes")
    _assert(result["status"] == "Unanswerable", "status=Unanswerable")


# ============================================================
# Scenario 6: 정상 3-group 분포 — keep_hard + drop_hard + uncertain 모두 존재
# ============================================================
def test_balanced_three_group_distribution():
    print("\n[test] balanced 3-group distribution")
    mock = json.dumps([
        {"column": "orders.total", "keep": True, "reason": "needed"},
        {"column": "orders.note", "keep": False, "reason": "irrelevant"},
    ])
    flt = _make_filter(mock_response=mock)
    result = flt.refine(
        query="user total spending",
        subgraph={
            "users": ["id", "name", "age"],
            "orders": ["total", "note"],
        },
        db_id=None,
        gat_scores={
            "users.id": 0.92,    # keep_hard
            "users.name": 0.70,  # keep_hard
            "users.age": 0.15,   # drop_hard
            "orders.total": 0.55,  # uncertain
            "orders.note": 0.50,   # uncertain
        },
    )
    _assert(result["stats"]["keep_hard"] == 2, "2 keep_hard")
    _assert(result["stats"]["drop_hard"] == 1, "1 drop_hard")
    _assert(result["stats"]["uncertain"] == 2, "2 uncertain → LLM")
    _assert(result["stats"]["lm_keep"] == 1, "LLM kept 1 of uncertain")
    final = set(result["final_nodes"])
    _assert({"users.id", "users.name", "orders.total"}.issubset(final), "keep_hard + lm_keep present")
    _assert("users.age" not in final, "drop_hard excluded")
    _assert("orders.note" not in final, "LLM-rejected uncertain excluded")
    _assert(len(flt.client.calls) == 1, "exactly one LLM call")


# ============================================================
# Scenario 7: FK hardcode — drop_hard 라도 FK column 이면 final 에 보존
# ============================================================
def test_fk_hardcode_preserves_drop_hard_fk_column():
    print("\n[test] FK column survives even when score < θ_drop")
    flt = _make_filter(fk_pk_hardcode=True)
    metadata = {"fk_to_id": {"users.id->orders.user_id": 0}}
    result = flt.refine(
        query="q",
        subgraph={"users": ["id"], "orders": ["user_id"]},
        db_id=None,
        gat_scores={"users.id": 0.05, "orders.user_id": 0.05},  # 둘 다 drop_hard
        metadata=metadata,
    )
    _assert(len(flt.client.calls) == 0, "LLM not called")
    _assert(result["stats"]["drop_hard"] == 2, "both initially drop_hard")
    _assert(result["stats"]["struct"] == 2, "both rescued by FK hardcode")
    _assert(
        set(result["final_nodes"]) == {"users.id", "orders.user_id"},
        "FK columns preserved",
    )


# ============================================================
# Scenario 8: parse fail → recall-safe fallback (S_uncertain 전부 keep)
# ============================================================
def test_unparseable_response_recall_safe_fallback():
    print("\n[test] unparseable LLM response → keep all S_uncertain (recall-safe)")
    flt = _make_filter(mock_response="totally garbage no json here")
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"]},
        db_id=None,
        gat_scores={"users.id": 0.50, "users.name": 0.55},  # 둘 다 uncertain
    )
    _assert(result["stats"]["uncertain"] == 2, "both uncertain")
    _assert(result["stats"]["lm_keep"] == 2, "fallback kept all")
    _assert(
        set(result["final_nodes"]) == {"users.id", "users.name"},
        "recall-safe fallback preserves all uncertain",
    )
    _assert(
        result["filter_info"].get("filter_kept_via_fallback") is True,
        "filter_info flags fallback",
    )


# ============================================================
# Scenario 9: theta_drop > theta_keep 잘못된 설정 시 ValueError
# ============================================================
def test_invalid_thresholds_raise():
    print("\n[test] theta_drop > theta_keep raises ValueError")
    try:
        ScoreGatedBatchExtractiveFilter(
            theta_keep=0.40, theta_drop=0.65, provider=None,
        )
        _assert(False, "expected ValueError but none raised")
    except ValueError:
        _assert(True, "ValueError raised as expected")


# ============================================================
# Scenario 10: line-pattern fallback parser
# ============================================================
def test_line_pattern_parser_fallback():
    print("\n[test] non-JSON list but per-line 'col: yes' parses via fallback")
    mock = (
        "Here are my decisions:\n"
        "users.id: yes (subject key)\n"
        "users.name: no (not referenced)\n"
        "orders.total: yes (aggregated)\n"
    )
    flt = _make_filter(mock_response=mock)
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"], "orders": ["total"]},
        db_id=None,
        gat_scores=None,  # 전부 uncertain
    )
    _assert(result["stats"]["lm_keep"] == 2, "2 columns kept via line fallback")
    final = set(result["final_nodes"])
    _assert("users.id" in final and "orders.total" in final, "yes-marked kept")
    _assert("users.name" not in final, "no-marked dropped")


# ============================================================
# Scenario 11: step_mode="step_0" — FK/PK 만 keep, 나머지 모두 drop, LLM 없음
# ============================================================
def test_step_mode_step_0():
    print("\n[test] step_mode='step_0': FK/PK only, no LLM, drop everything else")
    flt = _make_filter(step_mode="step_0")
    metadata = {"fk_to_id": {"users.id->orders.user_id": 0}}
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"], "orders": ["user_id", "total"]},
        db_id=None,
        gat_scores={"users.id": 0.90, "users.name": 0.85,
                    "orders.user_id": 0.80, "orders.total": 0.75},
        metadata=metadata,
    )
    _assert(result["stats"]["step_mode"] == "step_0", "stats records step_mode")
    _assert(len(flt.client.calls) == 0, "no LLM call in step_0")
    _assert(result["stats"]["lm_keep"] == 0, "lm_keep=0")
    _assert(result["stats"]["keep_hard"] == 0, "keep_hard=0 (step 1 skipped)")
    _assert(result["stats"]["drop_hard"] == 0, "drop_hard=0 (step 1 skipped)")
    _assert(result["stats"]["uncertain"] == 0, "uncertain=0 (step 1 skipped)")
    _assert(result["stats"]["struct"] == 2, "FK columns recovered")
    _assert(
        set(result["final_nodes"]) == {"users.id", "orders.user_id"},
        "final = FK columns only — high-score non-FK columns excluded",
    )


# ============================================================
# Scenario 12: step_mode="step_0+1" — score-gate keep + struct, no LLM
# ============================================================
def test_step_mode_step_0_plus_1():
    print("\n[test] step_mode='step_0+1': score-gate only, no LLM, drop uncertain")
    flt = _make_filter(step_mode="step_0+1")
    metadata = {"fk_to_id": {"users.id->orders.user_id": 0}}
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"], "orders": ["user_id", "total"]},
        db_id=None,
        gat_scores={
            "users.id": 0.05,         # drop_hard but rescued by FK hardcode (struct)
            "users.name": 0.85,       # keep_hard
            "orders.user_id": 0.50,   # uncertain — dropped (no LLM)
            "orders.total": 0.15,     # drop_hard
        },
        metadata=metadata,
    )
    _assert(result["stats"]["step_mode"] == "step_0+1", "stats records step_mode")
    _assert(len(flt.client.calls) == 0, "no LLM call in step_0+1")
    _assert(result["stats"]["lm_keep"] == 0, "lm_keep=0 (step 2 skipped)")
    _assert(result["stats"]["keep_hard"] == 1, "users.name keep_hard")
    _assert(result["stats"]["drop_hard"] == 2, "users.id + orders.total drop_hard")
    _assert(result["stats"]["uncertain"] == 1, "orders.user_id uncertain (dropped)")
    _assert(result["stats"]["struct"] == 2, "users.id + orders.user_id rescued as FK")
    final = set(result["final_nodes"])
    _assert(
        final == {"users.id", "users.name", "orders.user_id"},
        "final = keep_hard ∪ struct (uncertain dropped without LLM)",
    )


# ============================================================
# Scenario 13: step_mode="step_0+1+2" explicit — default 와 동일 거동 확인
# ============================================================
def test_step_mode_full_default_matches_explicit():
    print("\n[test] step_mode='step_0+1+2' explicit == default behavior")
    mock = json.dumps([{"column": "orders.note", "keep": True, "reason": "ok"}])
    flt = _make_filter(mock_response=mock, step_mode="step_0+1+2")
    result = flt.refine(
        query="q",
        subgraph={"users": ["id"], "orders": ["note"]},
        db_id=None,
        gat_scores={"users.id": 0.95, "orders.note": 0.50},
    )
    _assert(result["stats"]["step_mode"] == "step_0+1+2", "explicit mode logged")
    _assert(len(flt.client.calls) == 1, "LLM called once for uncertain bucket")
    _assert(result["stats"]["keep_hard"] == 1, "users.id keep_hard")
    _assert(result["stats"]["uncertain"] == 1, "orders.note uncertain")
    _assert(result["stats"]["lm_keep"] == 1, "LLM kept orders.note")
    _assert(
        set(result["final_nodes"]) == {"users.id", "orders.note"},
        "behaves identically to default mode",
    )


# ============================================================
# Scenario 14: score collapse detected → XiYan-equivalent fallback
# ============================================================
def test_score_collapse_triggers_xiyan_fallback():
    print("\n[test] score collapse (std < threshold) → all → S_uncertain → LLM")
    mock = json.dumps([
        {"column": "users.id", "keep": True, "reason": "ok"},
        {"column": "users.name", "keep": True, "reason": "ok"},
    ])
    flt = _make_filter(mock_response=mock, score_collapse_threshold=0.05)
    # 모든 score 가 거의 동일 (std ≈ 0.0008) — over-smoothing era 모사
    collapsed_scores = {"users.id": 0.501, "users.name": 0.500, "users.age": 0.502}
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name", "age"]},
        db_id=None,
        gat_scores=collapsed_scores,
    )
    _assert(
        result["stats"]["score_collapse_detected"] is True,
        "collapse flagged in stats",
    )
    _assert(
        result["filter_info"]["filter_score_collapse_detected"] is True,
        "collapse flagged in filter_info",
    )
    _assert(result["stats"]["keep_hard"] == 0, "no keep_hard during collapse")
    _assert(result["stats"]["drop_hard"] == 0, "no drop_hard during collapse")
    _assert(result["stats"]["uncertain"] == 3, "all routed to S_uncertain")
    _assert(len(flt.client.calls) == 1, "LLM called once (XiYan-equivalent)")
    _assert(result["stats"]["lm_keep"] == 2, "LLM kept 2")


# ============================================================
# Scenario 15: collapse threshold=None → 비활성화, 정상 score-gate 작동
# ============================================================
def test_score_collapse_threshold_disabled():
    print("\n[test] score_collapse_threshold=None disables detection")
    flt = _make_filter(score_collapse_threshold=None)
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"]},
        db_id=None,
        # 매우 작은 std 이지만 감지 비활성화 → 정상 gate
        gat_scores={"users.id": 0.700, "users.name": 0.701},
    )
    _assert(
        result["stats"]["score_collapse_detected"] is False,
        "collapse detection disabled",
    )
    _assert(result["stats"]["keep_hard"] == 2, "정상 score-gate: 둘 다 keep_hard")
    _assert(len(flt.client.calls) == 0, "no LLM call when all keep_hard")


# ============================================================
# Scenario 16: invalid step_mode → ValueError
# ============================================================
def test_invalid_step_mode_raises():
    print("\n[test] invalid step_mode raises ValueError")
    try:
        ScoreGatedBatchExtractiveFilter(step_mode="bogus", provider=None)
        _assert(False, "expected ValueError but none raised")
    except ValueError:
        _assert(True, "ValueError raised for invalid step_mode")


def run_all():
    tests = [
        test_empty_subgraph_short_circuits,
        test_no_scores_routes_all_to_llm,
        test_partial_scores_route_missing_to_uncertain,
        test_all_keep_hard_skips_llm,
        test_all_drop_hard_skips_llm_and_returns_empty,
        test_balanced_three_group_distribution,
        test_fk_hardcode_preserves_drop_hard_fk_column,
        test_unparseable_response_recall_safe_fallback,
        test_invalid_thresholds_raise,
        test_line_pattern_parser_fallback,
        test_step_mode_step_0,
        test_step_mode_step_0_plus_1,
        test_step_mode_full_default_matches_explicit,
        test_score_collapse_triggers_xiyan_fallback,
        test_score_collapse_threshold_disabled,
        test_invalid_step_mode_raises,
    ]
    failures: List = []
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
