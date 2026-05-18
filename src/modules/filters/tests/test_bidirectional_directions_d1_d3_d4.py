"""Smoke tests for Wave 8 M4 Bidirectional 발전 D1 + D3 + D4.

학술 agent §1+§3+§4 + DECISIONS 2026-05-18 §2 정합.
M4 (BidirectionalFilter) 위의 wrapper 들. LLM 은 mock, D3 의 DB 실행은 임시
sqlite 파일 (smoke 용).
"""
import json
import os
import sqlite3
import sys
import tempfile
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("VLLM_API_KEY", "dummy")

from modules.filters.bidirectional_decompose_filter import (  # noqa: E402
    BidirectionalDecomposeFilter,
)
from modules.filters.bidirectional_verify_filter import (  # noqa: E402
    BidirectionalVerifyLoopFilter,
)
from modules.filters.bidirectional_value_hint_filter import (  # noqa: E402
    BidirectionalValueHintFilter,
)


class _SequentialMock:
    """generate_text(...) — 순서대로 응답 dequeue."""

    def __init__(self, responses: List[str]):
        self._q = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate_text(self, prompt: str, model: str, temperature: float, **kw) -> str:
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        if not self._q:
            return ""
        return self._q.pop(0)


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


# ============================================================
# D1 — Question Decomposition + Multi-Backward
# ============================================================
def _make_d1(responses: List[str], forward_per_sub_q: bool = False,
             max_sub_q: int = 5) -> BidirectionalDecomposeFilter:
    flt = BidirectionalDecomposeFilter(
        model_name="mock", provider=None, db_dir="/x",
        num_examples=0, sanitize_output=True,
        d1_max_sub_questions=max_sub_q,
        d1_forward_per_sub_q=forward_per_sub_q,
    )
    mock = _SequentialMock(responses)
    flt.client = mock
    # M4 의 client 도 동일 mock — 모든 LLM call 이 같은 queue 에서 dequeue
    flt.m4.client = mock
    return flt


def test_d1_parse_sub_questions():
    print("\n[test] _parse_json_array clips to cap + filters empty")
    out = BidirectionalDecomposeFilter._parse_json_array(
        '["q1", "q2", "", "q3", "q4", "q5", "q6"]', cap=5,
    )
    _assert(out == ["q1", "q2", "q3", "q4", "q5"], f"capped, got {out}")


def test_d1_parse_sub_questions_malformed():
    print("\n[test] _parse_json_array fallback to [] on garbage")
    out = BidirectionalDecomposeFilter._parse_json_array("not json", cap=5)
    _assert(out == [], "garbage → empty")


def test_d1_decompose_fail_uses_m4_baseline():
    print("\n[test] D1 decompose returning empty list → M4 baseline fallback")
    # M4: forward 1 + backward 1, then D1 decompose 1 = 3 calls
    m4_fwd = json.dumps({"users": ["id"]})
    m4_bwd = json.dumps({"users": ["name"]})
    decomp = "not a valid json array"     # fallback path
    flt = _make_d1([m4_fwd, m4_bwd, decomp])
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name", "age"]},
        db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id", "users.name"}, f"M4 baseline kept, got {sorted(final)}")
    _assert(result["filter_info"]["filter_decompose_failed"] is True, "decompose fail flag")
    _assert(result["stats"]["num_sub_questions"] == 0, "0 sub-q")


def test_d1_multi_backward_unions_to_m4():
    print("\n[test] D1 v1: sub-q backward 들이 M4 baseline 에 union")
    m4_fwd = json.dumps({"users": ["id"]})
    m4_bwd = json.dumps({"users": ["name"]})
    decomp = json.dumps(["who is the user?", "what is the age?"])
    bsub1 = json.dumps({"users": ["email"]})     # M4 미포함 → added
    bsub2 = json.dumps({"users": ["age"]})       # M4 미포함 → added
    flt = _make_d1([m4_fwd, m4_bwd, decomp, bsub1, bsub2])
    result = flt.refine(
        query="user info",
        subgraph={"users": ["id", "name", "email", "age"]},
        db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id", "users.name", "users.email", "users.age"},
            f"union of M4 + sub-q bwds, got {sorted(final)}")
    info = result["filter_info"]
    _assert(info["filter_num_sub_questions"] == 2, "2 sub-q parsed")
    _assert(info["filter_added_by_multi_backward"] == 2, "email + age added")
    # LLM calls: M4 (2) + decompose (1) + 2 sub-q backward = 5 total. D1 own = 3.
    _assert(info["filter_d1_llm_calls"] == 3, f"d1=3 (decompose + 2 sub-q bwd), got {info['filter_d1_llm_calls']}")


def test_d1_forward_per_sub_q_v2_runs_more_llm_calls():
    print("\n[test] D1 v2 (forward_per_sub_q=True): forward 도 sub-q별")
    m4_fwd = json.dumps({"users": ["id"]})
    m4_bwd = json.dumps({"users": ["name"]})
    decomp = json.dumps(["q1", "q2"])
    bsub1 = json.dumps({"users": ["email"]})
    fsub1 = json.dumps({"users": ["age"]})
    bsub2 = json.dumps({"orders": ["total"]})
    fsub2 = json.dumps({"orders": ["status"]})
    flt = _make_d1(
        [m4_fwd, m4_bwd, decomp, bsub1, fsub1, bsub2, fsub2],
        forward_per_sub_q=True,
    )
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name", "email", "age"], "orders": ["total", "status"]},
        db_id=None,
    )
    final = set(result["final_nodes"])
    _assert(
        final == {"users.id", "users.name", "users.email", "users.age",
                  "orders.total", "orders.status"},
        f"v2 includes per-sub-q forward, got {sorted(final)}",
    )
    info = result["filter_info"]
    # M4(2) + decompose(1) + 2 sub-q × (backward + forward) = 7 LLM. D1 own = 5
    _assert(info["filter_d1_llm_calls"] == 5, f"d1=5, got {info['filter_d1_llm_calls']}")


def test_d1_caps_sub_questions():
    print("\n[test] d1_max_sub_questions caps LLM count")
    m4_fwd = json.dumps({"users": ["id"]})
    m4_bwd = json.dumps({"users": ["name"]})
    decomp = json.dumps(["q1", "q2", "q3", "q4", "q5", "q6", "q7"])
    # cap = 3 → 3 sub-q backward
    bsubs = [json.dumps({"users": ["age"]}) for _ in range(3)]
    flt = _make_d1([m4_fwd, m4_bwd, decomp] + bsubs, max_sub_q=3)
    result = flt.refine(query="q", subgraph={"users": ["id", "name", "age"]}, db_id=None)
    info = result["filter_info"]
    _assert(info["filter_num_sub_questions"] == 3, "capped to 3 sub-q")
    _assert(info["filter_d1_llm_calls"] == 4, "1 decompose + 3 bsub")


def test_d1_invalid_max_sub_questions_raises():
    print("\n[test] d1_max_sub_questions < 1 → ValueError")
    try:
        BidirectionalDecomposeFilter(
            model_name="m", provider=None, d1_max_sub_questions=0,
        )
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_d1_empty_subgraph_short_circuit():
    print("\n[test] D1 empty subgraph → Unanswerable, no LLM call")
    flt = _make_d1([])
    result = flt.refine(query="q", subgraph={}, db_id=None)
    _assert(result["status"] == "Unanswerable", "Unanswerable")
    _assert(len(flt.client.calls) == 0, "no LLM")


# ============================================================
# D3 — Self-Verification Loop
# ============================================================
def _make_d3(responses: List[str], max_rounds: int = 2) -> BidirectionalVerifyLoopFilter:
    flt = BidirectionalVerifyLoopFilter(
        model_name="mock", provider=None, db_dir="/x",
        num_examples=0, sanitize_output=True, d3_max_rounds=max_rounds,
    )
    mock = _SequentialMock(responses)
    flt.client = mock
    flt.m4.client = mock
    return flt


def test_d3_parse_missing_column():
    print("\n[test] D3 parse_missing_from_error extracts 'no such column' hints")
    hints = BidirectionalVerifyLoopFilter.parse_missing_from_error(
        "no such column: users.email; no such table: orders"
    )
    _assert("users.email" in hints, "users.email hint")
    _assert("orders" in hints, "orders table hint")


def test_d3_parse_missing_empty():
    print("\n[test] D3 parse on empty msg → []")
    _assert(BidirectionalVerifyLoopFilter.parse_missing_from_error("") == [], "empty")
    _assert(BidirectionalVerifyLoopFilter.parse_missing_from_error(None) == [], "None")


def test_d3_recover_only_from_extractor():
    print("\n[test] D3 recover ignores hints not in extractor_output (no hallucination)")
    recovered = BidirectionalVerifyLoopFilter.recover_from_extractor(
        hints=["users.email", "users.fake_col", "ghost.x"],
        extractor_output={"users": ["id", "email"]},
        current_schema={"users": ["id"]},
    )
    _assert(recovered == {"users": ["email"]}, f"only existing 'email', got {recovered}")


def test_d3_recover_skips_already_in_current_schema():
    print("\n[test] D3 recover skips columns already in current_schema")
    recovered = BidirectionalVerifyLoopFilter.recover_from_extractor(
        hints=["users.id"],
        extractor_output={"users": ["id", "name"]},
        current_schema={"users": ["id"]},
    )
    _assert(recovered == {}, "id already present → skip")


def test_d3_invalid_max_rounds_raises():
    print("\n[test] D3 invalid max_rounds → ValueError")
    try:
        BidirectionalVerifyLoopFilter(
            model_name="m", provider=None, d3_max_rounds=10,
        )
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_d3_db_execute_success_breaks_loop():
    """smoke test with real sqlite — Sketch SQL 가 valid 면 loop break."""
    print("\n[test] D3 valid SQL on real sqlite → success, loop breaks at round 1")
    tmp_db = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp_db.close()
    try:
        conn = sqlite3.connect(tmp_db.name)
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'alice')")
        conn.commit()
        conn.close()

        # db_id = basename without .sqlite, expected DB path = db_dir/db_id/db_id.sqlite
        # 임시 db 를 그 구조에 맞추기
        db_id = "smoke_d3"
        db_dir = os.path.join(os.path.dirname(tmp_db.name), "d3_root")
        os.makedirs(os.path.join(db_dir, db_id), exist_ok=True)
        target_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        os.rename(tmp_db.name, target_path)

        flt = BidirectionalVerifyLoopFilter(
            model_name="mock", provider=None, db_dir=db_dir,
            num_examples=0, sanitize_output=True, d3_max_rounds=2,
        )
        # M4: fwd + bwd, then D3 sketch (round 1, valid SQL)
        m4_fwd = json.dumps({"users": ["id"]})
        m4_bwd = json.dumps({"users": ["name"]})
        sketch_ok = "SELECT id, name FROM users"
        mock = _SequentialMock([m4_fwd, m4_bwd, sketch_ok])
        flt.client = mock
        flt.m4.client = mock
        result = flt.refine(
            query="show users",
            subgraph={"users": ["id", "name"]},
            db_id=db_id,
        )
        _assert(result["stats"]["avg_rounds_used"] == 1,
                f"1 round (success → break), got {result['stats']['avg_rounds_used']}")
        _assert(result["stats"]["verify_success_rate"] == 1.0,
                "100% success")
        _assert(result["stats"]["d3_llm_calls"] == 1, "1 sketch LLM call")
    finally:
        # cleanup
        try:
            target_path  # noqa: F841
            os.remove(os.path.join(db_dir, db_id, f"{db_id}.sqlite"))
            os.rmdir(os.path.join(db_dir, db_id))
            os.rmdir(db_dir)
        except Exception:
            pass


def test_d3_db_execute_error_recovers_columns():
    print("\n[test] D3 sketch SQL fails with 'no such column' → recover from extractor")
    tmp_db = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp_db.close()
    try:
        conn = sqlite3.connect(tmp_db.name)
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT, email TEXT)")
        conn.commit()
        conn.close()
        db_id = "smoke_d3_err"
        db_dir = os.path.join(os.path.dirname(tmp_db.name), "d3_err_root")
        os.makedirs(os.path.join(db_dir, db_id), exist_ok=True)
        target_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        os.rename(tmp_db.name, target_path)

        flt = BidirectionalVerifyLoopFilter(
            model_name="mock", provider=None, db_dir=db_dir,
            num_examples=0, sanitize_output=True, d3_max_rounds=2,
        )
        # M4 baseline: {users: [id]}; subgraph has email also
        m4_fwd = json.dumps({"users": ["id"]})
        m4_bwd = json.dumps({})
        # round 1: SQL references nonexistent 'phone' column → error
        sketch1 = "SELECT users.phone FROM users"
        # round 2: SQL valid using newly recovered (none — phone not in extractor either)
        sketch2 = "SELECT id, email FROM users"
        mock = _SequentialMock([m4_fwd, m4_bwd, sketch1, sketch2])
        flt.client = mock
        flt.m4.client = mock
        result = flt.refine(
            query="show users",
            subgraph={"users": ["id", "email", "name"]},
            db_id=db_id,
        )
        # 'phone' 은 extractor 에 없음 → recovered 없음 → 조기 종료 (round 2 진행 안 함)
        _assert(result["stats"]["avg_rounds_used"] == 1, "1 round (hint not in extractor → break)")
        _assert(result["stats"]["recovered_count"] == 0, "no recovery (phone hallucination)")
    finally:
        try:
            os.remove(target_path)
            os.rmdir(os.path.join(db_dir, db_id))
            os.rmdir(db_dir)
        except Exception:
            pass


def test_d3_empty_subgraph_short_circuit():
    print("\n[test] D3 empty subgraph → Unanswerable, no LLM call")
    flt = _make_d3([])
    result = flt.refine(query="q", subgraph={}, db_id=None)
    _assert(result["status"] == "Unanswerable", "Unanswerable")
    _assert(len(flt.client.calls) == 0, "no LLM")


# ============================================================
# D4 — Value Hint Forward 강화
# ============================================================
def test_d4_match_values_exact_high_confidence():
    print("\n[test] D4 match: exact match → confidence='high'")
    evidence = BidirectionalValueHintFilter.match_values_to_columns(
        value_mentions=["seoul", "2020"],
        col_samples={"users.city": ["Seoul", "Busan"], "users.year": ["2019", "2020"]},
    )
    _assert(evidence["users.city"]["confidence"] == "high", "city exact")
    _assert(evidence["users.year"]["confidence"] == "high", "year exact")


def test_d4_match_partial_medium_confidence():
    print("\n[test] D4 match: partial → confidence='medium'")
    evidence = BidirectionalValueHintFilter.match_values_to_columns(
        value_mentions=["South Korea"],
        col_samples={"users.country": ["Korea", "Japan"]},
    )
    _assert(evidence["users.country"]["confidence"] == "medium",
            f"partial → medium, got {evidence['users.country']['confidence']}")


def test_d4_match_no_match_empty():
    print("\n[test] D4 match: no match → empty evidence")
    evidence = BidirectionalValueHintFilter.match_values_to_columns(
        value_mentions=["xyz"], col_samples={"users.city": ["Seoul"]},
    )
    _assert(evidence == {}, "no match")


def test_d4_format_evidence_text():
    print("\n[test] D4 format_value_evidence groups by confidence")
    text = BidirectionalValueHintFilter.format_value_evidence({
        "users.year": {"matched_values": ["2020"], "confidence": "high"},
        "users.country": {"matched_values": ["Korea"], "confidence": "medium"},
    })
    _assert("HIGH" in text and "MEDIUM" in text, "both labels present")
    _assert(text.index("HIGH") < text.index("MEDIUM"), "high first")


def _make_d4(responses: List[str], forced_include: bool = False,
             col_samples: Dict[str, List[str]] = None,
            ) -> BidirectionalValueHintFilter:
    flt = BidirectionalValueHintFilter(
        model_name="mock", provider=None, db_dir="/x",
        num_examples=0, sanitize_output=True,
        d4_forced_include=forced_include,
    )
    mock = _SequentialMock(responses)
    flt.client = mock
    flt.m4.client = mock
    # Override DB fetch (avoid real sqlite)
    flt._fetch_column_examples = lambda subgraph, db_id: dict(col_samples or {})
    return flt


def test_d4_v1_value_hint_enhanced_forward():
    print("\n[test] D4 v1: value extract + enhanced forward + M4 backward union")
    # Sequence: D4 value_extract + D4 enhanced forward + M4 backward (no M4 full call in v1)
    val_extract = json.dumps(["seoul"])
    fwd = json.dumps({"users": ["city", "id"]})
    bwd = json.dumps({"users": ["name"]})
    flt = _make_d4(
        [val_extract, fwd, bwd],
        forced_include=False,
        col_samples={"users.city": ["Seoul"], "users.name": ["alice"]},
    )
    result = flt.refine(
        query="who lives in seoul",
        subgraph={"users": ["id", "name", "city"]},
        db_id="anydb",  # 임의 (실제 fetch 는 mock 으로 override)
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id", "users.name", "users.city"},
            f"fwd+bwd union, got {sorted(final)}")
    info = result["filter_info"]
    _assert(info["filter_evidence_high_count"] == 1, "seoul matched city as high")
    _assert(info["filter_d4_llm_calls"] == 2, "1 value extract + 1 enhanced forward")


def test_d4_v3_forced_include_high_confidence():
    print("\n[test] D4 v3 (forced_include=True): high-conf column 강제 retain")
    val_extract = json.dumps(["seoul"])
    # M4 baseline: forward + backward (m4 호출됨)
    m4_fwd = json.dumps({"users": ["id"]})
    m4_bwd = json.dumps({"users": ["name"]})
    flt = _make_d4(
        [val_extract, m4_fwd, m4_bwd],
        forced_include=True,
        col_samples={"users.city": ["Seoul"], "users.name": ["alice"]},
    )
    result = flt.refine(
        query="who lives in seoul",
        subgraph={"users": ["id", "name", "city"]},
        db_id="anydb",
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id", "users.name", "users.city"},
            f"M4 + forced city, got {sorted(final)}")
    info = result["filter_info"]
    _assert(info["filter_forced_count"] == 1, "city forced from high evidence")
    _assert(info["filter_d4_llm_calls"] == 1, "1 value extract (no extra forward in v3)")


def test_d4_v3_forced_skips_when_not_in_subgraph():
    print("\n[test] D4 v3: high-conf col 이 subgraph 에 없으면 forced 안 함 (no halluc)")
    val_extract = json.dumps(["mars"])
    m4_fwd = json.dumps({"users": ["id"]})
    m4_bwd = json.dumps({})
    flt = _make_d4(
        [val_extract, m4_fwd, m4_bwd],
        forced_include=True,
        # planet.name 은 subgraph 에 없음
        col_samples={"planet.name": ["Mars", "Venus"]},
    )
    result = flt.refine(
        query="planets",
        subgraph={"users": ["id"]},  # planet 없음
        db_id="anydb",
    )
    final = set(result["final_nodes"])
    _assert(final == {"users.id"}, "planet not in subgraph → not forced")
    _assert(result["filter_info"]["filter_forced_count"] == 0, "no force")


def test_d4_gold_kwarg_records_evidence_gold_precision():
    print("\n[test] D4 gold kwarg → evidence_gold_precision recorded")
    val_extract = json.dumps(["seoul"])
    m4_fwd = json.dumps({"users": ["id"]})
    m4_bwd = json.dumps({})
    flt = _make_d4(
        [val_extract, m4_fwd, m4_bwd],
        forced_include=True,
        col_samples={"users.city": ["Seoul"]},
    )
    result = flt.refine(
        query="q", subgraph={"users": ["id", "city"]}, db_id="any",
        gold={"users": ["city"]},
    )
    _assert(result["filter_info"]["filter_evidence_gold_precision"] == 1.0,
            "city evidence is gold → precision 1.0")


def test_d4_empty_subgraph_short_circuit():
    print("\n[test] D4 empty subgraph → Unanswerable, no LLM call")
    flt = _make_d4([])
    result = flt.refine(query="q", subgraph={}, db_id=None)
    _assert(result["status"] == "Unanswerable", "Unanswerable")
    _assert(len(flt.client.calls) == 0, "no LLM")


# ============================================================
# YAML-style registry build
# ============================================================
def test_yaml_build_all_three_d_filters():
    print("\n[test] registry.build instantiates D1/D3/D4 filters from yaml-style")
    from modules.registry import build
    d1 = build("filter", {
        "name": "BidirectionalDecomposeFilter",
        "params": {"model_name": "mock", "provider": None, "db_dir": "/x",
                   "num_examples": 0, "d1_max_sub_questions": 3,
                   "d1_forward_per_sub_q": True},
    })
    d3 = build("filter", {
        "name": "BidirectionalVerifyLoopFilter",
        "params": {"model_name": "mock", "provider": None, "db_dir": "/x",
                   "num_examples": 0, "d3_max_rounds": 2, "d3_db_timeout_s": 5.0},
    })
    d4 = build("filter", {
        "name": "BidirectionalValueHintFilter",
        "params": {"model_name": "mock", "provider": None, "db_dir": "/x",
                   "num_examples": 0, "d4_forced_include": True},
    })
    _assert(d1.d1_max_sub_questions == 3, "D1 cap persisted")
    _assert(d1.d1_forward_per_sub_q is True, "D1 v2 persisted")
    _assert(d3.d3_max_rounds == 2, "D3 max rounds")
    _assert(d4.d4_forced_include is True, "D4 v3 forced")


def run_all():
    tests = [
        # D1
        test_d1_parse_sub_questions,
        test_d1_parse_sub_questions_malformed,
        test_d1_decompose_fail_uses_m4_baseline,
        test_d1_multi_backward_unions_to_m4,
        test_d1_forward_per_sub_q_v2_runs_more_llm_calls,
        test_d1_caps_sub_questions,
        test_d1_invalid_max_sub_questions_raises,
        test_d1_empty_subgraph_short_circuit,
        # D3
        test_d3_parse_missing_column,
        test_d3_parse_missing_empty,
        test_d3_recover_only_from_extractor,
        test_d3_recover_skips_already_in_current_schema,
        test_d3_invalid_max_rounds_raises,
        test_d3_db_execute_success_breaks_loop,
        test_d3_db_execute_error_recovers_columns,
        test_d3_empty_subgraph_short_circuit,
        # D4
        test_d4_match_values_exact_high_confidence,
        test_d4_match_partial_medium_confidence,
        test_d4_match_no_match_empty,
        test_d4_format_evidence_text,
        test_d4_v1_value_hint_enhanced_forward,
        test_d4_v3_forced_include_high_confidence,
        test_d4_v3_forced_skips_when_not_in_subgraph,
        test_d4_gold_kwarg_records_evidence_gold_precision,
        test_d4_empty_subgraph_short_circuit,
        # Common
        test_yaml_build_all_three_d_filters,
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
