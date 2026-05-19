"""Smoke tests for Wave 11 Schema Serialization Direction C (2026-05-19).

5 cells (DECISIONS 2026-05-19 §3 Phase A):
  - C-v1   : source_tagged
  - C-v2   : question_enrichment
  - C-v3a  : flat_merged_fk
  - C-v3b  : flat_merged_no_fk
  - Comb-C : tagged_enriched

각 serializer 가 sample query 위 정상 동작 + Schema Content Invariance 검증
(Filter 가 선택한 column 집합이 직렬화 후에도 변하지 않음).
"""
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("VLLM_API_KEY", "dummy")

from serializers.source_tagged_serializer import (  # noqa: E402
    tag_columns,
    format_tagged_schema,
    _normalize_set,
)
from serializers.flat_merged_serializer import format_flat_schema  # noqa: E402
from serializers.question_enricher import (  # noqa: E402
    enrich_question,
    EnrichmentCache,
    format_schema_for_enrichment,
    ENRICHMENT_SYSTEM_PROMPT,
)


class _MockLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def generate_text(self, prompt: str, model: str, temperature: float, **kw) -> str:
        self.calls += 1
        return self.response


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


def _columns_from_schema_str(text: str, m4_output: Dict[str, List[str]]) -> set:
    present = set()
    for t, cols in m4_output.items():
        for c in cols:
            if f"{t}.{c}" in text or (f"Table: {t}" in text and f"- {c}" in text):
                present.add(f"{t}.{c}")
    return present


# ============================================================
# C-v1 — source_tagged_serializer
# ============================================================
def test_c_v1_tag_columns_categorizes_F_B_FB():
    print("\n[test] C-v1 tag_columns: [F+B] / [F] / [B]")
    fwd = {("users", "id"), ("users", "name")}
    bwd = {("users", "id"), ("orders", "total")}
    union = {("users", "id"), ("users", "name"), ("orders", "total")}
    tags = tag_columns(fwd, bwd, union)
    _assert(tags[("users", "id")] == "[F+B]", "users.id F+B")
    _assert(tags[("users", "name")] == "[F]", "users.name F only")
    _assert(tags[("orders", "total")] == "[B]", "orders.total B only")


def test_c_v1_normalize_set_accepts_strings_and_tuples():
    print("\n[test] _normalize_set: list[str] + list[tuple] + None")
    a = _normalize_set(["users.id", "orders.total"])
    _assert(a == {("users", "id"), ("orders", "total")}, "from strings")
    b = _normalize_set([("users", "id")])
    _assert(b == {("users", "id")}, "from tuples")
    c = _normalize_set(None)
    _assert(c == set(), "from None")


def test_c_v1_format_tagged_schema_includes_legend_and_tags():
    print("\n[test] C-v1 format_tagged_schema: legend + per-col tag")
    m4 = {"users": ["id", "name"], "orders": ["total"]}
    fwd = ["users.id", "users.name"]
    bwd = ["users.id", "orders.total"]
    text = format_tagged_schema(m4, fwd, bwd)
    _assert("[F+B]=High confidence" in text, "legend present")
    _assert("- id [F+B]" in text, "id tagged [F+B]")
    _assert("- name [F]" in text, "name tagged [F]")
    _assert("- total [B]" in text, "total tagged [B]")


def test_c_v1_schema_content_invariance():
    print("\n[test] C-v1 invariance: 모든 M4 col 이 schema text 에 표현")
    m4 = {"users": ["id", "name", "age"], "orders": ["total", "user_id"]}
    text = format_tagged_schema(
        m4, ["users.id"],
        ["users.name", "orders.total", "orders.user_id", "users.age"],
    )
    present = _columns_from_schema_str(text, m4)
    expected = {f"{t}.{c}" for t, cs in m4.items() for c in cs}
    _assert(present == expected, f"all M4 cols present, got {sorted(present)}")


# ============================================================
# C-v3 — flat_merged_serializer
# ============================================================
def test_c_v3b_flat_no_fk():
    print("\n[test] C-v3b flat (no FK)")
    m4 = {"users": ["id", "name"], "orders": ["total"]}
    text = format_flat_schema(m4, fk_relations=None)
    _assert(text.startswith("[Available Columns]"), "header present")
    _assert("users.id" in text and "users.name" in text and "orders.total" in text,
            "all 3 cols flat-listed")
    _assert("Foreign Key Relations" not in text, "no FK section in v3b")


def test_c_v3a_flat_with_fk():
    print("\n[test] C-v3a flat with FK relations (filtered to union)")
    m4 = {"users": ["id"], "orders": ["user_id"]}
    fks = [
        ("orders", "user_id", "users", "id"),       # both endpoints in union
        ("orders", "product_id", "products", "id"), # neither in union — should skip
    ]
    text = format_flat_schema(m4, fk_relations=fks)
    _assert("[Foreign Key Relations]" in text, "FK section present")
    _assert("orders.user_id -> users.id" in text, "in-union FK shown")
    _assert("products" not in text, "out-of-union FK skipped")


def test_c_v3_schema_content_invariance():
    print("\n[test] C-v3 invariance: M4 col 집합이 텍스트에 모두 노출")
    m4 = {"a": ["x", "y"], "b": ["z"]}
    text = format_flat_schema(m4, fk_relations=None)
    expected = {"a.x", "a.y", "b.z"}
    present = {c for c in expected if c in text}
    _assert(present == expected, f"all flat cols present, got {sorted(present)}")


def test_c_v3a_fk_relations_empty_list_no_section():
    print("\n[test] C-v3a empty fk_relations -> no FK section")
    m4 = {"users": ["id"]}
    text = format_flat_schema(m4, fk_relations=[])
    _assert("[Foreign Key Relations]" not in text, "empty fks -> no section")


def test_c_v3_malformed_fk_skipped():
    print("\n[test] C-v3a malformed FK tuple -> skip silently")
    m4 = {"a": ["x"]}
    text = format_flat_schema(m4, fk_relations=[("a", "x", "b")])  # 3-tuple, malformed
    _assert("[Foreign Key Relations]" not in text, "malformed skipped")


# ============================================================
# C-v2 — question_enricher
# ============================================================
def test_c_v2_format_schema_for_enrichment_no_full_schema():
    print("\n[test] C-v2 format_schema: M4 schema only (No Full Schema)")
    m4 = {"users": ["id", "name"]}
    text = format_schema_for_enrichment(m4)
    _assert("Table: users" in text, "table header present")
    _assert("- id" in text and "- name" in text, "cols listed")
    _assert("orders" not in text, "no other tables")


def test_c_v2_enrich_question_cache_miss_then_hit():
    print("\n[test] C-v2 enrich: cache miss -> LLM, hit -> 0 LLM")
    cache = EnrichmentCache()
    llm = _MockLLM(response="Enriched: list users.id and users.name where active.")
    out1 = enrich_question(
        question="who are active users?", m4_schema={"users": ["id", "name"]},
        few_shot_examples=[], llm_client=llm, model_name="m", cache=cache,
    )
    _assert(llm.calls == 1, "1 LLM call on miss")
    _assert("Enriched:" in out1, "enriched returned")
    out2 = enrich_question(
        question="who are active users?", m4_schema={"users": ["id", "name"]},
        few_shot_examples=[], llm_client=llm, model_name="m", cache=cache,
    )
    _assert(llm.calls == 1, "still 1 LLM call (cache hit)")
    _assert(out2 == out1, "same enriched returned")
    _assert(cache.stats()["hits"] == 1, "1 hit recorded")


def test_c_v2_enrich_different_schema_invalidates_cache():
    print("\n[test] C-v2 different M4 schema -> cache miss")
    cache = EnrichmentCache()
    llm = _MockLLM(response="enriched v1")
    enrich_question("q", {"users": ["id"]}, [], llm, "m", cache=cache)
    llm.response = "enriched v2"
    out2 = enrich_question("q", {"users": ["id", "name"]}, [], llm, "m", cache=cache)
    _assert(llm.calls == 2, "2 LLM calls (different schema)")
    _assert(out2 == "enriched v2", "new response for new schema")


def test_c_v2_enrich_llm_error_fallback_to_original():
    print("\n[test] C-v2 LLM exception -> fallback original (recall-safe)")
    class _ErrLLM:
        def generate_text(self, *a, **kw):
            raise RuntimeError("network down")
    out = enrich_question(
        question="original question?", m4_schema={"users": ["id"]},
        few_shot_examples=[], llm_client=_ErrLLM(), model_name="m",
        cache=EnrichmentCache(), fallback_on_error=True,
    )
    _assert(out == "original question?", "fallback to original")


def test_c_v2_few_shot_inserted_in_prompt():
    print("\n[test] C-v2 few-shot 예시가 prompt 에 inject")
    captured = []
    class _Cap:
        def generate_text(self, prompt, model, temperature, **kw):
            captured.append(prompt)
            return "enriched"
    few_shots = [
        {"question": "ex Q1", "schema": {"t1": ["c1"]}, "enriched_question": "ex E1"},
        {"question": "ex Q2", "schema": {"t2": ["c2"]}, "enriched_question": "ex E2"},
    ]
    enrich_question(
        question="user Q", m4_schema={"u": ["i"]},
        few_shot_examples=few_shots, llm_client=_Cap(), model_name="m",
        cache=EnrichmentCache(),
    )
    p = captured[0]
    _assert("Example 1" in p and "Example 2" in p, "2 examples present")
    _assert("ex E1" in p and "ex E2" in p, "enriched samples in prompt")
    _assert("Your Task" in p, "task section present")
    _assert(ENRICHMENT_SYSTEM_PROMPT.split("\n")[0] in p, "system prompt inlined")


def test_c_v2_no_full_schema_in_prompt():
    print("\n[test] C-v2 Wave 11 핵심 제약: Full Schema 미포함")
    captured = []
    class _Cap:
        def generate_text(self, prompt, model, temperature, **kw):
            captured.append(prompt)
            return "ok"
    enrich_question(
        question="q", m4_schema={"users": ["id"]},
        few_shot_examples=[], llm_client=_Cap(), model_name="m",
        cache=EnrichmentCache(),
    )
    p = captured[0]
    _assert("Table: users" in p, "M4 table listed")
    _assert("- id" in p, "M4 col listed")
    for forbidden in ["orders", "products", "transactions", "customers"]:
        _assert(forbidden not in p, f"'{forbidden}' NOT in prompt")


# ============================================================
# 5 cells 통합 — Schema Content Invariance
# ============================================================
def test_schema_invariance_all_5_cells():
    print("\n[test] Invariance: 5 cells all preserve M4 column set")
    m4 = {"users": ["id", "name"], "orders": ["total", "user_id"]}
    fwd = ["users.id", "users.name"]
    bwd = ["users.id", "orders.total", "orders.user_id"]
    fks = [("orders", "user_id", "users", "id")]
    expected = {"users.id", "users.name", "orders.total", "orders.user_id"}

    text_v1 = format_tagged_schema(m4, fwd, bwd)
    _assert(_columns_from_schema_str(text_v1, m4) == expected, "C-v1 cols invariant")

    text_v3a = format_flat_schema(m4, fk_relations=fks)
    pres_v3a = {c for c in expected if c in text_v3a}
    _assert(pres_v3a == expected, "C-v3a cols invariant")

    text_v3b = format_flat_schema(m4, fk_relations=None)
    pres_v3b = {c for c in expected if c in text_v3b}
    _assert(pres_v3b == expected, "C-v3b cols invariant")

    sch_enrich = format_schema_for_enrichment(m4)
    pres_v2 = {f"{t}.{c}" for t, cs in m4.items() for c in cs
               if f"Table: {t}" in sch_enrich and f"- {c}" in sch_enrich}
    _assert(pres_v2 == expected, "C-v2/Comb-C enrichment schema cols invariant")


# ============================================================
# Pipeline integration — BidirectionalFilter F/B set logging
# ============================================================
def test_wave11_pipeline_loads_few_shots_and_filters_by_db_id():
    print("\n[test] schema_linking 의 few-shot loader: JSON 형식 정합 + db_id leakage filter")
    # JSON 파일 형식 모사: planning/few_shot_examples_wave11_2026-05-19.json 정합
    import tempfile, json as _json
    raw = {
        "_meta": {"note": "wave11 few-shots"},
        "examples": [
            {"db_id": "california_schools", "difficulty": "simple",
             "original_question": "Q1", "filtered_schema": {"frpm": ["col1"]},
             "enriched_question": "E1"},
            {"db_id": "card_games", "difficulty": "moderate",
             "original_question": "Q2", "filtered_schema": {"cards": ["id"]},
             "enriched_question": "E2"},
            {"db_id": "card_games", "difficulty": "simple",
             "original_question": "Q3", "filtered_schema": {"cards": ["name"]},
             "enriched_question": "E3"},
            {"db_id": "european_football_2", "difficulty": "challenging",
             "original_question": "Q4", "filtered_schema": {"Match": ["date"]},
             "enriched_question": "E4"},
        ],
    }
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    _json.dump(raw, tf)
    tf.close()
    try:
        from pipeline.schema_linking import SchemaLinkingPipeline
        # Bypass __init__ — 필요한 attribute 만 set 후 helper 호출
        pipe = SchemaLinkingPipeline.__new__(SchemaLinkingPipeline)
        pipe._enrichment_few_shots = []
        # 직접 loader logic mimick — JSON 로드 + 정규화
        with open(tf.name, "r", encoding="utf-8") as f:
            loaded = _json.load(f)
        ex_list = loaded["examples"]
        pipe._enrichment_few_shots = [
            {
                "question": ex.get("original_question", ""),
                "schema": ex.get("filtered_schema", {}),
                "enriched_question": ex.get("enriched_question", ""),
                "db_id": ex.get("db_id"),
                "difficulty": ex.get("difficulty"),
            }
            for ex in ex_list
        ]
        # data leakage filter — test_db_id = "card_games"
        out = pipe._select_few_shots_for_query(
            test_db_id="card_games", max_examples=8,
        )
        db_ids_in_out = {ex["db_id"] for ex in out}
        _assert("card_games" not in db_ids_in_out,
                f"card_games filtered out, got {db_ids_in_out}")
        _assert(len(out) == 2,
                f"2 examples remain (california_schools + european_football_2), got {len(out)}")

        # 다른 test_db_id → 모든 examples 사용 가능
        out_all = pipe._select_few_shots_for_query(
            test_db_id="other_db", max_examples=8,
        )
        _assert(len(out_all) == 4, f"all 4 examples for unrelated test_db, got {len(out_all)}")
    finally:
        os.unlink(tf.name)


def test_wave11_pipeline_round_robin_difficulty_sampling():
    print("\n[test] _select_few_shots round-robin difficulty 분포")
    from pipeline.schema_linking import SchemaLinkingPipeline
    pipe = SchemaLinkingPipeline.__new__(SchemaLinkingPipeline)
    pipe._enrichment_few_shots = [
        {"question": f"Q{i}", "schema": {}, "enriched_question": f"E{i}",
         "db_id": f"db{i}", "difficulty": d}
        for i, d in enumerate(
            ["simple"] * 4 + ["moderate"] * 4 + ["challenging"] * 4
        )
    ]
    out = pipe._select_few_shots_for_query(test_db_id=None, max_examples=6)
    diffs = [ex["difficulty"] for ex in out]
    _assert(len(out) == 6, "6 examples returned")
    # 순서: simple/moderate/challenging round-robin → 2 simple + 2 moderate + 2 challenging
    _assert(diffs.count("simple") == 2, f"2 simple, got {diffs.count('simple')}")
    _assert(diffs.count("moderate") == 2, f"2 moderate, got {diffs.count('moderate')}")
    _assert(diffs.count("challenging") == 2, f"2 challenging, got {diffs.count('challenging')}")


def test_bidirectional_filter_exposes_forward_backward_set():
    print("\n[test] BidirectionalFilter stats + filter_info expose forward_set/backward_set")
    from modules.filters.bidirectional_filter import BidirectionalFilter

    class _SeqLLM:
        def __init__(self, responses):
            self._q = list(responses)
        def generate_text(self, prompt, model, temperature, **kw):
            return self._q.pop(0) if self._q else ""

    flt = BidirectionalFilter(
        model_name="mock", provider=None, db_dir="/x", num_examples=0,
    )
    fwd_resp = json.dumps({"users": ["id", "name"]})
    bwd_resp = json.dumps({"users": ["id"], "orders": ["total"]})
    flt.client = _SeqLLM([fwd_resp, bwd_resp])
    result = flt.refine(
        query="q",
        subgraph={"users": ["id", "name"], "orders": ["total"]},
        db_id=None,
    )
    stats = result["stats"]
    info = result["filter_info"]
    _assert("forward_set" in stats and "backward_set" in stats,
            "stats has forward_set + backward_set (Wave 11 naming)")
    _assert(set(stats["forward_set"]) == {"users.id", "users.name"},
            f"forward_set correct, got {sorted(stats['forward_set'])}")
    _assert(set(stats["backward_set"]) == {"users.id", "orders.total"},
            f"backward_set correct, got {sorted(stats['backward_set'])}")
    _assert("filter_forward_set" in info and "filter_backward_set" in info,
            "filter_info has filter_forward_set + filter_backward_set")
    # C-v1 end-to-end smoke
    text = format_tagged_schema(
        m4_output={"users": ["id", "name"], "orders": ["total"]},
        forward_set=stats["forward_set"],
        backward_set=stats["backward_set"],
    )
    _assert("- id [F+B]" in text, "C-v1 users.id [F+B]")
    _assert("- name [F]" in text, "C-v1 users.name [F]")
    _assert("- total [B]" in text, "C-v1 orders.total [B]")


def run_all():
    tests = [
        test_c_v1_tag_columns_categorizes_F_B_FB,
        test_c_v1_normalize_set_accepts_strings_and_tuples,
        test_c_v1_format_tagged_schema_includes_legend_and_tags,
        test_c_v1_schema_content_invariance,
        test_c_v3b_flat_no_fk,
        test_c_v3a_flat_with_fk,
        test_c_v3_schema_content_invariance,
        test_c_v3a_fk_relations_empty_list_no_section,
        test_c_v3_malformed_fk_skipped,
        test_c_v2_format_schema_for_enrichment_no_full_schema,
        test_c_v2_enrich_question_cache_miss_then_hit,
        test_c_v2_enrich_different_schema_invalidates_cache,
        test_c_v2_enrich_llm_error_fallback_to_original,
        test_c_v2_few_shot_inserted_in_prompt,
        test_c_v2_no_full_schema_in_prompt,
        test_schema_invariance_all_5_cells,
        test_wave11_pipeline_loads_few_shots_and_filters_by_db_id,
        test_wave11_pipeline_round_robin_difficulty_sampling,
        test_bidirectional_filter_exposes_forward_backward_set,
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
