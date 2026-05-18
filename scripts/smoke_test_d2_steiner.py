"""Smoke test — Wave 8 D2 FK/PK Connectivity Steiner Closure (LLM 0×).

DECISIONS 2026-05-18 Wave 8 §2 D2 + 학술 agent §2 정합.

검증 범위:
  1. steiner_closure (direct FK): california_schools 의 schools/frpm 사이
     frpm.CDSCode FK 추가 검증
  2. steiner_closure (테이블 1개) — early return (no-op)
  3. steiner_closure_with_bridge — 직접 FK 가 이미 있는 경우 (bridge 불필요)
  4. steiner_closure (added_only 측정) — added/gold count, steiner_precision sanity
  5. D2SteinerFilter (LLM 차단 — M4 monkey-patch) end-to-end refine() 검증

LLM 호출 없음 — algorithm + monkey-patched M4 stub.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Path setup — src/ 추가
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from modules.filters.steiner_closure import (
    build_local_fk_graph,
    count_added_gold,
    steiner_closure,
    steiner_closure_with_bridge,
)


def load_california_schools_meta():
    import json

    p = ROOT / "data/processed/db_fk_metadata.json"
    if not p.exists():
        print(f"[ERR] db_fk_metadata.json 미존재: {p}")
        print("Run: python -m modules.builders.db_fk_extractor")
        sys.exit(1)
    payload = json.loads(p.read_text())
    return payload.get("california_schools", {})


def test_direct_fk_closure():
    print("=" * 60)
    print("Test 1: direct FK closure — schools + frpm")
    print("=" * 60)
    meta = load_california_schools_meta()
    print(f"  fk_constraints count: {len(meta.get('fk_constraints', []))}")

    # Extractor 후보 — 두 테이블의 다양한 column 포함
    extractor_output = {
        "schools": ["CDSCode", "School", "District", "City", "County"],
        "frpm": [
            "CDSCode",
            "School Name",
            "Enrollment (K-12)",
            "Free Meal Count (K-12)",
        ],
        "satscores": ["cds", "sname", "NumTstTakr"],
    }

    # M4 union — frpm.CDSCode 가 빠진 상태 (직접 FK 컬럼 누락 시나리오)
    m4_union = {
        "schools": ["CDSCode"],
        "frpm": ["School Name"],
    }

    closed, added = steiner_closure(m4_union, extractor_output, meta)
    print(f"  M4 union: {m4_union}")
    print(f"  closed: {closed}")
    print(f"  added: {added}")

    # 검증: frpm.CDSCode 가 추가되어야 함 (FK from frpm to schools)
    assert "frpm" in added, f"frpm should be in added: {added}"
    assert "CDSCode" in added["frpm"], (
        f"frpm.CDSCode should be added (FK to schools.CDSCode): added={added}"
    )
    # closed 에는 모두 포함되어야 함
    assert "CDSCode" in closed["frpm"]
    assert "School Name" in closed["frpm"]
    assert "CDSCode" in closed["schools"]
    print("  [PASS] frpm.CDSCode FK column added by direct closure")


def test_single_table_noop():
    print("=" * 60)
    print("Test 2: single-table m4_union → no-op")
    print("=" * 60)
    meta = load_california_schools_meta()
    extractor_output = {
        "schools": ["CDSCode", "School"],
    }
    m4_union = {"schools": ["School"]}
    closed, added = steiner_closure(m4_union, extractor_output, meta)
    print(f"  closed: {closed}")
    print(f"  added: {added}")
    assert added == {}, f"single-table should produce empty added: {added}"
    assert closed == {"schools": ["School"]}
    print("  [PASS] single-table early return")


def test_bridge_path_already_direct():
    print("=" * 60)
    print("Test 3: bridge variant — direct FK existing (no new bridge)")
    print("=" * 60)
    meta = load_california_schools_meta()
    extractor_output = {
        "schools": ["CDSCode", "School"],
        "frpm": ["CDSCode", "School Name"],
    }
    m4_union = {
        "schools": ["School"],
        "frpm": ["School Name"],
    }
    closed, added = steiner_closure_with_bridge(
        m4_union, extractor_output, meta, max_bridge_depth=1
    )
    print(f"  closed: {closed}")
    print(f"  added: {added}")
    # Direct FK 가 존재 (frpm.CDSCode ↔ schools.CDSCode) — direct closure 단계에서 추가됨
    assert "frpm" in added and "CDSCode" in added["frpm"]
    assert "schools" in added and "CDSCode" in added["schools"]
    print("  [PASS] bridge variant correctly identifies direct FK")


def test_added_gold_precision():
    print("=" * 60)
    print("Test 4: added_only / gold counts — Steiner Precision")
    print("=" * 60)
    meta = load_california_schools_meta()
    extractor_output = {
        "schools": ["CDSCode", "School", "District"],
        "frpm": ["CDSCode", "School Name", "Enrollment (K-12)"],
        "satscores": ["cds", "sname"],
    }
    m4_union = {
        "schools": ["School"],
        "frpm": ["School Name"],
        "satscores": ["sname"],
    }
    # Gold 정답 — FK 컬럼 모두 정답에 포함되어 있다고 가정
    gold = {
        "schools": ["School", "CDSCode"],
        "frpm": ["School Name", "CDSCode"],
        "satscores": ["sname", "cds"],
    }

    closed, added = steiner_closure(m4_union, extractor_output, meta)
    added_count = sum(len(v) for v in added.values())
    added_gold = count_added_gold(added, gold)
    precision = added_gold / added_count if added_count > 0 else None

    print(f"  added: {added}")
    print(f"  added_count={added_count}, added_gold={added_gold}, "
          f"precision={precision:.4f}" if precision else
          f"  added_count={added_count}, added_gold={added_gold}, precision=None")
    assert added_count > 0
    assert added_gold == added_count, (
        f"All added FK cols are FK gold in this test: {added_gold}/{added_count}"
    )
    assert precision == 1.0
    print("  [PASS] Steiner Precision = 1.0 (all added cols are gold)")


def test_d2_filter_with_monkeypatched_m4():
    print("=" * 60)
    print("Test 5: D2SteinerFilter end-to-end (M4 monkey-patched, LLM 0×)")
    print("=" * 60)

    # M4 stub — call_forward / call_prompt / refine 차단
    from modules.filters import d2_steiner_filter as d2_mod
    from modules.filters.bidirectional_filter import BidirectionalFilter

    extractor_output = {
        "schools": ["CDSCode", "School", "District"],
        "frpm": ["CDSCode", "School Name"],
        "satscores": ["cds", "sname"],
    }
    # M4 가 반환할 stub union (frpm.CDSCode 누락 → Steiner 가 회복해야 함)
    stub_m4_final_nodes = ["schools.School", "frpm.School Name", "satscores.sname"]
    stub_m4_reasoning = "[M4 stub] union=3 nodes"
    stub_m4_filter_info = {"filter_type": "BidirectionalFilter", "stub": True}
    stub_m4_stats = {
        "forward_count": 3, "backward_count": 0, "union_count": 3,
    }

    def stub_refine(self, query, subgraph, db_id=None, gold=None, **kwargs):
        return {
            "status": "Answerable",
            "final_nodes": list(stub_m4_final_nodes),
            "reasoning": stub_m4_reasoning,
            "stats": stub_m4_stats,
            "filter_info": dict(stub_m4_filter_info),
        }

    # __init__ 가 _make_llm_client 를 호출하므로 monkey-patch
    original_init = BidirectionalFilter.__init__
    original_refine = BidirectionalFilter.refine

    def stub_init(self, *args, **kwargs):
        # 실제 LLM client 생성 skip — 필요한 minimal attributes 만 세팅
        self.model_name = kwargs.get("model_name", "stub")
        self.forward_section = kwargs.get("forward_section", "stub")
        self.backward_section = kwargs.get("backward_section", "stub")
        self.bidirectional_forward_prompt_mode = kwargs.get(
            "bidirectional_forward_prompt_mode"
        )
        self.bidirectional_forward_voting_strategy = kwargs.get(
            "bidirectional_forward_voting_strategy", "MAJORITY"
        )
        self._forward_is_voting = False

    BidirectionalFilter.__init__ = stub_init
    BidirectionalFilter.refine = stub_refine

    try:
        # FK metadata cache 강제 리셋
        d2_mod.D2SteinerFilter._DB_FK_METADATA_CACHE = None
        d2_mod.D2SteinerFilter._DB_FK_METADATA_CACHE_PATH = None

        flt = d2_mod.D2SteinerFilter(
            model_name="stub-model",
            provider="glm",
            d2_steiner_closure=True,
            d2_1hop_bridge=False,
            db_fk_metadata_path=str(ROOT / "data/processed/db_fk_metadata.json"),
        )
        result = flt.refine(
            query="dummy",
            subgraph=extractor_output,
            db_id="california_schools",
            gold={
                "schools": ["School", "CDSCode"],
                "frpm": ["School Name", "CDSCode"],
                "satscores": ["sname", "cds"],
            },
        )

        print(f"  final_nodes: {result['final_nodes']}")
        print(f"  reasoning: {result['reasoning']}")
        print(f"  stats: {result['stats']}")

        # 검증: frpm.CDSCode + schools.CDSCode + satscores.cds 가 추가되어야 함
        final_set = set(result["final_nodes"])
        assert "frpm.CDSCode" in final_set, (
            f"frpm.CDSCode missing: {final_set}"
        )
        assert "schools.CDSCode" in final_set
        assert "satscores.cds" in final_set
        # 측정 메타
        info = result["filter_info"]
        assert info.get("filter_d2_steiner_closure") is True
        assert info.get("filter_d2_variant") == "direct_fk"
        assert int(info.get("filter_d2_added_count", 0)) >= 2
        assert info.get("filter_d2_added_gold_count") == info.get(
            "filter_d2_added_count"
        )  # all added are gold
        assert info.get("filter_d2_steiner_precision") == 1.0
        # M4 nested filter_info 포함 확인
        assert "filter_m4_filter_info" in info
        print(
            "  [PASS] D2SteinerFilter end-to-end: added_count="
            f"{info['filter_d2_added_count']}, "
            f"precision={info['filter_d2_steiner_precision']}"
        )
    finally:
        BidirectionalFilter.__init__ = original_init
        BidirectionalFilter.refine = original_refine


def main():
    print("\n" + "=" * 60)
    print("Wave 8 D2 FK/PK Connectivity Steiner Closure — Smoke Test")
    print("=" * 60 + "\n")
    test_direct_fk_closure()
    print()
    test_single_table_noop()
    print()
    test_bridge_path_already_direct()
    print()
    test_added_gold_precision()
    print()
    test_d2_filter_with_monkeypatched_m4()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
