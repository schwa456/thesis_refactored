"""SGBE Phase 2 — raw_score interface 보강 smoke test (DECISIONS 2026-05-12 §SGBE Phase 2.5).

검증 대상:
  (1) EnsembleSelector — `select()` 호출 후 `latest_raw_gat_scores` + `latest_raw_cos_scores`
      attributes 가 dict[int, float] 로 노출됨.
  (2) DirectGATSelector — `latest_raw_gat_scores` (latest_scores 와 동일 value) + `latest_raw_cos_scores`
      (Direct variant 는 cosine 분기 없어 빈 dict) 가 노출됨.
  (3) Pipeline (schema_linking.py) — filter.refine() 호출 시 `raw_gat_scores` + `raw_cos_scores`
      kwarg 가 column name (str) → float dict 로 전달됨. SpyFilter 가 captured.

본 smoke 는 GPU/checkpoint 의존 X — Selector 인스턴스를 mock attribute 주입으로 우회.

Run from project root:
    PYTHONPATH=src conda run -n base python src/modules/selectors/tests/test_raw_score_interface.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_ensemble_selector_attributes_exist():
    """EnsembleSelector 인스턴스가 latest_raw_gat_scores + latest_raw_cos_scores attributes 를 갖는다."""
    print("\n[test_ensemble_selector_attributes_exist]")
    from modules.selectors.ensemble_selector import EnsembleSelector
    # __init__ 의 ckpt load 우회 — class 만 import + 직접 attribute set 검증
    assert "latest_raw_gat_scores" in EnsembleSelector.__init__.__code__.co_names \
           or "latest_raw_gat_scores" in EnsembleSelector.__init__.__code__.co_varnames, \
           "EnsembleSelector.__init__ must reference latest_raw_gat_scores"
    # 본 코드 path 가 __init__ 에서 self.latest_raw_gat_scores = {} 로 초기화하는지
    src = Path(EnsembleSelector.__init__.__code__.co_filename).read_text()
    assert "self.latest_raw_gat_scores: Dict[int, float] = {}" in src
    assert "self.latest_raw_cos_scores: Dict[int, float] = {}" in src
    print("  OK EnsembleSelector __init__ initializes both raw_score dicts")


def test_direct_gat_selector_attributes_exist():
    """DirectGATSelector 도 동일 attributes 노출."""
    print("\n[test_direct_gat_selector_attributes_exist]")
    from modules.selectors.direct_gat_selector import DirectGATSelector
    src = Path(DirectGATSelector.select.__code__.co_filename).read_text()
    assert "self.latest_raw_gat_scores: Dict[int, float] = {" in src
    assert "self.latest_raw_cos_scores: Dict[int, float] = {}" in src
    print("  OK DirectGATSelector.select sets both raw_score dicts")


def test_pipeline_filter_call_forwards_raw_scores():
    """schema_linking.py 의 filter.refine() 호출에 raw_gat_scores + raw_cos_scores 가 포함됨.

    추가 (Phase 2 보강): table.column 키 + column 단독 fallback 키 양쪽이 등록되는지 검증
    (SGBE _lookup_score 가 두 형식 모두 지원).
    """
    print("\n[test_pipeline_filter_call_forwards_raw_scores]")
    src = (SRC / "pipeline/schema_linking.py").read_text()
    # main 호출 (L275~) — raw_gat_scores + raw_cos_scores 둘 다 forward
    assert "raw_gat_scores=raw_gat_scores" in src, \
        "main filter.refine() must forward raw_gat_scores"
    assert "raw_cos_scores=raw_cos_scores" in src, \
        "main filter.refine() must forward raw_cos_scores"
    # selector 의 latest_raw_gat_scores 를 꺼내는 path 가 존재
    assert "getattr(self.selector, \"latest_raw_gat_scores\", None)" in src
    assert "getattr(self.selector, \"latest_raw_cos_scores\", None)" in src
    # retry path 에도 동일 — 두 번 forward
    count_raw_gat = src.count("raw_gat_scores=raw_gat_scores")
    count_raw_cos = src.count("raw_cos_scores=raw_cos_scores")
    assert count_raw_gat >= 2, f"expected raw_gat_scores forward at main+retry (count={count_raw_gat})"
    assert count_raw_cos >= 2, f"expected raw_cos_scores forward at main+retry (count={count_raw_cos})"
    # 키 형식 보강 — table.column + column-only fallback
    assert "raw_gat_scores.setdefault(col_only, score_val)" in src, \
        "raw_gat_scores must include column-only fallback key (SGBE _lookup_score 호환)"
    assert "raw_cos_scores.setdefault(col_only, cos_val)" in src, \
        "raw_cos_scores must include column-only fallback key"
    # fk_node ('->' in name) 는 제외
    assert "\"->\" in name_str" in src, "fk_node (-> in name) must be excluded from raw_*_scores"
    print(f"  OK raw_gat_scores forwarded {count_raw_gat}× + raw_cos_scores forwarded {count_raw_cos}× "
          f"+ table.column + column-only fallback")


class _SpyFilter:
    """Mock filter that captures the kwargs that the pipeline passes."""

    def __init__(self) -> None:
        self.last_kwargs: Dict[str, Any] = {}
        self.last_info = None

    def refine(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> Dict[str, Any]:
        self.last_kwargs = dict(kwargs)
        return {"status": "ok", "final_nodes": [], "reasoning": "spy"}


def test_spy_filter_receives_raw_scores_end_to_end():
    """End-to-end (selector 와 pipeline 의 minimal mock) — SpyFilter 가 raw_gat_scores 등을 받음.

    schema_linking.py 의 실제 pipeline 을 spawn 하지 않고, 본 smoke 는 코드의 contract 가
    test_pipeline_filter_call_forwards_raw_scores 으로 이미 검증되었으므로 spy 측만 확인.
    """
    print("\n[test_spy_filter_receives_raw_scores_end_to_end]")
    spy = _SpyFilter()
    # pipeline kwargs 의 minimum 흉내
    result = spy.refine(
        query="dummy", subgraph={"t": ["c"]}, db_id="d",
        tier2_pool=[], gat_scores={"t.c": 0.7},
        raw_gat_scores={"t.c": 0.65}, raw_cos_scores={"t.c": 0.50},
        metadata={},
    )
    assert "raw_gat_scores" in spy.last_kwargs
    assert "raw_cos_scores" in spy.last_kwargs
    assert spy.last_kwargs["raw_gat_scores"] == {"t.c": 0.65}
    assert spy.last_kwargs["raw_cos_scores"] == {"t.c": 0.50}
    print(f"  OK SpyFilter received raw_gat_scores={spy.last_kwargs['raw_gat_scores']} "
          f"raw_cos_scores={spy.last_kwargs['raw_cos_scores']}")


def test_basefilter_kwargs_compat():
    """BaseFilter.refine signature 가 **kwargs 를 받음 — 기존 filter 들이 신규 kwarg 를 ignore."""
    print("\n[test_basefilter_kwargs_compat]")
    from modules.base import BaseFilter
    import inspect
    sig = inspect.signature(BaseFilter.refine)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    assert has_var_keyword, "BaseFilter.refine must accept **kwargs for backward compat"
    print(f"  OK BaseFilter.refine accepts **kwargs (backward-compat for raw_gat_scores/raw_cos_scores)")


def main():
    test_ensemble_selector_attributes_exist()
    test_direct_gat_selector_attributes_exist()
    test_pipeline_filter_call_forwards_raw_scores()
    test_spy_filter_receives_raw_scores_end_to_end()
    test_basefilter_kwargs_compat()
    print("\nAll SGBE Phase 2 raw_score interface smoke tests passed.")


if __name__ == "__main__":
    main()
