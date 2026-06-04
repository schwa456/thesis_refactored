# Filter (LLM Schema Linking Refinement) 모듈

> **루트 CLAUDE.md 참조 (읽기 전용)**: 실험 실행, 디렉토리 구조, 문서화 규칙 등 프로젝트 전역 규칙은 [/home/hyeonjin/thesis_refactored/CLAUDE.md](/home/hyeonjin/thesis_refactored/CLAUDE.md)를 반드시 먼저 읽고 따른다. 단, 루트 CLAUDE.md는 수정하지 않는다 — 수정이 필요하면 루트 세션에 요청한다.

## 이 세션의 집중 주제
**LLM 기반 최종 refine**. Subgraph에서 불필요한 노드 제거 + 필요 시 복원.
Builder/Selector/Extractor 내부는 가급적 언급하지 않고,
**"PCST가 넘긴 subgraph를 LLM이 어떻게 검증/정제/확장하느냐"** 관점만 유지.

## 현재 Filter 구현
[이 폴더](.):
- **xiyan_filter.py** `XiYanFilter` — 메인 pruning baseline (Qwen3-Coder-30B-A3B-Instruct-FP8)
- **agents.py** `SingleAgentFilter`, `AdaptiveMultiAgentFilter` — Semantic+Structural+Skeptic
- **reflection_filter.py** `ReflectionFilter` — Propose → Critique → Revise (원 subgraph 밖 노드 재도입 허용)
- **verifier_filter.py** `VerifierFilter` — XiYan 초기 필터 + NL unit test 생성/검증 + missing 복원
- **bidirectional_agent_filter.py** `TieredBidirectionalAgentFilter` — Prune(Tier-1) + Restore(Tier-1 dropped ∪ Tier-2)
- **adaptive_depth_filter.py** `AdaptiveDepthFilter` — GAT confidence 기반 depth 분기 (XiYan / Reflection / Bidirectional)
- **stacked_filter.py** — 여러 Filter 직렬 연결
- **bidirectional_filter.py** `BidirectionalFilter` — M4 anchor (Forward + Backward union, 2 LLM/q)
- **d2_steiner_filter.py** `D2SteinerFilter` — Wave 8 D2: M4 위에 FK/PK Connectivity Steiner Closure (LLM 0× 추가, direct_fk / bridge_1hop variants). 의존: `steiner_closure.py` algorithm utility + `builders/db_fk_extractor.py` 메타데이터.
- **tools/graph_tools.py** — F3용 graph-native tool (get_neighbors, get_fk_path, get_gat_score, get_column_examples, get_tier)

## V7-W1 FKH (FK Hint 직렬화, 2026-06-05)
RFP #3 spec ([planning/extractor/scholar_agent_extractor_rfp_2026-06-04.md §3](../../../planning/extractor/scholar_agent_extractor_rfp_2026-06-04.md)) 정합 — Extractor 수정 없이 LLM prompt 의 schema 부분 위 FK 관계 명시.
- **구현 위치**: `XiYanFilter` 위 `fk_hint_format`/`fk_hint_position` 옵션 (default `"none"`/`"inline"` = baseline 호환). `_serialize_schema()` helper 가 base mschema 위 [src/utils/schema_serializer.py](../../utils/schema_serializer.py) 의 `serialize_schema_with_fk_hints()` 호출.
- **5 cells × 5 seeds = 25 configs**: `configs/experiments/abl/v7_extractor_redesign/fkh_*_seed*.yaml`. cell 정의:
  | cell_id | fk_hint_format | fk_hint_position |
  |---------|----------------|------------------|
  | fkh_00  | none           | inline           |
  | fkh_01  | explicit       | inline           |
  | fkh_02  | explicit       | prefix           |
  | fkh_03  | compact        | inline           |
  | fkh_04  | compact        | suffix           |
- **게이트**: EX ≥ baseline c01_01_wave7_relog (0.5176) + 0.0030, 5 seeds 평균 p < 0.05. R/P/F1 동일 보장 (extracted_nodes/edges 자체 변경 없음 — 변화 시 구현 버그).
- **smoke test**: `scripts/smoke_test_v7_w1_fkh.py` (LLM 0×, CPU only, 6/6 PASS — fkh_00~04 + invariance).
- **launcher**: `scripts/run_v7_w1_fkh_sweep.sh` (25 configs 순차, GPU 0,1 외부 export).
- **FK 정보 소스**: `metadata['fk_to_id']` keys (`"src_t.src_c->dst_t.dst_c"` 형식, builder `_generate_fk_descriptions()` 정합) → extracted_subgraph 위 양쪽 column 모두 존재하는 FK 만 hint 위 포함 (context window 보호).

## XiYan Filter 핵심
- **Value Retrieval**: DB에서 실제 값 샘플을 프롬프트에 포함
- prompt에 column-level 예시값 제공하여 LLM이 schema-value 연결 판단
- Output: `{status, final_nodes, reasoning}`

## 인터페이스 계약
`refine(query, subgraph, db_id, tier2_pool=None, gat_scores=None, metadata=None, **kwargs)` → `Dict`
- `subgraph`: `{table_name: [col1, col2, ...]}` (PCST 출력에서 변환)
- `db_id`: DB 경로 조회용 (value retrieval에서 필요)
- `tier2_pool`, `gat_scores`, `metadata`: F3용 확장 signature (기존 filter는 무시)
- 반환: `final_nodes` (list of "table.column" or "table"), `status`, `reasoning`

## 성능 분석 (HISTORY §6-16, §9)

### 2×2×2 baseline 맥락
- Filter 없음: 어떤 조합도 P<0.40
- XiYan 추가 시 Precision +0.40~0.45 급등 (0.35→0.79)

### a05 Agentic Filter Ablation (anchor: a03_17, F1=0.6940)
| Filter | Recall | Precision | F1 | vs anchor |
|--------|--------|-----------|----|-----------|
| XiYan (anchor) | 0.6761 | 0.7128 | 0.6940 | — |
| AdaptiveMultiAgent (a05_01) | 0.3770 | 0.6276 | 0.4713 | **−22.3%p** 실패 |
| **ReflectionFilter 1iter (a05_02)** | **0.7320** | 0.6833 | **0.7068** | **+1.3%p (신기록)** |
| VerifierFilter (a05_04) | 0.7093 | 0.6676 | 0.6878 | −0.6%p |
| ReflectionFilter 3iter (a05_03) | 진행 중 | | | |
| Tiered Bidirectional / AdaptiveDepth / Retry | 대기 | | | |

### 핵심 관찰
- **ReflectionFilter가 recall 천장 돌파**: XiYan 대비 R +0.06 (0.6761→0.7320). Critique이 원 subgraph 밖 노드 재도입을 허용하는 구조가 핵심.
- **VerifierFilter < ReflectionFilter**: F1 0.6878 vs 0.7068. Generate-then-check 분리 구조가 통합 critique-revise보다 약함.
- **AdaptiveMultiAgent 실패**: 3-agent consensus 과보수적 교집합화 + JSON parsing 실패로 R 파괴.
- **Direct variant의 recall 천장** (R≈0.68~0.73) 은 Selector/Extractor 단계가 결정; Filter가 Reflection restore path로 일부 돌파 가능.

### Prune-Only 한계 (이전 분석, Ensemble+Adaptive 파이프라인)
- **Filter✗ 노드 특성**: TP와 score 차이가 크지 않음 (TP mean 0.7108 vs Filter✗ mean 0.6394, 0.07 차이)
- **Filter✗는 제거되지 말았어야 할 고신뢰 gold 포함**
- XiYan recall 손실 ~0.15 — agentic restore path 필요의 근거

## 사용하는 LLM
- **Qwen3-Coder-30B-A3B-Instruct-FP8** (현재 기본, vLLM on GPU 2+3, localhost:8000)
- **GPT-4o-mini** (a05_11/12 backbone 민감도 검증용)
- API endpoint 호출, prompt caching 활용

## a05 계열 실험 전체 (HISTORY §6-16)
configs/experiments/abl/a05_filter_agentic/ 하위 12개, anchor는 `abl_a03_17` (SuperNode-Direct + Fixed PCST + XiYan).
- 완료: a05_01, a05_02, a05_04
- 진행: a05_03
- 대기: a05_05~12 (Tiered Bidirectional, AdaptiveDepth, Stacked, Retry, GPT-4o-mini)

## 진행중/추후 고려
- **F3 Tiered Bidirectional 검증**: Tier-1(PCST subgraph) + Tier-2(selector-positive but PCST-rejected) 분리 prompt
- **F4 Uncertainty gating**: GAT confidence 기반 XiYan/Reflection/Bidirectional 분기
- **F5 Extraction retry**: Unanswerable verdict → Extractor cost 완화 재호출 (pipeline reverse loop)
- **Reflection + Enriched/Triplet Builder 결합**: 각각의 최고점 시너지 (F1 0.7068 × P 0.81)
- Ensemble Filter (Reflection + Verifier)
- LLM 교체 영향 분리 관측 (Qwen vs GPT-4o-mini)

## 프롬프트 엔지니어링 주의
- Schema 정보가 너무 많으면 LLM이 과잉 제거 경향
- Example values는 정확해야 함 (hallucination 방지)
- `reasoning` 필드는 LLM이 직접 설명하는 근거 — 디버깅에 중요
- JSON parsing 실패 시 기본 fallback이 Unanswerable이면 recall 파괴 (a05_01 교훈) — XiYan 결과 유지로 폴백 권장

## 분석 산출물
- [/home/hyeonjin/thesis_refactored/notebooks/analysis_results/full_ablation_2x2x2.md](/home/hyeonjin/thesis_refactored/notebooks/analysis_results/full_ablation_2x2x2.md) — Filter 유무 비교
- [/home/hyeonjin/thesis_refactored/notebooks/analysis_results/per_stage_failure_analysis.md](/home/hyeonjin/thesis_refactored/notebooks/analysis_results/per_stage_failure_analysis.md) — Filter✗ 분석
