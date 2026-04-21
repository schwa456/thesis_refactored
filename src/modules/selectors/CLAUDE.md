# Seed Selector 모듈

> **루트 CLAUDE.md 참조 (읽기 전용)**: 실험 실행, 디렉토리 구조, 문서화 규칙 등 프로젝트 전역 규칙은 [/home/hyeonjin/thesis_refactored/CLAUDE.md](/home/hyeonjin/thesis_refactored/CLAUDE.md)를 반드시 먼저 읽고 따른다. 단, 루트 CLAUDE.md는 수정하지 않는다 — 수정이 필요하면 루트 세션에 요청한다.

## 이 세션의 집중 주제
**노드 별 score 계산**. Query-Node 유사도의 품질, GAT 구조 반영, score calibration.
Builder/Extractor/Filter 내부는 가급적 언급하지 않고,
**"Selector가 내는 score가 하류에 넘겨질 때 충분한 변별력을 가지는가"** 관점만 유지.

## 핵심 설계 원칙 (착각하지 말 것)
**Selector의 top-k는 PCST에 대한 게이트가 아니다.**
- Selector는 score를 계산하고 `candidates`를 그대로 반환 (즉, 모든 노드가 하류로 감)
- PCST는 전체 node_scores를 받아 prize를 계산함
- 따라서 "Seed✗ = top-k 탈락"이라는 분석은 **잘못된 분류**
- Selector의 실질적 기여는 **score quality 자체**

**예외**: `DirectGATSelector(apply_threshold=True)` + `SteinerBackbonePCSTExtractor` 조합은
binary pass가 seed로 전달되어 `seed_nodes` 기반 확장이 일어남 (HISTORY §6-14).

## 현재 Selector 구현
- **VectorOnlySelector** — Cosine 유사도만
- **EnsembleSelector** ([ensemble_selector.py](ensemble_selector.py)) — 2×2×2 best 구성
  - `score = α·cos(e_Q, e_ni) + (1−α)·MLP(z_Q, z_ni)`
  - α=0.85 (cosine 85% + GAT 15%)
  - GAT projector로부터 compute_scores 호출
- **LinkAlignSelector** — LinkAlign 베이스라인
- **GATClassifierSelector** — GAT 단독 (legacy)
- **DirectGATSelector** — Projector 제거, BCE-only Direct head 기반
  - Mode: `query_conditioned` (Concat) 또는 `query_supernode` (SuperNode)
  - `apply_threshold=True` 시 sigmoid ≥ threshold 노드만 반환 (binary)
  - Direct variant 전용: checkpoint `*_direct.pt` 사용
- **NeurosymbolicL1Selector** ([neurosymbolic_l1_selector.py](neurosymbolic_l1_selector.py)) — **S-V**, EnsembleSelector 확장
  - `boosted = ensemble_scores + λ · reach_mask` (single-hook override, 최소 침습)
  - reach_mask: anchor 테이블에서 FK-reachable 한 table/column/fk_node = 1.0, else 0.0
  - Anchor: question 토큰 (min_len=3) ↔ table/column 이름 snake_case word 매칭
  - 의존: Builder B-III `metadata['fk_reachability']`. 키 없으면 ensemble 로 graceful fallback
  - **Status**: 구현 + smoke 통과 (HISTORY §6-21). End-to-end F1 pending — `abl_sel_ns_l1_01` (λ=0.1), anchor `s03_a02_03_xiyan_filter`
- **FixedTopKSelector / AdaptiveSelector** — utility

## Score 분석 결과 (Ensemble baseline, HISTORY §4)
- **ROC-AUC**: Cosine 0.741 vs Ensemble **0.776** (+0.035)
- **PR-AUC**: Cosine 0.243 vs Ensemble **0.317** (+0.074, 더 큰 개선)
- **Gold-NonGold gap**: Cosine 0.108 vs Ensemble **0.227** (2배 이상)
- **GAT 기여도 (P80 threshold 기준)**:
  - GAT rescued: 544 gold (5.3%) — cosine 탈락했으나 ensemble이 구출
  - GAT hurt: 330 gold (3.2%) — 반대 케이스
  - **순 기여 +214 (2.1%)**
- **구조적 한계**: 38.9% gold가 두 방법 모두에서 threshold 미만 (임베딩 자체 한계)

## Direct Variant 관찰 (HISTORY §6-10 ~ §6-15)
- **Direct(BCE only) < Projector(BCE+InfoNCE)**: Q6/Q7 (F1 0.52~0.53) vs Q4/Q5 (F1 0.59~0.60)
  - DualTowerProjector + InfoNCE가 score ranking 품질에 기여
- **Direct + Binary threshold + Fixed PCST + XiYan (a03_17)**: **F1=0.6940** (Direct variant 신기록)
  - SuperNode-Direct(τ=0.5) + `PCSTExtractor`(fixed) + XiYan
  - Binary gating이 Selector 단계에서는 유효 (P +0.07~0.12), 단 AdaptivePCST와 결합하면 역효과
- **SuperNode > QCond** (Direct, binary R 기준): 0.6261 vs 0.4871

## 학습 체계
- Loss: BCE (pos_weight=100) + InfoNCE (λ=0.5, T=0.07) — Projector
- Direct variant: BCE only
- InfoNCE hard negative mining: top-15
- Architecture: HeteroGATv2 (3 layers, 4 heads) + DualTowerProjector 또는 DirectClassifierHead

## Checkpoint 목록 (HISTORY §5)
| ID | File | Variant | Notes |
|----|------|---------|-------|
| T4 | `best_gat_model.pt` | Projector BCE+InfoNCE | Ensemble/Cosine 기본 |
| T5 | `best_gat_enriched.pt` | Projector + Enriched features | E1 F1=0.7327 |
| T6 | `best_gat_query_conditioned.pt` | Projector + QCond Concat | Q1/Q4 |
| T7 | `best_gat_query_supernode.pt` | Projector + SuperNode | Q2/Q3/Q5 |
| T8 | `best_gat_query_conditioned_direct.pt` | Direct(BCE) QCond | Q6, a03_13~15, val R=0.5914 |
| T9 | `best_gat_query_supernode_direct.pt` | Direct(BCE) SuperNode | Q7, a03_16~18, val R=0.5548 |

## 인터페이스 계약
`select(scores, candidates, question, graph_data, metadata, **kwargs)` → `List[int]` (seeds)
- 대부분 구현에서 candidates를 그대로 반환해도 PCST 동작
- `DirectGATSelector(apply_threshold=True)`는 실제로 필터링된 seed 반환
- `self.latest_scores`에 전체 노드 score 저장 (파이프라인이 이걸 꺼내 PCST에 전달)

## 추후 고려할 축
- α 튜닝 (현재 0.85 고정)
- Ensemble 외 대안 (concat + MLP, cross-attention 등)
- Query-side encoding 다변화 (mean pool vs token-level MaxSim)
- 학습시 curriculum learning (easy → hard negatives)
- FK 노드 supervised training (bridge table 인식 강화)

## PLAN 축별 구현 상태 (루트 EXPERIMENT_PLAN 기준)
- ✅ **S-V (Neurosymbolic L1)**: 구현 완료, pilot `abl_sel_ns_l1_01` 대기 (vLLM 필요). λ sweep 후속.
- ⏳ **S-III (EHGAT)**: Builder B-II smoke OK, 본 세션에서 LineGraph encoder + Selector 본체 구현 필요
- ⏳ **S-II (RFM Encoder)**: Builder B-I smoke OK, api_handler logprobs 확장 및 RFM zero-shot selector 래핑 필요
- ⏳ **S-IV (Extractive LLM)**: api_handler logprobs 선행 작업 이후
- ⏳ **S-I (RL/GRPO)**: 최우선 순위 아님, 마지막

## 분석 산출물
- [/home/hyeonjin/thesis_refactored/notebooks/analysis_results/selector_analysis.md](/home/hyeonjin/thesis_refactored/notebooks/analysis_results/selector_analysis.md)
- [/home/hyeonjin/thesis_refactored/notebooks/analysis_results/ensemble_contribution_analysis.md](/home/hyeonjin/thesis_refactored/notebooks/analysis_results/ensemble_contribution_analysis.md)
