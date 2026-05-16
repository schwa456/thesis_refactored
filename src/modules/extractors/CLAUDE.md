# Subgraph Extractor (PCST) 모듈

> **루트 CLAUDE.md 참조 (읽기 전용)**: 실험 실행, 디렉토리 구조, 문서화 규칙 등 프로젝트 전역 규칙은 [/home/hyeonjin/thesis_refactored/CLAUDE.md](/home/hyeonjin/thesis_refactored/CLAUDE.md)를 반드시 먼저 읽고 따른다. 단, 루트 CLAUDE.md는 수정하지 않는다 — 수정이 필요하면 루트 세션에 요청한다.

## 이 세션의 집중 주제
**PCST 기반 subgraph 추출**. Prize 설계, Cost 설계, threshold 정책, 그래프 알고리즘 트레이드오프.
Builder/Selector/Filter 내부는 가급적 언급하지 않고,
**"주어진 node_scores와 그래프 구조에서 subgraph를 어떻게 뽑느냐"** 관점만 유지.

## Prize-Cost 스케일 불일치 (진단 완료)
### 분석 결과
- **Prize**: `max(score − threshold, 0)` — positive prize 중앙값 ~0.086
- **Cost** (AdaptivePCST 기본값):
  - `belongs_to = 0.01` → prize 중앙값의 0.12배 → **제약이 안 됨** (90.9% 노드가 감당 가능)
  - `fk = 0.05` → prize 중앙값의 0.58배 → 간신히 감당
  - `macro = 0.5` → prize 중앙값의 5.82배 → **사실상 불가능** (0.6%만 감당)
- **결과**: 3-table JOIN 쿼리에서 테이블 간 연결 실패 빈번

### 이 문제에 대한 대응 (모두 실험 완료)
- **Idea 2 (Product Cost)**: `cost = bt/fk/macro_weight × score_u × score_v` — `ProductCostPCSTExtractor`
- **Idea 4 (Component-Aware)**: component별 독립 threshold — `ComponentAwareProductCostPCSTExtractor`
- **Idea 2+4 결합**: I24b F1=0.7063
- **BO (§6-6)**: `ScoreDrivenPCSTExtractor` (bt=1.955/fk=2.779/macro=3.439/ε=0.009), BO2 F1=0.6751
- **Idea 3 Steiner backbone**: seed 간 Steiner tree 2-근사 → PCST expansion

## Extractor 계층 (HISTORY §10-3)
[pcst.py](pcst.py) 및 관련 파일:
- **PCSTExtractor** (Basic) — `node_threshold=0.1`, fixed cost
- **AdaptivePCSTExtractor** — per-query P80 threshold, prize 개수 clamp (3~25)
- **DynamicPCSTExtractor** — Hub discount (`C / (1 + γ·log(1+deg_v))`)
- **UncertaintyPCSTExtractor** — Prize에 power scaling (`prize^α`)
- **DynamicUncertaintyPCSTExtractor** — Dynamic + Uncertainty 결합
- **GATAwarePCSTExtractor** — GAT score를 prize로 직접 사용 (legacy, scale mismatch로 실패)
- **PPRPCSTExtractor** — PPR로 prize 확산
- **EdgePrizePCSTExtractor** — Triplet edge embedding 기반 edge prize (`topk_e=5`)
- **ProductCostPCSTExtractor** — Score-product edge cost (Idea 2)
- **ComponentAwareProductCostPCSTExtractor** — Idea 2+4 결합
- **ScoreDrivenPCSTExtractor** — BO로 튜닝된 cost weights
- **SteinerBackbonePCSTExtractor** — `backbone_bonus=0.5` bridge 강제 포함
- **MSTExtractor** — Metric closure 기반 Steiner 2-근사 (Kou-Markowsky-Berman 1981, 단독 드묾). [2026-04-27] `seed_mode ∈ {"topk","threshold"}` 추가:
  - `topk` (default, backward compatible) — 외부 전달 `seed_nodes` (Selector top-k) 사용
  - `threshold` — `node_scores > score_threshold` (default 0.1) 인 노드를 자체 산출 → Basic PCST 와 동일 candidate pool. 발표 §14.3 narrative 정확성 (top-k 한정 vs score-threshold widening 의 R 영향 isolate) 위해 도입.
- **MSTKruskalExtractor** [2026-04-27] — "진짜" MST. `node_scores > score_threshold` 인 노드의 induced subgraph 위 `networkx.minimum_spanning_tree` (Kruskal default). disconnected 시 minimum spanning forest 자동 반환. Steiner Tree 와 차이: terminal subset connecting + Steiner point 추가가 아니라 induced subgraph 의 모든 vertex spanning + Steiner point 없음. 발표 narrative 의 "MSTExtractor 명명 → 실제로 Steiner 2-approx" 정정 + 사용자 의도된 진짜 MST 측정용. 관련 configs: `abl/.../plain_ens_a05_mst_kruskal_*.yaml`.
- **MSTPCSTUnionExtractor** [2026-04-27] — `MSTKruskalExtractor` (score_threshold=0.1) ∪ `PCSTExtractor` (node_threshold=0.1, default cost) union. set 합집합으로 노드/엣지 통합 후 sorted canonical tuple form 으로 반환. `last_info` 에 `mst_node_count`, `pcst_node_count`, `intersection_count`, `mst_only_count`, `pcst_only_count` 기록 → analyzer 가 union 의 추가 노드 비율 정량 가능. 의도: MST Kruskal anchor (F1=0.8642) 의 R 상한 검증 + Filter 의 union 처리 능력 정량 (시나리오 A: F1>anchor → multi-extractor union 우세, B: ≈anchor → MST 만으로 충분, C: <anchor → over-include noise). 관련 configs: `abl/.../plain_ens_a05_mst_pcst_union_*.yaml`.
  - **[2026-05-16 Phase 4.1]** `seed_selection_mode ∈ {"threshold","integrated_score"}` + `alpha ∈ [0,1]` 추가. `"threshold"` (default, backward compat) = 위 기본 동작. `"integrated_score"` 모드는 `s_integrated(v) = α·𝟙[v ∈ Selector_TopK] + (1-α)·𝟙[s_v ≥ score_threshold]` 로 양쪽 sub-extractor 의 새 `node_scores` 를 만들고 sub-extractor threshold 0.0 으로 override 하여 전달. effective seed = `s_integrated > 0` (= TopK ∪ threshold-pass). α=0.0 → threshold only (≈ c01_01 anchor), α=1.0 → Selector TopK only, middle α → union with α-weighted PCST prize. `seed_nodes` 는 `pipeline/schema_linking.py:197` 가 Selector top-K 를 그대로 전달. `last_info` 에 `seed_selection_mode`, `alpha`, `integrated_topk_only`, `integrated_threshold_only`, `integrated_intersection`, `integrated_positive_total` 추가. 관련 configs: `abl/c04_phase4_alpha_sweep/p4_01..06`. 근거: planning/DECISIONS.md 2026-05-16 §3 Phase 4.1.
- **TopK / None** — utility
- **HybridFKPriorPCSTExtractor** [E-III, 2026-04-20] — Adaptive PCST + FK topology prior. Top-k anchor 간 FK shortest-path 위 `table_to_table` edge 비용 `× fk_path_discount` (기본 0.3), 선택적 bridge (non-anchor) table prize boost (`bridge_bonus × max_prize`). `graph_data['fk_shortest_paths']` (Builder B-III 제공) 우선, 없으면 networkx on-the-fly fallback. 의도: PCST 의 prize-cost 스케일 불일치를 FK topology 기반으로 보정 (3-table JOIN bridge 포함율 개선 기대). 관련 configs: `abl/a06_ext_fkprior/a06_01~04`.
- **PathfindingEnsemblePCSTExtractor** [E-II, 2026-04-20] — Base AdaptivePCST ⊕ Steiner 2-approx pathfinder. `mode ∈ {union, 2pass, intersection}` — union=PCST∪path, 2pass=path 노드 score boost 후 PCST 재실행, intersection=보수적. `k_anchors` (기본 5) 테이블 anchor 는 `node_scores[0:num_t]` top-k. MSTExtractor 의 `steiner_tree_2approx` 재활용. 의도: PCST 가 놓치는 bridge 노드를 pathfinder 로 결정론적 보완. 관련 configs: `abl/a07_ext_path_ensemble/a07_01~05`.
- **LouvainCommunityPCSTExtractor** [E-I, 2026-04-20] — Table FK subgraph 위 Louvain community detection (`networkx.algorithms.community.louvain_communities`) → top-M community 의 tables + columns + FKs 만 유지한 masked graph 위에 Base PCST. `adaptive_coverage=True` 시 고score 테이블이 전부 덮일 때까지 community 자동 확장. ComponentAware 와의 차이: CA 는 PCST **이후** component, 여기는 **입력 그래프** community. 관련 configs: `abl/a08_ext_louvain/a08_01~03`.

## 인터페이스 계약
`extract(graph_data, node_scores, seed_nodes=None, **kwargs)` → `(List[int], List[Tuple[int,int]])`
- `graph_data`: Builder가 넘긴 metadata (`edges`, `edge_types` 포함)
- `node_scores`: 모든 노드의 score (Selector가 제공)
- `seed_nodes`: **대부분의 구현에서 무시됨**. 예외: `SteinerBackbonePCSTExtractor`는 `seed_nodes`를 실제로 사용

## PCST 라이브러리
- `pcst_fast` 사용
- **주의**: numpy 2.x 비호환 버그 → `numpy<2` (1.26.4) 사용
- `pcst_fast.pcst_fast(edges, prizes, costs, root=-1, num_clusters=1, 'gw', verbose=0)`
- Prize가 모두 0이면 빈 subgraph 반환

## 평가 지표 및 주요 수치 (HISTORY §4, §6)
- **Basic PCST + XiYan > Adaptive PCST + XiYan (F1 기준)**: 0.7863 vs 0.6987
  - XiYan이 pruning 잘하므로 PCST는 넓게 포함시키는 게 유리
- **a03_17 (SuperNode-Direct + Fixed PCST + XiYan)**: F1=0.6940 (Direct variant 최고)
  - Fixed PCST가 SuperNode의 강한 recall(0.7982)을 손실 없이 유지
- **Fixed PCST > SteinerBackbone** 일관 (+XiYan): a03_14>a03_15, a03_17>a03_18
- **Fixed PCST > AdaptivePCST** 일관 (+XiYan): a03_14>a03_08, a03_17>a03_12
- **PCST✗** = gold인데 PCST 출력에 없음 (하류 Filter로 못 넘어감)
  - Ensemble+Adaptive+NoFilter (#7): PCST✗ 5,358 / 10,252 gold (52%)
  - Basic PCST: PCST✗ 노드 중 threshold 이상 비율 86–99% (점수 충분, 그래프 비용 때문에 탈락)
  - Adaptive PCST: 37–39% (점수 자체가 부족)

## 하이퍼파라미터 맵
- AdaptivePCST: `percentile=80.0`, `min/max_prize_nodes=3/25`, `node_threshold=0.0`
- 공통 cost: `base_cost=0.05`, `belongs_to_cost=0.01`, `fk_cost=0.05`, `macro_cost=0.5`
- Basic PCST: `node_threshold=0.1`
- Steiner: `backbone_bonus=0.5`
- MSTExtractor: `seed_mode='topk'` (또는 `'threshold'`), `score_threshold=0.1` (threshold mode 시)
- MSTKruskalExtractor: `score_threshold=0.1`
- MSTPCSTUnionExtractor: `score_threshold=0.1` (양쪽 통일), PCST cost params 는 PCSTExtractor default (`base_cost=1.0, belongs_to_cost=0.01, fk_cost=0.05, macro_cost=0.5, hub_discount=0.2`). [Phase 4.1] `seed_selection_mode='threshold'` (default) | `'integrated_score'`, `alpha=1.0` (default 0~1)
- ProductCost: `bt_weight=0.1, fk_weight=0.2, macro_weight=0.5, min_cost=0.0001, percentile=80`
- ScoreDriven (BO2): `bt=1.955, fk=2.779, macro=3.439, ε=0.009`
- HybridFKPrior (E-III): `k_anchors=5, fk_path_discount=0.3, bridge_bonus=0.0, max_anchor_pairs=20`
- PathfindingEnsemble (E-II): `mode='union', k_anchors=5, prize_boost=0.2`
- LouvainCommunity (E-I): `resolution=1.0, top_m_communities=2, min_communities_to_prune=2, adaptive_coverage=False, seed=42`

## 분석 산출물
- [/home/hyeonjin/thesis_refactored/notebooks/analysis_results/per_stage_failure_analysis.md](/home/hyeonjin/thesis_refactored/notebooks/analysis_results/per_stage_failure_analysis.md)

## 추후 고려할 축
- Budget constrained PCST (k-MST 제약)
- Multi-objective PCST (recall/precision 동시 최적화)
- Filter와의 역피드백 loop (F5 방향: Unanswerable verdict 시 cost 완화 재호출)
- Prize에 uncertainty/confidence scaling
- Component 분해와 agentic filter의 결합

**닫힌 주제**: 방안 A(Score-driven dynamic cost), 방안 B(BO)는 둘 다 완료. 더 이상 꺼내지 말 것.
