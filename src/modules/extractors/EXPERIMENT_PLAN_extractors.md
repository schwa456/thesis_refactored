# Extractor (PCST) 모듈 실험 계획 — 9 아키텍처 제안 중 Extractor 관련 축

> **⚑ 먼저 루트 계획을 읽을 것**: [/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md](/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md) — 전 모듈 통합 로드맵, Cross-Module Dependency, 통합 실험(int_01~08), 우선순위 Phase A~E, 논문 매핑이 거기에 있다. **루트 PLAN은 수정하지 않는다** — 수정이 필요하면 루트 세션에 요청.
> **이 파일의 역할**: 루트 PLAN에서 Extractor에 할당된 3축(E-I/E-II/E-III)의 **모듈 내부 구현 상세**만 담는다.
>
> **현재 진입점**: `AdaptivePCSTExtractor` (baseline), `PCSTExtractor` (fixed θ=0.1), `ScoreDrivenPCSTExtractor` / `ProductCostPCSTExtractor` (방안 A/B 완료 — 더 이상 꺼내지 않음).
> **핵심 과제**: PCST 단독의 한계 (3-table JOIN 연결 실패, macro cost 0.5가 prize 중앙값 대비 6배).
> **선결 의존성**: E-III는 Builder B-III의 FK reachability matrix에 의존. 루트 PLAN Phase A 완료 전에는 E-III 블록.

---

## 이 모듈이 받아야 할 3가지 제안

| # | 이름 | Extractor의 역할 | 우선순위 |
|---|------|-----------------|---------|
| E-I | **Louvain Community PCST** (원안 #4) | Community detection 선행 → component별 독립 PCST | 중 |
| E-II | **Pathfinding Extractor + PCST Ensemble** (원안 #5) | Top-k anchor 간 shortest/Steiner path → PCST와 합집합 | 상 |
| E-III | **Symbolic-Neural Layer 2: Hybrid PCST with FK prior** (원안 #9 Layer 2) | FK reachability로 edge cost 동적 조정 + bridge 강제 포함 | 상 |

---

## E-I. Louvain Community PCST

### 동기
- 현재 `AdaptivePCSTExtractor` 는 전역 P80 threshold로 모든 노드 동일 기준 — large DB에서 의미 군집(domain cluster)을 무시.
- 이미 `ComponentAwareProductCostPCSTExtractor` 가 connected-component 단위 독립 threshold를 구현 — **그러나 "component"는 PCST 결과의 후처리**, 제안 #4는 **입력 그래프의 community detection**.
- Louvain 으로 테이블을 **사전 cluster** → 쿼리가 target하는 community만 PCST 수행 → 불필요 macro edge 가지치기.

### 설계 요소
- **Stage 1**: Builder의 `table_to_table` + `fk` 그래프에서 Louvain community detection.
- **Stage 2**: 쿼리-각 community 유사도 scoring (community member의 평균 score).
- **Stage 3**: Top-M community에 속한 노드만 PCST에 전달, 나머지는 제외.
- **Stage 4**: 선택된 community 간 macro edge cost를 낮춤 (cross-community 연결 장려).

### 인터페이스
```python
class LouvainCommunityPCSTExtractor(BasePCSTExtractor):
    def __init__(self, resolution=1.0, top_m_communities=2, ...):
        ...
    def extract(self, graph_data, node_scores, seed_nodes=None, **kwargs):
        communities = louvain(graph_data)
        selected = top_m_communities_by_score(communities, node_scores)
        masked_graph = mask_outside_communities(graph_data, selected)
        return run_pcst(masked_graph, node_scores, ...)
```

### 의존성 / 주의
- `networkx` 또는 `python-louvain` 추가 의존.
- 단일 쿼리가 multi-community에 걸치는 경우 failure mode: **M 자동 확장** 또는 coverage check.
- **이미 closed된 방안 A/B와 혼동 금지** — Louvain은 "community 분할 선행" 축으로 별개.

### 예상 실험
| 실험 ID | 파라미터 | 비고 |
|---------|---------|-----|
| `abl_ext_louvain_01` | resolution=1.0, top_m=2 | Pilot |
| `abl_ext_louvain_02` | resolution=0.5 (더 큰 community) | |
| `abl_ext_louvain_03` | top_m=3, adaptive | Coverage 기반 자동 확장 |

### 검증
- 3-table JOIN 쿼리에서 필요한 3개 테이블이 모두 선택 community에 포함되는지.
- Large DB(컬럼 50+)에서 특히 latency/quality 개선.

---

## E-II. Pathfinding Extractor + PCST Ensemble (★ 핵심 신규)

### 동기
- PCST는 globally-optimal subgraph 찾지만 cost/prize balance가 맞지 않으면 **bridge 노드 빠짐**.
- Pathfinding (Steiner tree / shortest path)은 결정론적으로 "anchor 간 연결"을 보장.
- 둘의 **합집합** 이 single PCST보다 robust: PCST의 semantic coverage + Pathfinder의 connectivity.

### 설계 요소
- **Anchor 식별**: Selector score top-k (기본 k=5, 테이블 위주).
- **Pathfinding**:
  - `SteinerBackbonePCSTExtractor` 에 이미 구현된 metric-closure 2-approx 재활용.
  - 또는 all-pairs shortest path (small graph).
- **Ensemble 규칙**:
  - Union: `nodes = pcst_nodes ∪ steiner_path_nodes`
  - Weighted: PCST 결과에 path nodes prize boost 후 재실행 (2-pass)
  - Intersection ∪ Anchor-only (conservative)

### 인터페이스
```python
class PathfindingEnsemblePCSTExtractor(BasePCSTExtractor):
    def __init__(self, pcst_base_cls=AdaptivePCSTExtractor, k_anchors=5, mode='union'):
        self.pcst = pcst_base_cls()
        self.pathfinder = MSTExtractor()  # 기존 metric closure 재사용
    def extract(self, graph_data, node_scores, seed_nodes=None, **kwargs):
        anchors = top_k_tables_by_score(node_scores, metadata, k=self.k_anchors)
        pcst_nodes, pcst_edges = self.pcst.extract(graph_data, node_scores, anchors)
        path_nodes, path_edges = self.pathfinder.extract(graph_data, node_scores, anchors)
        if self.mode == 'union':
            return list(set(pcst_nodes) | set(path_nodes)), list(set(pcst_edges) | set(path_edges))
        elif self.mode == '2pass':
            # boost path nodes then rerun PCST
            boosted_scores = boost(node_scores, path_nodes, gamma=0.2)
            return self.pcst.extract(graph_data, boosted_scores, anchors)
```

### 의존성 / 주의
- **기존 재활용 가능**: `MSTExtractor` (metric closure Steiner 2-approx), `SteinerBackbonePCSTExtractor` (`backbone_bonus`).
- Anchor 품질이 결정 — Selector의 top-k table 정확도 낮으면 오답 path.
- Union mode는 precision 하락 가능 (대량 노드 유입).

### 예상 실험
| 실험 ID | mode | k_anchors | 비고 |
|---------|------|-----------|-----|
| `abl_ext_path_01` | union | 5 | Baseline |
| `abl_ext_path_02` | 2pass | 5 | Boost-rerun |
| `abl_ext_path_03` | union | 3 | Anchor 축소 |
| `abl_ext_path_04` | 2pass | 10 | Anchor 확대 |

### 검증
- Bridge table 인식율 (3-table JOIN 쿼리에서 중간 테이블 포함 %)
- PCST✗ gold 중 path 경로로 복구된 비율.
- Precision 하락 폭이 Filter로 회복 가능한지 (+XiYan에서 평가).

### 학술 기여
- "Deterministic path-based completion of prize-cost optimization" — PCST 단독 한계의 명시적 보완.
- AdaptivePCST + XiYan F1 0.6987 → Path ensemble + XiYan으로 유의미 개선 기대.

---

## E-III. Symbolic-Neural Layer 2 — Hybrid PCST with FK Prior

### 동기
- 제안 #9 Layer 2 = **Layer 1의 symbolic FK 정보를 PCST cost 조정에 직접 주입**.
- Builder가 제공한 `fk_reachability`, `fk_shortest_paths`, `fk_components` 를 **edge cost를 동적으로 낮추거나 bridge를 강제 포함** 하는 데 사용.
- 이미 `ScoreDrivenPCSTExtractor` (BO tuned) 가 cost를 score와 결합 — 여기서는 **FK graph의 구조적 신호** 를 추가.

### 설계 요소
- **Cost adjustment**:
  - FK edge for (anchor_i → anchor_j shortest path): cost × 0.3 (path 상 edge 할인)
  - Cross-community FK: cost × 0.5 (bridge 장려)
  - Non-path FK: cost × 1.0 (기본값)
- **Bridge force-include**: anchor 간 shortest FK path 상 node를 `backbone_bonus` 로 강제.
- **Integration with Path Ensemble (E-II)**: path_nodes를 prize boost.

### 인터페이스
```python
class HybridFKPriorPCSTExtractor(BasePCSTExtractor):
    def __init__(self, fk_path_discount=0.3, bridge_bonus=0.5, use_reachability=True):
        ...
    def extract(self, graph_data, node_scores, seed_nodes=None, **kwargs):
        metadata = kwargs["metadata"]
        fk_paths = metadata["fk_shortest_paths"]
        anchors = top_k_tables_by_score(node_scores, metadata, k=5)
        # 1. path edges 식별
        path_edge_ids = set()
        for i, j in combinations(anchors, 2):
            for edge_id in fk_paths.get((i, j), []):
                path_edge_ids.add(edge_id)
        # 2. cost 조정
        costs = compute_costs_with_discount(graph_data, path_edge_ids, self.fk_path_discount)
        # 3. bridge node prize 가산 (optional)
        prizes = compute_prizes_with_bridge_bonus(node_scores, path_edge_ids, self.bridge_bonus)
        return pcst_fast(edges, prizes, costs, ...)
```

### 의존성 / 주의
- **Builder B-III (FK reachability)** 선결.
- **방안 A/B와 구분**: A/B는 cost를 score-driven으로 조정, Layer 2는 **FK graph topology 기반** — 직교 축.
- FK path가 공집합인 경우 (disconnected DB) fallback 필요.

### 예상 실험
| 실험 ID | 파라미터 | 비고 |
|---------|---------|-----|
| `abl_ext_fkprior_01` | discount=0.3, no bridge bonus | 순수 cost 조정 |
| `abl_ext_fkprior_02` | discount=0.3, bridge_bonus=0.5 | 결합 |
| `abl_ext_fkprior_03` | discount=0.5, bridge_bonus=1.0 | Aggressive |

### 검증
- 3-table JOIN 쿼리에서 FK bridge 포함 비율 (현재 대비).
- Macro edge 사용량 (cost가 크게 쿠션되는지).
- Precision 하락 없이 Recall 상승해야 함.

### 학술 기여
- "Cost-adjustment via symbolic FK topology enables prize-cost scale mismatch resolution without score hacking."
- Neurosymbolic 프레이밍의 Extractor 기여 축.

---

## 통합 실험 로드맵 (Extractor 관점)

| Phase | 실험 | 의존 | 비고 |
|-------|------|------|-----|
| E1 | `abl_ext_fkprior_*` | Builder B-III | 가장 저비용, 먼저 실행 |
| E2 | `abl_ext_path_*` | 없음 (MSTExtractor 재활용) | 구현 빠름, 기대 효과 큼 |
| E3 | `abl_ext_louvain_*` | networkx 의존 | Large DB 특화 |
| E4 | `abl_ext_pathfkprior` | B-III + E-II 통합 | 상한 탐색 |

**고정 조건**: Selector는 `EnsembleSelector` 또는 `DirectGATSelector(SuperNode)` 기본. Filter는 XiYan 기본 (변경은 filter 세션 담당).

## 변경될 파일

| 파일 | 변경 |
|------|------|
| [pcst.py](pcst.py) | `LouvainCommunityPCSTExtractor`, `PathfindingEnsemblePCSTExtractor`, `HybridFKPriorPCSTExtractor` 추가 |
| [mst.py](mst.py) | 기존 `MSTExtractor` 재활용 (변경 없음) |
| [baseline.py](baseline.py) | — |
| `src/pipeline/schema_linking.py` | 새 extractor 분기 |
| `configs/experiments/abl/ext_*/` | 11+ yaml |

## 인터페이스 계약 (유지)
- `extract(graph_data, node_scores, seed_nodes=None, **kwargs)` → `(List[int], List[Tuple[int,int]])`
- `**kwargs` 로 `metadata` 전달받아 FK reachability 등 consult.
- Prize=0 처리, numpy<2 호환 유지.

## 닫힌 주제 재확인
- **방안 A (Score-driven cost)**: `ScoreDrivenPCSTExtractor` (BO2 tuned). **새 실험에서 기본값 인용은 가능, 변형 제안 금지.**
- **방안 B (BO)**: 동일. F1=0.6751.
- **Idea 2 (Product Cost)**, **Idea 4 (Component-Aware)**: 모두 완료. I24b F1=0.7063.
- 위 3개와 **교차 실험 (기존 × 신규 축)** 은 가능하나 기존 단독 변형은 재실행하지 않음.

## Phase 4.1 — Selector + Extractor 통합 점수 mode (2026-05-16, 학술 Agent Plan §4.1)

### 동기
- 현 anchor (c01_01 `MSTPCSTUnionExtractor`) 의 seed 선택은 `node_scores ≥ score_threshold(=0.1)` 단독 — Selector top-K (예: 20) 결과는 score-threshold 와 무관하게 폐기.
- Selector 와 Extractor 의 **co-design integration evidence** 가 paper §3.5 axis #5/#6/#7 narrative 보강 필요.
- closure 정합 (Wave 5 closure final, R 갱신 시도 중단) 위에서, R 상승 lever 보다는 **mechanism contribution 정량** 축으로 재정의됨.

### 신규 mode 정의
[mst_pcst_union.py:MSTPCSTUnionExtractor](mst_pcst_union.py) 에 `seed_selection_mode` 파라미터 추가:
- `"threshold"` (default, **backward compat**): 기존 동작 (양쪽 sub-extractor 가 `node_scores ≥ score_threshold` 자체 적용).
- `"integrated_score"` (신규): 통합 점수 mode.
    ```
    s_integrated(v) = α · 𝟙[v ∈ Selector_TopK] + (1-α) · 𝟙[node_scores[v] ≥ score_threshold]
    ```
    - s_integrated ∈ {0, α, 1-α, 1} 4-valued discrete blend.
    - 양쪽 sub-extractor (MSTKruskal, PCST) 의 threshold 를 일시 0.0 으로 override 한 뒤, `s_integrated` 를 `node_scores` 자리에 전달 → s_integrated > 0 (= TopK ∪ threshold-pass) 인 노드만 effective seed 진입.
    - PCST prize 는 s_integrated 값 자체 (max(s − 0, 0) = s) → α 가 노드별 prize 가중치로 직접 반영됨.

### α grid 의 의미
| α | s_integrated 식 | TopK-only prize | threshold-only prize | intersection prize | 효과 |
|---|---|---:|---:|---:|---|
| 0.0 | 𝟙[s ≥ θ] | 0 | 1.0 | 1.0 | threshold only (≈ anchor c01_01) |
| 0.2 | 0.2·𝟙[TopK] + 0.8·𝟙[s≥θ] | 0.2 | 0.8 | 1.0 | threshold-leaning |
| 0.4 | 0.4·𝟙[TopK] + 0.6·𝟙[s≥θ] | 0.4 | 0.6 | 1.0 | threshold-leaning |
| 0.6 | 0.6·𝟙[TopK] + 0.4·𝟙[s≥θ] | 0.6 | 0.4 | 1.0 | TopK-leaning |
| 0.8 | 0.8·𝟙[TopK] + 0.2·𝟙[s≥θ] | 0.8 | 0.2 | 1.0 | TopK-leaning |
| 1.0 | 𝟙[TopK] | 1.0 | 0 | 1.0 | TopK only |

> α=0.2~0.8 은 TopK ∪ threshold-pass union 으로 effective seed set 동일하지만 PCST prize 분포가 달라 → bridge 노드 포함 양상 변화 측정.

### 인터페이스 계약 (변경 없음)
`extract(graph_data, node_scores, seed_nodes=None, **kwargs)`
- `seed_nodes` 는 [pipeline/schema_linking.py:197](../../pipeline/schema_linking.py) 에서 Selector top-K 가 그대로 전달됨 (기존 contract 활용, 추가 인터페이스 변경 없음).
- `seed_selection_mode="threshold"` (default) 인 경우 `seed_nodes` 무시 — 기존 모든 config 가 변경 없이 동작.

### 실험 설정
| 실험 ID | α | 고정 (θ, K) | 비고 |
|---------|---|------------|-----|
| `p4_01_alpha_0.0` | 0.0 | (0.1, 20) | threshold only (≈ c01_01) |
| `p4_02_alpha_0.2` | 0.2 | (0.1, 20) | threshold-leaning blend |
| `p4_03_alpha_0.4` | 0.4 | (0.1, 20) | threshold-leaning blend |
| `p4_04_alpha_0.6` | 0.6 | (0.1, 20) | TopK-leaning blend |
| `p4_05_alpha_0.8` | 0.8 | (0.1, 20) | TopK-leaning blend |
| `p4_06_alpha_1.0` | 1.0 | (0.1, 20) | TopK only |

- Anchor stack: QCondGAT + MSTPCSTUnion + XiYanFilter (GLM) + LLMSQLGenerator (GLM).
- Configs: [configs/experiments/abl/c04_phase4_alpha_sweep/](../../../configs/experiments/abl/c04_phase4_alpha_sweep/).
- 측정: TCR_new + Filter Pruning Ratio + R/P/F1/EX + integrated_* telemetry (`integrated_topk_only`, `integrated_threshold_only`, `integrated_intersection`, `integrated_positive_total`) — `extractor_info` 에 자동 기록.

### 검증 포인트
- α=0.0 의 R/P/F1 이 c01_01 (R=0.8654 P=0.8675 F1=0.8664) 와 sub-noise (|Δ| < 0.005) 정합 — backward-compat 의 deterministic 확인.
- α=1.0 의 effective seed = Selector top-20 만 → R 손실 (threshold-pass 의 추가 노드 폐기) 정도 정량.
- α 중간값에서 prize 가중치 차이가 PCST bridge 포함 양상에 미치는 효과 — `integrated_positive_total` 동일하나 `extractor_num_selected_nodes` / Filter Pruning Ratio 차이 측정.

### 산출
- [notebooks/analysis_results/phase4_1_integrated_alpha_sweep_2026-05-XX.md](../../../notebooks/analysis_results/) (root + analyzer 단계).
- paper §3.5 axis #5/#6/#7 narrative 보강 candidate.

## 검증 방법 (모듈 내)
- **Recall/Precision/F1 4소수점**.
- **3-table JOIN 특화 분석**: 이 쿼리군에서 bridge 포함 비율, PCST✗ 복구율.
- **Latency**: Louvain/Pathfinder는 쿼리당 +수십 ms 허용 범위.
