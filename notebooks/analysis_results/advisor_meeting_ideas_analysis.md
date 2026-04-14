# 지도교수 면담 아이디어 분석 — 구현 방법 및 이론적 근거

## Context

예심 통과 후(2026-04-10) 지도교수님과 면담에서 5가지 파이프라인 개선 아이디어가 제안되었다. 현재 파이프라인(GraphBuilder → PLM Encoder → Ensemble Seed Selector → Adaptive PCST → Auto JOIN Keys → XiYan Filter)의 핵심 병목을 해소하고 차별성을 강화하기 위한 방향이다.

**현재 파이프라인의 정량적 문제점** (prior analysis 기반):
- PCST macro_cost(0.5) >> 중간 prize(0.086) → table-table 연결이 사실상 차단됨
- GAT 순 기여: +214 gold nodes(2.1%)로 미미 — α=0.85에서 GAT 비중 15%만 반영
- FK 노드에 학습 label이 없음 (BCE/InfoNCE 모두 미적용, message passing을 통한 간접 학습만)
- 기존 `MSTExtractor`는 seed 간 직접 연결만 확인하는 naive 구현 (metric closure 미사용)
- **GAT가 query를 전혀 보지 못함** — message passing이 schema 구조만으로 수행되어 query-conditioned attention 불가

---

## 아이디어 1: Bridge Table Awareness in GAT Scoring

### 핵심 관찰
> "GAT-MLP(Ensemble Seed Selection)를 거친 결과가 기존 연구와 다르려면, Query에 나타나는 Column 뿐만 아니라 그 사이를 이어주는 Node에도 Seed Selection 단계부터 점수가 높게 나와야 한다."

### 현재 한계 진단

현재 GAT의 bridge table scoring이 약한 근본 원인:

1. **Label 부재**: `bird_dataset.py:67-84`에서 gold label은 `parse_sql_elements(gold_sql)`로 생성하며, 이는 SQL의 모든 Table/Column을 추출한다. **Bridge table은 SQL의 FROM/JOIN에 등장하므로 이미 gold label에 포함된다.** 그러나 "bridge이기 때문에 중요하다"는 구조적 역할이 반영되지 않는다.

2. **FK 노드 무감독**: `train_gat.py:232-233`에서 `hasattr(batch[n_type], 'y')`를 체크하는데, FK 노드에는 `y` attribute이 설정되지 않아 **BCE와 InfoNCE 모두 skip**된다. FK 노드는 GAT message passing의 gradient 흐름으로만 간접 학습된다.

3. **Ensemble 가중치**: α=0.85에서 GAT가 bridge table을 인식하더라도, 최종 ensemble 점수에서 15%만 반영되어 cosine 유사도가 지배적이다.

### 구현 방안

**A. FK 노드 label 추가 (최소 변경)**
- `bird_dataset.py`에서 gold table 쌍을 연결하는 FK 노드에 `y_fk = 1.0` 부여
- `train_gat.py`에서 fk_node도 BCE + InfoNCE loss에 포함

**B. Bridge-aware auxiliary loss (권장)**
- Gold SQL에서 JOIN clause에만 등장하는 테이블을 별도로 식별 (`parse_sql_join_tables()`)
- `y_bridge` label을 추가하고, auxiliary loss term `lambda_bridge * BCE(logits_bridge, y_bridge)` 적용
- Bridge table의 경우 pos_weight를 더 높게 설정하여 희소한 bridge 감독 신호를 강화

**C. α 조정 실험 (즉시 실행 가능)**
- α를 0.85 → 0.70~0.75로 낮춰 GAT 기여도를 증폭
- Config만 변경하면 되므로 가장 빠른 검증 방법

### 이론적 근거

| 참고문헌 | 핵심 주장 | 관련성 |
|---------|---------|-------|
| RAT-SQL (Wang et al., ACL 2020) | Relation-aware self-attention으로 스키마 관계를 명시적으로 모델링하여 Spider에서 +8.7% 향상 | Schema 구조 인코딩이 bridge table 포착에 핵심적 |
| ShadowGNN (Chen et al., NAACL 2021) | Graph projection으로 도메인 독립적 스키마 추상화, join path를 abstract representation으로 처리 | Bridge node의 구조적 역할을 추상화하여 학습 |
| RESDSQL (Li et al., AAAI 2023) | Cross-encoder로 스키마 요소를 ranking하여 관련 항목만 선별적으로 인코딩 | 구조적 관련성까지 포함한 ranking이 schema linking 품질 결정 |
| GATv2 (Brody et al., ICLR 2022) | Dynamic attention이 static GAT보다 표현력이 높으며, neighbor 중요도를 query-conditioned로 학습 가능 | 현재 모델이 사용 중이지만, bridge 학습 신호가 부족하면 attention이 제대로 학습되지 않음 |

### 판단

- **타당성**: 높음 — 현재 GAT의 기여가 2.1%에 불과한 근본 원인을 직접 해소
- **구현 우선순위**: 방안 C(α 조정) → A(FK label) → B(bridge loss) 순으로 점진적 시도
- **주의점**: 방안 B는 SQL parsing의 edge case(implicit JOIN via comma-separated FROM)를 주의해야 하며, bridge table이 전체 학습 샘플 대비 희소하므로 class balancing 필요

---

## 아이디어 2: PCST Cost를 양 Node Prize의 곱으로 설정

### 핵심 관찰
> "PCST 관련 Cost를 양 Node의 Prize의 곱의 형태로 주는 것은 어떤가?"

### 현재 한계 진단

현재 `AdaptivePCSTExtractor`(`pcst.py:264-315`)의 구조:
- Prize: `max(score - P80_threshold, 0)` → 범위 0.0~0.4, 중간값 ~0.086
- Cost: edge type별 고정값 (belongs_to=0.01, fk=0.05, **macro=0.5**)
- **핵심 문제**: macro_cost(0.5) / median_prize(0.086) ≈ 5.8x → table-table 연결이 사실상 불가능

`pcst_fast`의 Goemans-Williamson primal-dual 알고리즘은 "prize 수집 이득 vs. cost 지불 비용"을 비교하여 노드/엣지 포함 여부를 결정한다. Prize와 Cost가 같은 스케일이어야 의미 있는 최적화가 이루어진다.

### 구현 방안

**Product Cost 정의**: edge (u, v)에 대해

```
prizes = max(score - threshold, 0)  # 기존과 동일
cost(u, v) = type_base × (1 - norm_score_u) × (1 - norm_score_v)
```

여기서 `norm_score`는 해당 query의 score를 [0, 1]로 min-max 정규화한 값.

**동작 원리**:
- 양 끝 노드 점수가 모두 높을 때: `(1-high) × (1-high) ≈ 0` → **cost ≈ 0, 연결 장려**
- 한쪽이라도 점수가 낮을 때: `(1-high) × (1-low) ≈ 0.x` → **cost 적당, 신중한 연결**
- 양 끝 모두 낮을 때: `(1-low) × (1-low) ≈ 1.0` → **cost ≈ type_base, 연결 억제**

**type_base 초기값**: belongs_to=0.1, fk=0.2, macro=0.5 (prize 범위와 동일 스케일)

**구현 위치**: `pcst.py`에 `ProductCostPCSTExtractor` 클래스 추가, `AdaptivePCSTExtractor` 상속

```python
@register("extractor", "ProductCostPCSTExtractor")
class ProductCostPCSTExtractor(AdaptivePCSTExtractor):
    def __init__(self, bt_weight=0.1, fk_weight=0.2, macro_weight=0.5,
                 min_cost=1e-4, **kwargs):
        ...

    def extract(self, graph_data, node_scores, seed_nodes=None, **kwargs):
        # 1. Adaptive threshold & prizes (부모 클래스 로직 재사용)
        # 2. Score 정규화 (min-max)
        # 3. Edge별 product cost 계산
        # 4. pcst_fast 실행
```

### 이론적 근거

| 참고문헌 | 핵심 주장 | 관련성 |
|---------|---------|-------|
| Leitner & Raidl (2010), "Prize collecting Steiner trees with node degree dependent costs", *Computers & Operations Research* | PCST에서 edge cost를 노드 속성(degree)의 함수로 정의하는 변형을 제안. 통신 네트워크에서 노드 장비 비용이 연결 수에 따라 달라지는 상황을 모델링 | **직접적 선례**: edge cost가 endpoint node의 속성에 의존하는 PCST variant의 이론적 정당성 |
| G-Retriever (He et al., NeurIPS 2024) | PCST를 Knowledge Graph에서 query-relevant subgraph 추출에 적용. Node prize를 query 관련성으로 설정 | Prize-aware graph extraction이 실제 검색 시스템에서 효과적임을 실증 |
| Goemans & Williamson (1995), Primal-Dual for PCST | pcst_fast의 기반 알고리즘. **임의의 비음수 cost에 대해 2-근사 보장**이 유지됨 | Product form cost가 비음수이면 알고리즘의 근사 보장이 유효 |
| Prize-Collecting Steiner Tree: A 1.79 Approximation (2024, arXiv) | PCST 최신 1.7994-근사 달성 | 동적 cost 구조에서도 근사 알고리즘 적용 가능 |

### 판단

- **타당성**: 매우 높음 — 가장 큰 정량적 병목(prize-cost 스케일 불일치)을 구조적으로 해소
- **구현 복잡도**: 낮음 — `extract()` 내 cost 계산 로직만 변경
- **예상 효과**: Recall +3~8%, F1 +2~5% (특히 multi-table query에서)
- **위험**: 양쪽 모두 중간 점수(~0.5)인 경우 cost가 과도하게 낮아질 수 있음 → `min_cost` floor로 방어
- **구현 우선순위**: **1순위** — 코드 변경 최소, 효과 최대

---

## 아이디어 5: Query-Conditioned GAT (Query 정보를 GAT에 주입)

### 핵심 관찰
> "GAT 모델에도 Query 정보를 줘야 한다. Super Node를 만들어서 붙이든, 모든 Node Embedding에 Concat을 하든 하는 방식으로."

### 현재 한계 진단

현재 `SchemaHeteroGAT`(`gat_network.py:56-78`)의 `forward(self, x_dict, edge_index_dict)`는 **query embedding을 전혀 받지 않는다.** Query 정보는 GAT 이후 DualTowerProjector에서 비로소 등장하여, cosine similarity로 post-hoc 비교만 수행한다.

이는 **GAT의 attention이 query-agnostic**하다는 것을 의미한다. 어떤 query가 들어오든 동일한 schema graph에 대해 동일한 node embedding을 생산한다. "이 query에서 어떤 table 간 연결이 중요한가?"를 message passing 단계에서 판단하지 못하므로, bridge table detection이 query-conditioned attention 없이 일반적 구조 패턴에만 의존하게 된다.

### 두 가지 구현 방안 비교

#### 방안 A: Super Node (Query Node)

Query embedding을 별도 노드로 추가하여 모든 schema 노드와 양방향 edge로 연결.

```
수정 전: Table ↔ Column ↔ FK_Node ↔ Column ↔ Table
수정 후: Query ↔ Table ↔ Column ↔ FK_Node ↔ Column ↔ Table
                ↕            ↕                        ↕
               Query       Query                    Query
```

**장점**:
- Query가 매 GATv2Conv layer마다 message passing에 참여 → query-conditioned attention
- 양방향: schema → query (어떤 schema가 query와 관련?) + query → schema (query가 어떤 schema에 집중?)
- Gilmer et al. (2017)의 virtual node trick에 의해 O(1) depth로 장거리 정보 전파
- 이론적으로 full self-attention(Transformer)을 근사 가능 (ICLR 2023, "Revisiting Virtual Nodes in GNNs")

**단점**:
- 2N개 edge 추가 (N = 전체 schema 노드 수, 보통 100~500) → 메모리 ~20-30% 증가
- Over-smoothing 위험: query node가 후반 layer에서 모든 노드를 지배할 수 있음
- HeteroConv에 새로운 edge type (`query↔table`, `query↔column`, `query↔fk_node`) 추가 필요
- Batch 처리 시 각 sample마다 별도 query node 관리 필요

**구현 변경점**:
- `gat_network.py`: forward()에 `query_emb` 매개변수 추가, 새 edge type GATv2Conv 추가
- `train_gat.py`: training loop에서 query_emb를 GAT에 전달
- `bird_dataset.py`: HeteroData에 `data['query_node']` 추가
- `gat_classifier_selector.py`: 추론 시 query_emb를 GAT에 전달

#### 방안 B: Concatenation (Node Feature Augmentation)

Query embedding을 모든 node feature에 concat하여 GAT의 첫 입력을 augment.

```python
# 기존: x_table = [table_emb]  (384-dim)
# 변경: x_table = [table_emb || query_emb]  (384+384 = 768-dim)
```

**장점**:
- 구현 최소 변경: `lin_dict`의 in_channels를 384→768로 변경하면 됨
- 그래프 구조 변경 없음 (edge type 추가 불필요)
- 메모리 오버헤드 작음 (~2-5%)
- Pre-trained GAT에도 적용 가능 (projection layer만 fine-tune)

**단점**:
- Query 정보가 첫 layer에서만 직접 입력 → 깊은 layer에서 희석됨
- 단방향: schema는 query를 "본다"이지만, query가 schema로부터 업데이트를 받지는 않음
- Feature dimension 2배 → 첫 Linear layer의 파라미터 2배

**구현 변경점**:
- `gat_network.py`: forward()에 `query_emb` 추가, `lin_dict`의 in_channels 조정
- `train_gat.py`: query_emb를 각 batch에서 node별로 expand하여 concat
- 나머지 코드 변경 최소

### 권장 접근 순서

1. **방안 B (Concatenation)** 먼저 — 구현이 간단하고 빠르게 효과 검증 가능
2. 효과가 있으면 **방안 A (Super Node)** 로 확장하여 full query conditioning 실험
3. 두 방안 모두 ablation에 포함하여 논문에 비교 분석

### 이론적 근거

| 참고문헌 | 핵심 주장 | 관련성 |
|---------|---------|-------|
| Bogin et al. (ACL 2019), "Representing Schema Structure with GNN for Text-to-SQL Parsing" | Query-conditioned relevance score(ρ_v = max_i p_link(v\|x_i))를 GNN 노드 초기값에 곱하여 soft pruning. Multi-table 정확도 14.6%→26.8% | **직접적 선례**: query 정보를 GNN 입력에 주입하여 schema linking 성능 대폭 향상 |
| RAT-SQL (Wang et al., ACL 2020) | [question tokens \| schema tokens] 통합 시퀀스에서 relation-aware self-attention 적용. Spider +8.7% | Question과 schema를 동일 attention 공간에서 처리하면 구조적 관계 학습 극대화 |
| S²SQL (Lyu et al., ACL Findings 2022) | Question-Schema interaction graph를 구성하여 question token도 그래프 노드로 포함. SOTA 달성 | Super node 방안의 직접적 선례: question을 그래프의 일부로 모델링 |
| Gilmer et al. (ICML 2017), "Neural Message Passing for Quantum Chemistry" | Virtual node(master node)를 모든 노드에 연결하여 O(1) depth 장거리 통신 실현 | Super node의 이론적 정당성: MPNN + virtual node ≈ Transformer attention |
| "Revisiting Virtual Nodes in GNNs" (ICLR 2023) | Virtual node가 over-smoothing을 야기할 수 있으나, 적절한 residual connection으로 완화 가능. Graph classification 에서 1-3% 향상 | Over-smoothing 위험과 해결책을 모두 제시 |
| ShadowGNN (Chen et al., NAACL 2021) | Schema를 abstract representation으로 투영 후 relation-aware transformer로 question-schema alignment | Query-independent GNN + post-hoc alignment의 한계를 보여줌 (현재 아키텍처의 문제와 동일) |

### 판단

- **타당성**: 매우 높음 — 현재 GAT의 가장 근본적 한계(query-agnostic)를 해소. 아이디어 1(bridge awareness)의 **전제 조건**이기도 함 (query를 모르면 어떤 bridge가 중요한지 판단 불가)
- **구현 복잡도**: 방안 B는 낮음, 방안 A는 중간
- **예상 효과**: GAT 기여도를 2.1% → 5~15%로 끌어올릴 잠재력. 아이디어 1과 결합 시 시너지 극대화
- **구현 우선순위**: **아이디어 1보다 선행 필요** — query conditioning 없이 bridge detection을 강화해도 효과 제한적
- **주의점**: in_channels 변경 시 기존 체크포인트(`best_gat_model.pt`)와 호환 불가 → 새로운 체크포인트명 사용 (이미 `best_gat_enriched.pt` 패턴 확립됨)

---

## 아이디어 3: MST Backbone + PCST Expansion

### 핵심 관찰
> "Ensemble Seed Selector가 Seed Node를 정한 뒤 해당 Seed Node 끼리 MST를 통해 확실하게 Connectivity를 확보한 뒤에, PCST로 추가 Node에 대한 Expansion 및 Pruning을 고려해보는 건 어떻겠느냐?"

### 현재 한계 진단

1. **seed_nodes 미활용**: `schema_linking.py:153-156`에서 `seed_nodes=seeds`를 extractor에 넘기지만, `AdaptivePCSTExtractor`는 이를 **무시**한다. Selector의 top-k 선택이 실질적으로 활용되지 않고 있음.

2. **기존 MSTExtractor의 결함**: `mst.py:25`에서 `G.subgraph(seed_nodes)`로 seed 간 **직접 연결된 edge만** 포함하는 subgraph를 만들어 MST를 구한다. 그러나 schema graph에서 두 table은 보통 column→FK→column을 거쳐 2~4 hop 떨어져 있으므로, 직접 연결이 없어 **대부분의 경우 disconnected subgraph**가 반환된다.

3. **올바른 접근**: Steiner Tree의 고전적 2-근사 알고리즘(Kou, Markowsky, Berman 1981)은 **metric closure**(seed 간 최단 경로 가중치로 완전 그래프 생성) → MST → 원래 경로로 복원의 3단계로 구성된다.

### 구현 방안 (Two-Phase Extraction)

**Phase 1: Steiner Tree Backbone**
```python
# 1. 전체 schema graph를 NetworkX로 구성
# 2. Seed nodes (top-k from Selector) 간 all-pairs shortest paths
# 3. Metric closure: seed 간 완전 그래프, 가중치 = shortest path length
# 4. MST on metric closure
# 5. MST edge → 원래 graph의 shortest path로 복원 → backbone nodes/edges
```

**Phase 2: PCST Expansion**
```python
# Backbone nodes에 bonus prize 부여 (예: prize += max_prize * 0.5)
# 또는 backbone edges의 cost를 0으로 설정
# 전체 graph에서 PCST 실행 → backbone을 포함하면서 추가 확장/가지치기
```

**핵심**: Phase 1이 connectivity를 **보장**하고, Phase 2가 추가 관련 노드를 **발견**하는 역할 분담.

### 이론적 근거

| 참고문헌 | 핵심 주장 | 관련성 |
|---------|---------|-------|
| Kou, Markowsky, Berman (1981), "A fast algorithm for Steiner trees" | Metric closure MST 기반 Steiner tree 2-근사 알고리즘. 시간복잡도 O(\|S\|·(V+E log V)) (S=seed 수) | **직접적 이론 기반**: Phase 1의 backbone 구축이 이 알고리즘 |
| Zelikovsky (1993), Steiner tree 1.55-근사 | MST 기반 초기해에 Steiner point를 반복 추가하여 개선 | Phase 2(PCST expansion)가 Steiner point 추가와 유사 |
| DIN-SQL (Pourreza & Rafiei, NeurIPS 2023) | Text-to-SQL을 decomposition 기반으로 4단계로 분리하여 성능 향상 | Multi-phase 분해 접근법의 실증적 효과 |
| DAIL-SQL (Gao et al., VLDB 2024) | Schema linking → skeleton parsing → generation의 다단계 파이프라인 | Phase별 역할 분담이 전체 정확도 향상에 기여 |

### 판단

- **타당성**: 중간 — 아이디어 자체는 건전하나, **아이디어 2(Product Cost)가 macro_cost 문제를 해결하면 MST backbone의 추가 이득이 제한적**일 수 있음
- **구현 복잡도**: 중간 — metric closure + MST는 NetworkX로 구현 가능하지만, Phase 1-2 연결부의 prize/cost 조정 설계가 필요
- **예상 효과**: Recall +2~6% (multi-table query), Precision -1~3% (backbone 강제 포함에 의한 노이즈)
- **핵심 위험**: seed 품질에 강하게 의존 — seed에서 누락된 table은 backbone에도 없으므로 PCST expansion이 독립적으로 발견해야 함
- **구현 우선순위**: **4순위** — 아이디어 2 실험 후 bridge recall이 여전히 부족할 때 고려

---

## 아이디어 4: Connected Component 분리 후 독립 실행

### 핵심 관찰
> "실제 산업 현장에서 전체 DB가 모두 연결되지 않았을 수도 있다. 전체 Graph의 Connected Component를 먼저 구하게 한 다음 각 Connected Component에 대해 이 프레임워크를 실행하게 하는 게 낫지 않겠느냐?"

### 현재 한계 진단

현재 `pcst_fast`는 `root=-1`로 호출되어(`pcst.py:58`) disconnected graph를 forest로 처리한다. 그러나 전체 graph에서 global하게 prize-cost tradeoff를 계산하므로, component 간 prize 분포 차이가 반영되지 않는다. 특히 `AdaptivePCSTExtractor`의 P80 threshold는 **전체 노드의 score 분포**에서 계산되므로, 작은 component의 노드가 큰 component의 분포에 의해 threshold가 결정되는 문제가 있다.

### 구현 방안

```python
class ComponentAwareMixin:
    def _decompose(self, edges, num_nodes):
        """Union-Find로 connected components 분리. O(V + E)."""
        parent = list(range(num_nodes))
        def find(x): ...
        def union(a, b): ...
        for u, v in edges:
            union(u, v)
        # component별 (node_ids, edge_ids) 반환
```

**적용 방식**: Decorator/Mixin 패턴으로 **기존 어떤 PCST extractor에도 부착 가능**.

```python
@register("extractor", "ComponentAwareAdaptivePCSTExtractor")
class ComponentAwareAdaptivePCSTExtractor(ComponentAwareMixin, AdaptivePCSTExtractor):
    def extract(self, graph_data, node_scores, ...):
        components = self._decompose(edges, len(node_scores))
        results = []
        for comp_nodes, comp_edges, comp_edge_types in components:
            if max(node_scores[n] for n in comp_nodes) < threshold:
                continue  # prize가 없는 component는 skip
            # Component 내에서 독립적으로 adaptive threshold 계산 & PCST 실행
            comp_result = super().extract(comp_graph_data, comp_scores, ...)
            results.append(comp_result)
        return merge(results)
```

### 이론적 근거

| 참고문헌 | 핵심 주장 | 관련성 |
|---------|---------|-------|
| Tarjan (1975), Union-Find | Union-Find의 시간복잡도 O(V·α(V)) ≈ O(V). Connected component 탐색의 표준 알고리즘 | 구현 기반 |
| LinkAlign (EMNLP 2025) | Multi-database 환경에서 database retrieval → schema item grounding의 2단계 분리. Database를 먼저 선택하는 것이 schema decomposition의 한 형태 | Schema decomposition의 실증적 효과 |
| RDB2G-Bench (2024) | 50개 실제 기업 DB 분석: 수백 테이블, 수천 컬럼. 적은 테이블만 선택적으로 사용하는 것이 성능과 효율 모두 향상 | 실제 대규모 DB에서 schema decomposition의 필요성 |
| PCST 분해 정리 | PCST의 목적함수는 connected components에 대해 **가법적으로 분해 가능**. 즉, 각 component에서 독립적으로 최적해를 구하면 전체 최적해와 동등 | 이론적 정당성: component별 독립 실행이 근사 손실 없음 |

### 판단

- **타당성**: 높음 — 구현이 매우 단순하고, 수학적으로 정당하며, 기존 어떤 extractor에도 적용 가능
- **구현 복잡도**: 매우 낮음 — ~30줄 코드
- **예상 효과**: BIRD dev에서는 미미(대부분 connected), 실제 production에서는 유의미
- **부가 이점**: component별로 **독립적 adaptive threshold** 계산 가능 → 스키마 크기에 따른 threshold 편향 해소
- **구현 우선순위**: **2순위** — 구현 비용 대비 이론적 기여가 확실하고, 다른 아이디어와 직교적으로 결합 가능

---

## 종합 우선순위 및 실행 계획

| 순위 | 아이디어 | 구현 복잡도 | 예상 F1 개선 | 이론적 기여 | 논문 차별성 |
|-----|---------|-----------|------------|-----------|-----------|
| **1** | 2. Product Cost PCST | 낮음 | +2~5% | PCST variant 제안 | Prize-Cost scale 일치 원리 |
| **2** | 4. Connected Component | 매우 낮음 | +0~1% (BIRD) | 분해 정리 적용 | 산업 적용성 강조 |
| **3** | 1. Bridge Table GAT | 중간~높음 | +1~4% | 구조적 label 설계 | GAT의 bridge 인식 능력 |
| **4** | 3. MST + PCST | 중간 | +1~3% | Steiner 2-근사 활용 | Two-phase extraction |
| **5** | 5. Query-Conditioned GAT | 중간 | GAT 기여 2.1%→5~15% | Query-aware GNN | 아이디어 1의 전제조건, query-conditioned attention |

### 결합 가능성
- **2+4**: 자연스럽게 결합 (Component Mixin + ProductCost Extractor)
- **1+2**: 자연스럽게 결합 (더 좋은 GAT 점수 → 더 정확한 product cost)
- **5→1**: 아이디어 5는 아이디어 1의 전제조건 (query conditioning 없이 bridge detection 강화 효과 제한적)
- **2+3**: 결합 가능하나 잠재적 중복 (둘 다 bridge connectivity 해소 목적)
- **1+2+4+5**: 최종 권장 조합

### 검증 방법

각 아이디어를 기존 2×2×2 ablation 프레임워크 위에서 실험:
1. 아이디어 2: `ProductCostPCSTExtractor`로 8개 ablation cell 재실행, 기존 결과와 비교
2. 아이디어 4: Mixin 적용 후 동일 비교
3. 아이디어 5: 방안 B(Concat) → 방안 A(Super Node) 순서로 GAT 재학습 후 비교
4. 아이디어 1: α 조정 → FK label → bridge loss 순서로 점진적 실험 (아이디어 5 적용 후)
5. 아이디어 3: 아이디어 2 실험 후 bridge recall이 부족할 때만 추가 검증

### 변경될 파일

| 파일 | 아이디어 | 변경 내용 |
|------|---------|---------|
| `src/modules/extractors/pcst.py` | 2, 3, 4 | ProductCostPCSTExtractor, ComponentAwareMixin, MSTBackbonePCSTExtractor 추가 |
| `src/modules/extractors/mst.py` | 3 | Metric closure 기반 Steiner tree 2-근사로 교체 |
| `src/data/bird_dataset.py` | 1 | FK node label 추가, bridge table label 추가 |
| `src/train_gat.py` | 1, 5 | FK node loss 포함, bridge auxiliary loss 추가, query_emb를 GAT에 전달 |
| `src/models/gat_network.py` | 5 | forward()에 query_emb 매개변수 추가, Concat 또는 Super Node 구현 |
| `src/modules/selectors/gat_classifier_selector.py` | 5 | 추론 시 query_emb를 GAT에 전달 |
| `src/utils/evaluator.py` | 1 | `parse_sql_join_tables()` 함수 추가 |
| `configs/experiments/` | 전체 | 각 아이디어별 실험 config 추가 |

### 산출물

- **파일**: `notebooks/analysis_results/advisor_meeting_ideas_analysis.md` — 본 분석 문서를 MD로 저장
