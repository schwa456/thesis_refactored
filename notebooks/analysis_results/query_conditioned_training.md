# Query-Conditioned GAT Training

지도교수 면담 아이디어 5번("GAT 모델에도 Query 정보를 줘야 한다")의 구현 기록.
현재 GAT은 `x_dict`와 `edge_index_dict`만 받으며 query embedding을 전혀 보지 못한다.
DualTowerProjector 단계에서야 cosine similarity로 post-hoc 비교할 뿐이다.
이 문서는 query 정보를 GAT 내부로 주입하기 위한 두 가지 방안(Concat, Super Node)과,
query 중복을 제거하기 위한 Direct 변형의 설계/코드/실행 절차를 정리한다.

---

## 1. 기존 파이프라인의 한계

- `SchemaHeteroGAT.forward(x_dict, edge_index_dict)` — query 미입력
- `DualTowerProjector`에서만 query ↔ node cosine 유사도 계산
- Message passing은 schema 구조만으로 수행되어 **query-conditioned attention 불가**
- 결과: GAT 순 기여가 +2.1%(prior ensemble 분석)에 그침

Query가 **파이프라인에서 3회 중복**되는 구조적 문제도 존재:

1. (Selector) raw cosine baseline에서 query 사용
2. (GAT 후) DualTowerProjector 재입력 시 query 사용
3. (PCST 이후) XiYan Filter에서 query 사용

→ GAT 내부로 query를 직접 주입(1회)하면 중복 제거 가능하며, Loss도 BCE 단독으로 단순화 가능.

---

## 2. 두 가지 주입 방안

### 방안 A: Query Concatenation (`query_conditioned=True`)

모든 schema 노드 feature에 query embedding을 concat하여 GAT 첫 입력을 augment.

```python
# src/models/gat_network.py:22
effective_in = in_channels * 2 if query_conditioned else in_channels
# 입력 차원 384 → 768 (query 384 + node 384)
```

```python
# src/models/gat_network.py:82-90
if self.query_conditioned and query_emb is not None:
    if query_emb.dim() == 1:
        query_emb = query_emb.unsqueeze(0)
    augmented = {}
    for nt, x in x_dict.items():
        q_exp = query_emb.expand(x.size(0), -1)
        augmented[nt] = torch.cat([x, q_exp], dim=-1)
    x_dict = augmented
```

**장점**
- 구현 최소 변경: `lin_dict`의 in_channels 384→768만 조정
- 그래프 구조 변경 없음 (edge type 추가 불필요)
- 메모리 오버헤드 낮음 (~2-5%)

**단점**
- Query 정보가 **첫 layer에서만 직접 입력** → 깊은 layer에서 희석
- 단방향: schema는 query를 보지만, query가 schema로부터 업데이트되지는 않음
- 첫 Linear layer의 파라미터 2배

### 방안 B: Query Super Node (`query_supernode=True`)

Query embedding을 별도 노드 `query_node`로 추가하고, 모든 schema 노드와 **양방향 edge**로 연결.
Bipartite injection은 `BIRDSuperNodeDataset`(`src/data/bird_dataset.py:134-181`)에서 수행.

```python
# src/data/bird_dataset.py:151-172
data['query_node'].x = q_emb  # [1, 384]
for schema_nt in ['table', 'column', 'fk_node']:
    num_nodes = data[schema_nt].x.size(0)
    src = torch.zeros(num_nodes, dtype=torch.long)   # query idx = 0
    dst = torch.arange(num_nodes, dtype=torch.long)
    data['query_node', f'attends_to_{schema_nt}', schema_nt].edge_index = \
        torch.stack([src, dst], dim=0)
    data[schema_nt, f'attended_by_{schema_nt}', 'query_node'].edge_index = \
        torch.stack([dst, src], dim=0)
```

모델 측에서는 supernode 전용 edge type을 각 layer에 주입한다:

```python
# src/models/gat_network.py:48-54
if query_supernode:
    for schema_nt in ['table', 'column', 'fk_node']:
        supernode_edge_types[('query_node', f'attends_to_{schema_nt}', schema_nt)] = \
            GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
        supernode_edge_types[(schema_nt, f'attended_by_{schema_nt}', 'query_node')] = \
            GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
```

**장점**
- Query가 매 GATv2Conv layer마다 message passing에 참여 → **진정한 query-conditioned attention**
- 양방향: schema → query (어떤 schema가 query와 관련?) + query → schema (query가 어떤 schema에 집중?)
- Gilmer et al. (2017) virtual node trick: O(1) depth 장거리 정보 전파

**단점**
- 2N개 edge 추가 → 메모리 ~20-30% 증가
- Over-smoothing 위험 (query node가 후반 layer에서 모든 노드를 지배)
- Batch 처리 시 sample별 query node 관리 필요
- 새로운 edge type 6종 추가 (HeteroConv 확장)

---

## 3. Direct 변형 (BCE-only, Projector 제거)

`train_gat.py`의 기존 경로는 GAT → DualTowerProjector → InfoNCE + BCE (query 2회 사용).
Query-Conditioned GAT에서는 query가 **GAT 내부에서 이미 반영**되므로 DualTowerProjector 재입력이 중복.

`train_gat_direct.py`는 이를 제거한 변형:

- GAT forward **1회**만 수행
- `DirectClassifierHead`(MLP: in→hidden→hidden/2→1)로 바로 binary logit 예측
- Loss: **BCE only** (InfoNCE 제거 — 공통 query 공간이 없으므로 contrastive 불가)
- 체크포인트에 `gat_state_dict` + `classifier_state_dict` 저장

```python
# src/train_gat_direct.py:163-172
classifier_types = ['table', 'column', 'fk_node']
classifier_heads = nn.ModuleDict({
    nt: DirectClassifierHead(
        in_dim=cfg['model']['out_channels'],
        hidden_dim=cfg['model'].get('classifier_hidden', 256),
        dropout=cfg['model'].get('dropout', 0.1),
    ).to(device)
    for nt in classifier_types
})
```

추론은 `src/modules/selectors/direct_gat_selector.py`의 `DirectGATClassifierSelector`가 담당한다.

---

## 4. 4가지 변형 요약

| 변형 | Train script | 모델 플래그 | 체크포인트 | Loss |
|---|---|---|---|---|
| Query-Conditioned (non-direct) | `train_gat.py` | `query_conditioned=true` | `best_gat_query_conditioned.pt` | BCE + InfoNCE (Projector) |
| Query-Supernode (non-direct) | `train_gat.py` | `query_supernode=true` | `best_gat_query_supernode.pt` | BCE + InfoNCE (Projector) |
| Query-Conditioned Direct | `train_gat_direct.py` | `query_conditioned=true` | `best_gat_query_conditioned_direct.pt` | BCE only |
| Query-Supernode Direct | `train_gat_direct.py` | `query_supernode=true` | `best_gat_query_supernode_direct.pt` | BCE only |

Config 위치: `configs/training/train_gat_query_*.yaml`
Experiment alpha: qcond=0.85, supernode=0.70 (`configs/experiments/experiment_*_idea24_xiyan.yaml`)

---

## 5. 학습 실행 커맨드

```bash
cd /home/hyeonjin/thesis_refactored

# Query-Conditioned Direct
python src/train_gat_direct.py --config configs/training/train_gat_query_conditioned_direct.yaml

# Query-Supernode Direct
python src/train_gat_direct.py --config configs/training/train_gat_query_supernode_direct.yaml

# Non-Direct (Projector + InfoNCE) 변형
python src/train_gat.py --config configs/training/train_gat_query_conditioned.yaml
python src/train_gat.py --config configs/training/train_gat_query_supernode.yaml
```

공통 하이퍼파라미터: epochs=300, batch_size=8, lr=1e-4, wd=1e-5, pos_weight=100, recall_k=15.

---

## 6. 이론적 근거

| 참고문헌 | 핵심 주장 | 관련성 |
|---|---|---|
| Bogin et al. (ACL 2019), "Representing Schema Structure with GNN for Text-to-SQL Parsing" | Query-conditioned relevance score를 GNN 노드 초기값에 곱하여 soft pruning. Multi-table 14.6%→26.8% | 직접적 선례 |
| RAT-SQL (Wang et al., ACL 2020) | [question \| schema] 통합 시퀀스에서 relation-aware self-attention. Spider +8.7% | Question/schema 통합 attention |
| S²SQL (Lyu et al., ACL Findings 2022) | Question-Schema interaction graph. Question token을 그래프 노드로 포함 | Super Node 방안의 선례 |
| Gilmer et al. (ICML 2017), Neural Message Passing | Virtual node trick: 모든 노드에 연결된 master node로 O(1) 장거리 통신 | Super node의 이론적 정당성 |
| "Revisiting Virtual Nodes in GNNs" (ICLR 2023) | Virtual node가 over-smoothing을 야기할 수 있으나, residual로 완화 가능. Graph classification +1-3% | 위험과 해결책 |
| ShadowGNN (Chen et al., NAACL 2021) | Query-independent GNN + post-hoc alignment의 한계 | 현재 아키텍처의 문제 |

---

## 7. 주의점

- `in_channels` 변경 시 기존 체크포인트와 **호환 불가** → 별도 checkpoint_name 사용 (이미 분리됨).
- BIRDSuperNodeDataset는 `BIRDGraphDataset`을 wrapping하므로 cache는 base dataset에만 저장된다.
  Supernode 주입은 매 training run마다 `_inject_supernode()`에서 수행된다.
- Query-Conditioned + Direct는 query가 파이프라인에서 한 번만 사용되므로 **추론 속도가 약 20% 향상**.
- Non-Direct 변형은 DualTowerProjector의 공동 임베딩 공간이 필요한 cosine baseline 실험을 위해 유지.
