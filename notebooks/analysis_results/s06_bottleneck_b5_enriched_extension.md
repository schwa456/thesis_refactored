# s06_a01_07 (B5 Enriched) — 3축 병목 분석 확장

**분석일**: 2026-04-21
**대상**: `s06_a01_07_b5_enriched_dual_stream` (B5 구조 + Enriched features)
**비교 기준**: [`s06_bottleneck_comparison.md`](s06_bottleneck_comparison.md) (B0~B5)
**체크포인트**: `/SSL_NAS/peoples/khj/thesis/checkpoints/s06_gat_bottleneck_fix/best_gat_s06_a01_07_b5_enriched.pt`
**산출물 경로**:
- `outputs/analysis/s06_bottleneck_b5_enriched/` (B5E 단독)
- `outputs/analysis/s06_bottleneck_merged/` (B0~B5 + B5E cross-model)

---

## 1. B5E 개요

| 항목 | 값 |
|---|---|
| Builder | EnrichedHeteroGraphBuilder (tables.json 자연어명 + database_description/*.csv) |
| Model | B5 동일 — L=2, PairNorm, IR α=0.2, JK=concat, Dual-Stream, ListNet, AC(0.3, 0.85) |
| Batch | 8 (B5는 1, batched dual_stream으로 가속) |
| 총 학습 시간 | 9h 14m (300/300 epoch, B5 ~29h → **3.1× 단축**) |

---

## 2. Step 1 — Loss/Recall Trajectory

| ID | Best R@15 | Best Ep | Final R@15 | Epochs | Final Loss |
|----|-----------|---------|------------|--------|------------|
| B0 | 0.5738 | 298 | 0.5713 | 300 | 0.7494 |
| B5 | **0.6073** | **62** | 0.6021 | 285 | 1.1617 |
| **B5E** | **0.6016** | **60** | 0.5969 | 300 | 1.1382 |

### 핵심 관찰
- **B5E < B5 by −0.0057** — Enriched features가 R@15를 개선하지 못함 (기대와 반대 방향)
- **수렴 에폭 동일 (B5=62 / B5E=60)** — 구조적 수렴 속도는 동일
- **60 epoch 이후 240 epoch 플래토** — over-training 여지 없음, 80~100 epoch로 단축 가능
- Final loss가 소폭 낮음 (1.14 vs 1.16) — Enriched가 train fit은 약간 향상, dev 일반화엔 미반영

### 해석
Enriched features는 BIRD train에는 있지만 dev가 동일 DB 분포이므로 OOD 완화 효과가 이론적으로 제한적. 추가된 자연어 설명이 **train에서 과적합 신호로 작용**했을 가능성 → dev에서 오히려 미세하게 hurt.

플롯: `s06_bottleneck_b5_enriched/objective_mismatch_B5E.png`

---

## 3. Step 2 — Over-smoothing (Intra-Table Cosine)

| ID | L0_PLM | L1_GAT | L2_GAT | L_out | ΔL0→L_out |
|----|--------|--------|--------|-------|-----------|
| B0 (L=3) | 0.657 | 0.851 | 0.891 | 0.833 | +0.176 |
| B4 (L=3) | 0.657 | 0.386 | 0.947 | 0.562 | −0.095 |
| B5 (L=2) | 0.657 | 0.373 | **0.920** | **0.357** | −0.300 |
| **B5E (L=2)** | **0.636** | 0.430 | **0.978** | **0.329** | **−0.307** |

### 핵심 관찰
- **L0_PLM 감소 (0.657 → 0.636)**: Enriched 텍스트가 column 간 원본 임베딩을 **더 분산** 시킴 (의도대로 작동). Baseline PLM features보다 sibling discriminability가 높은 시작점.
- **L2_GAT 상승 (0.920 → 0.978, 거의 collapse)**: 2-layer GAT 후엔 오히려 **B5보다 동일 테이블 column이 더 비슷해짐**. `column→belongs_to→table` attention entropy가 매우 높아 (≈1.95) table 중심 pooling이 강하게 일어남 → richer features가 오히려 table-centric homogenization을 강화하는 역설.
- **L_out 추가 감소 (0.357 → 0.329)**: Dual-Stream fusion이 L2 collapse를 뚫고 최종 표현을 더 분산. **"Fusion이 GAT 병리를 사후 교정"** 하는 그림이 B5E에서 더 뚜렷.

### 축별 기여 해석 (B5 → B5E)
| 축 | B5 | B5E | Δ |
|----|----|----|---|
| L0 시작점 | 0.657 | 0.636 | **−0.021** (더 좋음) |
| L1 통과 후 | 0.373 | 0.430 | +0.057 (악화) |
| L2 통과 후 | 0.920 | 0.978 | +0.058 (심화) |
| L_out (fusion) | 0.357 | 0.329 | **−0.028** (더 좋음) |

→ **GAT 구간은 B5E가 불리, Fusion이 뒤집음.** 최종 표현 품질 자체는 B5E가 약간 우위.

플롯: `s06_bottleneck_merged/cross_model_oversmoothing.png`, `s06_bottleneck_b5_enriched/B5E/oversmoothing_trajectory.png`, `s06_bottleneck_b5_enriched/B5E/tsne_query0.png`

---

## 4. Step 3 — Gradient Flow & Attention Entropy

### 4-1. Gradient per parameter group

| 그룹 | B5 | B5E | 해석 |
|------|----|----|------|
| `lin_dict` | 0.43 | **1.13** | Enriched input이 더 큰 gradient 요구 (L0 embedding 갱신 증가) |
| `conv_L1` | 0.043 | **0.171** | GAT conv1도 4× 더 많이 학습 |
| `conv_L2` | 0.030 | 0.042 | L2는 소폭 증가 |
| `jk_lin` | 0.144 | **0.515** | JK concat projection 3.6× |
| `skip_dict` | 0.125 | **0.545** | Skip 경로 4.4× |
| `query_encoder` | 0.629 | **1.349** | Query 분기 2.1× |
| `fusion_head` | 0.592 | **1.827** | Fusion 3.1× |
| `grad_ratio (L2/L1)` | **0.687** | **0.244** | Conv 경로 gradient 불균형 심화 |

### 4-2. Attention Entropy

| edge-type | L1 (B5E) | L2 (B5E) |
|-----------|----------|----------|
| `column→belongs_to→table` | **1.945** | 1.860 (여전히 max entropy, 분산) |
| `fk_node→points_to→column` | 0.758 | 0.756 (sharp, 유지) |
| `table→table_to_table→table` | 0.617 | 0.613 (JOIN, 유지) |

### 핵심 관찰
- **모든 파라미터 그룹의 gradient가 B5 대비 2~4× 증가** — Enriched input은 학습 신호를 **양적으로** 크게 키우지만, **질적으로 R@15로 변환되지 못함**.
- **grad_ratio 0.244**: conv_L2 gradient가 conv_L1 대비 1/4 수준. 2-layer 구조에서 L2가 이미 수렴한 상태로 간주 가능. 절대값(0.042)은 vanish threshold(0.04 근방) 근처지만, 비슷한 크기의 B5(0.030)가 잘 학습된 것으로 보아 vanish 위기는 아님.
- **Fusion과 Query encoder가 compensation 역할**: B5E에서 fusion_head gradient가 1.83으로 최대 — Fusion이 **"GAT가 만든 L2 collapse를 뚫고 sibling을 분리"** 하는 역할을 전담. B5 대비 이 역할이 더 비대해짐.
- **Attention entropy 패턴은 B5와 거의 동일** — Enriched features가 edge weight 학습에는 영향 못 미침. 구조(Dual-Stream + Fusion)에만 의존.

플롯: `s06_bottleneck_merged/cross_model_grad_ratio.png`, `s06_bottleneck_b5_enriched/B5E/gradient_flow.png`, `s06_bottleneck_b5_enriched/B5E/attention_entropy.png`

---

## 5. 종합 해석 — 왜 B5E가 B5보다 못했는가

### 가설 1: Enriched features가 L0를 분산시켰으나, GAT가 그 분산을 유지 못함
- L0 분산 0.657 → 0.636 (좋은 출발점)
- L2에서 B5보다 더 collapse (0.920 → 0.978)
- `column→belongs_to→table` edge가 **같은 table 컬럼들을 더 비슷하게 만드는 평균화 연산**인데, 컬럼들 자체가 풍부한 정보를 가지면 table 노드가 더 중요한 정보를 축적 → sibling column 구분 불필요로 해석

### 가설 2: Train-time over-fitting — Enriched features의 train-specific bias
- Final train loss가 B5E(1.138)가 B5(1.162)보다 낮음 — train fit 개선
- Dev R@15는 오히려 하락 — 일반화 실패
- Enriched text가 train DB별 특성에 더 fit → dev DB에서 score calibration drift 가능성

### 가설 3: 2-layer는 Enriched에 적은 깊이
- B0~B4는 3-layer였고, B5만 2-layer. 2-layer는 B5의 "구조+ListNet+AC" 조합에서 필요충분
- 그러나 Enriched 정보를 전파/추상화하려면 추가 hop이 필요할 수 있음
- 검증 필요: B5E L=3 재학습

---

## 6. 추천 후속 실험

1. **B5E L=3 재학습** — Enriched에 깊이가 필요한지 확인. 2-layer 가설 재검증.
2. **B5E Early stop @ E60** — 300 epoch 무용, 60 epoch 체크포인트를 실전 투입하여 동일 성능 확인 → 학습시간 절감.
3. **"Enriched features 단독" 효과 분리** — B0 또는 B4에 Enriched만 적용 (dual-stream/JK 없이). Enriched의 단순 기여도 측정.
4. **Enriched가 Extractor/Filter에 주는 영향** — Seed score 품질은 같거나 약간 저하지만, PCST에서 더 좋은 후보를 남기는지 확인 (E2E F1로).
5. **B5E + Neurosymbolic L1 (S-V)** — FK-reachability prior가 Enriched GAT와 결합 시 시너지 측정 (`abl_sel_ns_l1_02`로 예약 가능).

---

## 7. 결론

- **R@15 관점**: B5 (0.6073) > B5E (0.6016) — **Enriched가 순효과 없음** (−0.0057).
- **Over-smoothing 관점**: B5E가 L2에서 더 collapse되지만 Fusion이 이를 L_out에서 더 많이 교정 (−0.307 vs −0.300) — 구조적으로는 소폭 개선.
- **학습 부담**: B5E가 모든 파라미터에 2~4× 큰 gradient 부담 → 추가 학습 신호가 R@15로 전환되지 않음.
- **결정적 관찰**: **Fusion head가 병목이자 구원자**. L2 collapse는 심화되었지만 Fusion이 더 큰 용량으로 보완. Dual-Stream 구조 없이 Enriched만 쓰면 L2 collapse가 그대로 downstream에 전파될 가능성.

→ **B5E는 학습 시간 단축(3.1×)과 L_out 품질 향상에는 기여하나, R@15 단일 지표로는 B5가 우위**. 다운스트림 E2E F1에서 재평가 필요.

---

## 8. 파일 위치

```
outputs/analysis/s06_bottleneck_b5_enriched/
├── batch_summary.json
├── objective_mismatch_B5E.png
├── cross_model_oversmoothing.png  (B5E 단독 — merged 폴더 참조)
├── cross_model_grad_ratio.png      (B5E 단독)
└── B5E/
    ├── oversmoothing_trajectory.png
    ├── tsne_query0.png
    ├── attention_entropy.png
    └── gradient_flow.png

outputs/analysis/s06_bottleneck_merged/
├── batch_summary.json              ← B0~B5 + B5E 통합
├── cross_model_oversmoothing.png   ← 7 모델 비교
├── cross_model_grad_ratio.png      ← 7 모델 비교 (B5E 빨간색)
└── cross_model_best_recall.png     ← Best R@15 비교 (B5E 빨간색, B5 cyan)
```

분석 스크립트:
- `src/analysis/gat_bottleneck_analysis_v2.py --models B5E --output_dir outputs/analysis/s06_bottleneck_b5_enriched`
- `src/analysis/merge_b5e_bottleneck.py`
