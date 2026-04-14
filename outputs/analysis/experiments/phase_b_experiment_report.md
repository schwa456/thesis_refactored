# Phase B 실험 결과 보고서

**실험 일시**: 2026-04-06  
**BIRD Benchmark**: 1,534 dev queries, 11 databases  
**공통 조건**: LocalPLMEncoder, SQL Generator 없음 (EX=0)

---

## 1. 실험 설계

| 실험 | Selector | Extractor | JoinKeys | Filter | 목적 |
|---|---|---|---|---|---|
| **B-0** (baseline) | VectorOnly (top_k=20) | PCST (threshold=0.1) | X | None | Local baseline 재현 |
| **B-1** | VectorOnly (top_k=20) | **AdaptivePCST** (p80) | X | None | Adaptive threshold 효과 |
| **B-2** | **Ensemble** (alpha=0.85) | PCST (threshold=0.1) | X | None | Score ensemble 효과 |
| **B-combined** | **Ensemble** (alpha=0.85) | **AdaptivePCST** | **O** | None | B-1+B-2+B-3 결합 |
| **B-4b** | **Ensemble** (alpha=0.85) | **AdaptivePCST** | **O** | **XiYanFilter** | Filter 추가 효과 |

---

## 2. 종합 결과

| 실험 | Recall | Precision | F1 | Avg Tables | Avg Cols |
|---|---|---|---|---|---|
| **B-0** (baseline) | 0.9489 | 0.1570 | 0.2694 | 5.7 | 47.7 |
| **B-1** (Adaptive) | 0.6719 | 0.3745 | 0.4809 | 2.6 | 9.5 |
| **B-2** (Ensemble) | 0.9679 | 0.1293 | 0.2282 | 6.7 | 56.5 |
| **B-combined** (no filter) | 0.7210 | 0.3471 | 0.4686 | 3.4 | 12.1 |
| **B-4b** (XiYanFilter) | 0.6244 | **0.7930** | **0.6987** | **1.7** | **3.2** |
| *(참고) Gold 평균* | — | — | — | *1.9* | *3.8* |

### F1 기준 순위: B-4b (0.70) > B-1 (0.48) > B-combined (0.47) > B-0 (0.27) > B-2 (0.23)

---

## 3. 핵심 발견

### 3.1 Adaptive PCST가 구조적 개선의 핵심 (B-1)

- **F1: +78.5% 상대 개선** (0.2694 → 0.4809)
- Precision: 0.1570 → 0.3745 (**2.4배 상승**)
- 선택 노드 수: 47.7 cols → 9.5 cols (gold 평균 3.8에 근접)
- **진단**: B-0의 고정 threshold(0.1)는 사실상 전체 스키마를 선택. Adaptive percentile이 per-query 분포에 맞게 조정

### 3.2 Score Ensemble은 고정 threshold와 궁합이 나쁨 (B-2)

- Ensemble이 Recall을 94.9% → 96.8%로 올렸지만
- 동일 고정 threshold(0.1)에서 더 많은 노드를 포함 → Precision 악화
- **교훈**: Ensemble은 반드시 Adaptive threshold와 함께 사용해야 효과적

### 3.3 XiYanFilter가 Precision-Recall 균형을 최적화 (B-4b)

B-combined → B-4b (XiYanFilter 추가):
- **Precision: 0.3471 → 0.7930 (+128.6% 상대 개선)**
- Recall: 0.7210 → 0.6244 (-9.7%p 하락)
- **F1: 0.4686 → 0.6987 (+49.1% 상대 개선)**
- 선택 컬럼: 12.1 → **3.2** (gold 평균 3.8보다 작음!)

**XiYanFilter의 역할**: DB value example을 포함한 프롬프트로 LLM이 정확한 컬럼 선택을 수행. 불필요한 컬럼을 효과적으로 제거하여 Precision을 크게 개선.

### 3.4 Recall 하락 분석

Filter 추가로 Recall이 0.72→0.62로 하락한 원인:
1. **Filter 입력의 Recall이 상한선** — B-combined에서 이미 놓친 28%는 복구 불가
2. **LLM false negative** — 필요한 컬럼을 불필요하다고 잘못 제거하는 케이스 존재
3. **개선 가능성**: 프롬프트에 "의심스러우면 포함" 지침 추가, Recall floor 보장 로직

---

## 4. DB별 상세 분석

### B-4b (XiYanFilter) — DB별 F1

| DB | Recall | Precision | F1 | 쿼리 수 |
|---|---|---|---|---|
| thrombosis_prediction | 0.7417 | 0.8490 | **0.7917** | 163 |
| superhero | 0.6909 | 0.9057 | **0.7839** | 129 |
| formula_1 | 0.7563 | 0.7987 | **0.7769** | 174 |
| codebase_community | 0.6874 | 0.8393 | **0.7558** | 186 |
| student_club | 0.6961 | 0.7908 | **0.7404** | 158 |
| financial | 0.6744 | 0.8076 | **0.7350** | 106 |
| european_football_2 | 0.5968 | 0.7119 | 0.6493 | 129 |
| card_games | 0.5139 | 0.7100 | 0.5962 | 191 |
| toxicology | 0.4387 | 0.8322 | 0.5745 | 145 |
| california_schools | 0.4806 | 0.6349 | 0.5471 | 89 |
| debit_card_specializing | 0.3961 | 0.7969 | 0.5292 | 64 |

### B-combined vs B-4b 비교 (Filter 효과)

| DB | B-combined F1 | B-4b F1 | 차이 | 분석 |
|---|---|---|---|---|
| thrombosis | 0.3998 | **0.7917** | **+39.2%p** | Filter가 over-selection 대폭 정리 |
| codebase_community | 0.3956 | **0.7558** | **+36.0%p** | 대규모 DB에서 Filter 효과 극대 |
| european_football | 0.2099 | **0.6493** | **+43.9%p** | 복잡한 스키마 정리 |
| formula_1 | 0.4535 | **0.7769** | **+32.3%p** | |
| card_games | 0.2255 | **0.5962** | **+37.1%p** | |
| superhero | 0.5856 | **0.7839** | **+19.8%p** | |
| student_club | 0.5499 | **0.7404** | **+19.1%p** | |
| financial | 0.5283 | **0.7350** | **+20.7%p** | |
| california_schools | 0.2365 | **0.5471** | **+31.1%p** | |
| toxicology | 0.5846 | 0.5745 | -1.0%p | 유일하게 하락 — Recall 손실 > Precision 이득 |
| debit_card | 0.5359 | 0.5292 | -0.7%p | 거의 동일 |

**11개 DB 중 9개에서 F1 개선, 2개에서 미세 하락** — Filter 효과가 전반적으로 매우 유효

---

## 5. Ablation 요약: 각 기법의 기여도

| 기법 | F1 기여 | 설명 |
|---|---|---|
| **Adaptive PCST (B-1)** | +0.21 (0.27→0.48) | 가장 핵심적 개선. 과잉 선택 방지 |
| **XiYanFilter (B-4b)** | +0.23 (0.47→0.70) | Precision을 gold 수준으로 끌어올림 |
| **Ensemble (B-combined)** | +0.05 (Recall 기준) | Recall 4.9%p 개선, F1 기여는 제한적 |
| **JoinKeys (B-3)** | 미측정 (ablation 필요) | Ensemble과 분리 측정 필요 |

### 누적 F1 개선 경로
```
B-0 (0.27) → [+Adaptive PCST] → B-1 (0.48) → [+Ensemble+JoinKeys] → B-combined (0.47) → [+XiYanFilter] → B-4b (0.70)
```

---

## 6. Recall 하락 완화를 위한 향후 개선안

1. **Percentile 하향 조정** (p80 → p70): 입력 Recall을 높여 Filter 전 상한선 확대
2. **Filter 프롬프트 개선**: "Uncertain? Include it." 보수적 지침 추가
3. **Recall Floor 로직**: Filter 후에도 seed top-K 노드 강제 포함
4. **2-iteration XiYan**: 1차에서 넓게 선택 후 2차에서 정제 (현재 1-iteration)
5. **Ensemble alpha 튜닝**: DB별 최적 alpha 탐색

---

## 7. Summary

```
experiment               recall  precision     f1  avg_tables  avg_cols
b0_raw_pcst_baseline     0.9489     0.1570 0.2694         5.7      47.7
b1_adaptive_pcst         0.6719     0.3745 0.4809         2.6       9.5
b2_ensemble              0.9679     0.1293 0.2282         6.7      56.5
b_combined               0.7210     0.3471 0.4686         3.4      12.1
b4_xiyan_filter          0.6244     0.7930 0.6987         1.7       3.2
```
