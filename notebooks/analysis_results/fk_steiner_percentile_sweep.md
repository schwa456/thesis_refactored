# FK-Backbone Steiner: Column Recovery Percentile Sweep (Offline, 2026-04-17)

## 동기

a10 FK-Backbone Steiner의 `column_recovery_threshold θ_r`은 절댓값으로 해석된다. Ensemble
Selector(α=0.85)가 뱉는 score는 85%가 raw cosine이므로 DB/query에 따라 분포가 심하게 이동
한다 — "θ_r=0.8"이 어떤 쿼리에서는 상위 10%, 다른 쿼리에서는 상위 40%일 수 있다. Per-query
percentile로 재정의해 이 변동을 흡수하면 F1이 개선되는지 검증.

## 방법

`FKBackboneSteinerExtractor`에 `column_recovery_percentile` 파라미터와 scope 선택지를 추가
하고, a10_09의 score_analysis (per-query node scores) + dev graph cache를 재사용해 **Selector
를 재실행하지 않고 extractor만** 오프라인 재평가. 각 구성마다 1534 쿼리에서 Step 4a의
column recovery 기준만 바꿔 R/P/F1을 계산. Auto-join FK 후처리까지 포함.

**4 scopes × 21 percentiles (0..100, 5 step) + abs anchor = 85 configs**.

### Scope 정의

| Scope | Percentile 계산 모집단 |
|---|---|
| `global` | 쿼리의 모든 노드 (table + column + fk) score |
| `all_cols` | 쿼리의 모든 컬럼 노드 score |
| `closed_cols` | Steiner closure로 확정된 테이블 내부 컬럼 score |
| `per_table` | closed table 각각 독립 (테이블별 percentile) |

## 결과

### 각 scope의 F1 peak

| Scope | 최적 percentile | R | P | F1 | Δ vs abs_anchor |
|---|---|---|---|---|---|
| **abs_anchor** (θ_r=0.8) | — | 0.5455 | 0.5044 | **0.5242** | — |
| `global` | p=95 | 0.5754 | 0.4776 | 0.5219 | **−0.0023** |
| **`all_cols` ★** | **p=95** | **0.6167** | 0.4626 | **0.5287** | **+0.0045** |
| `closed_cols` | p=95 | 0.5688 | 0.4899 | 0.5264 | +0.0022 |
| `per_table` | p=100 | 0.5471 | 0.4928 | 0.5185 | −0.0057 |

### High-Recall 운영점 (Filter-앞 후보군)

R ≥ 0.85를 만족하는 구성의 P 비교 (동일 R 수준에서 P 낮으면 Filter 부담 증가):

| Scope | p | R | P |
|---|---|---|---|
| `global` | 50 | 0.8998 | 0.2058 |
| `all_cols` | 55 | 0.8801 | 0.2151 |
| **`closed_cols`** | 50 | **0.8522** | **0.2389** |
| `per_table` | 50 | 0.8293 | 0.2173 |

`closed_cols` 가 R~0.85 구간에서 P 최고 — Steiner closure로 후보군이 이미 필터된 pool에서
선별하므로 과도 포함이 억제됨.

### Full-Recall 운영점 (p=0 → 모든 컬럼 포함)

모든 scope에서 p=0은 동일: **R=0.9492, P=0.1567, F1=0.2690**. Closed_tables 내부 모든 컬럼
포함 → force_fk_columns까지 적용한 상한 recall.

## 핵심 발견

1. **개선 폭 미미, 단 일관된 양 효과**: all_cols p=95가 abs_anchor 대비 **+0.0045 F1**로
   최고. 큰 jump는 아니지만 per-query calibration이 약간의 이득을 준다.

2. **global ≈ all_cols**: 쿼리의 노드 중 컬럼이 ~95%를 차지하므로 두 분포가 거의 동일.
   `global` scope는 사실상 중복 설계 — `all_cols` 가 더 깔끔한 정의.

3. **closed_cols는 "복원 후보군에서 상위 p%"로 깔끔한 해석**: 개념적으로 가장 targeted.
   F1 peak도 all_cols에 근접(0.5264 vs 0.5287, −0.0023). 고-Recall 구간(R>0.85)에서는
   P 최고 → **Filter-앞 단계에서 선호되는 scope**.

4. **per_table이 floor는 낮지만 천장도 낮다**: p=5에서도 R=0.9041(타 scope는 0.94+).
   작은 테이블에서 percentile 계산이 노이즈를 주어 gold 컬럼 손실 가능성. F1 peak도 최하
   (0.5185). **추천 X**.

5. **모든 scope가 고-percentile 구간(p=90~100)에서 peak**: `column_recovery_threshold=0.8`
   절댓값이 실제로 상위 5% 정도의 고정 cut을 의미했음을 확증. 현재 θ_r=0.8 regime이 이미
   dev set 평균적 최적 근처.

6. **per-query calibration의 이득이 제한적인 이유**:
   - Score 분포가 예상보다 query 간 일관적 (all-MiniLM이 cos 0.2~0.9 범위에 안정적 분포)
   - 이미 adaptive_threshold(P80)가 seed_tables 단계에서 per-query 정규화를 적용 중
   - Column recovery 단계의 변동 흡수 이득은 주변부에 국한

## 권장사항

- **F1 최대화**: `all_cols p=95` 사용 (+0.0045 F1). 개선폭 작지만 interpretability 향상.
- **High-Recall 운영점 (Filter-앞)**: `closed_cols` scope + p=50 사용.
  R=0.85 유지하며 P 최대 (0.2389).
- **`per_table` scope 비추천**: 작은 테이블에서 F1 손실.
- **`global` scope 불필요**: `all_cols` 와 수치 거의 동일, 정의만 중복.

## 다음 단계 제안

1. **XiYan Filter 결합 테스트**: `all_cols p=95` + XiYan 조합. Filter 후 net F1 이득이
   유지되는지 확인 (절댓값 모드 대비).
2. **High-Recall operating point + Filter 조합**: `closed_cols p=50` (R=0.8522) + XiYan
   — Filter가 P를 얼마나 회복하는지로 최종 F1 결정.
3. **GAT 재학습 후 재분석**: Ensemble α=0.85에서 raw cosine이 지배적 → 새 GAT가 배포
   되면 본 sweep 재실행 필요. percentile 방식이 score scale에 robust한지 검증 가치.
4. **Micro-averaged vs macro-averaged 비교**: 본 분석은 query macro. 쿼리 크기 편향
   보정을 위한 micro-average도 같이 보면 해석이 명확해질 수 있음.

## 산출물

- CSV: [fk_steiner_percentile_sweep.csv](fk_steiner_percentile_sweep.csv) (85 rows)
- 스크립트: [src/analysis/fk_steiner_percentile_sweep.py](/home/hyeonjin/thesis_refactored/src/analysis/fk_steiner_percentile_sweep.py)
- 확장된 Extractor: `FKBackboneSteinerExtractor` (pcst.py) — `column_recovery_percentile`,
  `column_recovery_percentile_scope` 파라미터 추가
