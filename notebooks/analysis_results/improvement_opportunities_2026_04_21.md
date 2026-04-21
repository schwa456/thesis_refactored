# 향후 성능 개선 기회 분석 (2026-04-21)

**작성 세션**: analyzer (post-hoc only)
**기반**: EXPERIMENT_HISTORY.md (1140줄), EXPERIMENT_CATALOG.md, `notebooks/analysis_results/*.md`, `outputs/summary_all.csv`, `outputs/experiments/abl/a05_filter_agentic/summary_all.csv`, `s06_bottleneck_*.md`
**출력 규격**: Recall / Precision / F1, 소수점 4자리

---

## 0. 현 리더보드 스냅샷

| 지표 | 실험 ID | Recall | Precision | F1 |
|------|---------|--------|-----------|-----|
| **Best F1** | `abl_ens_basic_xiyan` (구 2×2×2 #6, Ens+Basic+XiYan) | 0.8149 | 0.7597 | **0.7863** |
| Best P (Enriched) | Enriched builder + XiYan | — | **0.8147** | — |
| Best P (EdgePrize) | EdgePrize PCST + XiYan | — | 0.8139 | — |
| Recall 최고 (w/ filter) | `a05_02` ReflectionFilter 1iter | **0.7320** | 0.6833 | 0.7068 |
| Recall 최고 (no filter) | `a10` FKBackboneSteiner θ_r=0.5 | **0.8565** | 0.0732 | 0.1348 |
| GAT Val R@15 | `s06_a01_06` B5 (2-layer, Dual-Stream) | **0.6073** | — | — |

**관찰 1**: 최고 F1 0.7863 이후 2×2×2 8셀 기본 매트릭스는 포화. 추가 개선은 "기본 매트릭스 밖"에서 나올 수밖에 없음.
**관찰 2**: Reflection은 XiYan 대비 **Recall +0.0559 / Precision −0.0295 / F1 +0.0128** 변화를 실측했으나 (`a03_17 vs a05_02`), 아직 **최상위 anchor에 적용되지 않았음**.
**관찰 3**: Enriched 빌더는 Precision 천장(0.8147)을 세웠지만 Recall 회복 필터와 결합된 적이 없음.

---

## 1. 모듈별 병목 진단

### 1-1. Selector
- **GAT 기여도 dilution**: `selector_analysis.md`에서 α=0.85 (Cos 85% + GAT 15%)가 기본. Ensemble AUC 이득의 절반 이하만 top-k에 반영.
- **Val R@15 천장 0.6073 (B5)**: Enriched로 확장해도 −0.0057 (`s06_bottleneck_b5_enriched_extension.md`). 구조적 포화.
- **LDBO 결론**: train-internal DB diversity ≪ train↔dev gap → **encoder-level intervention** 필요. Head retrain만으로 oracle Dev AUC +0.048 (0.7067→0.7548), upstream 표현 자체가 병목.
- **NS-L1 (FK-reachability prior)**: 구현 완료 (`abl_sel_ns_l1_01` smoke verified 2026-04-20), e2e는 vLLM 서버 대기.

### 1-2. Extractor
- **Cost 계열 포화**: BO (`abl_ext_pcst_bo`) F1=0.6751 plateau. 방안 A·B 모두 종료.
- **FK-Steiner 미활용 upside**: θ_r=0.5에서 R=0.8565 (현 anchor 대비 +0.0416). Precision이 0.0732로 낮지만 **filter로 회수 가능한 recall pool**이 존재.
- **Offline sweep (`fk_steiner_percentile_sweep.md`)**: `all_cols p=95` F1=0.5287 (+0.0045 vs abs_anchor), `closed_cols p=50` R=0.8522. 적용 대기.

### 1-3. Filter
- **Reflection 1iter vs 3iter 수렴**: `a05_02` F1=0.7068 ≈ `a05_03` F1=0.7071 — 3iter diminishing return. 1iter로 충분.
- **Reflection이 XiYan 대비 recall 회복**: XiYan은 F1 천장을 세우는 대신 R을 깎음. Reflection은 반대 방향. **둘을 stacking하면 Precision×Recall 교차 이득** 가능성.
- **SymbolicVerifier (a05_19~22)**: config 4개 존재, 출력 비어 있음 — **예약만 되고 실행 안 됨**.
- **AdaptiveDepth / Bidirectional**: 인터페이스는 `filter_info` 표준화 후 준비 완료 (stage_infos, route_distribution 수집 가능), 실험 config 미작성.

### 1-4. Encoder / Builder
- **B-I/B-II/B-III builder infra 예약 (`abl_build_01~03`)**: RFM tokens, LineGraph, FK-reachability metadata — 아직 실행 전.
- **B5E paradox**: Enriched가 L0 spread는 개선(0.657→0.636)하나 L2 GAT에서 collapse 심화(0.920→0.978). Fusion이 뒤집지만 Val R@15은 −0.0057. **Enriched 단독 Selector/Extractor 효과는 측정되지 않음** (Filter 없이 E2E F1로도 측정 안 됨).

---

## 2. Tier-1 — 즉시 실행 권장 (High ROI, Low Cost)

### T1-A. ReflectionFilter on Enriched-anchor
- **규칙**: Enriched 빌더 + Basic/Adaptive PCST + **ReflectionFilter 1iter** (XiYan 대체)
- **근거**: Enriched는 precision 0.8147을 찍었지만 recall이 작아져 F1 0.6-후반대로 묶여 있음 (HISTORY §Phase C-extension). Reflection은 anchor 대비 R +0.0559. Enriched의 precision을 유지한 채 recall을 끌어올리면 F1 신기록 후보.
- **예상 F1**: 현 Enriched+XiYan F1을 `a03_17→a05_02` 관측 delta(+0.0128)로 transfer → 대략 **0.79~0.80 구간**.
- **실행 비용**: 기존 Enriched graph cache 재사용, filter만 교체. 1 config.

### T1-B. ReflectionFilter on 최고 F1 anchor (Ens+Basic+XiYan)
- **규칙**: `abl_ens_basic_xiyan` (현재 F1=0.7863)의 XiYan을 **XiYan→Reflection stacking** 또는 **Reflection 단독**으로 재실험
- **근거**: 현재 anchor는 Recall 0.8149로 추가 여유 있음. XiYan의 −Recall을 Reflection이 보강하면 F1 ≥ 0.7900 가능성. Stacking은 StackedFilter 기존 인프라 사용.
- **예상 F1**: Reflection 단독 0.77~0.79, Stack XiYan→Reflection 0.79~0.81.
- **주의**: StackedFilter의 `short_circuited_at` 텔레메트리로 Reflection이 실제로 트리거되는 비율 확인 필수.

### T1-C. FK-Backbone Steiner (θ_r=0.5) + XiYan
- **규칙**: `fk_backbone_steiner θ_r=0.5` (R=0.8565) 추출기 + **XiYan filter 부착** — 한 번도 해 본 적 없음
- **근거**: FK-Steiner의 R이 현 anchor보다 +0.0416 높음. XiYan이 평균적으로 P는 살리고 R은 −0.04 깎는다고 해도 최종 F1이 현재 0.7863을 상회할 가능성 존재. 저 R=0.8565는 0.5241(no-filter)과 P가 0.0732로 낮지만, filter 통과 후 P는 어차피 filter가 결정.
- **예상 F1**: 경험적 delta transfer 기준 **0.72~0.76** (Recall upside 크지만 가설 위에 가설).
- **리스크**: FK-Steiner의 후보 노드 수가 많아 XiYan 호출 cost 증가. 프로파일링 먼저.

### T1-D. NeurosymbolicL1 Selector (S-V) E2E
- **규칙**: `abl_sel_ns_l1_01` — smoke verified, **vLLM 서버 기동 후 E2E F1 측정**.
- **근거**: FK-reachability additive prior λ·reach_mask는 LDBO가 지적한 train↔dev gap 중 **OOD 컬럼에서 FK 경로를 통해 신호 복구**하는 가장 직접적인 보정. 효과가 있다면 Enriched보다 구조적.
- **예상 F1**: 불확실. Recall 우선 증가 → filter 조합 재탐색 필요.

---

## 3. Tier-2 — 중간 비용, 확실한 검증 가치

### T2-A. SymbolicVerifierFilter (a05_19~22) 실행
- 현재 config 4개 작성만 되어 있고 output 비어 있음. PCST 출력에 대한 symbolic schema validation은 **hallucination filtering의 최저비용 하한**이므로 baseline으로라도 필요.

### T2-B. Implicit FK Inference in Builder
- BIRD에서 **6.5% joins are non-declared** (column name match only)이라는 미측정 손실. Builder에서 column-name heuristic으로 implicit FK edge 추가 → Selector/Extractor에 자동 반영. `abl_build_02` (FK-reachability metadata) 확장으로 수렴 가능.

### T2-C. B5 체크포인트를 Full Pipeline에 투입
- B5 (R@15=0.6073)는 현재 Val Recall에서만 측정됨. **E2E F1 / precision 측정 안 됨**. 기존 2×2×2 최고 셀 (Ens+Basic+XiYan)의 GAT를 B5로 교체한 실험이 누락됨.

### T2-D. StackedFilter 다층 구성 (Reflection → XiYan → SymVerify)
- StackedFilter 텔레메트리(`short_circuited_at`)로 각 stage 기여도 분해 가능. recall 회복 → precision 선별 → symbolic 검증의 **파이프라인 단조성** 실증.

---

## 4. Tier-3 — 연구 확장 (High Upside, High Cost)

### T3-A. Domain-Adversarial GAT Training (DANN)
- LDBO가 확정적으로 가리킨 **encoder-level intervention**. train DB 분포를 discriminator로 제거해 dev OOD에서 Val R@15 위로 뚫기. B5 구조 위에 추가.

### T3-B. B5 + Query-Conditioned Projector + Enriched combined
- T6/T7 QCond + T5 Enriched + B5 구조 세 가지가 각자 다른 축에서 기여한 것으로 보였으나 **3-way combine이 없음**. B5E가 실패한 이유 중 "2-layer는 enriched에 부족"이 맞다면 QCond가 hop 역할 대체 가능성.

### T3-C. Builder B-II/B-III (LineGraph / FK-metadata) → S-II/S-III wiring
- 예약된 builder infra가 Selector 변형(S-II/S-III)과 연동되지 않은 채 남아 있음. 두 축을 함께 운용할 때의 시너지 미측정.

---

## 5. Tier-4 — Skip 권장 (근거 있음)

| 제안 | Skip 이유 |
|------|----------|
| 추가 edge-cost 튜닝 | 방안 A(score-driven) + 방안 B(BO) 모두 완료, plateau F1=0.6751 |
| Direct v1/v2 GAT | HISTORY에서 closed 상태 |
| GPT-4o-mini filter | a05_14/15/17 대비 Qwen3-Coder-30B이 R/P 양쪽 우위로 확인 |
| `per_table` percentile | `all_cols` 대비 FK-Steiner sweep에서 열세 (`fk_steiner_percentile_sweep.md`) |
| ProductCost 추가 BO | a09 sweep 이미 전수, ComponentAware가 대체 |

---

## 6. F1 기대 delta 요약

| 제안 | Anchor 실측 F1 | 관측된 transfer delta | 예상 신 F1 |
|------|---------------|----------------------|------------|
| T1-A Reflection × Enriched | ~0.67 (Enriched+XiYan 추정) | +0.0128 (recall-gain side) | **0.78~0.80** |
| T1-B Reflection on best anchor | 0.7863 | +0.01~+0.03 (upside) / −0.02 (downside) | **0.77~0.81** |
| T1-C FK-Steiner(θ=0.5) + XiYan | 0.5241 (no-filter) | XiYan transfer +0.25 (Basic→Basic+XiYan 실측) | **0.72~0.76** (넓은 구간) |
| T1-D NS-L1 E2E | — (미측정) | FK-prior 기여 추정 +Recall | **불확실, recall-breaking 기대** |
| T2-C B5 + best filter | 0.7863 | Val R@15 +0.0335 → E2E +α | **0.78~0.80** |

---

## 7. 실행 순서 제안 (루트 세션에 에스컬레이션할 내용)

1. **T1-A (Reflection × Enriched)** — 1 config, 가장 저비용, Precision 천장 유지한 채 Recall 회복 검증.
2. **T1-B (Reflection on best anchor)** — 2 config (단독 / stack), 신기록 가장 가능성 높음.
3. **T1-D (NS-L1 E2E)** — vLLM 기동 직후 즉시 실행 (이미 대기 중).
4. **T1-C (FK-Steiner θ=0.5 + XiYan)** — LLM call cost 사전 프로파일링.
5. **T2-A (SymVerify a05_19~22)** — baseline filler.
6. **T2-C (B5 E2E)** — GAT 체크포인트 교체.
7. **T2-B (Implicit FK builder)** — 6.5% BIRD structural miss 보정.
8. **T3-A (DANN)** — 장기 연구 축.

---

## 8. 근거 링크

- 리더보드 / 2×2×2: [EXPERIMENT_HISTORY.md](../../EXPERIMENT_HISTORY.md) Phase C
- GAT 병목: [s06_bottleneck_comparison.md](s06_bottleneck_comparison.md), [s06_bottleneck_b5_enriched_extension.md](s06_bottleneck_b5_enriched_extension.md)
- FK-Steiner sweep: [fk_steiner_percentile_sweep.md](fk_steiner_percentile_sweep.md)
- Selector LDBO / Head retrain: [selector_analysis.md](selector_analysis.md), HISTORY §7-2·§7-3
- Filter Reflection 실측: `outputs/experiments/abl/a05_filter_agentic/summary_all.csv` (a05_02, a05_03)
- Advisor 5 ideas: [advisor_meeting_ideas_analysis.md](advisor_meeting_ideas_analysis.md)

---

## 9. 메모

- 이 분석은 **analyzer 세션 산출물**로, 새 실험 실행 / config 생성 / EXPERIMENT_*.md 수정은 **루트 세션 책임**.
- 루트 CLAUDE.md의 "#8 E+A+X (F1=0.4936)" 표기와 HISTORY의 "Best F1=0.7863 (#6)" 간 불일치가 있음 → **루트 세션에 수정 요청 필요**.
- 모든 F1 추정값은 기존 delta transfer에 기반한 단순 외삽. 실측 시 ±0.02 범위 오차 감수.
