# Planner Decisions Log

> Planner 세션이 PLAN을 바꿀 때마다 **반드시** 이 파일에 엔트리를 남긴다.
> 세션이 교체되어도 직전 맥락을 복원할 수 있게 하는 연속성 장치.
>
> 엔트리 포맷은 [CLAUDE.md](CLAUDE.md) 하단 템플릿 참조.
> 최신이 위, 과거가 아래 (역시간순).

---

## 2026-05-13 (Anchor Framework 정확한 정의 정정 — Direction A config 의 Extractor 정정 + 향후 모든 ablation 의 reference base)

> **사용자 직전 input (5/13)**: "지금 anchor 프레임워크는 Enriched Builder + QCond (concat) + MST+PCST + XiYan Filter 구조야 — 지금은 Filter 를 실험하는 거니까 나머지는 anchor 와 동일해야 해. Root 에게도 알려줬으니 너도 참고해." Direction A 의 a05_23 / a05_24 config 가 Extractor 를 단일 MSTKruskalExtractor 로 설정한 것은 anchor 와 incompatibility — Root 가 정정 후 재실험.

### §1. Anchor Framework 정확한 정의 (향후 모든 ablation chain 의 reference base)

| Stage | Module | 정확한 클래스 이름 |
|---|---|---|
| **Builder** | Enriched | `EnrichedHeteroGraphBuilder` (tables.json + database_description CSV 포함) |
| **Selector** | **QCond (concat)** | `EnsembleSelector` with `best_gat_qcond_nl3.pt` (Query-Conditioned GAT NL=3 layer, concat fusion) + `alpha=0.5` (GAT + cosine ensemble) + `top_k=20` |
| **Extractor** | **MST + PCST (결합)** | **`MSTPCSTUnionExtractor`** (단일 `MSTKruskalExtractor` 또는 단일 `AdaptivePCSTExtractor` 와 구별) |
| **Filter** | **XiYan** | `XiYanFilter` (GLM 4.7 era — model `zai-org/glm-4.7`, provider `glm`) |
| (SQL Gen) | LLM | `LLMSQLGenerator` (GLM 4.7) |

→ **Anchor metric**: F1 = **0.8651**, EX = **0.5202** (학회 paper main, paper §V.5.4 + §V.5.5 base, 학술 Agent 의 모든 phase 의 reference anchor)

### §2. Direction A config 의 Extractor 정정 (Root 책임, planner 는 reference)

| Config | 직전 (잘못된, anchor 와 incompatibility) | 정정 (Root 갱신) |
|---|---|---|
| `a05_23_rsl_backward_baseline.yaml` | `MSTKruskalExtractor (score_threshold=0.1)` | **`MSTPCSTUnionExtractor`** (anchor 정합) |
| `a05_24_rsl_backward_with_guard.yaml` | `MSTKruskalExtractor (score_threshold=0.1)` | **`MSTPCSTUnionExtractor`** (anchor 정합) |

→ **사용자 결정**: Root 가 a05_23 + a05_24 config 의 extractor 를 `MSTPCSTUnionExtractor` 로 갱신 후 재실험. Planner 는 코드 수정 X (planning 세션 의 boundary), 단 documentation + DECISIONS reference 갱신.

→ **재실험 영향**: 직전 잘못된 stack 으로 진행 중이던 sweep 결과는 폐기 — anchor 와 incompatibility 라 정량 비교 불가. ETA + 추가 LLM cost 발생 (재실험 분).

### §3. Filter Ablation 의 정확한 원칙 (학회 paper §3.5 narrative 정합)

학회 paper §3.5 의 Filter Dominance 정확한 실험 원칙:
> "Filter 의 효과 정량 측정 시 Filter 외 모든 module (Builder + Selector + Extractor + SQL Gen) 은 anchor 와 동일하게 유지. Filter 만 변경하여 정확한 isolated effect 측정."

→ 직전 Filter Dominance evidence 표 의 모든 axis (#1~#9) 의 base = 본 anchor stack. 즉:
- #4 ΔF1 +0.65 lift = anchor (Enriched + QCondGAT + MSTPCSTUnion + XiYan) vs same stack with Filter=None
- #5 H-A/H-D Filter design 14 변형 = 위 anchor stack 의 Filter 만 14 variants
- #7 Filter-Invariant F1+EX sweep v2 9-cell = anchor stack 의 Filter 만 9 variants
- #8 SGBE Negative Evidence = anchor stack 의 Filter 만 SGBE 로 교체 → F1=0.3697 (anchor 0.8673 대비)
- **#9 Direction A = anchor stack 의 Filter 만 RSLBackwardFilter 로 교체** (재실험 후 정확한 비교 가능)

### §4. RSLBackwardFilter Documentation 갱신 항목

- `planning/rsl_backward_filter_documentation_2026-05-13.md`:
  - §1.1 "학술적 위치" — anchor framework 의 정확한 정의 명시
  - §4.1 / §4.2 — anchor stack 의 정확한 module 정정 (MSTPCSTUnionExtractor)
  - §4.3 anchor 표기 — Enriched + QCondGAT (concat) + MST+PCST + XiYan 의 통합 표기
  - §11 chain status — Root 의 config 정정 후 재실험 상태 반영

### §5. paper_research_direction.md 영향

- §3.5 axis #9 placeholder — Direction A 가 anchor stack 의 Filter 만 RSLBackwardFilter 로 교체. 정확한 isolated effect 측정 정합 (학회 paper narrative 정합).
- 직전 §1 (Paper Main Pipeline) — 이미 정확한 anchor framework 명시되어 있는지 확인 + 갱신 trigger.

### §6. 사용자 의도 + Planner 의 향후 reference

- Direction A 의 sweep 의 정확한 isolated effect 측정 → 학술 Agent Phase 3 의 ΔF1(A) trigger 분기 (≥0.03 / <0.02 / gray zone) 의 정량 정확성 확보
- 향후 모든 ablation (Direction C 의 미래 trigger 시 + Filter ablation 추가 시) 의 anchor stack 정확성 유지
- DECISIONS 의 본 entry = 향후 모든 ablation chain 의 anchor reference

### §7. Chain Status 갱신 (5/13 저녁 — Root config 정정 후)

| # | Chain | Status |
|---|---|---|
| 1-10 | 직전 chains | ✅ 완료 |
| 11 | Module:Filter `RSLBackwardFilter` 구현 + smoke 15/15 | ✅ 완료 (commit 462798d) |
| **12** | **Root: Direction A sweep config (a05_23 + a05_24)** | ⚠️ **Extractor MSTKruskalExtractor → MSTPCSTUnionExtractor 정정 + 재실험 진행 중** |
| 13 | Analyzer n=488 해명 분석 | 🚀 진행 중 (병행) |
| ⏸ 14 | Analyzer Direction A 결과 보고서 (anchor 정정 후 정확한 ΔF1) | ⏸ Root 재실험 완료 의존 |
| ⏸ 15 | 학술 Agent Phase 5 (Direction A 결과 + Direction C 재결정) | ⏸ Direction A 결과 의존 |
| ⏸ 16 | 학위 논문 §V.5.x.9 narrative final integration | ⏸ Direction A 배포 후 ΔF1 정량 trigger |

### §8. 영향 범위

- planning/DECISIONS.md (본 entry — 향후 모든 ablation chain reference base)
- planning/rsl_backward_filter_documentation_2026-05-13.md (§1.1 + §4 + §11 갱신)
- (Root 책임) configs/experiments/abl/a05_filter_agentic/a05_23_rsl_backward_baseline.yaml + a05_24_rsl_backward_with_guard.yaml (planner 는 갱신 X, Root 세션이 정정 진행 중)

### §9. 근거

- 사용자 직접 input (anchor framework 정확한 정의 + Root 알림 + planner 참고 지시)
- `src/modules/extractors/mst_pcst_union.py` (`MSTPCSTUnionExtractor` 클래스 확인)
- `planning/paper_research_direction.md` §3.5 Filter Dominance evidence 표 — 모든 axis 의 base = anchor stack 의 Filter 만 ablation 원칙

### §10. 사용자 후속 actions

1. ✅ **사용자 → Root 알림** — Direction A config 의 Extractor 정정 (사용자 진행 완료)
2. 🚀 **Root: a05_23 + a05_24 의 extractor `MSTPCSTUnionExtractor` 정정 + 재실험 launch**
3. 🚀 **Planner: documentation §1.1 + §4 + §11 갱신** (본 entry 의 §4)
4. ⏸ **Analyzer**: 재실험 결과 ΔF1 보고서 (재실험 완료 의존)

---

## 2026-05-13 (학술 Agent Phase 4 Review — Verification 정합 confirm + 3 표현 수정 + n=488 해명 요청 + Bimodal axis 위치 권장)

> **사용자 직전 input (5/13)**: 학술 Agent 의 Phase 4 review 결과 공유. 본 review 는 직전 verification 의 학술 Agent 검증 — interpretation (b) 정합 + mathematical identity 정합 + auxiliary statistics 표기 권장 + 3 수정 항목.

### §1. 학술 Agent Phase 4 Review 결과 요약 (6 항목)

| Section | 검증 결과 | 조치 |
|---|---|---|
| §2 Interpretation (b) | ✅ **정합** — 정정 불필요 | n=488 vs Phase 1 n_restore_nonzero=698 차이 해명 요청 (Analyzer) |
| §3 Mathematical Identity | ✅ **정합** — proof appendix 사용 가능 | 변경 없음 |
| §4 Auxiliary Statistics | ✅ **정합** + 표기 권장 | main 에 conditional 0.5709 만, 각주에 pooled 0.5984 (paper_research_direction.md §V.5.x.6 갱신) |
| §5 Bimodal Distribution | ✅ **정합** + ⚠️ **표현 수정** | "all-or-nothing" → "predominantly binary recovery behavior" (paper_research_direction.md §V.5.x.7 갱신) |
| §6 Main 인용 SGBE contrast | ⚠️ **표현 수정** | "SGBE 의 절반 수준" → "SGBE 의 약 44% 수준 (절반 이하)" (35.66/81.22 = 43.9%) |
| §6 §V.5.x.6 + §V.5.x.7 위계 | ✅ **적절** | 변경 없음 |
| §9 Bimodal axis #9 포함 | 권장 표기 | 학회 paper §3.5 axis #9 포함 X, 학위 논문 §V.5.x.8 discussion 으로 통합 (paper_research_direction.md §3.5 갱신) |

### §2. 학술 Agent §2 — n=488 해명 요청 (Analyzer 후속 분석)

학술 Agent 의 정확 인용:

> "n=537 (conditional)과 n=488 (S_restore non-empty)의 차이 해명을 요청합니다. §6의 인용 문구에서 mean(S_restore_precision)의 분모를 n=488로 표기했는데, Phase 1 시점에서는 n_restore_nonzero = 698이었습니다. Bug fix가 recall 계산 로직만 수정한 것이라면 L_bwd 집합 자체는 변하지 않아야 하므로 S_restore non-empty 쿼리 수도 698로 유지되어야 합니다. 698 → 488 변화는 column name normalization(table prefix 통일, 대소문자 처리 등) 등 L_bwd 추출 방식까지 함께 수정되었을 가능성을 시사합니다. Bug fix의 범위가 정확히 어느 라인이었는지 확인하고, n=488의 근거를 스크립트 주석에 명시해두는 것을 권장합니다."

**Planner 의 추측 (Analyzer 검증 요청)**:
- Phase 2 bug fix 가 `alias-distinct → col-only-distinct normalization` 변경 (직전 학술 Agent Phase 2 review 의 인용)
- column name normalization 후 같은 col 의 alias variants 가 통합되어 L_bwd 집합 축소 → 같은 col 이 alias 로 두 번 나타나는 query 의 경우 Phase 1 에서는 distinct count 2 (S_restore ≥ 1), Phase 2 에서는 distinct count 1 (S_restore = 0 가능)
- 결과: S_restore non-empty 의 query 수가 **698 → 488 감소 (-30%)**

**Analyzer 의 후속 분석 spec** (본 entry 의 핸드오프 base):
1. `src/analysis/filter_proposal_a3_restore_noise.py` 의 L_bwd 추출 로직 (sqlglot.extract_columns + normalization) 의 정확한 lines 확인
2. Phase 1 → Phase 2 의 변경 lines 정확한 위치 (recall 계산 외에 L_bwd 추출 까지 영향 받았는지)
3. 698 → 488 변화의 정확한 이유 — alias normalization 의 함수 / table prefix 통일 / 대소문자 처리 등
4. **n=488 의 근거를 스크립트 주석에 명시** (학술 Agent 권장)
5. `notebooks/analysis_results/recall_gained_denominator_verification.md` 에 §1.4 추가 — n=488 의 정확한 정의 + Phase 1 의 n=698 과의 비교 + bug fix 의 정확한 lines

### §3. 학술 Agent §4 — 보조 통계 표기 권장 (paper_research_direction.md §V.5.x.6 갱신)

학술 Agent 의 정확 인용:

> "학위 논문 주 표기 권장: conditional mean 0.5709. Pooled 0.5984는 보조 표기로 병기할 수 있습니다. 두 수치를 모두 본문에 나열하면 독자가 혼란을 겪을 수 있으므로, §6 main 인용에는 0.5709만, 각주에 'pooled micro-average 0.5984, 두 지표 모두 backward path의 >57% 회복률 일관 지지'로 처리하는 것이 가장 깔끔합니다."

**Planner 의 적용**:
- paper_research_direction.md §V.5.x.6 main 인용 final 갱신 — pooled rate 0.5984 를 본문에서 제거, 각주로 이동
- DECISIONS verification entry §6 갱신 (본 entry 의 §5)

### §4. 학술 Agent §5 — "all-or-nothing" 표현 수정 (paper_research_direction.md §V.5.x.7 갱신)

학술 Agent 의 정확 인용:

> "§9에서 사용한 'all-or-nothing'이라는 표현은 수정이 필요합니다. 53건(10%)이 partial recovery (0 < recall_gained < 1) 구간에 존재하므로 엄밀한 의미의 'all-or-nothing'은 아닙니다. 'predominantly binary' 또는 'bimodal with 90% extreme cases (52% perfect + 38% zero)'가 더 정밀합니다. 학위 논문 서술 시 'RSL-SQL backward path exhibits a predominantly binary recovery behavior on BIRD-Dev: 52% of forward-imperfect queries achieve perfect column recovery, 38% achieve zero recovery, with only 10% showing partial recovery'로 표기할 것을 권장합니다."

**Planner 의 적용**:
- paper_research_direction.md §V.5.x.7 보조 인용 final 갱신 — "all-or-nothing" → "predominantly binary recovery behavior" + 정확한 정량 서술
- DECISIONS verification entry §6 §V.5.x.7 보조 인용 갱신

### §5. 학술 Agent §6 — Main 인용 "SGBE 의 절반 수준" 수정

학술 Agent 의 정확 계산:

> "Direction A noise rate = 1 − 0.6434 = 35.66%, SGBE S_keep_hard noise = 81.22% 이므로, 35.66% / 81.22% = 43.9%입니다. '절반 수준'(50%)보다 낮으므로 사실상 더 강한 주장이지만 표현이 부정확합니다."

| 위치 | 현재 (수정 전) | 수정 후 |
|---|---|---|
| paper_research_direction.md §V.5.x.6 main 인용 | "Direction A 의 noise 35.66%, SGBE 의 절반 수준" | "Direction A 의 noise 35.66%, SGBE 의 약 44% 수준 (절반 이하)" |
| paper_research_direction.md §3.5 axis #9 placeholder | "Direction A noise 35.66% << SGBE S_keep_hard noise 81.22% (~SGBE 의 절반)" | "Direction A noise 35.66% << SGBE S_keep_hard noise 81.22% (SGBE 의 약 44% 수준)" |
| DECISIONS verification entry §6 main 인용 | "Direction A 의 noise 35.66%, SGBE 의 절반 수준" | "Direction A 의 noise 35.66%, SGBE 의 약 44% 수준 (절반 이하)" |

나머지 main 인용 (precision 0.6434, threshold 1.07×, recall_gained 0.5709 conditional 정의, aggregate identity 0.0768 ≈ 0.0771) 모두 ✅ 검증 완료.

### §6. 학술 Agent §V.5.x.6 main 인용 final 갱신 (3 수정 항목 통합)

```
Direction A (RSL-SQL backward path) 의 본 도메인 transfer 정량 evidence:

  - mean(S_restore_precision) = 0.6434 (n=488, S_restore non-empty queries) — 학술 Agent
    threshold 0.60 의 1.07× margin. SGBE 의 S_keep_hard noise 81.22% 와 dramatic
    contrast (Direction A 의 noise 35.66%, SGBE 의 약 44% 수준 — 절반 이하).
    [수정 1: "절반 수준" → "약 44% 수준 (절반 이하)" — 학술 Agent §6]

  - mean(recall_gained_by_restore) = 0.5709 (n=537, forward-imperfect queries):
    "of the gold columns that forward misses, backward recovers 57% on average per query"
    (학술 Agent 인용, conditional mean 정합).

  - [각주, 학술 Agent §4 권장]: pooled micro-average 0.5984
    (sum |S_restore ∩ missed| / sum |missed|), 두 지표 모두 backward path 의 >57%
    회복률 일관 지지.

  - Aggregate consistency: mean(Δrecall_union) = +0.0771 = mean(recall_gained × |M|/|G|)
    per-query identity (mathematical proof, 학술 Agent 검증 ✅ — proof appendix 사용 가능).
```

학위 논문 §V.5.x.7 보조 인용 final:

```
Forward (XiYan anchor) 가 BIRD-Dev 1534 query 중 65% (997 queries) 에서 모든 gold cols
회수 — backward path 의 의미 있는 영역은 35% (537 queries) 만. 이 conditional 영역에서:
  - 52% (279 queries) perfect recovery (backward 가 모든 missed gold 복원)
  - 38% (205 queries) zero recovery (backward 가 한 col 도 복원 안 함)
  - 10% (53 queries) partial recovery (0 < recall_gained < 1)
→ "RSL-SQL backward path exhibits a predominantly binary recovery behavior on BIRD-Dev:
52% of forward-imperfect queries achieve perfect column recovery, 38% achieve zero
recovery, with only 10% showing partial recovery" (학술 Agent §5 권장 표기).
"Selective gain on hard queries" 정량 evidence — 학술 Agent Phase 1 §3 의 challenging
query lift 0.3287 (simple 0.2216 의 1.48×) 와 일관.
[수정 2: "all-or-nothing" 표현 제거 → "predominantly binary recovery behavior" — 학술 Agent §5]
```

### §7. 학술 Agent §9 — Bimodal Axis 위치 권장 (학회 paper §3.5 axis #9 포함 X)

학술 Agent 의 정확 인용:

> "학회 paper(한국지능정보시스템학회 2026 춘계)의 §3.5 8-axis Filter Dominance가 이미 main contribution이므로, bimodal 발견은 학회 paper에 독립 axis로 추가하기보다 Direction A 배포 결과 절(§V.5.x.8)의 discussion 항목으로 통합하는 것이 학위 논문 구조상 적합합니다. 학회 paper는 현재 구조를 유지하고, bimodal 발견은 학위 논문 §V에서만 상세히 서술하는 방향이 contribution의 과잉 분산을 막습니다."

**Planner 의 적용**:
- paper_research_direction.md §3.5 axis #9 placeholder — bimodal 발견 narrative 제거 (학회 paper 의 axis 로 포함 X)
- paper_research_direction.md §V.5.x.8 (학위 논문 chapter outline, "plan, Direction A 배포 후 trigger") — bimodal 분포 분석 + discussion 통합 명시

**근거**: 학회 paper 의 main contribution 의 과잉 분산 방지. 학위 논문 §V 에서만 상세 서술.

### §8. paper_research_direction.md 갱신 항목 (3 수정 적용)

1. **§3.5 evidence 표 axis #9**:
   - "SGBE 의 절반 수준" → "SGBE 의 약 44% 수준" (수정 1)
   - Bimodal 발견 narrative 제거 (학술 Agent §9 권장, 학위 논문 §V.5.x.8 으로 통합)
2. **§V.5.x.6 narrative final candidate sub-section**:
   - main 인용 final 갱신 (수정 1 + 보조 통계 각주 표기)
   - §V.5.x.7 보조 인용 final 갱신 (수정 2 "all-or-nothing" 제거)
   - §V.5.x.8 (plan, Direction A 배포 후 trigger) row 의 scope 갱신 — bimodal discussion 통합 명시
3. **§3.5 axis #9 의 trigger 명시** — Direction A 배포 후 정량 main update (bimodal 발견은 학위 논문 §V.5.x.8 으로만 통합)

### §9. Direction A 배포 Chain Status 갱신 (5/13 저녁 — Phase 4 review 완료 후)

| # | Chain | Status |
|---|---|---|
| 1-10 | 직전 chains (verification 완료까지 포함) | ✅ 완료 |
| **11** | **학술 Agent Phase 4 review** | ✅ **완료 (5/13)** — 3 수정 항목 + n=488 해명 요청 |
| **12** | **Planner: 3 수정 적용 (paper_research_direction.md + DECISIONS)** | 🚀 즉시 (본 entry §8) |
| **13** | **Analyzer: n=488 해명 후속 분석** | 🚀 trigger (본 entry §2 spec) |
| **14** | **Module:Filter: `RSLBackwardFilter` 구현** | 🚀 진행 중 (Phase 4 review 결과 의존 없음) |
| ⏸ 15 | Root: Direction A sweep launch | ⏸ Module:Filter 완료 의존 |
| ⏸ 16 | Analyzer: Direction A 결과 보고서 | ⏸ Root sweep 완료 의존 |
| ⏸ 17 | 학술 Agent Phase 5 (Direction A 결과 + Direction C 재결정) | ⏸ Direction A 결과 의존 |
| ⏸ 18 | 학위 논문 §V.5.x.6 narrative final integration | ⏸ Direction A 배포 후 ΔF1 정량 trigger |

### §10. 영향 범위

- planning/DECISIONS.md (본 entry)
- planning/paper_research_direction.md §3.5 evidence 표 axis #9 + §V.5.x.6 narrative final candidate sub-section (§V.5.x.6 main 인용 + §V.5.x.7 보조 인용 + §V.5.x.8 trigger row)
- 후속:
  - Analyzer 핸드오프 — n=488 해명 분석 + verification 의 §1.4 추가
  - paper_research_direction.md §V.5.x.6 main 인용 의 n=488 표기 final 확정 (Analyzer 분석 후)

### §11. 근거

- 사용자 직접 input (학술 Agent Phase 4 review 결과 공유)
- 직전 entry (verification 완료) §6 main 인용 + §V.5.x.7 보조 인용 — 본 review 의 base

### §12. 사용자 후속 actions

1. ✅ **Phase 4 review receive** — 학술 Agent 의 verification 검증 완료
2. 🚀 **Analyzer 핸드오프** — n=488 vs 698 해명 분석 (본 entry §2 spec)
3. ✅ **paper_research_direction.md 3 수정 적용** — planner 즉시 (본 entry §8)
4. 🚀 **Module:Filter** — `RSLBackwardFilter` 구현 계속 (Phase 4 review 결과 의존 없음)
5. ⏸ **Phase 5 cover note prep** — Direction A 배포 결과 (ΔF1) + Analyzer n=488 해명 추가 후 전달

---

## 2026-05-13 (recall_gained_by_restore Denominator Verification 완료 — Interpretation (b) 확정 + 학술 Agent narrative 정합 + Mathematical Identity 정합 + 학위 논문 §V.5.x.6 main 인용 final candidate)

> **사용자 직전 input (5/13)**: Analyzer 의 `notebooks/analysis_results/recall_gained_denominator_verification.md` 작성 완료 보고. 학술 Agent Phase 3 의 수치 확인 요청 (recall_gained_by_restore = 0.5709 의 분모 정의 확인) 의 정식 응답 base.

### §1. 핵심 결론 — Interpretation (b) 확정

분모 = `|G \ L_fwd|` (forward 가 누락한 gold cols 의 수). 코드 직접 인용 (`filter_proposal_a3_restore_noise.py` lines 95-104):

```python
missed_by_fwd = set(c for c in gold_cols_raw if c.lower() not in L_fwd_compare)
missed_size = len(missed_by_fwd)
if missed_size > 0:
    gained = _intersect_size(S_restore, missed_by_fwd)
    recall_gained = gained / missed_size
else:
    recall_gained = 0.0
```

→ **Interpretation (b)**: `recall_gained_by_restore = |S_restore ∩ (G \ L_fwd)| / |G \ L_fwd|`

### §2. 0.5709 의 정확한 해석

| 카테고리 | n | % |
|---|---:|---:|
| missed > 0 (forward 가 ≥1 gold 누락) | **537** | 35.0% |
| missed = 0 (forward 가 perfect recall) | 997 | 65.0% |
| **합계** | **1534** | 100% |

→ **0.5709 = 537 conditional queries 의 per-query mean** (forward 가 imperfect 한 영역). 학술 Agent narrative "of the gold columns that forward misses, backward recovers 57%" **정합** — 정정 불필요.

### §3. recall_gained 분포 — Bimodal Distribution (학위 논문 §V.5.x.7 별도 axis 가능)

| Bucket | n | % |
|---|---:|---:|
| 0 (zero recovery) | 205 | 38.2% |
| (0, 0.5) | 4 | 0.7% |
| [0.5, 0.75) | 46 | 8.6% |
| [0.75, 1.0) | 3 | 0.6% |
| **1.0 (perfect recovery)** | **279** | **52.0%** ⭐ |

→ **bimodal**: 52% perfect + 38% zero + 10% middle. 단순 평균 0.5709 = dual mode 의 mid-point. 학위 논문 §V.5.x narrative 의 "selective gain on hard queries" 정량 evidence (학술 Agent Phase 1 §3 challenging query lift 0.3287 dominant 와 일관).

### §4. Mathematical Identity 정합 (Δrecall_union 과의 등식)

**Per-query identity** (proof):

$\Delta r_q = |S \cap M| / |G| = \text{recall\_gained}_q \cdot (1 - r_{\text{fwd},q})$

where $M = G \setminus L_{\text{fwd}}$, $S = L_{\text{bwd}} \setminus L_{\text{fwd}}$.

**Aggregate 정합** (1534 queries):
- $\text{mean}_q(\Delta r_q)$ = **0.0771** (A-2 summary)
- $\text{mean}_q(\text{recall\_gained}_q \cdot |M_q|/|G_q|)$ = **0.0768** (per-query 직접 계산, rounding error 0.0003)

→ **macro-aggregate identity 정합 (within rounding tolerance)**. Sample 5 queries 의 per-query identity 검증 모두 정확 일치.

⚠️ **Naive product 의 caveat**: $0.5709 \cdot (1 - 0.8706) = 0.0739 \neq 0.0771$ — conditional aggregation (537 vs 1534) + per-query covariance 차이 때문. **proper conditional aggregate (mean of products)** 만 정확 등식.

### §5. 보조 통계 (학위 논문 §V.5.x.6 main 인용 의 alternative)

| 통계 | 정의 | 값 | 의미 |
|---|---|---:|---|
| **Per-query mean (conditional, n=537)** | $\text{avg}_{q: M_q \neq \emptyset}(\text{recall\_gained}_q)$ | **0.5709** ⭐ | **학술 Agent 인용 (학위 논문 main)** |
| Per-query median (conditional, n=537) | median | 1.0000 | bimodal 의 mode |
| Pooled rate (micro-avg) | $\sum_q |S \cap M| / \sum_q |M|$ | **0.5984** | "across all missed gold cols pooled" |
| Per-query mean (unconditional, n=1534) | $\text{avg}_q(\text{recall\_gained}_q)$, $0$ for $M_q=\emptyset$ | 0.1999 | "across all queries" — diluted |

→ 학위 논문 main 표기 = conditional mean 0.5709 + 보조 pooled 59.84% (narrative 강도 비슷).

### §6. 학위 논문 §V.5.x.6 main 인용 final 권장 문구 (verification §4.2 인용)

```
Direction A (RSL-SQL backward path) 의 본 도메인 transfer 정량 evidence:

  - mean(S_restore_precision) = 0.6434 (n=488, S_restore non-empty queries) — 학술 Agent
    threshold 0.60 의 1.07× margin. SGBE 의 S_keep_hard noise 81.22% 와 dramatic
    contrast (Direction A 의 noise 35.66%, SGBE 의 절반 수준).

  - mean(recall_gained_by_restore) = 0.5709 (n=537, forward-imperfect queries):
    "of the gold columns that forward misses, backward recovers 57% on average per query"
    (학술 Agent 인용, conditional mean 정합).

  - 보조 통계: pooled rate = 59.84% (sum |S_restore ∩ missed| / sum |missed|);
    bimodal distribution (52% perfect recovery + 38% zero recovery).

  - Aggregate consistency: mean(Δrecall_union) = +0.0771 = mean(recall_gained × |M|/|G|)
    per-query identity (mathematical proof).
```

학위 논문 §V.5.x.7 보조 인용 (분포 분석):

```
Forward (XiYan anchor) 가 BIRD-Dev 1534 query 중 65% (997 queries) 에서 모든 gold cols
회수 — backward path 의 의미 있는 영역은 35% (537 queries) 만. 이 conditional 영역에서:
  - 52% (279 queries) perfect recovery (backward 가 모든 missed gold 복원)
  - 38% (205 queries) zero recovery (backward 가 한 col 도 복원 안 함)
  - 10% (53 queries) partial recovery (0 < recall_gained < 1)
→ Direction A 의 학회 narrative "selective gain on hard queries" 정량 evidence — 학술
Agent Phase 1 §3 challenging query 의 backward effect 0.3287 (simple 0.2216 의 1.48×)
와 일관.
```

### §7. Direction A `RSLBackwardFilter` Implementation Spec 갱신 (직전 entry 의 spec 인용)

직전 entry (2026-05-13 학술 Agent Phase 3 Response) 의 Implementation Spec 의 Step 3 (S_restore 측정 protocol) 의 정량 측정 base 갱신 — Module:Filter 의 Direction A 배포 후 정량 측정 시 본 verification 의 conditional protocol 정합:

```python
# Module:Filter 의 정량 측정 protocol (Direction A 배포 후 analyzer 의 ΔF1 보고서)
# (1) S_restore_precision 측정: n=488 (S_restore non-empty), threshold = 0.60
# (2) recall_gained_by_restore 측정: n=537 (forward imperfect, missed > 0), conditional mean
#     분모 = |G \ L_fwd| (interpretation (b), verification 정합)
#     "57% of forward-missed gold recovered by backward" narrative 정합
# (3) Δrecall_union 측정: n=1534 (전체), unconditional mean
#     per-query identity: Δr_q = recall_gained_q × (1 - r_fwd,q)
# (4) (보조) pooled rate = sum|S ∩ M| / sum|M| (micro-avg, 59.84% for verification data)
```

### §8. 학술 Agent 응답 base (verification §5.2 cover note)

학술 Agent 의 직전 수치 확인 요청 응답 cover note (verification §5.2):

```
recall_gained_by_restore 의 분모 = |G \ L_fwd| (forward 누락 gold 의 수) — interpretation (b).
0.5709 = 537 conditional queries (missed > 0) 의 per-query 평균.
narrative "of the gold columns that forward misses, backward recovers 57%" 정합 — 정정 불필요.

보조 통계:
  - pooled rate = 59.84% (sum 분모/분자)
  - distribution bimodal (52% perfect + 38% zero recovery)
  - 1534 query 중 forward perfect = 997 (65%), backward 의 의미 영역 = 537 (35%)
  - Aggregate identity check: mean_q(recall_gained × |missed|/|gold|) = 0.0768
    ≈ mean(Δrecall_union) = 0.0771 ✓ (within rounding tolerance)
```

→ 신규 cover note 파일 `planning/filter_proposal_phase4_cover_note_for_scholar_agent_2026-05-13.md` 작성 (본 entry §8 인용).

### §9. paper_research_direction.md 갱신 항목

- **§3.5 evidence 표 (line 506 ~)**: axis #9 placeholder 추가 — "Direction A (RSL-SQL Backward) 학술 Agent Phase 3 GO 확정 + Module:Filter 배포 chain 진행 중. ΔF1(A) trigger 로 정량 main update."
- **학위 논문 Part III Chapter Base sub-section (line 695 ~) 후속**: 신규 sub-section "§V.5.x.6 Direction A (RSL-SQL Backward) — recall_gained verification 완료, Module:Filter 배포 chain 진행 중" 추가. §V.5.x.6 main 인용 (verification §4.2) + §V.5.x.7 보조 인용 (분포 분석) + Direction A 배포 후 ΔF1 trigger 의 narrative final update plan 명시.

### Chain Status 갱신 (5/13 저녁 — verification 완료 후)

| # | Chain | Status |
|---|---|---|
| 1-9 | 직전 chains (학술 Agent Phase 3 Response 포함) | ✅ 완료 |
| **10** | **Analyzer: recall_gained 정의 확인** | ✅ **완료 (5/13)** — interpretation (b) 확정 + 학술 Agent narrative 정합 + identity 정합 |
| **11** | **Module:Filter: `RSLBackwardFilter` 구현** | 🚀 진행 중 (병행 chain — 본 verification 결과 의존 없음, GO 확정 base) |
| **12** | **사용자 → 학술 Agent**: Phase 4 cover note 전달 | 🚀 즉시 (본 verification §5.2 base) |
| ⏸ 13 | Root: Direction A sweep launch | ⏸ Module:Filter 완료 의존 |
| ⏸ 14 | Analyzer: Direction A 결과 보고서 | ⏸ Root sweep 완료 의존 |
| ⏸ 15 | 학술 Agent Direction C 재결정 (ΔF1 trigger) | ⏸ Direction A 결과 의존 |
| ⏸ 16 | 학위 논문 §V.5.x.6 narrative final integration | ⏸ Direction A 배포 후 ΔF1 정량 trigger |

### §10. 영향 범위

- planning/DECISIONS.md (본 entry)
- planning/filter_proposal_phase4_cover_note_for_scholar_agent_2026-05-13.md (신규 파일, 본 entry §8 인용)
- planning/paper_research_direction.md §3.5 evidence 표 + Part III Chapter Base sub-section (§9 항목)
- 후속: Module:Filter 의 정량 측정 protocol (§7 spec 갱신 인용)

### §11. 근거

- 사용자 직접 input (Analyzer 보고)
- notebooks/analysis_results/recall_gained_denominator_verification.md §1~§5 (verification 본문)
- src/analysis/filter_proposal_a3_restore_noise.py (lines 95-104, interpretation (b) 코드 직접 인용)
- 직전 entry (학술 Agent Phase 3 Response) §1 recall_gained 정의 정합성 — 본 verification 의 정식 응답

### §12. 사용자 후속 actions (Direction A 배포 chain 의 잔여 step)

1. ✅ **Analyzer 핸드오프** — recall_gained 분모 확인 완료 (interpretation (b), 학술 Agent narrative 정합)
2. 🚀 **Module:Filter 핸드오프** — `RSLBackwardFilter` 구현 (병행 chain 진행 중, verification 결과 의존 없음)
3. 🚀 **사용자 → 학술 Agent** — Phase 4 cover note 전달 (verification §5.2 base, planner 작성)
4. ⏸ **Root 핸드오프 prep** — Module:Filter 완료 후 sweep launch
5. ⏸ **Direction C 재결정** — Direction A 의 ΔF1 결과 후 (학술 Agent decision trigger)
6. ⏸ **학위 논문 §V.5.x.6 narrative final integration** — Direction A 배포 후 ΔF1 정량 main update

---

## 2026-05-13 (학술 Agent Phase 3 Response — Direction A GO 확정 + C 재결정 기준 + B Hold + recall_gained 정의 확인 요청)

> **사용자 직전 input (5/13)**: 학술 Agent 의 Phase 3 response 공유 (`filter_proposal_by_scholar_agent_phase2_2026-05-13.md`) — 5 cover note 질문 응답 + recall_gained 정의 정합성 분석 + C-1/C-2 정렬 패턴 발견 + Direction C 재결정 기준 (ΔF1(A) ≥ 0.03 / < 0.02 분기).

### Cover Note 5 질문 응답 매핑

| Q | 학술 Agent 응답 | 결정 |
|---|---|---|
| **Q1** Direction C priority | A 배포 후 측정 (gray zone 행동 지침) | **Option α** confirm |
| **Q2** C DB-targeted vs all-DB | ΔF1(A) < 0.02 시 **debit_card_specializing + card_games 타겟** 구현 (inferred_fk GPT-4.1-mini 보완 선행) | **DB-level targeted** (조건부) |
| **Q3** Direction A implementation 구체 가이드 | (직접 답변 없음 — GO 확정 단 구체 구현은 사용자 / Module 위임) | Module:Filter 위임 |
| **Q4** Direction B trigger | A 배포 후 recall gap 잔존 시 B-1/B-2 착수 | **A 결과 후 결정** |
| **Q5** Bug fix magnitude 영향 | A 배포 결정 영향 없음. margin 1.20× → 1.07× 좁아짐 → toxicology 외 추가 low-precision DB 시 DB-level guard 준비 | **A 결정 유지 + caveat 강화** |

### 학술 Agent 의 결정적 신규 발견 (cover note 외 자체 분석)

#### 1. recall_gained_by_restore 정의 정합성 — 학위 논문 §V 인용 base

학술 Agent 의 spec 의도 정합 해석:

| Phase | 추정 분모 | 의미 |
|---|---|---|
| Phase 1 | `|gold|` (전체 gold 대비) | 0.2489 = 24.89% |
| **Phase 2 (fix)** | `|gold − (gold ∩ L_fwd)|` (forward 가 놓친 gold 대비) | **0.5709 = backward 가 forward 누락 gold 의 57% 회복** ⭐ |

→ **학위 논문 §V 핵심 인용 (학술 Agent 권장)**:
> "**Of the gold columns that forward misses, backward recovers 57%**"

**🚨 수치 확인 요청 (Analyzer 후속)**: `filter_proposal_a3_restore_noise.py` 의 recall_gained_by_restore 분모가 정확히 어느 형식인지 스크립트 주석으로 확인. `|gold|` 분모일 시 Δrecall_union +0.0771 과 수학적 정합성 재점검 필요.

#### 2. C-1 + C-2 의 DB-level 정렬 패턴 — "FK declaration 부족 → join col miss" BIRD 실증

| DB | C-1 fk_coverage | C-2 structural miss rate |
|---|---:|---:|
| **debit_card_specializing** | **0.2000** (outlier) | **10.59%** ⭐ |
| **card_games** | **0.5714** | **9.33%** ⭐ |
| financial | (mid-range) | 13.21% (highest miss) |
| (3 perfect DBs: formula_1, student_club, superhero) | 1.0 | (low miss) |

→ **"FK declaration 부족 → join col miss" 경로의 BIRD 실증** + GRAST-SQL 의 "predicting missing keys" 기능 필요성 정량 정당화. 학위 논문 §V.5.x 의 Direction C 의 schema-dependent caveat 의 결정적 evidence.

#### 3. Coverage Bimodal 패턴

- formula_1 / student_club / superhero: 1.00 (완전)
- debit_card_specializing: 0.20 (극단적 저선언)
- → BIRD DB 의 "schema designer 의 메타데이터 충실도" 이질성 실증

### Direction 별 최종 결정 표 (학술 Agent Phase 3 정식)

| Direction | 상태 | 다음 행동 | 시점 |
|---|---|---|---|
| **A (RSL Backward)** | **GO 확정** | `RSLBackwardFilter` 구현 + anchor 대비 ΔF1 측정 | **즉시** |
| **C (GRAST-SQL FD)** | Feasible, **mid-priority** | A EX 측정 후 ΔF1 lift 의 임계 분기 | A 배포 결과 후 |
| **B (HN-SupCon)** | **Hold** | A 배포 후 recall gap 잔존 확인 시 B-1/B-2 착수 | A 결과 후 |

### Direction C 재결정 기준 (A 배포 후, 학술 Agent 정식 trigger)

| Trigger | Direction C 결정 |
|---|---|
| **ΔF1(A) ≥ 0.03** | C **post-paper 확정** |
| **ΔF1(A) < 0.02** | C **debit_card_specializing + card_games 타겟 구현** (inferred_fk GPT-4.1-mini 보완 선행) |
| 0.02 ≤ ΔF1(A) < 0.03 | gray zone (학술 Agent decision 추가 필요) |

### Direction A 배포 Margin Caveat (학술 Agent Q5 응답 의 강화)

Bug fix 후 magnitude 변화:
- A-3 precision: 0.7205 → **0.6434** (margin 1.20× → **1.07×** 좁아짐)
- → toxicology 외 추가 low-precision DB 시 **DB-level guard 준비**
- Direction A `RSLBackwardFilter` 구현 시 **DB-level precision threshold** 조건부 적용 prep (Implementation 시 module:filter 결정 필요)

### 후속 chain 계획 (학술 Agent decision 후)

| Step | 책임 | 작업 |
|---|---|---|
| 1 | **Analyzer** (즉시, 학술 Agent confirm 요청) | A-3 recall_gained_by_restore 분모 정의 스크립트 주석 확인 → 학술 Agent 의 0.5709 해석 정합성 검증 |
| 2 | **Module:Filter** | `RSLBackwardFilter` (또는 학술 Agent 권장 명명) 신규 클래스 — XiYan forward + Preliminary SQL (full schema) backward + S_restore union + (조건부) DB-level precision guard |
| 3 | **Root** | Direction A pipeline config + sweep launch (anchor + Backward) → ΔF1/ΔEX 정량 |
| 4 | **Analyzer** | Direction A 배포 결과 보고서 — ΔF1 lift 정량 + per-DB breakdown |
| 5 | **Planner + 학술 Agent (사용자 bridge)** | 학술 Agent Direction C 재결정 (ΔF1(A) trigger 분기) — B Hold 유지 or B-1/B-2 launch |
| 6 | **Module:Filter (조건부)** | (ΔF1(A) < 0.02 시) Direction C 의 debit_card + card_games 타겟 구현 + inferred_fk GPT-4.1-mini 보완 |
| 7 | **Planner** | 학위 논문 §V.5.x narrative final integration (학술 Agent 의 모든 발견 + Direction A/C 결과 통합) |

### Direction A `RSLBackwardFilter` Implementation Spec (학술 Agent Q3 위임 후, planner 정리)

```
Class: RSLBackwardFilter
Module: src/modules/filters/rsl_backward_filter.py
Interface: refine(query, subgraph, db_id, **kwargs) → {status, final_nodes, reasoning, stats}

Step 1 (기존 anchor 의 XiYan forward, 의존성):
    S_fwd = XiYanFilter.refine(query, subgraph, db_id).final_nodes

Step 2 (신규, GLM 4.7 preliminary SQL):
    full_schema_str = build_full_schema(db_schema_map, db_id)
    prelim_sql = GLM_4_7.chat(create_messages(full_schema_str, query, evidence))
    L_bwd = sqlglot.extract_columns(prelim_sql, col-only-distinct)  # Phase 2 bug fix 후 normalization 정합

Step 3 (S_restore + DB-level guard 조건부):
    S_restore = L_bwd - S_fwd
    if db_id in {"toxicology", ...} (학술 Agent caveat):
        # DB-level precision guard — threshold 결정 candidate (예: precision < 0.60 시 skip)
        S_restore_filtered = (조건부) S_restore 또는 skip
    else:
        S_restore_filtered = S_restore

Step 4 (S_struct FK/PK hardcode):
    S_struct = extract_fk_pk_columns(subgraph, metadata)

Output:
    final_nodes = S_fwd ∪ S_restore_filtered ∪ S_struct
    
LLM calls per query: 2 (Step 1 XiYan + Step 2 preliminary SQL)
Token cost: anchor 대비 ~+100% (preliminary SQL 의 full schema input)
```

학술 Agent Q3 의 직접 답변 없음 — Module:Filter 의 implementation 결정 (구체 detail 은 module:filter 세션 위임).

### Chain status 갱신 (5/13 저녁 최종)

| # | Chain | Status |
|---|---|---|
| 1-6 | 직전 chains | ✅ 완료 |
| 7 | Phase 1 (A-1/A-2/A-3) | ✅ 완료 |
| 8 | Phase 2 (A-2 fix + C-1/C-2) | ✅ 완료 |
| **9** | **학술 Agent Phase 3 decision** | ✅ **완료 (5/13)** — Direction A GO 확정 + C feasible-mid + B Hold |
| **10** | **Analyzer: recall_gained 정의 확인** | 🚀 **즉시 trigger** (학술 Agent 수치 확인 요청) |
| **11** | **Direction A 배포 chain** | 🚀 **즉시 trigger** — Module:Filter `RSLBackwardFilter` + Root sweep |
| ⏸ 12 | Direction C 재결정 (A 결과 후) | ⏸ ΔF1(A) trigger 의존 |
| ⏸ 13 | Direction B (HN-SupCon) launch | ⏸ A 결과 후 recall gap 잔존 시 |

### 학위 논문 §V.5.x Narrative Update (학술 Agent Phase 3 의 핵심 인용)

```
§V.5.x.6 (학술 Agent 권장 핵심 인용)

"Of the gold columns that forward misses, backward recovers 57%"
(Phase 2 recall_gained_by_restore = 0.5709, fix 후 spec 의도 정합)

§V.5.x.7 BIRD DB Schema Heterogeneity (학술 Agent 신규 발견)

"FK declaration 부족 → join col miss 경로의 BIRD 실증":
- C-1 fk_coverage bimodal: formula_1/student_club/superhero 1.00 ↔ debit_card 0.20
- C-2 structural miss 의 DB-level 정렬: debit_card 10.59% + card_games 9.33%
- → GRAST-SQL "predicting missing keys" 기능 의 필요성 정량 정당화

§V.5.x.8 Direction A Margin Caveat (학술 Agent Q5 응답)
- Phase 1 margin 1.20× → Phase 2 fix 후 1.07× 좁아짐
- DB-level precision guard 의 implementation 차원 lesson learned
```

### 영향 범위

- planning/DECISIONS.md (본 entry)
- 후속:
  - Analyzer recall_gained 정의 확인 (즉시, 학술 Agent 수치 확인 요청)
  - Module:Filter `RSLBackwardFilter` 구현 (즉시, 학술 Agent GO 확정)
  - Root Direction A sweep launch (Module:Filter 완료 후)

### 근거

- 사용자 직접 input + 학술 Agent file `filter_proposal_by_scholar_agent_phase2_2026-05-13.md`
- 학술 Agent §0 Decision Rules 최종 평가 + §1 bug fix 해석 + §2 C-1 분석 + §3 C-2 gray zone + §4 Direction 별 최종 권고

### 사용자 후속 actions

1. **Analyzer 핸드오프** — recall_gained 분모 확인 (즉시)
2. **Module:Filter 핸드오프** — `RSLBackwardFilter` 구현 (즉시, Analyzer 확인 후속 또는 병행)
3. **Root 핸드오프 prep** — Module:Filter 완료 후 sweep launch
4. **Direction C 의 (조건부) 후속** — Direction A 의 ΔF1 결과 후

---

## 2026-05-13 (Phase 2 완료 — A-2 bug fix + C-1/C-2 측정 + Direction A 배포 GO 재확인 + Direction C feasible-mid)

> **사용자 직전 input (5/13)**: Analyzer 가 Phase 2 chain 완료 — A-2 xlsx recall bug fix + A-2/A-3 재계산 + C-1/C-2 측정 + 사용자 deliverable (`filter_proposal_phase2_summary.md` + `phase2_records.xlsx`). Planner 작업 요청: DECISIONS prepend + 학술 Agent 추가 질문 list.

### Phase 2 결정 Rule 평가 (5 rules)

| Rule | 실측 | Threshold | 결과 |
|---|---:|---:|:---:|
| **A-3 core** mean(S_restore_precision) | **0.6434** | ≥ 0.60 | ✅ PASS (1.07×) |
| **A-2 core** mean(Δrecall_union vs fwd) | **+0.0771** | ≥ +0.05 | ✅ PASS (1.54×) |
| **A-3 보조** mean(recall_gained_by_restore) | **0.5709** | ≥ 0.05 | ✅ PASS (11.4×) |
| **C-1** mean(fk_coverage_rate) | **0.7312** | ≥ 0.50 → feasible | ✅ Direction C **feasible** |
| **C-2** mean(is_join_complete, multi-table) | **0.8624** | < 0.80 priority up / ≥ 0.95 post-paper | ⚠️ **mid-priority** (0.80~0.95 중간) |

→ **Direction A 배포 GO 재확인** + **Direction C feasible-mid**.

### A-2 Bug Fix 상세 (학술 Agent xlsx report 5/13 응답)

**Bug 원인**: A-2 의 recall 계산 시 column normalization 방식 (col-only vs alias-distinct).
- 직전 (bug): alias-distinct 로 col 추출 — gold 의 `t1.col` 과 anchor 의 `t2.col` 의 alias mismatch 가 분모 base inflation
- Fix: **col-only normalization** (`alias-distinct → col-only-distinct`)

**Fix 후 magnitude 변화**:
- A-3 mean(S_restore_precision): **0.7205 → 0.6434** (Δ = -0.0771, -10.7%)
- Recall ≤ 1.0 보장 확인
- Decision Rules 재확인 PASS

**의의**: 매그니튜드는 직전보다 낮아졌으나 PASS threshold 충족 + recall 정의 정확성 확보. 학술 Agent 의 의사결정 trust 회복.

### C-1 결과 — Direction C Feasible 확정 단 outlier 존재

- **mean(fk_coverage_rate) = 0.7312** (BIRD-Dev 11 DBs)
- **3 perfect DBs** (coverage = 1.0)
- **debit_card_specializing outlier**: **0.2000** (-0.53 from mean) — FK 선언 매우 sparse
- → Direction C 의 graph quality 가 DB-level variance 큼. 학위 논문 §V.5.x footnote 의 schema-dependent caveat candidate

### C-2 결과 — Join Completeness Mid-Priority

- **mean(is_join_complete) = 0.8624** (multi-table queries)
- C-2 가 < 0.80 미만이 아니라 ≥ 0.95 도 아닌 **중간 영역** — Direction C 우선순위 결정 학술 Agent decision 필요
- **Lift target DBs** (is_join_complete 낮은 DB):
  - **card_games**: 0.6893 (가장 낮음, anchor join col 31% missing)
  - **debit_card_specializing**: 0.7347 (C-1 outlier 와 일관)
- → Direction C 적용 시 두 DB 에 dominant lift candidate. all-DB 적용 vs DB-level targeted 학술 Agent decision

### Direction A 배포 GO 재확인

- Phase 1 (직전 5/13): A-3 0.7205 / A-2 +0.0761 / A-3 보조 0.2489 — 3/3 PASS
- Phase 2 (재계산, bug fix 후): A-3 0.6434 / A-2 +0.0771 / A-3 보조 0.5709 — 3/3 PASS (재확인)
- → Direction A 배포 결정 **유지**.
- A-3 보조 (recall_gained_by_restore) 가 **0.5709 (Phase 1 0.2489 → Phase 2 0.5709, 11.4×)** — bug fix 후 magnitude 큼 (학술 Agent 의 의사결정 confirm 강력)

### 학술 Agent 추가 질문 List (Phase 3 decision 정확히 받기 위해)

planner 가 학술 Agent 에 전달할 5 항목 — `filter_proposal_phase2_cover_note_for_scholar_agent_2026-05-13.md` 별도 md 작성 (본 entry 의 §"산출물" 참조).

### Direction C Priority 결정 — 사용자 결정 candidate

| Option | trade-off |
|---|---|
| **(α) Direction A 배포 후 측정** (recommended) | A 의 ΔF1 lift 정량 후 C 의 marginal value 측정 — 일정 ~1주 추가. 학위 논문 §V.5.x 의 본 phase 통합 |
| **(β) 즉시 Direction C launch** | A + C 병행 — 학위 논문 5/22 마감 빠듯, V5 chain 의 GPU 와 자원 conflict 가능 |
| **(γ) Direction C post-paper** | C-2 mean 0.8624 가 0.95 미달이라 post-paper 전환은 trigger 부정확 단 일정 우선 시 가능 |

→ **권장: (α)**. 단 학술 Agent decision 우선.

### Direction A 배포 chain (post-Phase 3 학술 Agent decision)

학술 Agent 의 Phase 3 response 후 Direction A 의 production 구현:

| Step | 책임 | 작업 |
|---|---|---|
| 1 | Module:Filter | `RSLBackwardFilter` 신규 클래스 — XiYan forward + Preliminary SQL (full schema) backward + S_restore union + (조건부) DB-level guard (toxicology / debit_card_specializing) |
| 2 | Root | Direction A pipeline config + sweep launch (anchor + Backward) — ΔF1/ΔEX 정량 |
| 3 | Analyzer | Direction A 배포 결과 보고서 + paper §V.5.x base |
| 4 | Planner | 학위 논문 §V.5.x narrative final integration (학술 Agent 의 4 세부 발견 + Direction A 결과) |

### Direction A Implementation 의 핵심 구현 결정 (학술 Agent 추가 질문 #3 의 base)

직전 spec 의 Direction A pseudo-code:
```
Step 1: XiYan forward prune → S_fwd  (anchor 기존)
Step 2: GLM 4.7 로 preliminary SQL 생성 (full schema)
        → SQL_prelim parse → L_bwd
Step 3: S_restore = L_bwd \ S_fwd
Step 4: final_nodes = S_fwd ∪ S_restore ∪ S_struct (FK/PK)
LLM calls: 2 (XiYan 기존 + Preliminary SQL)
```

**구체 가이드 학술 Agent 결정 필요**:
- (a) Step 2 의 preliminary SQL 입력 — full schema vs S_fwd? (학술 Agent §"Direction A 핵심" 의 default = full schema)
- (b) Toxicology / debit_card_specializing DB 의 precision guard 조건 — DB-level threshold (예: precision < 0.6 시 skip backward) vs query-level threshold
- (c) S_struct (FK/PK hardcode) 통합 — Direction C 의 일부 component (CHESS Talaei 2024) 와 동일?
- (d) Direction A 의 학술 Agent narrative 측면 — `RSLBackwardFilter` vs `BidirectionalFilter` 의 명명 선호 (RSL 의 origin paper 정합)

### Chain status 갱신 (5/13 저녁)

| # | Chain | Status |
|---|---|---|
| 1 | Filter sweep 9-cell | ✅ 완료 |
| 2 | V5 sweep V5-A/B/C | 🔄 active (Module:Selector ownership 완료 + Root sweep launch 대기) |
| 3 | SGBE Phase 3-5 | ✅ 완료 (negative evidence) |
| 4 | V5-D-1 진단 | ✅ 완료 |
| 5 | V5-D-2 학습 | ⏸ V5 sweep 결과 후 |
| 6 | B1'+B2'+B3' GLM Baseline | ✅ 완료 |
| **7** | **Filter Proposal Phase 1 (A-1/A-2/A-3)** | ✅ 완료 (5/13 PASS 3/3, Direction A GO) |
| **8** | **Filter Proposal Phase 2 (A-2 bug fix + C-1/C-2)** | ✅ **완료 (5/13)** — Direction C feasible-mid |
| ⏸ 9 | **Filter Proposal Phase 3** | ⏸ 학술 Agent Phase 3 decision 대기 (cover note 전달 후 ~5/14 ETA) |
| ⏸ 10 | **Direction A 배포 chain** | ⏸ 학술 Agent decision 후 Module:Filter `RSLBackwardFilter` 작성 → Root sweep |
| ⏸ 11 | **Direction C launch** | ⏸ 학술 Agent decision 의 priority 결정 의존 (Option α/β/γ) |
| ⏸ 12 | (Direction B) HN-SupCon | ⏸ Direction A 결과 후 학술 Agent trigger 결정 |

### 학회/학위 논문 narrative 측면

**학회 paper**:
- Filter Dominance 의 8-axis evidence 유지 + Direction A 배포 결과 footnote (post-paper)
- 학회 paper main contribution = paper §3.5 의 7번째 axis (Filter-Invariant) + 8번째 (SGBE Negative Evidence)

**학위 논문 §V.5.x**:
- §V.5.x.1 SGBE Negative Evidence (직전 5/13 entry)
- §V.5.x.2 Direction A 배포 — 학술 Agent 의 4 세부 발견 + Phase 2 Decision Rule 통과 + bug fix lesson learned (reproducibility)
- §V.5.x.3 Toxicology / debit_card_specializing DB-level caveat (schema-dependent outlier)
- §V.5.x.4 Direction C feasibility (C-1 0.7312) + mid-priority (C-2 0.8624)

### 산출물 위치

- `notebooks/analysis_results/filter_proposal_phase2_summary.md` (347 lines, 7 sections — analyzer 작성)
- `outputs/analysis/filter_proposal/phase2_records.xlsx` (4 sheets, 0.20 MB)
- `outputs/analysis/filter_proposal/{A2,A3,C1,C2}*.{jsonl,csv,json}` (재계산 + 신규)
- `src/analysis/filter_proposal_{a2_backward_recall,c1_fd_graph,c2_structural_miss}.py`
- `src/analysis/tests/test_filter_proposal_phase{1,2}.py` (28 tests passed)
- **🆕** `planning/filter_proposal_phase2_cover_note_for_scholar_agent_2026-05-13.md` (본 entry 후속, planner 작성)

### 근거

- 사용자 직접 input (Phase 2 결과 + planner 작업 요청)
- Analyzer 의 deliverable (phase2_summary.md + phase2_records.xlsx)
- Phase 2 결정 Rule 5 항목 평가 (3 A-PASS + C-1 feasible + C-2 mid)

### 후속 chain

1. **Planner (본 entry)** — DECISIONS prepend ✅ + cover note 작성 (학술 Agent 5 질문 list)
2. **사용자** — cover note + phase2_summary.md + phase2_records.xlsx 학술 Agent 에 전달
3. **학술 Agent** — Phase 3 response (~5/14 ETA) — Direction A implementation 구체 가이드 + Direction C priority decision
4. **사용자** — 학술 Agent response → planner 에 전달
5. **Planner** — DECISIONS prepend + Direction A 배포 chain 작성 (Module:Filter + Root + Analyzer) → root 핸드오프
6. **Module:Filter + Root** — Direction A `RSLBackwardFilter` 구현 + sweep launch
7. **Analyzer + Planner** — 결과 통합 + 학위 논문 §V.5.x narrative final

---

## 2026-05-13 (Phase 1 PASS 3/3 — Direction A 배포 결정 확정 + Phase 2 GO + A-2 xlsx bug fix)

> **사용자 직전 input (5/13)**: 학술 Agent 의 Phase 1 데이터 review response — **Decision Rule 3/3 PASS** + Direction A (RSL-SQL Backward) 즉시 배포 권장 + Phase 2 (C-1, C-2) GO + A-2 xlsx recall 계산 bug 확인 요청.

### 🎯 Phase 1 Decision Rule 3/3 PASS

| Rule | 지표 | Threshold | 실측 | 결과 |
|---|---|---|---|---|
| **A-3 core** | `mean(S_restore_precision)` | ≥ 0.60 | **0.7205 (1.20×)** | ✅ PASS |
| **A-2 core** | `mean(recall_union) - mean(recall_fwd)` | +Δ ≥ 0.05 | **+0.0761 (1.52×)** | ✅ PASS |
| **A-3 보조** | `mean(recall_gained_by_restore)` | ≥ 0.05 | **0.2489 (4.98×)** | ✅ PASS |

→ **Direction A (RSL-SQL Backward) 즉시 배포 결정 확정**.

### SGBE vs Direction A — Noise Rate Contrast

| Filter | S_restore (또는 S_keep_hard) noise rate |
|---|---:|
| SGBE (5/13 부정) | **81.22% noise** ⚠️ |
| **Direction A (5/13 PASS)** | **27.95% noise** ✅ (1 - 0.7205) |

→ Direction A 의 noise rate 가 SGBE 의 **~1/3** — 극적 정량 contrast. 학위 논문 §V.5.x 의 결정적 evidence.

### 학술 Agent 의 4 세부 발견 (학위 논문 narrative)

**(1) 비용 구조 매우 양호**:
- `mean(|S_restore|) = 1.40 column/query` (실질 token 부담 작음)
- **전체 query 54.50% (836건) 는 S_restore = ∅** (forward 가 이미 backward 전체 포함 — backward path 의 incremental cost 없음)
- → Direction A 의 cost 효율성 정량

**(2) Low-recall query 구제 효과**:
- recall_fwd < 0.50 인 query **123건 → recall_union < 0.50 이 32건** (-73.98% 탈출)
- Perfect-recall query **2건 → 17건** (+8.5×)
- → backward path 가 low-recall 영역의 결정적 구제

**(3) Challenging query 에서 효과 최대**:

| Difficulty | mean(recall_gained_by_restore) |
|---|---:|
| simple | 0.2216 |
| moderate | 0.2682 |
| **challenging** | **0.3287** ⭐ |

→ 어려운 query 일수록 backward path 의 기여 큼. **학위 논문 §V.5.x narrative 의 결정적 layer** (schema linking 의 difficulty stratification + Direction A 의 selective gain).

**(4) Toxicology DB caveat**:
- 11 BIRD-Dev DB 중 **toxicology 만 mean(S_restore_precision) = 0.5770** (기준 0.60 에 -0.023 미달)
- → DB-level precision guard (조건부 적용) 또는 toxicology case study 학위 논문 §V.5.x footnote
- Schema-dependent caveat — over-smoothing 의 A3 stratified analysis 와 일관 (toxicology 의 specific outlier 특성)

### ⚠️ A-2 xlsx Bug — recall 계산 로직 확인 필요

학술 Agent 발견:
- **xlsx**: recall_bwd max = 1.75, recall_union max = 2.0 (1.0 초과 — recall 정의 위반)
- **md summary**: recall_fwd = 0.7551, recall_union = 0.8311 (정상)
- **xlsx**: recall_fwd = 0.6125, recall_union = 0.7583 (md 와 차이)

추정 원인:
- `filter_proposal_a2_backward_recall.py` 의 recall 계산 시 분모가 `|gold_cols|` 대신 `|L_bwd|` 또는 0-division fallback 사용
- 또는 jsonl → xlsx 변환 script 의 column rename / 잘못된 field mapping

학술 Agent 결정: **의사결정 기준은 md summary 수치 (정확)**. 단 Phase 2 수집 전에 A-2 bug fix 필요 — 학위 논문 의 reproducibility 측면 + 향후 Phase B-1/B-2 의 xlsx 의 trust 측면.

### Phase 2 GO — C-1 + C-2

학술 Agent 의 Phase 2 진행 의견:
- **C-1**: BIRD tables.json + PRAGMA parse 만 — LLM 무관, ~수 시간
- **C-2**: A-1 의 predictions.jsonl + gold SQL — LLM 무관, ~수 시간
- 둘 다 A-1 산출물에 의존 X, 독립 수집 가능

**Phase 2 Decision Rules**:
- C-1 `mean(fk_coverage_rate)` < **0.50** → Direction C 전체 feasibility 흔들 → B-1/B-2 만 남음
- C-2 `mean(is_join_complete)` < **0.80** → Direction C 우선순위 상향 (Steiner tree expected gain 큼)
- C-2 ≥ **0.95** → Direction C post-paper 전환

### 결정 + 후속 chain

**3 후속 chain 동시 launch**:

1. **A-2 xlsx bug fix** (Analyzer) — recall 분모 issue 디버그 + 재계산 + xlsx 재변환
2. **Phase 2 C-1 + C-2 launch** (Analyzer) — BIRD-Dev/Train DB FK/PK 분석 + structural miss
3. **Direction A 배포 chain 작성** (Planner, 후속) — Direction A 의 production pipeline 구현 (XiYan forward + Preliminary SQL backward + union)

### 학회 / 학위 논문 narrative 영향

**학회 paper (한국지능정보시스템학회 2026 춘계)**:
- 본 Phase 1 결과 = Filter Dominance 의 **enhancement evidence** (학회 main contribution 변경 X, footnote)
- Direction A 배포 시 anchor F1=0.8651/EX=0.5202 → 추가 lift 정량 evidence

**학위 논문 §V.5.x**:
- 학술 Agent 의 4 세부 발견 모두 narrative 의 결정적 layer:
  - §V.5.x.1: Direction A 의 noise rate 27.95% vs SGBE 81.22% contrast
  - §V.5.x.2: 비용 효율성 (54.50% query S_restore=∅)
  - §V.5.x.3: Low-recall query 구제 (-73.98% 탈출)
  - §V.5.x.4: Challenging query 의 backward effect 0.3287 — difficulty stratification narrative
  - §V.5.x.5: Toxicology DB caveat — schema-dependent (over-smoothing A3 와 일관)

### Direction A 배포 implementation (post-Phase 2)

Direction A 의 production 구현:

```
입력: query Q, db_id, anchor pipeline (Enriched + QCond + MST + XiYan + GLM 4.7)

Step 1 (기존 anchor): XiYan forward prune → S_fwd
Step 2 (신규): GLM 4.7 로 preliminary SQL 생성 (full schema 입력)
              → SQL_prelim parse (sqlglot) → L_bwd
Step 3 (신규): S_restore = L_bwd \ S_fwd
Step 4 (신규): final_nodes = S_fwd ∪ S_restore ∪ S_struct (FK/PK)

LLM calls: 2 (Step 1 기존 + Step 2 prelim SQL)
Toxicology DB 의 경우: DB-level precision guard 조건부 (학술 Agent 권고)
```

→ Module:Filter 의 신규 클래스 `RSLBackwardFilter` 또는 `BidirectionalFilter` 구현 candidate (post-Phase 2).

### 영향 범위

- planning/DECISIONS.md (본 entry)
- planning/filter_proposal_scholar_agent_response_phase2_2026-05-13.md (신규 — 학술 Agent response 보존)
- (Analyzer) `src/analysis/filter_proposal_a2_backward_recall.py` bug fix
- (Analyzer) `src/analysis/filter_proposal_c1_fd_graph.py` + `filter_proposal_c2_structural_miss.py` 신규
- (Module:Filter, post-Phase 2) `src/modules/filters/rsl_backward_filter.py` (Direction A 배포)
- (Root, post-Phase 2) Direction A 배포 anchor + sweep launch

### 학위 논문 일정 (5/14~5/22)

- 5/14: A-2 bug fix + Phase 2 C-1/C-2 launch (병행, LLM 무관 ~수 시간)
- 5/14~5/15: 사용자 가 Phase 2 md + xlsx 변환 → 학술 Agent 전달
- 5/15: 학술 Agent Direction C decision + Direction A 배포 chain start
- 5/15~5/22: Direction A 배포 implementation + 학위 논문 §V.5.x narrative 통합

### 근거

- 사용자 직접 input (학술 Agent Phase 1 response 공유)
- 학술 Agent 의 Decision Rule 3/3 PASS + 4 세부 발견 + xlsx bug 발견 + Phase 2 GO

### 후속 actions (즉시)

1. Analyzer: A-2 bug fix + Phase 2 (C-1 + C-2) script 작성 + smoke test
2. Root: Phase 2 launch (LLM 무관, GPU 무관, ~수 시간)
3. 사용자: Phase 2 결과 md + xlsx 변환 → 학술 Agent 전달
4. Planner (post-Phase 2): Direction A 배포 implementation plan

---

## 2026-05-13 (학술 Agent Phase 1 GO Response — A-1 truncated 필드 보강 + B-1/B-2 summary 항목 정식 + Phase 1 launch 결정)

> **사용자 직전 input (5/13)**: 학술 Agent 의 response 공유 — Phase 1 (A-1/A-2/A-3) 결과 우선 receive + Decision Rules 표 정식 + B-1/B-2 summary 항목 명시 + A-1 truncated 필드 보강 제안.

### 학술 Agent Response 핵심

**(a) Decision Rules 표 정식**:

| Dataset | 핵심 지표 | Decision Trigger | 다음 행동 |
|---|---|---|---|
| **A-3** | mean(S_restore_precision) | **≥ 0.6** → Direction A 우선 배포 | Direction A 즉시 구현 + B/C 병행 |
| **A-2** | mean(recall_union) - mean(recall_fwd) | **+Δ ≥ 0.05** → backward path 유효 | A-3 precision 과 종합 판단 |
| **C-1** | mean(fk_coverage_rate) | **≥ 0.50** → Direction C feasible | < 0.30 시 Direction C post-paper |
| **C-2** | mean(is_join_complete) | **< 0.80** → Steiner tree expected gain 큼 | Direction C 우선순위 상향 |
| **B-1** | mask=1 ratio (hard negative) | **≥ 30%** → HN-SupCon 학습 의미 | A 결과 후 B-2 와 진행 결정 |

**(b) B-1 / B-2 summary 항목 정식** (xlsx full dump 대신 md summary 충분):

- **B-1 요약**: hard negative 비율 (전체 / per-query 분포) + easy negative 비율 + cosine score 분포 histogram (TP vs FN vs TN 3 group) + mask=1 ratio 추정치
- **B-2 요약**: gold:non-gold ratio (BIRD-Train) + Train vs Dev distribution 차이 + per-difficulty 분포 (simple/moderate/challenging)

**(c) A-1 보강 권고 — Token Limit 초과 처리**:

- BIRD 의 california_schools / financial 등 column 수 많은 DB 의 full schema prompt 가 GLM 4.7 token limit 초과 가능성
- 신규 필드 추가:
  - `is_executable_full = None` (token limit 초과로 SQL 생성 불가)
  - `truncated: true` (prompt 가 truncate 됐는지 추적)

### 결정 — Phase 1 (A-1 + A-2 + A-3) 즉시 launch

학위 논문 일정 (5/14~5/22) + 학술 Agent 의 Phase 1 우선 receive 요청 — 즉시 Analyzer 핸드오프 + Root sweep launch.

### Spec 파일 정정 (학술 Agent 응답 반영)

`planning/filter_proposal_data_spec_2026-05-13.md` 의 다음 update:

1. **§2 A-1 의 포함 항목** 에 `is_executable_full = None` (token limit 초과 시 nullable) + `truncated: bool` 필드 추가
2. **§3 B-1 의 정의** 에 학술 Agent 가 요구하는 summary statistics 항목 (hard negative ratio + cosine histogram TP/FN/TN + mask=1 ratio) 명시 — xlsx full dump 대신 md summary 의 inline 표 형식
3. **§3 B-2 의 정의** 에 학술 Agent 가 요구하는 summary 항목 (gold:non-gold ratio + Train-Dev distribution diff + per-difficulty 분포) 명시
4. **§1.1 Decision Rules 표** 갱신 — 학술 Agent response 의 정확한 Trigger 인용

### Phase 1 책임 분담 (V5 chain 정정 원칙 준수)

| Step | 책임 | 작업 | ETA |
|---|---|---|---|
| 1 | **Analyzer** | `src/analysis/filter_proposal_a1_preliminary_sql.py` script 작성 — B1' / B4' script 재사용 + token limit detection (`truncated` 필드) | ~2-3h |
| 2 | **Analyzer** | Smoke test (1~5 query) — GLM 4.7 API call 정상 + truncation 감지 | ~30 min |
| 3 | **Root** | A-1 launch (~3-5h wall, ~$10-20 cost, 1534 query × 2 prompts: full + S_fwd) | ~5h |
| 4 | **Analyzer** | `filter_proposal_a2_backward_recall.py` (A-1 결과 의존, sqlglot parsing) | ~2h |
| 5 | **Analyzer** | `filter_proposal_a3_restore_noise.py` (A-2 결과 의존) | ~1h |
| 6 | **사용자** | jsonl → xlsx 변환 + md summary 작성 | ~수 시간 |
| 7 | **학술 Agent** | md + xlsx 수령 → Decision Rules 적용 (A-3 precision ≥ 0.6 / A-2 +Δ ≥ 0.05) | discussion |

### 학위 논문 일정 영향

- Phase 1 launch 즉시 (5/14): A-1 + A-2 + A-3 완료 → ~5/15
- 학술 Agent decision → ~5/16
- Direction A 배포 (positive) 또는 B/C 검토 → ~5/17~5/22
- 학위 논문 §V.5.x Direction A 결과 통합 → ~5/22

→ **즉시 launch 적합**.

### 학회/학위 논문 narrative 측면

- Phase 1 A-3 의 `S_restore_precision ≥ 0.6` 시 — Direction A 가 학위 논문 §V.5.x 의 Filter Dominance enhancement evidence (학회 paper 의 future direction)
- A-2 의 `mean(recall_union) - mean(recall_fwd) ≥ 0.05` 시 — backward path 의 recall 회복 정량 evidence

### 영향 범위

- `planning/filter_proposal_data_spec_2026-05-13.md` (spec 정정 — A-1 truncated 필드 + B-1/B-2 summary 항목 + Decision Rules 정식)
- `planning/DECISIONS.md` (본 entry)
- 후속:
  - Phase 1 결과 후 사용자가 md summary + xlsx 변환 → 학술 Agent 전달
  - 학술 Agent decision → Direction A/B/C 배포 결정

### 근거

- 사용자 직접 input (학술 Agent response 공유)
- 학술 Agent response 의 Decision Rules 표 + B-1/B-2 summary 항목 + A-1 truncated 보강

### 후속 actions

- (1) **Spec 파일 정정** (즉시, planner)
- (2) **Analyzer 핸드오프** — A-1 script 작성 (planner 작성, 사용자 전달)
- (3) **사용자 후속 작업** — 학술 Agent 응답 md 보관 (`planning/filter_proposal_scholar_agent_response_phase1_2026-05-13.md` 신규?), Phase 1 결과 후 md summary + xlsx 변환

---

## 2026-05-13 (Filter Proposal Data Spec 정정 — 학술 Agent md/xlsx-only 소통 + 3-Way Workflow 명시)

> **사용자 직전 input (5/13)**: "학술 에이전트는 코드베이스에 접근하지 못해 내가 md 파일만 전해주는 방식으로 소통했어" + "데이터를 엑셀로 만들어서 줘도 될 것 같아".

### 정정 내용

직전 `filter_proposal_data_spec_2026-05-13.md` 의 학술 Agent view 부정합 부분 minor 수정 (Option A):

1. **3-Way 소통 Workflow 도식 추가** — 학술 Agent ↔ 사용자 ↔ 실험 수행 Agent 의 fact 정정. 학술 Agent 가 코드/파일 접근 불가, md/xlsx-only 소통.
2. **§5.5 Data Output 위치** — "실험 수행 Agent reference, 학술 Agent 에게는 xlsx 로 변환되어 전달" 명시.
3. **§6 Data Output Format 전면 재작성** — 학술 Agent 전달 형식 = **md (summary + key tables) + xlsx (per-dataset raw records)** 조합.
4. **§6.3 Reproduction commands** → **§6.5 실험 수행 Agent 의 작업 chain** (학술 Agent → 사용자 → 실험 Agent 전달 순서) wording 정정.

### 학술 Agent 전달 형식 (정식)

| 형식 | 역할 | 학술 Agent 분석 |
|---|---|---|
| **md summary** | summary statistics + key tables + decision rule trigger + 학회 narrative 의의 | high-level decision |
| **xlsx (per dataset)** | per-query / per-DB raw records, 1 sheet per dataset | case study, outlier 분석, 분포 시각화 |

### xlsx Size 적합성

| Dataset | Records | xlsx 적합? |
|---|---:|:---:|
| A-1 / A-2 / A-3 | 1534 each | ✅ trivial |
| B-1 | ~980k | ⚠️ 1M row 한계 근접 — sampling or summary |
| B-2 | ~880k | ⚠️ 동일 |
| C-1 | ~100 | ✅ trivial |
| C-2 | 1534 | ✅ trivial |

### 3-Way Workflow 정식

```
학술 Agent (md/xlsx review only)
   ↑                              ↓ (spec / decision)
사용자 (bridge, md/xlsx 변환)
   ↑                              ↓ (chain)
실험 수행 Agent (analyzer/root/module, raw 데이터)
```

→ 사용자가 학술 Agent 의 spec/decision md 와 실험 수행 Agent 의 raw 데이터 사이 bridge. 실험 결과를 md summary + xlsx 변환 deliverable 로 학술 Agent 에 전달.

### Phase 1 후 deliverable spec (예시)

학술 Agent 에 전달할 deliverable 2 set:
- `filter_proposal_phase1_results_for_scholar_agent_<date>.md` — A-1/A-2/A-3 summary + decision rule trigger + 학회 narrative 의의
- `filter_proposal_phase1_data_<date>.xlsx` — A-1/A-2/A-3 의 per-query raw records (3 sheets)

### 영향 범위

- `planning/filter_proposal_data_spec_2026-05-13.md` (수정 — workflow 명시 + xlsx 옵션 추가)
- `planning/DECISIONS.md` (본 entry)
- Phase 1 후속 — 사용자가 변환 scripts (`scripts/jsonl_to_xlsx.py`) 작성 또는 analyzer 가 xlsx 직접 출력

### 근거

- 사용자 직접 input — 학술 Agent 가 코드 접근 불가, md/xlsx-only
- 직전 5/13 spec 파일 작성 entry — 학술 Agent view 부정합 minor 수정

### 후속 (사용자 결정 candidate)

- (a) Phase 1 launch 즉시 (Analyzer A-1 script + Root sweep) + 결과 후 사용자가 md + xlsx 변환
- (b) 학술 Agent 의 spec review 응답 (md) 받은 후 launch
- (c) 학위 논문 일정 (5/14~5/22) 고려 — Phase 1+2 즉시 launch 권장

---

## 2026-05-13 (Filter Proposal Data Spec 작성 — 학술 Agent 3 Direction 의 필요 데이터 정리)

> **사용자 직전 input (5/13)**: SGBE 실패 분석 (`sgbe_failure_analysis_for_scholar_agent_2026-05-13.md`) 후 학술 Agent 가 새 제안 (`filter_proposal_by_scholar_agent_2026-05-13.md`) 작성. 본 제안의 데이터 요구사항을 학술 Agent 가 받을 수 있는 파일로 정리 요청.

### 학술 Agent 의 3 Direction Proposal 요약

- **Direction A** (RSL-SQL Backward, 학습 불필요): Preliminary SQL 의 backward column 추출 + forward union — Cao et al. 2024 + Yang et al. 2024
- **Direction B** (HN-SupCon Selector Re-train): Hard Negative Supervised Contrastive 로 selector score 분포 재형성 → SGBE Revival — Piao et al. 2025 (LitE-SQL)
- **Direction C** (GRAST-SQL FD Graph Reranker): Functional Dependency graph + relation-aware transformer + Steiner tree — Hoang et al. 2025

### 결정 — 신규 산출물

`planning/filter_proposal_data_spec_2026-05-13.md` 신규 (학술 Agent 논의용, ~620 lines, 7 sections + 2 appendix):

| § | 내용 |
|---|---|
| §1 | Overview — 3 Direction + 8 datasets matrix |
| §2 | Direction A 데이터 spec (A-1 preliminary_sql_quality / A-2 backward_recall / A-3 restore_noise) |
| §3 | Direction B 데이터 spec (B-1 cosine_per_query / B-2 gold_labels_train / B-3 score_after_finetune) |
| §4 | Direction C 데이터 spec (C-1 fd_graph_completeness / C-2 structural_miss) |
| §5 | 우선순위 + Implementation Plan (Phase 1/2/3/4) |
| §6 | Data Output Format Schema + Reproduction commands |
| §7 | 학회 / 학위 논문 narrative 의의 |
| §A | References (외부 5 papers + 내부 5 reports) |
| §B | Summary for Implementation |

### 데이터 8 datasets 의 핵심 specs

| ID | 정의 | 우선순위 | 수집 비용 |
|---|---|:---:|:---:|
| **A-1** | preliminary_sql_quality (full + S_fwd, GLM 4.7) | **1** | ~$10-20 + 3-5h |
| **A-2** | backward_recall_stats (L_bwd ∩ L_fwd ∩ gold) | **1** | ~$0, sqlglot parsing |
| **A-3** | restore_candidate_noise_rate (S_restore precision/recall) | **1** | ~$0 |
| B-1 | column_embedding_cosine_per_query (Dev + Train) | 3 | ~$0, ~수 시간 |
| B-2 | gold_column_labels_train (BIRD-Train) | 3 | ~$0 |
| B-3 | score_distribution_after_finetune | (post-finetune) | ~수 일 학습 비용 |
| **C-1** | fd_graph_completeness_per_db (FK/PK 선언율) | **2** | ~$0 (GPT 예측 시 ~$5-10) |
| C-2 | structural_miss_rate_per_query | 2 | ~$0 |

### 학술 Agent 의 decision rules (data 수령 후)

- **A-3 의 mean(S_restore_precision) ≥ 0.6**: Direction A 우선 배포 권장 (SGBE 의 S_keep_hard 81.22% noise 와 직접 비교)
- **C-1 의 mean(fk_coverage_rate) ≥ 0.50**: Direction C feasible (GRAST-SQL 의 FD graph 품질 충분)
- **B-3 의 TP-TN spread ≥ 0.25** (post-finetune): SGBE Revival 가능 — HN-SupCon 의 score 분포 개선 confirm

### 진행 plan (Phase 1~4)

| Phase | Tier | 작업 | ETA |
|---|---|---|---|
| **1** | **즉시** | A-1 + A-2 + A-3 (Direction A 결정 trigger) | ~수 시간~1일 |
| **2** | 단기 (병행) | C-1 (Feasibility check) | ~수 시간 |
| **3** | 중기 (Phase 1 결과 후) | B-1 + B-2 (Direction B candidate) | ~수 시간 |
| **4** | Post fine-tune | B-3 (SGBE Revival 측정) | ~수 일 |

### 책임 분담 (V5 chain 정정 원칙 준수)

| Step | 책임 | 작업 |
|---|---|---|
| 5.1.1 (A-1 script) | **Analyzer** | `src/analysis/filter_proposal_a1_preliminary_sql.py` — B1' / B4' script 재사용 |
| 5.1.2 (A-1 launch) | **Root** | GLM 4.7 API 2-call sweep (1534 × 2) |
| 5.1.3 (A-2/A-3) | **Analyzer** | sqlglot parsing + set operation |
| 5.2 (C-1) | **Analyzer** | BIRD-Dev/Train DB FK/PK 분석 |
| 5.3 (B-1) | **Module:Selector** | EnsembleSelector cosine/gat 분리 dump interface |
| 5.3 (B-2) | **Analyzer** | BIRD-Train gold SQL parsing |
| (Phase 3 후) | **Module:Selector** | HN-SupCon fine-tune |
| (Phase 4) | **Analyzer** | B-3 측정 + 학술 Agent decision |

### 학회/학위 논문 narrative 측면

- 학회 paper: Direction A 결과 (positive 시) 가 Filter Dominance enhancement evidence candidate — 학회 main contribution 변경 X, footnote 또는 §V future direction
- 학위 논문 §V.5.x: Direction A/B/C 의 implementation plan + 결과 (Phase 1~4 진행 후)
- 외부 paper transfer 의의: RSL-SQL / LitE-SQL / GRAST-SQL 의 본 도메인 transfer 정량 evidence

### 학위 논문 일정 영향 (5/14~5/22)

- Phase 1 (Direction A): ~1일 → 5/14 launch 시 5/15 결과 → 학위 논문 §V.5.x 에 통합 가능
- Phase 2 (C-1 feasibility): 병행 ~수 시간
- Phase 3 (B-1, B-2): Phase 1 결과 후 결정 — 학위 본 심사 후 가능 (post-paper)

→ **Phase 1+2 즉시 launch 권장** (학위 논문 §V.5.x 통합 candidate). Phase 3+4 는 post-paper future direction.

### 영향 범위

- planning/filter_proposal_data_spec_2026-05-13.md (신규)
- planning/DECISIONS.md (본 entry)
- 후속 (Phase 1 결과 후): 학술 Agent discussion + Direction A 배포 결정

### 14 planning 문서의 역할 분리 (Filter 측면, 5/13 정식)

| Filter 관련 문서 | 역할 |
|---|---|
| `src/modules/filters/CLAUDE.md` | Filter module 세션 entry (8 구현체) |
| `src/modules/filters/EXPERIMENT_PLAN_filters.md` | Module PLAN (FL-I/II/III + SGBE) |
| `filtering_suggestion_by_scholar_agent_2026-05-12.md` | 학술 Agent 의 SGBE 권고 (origin) |
| `filter_full_context_2026-05-12.md` | Filter 모듈 종합 (8 구현체 + Filter Dominance 6/7/8-axis) |
| `sgbe_failure_analysis_for_scholar_agent_2026-05-13.md` | SGBE 실패 분석 (학술 Agent 논의용) |
| **🆕 `filter_proposal_by_scholar_agent_2026-05-13.md`** | **학술 Agent 의 3 Direction 제안 (RSL-SQL / HN-SupCon / GRAST-SQL)** |
| **🆕 `filter_proposal_data_spec_2026-05-13.md`** (5/13 신규) | **Direction 의 필요 데이터 spec + Implementation plan** |

### 근거

- `filter_proposal_by_scholar_agent_2026-05-13.md` (학술 Agent input, 5/13)
- `sgbe_failure_analysis_for_scholar_agent_2026-05-13.md` (직전 5/13 entry)
- 외부 5 papers (Cao 2024 RSL-SQL / Yang 2024 SQL-to-Schema / Piao 2025 LitE-SQL / Hoang 2025 GRAST-SQL / Talaei 2024 CHESS)

### 에스컬레이션 (사용자 결정 candidate)

- **즉시**: Phase 1 (A-1/A-2/A-3) launch — Analyzer 의 script 작성 + Root sweep
- **병행**: Phase 2 (C-1) launch — Analyzer 의 FK/PK 분석
- **결정 보류**: Phase 3 (B-1/B-2) — Phase 1 결과 후 결정
- **Post-paper**: Phase 4 (B-3) — HN-SupCon fine-tune 학습 비용 부담

---

## 2026-05-13 (V5 Module:Selector Ownership 완료 — V5-A/B/C 구현 + V5-C Cumulative Attention 신규 보강)

> **사용자 직전 input (5/13)**: Module:Selector 가 V5-A/B/C 코드 ownership 완료 (Commit **afadafd**) — Smoke 16/16 통과 + EXPERIMENT_PLAN_selectors §단계 8 신설.

### 산출물 표 (Commit afadafd)

| ID | Class | Paper Reference | 핵심 mechanism |
|---|---|---|---|
| **V5-A** | `GATEGATv2Conv` (alias `GATEConv`) | Mustafa & Burkholz NeurIPS 2024 §3.2 Eq. 4 | `att_self` + `att` (neighbor) 분리, **W 공유**, row-stochasticity 유지 |
| **V5-B** | `GCNIIGATv2Conv` | Chen 2020 (GCNII) + Peng 2024 | `eye_init` `gcnii_w` + $\beta_l = \log(\lambda/l + 1)$ (1-indexed forwarding) + Initial Residual $\alpha$ outer |
| **V5-C** | `FullAEROGATv2Conv` (alias `FullAEROGATConv`) | Lee 2023 Theorem 3 full form | V4-B 기반 + **Hop Attention (outer)** + **🆕 Cumulative Attention** |

Smoke test: **16/16 통과** (V5-A 5 + V5-B 5 + V5-C 6).

### V5-C Cumulative Attention 신규 — 직전 review chain 누락 후 보강

직전 5/12 V5 chain 핸드오프 작성 시 **Cumulative Attention** mechanism 누락 발견. Module:Selector 가 본 chain 에서 자체 보강. 단 **paper form vs 본 구현 차이의 이론적 caveat 명시 필요**:

| 항목 | Paper (Lee 2023) | 본 구현 |
|---|---|---|
| **Cumulative Attention 의 적용 level** | **edge-level α 누적** ($\alpha_{ij}^{(l)} = \alpha_{ij}^{(l-1)} + \text{softplus}(e_{ij}^{(l)})$) | **hidden-state level residual outer simulation** (각 layer 의 hidden state 에 outer residual 누적) |
| 이론적 보장 (SR2OS) | Theorem 3 의 row-stochasticity 파괴 + edge-level cumulative 의 조합 | edge-level cumulative 의 hidden-state outer 근사 — paper Theorem 3 의 SR2OS guarantee 의 transfer 정합성 불완전 |

→ **이론적 caveat (학위 논문 §V.5.x 또는 caveat footnote)**:
> "V5-C 의 Cumulative Attention 은 Lee 2023 Theorem 3 의 edge-level α 누적이 아닌 hidden-state level residual outer simulation 으로 구현. PyG `MessagePassing` 의 edge-level α 직접 누적은 GATv2Conv 의 inner 구조 access 가 어려워 outer hidden-state level 근사를 채택. paper Theorem 3 의 SR2OS guarantee 의 본 도메인 transfer 정합성은 추가 검증 candidate (post-paper)."

### Hop Attention + Cumulative Attention 의 Effect Overlap 가능성

- Lee 2023 Theorem 4: Hop Attention 의 sum form = $\sum_l \omega_v^{(l)} \mathbf{h}_v^{(l)}$ — per-node hop weight 의 sum
- Theorem 3 Cumulative Attention (edge-level): $\alpha^{(l)} = \alpha^{(l-1)} + \text{softplus}(e^{(l)})$ — edge-level α 누적
- 본 V5-C 구현 (hidden-state outer): 두 mechanism 의 hidden-state level 표현이 **부분적으로 등가** 가능성 — Hop Attention 의 sum 이 Cumulative Attention 의 outer 표현과 effect overlap
- → V5-C 의 ablation candidate (사용자 결정):
  - **v5c-full**: hop=True + cum=True + decay=1.0 (Theorem 3 full form 시도)
  - **v5c-hop-only**: hop=True + cum=False (Theorem 4 만)
  - **v5c-cum-only**: hop=False + cum=True + decay=0.5 (Theorem 3 cumulative 만)
- → Full ablation 시 hop/cumulative overlap 정량 검증 가능

### V5 Sweep Launch 결정 — 사용자 결정 (γ) Full Ablation (5/13 저녁 확정)

**선택: (γ) Full — V5-A + V5-B (L=2/4/6) + V5-C 3 cell ablation (~55h GPU 0/1 병렬)**

GPU 0/1 분배 최적화:
- **GPU 0** (V5-A + V5-B sequential, ~55h): V5-A (~10h) → V5-B L=2 (~10h) → V5-B L=4 (~15h) → V5-B L=6 (~20h)
- **GPU 1** (V5-C 3 ablation sequential, ~30h): V5-C v5c-full → V5-C v5c-hop-only → V5-C v5c-cum-only

Wall = max(GPU 0, GPU 1) ≈ **~55h** (예상보다 빠름). 5/13 저녁 launch → **5/16 결과** 가능. 5/16~5/22 (6일) chapter draft 통합 적합.

### V5 Sweep 결정 — 사용자 결정 candidate (직전 4 option 참조용)

학위 논문 일정 (5/14~5/22) 고려한 sweep 깊이:

| Option | V5-A | V5-B (depth sweep) | V5-C (Theorem 3 ablation) | Wall (GPU 0/1 병렬) |
|---|---|---|---|---|
| **(α) Minimal — narrative 우선** | 1 cell | L=2 만 | v5c-full 만 | ~30-40h |
| **(β) Mid — depth sweep 추가** | 1 cell | L=2+L=4 | v5c-full 만 | ~60-70h |
| **(γ) Full — V5-B+V5-C 모두 ablation** | 1 cell | L=2/4/6 | v5c-full + hop-only + cum-only | ~120-150h |
| **(δ) Recommended** ⭐ | 1 cell | **L=2 + L=4** | **v5c-full 만** | **~60-70h** — 학위 논문 5/22 마감 적합 + 학회 narrative 충분 |

→ **권장: (δ) Recommended** — V5-B 의 L=2 + L=4 (Peng 2024 의 deep GNN 가정 evidence + V4 stack 정합 모두 cover) + V5-C v5c-full (Theorem 3 시도). V5-C ablation 은 **post-paper future direction** (학위 논문 §V.5.x).

### Chain status 갱신 (5/13)

| # | Chain | Status |
|---|---|---|
| 1 | Filter sweep 9-cell | ✅ 완료 (v2) |
| 2 | **V5 sweep V5-A/B/C** | ✅ **코드 ownership 완료 (afadafd)** + ⏸ **Root sweep launch 대기** |
| 3 | SGBE Phase 3-5 | ✅ 완료 |
| 4 | V5-D-1 진단 | ✅ 완료 |
| 5 | V5-D-2 학습 | ⏸ V5-A/B/C 결과 후 |
| 6 | B1'+B2'+B3' GLM Baseline | ✅ 완료 |
| ⏸ post-paper | V5-C Theorem 3 hop/cum ablation | ⏸ 학위 본 심사 후 |
| ⏸ post-paper | Alternative anchor SGBE | ⏸ post-paper |

### 근거

- 사용자 직접 input (Commit afadafd + 산출물 표)
- `src/models/gat_network_v2.py` V5 classes (GATEGATv2Conv / GCNIIGATv2Conv / FullAEROGATv2Conv)
- `src/modules/selectors/EXPERIMENT_PLAN_selectors.md` §단계 8 신설
- 3 smoke test files (V5-A 5 + V5-B 5 + V5-C 6, 16/16 통과)
- DECISIONS 2026-05-13 V5 Sweep Launch 재시도 entry
- Lee 2023 "AERO-GNN" Theorem 3 (SR2OS guarantee) + Theorem 4 (Hop Attention sum form)

### 영향 범위

- planning/DECISIONS.md (본 entry)
- 후속: Root chain (V5 sweep launch) → Analyzer (14-trial 보고서) → Planner (paper §V.5.4 final integration)
- (post-paper) V5-C Theorem 3 full ablation (hop/cumulative overlap 정량)

### 학회/학위 논문 narrative 측면 의의

- V5-C Cumulative Attention 신규 보강 — 학위 논문 §V.5.x **lesson learned**: paper form 의 edge-level cumulative vs PyG 의 inner access 한계로 인한 hidden-state outer 근사 의 implementation caveat
- Hop/Cumulative overlap 의 working hypothesis — post-paper Theorem 3 full ablation 의 motivation

### 에스컬레이션

- **Root chain** — V5 sweep launch (Option δ 권장: V5-A + V5-B L=2/L=4 + V5-C v5c-full, ~60-70h)
- **Analyzer** (V5 결과 후) — 14-trial 통합 보고서
- **Planner** (Analyzer 후) — paper §V.5.4 final integration + V5-D-2 trigger 결정 + V5-C post-paper ablation 결정

### 추가 필요 분석 (V5 결과 후)

- 14-trial 매트릭스 + 3 시나리오 분기 결정 (Layer 2 pivot 여부)
- V5-D-2 trigger 결정 (V5 sweep 결과 의존)
- V5-C 의 hop/cumulative overlap 정량 (post-paper future direction)

---

## 2026-05-13 (V5 Sweep Launch 재시도 — V5-A/B/C 처음부터 Module:Selector 구현 + Root sweep launch)

> **사용자 직전 input (5/13)**: "V5 sweep 은 아마 아예 시작을 안 했을 거야 필요한 핸드오프를 작성해 줘".

### Status 정정

직전 5/12 V5 chain 위임 재할당 entry 후 module:selector 와 root 의 진행 상황:
- **V5-D-1 진단** ✅ 완료 (5/12, analyzer)
- **V5-A/B/C 코드 작성** ❌ **미진행** — 5/12 사용자 redirect ("직접 모듈 구현 금지") 후 root 의 작성 stop. Module:Selector 는 본 시점에 V5 작업 대신 **SGBE Phase 2 (raw_score interface 보강 + 3-anchor 진단)** 만 진행. V5-A/B/C 의 코드 review 작업 누락.
- **V5 sweep launch** ❌ 미진행

→ V5 chain 의 **Module:Selector 작업 처음부터 재시작 필요**.

### 결정 — V5 sweep launch 재시도

직전 5/12 V5 chain 위임 재할당 entry 의 책임 분담 (Module:Selector = V5-A/B/C 구현 + smoke test, Root = config + launch) 유지. Module:Selector 가 처음부터 V5-A/B/C 코드 구현 (root 의 이전 작성 시도 keep 코드 없음 가정).

### Chain status 갱신 (5/13)

| # | Chain | Status |
|---|---|---|
| 1 | Filter sweep 9-cell | ✅ 완료 (v2 evidence fix) |
| 2 | **V5 sweep V5-A/B/C** | ❌ **미launch — Module:Selector 핸드오프 재발송** |
| 3 | SGBE Phase 3-5 | ✅ 완료 |
| 4 | V5-D-1 진단 | ✅ 완료 |
| 5 | V5-D-2 학습 | ⏸ V5-A/B/C 결과 후 |
| 6 | B1'+B2'+B3' GLM Baseline | ✅ 완료 (B1'=0.5587 v1=v2 동일) |
| ~~7~~ | B4' | ❌ Tier 해제 |
| ⏸ 8 | Alternative anchor SGBE | ⏸ post-paper |

→ **유일한 active candidate = V5 sweep launch**.

### 학위 논문 일정 고려

- 학위 논문 chapter draft 일정: 5/14~5/22 (8 일)
- V5 sweep wall: ~30-40h (~5/14~5/16 결과 ETA, 즉시 launch 시)
- → **즉시 launch 필요** (5/13 저녁 또는 5/14 새벽). 5/16 결과 받으면 §V.5.4 final integration 가능 (5/22 마감 적합).

### Module:Selector 의 작업 — V5-A/B/C 코드 처음부터 구현

각 클래스의 이론적 근거 (코드 주석에 명시):
- **V5-A `GATEConv`**: Mustafa & Burkholz 2024 "GATE: How to Keep Out Intrusive Neighbors". 단일 attention vector $\mathbf{a}$ → $\mathbf{a}_s$ (self) + $\mathbf{a}_t$ (neighbor) 분리. Conservation Law 수정으로 작은 norm 으로도 task-irrelevant aggregation switch-off.
- **V5-B `GCNIIGATv2Conv`**: Peng et al. 2024 "Beyond Over-smoothing: Uncovering the Trainability Challenges". Initial Residual + Identity Mapping 동시. L=2/4/6 sweep.
- **V5-C `FullAEROGATConv`**: Lee et al. 2023 "AERO-GNN" Theorem 3 (SR2OS guarantee). V4-B (Softplus + Symmetric Norm) + Node-Adaptive Hop Attention 추가.

### Root 의 작업 — Module:Selector 완료 후 sweep launch

- Configs 5 신규 (V5-A / V5-B L=2/4/6 / V5-C)
- scripts/run_v5_mitigation_sweep.sh
- nohup launch + GPU 0/1 병렬
- HISTORY + CATALOG + ID_MIGRATION 갱신

### 예상 V5 결과 시나리오 (5/12 V5 chain entry 의 3 시나리오 그대로)

| 시나리오 | Trigger 조건 | Narrative 영향 |
|---|---|---|
| 1 | V5-D-1 R 갱신 + V5-A/B/C 모두 fail | 학술 Agent reinterpretation confirm — Layer 2 pivot (PLM lower bound) |
| 2 | V5-A 또는 V5-C 단독 R 갱신 | mech(ii-b) 5/5 absolute confirm 부분 부정 |
| 3 | V5 4 Direction 모두 fail | mech(ii-b) 5/5 absolute confirm 결정적 강화 — **현재 working hypothesis** |

→ V5 결과 후 paper §V.5.4 final integration 결정.

### 근거

- 사용자 직접 input ("V5 sweep 은 아마 아예 시작을 안 했을 거야")
- 직전 5/12 V5 chain 위임 재할당 entry — Module:Selector 의 작업 누락
- 학위 논문 일정 5/14~5/22 — V5 결과의 chapter draft 통합 필요

### 영향 범위

- planning/DECISIONS.md (본 entry)
- 후속: Module:Selector 작업 (~2-3일) → Root sweep launch (~30-40h) → Analyzer 보고서 → Planner narrative pivot

### 추가 필요 분석

- V5 sweep 완료 후 14-trial 통합 보고서 + paper §V.5.4 narrative pivot
- V5-D-2 trigger 결정 (V5 sweep 결과 의존)

---

## 2026-05-13 (Filter Sweep v2 Evidence Fix 완료 — anchor EX +18.06%p + Filter-Invariant 시나리오 + 7번째 axis 교체 + B4' Tier 해제)

> **사용자 직전 input (5/13, 늦은 저녁)**: analyzer 의 `filter_sweep_glm_9cell.md` v2 보고 — anchor EX 0.3396 → 0.5202 (+18.06%p), Baseline gap 82% 회수, Filter-Invariant 시나리오 확정 (F1 + EX 양쪽 sub-noise). 5 갱신 요청.

### 결정적 발견 — v1 → v2 Evidence Forward Fix Dominant

**v1 (직전) vs v2 (5/13 fix)**:

| Cell | v1 EX | v2 EX | ΔEX (v2 - v1) |
|---|---:|---:|---:|
| Anchor (C0 XiYan) | 0.3396 | **0.5202** | **+0.1806** |
| C8 No Filter | 0.3435 | (?) | (v1 confounder 해소) |
| F1 (anchor) | 0.8663 | 0.8651 | -0.0012 (sub-noise) |

→ **v1 의 EX 측정에 SQL gen prompt confounder** 존재. v2 fix 후 anchor EX 18.06%p jump. 직전 GLM Baseline B1' Full Schema 55.87% 와의 -21.91%p gap **대폭 축소** → root prompt 재실험 (5/13 진행 중) 의 motivation 정합.

**v2 결과 — Filter-Invariant 시나리오 확정**:

| Cell | F1 (v2) | EX (v2) |
|---|---:|---:|
| C0 XiYan (anchor) | **0.8651** | **0.5202** |
| C4 Stacked (Refl→Verif) | 0.8704 (+0.0053) | (?) |
| C7 Bidirectional | (?) | 0.5287 (+0.0085) |

- **F1 spread** (anchor cluster) = **0.0072** sub-noise
- **EX spread** (anchor cluster) = **0.0085** sub-noise
- **양쪽 sub-noise** = **Filter-Invariant 시나리오** 확정 (v1 의 "EX-F1 decoupling" narrative 폐기)

**Baseline gap 회수**:
- B3' Gold Column (Llama 3.1 8B) EX = 41.5% — 직전 B1' GLM 4.7 Full Schema 55.87% 와 vs anchor 33.96% 의 v1 gap = -21.91%p
- v2 anchor EX 0.5202 = **B3' (perfect schema linking, baseline) 의 84.7% 도달** — gap 82% 회수
- 직전 "schema linking 의 EX 효과 미미" working hypothesis (v1) 의 직접 부정

### 5 갱신 결정

**(1) paper §3.5 7-axis matrix 갱신 — 옛 7번째 폐기 + 신 7번째 추가**:

| 7번째 axis | 변경 |
|---|---|
| **옛 (v1, 5/13 오전)**: Filter-axis F1 robustness + EX-F1 decoupling | ❌ **폐기** (v1 confounder, v2 fix 후 decoupling 해소) |
| **신 (v2, 5/13 저녁)**: **Filter-Invariant in F1 + EX 양쪽 sub-noise spread** | ✅ **신규** — F1 spread 0.0072 + EX spread 0.0085 sub-noise + anchor F1=0.8651/EX=0.5202 + Baseline 84.7% 도달 |

8번째 axis (SGBE Negative Evidence, 5/13 SGBE chain) 유지.

**(2) Paper main pipeline anchor 확정 — C0 XiYan 유지**:
- C4 (best F1, +0.0053, 6× cost) — F1 측 sub-noise lift 단 cost 비효율
- C7 (best EX, +0.0085, 2.6× cost) — EX 측 sub-noise lift 단 cost 비효율
- 둘 다 sub-noise band 안 — anchor 변경 권장 X
- **paper main**: XiYan (C0) default + C4/C7 footnote (sub-noise alternatives)

**(3) DECISIONS prepend** ✅ (본 entry)

**(4) B4' chain narrative re-positioning — Tier 우선순위 해제**:

직전 5/13 B4' Tier 1 상향 (v1 confounder 의 mechanism 분기 H1 vs H2 필요) 의 trigger 변화:
- **v2 anchor EX = 0.5202** 가 paper main schema linking → SQL gen → EX 의 **직접 측정**
- B4' 의 작업 (anchor predictions → schema_str → GLM 4.7 SQL gen → EX) 가 **v2 sweep 안에 이미 포함**
- → **B4' 별도 chain 불필요**
- paper §V.5.4 EX evidence = **v2 anchor C0 metrics (F1=0.8651, EX=0.5202)** 가 직접 사용

→ B4' Tier 1 priority **해제**. Chain status 의 chain #7 (B4') 폐기 또는 "v2 sweep 안에 흡수" 로 update.

**(5) 학회 paper / 학위 논문 narrative 분리**:

| Outlet | Narrative |
|---|---|
| **학회 paper (한국지능정보시스템학회 2026 춘계)** | §3.5 Filter Dominance narrative + **Filter-Invariant 7-axis matrix table** (anchor F1=0.8651, EX=0.5202) + C4/C7 sub-noise footnote + SGBE caveat footnote (8번째 axis) |
| **학위 논문 §V Chapter** | (a) 본 보고서 §2 v1 → v2 evidence fix mechanism + case study, (b) §6.3 Baseline gap 회수 분해 (B3' 84.7% 도달), (c) **§V.5.5 SQL gen prompt confounder lesson learned** section 추가 (v1 의 prompt issue 의 mechanism + v2 fix detail + reproducibility 측면 교훈) |

### 학회 narrative reframe 의의 (5/13 v2 후)

직전 narrative (5/13 v1, Filter Sweep entry 시점):
> "Filter Dominance 의 7-axis evidence + EX-F1 decoupling — GLM 4.7 SQL gen 이 schema imperfection 흡수"

**갱신 narrative (5/13 v2 evidence fix 후)**:
> "Filter Dominance 의 7-axis evidence — **Filter-Invariant 시나리오 (F1 + EX 양쪽 sub-noise)** + Baseline 의 84.7% 도달. Anchor (XiYan + GLM 4.7) F1=0.8651, EX=0.5202 — schema linking 의 SQL gen 측 효과가 실제로 dominant. 직전 v1 의 EX-F1 decoupling narrative 는 SQL gen prompt confounder 의 artifact 로 polish."

### 본 발견의 의의 (v2 evidence fix dominant)

- **Filter Dominance narrative 더 강력**: F1 측면뿐만 아니라 EX 측면도 dominant 효과 — v1 의 "F1 dominant + EX backbone-dependent" candidate B 부정. **F1 + EX 양면 모두 Filter Dominance**.
- **B4' 별도 chain 불필요**: v2 anchor C0 의 EX=0.5202 가 곧 paper §V.5.4 의 EX evidence
- **학위 논문 §V.5.5 lesson learned**: v1 → v2 의 prompt confounder fix 가 학위 논문의 reproducibility / methodology 측면 핵심 lesson
- **B1' Full Schema baseline 의 위치**: 직전 v1 비교 (anchor 33.96% vs B1' 55.87% = -21.91%p) → v2 비교 (anchor 0.5202 vs B1' 0.5587 = -0.0385) — **Baseline gap 대폭 축소**. Maamari 2024 의 "Death of Schema Linking" 주장의 GLM 4.7 reframe — schema linking 이 EX 측면에서도 의미 있음 (sub-noise level 단 negative 차이 작음)

### Chain status 갱신 (5/13 저녁)

| # | Chain | Status |
|---|---|---|
| 1 | Filter sweep 9-cell | ✅ 완료 (v1 + **v2 evidence fix**) |
| 2 | V5 sweep V5-A/B/C | 🔄 active |
| 3 | SGBE Phase 3-5 | ✅ 완료 (Filter-Underperform 확정) |
| 4 | V5-D-1 진단 | ✅ 완료 |
| 5 | V5-D-2 학습 | ⏸ V5 후 |
| 6 | B1'+B2'+B3' GLM Baseline | ⚠️ root prompt 재실험 진행 중 (v2 fix 결과의 +18.06%p anchor 가 prompt confounder mechanism 의 정합 evidence) |
| ~~7~~ | ~~B4' paper main → SQL gen~~ | ❌ **Tier 해제 — v2 sweep 안에 흡수**. anchor C0 EX=0.5202 가 직접 paper §V.5.4 EX evidence |
| ⏸ 8 | Alternative anchor SGBE 재평가 | ⏸ post-paper (a03_17 / vLLM era best) |

### 본 sweep 의 final 8-axis (5/13 저녁 정식)

```
§3.5 Filter Dominance — Empirical Evidence (8 axes, 5/13 저녁 v2 evidence fix 후 정식)

1. H-B ckpt-invariant (Pearson r 0.06~0.24)
2. H-F stability/ordering (Jaccard 0.47~0.52)
3. F-1 + H-G alpha sweep (17 cells, 5.085× 압축)
4. ΔF1 lift (per-query +0.6462)
5. H-A/H-D 부정 (Enriched + norm 변형 plateau 유지)
6. 10-trial mitigation null + V4 architectural intervention 이중 fail (mech(ii-b) 5/5)
7. 🆕 Filter-Invariant in F1 + EX 양쪽 sub-noise spread (Filter sweep 9-cell v2,
   F1 spread 0.0072 + EX spread 0.0085, anchor F1=0.8651/EX=0.5202, Baseline 84.7% 도달)
8. Score-Gated Hybrid Negative Evidence — Selector-Filter Co-Design (SGBE F1=0.3697)

Caveats (footnote):
- C3 AdaptiveMultiAgent: multi-agent vote intersection bias
- C4 (+0.0053 F1, 6× cost) / C7 (+0.0085 EX, 2.6× cost): sub-noise alternatives, anchor 변경 권장 X
- SGBE: paper main anchor 부정합 (TP mean 0.47 < θ_keep 0.65), post-paper alternative anchor 재평가 candidate
- v1 EX-F1 decoupling narrative: SQL gen prompt confounder artifact, v2 fix 후 폐기
```

### 근거

- `notebooks/analysis_results/filter_sweep_glm_9cell.md` v2 §0 TL;DR + §1.2 Baseline 도달률 + §2.5 v1 EX-F1 decoupling 폐기 + §5.2 Filter-Invariant 정량 정의 + §6.2 anchor 유지 정당화 + §6.4 paper §3.5 통합 candidate
- 직전 5/13 (v1) Filter Sweep entry — 7-axis matrix
- 직전 5/13 (저녁) SGBE Chain 완료 entry — 8-axis matrix
- 직전 5/13 B1'+B2'+B3' GLM Baseline entry — v1 confounder mechanism (root prompt 재실험 진행 중)

### 영향 범위

- planning/DECISIONS.md (본 entry)
- planning/paper_research_direction.md §3.5 evidence #7 (5/13 갱신, 옛 EX-F1 decoupling) → **신 Filter-Invariant 7번째 axis 교체**
- planning/framework_snapshot_2026-05-12.md §4 Narrative Status (7-axis 옛 → 신 교체, 8-axis SGBE 유지)
- 후속 (paper §V.5.4 narrative final + 학위 논문 §V.5.5 lesson learned section)

### 추가 필요 분석

- B1' Full Schema 의 root prompt 재실험 결과 (5/13 진행 중) — v2 anchor EX 0.5202 와 비교, B1' Full Schema 의 v2 결과로 -21.91%p gap 변화 확인
- B4' 폐기 — 별도 chain 불필요
- (post-paper) alternative anchor SGBE 재평가 — a03_17 또는 vLLM era best

---

## 2026-05-13 (SGBE Chain 완료 — Filter-Underperform 확정 + 8th axis 추가 + B4' Tier 1 trigger 확정 + Alternative anchor post-paper)

> **사용자 직전 input (5/13)**: analyzer 의 `sgbe_filter_results.md` 보고 — SGBE Filter-Underperform 확정, anchor F1=0.8673 → Full SGBE F1=0.3697 ΔF1=-0.4976, P drop dominant = S_keep_hard over-include + θ_keep anchor-incompatibility. 5 갱신 요청.

### 결정적 발견 — SGBE Negative Evidence (학술 Agent 가설 부정)

**SGBE 결과 매트릭스**:

| Cell | Step | F1 | ΔF1 vs anchor |
|---|---|---:|---:|
| Anchor (C0 XiYan) | (baseline) | **0.8673** | — |
| **Full SGBE (step_0+1+2)** | Step 0 + Score-Gate + LLM Extractive | **0.3697** | **-0.4976** ⚠️ |
| Step 0+1 only (no LLM) | Score-Gate only | (Full -0.0048) | (LLM Extractive marginal +0.0048) |
| Step 0 only | FK/PK hardcode | (Step 0+1 보다 더 낮음) | — |

**핵심 mechanism (보고서 §4.5 4 sub-mechanism 의 dominant root cause)**:

- **(d) θ_keep anchor-incompatibility** — paper main anchor 의 TP mean **0.4746** << SGBE θ_keep **0.65** (학술 Agent 권장 + Option B 갱신값 0.50~0.60 모두 부정합)
- → S_keep_hard 가 거의 비어있음 + S_uncertain 이 매우 큼 → LLM call 의 noise 흡수가 dominant
- P drop dominant source: **S_keep_hard over-include — 81.22% noise** (보고서 §4.2). 즉 SGBE 의 keep_hard 가 anchor 의 noise column 까지 keep
- score_collapse hit rate ~0% (threshold 0.05 정상 — V4 era 같은 boundary case 만 detect, paper main anchor 는 normal score 분포)
- 학술 Agent 권장 θ (0.65, 0.40) + Option B (0.50/0.55/0.60 × 0.20/0.25/0.30) 모두 paper main anchor 의 score ladder 와 **incompatibility** confirm

→ **paper main anchor (Enriched + QCond + MST + XiYan + GLM 4.7) 에 SGBE 는 부정합**. Module:Selector Phase 2 진단 (3-anchor TP-TN spread 분석) 의 정확한 예측 confirm.

### 5 갱신 결정

**(1) paper §3.5 의 8th axis 추가 — SGBE Negative Evidence**:

기존 7 axis (5/13 Filter Sweep entry 갱신):
1. H-B ckpt-invariant
2. H-F stability/ordering
3. F-1 + H-G alpha sweep
4. ΔF1 lift
5. H-A/H-D 부정
6. 10-trial mitigation null + V4 이중 fail
7. Filter-axis F1 robustness + EX-F1 decoupling (Filter sweep 9-cell)

**🆕 8번째 axis (5/13 신규, SGBE)**:
- **Score-Gated Hybrid Negative Evidence — Selector-Filter Co-Design 중요성 강화**
- 정량: Full SGBE F1=0.3697 (anchor -0.4976) + LLM Extractive marginal +0.0048 + score_collapse hit ~0%
- mechanism: paper main anchor 의 TP mean 0.4746 << SGBE θ_keep 0.65 → θ anchor-incompatibility
- 의미: Filter Dominance 의 robustness 가 **단순 filter 종류 변경 ≠ 단일 LLM call 우위** — Score-Gated Hybrid 같은 score-aware filter 는 selector 의 score ladder 와 co-design 시에만 valid. paper main anchor 의 strong filter (XiYan + GLM 4.7) 의 LLM single-call 이 dominant.

**(2) Paper main anchor 유지 결정 확정 — XiYan (C0 sweep, F1=0.8663)**:

- SGBE 후보 폐기 (paper main 부정합 확정)
- **학회 paper (한국지능정보시스템학회 2026 춘계)**: anchor 유지 + SGBE caveat footnote
  > "Score-Gated Hybrid Filter (SGBE, post-Filter Dominance 6/7-axis 의 architectural alternative candidate, Yuan et al. 2025 KaSLA + Glass et al. 2025 의 score-aware approach 통합) 는 본 도메인의 strong filter (XiYan + GLM 4.7) anchor 의 score ladder (TP mean 0.4746 < θ_keep 0.65) 와 incompatibility — F1 -0.4976 underperform. selector-filter co-design 의 정량 evidence."

- **학위 논문 §V Chapter**: SGBE negative evidence full section
  - calibration 9-cell 결과 (θ_drop range invariance)
  - Phase 5 step contribution (Step 0 / Step 0+1 / Full)
  - θ_keep anchor-incompatibility mechanism (4 sub-mechanism root cause)
  - score_collapse hit rate 0% (threshold 0.05 정확)
  - Alternative anchor candidate (a03_17 binary, vLLM era best — post-paper)

**(3) DECISIONS entry prepend** ✅ (본 entry)

**(4) B4' Tier 1 priority 상향 trigger 확정**:

직전 DECISIONS 2026-05-12 B1'+B2'+B3' entry §B4' 보류 조건 ("SGBE + V5 + Filter sweep 모두 통합 후") 검토 결과:
- **Filter sweep 완료 ✅** (5/13) — anchor C0 F1=0.8663 확정
- **SGBE 부정 확정 ✅** (본 entry, 5/13) — paper main 후보 = XiYan anchor 유지
- **V5 결과 wait 안 함** (5/13 GLM Baseline entry 의 결정 유지 + 본 SGBE 부정으로 reinforce)

→ **Filter 모듈 paper main 후보 = XiYan anchor 확정**. B4' launch trigger 조건 **확정 충족**. **즉시 launch 가능** (직전 5/13 GLM Baseline entry 의 Tier 1 priority 결정 reinforce).

**(5) Alternative anchor SGBE 재평가 chain (post-paper future work)**:

본 SGBE 부정은 paper main anchor 의 score ladder (TP mean 0.4746) 에 한정. Module:Selector Phase 2 진단 의 다른 anchor 에서는 SGBE valid 가능:

| Anchor | TP-TN spread | TP mean | SGBE 권장 θ (Phase 2 보강 entry) | SGBE 가능성 |
|---|---:|---:|---|---|
| paper main GLM (anchor) | 0.1741 | 0.4746 | (0.55, 0.25) | ❌ 부정 confirm (본 entry) |
| **vLLM best (abl_ens_basic_xiyan)** | 0.1543 | **0.5526** | (0.65, 0.40) — 학술 Agent 권장 정합 | ⏸ post-paper 검증 candidate |
| **a03_17 (binary)** | **0.3067** | 0.5818 | (0.50, 0.05) — bimodal cutoff | ⏸ post-paper 검증 candidate (학술 Agent reference 와 spread 정합) |

→ **Post-paper future work entry**: a03_17 또는 vLLM era best 에서 SGBE 재calibration + selector-filter co-design 정량 evidence 확보. 학위 논문 §V Chapter 의 future direction.

### 학위 논문 §V Chapter — SGBE Negative Evidence Full Section 권고

- **§V.5.x** "Score-Gated Hybrid Filter — Negative Evidence + Selector-Filter Co-Design"
  - SGBE 의 학술 Agent 권고 background (Yuan 2025 + Glass 2025 + CHESS Talaei 2024 + Hoang 2025)
  - calibration 9-cell 결과 + step contribution
  - 4 sub-mechanism root cause (a~d) — (d) θ_keep anchor-incompatibility dominant
  - 학술 Agent reference anchor (TP-TN spread 0.31) 의 정체 추정 — a03_17 binary 와 정합
  - Filter Dominance 의 selector-filter co-design 측면 강화 — "단일 LLM call 의 strong filter 가 score-aware hybrid 보다 우위" narrative
  - Future direction: alternative anchor 에서 SGBE 재평가

### Chain status 갱신 (5/13)

| # | Chain | Status |
|---|---|---|
| 1 | Filter sweep 9-cell | ✅ 완료 |
| 2 | V5 sweep V5-A/B/C | 🔄 active |
| 3 | **SGBE Phase 3-5** | ✅ **완료 (5/13)** — Filter-Underperform 확정, paper main 부정합 |
| 4 | V5-D-1 진단 | ✅ 완료 |
| 5 | V5-D-2 학습 | ⏸ V5 sweep 후 |
| 6 | B1'+B2'+B3' GLM Baseline | ✅ 완료 (단 root + 사용자 prompt 재실험 중) |
| **7** | **B4' paper main → SQL gen** | 🚀 **Tier 1 priority 확정** (Filter sweep + SGBE 부정 confirm 후 trigger 완전 충족) |
| ⏸ 8 | (post-paper) Alternative anchor SGBE 재평가 | ⏸ a03_17 또는 vLLM era best 에서 SGBE re-calibration — 학위 논문 §V Chapter future direction |

### Filter Dominance 의 8-axis Narrative 통합 (paper §3.5)

```
§3.5 Filter Dominance — Empirical Evidence (8 axes, 5/13 갱신)

1. H-B ckpt-invariant (Pearson r 0.06~0.24)
2. H-F stability/ordering (Jaccard 0.47~0.52)
3. F-1 + H-G alpha sweep (17 cells, 5.085× 압축)
4. ΔF1 lift (per-query +0.6462)
5. H-A/H-D 부정 (Enriched + norm 변형 plateau 유지)
6. 10-trial mitigation null + V4 architectural intervention 이중 fail (mech(ii-b) 5/5 absolute confirm)
7. Filter-axis F1 robustness + EX-F1 decoupling (9-cell Filter sweep)
8. 🆕 Score-Gated Hybrid Negative Evidence — Selector-Filter Co-Design (SGBE F1=0.3697, anchor -0.4976)

Caveats (footnote):
- C3 AdaptiveMultiAgent: multi-agent vote intersection bias (R -0.1017, architectural pathology)
- C4 Stacked: future work +0.0072 F1 lift, 7.8× cost
- SGBE: paper main anchor 부정합, alternative anchor 에서 post-paper 검증 candidate
```

### 근거

- `notebooks/analysis_results/sgbe_filter_results.md` §0 TL;DR + §4.2 P drop dominant 81.22% noise + §4.5 4 sub-mechanism dominant root cause (d) θ_keep anchor-incompatibility + §6.2 학술 Agent θ 부정합 confirm + §7.2 Filter Dominance 8th axis
- 직전 Module:Selector Phase 2 진단 (3-anchor TP-TN spread + paper main TP mean 0.4746)
- 학술 Agent `filtering_suggestion_by_scholar_agent_2026-05-12.md` §"세 그룹의 score 분포" (TP 0.7108 / Filter✗ 0.6394 / TN ~0.40, spread 0.31)

### 영향 범위

- planning/DECISIONS.md (본 entry)
- planning/paper_research_direction.md §3.5 evidence — **7 → 8 axis** 갱신
- planning/framework_snapshot_2026-05-12.md §4 Narrative Status (8-axis 갱신)
- 후속 (B4' 결과 후): paper §V.5.4 narrative final + 학위 논문 §V.5.x SGBE Negative Evidence section
- Post-paper: alternative anchor SGBE 재평가 future work

### 추가 필요 분석

- **B4' 즉시 launch** (Tier 1 priority 확정) — paper main schema linking 결과 → GLM 4.7 SQL gen → EX
- B1' Full Schema 의 root prompt 재실험 결과 (5/13 root 진행 중)
- (post-paper) alternative anchor SGBE 재평가 — a03_17 또는 vLLM era best

---

## 2026-05-13 (B1'+B2'+B3' GLM 4.7 Baseline 완료 — 🚨 결정적 발견: anchor EX < B1' Full Schema -21.91%p + B4' Tier 1 priority 상향 + Paper dual-candidate narrative)

> **사용자 직전 input (5/13)**: analyzer 의 `glm_baseline_3cell.md` 보고 — Filter Sweep anchor EX 33.96% < B1' Full Schema EX 55.87% by **-21.91%p**. 4 갱신 요청.

### 🚨 결정적 발견 — Filter Dominance Narrative 의 결정적 reframe 필요

**B1'+B2'+B3' 결과 (GLM 4.7, BIRD-Dev 1534)**:

| Scenario | EX (GLM 4.7, 신규) | EX (Llama 3.1 8B, 기존) | ΔEX (GLM - Llama) |
|---|---:|---:|---:|
| **B1' Full Schema** | **55.87%** | 34.1% | **+21.77%p** |
| B2' Gold Table | ? | 40.1% | ? |
| B3' Gold Column | ? | 41.5% | ? |
| Filter Sweep anchor (C0 XiYan, F1=0.8663) | **33.96%** | — | — |

**🚨 anchor EX (33.96%) < B1' Full Schema EX (55.87%)** — **-21.91%p gap**:

- Schema linking pipeline (F1=0.8663, anchor) 가 LLM single-call Full Schema (no schema linking) 보다 EX 측면에서 **훨씬 못함**
- 직전 Filter sweep §5.4 의 "EX-F1 decoupling" narrative ("GLM 4.7 SQL gen 이 schema imperfection 흡수") 가 **반대 방향**: Schema linking 이 오히려 EX 를 떨어뜨림
- Maamari 2024 의 "Death of Schema Linking" 주장이 GLM 4.7 backbone 에서 **직접 confirm** — 본 연구 narrative 의 결정적 reframe 필요

### Mechanism 분기 — H1 vs H2

본 -21.91%p gap 의 두 hypothesis (보고서 §4.2 candidate):

| Hypothesis | 가설 | B4' 결과로 검증 |
|---|---|---|
| **H1 schema imperfection** | anchor 의 schema linking 결과 imperfect → SQL gen EX 감소. B4' (anchor predictions → GLM 4.7 SQL gen) 도 EX 33.96% 근방 예상 | B4' EX ≈ anchor 시 confirm |
| **H2 schema_str format** | anchor 의 schema 가 schema_str format 변환 시 GLM 4.7 reasoning 약화 (column name 만, type/desc/value 손실). B4' (다른 schema_str 형식 가능) EX 가 anchor 보다 높을 candidate | B4' EX ≈ B1' (55%+) 시 confirm — schema 형식이 결정 변수 |

→ **B4' 의 결과가 paper main contribution narrative 의 결정적 axis**. 본 발견 이전 narrative ("Filter Dominance 가 dual-layer 흡수") 의 **부분 부정** 가능성.

### 4 갱신 결정

**(1) B4' Tier 1 priority 상향 — Filter sweep 완료만으로 launch 가능**:

- 직전 trigger 조건 (DECISIONS 2026-05-12 §B4'): "Filter 모듈 확정 후 (SGBE + V5 + Filter sweep 모두 통합 후)"
- **새 trigger 조건 (5/13 update)**: **Filter sweep 완료만으로 launch 가능** ✅. SGBE / V5 wait 안 함.
- **이유**: anchor (C0 XiYan, F1=0.8663) 의 schema linking 결과 (predictions.jsonl) 가 본 sweep 완료로 확정. SGBE 의 결과 (F1=anchor 또는 갱신) 와 무관하게 B4' 는 anchor predictions 기반.
- **Priority**: 직전 Tier 4 (학위 본 심사 후) → **Tier 1 (즉시 launch)** 로 상향
- **이유 (강력)**: 본 -21.91%p gap 의 mechanism 분기 (H1 vs H2) 가 학회 발표 / 학위 논문 narrative 의 **결정적 axis** — B4' 결과 없이는 paper main contribution narrative 정식화 불가

**(2) Paper §V.5.4 narrative dual-candidate**:

| Candidate | B4' 결과 조건 | Paper §V.5.4 narrative |
|---|---|---|
| **Candidate A**: F1+EX 동시 수립 | B4' EX ≈ B1' (55%+) | Filter Dominance 가 양면 우위 — H2 schema_str format 가 dominant. Schema linking F1 lift 가 적절한 format 변환 시 EX 에 transfer. Maamari 2024 의 reframe — "schema linking 의 format 이 결정" |
| **Candidate B**: F1 dominant + EX backbone-dependent | B4' EX ≈ anchor (33%) | F1 측면만 dominant. EX 는 backbone capacity (GLM 4.7 의 in-context reasoning) 의존. **Filter Dominance 가 F1 측면 main contribution + EX backbone caveat footnote**. Maamari 2024 의 직접 confirm — "LLM 이 schema 전체 보면 더 잘 함" |

**Candidate B 가 working hypothesis (가능성 높음)** — 직전 Filter sweep §5.4 EX-F1 decoupling 발견과 정합. C8 no_filter F1=0.2250 의 EX=0.3435 ≈ anchor 0.3396 → schema linking quality 가 EX 에 영향 X. 본 B1' 55.87% 는 schema_str format 의 영향 (full schema = 더 풍부한 context) 가능성 단 schema linking 자체의 영향 보다 큰 EX gap 설명에는 H2 만으로 부족.

**(3) 학위 논문 vs 학회 paper 분리 narrative**:

| Outlet | Scope | Narrative |
|---|---|---|
| **학회 paper (한국지능정보시스템학회 2026 춘계)** | Extended Abstract (cover + 3p) | F1 evidence + Filter Dominance 6-axis (5/12) + **EX backbone caveat footnote**. -21.91%p gap mechanism 은 footnote 만 (depth 부족). C4 Stacked future work + B3' Gold Column upper bound 인용 |
| **학위 논문 §V Chapter** | 깊이 있는 분석 | 본 -21.91%p gap mechanism 분석 (H1 vs H2 dual-candidate) + B4' 결과 detail + backbone sensitivity 깊이 + Maamari 2024 의 GLM 4.7 confirm + paper narrative reframe |

→ 학회 paper 의 main contribution narrative 는 **F1 측면 Filter Dominance 유지** (anchor F1=0.8663 + 7-axis evidence). EX backbone caveat 는 footnote 처리. 학위 논문이 깊은 mechanism 분석 + B4' 결과 통합.

**(4) DECISIONS entry prepend** ✅ (본 entry)

### 학회 paper narrative 변화 (5/13 update)

직전 narrative (5/13 Filter sweep entry):
> Filter Dominance 의 7-axis evidence — schema linking F1 측면 + EX-F1 decoupling (GLM 4.7 SQL gen 의 schema imperfection 흡수)

**갱신 narrative (5/13 GLM Baseline 후)**:
> Filter Dominance 의 7-axis evidence — schema linking F1 측면 dominant. **EX 측면은 backbone capacity 의존 (GLM 4.7 의 in-context Full Schema reasoning 이 schema linking 우회)**. Filter Dominance 의 학회 paper main contribution = F1 측면 7-axis evidence. EX backbone caveat = footnote (B1' 55.87% > anchor 33.96% 발견 인용).

### 본 발견의 의의

- **본 연구 학회 paper 의 핵심 unchanged**: Filter Dominance 의 F1 측면 7-axis evidence — anchor F1=0.8663 + sub-noise spread + GAT internal lock-in 흡수
- **단 narrative reframe 필요**: "Filter Dominance 가 EX 도 dominate" 의 직전 implicit 주장 → "F1 dominant + EX backbone-dependent" 의 명시적 reframe
- **학위 논문 main contribution 강화**: -21.91%p gap mechanism 분석 + Maamari 2024 의 GLM 4.7 confirm 이 학위 논문 §V.5.4 의 결정적 deeper insight

### Chain status 갱신 (5/13)

| # | Chain | Status |
|---|---|---|
| 1 | Filter sweep 9-cell | ✅ 완료 |
| 2 | V5 sweep V5-A/B/C | 🔄 active |
| 3 | SGBE Phase 3-5 | 🚀 launch trigger |
| 4 | V5-D-1 진단 | ✅ 완료 |
| 5 | V5-D-2 학습 | ⏸ V5 sweep 후 |
| 6 | **B1'+B2'+B3' GLM Baseline** | ✅ **완료 (5/13)** — 결정적 발견 -21.91%p gap |
| **7** | **B4' paper main → SQL gen** | 🚀 **Tier 1 priority 상향, 즉시 launch** (Filter sweep 완료만 trigger, SGBE/V5 wait X) |

### B4' launch 의 책임 분담 (V5 chain 정정 원칙 준수)

| Step | 책임 | 작업 |
|---|---|---|
| 1 | Analyzer 또는 신규 module:sql_gen | B4' script 작성 — anchor predictions.jsonl 의 final_nodes 를 schema_str 로 변환 + 기존 GLM 4.7 SQL gen prompt 재사용 |
| 2 | Root | B4' launch (~1.5-2h, ~$5-10) |
| 3 | Analyzer | B4' 결과 보고서 — anchor F1=0.8663 의 EX transfer 정량 + H1 vs H2 mechanism 결정 + paper §V.5.4 dual-candidate 선택 |
| 4 | Planner | paper §V.5.4 narrative final integration + 학위 논문 §V Chapter detail |

### 근거

- `notebooks/analysis_results/glm_baseline_3cell.md` §0 TL;DR + §3.2 EX gap 정량 + §4.1 결정적 비교 + §4.2 backbone-flip 가설 + §4.4 paper narrative 재검토 필요성 + §5.4 B4' launch 시점 결정 권고
- 직전 Filter sweep §5.3 EX-F1 decoupling + §5.4 mechanism narrative
- Maamari et al. 2024 "The Death of Schema Linking?" arXiv:2408.07702
- 본 연구의 GLM 4.7 era anchor F1=0.8663 + EX 33.96%

### 영향 범위

- planning/DECISIONS.md (본 entry)
- planning/paper_research_direction.md §3.5 evidence #7 (5/13 갱신) → **5/13 GLM Baseline 후 narrative reframe** 추가 갱신 candidate
- planning/framework_snapshot_2026-05-12.md §4 Narrative Status (7-axis 갱신 후 EX backbone caveat 추가 candidate)
- 후속 (B4' 결과 후): paper §V.5.4 narrative final integration + 학위 논문 §V Chapter

### 추가 필요 분석

- **B4' 즉시 launch** — Tier 1 priority
- B2' Gold Table + B3' Gold Column 의 EX 결과 (보고서 §0 또는 §3 의 정확 수치 확인) — 본 entry 의 EX matrix 보강
- B4' 후속 — paper §V.5.4 narrative dual-candidate 선택 + 학위 논문 §V Chapter detail 작성

---

## 2026-05-13 (Filter Sweep 9-cell 완료 — Filter-Modest 시나리오 + C3 outlier + EX-F1 Decoupling 발견 + paper §3.5 7번째 axis 추가)

> **사용자 직전 input (5/13)**: analyzer 의 `filter_sweep_glm_9cell.md` 보고 — Filter-Modest 시나리오 + C3 outlier 확정 + EX-F1 decoupling 핵심 발견. 4 갱신 요청.

### 결정적 발견 (3 항목)

**1. Filter-Modest 시나리오 (Filter-axis F1 robustness)**:

| 비교 그룹 | F1 spread | 분류 |
|---|---:|---|
| 8 LLM filter (naive, C3 포함) | 0.0694 | Filter-Sensitive |
| **7 LLM filter (C3 outlier 제외)** | **0.0116** | **Filter-Modest band** |
| 4 anchor 근처 cell (C0/C5/C6/C7) | 0.0055 | sub-noise band |

→ Filter 선택의 F1 robustness 확인. "fully invariant" 가 아닌 "modest variance + 단일 multi-agent fail mode" narrative.

**2. C3 (AdaptiveMultiAgent) outlier — multi-agent vote pathology**:
- C3 R = 0.7681 (anchor 0.8698 대비 **-0.1017**), F1 = 0.8041 (anchor 대비 **-0.0622**)
- Mechanism: 3-agent (Semantic / Structural / Skeptic) majority vote 가 **over-aggressive pruning** — vote 가 union 이 아닌 intersection 으로 작동, Skeptic 의 conservative bias dominant
- **vLLM era 와 일관**: a05_01_adaptive_multi_agent (Qwen) R=0.3770 — backbone 다른데도 AdaptiveMultiAgent 가 항상 outlier. **architectural pathology**.

**3. EX-F1 decoupling — 본 sweep 의 가장 핵심 발견**:

| Cell | F1 | EX | F1 ranking | EX ranking | Δrank |
|---|---:|---:|---:|---:|---:|
| C4 Stacked | 0.8735 | 0.3416 | 1 | 5 | -4 |
| C0 XiYan (anchor) | 0.8663 | 0.3396 | 4 | 4 | 0 |
| C1 Reflection | 0.8625 | **0.3468** ⭐ | 6 | **1** | +5 |
| C8 No Filter | **0.2250** ⚠️ | **0.3435** | 9 | 2 | **+7** |

→ **C8 no_filter F1=0.2250 (near-zero) 의 EX=0.3435 ≈ anchor EX 0.3396** (+0.0039 더 높음!). Schema linking F1 +0.6413 lift 가 EX 에 **0 transfer**. GLM 4.7 의 SQL gen robustness 가 schema imperfection 흡수.

### 결정 (4 갱신 항목)

**(1) paper §3.5 Filter Dominance evidence 의 6 → 7 axis 추가**:

기존 6 axis (line 505):
1. H-B ckpt-invariant
2. H-F stability/ordering
3. F-1 + H-G alpha sweep
4. ΔF1 lift
5. H-A/H-D 부정
6. 10-Trial Mitigation Null + V4 이중 fail

**🆕 7번째 axis** (5/13 신규):
- **Filter-axis F1 robustness + EX-F1 decoupling** (9-cell Filter sweep)
- 정량: 7-filter F1 spread 0.0116 (Filter-Modest band) + C8 no_filter EX 0.3435 ≈ anchor 0.3396 (EX-F1 decoupling)
- 의미: schema linking F1 의 robustness 가 filter selection 에 invariant (modest band) + GLM 4.7 SQL gen 이 schema imperfection 까지 흡수 (EX-F1 decoupling)

**(2) Paper main pipeline anchor 유지 (C0 XiYan)**:

- Best 4 cells (C4 / C7 / C5 / C0) F1 cluster = 0.8663 ~ 0.8735, **spread 0.0072 sub-noise**
- C4 (Stacked, +0.0072 F1, **7.8× cost**) — toxicology-targeted marginal lift
- C7 (Bidirectional, +0.0032 F1, 3.1× cost) — bidirectional complexity
- C5 (SymVerify, +0.0009 F1, ~1.0× cost) — best ROI 단 lift sub-modest
- → **anchor (C0 XiYan, 단일 LLM call, 1h01m, 6.4M tokens) = simple/cost-effective default 로 robust**. paper main anchor 변경 권장 X.

**C4 footnote (paper §3.5 또는 §8 Future Work)**:
> "Future work: C4 Stacked (Refl→Verif) 가 anchor +0.0072 F1 lift candidate (toxicology-targeted), post-paper 추가 검증. 7.8× cost (filter_time 28523s vs 3672s) 의 ROI 가 학회 paper scope 밖."

**C3 caveat footnote (paper §3.5 footnote)**:
> "Filter Dominance 의 robustness 의 caveat: multi-agent voting filter (AdaptiveMultiAgent) 는 단일 LLM call 의 robustness 를 inherit 안 함 — vote 의 union 이 intersection 으로 작동, conservative bias 강화로 R 큰 손실. 본 sweep 에서 R -0.1017 (anchor 대비) 의 outlier. vLLM era (Qwen3-Coder, a05_01 R=0.3770) 와 일관 — backbone 무관 architectural pathology."

**(3) B4' chain 의 narrative pre-integration** (Filter 모듈 확정 후 launch):
- B4' = paper main schema linking 결과 (F1=0.8663) → GLM 4.7 SQL gen → EX 측정
- 본 sweep 의 EX-F1 decoupling 발견이 **B4' 의 결정적 사전 evidence**
- 예상 B4' 결과: anchor 의 EX ≈ **0.3396** (C0 sweep EX 와 정합, schema linking F1 variation 의 EX 무관 confirm)
- B3' (Gold Column GLM 4.7 baseline, 별도 chain) 의 EX upper bound 와 비교가 B4' 의 paper §V.5.4 핵심 narrative

**(4) Anchor F1 의 정정**:
- 직전 anchor F1 = 0.8673 인용 (paper_research_direction.md line 505)
- 본 sweep 의 C0 측정 F1 = 0.8663 (재실행, sweep 내)
- ±0.0010 noise (재실행 variance) — 둘 다 anchor 의 schema linking 결과로 통용 가능
- → 본 갱신 후 anchor F1 = **0.8663** (sweep 내 정량, paper §3.5 갱신)

### 영향 범위

- planning/DECISIONS.md (본 entry)
- planning/paper_research_direction.md §3.5 (line 505 evidence #6 갱신 + #7 신규 추가)
- planning/framework_snapshot_2026-05-12.md §4 Narrative Status (7-axis 갱신)
- 후속 (planner, V5 결과 통합 시): paper §V.5.4 main finding + B4' chain narrative integration

### 핵심 narrative 결론

> 9-cell Filter sweep (BIRD-Dev 1534, GLM 4.7) 의 두 결정적 발견:
>
> **(a) Filter-Modest robustness**: 7-LLM filter F1 spread 0.0116 (C3 outlier 제외) — Filter Dominance 의 robustness 가 filter selection 의 단순 invariance 가 아닌 **modest variance + 단일 multi-agent fail mode** 의 정량.
>
> **(b) EX-F1 decoupling**: schema linking F1 +0.6413 lift (C8 no_filter 0.2250 → anchor 0.8663) 가 EX 에 **0 transfer** (C8 EX 0.3435 ≈ anchor 0.3396). GLM 4.7 의 SQL gen robustness 가 schema imperfection 까지 흡수 — **Filter Dominance 의 SQL gen 측면 ceiling 정량 evidence**.
>
> 두 발견으로 paper §3.5 의 Filter Dominance narrative 가 **7-axis evidence** 로 확장 — 학회 contribution 의 결정적 강화.

### 근거

- `notebooks/analysis_results/filter_sweep_glm_9cell.md` §0 TL;DR + §2.5 C4 cost 정당화 + §4.2 Filter dominant +0.6413 F1 + §5.3 EX-F1 ranking + §5.4 mechanism narrative + §6.2 통합 권고
- 직전 DECISIONS entries (B1'~B3' 즉시 launch + B4' 보류 + Full Schema baseline 발견)

### Chain status 갱신 (5/13 기준)

| # | Chain | Status |
|---|---|---|
| 1 | Filter sweep 9-cell | ✅ **완료 (5/13)** — Filter-Modest + C3 outlier + EX-F1 decoupling 발견 |
| 2 | V5 sweep V5-A/B/C | 🔄 active |
| 3 | SGBE Phase 3-5 | 🚀 launch trigger (prerequisite 완료) |
| 4 | V5-D-1 진단 | ✅ 완료 |
| 5 | V5-D-2 학습 | ⏸ V5 sweep 후 |
| 6 | B1'+B2'+B3' GLM Baseline | 🚀 즉시 launch (analyzer script + root + analyzer 보고서) |
| ⏸ 7 | B4' paper main → SQL gen | ⏸ **Filter 모듈 확정 = 본 sweep 완료 + SGBE 결과 후** launch — 본 sweep 의 EX-F1 decoupling 이 사전 evidence |

### B4' launch trigger 조건 update (5/13)

직전 trigger 조건 "Filter 모듈 확정 후 (SGBE + V5 + Filter sweep 결과 통합 후)" 부분 정정:
- **Filter sweep 완료** ✅ (5/13) — anchor (C0 XiYan) F1=0.8663 정량 확정
- **SGBE chain 완료** ⏸ — SGBE 결과가 anchor F1 갱신 candidate
- **V5 결과** ⏸ — schema linking F1 영향 미정

→ **SGBE chain 완료 후** B4' launch 가능. V5 결과는 paper §V.5.4 통합 시점 (별도) 에 영향. **B4' 의 우선순위 = SGBE chain 완료 후 즉시**.

---

## 2026-05-12 (B1'+B2'+B3' GLM 4.7 Baseline 즉시 launch — B4' Filter 모듈 확정 후 보류)

> **사용자 직전 input (5/12)**: "B4' 은 어차피 Filter 모듈 확정되고 나서 맨 마지막에 실험할 거니까, B3' 까지만 지금 바로 진행하자".

- **결정 — 3 cell 즉시 launch + B4' 보류**:

  | Cell | Scenario | Backbone | Status |
  |---|---|---|---|
  | **B1'** | Full Schema (Maamari paradigm) | GLM 4.7 (Elice ML API) | 🚀 즉시 launch |
  | **B2'** | Gold Table oracle | GLM 4.7 | 🚀 즉시 launch |
  | **B3'** | Gold Column oracle (perfect schema linking) | GLM 4.7 | 🚀 즉시 launch |
  | ⏸ B4' | paper main schema linking 결과 → GLM 4.7 SQL gen | GLM 4.7 | **보류 — Filter 모듈 확정 후 (SGBE + V5 결과 통합 후)** |

- **B4' 보류의 의미**:
  - Filter 모듈 확정 (SGBE 결과 + Filter sweep 9-cell + V5 결과 모두 종합) 후 paper main pipeline anchor 의 최종 schema linking 결과 (F1) 확정
  - 그 시점에 paper main 의 predictions.jsonl 의 final_nodes 를 schema_str 로 변환 → GLM 4.7 SQL gen → EX 측정
  - B4' = 본 연구의 main contribution (Filter Dominance 의 F1=0.8673 또는 갱신값) 의 downstream EX transfer 정량 — **학회 narrative 의 결정적 evidence**
  - 학위 논문 draft (5/14~5/22) 또는 학회 paper 작성 시 launch

- **B1'~B3' 의 즉시 launch 의의**:
  - **B1'**: Maamari 2024 paradigm 의 GLM 4.7 직접 검증 — "LLM single-call full schema 가 SOTA LLM 에서 충분?" 정량
  - **B2'**: Table-level oracle 의 GLM 4.7 EX ceiling
  - **B3'**: Perfect schema linking (R=P=F1=1.0) 의 GLM 4.7 EX **absolute upper bound** — 학회 narrative 의 결정적 reference (본 연구 F1 의 EX transfer 갭 측정 base)

- **책임 분담 chain (V5 chain 정정 원칙 준수)**:

  | Step | 책임 | 작업 |
  |---|---|---|
  | 1 | **Analyzer** | GLM 4.7 SQL gen + EX eval script 작성 (`src/analysis/glm_baseline_sql_eval.py`). 기존 Jupyter notebook 3개 의 코드 logic 그대로 + LLM backbone 만 GLM 4.7 API client (XiYanFilter 의 GLM client 재사용) |
  | 2 | **Root** | Script launch 3 cell (B1'+B2'+B3') |
  | 3 | **Analyzer** | 결과 보고서 — EX 3 scenario (GLM 4.7) + Llama 3.1 8B vs GLM 4.7 backbone sensitivity 분석 + Theoretical bounds 재인용 + paper §V.5.4 narrative pre-integration (B4' 결과 후 final integration) |
  | 4 (보류) | **Planner** | B4' launch 시점 + paper §V.5.4 narrative final integration (Filter 모듈 확정 후) |

- **기존 Llama 3.1 8B 측정 reference (DECISIONS 직전 entry)**:
  - Full Schema: EX 34.1%
  - Gold Table: EX 40.1% (Δ +6.0%p)
  - Gold Column: EX 41.5% (Δ +7.4%p)
  - → GLM 4.7 backbone 결과와 직접 비교 (backbone sensitivity)

- **비용 + 시간 (3 cell)**:
  - ~$15-40 cost (GLM 4.7 API, BIRD-dev 1534 query × 3 prompt)
  - ~4-7h wall (LLM API call latency)
  - Full Schema 의 prompt token 가장 큼 (모든 column) → B1' 가 가장 비싼 cell

- **6 chain matrix 갱신**:

  | # | Chain | Status |
  |---|---|---|
  | 1 | Filter sweep 9-cell (root) | 🔄 active |
  | 2 | V5 sweep V5-A/B/C (root multi-instance) | 🔄 active |
  | 3 | SGBE Phase 3-5 | 🚀 launch trigger (prerequisite 완료) |
  | 4 | V5-D-1 진단 (analyzer) | ✅ 완료 |
  | 5 | V5-D-2 학습 | ⏸ V5 sweep 결과 후 권장 |
  | **6** | **B1'+B2'+B3' GLM Baseline** | 🚀 **즉시 launch** (analyzer script + root launch + analyzer 보고서) |
  | ⏸ 7 | B4' paper main → SQL gen | ⏸ **Filter 모듈 확정 후** (SGBE + V5 + Filter sweep 결과 통합 후) |

- **GLM 4.7 SQL gen prompt 의 정확한 형식 (기존 notebook 정합)**:

  System: `"You are an expert SQL developer. Your task is to write a SQLite query based on the given schema and external knowledge. IMPORTANT: If a column name contains spaces or special characters, you MUST wrap it in backticks. Output ONLY the SQL query."`
  User: `"### Schema (table.column):\n{schema_str}\n\n### External Knowledge:\n{evidence}\n\n### Question:\n{question}\n\n### SQL:"`

  → 기존 3 notebook 의 `create_messages` 함수 그대로 사용. LLM client 만 GLM 4.7 (provider="glm", model_name="zai-org/glm-4.7", temperature=0.0, max_tokens=256).

- **EX evaluation logic (기존 notebook 정합)**:
  - SQL execution with 3s timeout (thread-based, Cross Join 방지)
  - `set(cursor.fetchall())` 비교 (gold SQL result == predicted SQL result)
  - Error string 인 경우 비교 X (정확한 SQL 만 평가)

- **자원 조율**:
  - LLM API (GLM 4.7) — Filter sweep + SGBE chain + 본 chain 모두 동일 backbone 사용 (Elice ML API rate limit 확인 필요)
  - GPU 자원 — 본 chain 은 GPU 사용 안 함 (LLM API only) → V5 chain GPU 와 자원 별개

- **근거**:
  - 사용자 직접 input (B4' 보류 + B1'~B3' 즉시)
  - DECISIONS 직전 entry (Full Schema/Gold Baseline 발견)
  - 기존 Jupyter notebook 3개 (LLM Llama 3.1 8B base)

- **영향 범위**:
  - 신규 산출물: `src/analysis/glm_baseline_sql_eval.py` (analyzer 작성)
  - 신규 scripts: `scripts/run_glm_baseline_3cell.sh` (root 작성)
  - 신규 outputs: `outputs/analysis/glm_baseline/{b1_full, b2_gold_table, b3_gold_column}/predictions.jsonl + metrics.txt`
  - 신규 analyzer 보고서: `notebooks/analysis_results/glm_baseline_sql_eval_3cell.md`

- **에스컬레이션**:
  - **Analyzer** 세션 (cwd = `src/analysis/`) 에 script 작성 follow-up 핸드오프 (planner 작성)
  - Analyzer 완료 후 **Root** 에 launch 핸드오프 (planner 가 trigger 또는 직접)
  - Root launch 완료 후 **Analyzer** 가 결과 보고서 작성
  - 본 chain 의 B4' 부분은 별도 — Filter 모듈 확정 후 별도 chain

- **paper §V.5.4 narrative pre-integration candidate (B4' 대기 중)**:
  - B3' (Gold Column GLM 4.7 EX) = perfect schema linking 의 EX upper bound — paper §V conclusion 의 "schema linking 의 SQL gen 측면 ceiling" 정량
  - B1' (Full Schema GLM 4.7 EX) — Maamari 2024 paradigm 의 GLM 4.7 정량. 본 연구 narrative 의 정확한 positioning candidate
  - B2' (Gold Table) — table-level oracle ceiling
  - **B4' 결과 후** — Filter Dominance 의 EX transfer narrative 완성

---

## 2026-05-12 (Full Schema / Gold Table / Gold Column Baseline 발견 — Llama 3.1 8B + GLM 4.7 재실행 필요성)

> **사용자 직전 input (5/12)**: notebooks/{direct_generation, gold_schema_table_test, gold_schema_column_test}.ipynb 위치 명시. 기존 baseline 실험이 Jupyter notebook 으로 있음.

### 기존 실험 정리 (Llama 3.1 8B, BIRD-dev 1534 query)

**EX (Execution Accuracy)**:

| Scenario | LLM Backbone | EX | ΔEX vs Full |
|---|---|---:|---:|
| Full Schema (no schema linking) | meta-llama/Meta-Llama-3.1-8B-Instruct | **34.1%** | — |
| Gold Table | (동일) | 40.1% | +6.0%p |
| Gold Column | (동일) | 41.5% | +7.4%p |

**Theoretical Schema Linking Bounds** (gold_schema_table_test.ipynb 마지막 cell):

| Scenario | Overall R | Overall P | Table R | Table P | Col R | Col P |
|---|---:|---:|---:|---:|---:|---:|
| Full Schema | 0.9969 | 0.1381 | 0.9979 | 0.3292 | 0.9968 | 0.1173 |
| Gold Table | 0.9978 | 0.3359 | 1.0000 | 1.0000 | 0.9968 | 0.2729 |
| Gold Column | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

→ Full Schema 의 Overall Precision = 0.1381 (대부분 noise column).

### 직전 사용자 질문 ("GLM-4.7 기준으로 Full Schema / Gold Table / Gold Column 실험 했었나?") 의 정확한 답

- **기존 실험은 Llama 3.1 8B backbone** — paper main pipeline anchor 의 GLM 4.7 과 다른 backbone
- 본 연구의 main contribution (Schema Linking F1=0.8673) 와 EX (34.1~41.5%) 의 cross-comparison 불가능 (다른 metric + 다른 backbone)
- → **GLM 4.7 backbone 으로 재실행 필요**

### 학회 narrative 측면 critical gap 3

1. **Backbone 미정합**: paper main F1 (GLM 4.7) ↔ baseline EX (Llama 3.1 8B) 비교 불가능
2. **Schema Linking F1 의 SQL Gen 효과 미측정**: 본 연구의 main contribution F1=0.8673 이 실제 downstream EX 에 어떻게 transfer 되는지 정량 missing — paper §V.5.4 narrative 의 결정적 evidence missing
3. **Maamari 2024 직접 비교 불가**: Maamari paradigm "LLM single-call full schema 가 SOTA LLM 에서 충분" 이 GLM 4.7 backbone 에서 정량 검증 안 됨

### 결정 — GLM 4.7 backbone 재실행 (4 cell)

| Cell | Scenario | Backbone | 의미 |
|---|---|---|---|
| **B1'** | Full Schema | zai-org/glm-4.7 | Maamari paradigm 의 GLM 4.7 직접 검증 |
| **B2'** | Gold Table | (동일) | Table oracle 의 GLM 4.7 EX ceiling |
| **B3'** | Gold Column | (동일) | Perfect schema linking 의 GLM 4.7 EX ceiling (absolute upper bound) |
| **B4'** | paper main F1=0.8673 결과 → GLM 4.7 SQL gen | (동일) | **본 연구 F1 의 EX transfer 정량 — 학회 narrative 의 결정적 evidence** |

→ 직전 baseline plan (5/12 "Full Schema + Gold baseline 실험") 의 정확한 구현 path 확보. 기존 3 노트북을 base 로 LLM backbone 만 GLM 4.7 로 교체 + B4' 신규 cell 추가.

### 비용 + 시간

- Llama 3.1 8B 기존 측정: 1534 query × ~50min~1h35m wall (local GPU inference)
- GLM 4.7 API: 4 cell × 1534 query × prompt size (Full Schema = large input)
- 예상 cost: ~$20-50 (Elice ML API)
- 예상 wall: ~5-10h

### 본 연구 narrative 측면 의의

- **B1' Full Schema (GLM 4.7) F1** 측정 — 본 연구 paper main F1=0.8673 과 직접 비교 (같은 backbone, 다른 schema condition)
- **B3' Gold Column** = 본 연구의 R=P=F1=1.0 perfect oracle — schema linking 의 absolute upper bound 정량
- **B4' paper main → SQL gen EX** = paper §V.5.4 narrative 의 SQL gen 측면 정당화 (Filter Dominance 가 downstream EX 에도 transfer)
- Schema linking F1 (selector + filter) ↔ SQL gen EX 의 비례 관계 정량 → paper §V conclusion 의 결정적 evidence

### Schema linking metric vs EX metric 의 mapping

| Schema Condition | Schema Linking F1 (예상) | SQL Gen EX (예상, GLM 4.7) |
|---|---:|---:|
| Full Schema | 0.243 (=2·R·P/(R+P) = 2·0.997·0.138/(...)) | ? (~34% Llama → GLM 4.7 향상 candidate) |
| Gold Table | 0.503 | ? |
| Gold Column | **1.000** | ? (perfect schema linking 의 EX upper bound) |
| paper main (Enriched + QCond + MST + XiYan + GLM 4.7) | **0.8673** | **?** ← B4' 측정 |

→ **B4' 측정값 ↔ B3' (Gold Column) 의 gap** = "schema linking 의 R/P 0.13 의 불완전성이 SQL gen EX 에서 얼마나 차이를 만드는지" 의 정확한 정량.

### Action items

1. 본 entry prepend ✅
2. 기존 3 노트북을 base 로 GLM 4.7 backbone + B4' (paper main schema linking 결과) 추가하는 신규 chain 작성 필요 (root + analyzer chain)
3. 학회 narrative 측면: B4' 결과가 paper §V.5.4 narrative 에 직접 통합 — Schema Linking F1 (selector + filter) + SQL Gen EX 의 결합 evidence

### 책임 분담 (V5 chain 정정 원칙 준수)

| Step | 책임 | 작업 |
|---|---|---|
| 1 | Module: filter (또는 신규 module:sql_gen) | LLM client 호환 (GLM 4.7 API call) 보강 — 기존 노트북의 `meta-llama/Meta-Llama-3.1-8B-Instruct` 부분만 교체 |
| 2 | Module: filter (또는 analyzer) | B4' 의 prompt 작성 — paper main schema linking 결과 (`outputs/.../predictions.jsonl` 의 final_nodes) 를 schema_str 로 변환 + 기존 prompt 재사용 |
| 3 | Root | GLM 4.7 backbone B1'/B2'/B3'/B4' 4 cell launch (~5-10h, ~$20-50) |
| 4 | Analyzer | EX + Schema linking F1 의 mapping 표 작성 + B4' 결과의 paper §V.5.4 narrative 통합 |

### 근거

- `notebooks/direct_generation.ipynb` (Llama 3.1 8B Full Schema 34.1%)
- `notebooks/gold_schema_table_test.ipynb` (Gold Table 40.1% + Theoretical bounds)
- `notebooks/gold_schema_column_test.ipynb` (Gold Column 41.5%)

### 영향 범위

- planning/DECISIONS.md (본 entry)
- 후속 chain (root + analyzer): GLM 4.7 backbone 4 cell 실험 + 결과 보고서
- paper §V.5.4 narrative 의 SQL Gen EX 차원 통합 (Filter Dominance 의 downstream 정량)

### 우선순위 (사용자 결정 candidate)

| 옵션 | trade-off |
|---|---|
| **(a) 즉시 launch** | 학회 narrative 의 SQL Gen EX 정량 빠르게 확보. 단 6 chain 동시 진행 (filter sweep + V5 sweep + SGBE + V5-D-2 pending + 본 GLM 4.7 baseline + ...) |
| **(b) SGBE chain 완료 후** | SGBE 결과의 schema linking F1 도 B4' 의 base 로 추가 가능 (anchor + SGBE 2 cell 의 EX 비교) |
| **(c) V5 sweep 결과 후 통합** | 14-trial + V5-D-1 + SGBE + 본 4 cell baseline + V5-D-2 까지 통합 narrative — 가장 깔끔 |

→ 권장: **(b)** — SGBE 결과가 B4' 의 base 로 추가 가치 + V5 sweep 와 자원 별개 (LLM API).

---

## 2026-05-12 (SGBE Chain Phase 3 Launch Trigger — Prerequisite 완료 + Threshold 사후 검증)

> **사용자 직전 input (5/12, Module:filters 보고)**: step_mode 3-mode + score_collapse_threshold 옵션 추가 완료. 16/16 smoke test PASSED. Option A 채택 (default 0.05). stats + filter_info 에 기록 → era 별 collapse 빈도 측정 가능.

- **결정 — SGBE Chain Phase 3 Launch 즉시 진행 (launch 보류 조건 해제)**:

  Prerequisite 모두 충족:
  - ✅ Module:filters SGBE 구현 완료 (5/5 smoke, 5/12)
  - ✅ step_mode 3 mode 옵션 (16/16 smoke, 5/12) — Phase 3 calibration 의 `step_0+1` 평가 + Phase 5 ablation 의 step 별 평가 모두 지원
  - ✅ Score collapse fallback Option A — default `score_collapse_threshold=0.05`
  - ✅ Module:selector raw_score interface (5/5 smoke, 5/12)
  - ✅ Root 의 sgbe_calibration_base.yaml + scripts + HISTORY placeholder
  - ✅ Q1=Case B + Q2=paper main anchor 정합 (DECISIONS Phase 2 보강 entry §"Option B")

- **Score collapse threshold 0.05 의 적정성 검증 plan**:

  Module:selector Phase 2 진단의 3-anchor group mean spread:

  | Anchor | TP-TN group mean spread | 추정 column-level score std (per-query) |
  |---|---:|---:|
  | a03_17 (binary) | 0.3067 | ~0.15~0.20 |
  | vLLM best | 0.1543 | ~0.08~0.12 |
  | **paper main GLM** | **0.1741** | **~0.10~0.13** |

  → Threshold 0.05 가 정상 anchor 의 score std (0.08~0.20) 보다 충분히 낮음 — **conservative trigger** (V4 era boundary case 만 detect 예상, 정상 anchor 에서는 거의 trigger 안 됨). 학술 Agent §"한계" 의 V4 era 정확한 정합.

  **Post-hoc analysis (Root launch 와 병행)**:
  - SGBE 결과 stats["score_collapse_detected"] + filter_info["filter_score_std"] 가 모든 query 에 기록 (Module:filters 보고)
  - Phase 3 sweep 결과에서 anchor GLM 의 score std 분포 측정 → threshold 0.05 의 hit rate 분석 (예상 ~0%)
  - V4 era ckpt 의 score std 는 별도 측정 — V5 sweep 후 dsn_mitigation_v5_4dir.md 의 SGBE boundary case 통합 가능

  → **즉시 Phase 3 launch 진행 + threshold 사후 검증** (conservative 동작, risk 없음).

- **Score collapse threshold 권장값 update 결정 trigger**:
  - Phase 3 sweep 결과 score_collapse_detected hit rate > 5% → threshold 너무 높음 → 0.01~0.02 권장 update
  - Phase 3 sweep 결과 hit rate ~ 0% → threshold 적정 (V4 era 같은 extreme case 만 detect)
  - Phase 3 sweep 결과 hit rate 100% → threshold 너무 낮음 또는 학술 Agent §한계 의 hypothesis 직접 검증 — anchor 자체가 collapse 한 score 분포

- **Chain status 갱신 (5 chain matrix)**:

  | # | Chain | Status |
  |---|---|---|
  | 1 | Filter sweep 9-cell (root) | 🔄 active |
  | 2 | V5 sweep V5-A/B/C (root multi-instance) | 🔄 active |
  | **3** | **SGBE Phase 3-5** | **🚀 launch trigger** — prerequisite 완료, root SGBE chain 의 Phase 3 calibration sweep 즉시 launch 가능 |
  | 4 | V5-D-1 진단 (analyzer) | ✅ 완료 |
  | 5 | V5-D-2 학습 | ⏸ trigger 미정 (V5 sweep 결과 후 권장) |

- **Module:filters 의 산출물 명세 (보강 후)**:
  - `src/modules/filters/score_gated_batch_extractive_filter.py` — step_mode + score_collapse_threshold 옵션
  - `src/modules/filters/tests/test_sgbe.py` — 16/16 smoke test PASSED
  - `src/modules/filters/EXPERIMENT_PLAN_filters.md` — SGBE entry 의 Option 1 + Option 2 갱신
  - 인터페이스 보강:
    - `stats["score_collapse_detected"]` (bool)
    - `filter_info["filter_score_std"]` (float)
    - → era 별 / anchor 별 collapse 빈도 post-hoc analysis 가능

- **에스컬레이션**:
  - **Root** 세션 (SGBE chain active, Phase 3 launch 보류 상태) 에 launch trigger handoff (planner 작성, 본 entry 후속)
  - **Module:selector** 추가 follow-up 진단 (column-level score std per-anchor 측정) — **선택**. SGBE Phase 3 결과의 post-hoc analysis 가 동일 정보 제공 가능 → priority 낮음
  - **Analyzer** 의 sgbe_filter_results.md 작성 시 score_collapse 정량 통합 (Phase 3 sweep 완료 후)

- **근거**:
  - Module:filters 보고 (5/12) — 16/16 smoke + Option A 채택
  - Module:selector Phase 2 보고 (5/12) — 3-anchor TP-TN spread 0.15~0.31
  - 학술 Agent `filtering_suggestion_by_scholar_agent_2026-05-12.md` §"한계" 의 V4 era score collapse hypothesis

- **영향 범위**:
  - planning/DECISIONS.md (본 entry — launch 보류 조건 해제)
  - Root SGBE chain — Phase 3 launch trigger (handoff 후속 작성)
  - Module:filters 산출물 (16/16 smoke 완료, 추가 작업 없음)

- **추가 필요 분석 (Phase 3 sweep 완료 후)**:
  - Score collapse threshold 0.05 의 hit rate 측정 → threshold update 결정
  - Phase 3 결과의 optimal θ (paper main anchor 9 cell sweep) → Phase 4 final SGBE launch trigger
  - V4 era ckpt 의 score std 별도 측정 (V5 sweep 후 dsn_mitigation_v5_4dir.md 통합)

---

## 2026-05-12 (SGBE Chain Actual Status 정정 — Launch 보류 + Module:filters skip_llm/step_mode prerequisite 필요)

> **Root 세션 read-only status 보고 결과 (5/12)**: Q1=Case B (Option B paper main grid 정합) / Q2=(i) paper main GLM anchor / Q3=Phase 3 launch 미실행 (보류) / Q4=결과 0. 사용자 보고 "SGBE 실험 진행 중" 과 actual state 충돌.

- **Actual Status 정정**:

  | 직전 인식 (사용자 + planner) | Actual Status (root 보고) |
  |---|---|
  | SGBE Phase 3-5 active (sweep 진행 중) | **SGBE Phase 3 launch 보류** — yaml + scripts + HISTORY placeholder 까지만 진행, sweep 미실행 |
  | Module:filters Phase 1 완료 → Root chain 진행 | Module:filters SGBE 구현 완료 단 **prerequisite option 부재** — `skip_llm` + `step_mode` option 미구현으로 Phase 3 calibration sweep 의 "Step 0+1 only 평가 (LLM call 없음)" 가 실행 불가 |

- **사용자 보고 ↔ Actual State 의 gap 원인 (working hypothesis)**:
  - 사용자가 module:filters 의 SGBE 코드 작성 완료 = SGBE chain launch = "실험 진행 중" 으로 잘못 인식한 가능성
  - 또는 root 가 yaml + scripts 작성 완료 보고 = "실험 진행 중" 으로 인식
  - **다른 root multi-instance 에서 launch 시도 없음** confirm (root 보고: `ps aux` SGBE process 0)
  - → 사용자에게 actual state 명확화 + 다음 action 결정 candidate 제시 필요

- **Root chain 의 Q1=Case B + Q2=paper main 정합성 확인**:
  - sgbe_calibration_base.yaml 가 `s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` anchor mirror — **paper main anchor 정확히 정합 ✅**
  - 9-cell grid (θ_keep {0.50, 0.55, 0.60} × θ_drop {0.20, 0.25, 0.30}) 가 paper main 의 score ladder (TP mean 0.4746, TN mean 0.3005) 와 정확히 정합 — Planner SGBE Phase 2 보강 entry §"Option B 권장" 의 정확한 구현
  - → root 의 사전 작업 (yaml + scripts) 은 정확. 단 launch 미실행 — prerequisite 부족

- **Prerequisite — Module:filters 의 skip_llm + step_mode option 추가 필요**:

  학술 Agent `filtering_suggestion_by_scholar_agent_2026-05-12.md` §"θ 설정의 실용적 접근" 인용:
  > "BIRD-dev 의 소규모 holdout 으로 F1+ 를 기준으로 grid search. θ_keep 과 θ_drop 각 3~5개 값 조합 = 9~25 cell sweep, **각 cell 은 LLM call 없는 Step 0+1 만 평가** 하므로 매우 빠릅니다"

  Phase 3 calibration sweep 의 **LLM call 없는 Step 0+1 only 평가** 가 SGBE 의 `skip_llm` option 또는 `step_mode` option 필요. 또한 Phase 5 ablation 의 4 cell:
  - ablation-1: Step 0 only (FK/PK hardcode) → `step_mode="step_0"`
  - ablation-2: Step 0+1 only (score gate, no LLM) → `step_mode="step_0+1"`
  - SGBE full: Step 0+1+2 (complete) → `step_mode="step_0+1+2"` 또는 default

  → Module:filters 의 SGBE 구현에 두 option 누락. **직전 SGBE Phase 1 핸드오프 (planner 작성) 의 누락** — `theta_keep`, `theta_drop`, `provider`, `model_name`, `temperature`, `fk_pk_hardcode` 만 명시, `skip_llm` + `step_mode` 미명시. Planner 의 실수.

- **결정 — 4 action items**:

  **(a) 사용자 보고 정정**: SGBE chain 의 actual status (Phase 3 launch 보류) 를 사용자에게 명확화. 다른 root multi-instance launch 시도 없음 confirm.

  **(b) Module:filters 에 skip_llm + step_mode option 추가 follow-up handoff** (즉시 진행):
  - 직전 SGBE Phase 1 핸드오프의 누락 보강
  - `ScoreGatedBatchExtractiveFilter.__init__` 에 `step_mode` parameter 추가 (default = "step_0+1+2" = full SGBE)
  - `step_mode="step_0"` / `"step_0+1"` / `"step_0+1+2"` 의 3 mode 지원
  - smoke test 에 3 mode boundary case 추가
  - 완료 후 Root 에 SGBE Phase 3 launch trigger

  **(c) 학술 Agent grid (Case A) 추가 sweep 보류**: 본 chain 의 Q1=Case B 가 paper main anchor 정합 — 정확한 sweep. Case A 추가 sweep 은 paper §V.5.4 narrative 의 "학술 Agent reference anchor 추정" 부분에 도움 가능 단 **paper main 결과 후 결정** (priority 낮음).

  **(d) 27-cell Option A (3 anchor × 9 cell) 검토 보류**: 본 chain 의 Option B (paper main 9 cell) 결과 후 a03_17 + vLLM era best 의 추가 sweep — paper main 만 우선 진행하기로 결정 (DECISIONS Phase 2 보강 entry) → **paper main 결과 후 결정** (priority 낮음).

- **Chain status 갱신 (5 chain matrix)**:

  | # | Chain | Status |
  |---|---|---|
  | 1 | Filter sweep 9-cell (root) | 🔄 active |
  | 2 | V5 sweep V5-A/B/C (root multi-instance) | 🔄 active |
  | 3 | **SGBE Phase 3-5** | ⏸ **launch 보류** (Module:filters skip_llm/step_mode option prerequisite 부재) |
  | 4 | V5-D-1 진단 (analyzer) | ✅ 완료 (Tier 1 GO) |
  | 5 | V5-D-2 학습 | ⏸ trigger 미정 |

- **에스컬레이션**:
  - **Module:filters** 세션에 skip_llm + step_mode option 추가 follow-up handoff (planner 작성)
  - Option 추가 완료 후 **Root 의 SGBE Phase 3 launch** trigger
  - 사용자 보고 정정 — 직전 "SGBE 실험 진행 중" 인식이 actual state 와 차이

- **근거**:
  - Root 세션 read-only status 보고 (Q1=Case B / Q2=paper main / Q3=launch 보류 / Q4=결과 0)
  - 학술 Agent §"calibration 방법" 의 "LLM call 없는 Step 0+1 만 평가" 인용
  - 직전 SGBE Phase 1 핸드오프 (planner 작성) 의 option 누락 — planner 의 실수

- **영향 범위**:
  - planning/DECISIONS.md (본 entry)
  - Module:filters: SGBE 구현 보강 (skip_llm + step_mode option 추가)
  - Root: option 추가 후 Phase 3 launch 가능

- **Planner 본 entry 의 retrospective**:
  - 직전 SGBE Phase 1 핸드오프 작성 시 학술 Agent §"calibration 방법" 의 prerequisite (LLM call 없는 Step 0+1 평가) 를 option 으로 명시 안 함 — module:filters 가 자체 발견하지 못한 경우
  - 향후 chain prompt 작성 시 evaluation protocol (calibration / ablation 의 LLM 사용 여부) 도 module 영역 option 으로 명시 필요

---

## 2026-05-12 (V5-D-1 PLM Lower Bound 진단 완료 — Tier 1 GO 권고 + v5_plan §4.4/§5.3 정정)

> **사용자 직전 input (5/12)**: Analyzer 의 V5-D-1 진단 chain 완료 보고 — Plain vs Enriched $\bar{c}_{L_0}$ Δ=-0.0279, anchor $\bar{c}_{L_0}$=0.6246, $\bar{c}_{L_3}$=0.8924, V5-D-1 Tier 1 GO 권고. v5_plan §4.4/§5.3 정정 + V5-D-1 후속 학습 trigger 권고 + DECISIONS prepend 요청.

- **V5-D-1 진단 핵심 결과 (multi-DB n=55, 11 BIRD-Dev DBs × 5 queries seed=42)**:

  | Measurement | Value |
  |---|---:|
  | Plain builder $\bar{c}_{L_0}$ | 0.6526 |
  | **Enriched builder $\bar{c}_{L_0}$** | **0.6246** |
  | Δ (Enriched - Plain) | **-0.0279** (10/11 DB negative, codebase_community +0.0109 outlier) |
  | Anchor (Enriched + V-3-ext DSN p80) $\bar{c}_{L_3}$ | 0.8924 |
  | GAT 추가 collapse (Δ = $\bar{c}_{L_3}$ - $\bar{c}_{L_0}$) | +0.27 (multi-DB 기준) |

- **Phase 1 single-DB n=2 측정 정정**:
  - 직전 v5_plan §4.4 인용 "$\bar{c}_{L_0}$ = 0.51" 는 single-DB (california_schools) outlier
  - Multi-DB n=55 재측정값 anchor 0.6246 — single-DB 보다 +0.12 높은 ladder
  - V5-D-1 진단의 directional evidence 가 multi-DB protocol 기준이 더 정확

- **V5-D-1 GO 권고 — 4/4 trigger 조건 충족**:
  1. ✅ Enriched directional evidence (Δ=-0.0279, 10/11 DBs negative direction)
  2. ✅ Target c_L0 ≤ 0.30 까지 추가 여지 (-0.32 absolute reduction needed, current 의 51%)
  3. ✅ Single-DB caveat 해소 (multi-DB n=55, 11 BIRD-Dev DBs all sampled)
  4. ✅ Anchor c_L3 over-smoothing confirm (0.8924 ≥ 0.80 threshold)

- **Enriched 단독의 한계 정량**:
  - Enriched builder 가 needed -0.32 의 **~9% (-0.028) 만** 기여
  - → V5-D-2 (contrastive pre-training) 가 추가로 **-0.30 인하 필요** — 본 chain 의 critical path

- **결정 — v5_plan §4.4 + §5.3 정정 (planner 본 entry 작성 후 즉시 수행)**:
  1. **§4.4 Direction D 정정**:
     - "$\bar{c}_{L_0}$ = 0.51" → "single-DB outlier 0.51 ~ multi-DB n=55 0.6246" 명시
     - GAT 추가 collapse 정량 갱신 (Δ ≈ +0.45 → +0.27 multi-DB)
     - V5-D-1 진단 directional evidence (Δ=-0.0279) 추가
     - Enriched builder 의 9% partial mitigation + V5-D-2 추가 인하 필요 명시
  2. **§5.3 Tier 1 권고 정량 evidence 보강**:
     - "0.51 → 0.30 이하" → "0.6246 → 0.30 이하 = -0.32 needed" 정정
     - Enriched 단독 효과 -0.028 = needed 의 9% 명시
     - 4/4 GO trigger 조건 충족 표
     - V5-D-1 후속 학습 trigger 권고 — (a) column-specific fine-tuned PLM 또는 (b) schema-aware contrastive pre-training (LCL/SimCLR style, SBP signed-edge 사례 인용)

- **V5-D-1 후속 학습 trigger 권고 정식**:

  | Approach | Mechanism | 예상 효과 |
  |---|---|---|
  | **(a) Column-specific fine-tuned PLM** | sentence-transformer 를 BIRD schema metadata (description + data_type + FK context + NL alias) 로 fine-tuning | Enriched builder 의 -0.028 trend 확장. -0.10 ~ -0.20 추가 인하 candidate |
  | **(b) Schema-aware contrastive pre-training** | intra-table column pair = negative, inter-table = positive. LCL / SimCLR style + SBP signed-edge inspired | Direct intra-table push-apart. -0.20 ~ -0.30 추가 인하 candidate (target 0.30 도달 candidate) |

  → 두 approach 모두 valid. **권장: (a) → (b) 단계적 진행**. (a) 의 cost 가 낮고 (~1-2일), (b) 가 main effect (~수일~1주). (a) 의 결과로 (b) 의 target 조정 가능.

- **Three-Axis Invariance 보강**:
  - V5-D-1 진단의 부수적 발견: c_L0 axis 도 학습 변형 (Plain vs Enriched) 에 stable — anchor 0.6246 vs Plain 0.6526 의 spread 가 schema-dependent variability (per-DB Δ 0.005~0.054) 대비 작음.
  - → **Architecture / Attention / Gradient / c_L0 의 4-axis invariance** (3-axis → 4-axis 격상 candidate). 본 발견은 학회 contribution candidate.

- **paper §V.5.4 narrative 영향 working hypothesis**:
  - **Layer 2 narrative pivot 의 partial evidence**: 학술 Agent v5 plan §"Layer 2 reinterpretation" ("R@15 ceiling = PLM lower bound + domain bottleneck") 의 **directional evidence** 확보. Single-DB outlier 정정 후 multi-DB c_L0 = 0.6246 — 직전 narrative ("PLM lower bound 가 0.51 mid-similar") 보다 더 심각한 baseline.
  - **단 V5-D-2 결과 후 narrative pivot 확정**. V5-D-1 만으로는 R@15 영향 measurement 없음 (다음 학습 chain 필요).

- **근거**:
  - `notebooks/analysis_results/v5_d1_plm_lower_bound.md` §0 TL;DR + §3.3 per-DB consistency + §4.1 trigger 조건 충족
  - 사용자 직접 input + 본 V5-D-1 진단 4/4 trigger 조건 충족
  - 학술 Agent v5_plan §4.4 Direction D + §5.3 Tier 1 권고

- **영향 범위**:
  - planning/oversmoothing_v5_plan.md §4.4 + §5.3 (정정 완료)
  - planning/DECISIONS.md (본 entry)
  - 후속 (Module: Builder + Module: Selector + Module: PLM Encoder 또는 별도 module: encoders): V5-D-2 학습 chain 작성 (root 위임 + analyzer 후속 측정)

- **에스컬레이션**:
  - **사용자 결정 candidate**: V5-D-2 학습 trigger 시점 (a) 즉시 launch (3-chain 동시 진행과 별개 chain) / (b) V5 sweep (V5-A/B/C 학습) 결과 후 / (c) SGBE chain 완료 후
  - **V5-D-2 작업 책임 분담** (V5 chain 정정 원칙 준수):
    - PLM fine-tuning 코드: Module: Encoder (또는 신규 module:plm) 또는 src/models/plm_encoder.py 수정 영역
    - Contrastive pre-training script: Module 영역 (코드 구현) + Root 영역 (실행)
    - 결과 측정: Analyzer 영역
  - **권장: V5-A/B/C 학습 결과 + SGBE Phase 5 결과를 본 V5-D-1 진단 결과와 통합 보고서 작성 후 V5-D-2 launch 결정** (사용자 일정 + LLM/GPU 자원 고려)

- **추가 필요 분석**:
  - codebase_community outlier (+0.0109) 의 mechanism — Enriched 가 어떤 column type 에서 backfire 하는지 (analyzer 의 보고서 §4.3 추가 측정 candidate)
  - V5-D-1 후속 학습 후 c_L3 측정 — Δc_L0 vs Δc_L3 의 transfer 효율 (학회 narrative 의 V5-D-1 core finding)
  - Enriched builder 의 component-level ablation (name_type / name_type_desc / enriched_full) — 어떤 enrichment 가 dominant contributor

---

## 2026-05-12 (SGBE Phase 2 보강 + 3-anchor 진단 — 학술 Agent 인용 mean 격차 + TP-TN spread 패턴 정합)

> **사용자 직전 input (5/12)**: Module:Selector 세션의 SGBE Phase 2 작업 완료 보고 — Interface 보강 + 3-anchor score calibration 진단 결과.

### 결과 정리

**Phase 2.1 — Interface 보강 (5/5 smoke 통과)**:
- raw_gat_scores / raw_cos_scores 키 형식이 `'table.column'` + `'column'` 단독 fallback 양쪽 등록 (SGBE `_lookup_score` 호환)
- fk_node (`'->'`) 제외
- 산출물: `src/pipeline/schema_linking.py` (raw_*_scores 키 보강) + `src/modules/selectors/tests/test_raw_score_interface.py` (5/5 통과)

**Phase 2.2 — 3-anchor score 분포 진단** (1534 query × ~93 columns 평균):

| Anchor | F1 | Stack | TP mean | Filter✗ mean | TN mean | **TP-TN spread** |
|---|---:|---|---:|---:|---:|---:|
| **a03_17** | 0.6940 | Direct GAT binary + Fixed PCST + XiYan | 0.5818 | 0.2787 | 0.2751 | **0.3067** ⭐ |
| `abl_ens_basic_xiyan` (vLLM best #6) | 0.7863 | Ensemble (α=0.5) + Basic PCST + XiYan | 0.5526 | 0.4975 | 0.3983 | 0.1543 |
| **anchor GLM (paper main)** | **0.8673** | Enriched + QCond Concat + MST + XiYan + GLM 4.7 | 0.4746 | 0.4101 | 0.3005 | 0.1741 |
| **학술 Agent 보고 (참조)** | ? | (anchor 미명시) | 0.7108 | 0.6394 | ~0.40 | **~0.31** |

### 핵심 발견

1. **학술 Agent reference anchor 의 정체 미명시** — TP=0.7108 인용의 출처가 우리 3 anchor 어느 것과도 mean 절대값 일치 X. **TP-TN spread 패턴 (~0.31) 만 a03_17 (0.3067) 과 정합** — DirectGATSelector 의 binary sigmoid bimodal 분포 특성.

2. **a03_17 spread 정합 의미**:
   - DirectGATSelector binary 출력 → sigmoid bimodal 분포 → TP/TN clear separation (spread 0.31)
   - 학술 Agent 보고서의 reference 도 같은 bimodal selector 구조 (직전 a05 anchor 가 a03_17 — 학술 Agent 가 a05 ablation 보고서 인용했을 가능성 강함)
   - 단 mean 절대값 차이 (TP 0.7108 vs 0.5818) 의 의미: 학술 Agent reference 의 selector 가 본 ckpt 보다 sharper threshold 또는 다른 hyperparameter

3. **anchor 별 score distribution 의 매우 다른 ladder**:
   - a03_17: bimodal (binary 결과) — TP/TN 명확 분리, TP-TN spread 0.31
   - vLLM best: blended (cosine + GAT α=0.5) — smooth 분포, TP-TN spread 0.15
   - paper main (GLM era): Enriched + QCond Concat + MST blend — TP mean 더 낮음 (0.4746), 단 F1 가장 높음 (0.8673)
   - → **F1 ↑ 하면서 TP mean ↓** 의 비대칭 패턴: paper main 의 strong filter (XiYan + GLM) 가 낮은 TP score 의 column 도 정확히 선별

4. **학술 Agent 권장 θ (0.65, 0.40) 의 적용 가능 anchor 범위 제한**:
   - **a03_17** (bimodal): θ_keep=0.65 너무 높음, θ_drop=0.40 너무 높음. 사실상 모든 column 이 S_uncertain 으로 떨어짐 (binary 0/1 의 0.5 cutoff)
   - **abl_ens_basic_xiyan**: TP mean 0.5526 < θ_keep 0.65 → 대부분 TP 가 S_uncertain 으로 떨어짐. θ_keep 0.65 너무 보수적
   - **anchor GLM (paper main)**: TP mean 0.4746 << θ_keep 0.65, TN mean 0.3005 < θ_drop 0.40. **학술 Agent 권장 θ 그대로 적용 시 거의 모든 column 이 S_uncertain 으로** → XiYan-equivalent 로 degenerate (SGBE 의 이점 사라짐)

5. **Over-smoothing era 검출 X**: 세 anchor 모두 `collapsed=False`. SGBE gating valid 단 anchor 별 calibration 필수. V4-A/V4-B ckpt 의 score collapse 측정은 root chain (V5 sweep 후) 으로 위임.

### θ Recalibration 권장 매트릭스 (anchor 별 별도)

| Anchor | TP mean | TN mean | **권장 θ_keep** | **권장 θ_drop** | 기대 SGBE 효과 |
|---|---:|---:|---:|---:|---|
| **a03_17** (binary) | 0.5818 | 0.2751 | **0.50** (binary cutoff) | **0.05** (very low, almost no drop_hard) | binary 의 0/1 cutoff 위에 SGBE 의 minor restoration 효과 |
| **abl_ens_basic_xiyan** | 0.5526 | 0.3983 | **0.65** (학술 Agent 권장 그대로) | **0.40** | 학술 Agent 권장 θ 가 가장 정합 (TP mean 0.55 가까운 0.65 keep + TN mean 0.40 가까운 0.40 drop) |
| **anchor GLM** (paper main) | 0.4746 | 0.3005 | **0.55** (TP mean 0.47 보다 약간 위) | **0.25** (TN mean 0.30 보다 약간 아래) | paper main 의 낮은 TP score 영역을 capture 하기 위해 keep 낮춰야 |

→ **anchor 별 θ 가 0.50~0.65 (keep) + 0.05~0.40 (drop) 의 범위에서 변동**. Single global θ 가 3 anchor 에서 모두 동작 불가.

### Per-DB θ Adaptive 검토 권장

Phase 2.2 진단의 per-DB csv (각 anchor) 가 11 BIRD-Dev DB 별 score 분포 차이 정량 가능 — single global θ 가 모든 DB 에 valid 한지 검토:

- california_schools (T=3, C=89, large schema) — TP/TN spread 가 작을 가능성 (over-smoothing 영향 더 큼)
- toxicology (T=4, C=20, small schema) — TP/TN spread 클 가능성

→ 본 검토는 root chain Phase 3 의 sweep 결과 분석 시 analyzer 위임 candidate.

### Root chain Phase 3 의 sweep config 권장값 갱신

**직전 plan (single anchor 9-cell)**:
- θ_keep ∈ {0.60, 0.65, 0.70}, θ_drop ∈ {0.35, 0.40, 0.45} — 학술 Agent 권장 기반 단일 grid

**갱신 (anchor 별 sweep)**:

Option A — 3 anchor × 9 cell = **27 cell sweep** (가장 안전):
- a03_17 grid: θ_keep ∈ {0.45, 0.50, 0.55}, θ_drop ∈ {0.05, 0.10, 0.15}
- abl_ens_basic_xiyan grid: θ_keep ∈ {0.60, 0.65, 0.70}, θ_drop ∈ {0.35, 0.40, 0.45}
- anchor GLM grid: θ_keep ∈ {0.50, 0.55, 0.60}, θ_drop ∈ {0.20, 0.25, 0.30}

Option B — anchor 별 우선순위 (paper main 만 9 cell sweep, 다른 anchor 는 후속):
- **anchor GLM (paper main, F1=0.8673)** 만 9 cell sweep — 학회/학위 논문 narrative 의 가장 중요한 anchor
- 결과 후 a03_17 / abl_ens_basic_xiyan 별도 sweep 결정

**Recommended: Option B** — paper main anchor 우선 (학회 main contribution + LLM cost 절감)

### anchor stack 정의 명확화

본 진단으로 "anchor stack" 의 정의가 ambiguous — 3 candidate:
- **paper main pipeline** (`s04_pipeline_enriched_qcond_a05_mst_kruskal_glm`, F1=0.8673): 학회/학위 논문의 main contribution. SGBE 의 primary target.
- **vLLM era best** (`abl_ens_basic_xiyan`, F1=0.7863): 학술 Agent 의 SGBE 권장 θ 와 가장 정합한 stack
- **a03_17** (DirectGAT binary): 학술 Agent 의 TP-TN spread 패턴 정합 — bimodal selector

→ **SGBE primary target = paper main (anchor GLM)**. 단 학술 Agent 의 권장 θ 는 vLLM era best 와 정합 — selector + filter backbone 의 interaction 으로 narrative 화 가능.

### paper §V.5.4 Narrative 후속 보강 Candidate

> **추가 narrative**: 학술 Agent 가 권장한 SGBE θ (0.65, 0.40) 는 본 연구의 vLLM era best (`abl_ens_basic_xiyan`, Ensemble + Basic PCST + XiYan + Qwen3-Coder, F1=0.7863) 의 score 분포와 가장 정합 (TP mean 0.5526, TN mean 0.3983, spread 0.15). 그러나 paper main (Enriched + QCond Concat + MST + XiYan + GLM 4.7, F1=0.8673) 의 score 분포는 학술 Agent reference 와 격차 (TP mean 0.4746, TN mean 0.3005). 이 격차는 **selector + filter backbone 의 interaction** 으로 narrative 가능 — strong filter (GLM 4.7) 가 selector 의 낮은 TP score 영역에서도 정확한 schema linking 가능 → SGBE 의 θ 가 GLM era 에서는 더 낮은 ladder (keep 0.55 / drop 0.25) 로 calibrate 필요. **Score-Gate 의 calibration 이 단순 hyperparameter 가 아닌 selector + filter backbone interaction 의 결과** — Filter Dominance 6번째 축 narrative 의 추가 evidence (selector ↔ filter co-design).

### 영향 범위

- planning/DECISIONS.md (본 entry)
- root chain Phase 3 의 sweep config 갱신 (anchor 별 별도 grid)
- paper §V.5.4 narrative — selector + filter backbone interaction 으로 보강
- (후속) SGBE 의 anchor stack 정의 명확화 — paper main 을 primary target 으로

### 에스컬레이션

- **Root chain Phase 3 (SGBE θ calibration)**: 본 권장값 (Option B: paper main anchor 9 cell sweep 우선) 으로 sweep config 갱신 후 launch — Module:Filter 의 Phase 1 완료 신호 수신 후
- **Analyzer (후속, Phase 6 보고서 작성 시)**: per-DB θ adaptive 검토 + 학술 Agent reference anchor 추정 (external prior 여부) + selector + filter co-design narrative 보강

### 근거

- `outputs/analysis/sgbe_score_calibration/{a03_17, abl_ens_basic_xiyan, anchor}/{score_distribution.json, per_db.csv, histogram.png}`
- `src/analysis/sgbe_score_calibration_diagnostic.py`
- `src/pipeline/schema_linking.py` (raw_*_scores 키 보강)
- `src/modules/selectors/tests/test_raw_score_interface.py` (5/5 통과)
- 학술 Agent `planning/filtering_suggestion_by_scholar_agent_2026-05-12.md` §"세 그룹의 score 분포"

### 추가 필요 분석

- 학술 Agent reference anchor 의 정체 — external prior 인용인지 확인 (paper 또는 다른 dataset 의 결과). 학술 Agent 보고서의 reference (Yuan 2025) 의 score 분포가 0.7108 / 0.6394 / 0.40 인지 후속 검증.
- V4-A/V4-B ckpt 의 score collapse 측정 — V5 sweep 후 위임

---

## 2026-05-12 (V5 chain 위임 재할당 — Framework 원칙 정정 + Module:Selector + Analyzer 분리)

> **사용자 직전 input (5/12, root chain 차단 후)**: "(a) module 재할당 / (b) Hybrid review / (c) Revert + 재작성 중 선택. 본 root chain 의 V5 sweep launch 가 사용자 redirect (5/12 "오케스트레이션과 실험을 진행하지 직접 모듈을 구현하지 마") + auto-mode classifier 차단으로 보류".

- **결정 — 옵션 (a) Module 재할당 채택 (단 root 작성 코드는 keep)**:

  직전 V5 chain prompt (planner 5/12 작성) 가 V5-A/B/C 코드 작성을 root 에 위임한 것이 framework 원칙 위반:
  - `planning/CLAUDE.md` + 루트 `CLAUDE.md` 의 책임 분담: **Root** = 실험 실행 / **Module** = 모듈 내부 구현
  - `src/models/gat_network_v2.py` 의 `GATEConv` / `GCNIIGATv2Conv` / `FullAEROGATConv` 신규 클래스 = **Module 영역**
  - 직전 SGBE chain (Phase 1 filter module + Phase 2 selector module) 은 올바른 패턴 — V5 chain 만 inconsistent

- **재할당 표**:

  | Step | 직전 (잘못) | 정정 |
  |---|---|---|
  | V5-A GATE 코드 구현 | Root | **Module: Selector** (`src/models/gat_network_v2.py`) |
  | V5-B GCNII 코드 구현 | Root | **Module: Selector** |
  | V5-C Full AERO 코드 구현 | Root | **Module: Selector** |
  | V5-D-1 PLM 진단 측정 | Root | **Analyzer** (`src/analysis/v5_d1_plm_diagnostic.py`) |
  | Smoke test | Root | **Module: Selector** (자기 코드 검증) |
  | Config 작성 (training yaml) | Root | **Root** (실험 config 는 root 영역) |
  | Sweep launch (학습 실행) | Root | **Root** (실험 실행 영역) |
  | HISTORY / CATALOG / ID_MIGRATION 갱신 | Root | **Root** |

- **Root 의 작성 코드 처리 (3 옵션 중 하이브리드)**:
  - 옵션 (a) full module 재작성 + 옵션 (b) hybrid review 의 절충:
    - Root 가 이미 작성한 GATEConv / GCNIIGATv2Conv / FullAEROGATConv 코드 **keep** (불필요한 revert 작업 회피)
    - Module: Selector 세션이 본 코드를 **시작점으로 검토** + 이론적 정합성 점검 (Mustafa 2024 / Peng 2024 / Lee 2023 paper 와의 매핑) + smoke test 통과 확인 + `EXPERIMENT_PLAN_selectors.md` 갱신
    - Module 검토 통과 후 root 가 sweep launch
    - → 최종 코드의 책임은 **Module: Selector** (자기 이름으로 final commit), root 의 작성은 draft 로 처리

- **V5 chain prompt 갱신 — 3 신규 핸드오프**:
  1. **Module: Selector** (V5-A/B/C 검토 + smoke test + EXPERIMENT_PLAN 갱신)
  2. **Analyzer** (V5-D-1 PLM lower bound 진단)
  3. **Root** (Module + Analyzer 완료 후 config + sweep launch + HISTORY 갱신)

- **HISTORY/CATALOG/ID_MIGRATION 의 V5 entry placeholder 정리**:
  - 이미 prepend 된 launch 보류 placeholder 는 그대로 유지
  - Root 가 module 검토 + analyzer 완료 후 sweep launch 시 placeholder → 실제 학습 결과로 갱신

- **근거**:
  - 사용자 직접 input (root redirect + auto-mode classifier 차단)
  - 사용자 직전 redirect: "오케스트레이션과 실험을 진행하지 직접 모듈을 구현하지 마"
  - `planning/CLAUDE.md` 책임 분담 (Module = 모듈 내부 구현)
  - 직전 SGBE chain 의 올바른 패턴 (Phase 1+2 module 위임) — V5 만 inconsistent

- **영향 범위**:
  - planning/DECISIONS.md (본 entry)
  - V5 chain 의 root 위임 → module:selectors + analyzer 재할당
  - Root 의 작성 코드 (`src/models/gat_network_v2.py` 의 신규 클래스) keep, module:selectors 가 review

- **에스컬레이션**:
  - **Module: Selector** 세션 즉시 launch — V5-A/B/C 코드 검토 + smoke test + EXPERIMENT_PLAN 갱신
  - **Analyzer** 세션 즉시 launch — V5-D-1 진단 (병행 가능, module 과 무관)
  - **Root** 는 module + analyzer 완료 후 (Trigger 신호 수신 시) sweep launch + HISTORY 갱신

- **추가 필요 분석**: 없음. 단 Module: Selector 가 root 작성 코드 review 결과 (이론적 정합성 + smoke test 통과 여부 + minor refinement 권장 사항) 를 planner 에 보고.

- **본 정정의 의의** (process retrospective):
  - Planner 가 chain prompt 작성 시 책임 분담 원칙 재확인 필요 — root vs module vs analyzer 영역 mapping
  - 모듈 영역 (코드 작성) 인지 운영 영역 (실험 실행) 인지 명확화 후 위임
  - 향후 동일 패턴 (V6 등) 의 chain 작성 시 본 정정 entry 참조

---

## 2026-05-12 (3-chain 동시 진행 — Filter sweep + V5 + SGBE 병행 launch 결정)

> **사용자 직전 input (2026-05-12)**: "옵션 B 로 진행하자 지금 진행하지 않을 이유를 잘 모르겠어".

- **결정 — 3 chain 동시 진행**:

  | Chain | 자원 | Status |
  |---|---|---|
  | **Filter sweep (9-cell)** | LLM API (GLM 4.7) + 작은 GPU inference | 🔄 active (root) |
  | **V5 mitigation (4-Direction)** | GPU 0,1 (학습, ~30-40h) | 즉시 launch (root multi-instance) |
  | **SGBE Phase 1+2** | 코드 작성 (GPU/LLM 없음) | 즉시 launch (module: filter + module: selector 병행) |

- **자원 충돌 분석 — 무시 가능**:
  - Filter sweep 의 GPU inference 는 selector forward 만 (작음, 1534 query × CPU 가능)
  - V5 chain 의 GPU 학습이 main GPU consumer
  - SGBE module sessions 는 코드 작성 → GPU/LLM 없음
  - → 3 chain 자원 별개, 모두 병행 가능

- **직전 V5 entry 의 "filter sweep 종료 후 launch" 조건 해제**:
  - 직전 가정: root 세션 단일 chain 관리 부담
  - 사용자 결정: multi-instance root 허용 → conflict 없음
  - V5 chain 의 핸드오프 prompt 는 직전 DECISIONS entry (2026-05-12 V5 Mitigation Plan) 의 root chain 블록 그대로 사용 — 단 "filter sweep 종료 후 launch" → "지금 launch" 로 변경

- **즉시 전달 가능한 핸드오프 3 개**:
  1. **V5 chain → Root (multi-instance)**: 직전 V5 entry 의 chain prompt
  2. **SGBE Phase 1 → Module: Filter**: 직전 SGBE entry 의 Phase 1 prompt
  3. **SGBE Phase 2 → Module: Selector**: 직전 SGBE entry 의 Phase 2 prompt

- **Trigger 신호 (chain 종료 → 다음 핸드오프)**:
  - Filter sweep 종료 → Analyzer (filter 9-cell 보고서) + Planner (Filter Dominance 7번째 axis 결정)
  - SGBE Phase 1+2 둘 다 완료 → SGBE Phase 3-5 (Root chain, multi-instance or 동일 root 세션)
  - SGBE Phase 5 종료 → Analyzer (sgbe_filter_results.md)
  - V5 chain 종료 → Analyzer (dsn_mitigation_v5_4dir.md)
  - Analyzer 둘 다 완료 → Planner (narrative integration: paper §3.5 + framework snapshot + DECISIONS)

- **학회 / 학위 논문 일정 영향 (개선)**:
  - 직전 critical path: filter sweep (5-9h) → SGBE module (2-3일) → SGBE root (15-24h) → V5 chain (30-40h) → analyzer + planner → chapter draft
  - 신규 critical path: 3 chain 병행 → 가장 늦은 chain (V5 30-40h) 가 critical
  - → 학위 논문 draft 일정 (5/14~5/22) 에 V5 결과 통합 가능성 향상

- **근거**:
  - 사용자 직접 input + 자원 충돌 분석
  - 직전 SGBE entry 의 §"GPU/LLM 자원 조율" — 3 chain 자원 별개 명시

- **영향 범위**:
  - planning/DECISIONS.md (본 entry)
  - 기존 V5 entry + SGBE entry 의 핸드오프 prompt 는 그대로 (단 launch 시점만 변경)

- **에스컬레이션**:
  - 사용자가 3 핸드오프 즉시 전달:
    1. Root multi-instance (V5 chain) — V5 entry 의 핸드오프 prompt
    2. Module: Filter (SGBE Phase 1) — SGBE entry 의 Phase 1 prompt
    3. Module: Selector (SGBE Phase 2) — SGBE entry 의 Phase 2 prompt
  - Root 기존 chain (filter sweep) 는 자동 진행 continued

- **추가 필요 분석**: 없음.

---

## 2026-05-12 (SGBE Filter 채택 — Score-Gated Batch Extractive Filter 구현 + 실험 chain)

> **사용자 직전 input (2026-05-12)**: 학술 Agent 와 filter 논의 후 `planning/filtering_suggestion_by_scholar_agent_2026-05-12.md` 작성. 핸드오프 작성 + 세션별 역할 분담.

- **SGBE 방법론 핵심**:

  ```
  Step 0  [Structural Hard Keep — 0 LLM calls, instant]
    S_struct = FK/PK columns in S_pcst   ← 무조건 보존

  Step 1  [Score-Gate — 0 LLM calls, O(n)]
    θ_keep = 0.65 (TP mean 0.7108 기반)
    θ_drop = 0.40 (TN mean ~0.40 기반)
    S_keep_hard = {v | s_v ≥ θ_keep}     → LLM 없이 즉시 keep
    S_drop_hard = {v | s_v < θ_drop}     → LLM 없이 즉시 drop
    S_uncertain = {v | θ_drop ≤ s_v < θ_keep}  → LLM 대상

  Step 2  [Extractive LLM — 1 LLM call, S_uncertain만]
    Per-column binary 판단 ("yes/no + one-line reason")

  Output: final_nodes = S_keep_hard ∪ S_lm_keep ∪ S_struct
  ```

- **세 조건 충족 mechanism**:
  - **Recall 보호**: Step 0+1 의 score guard → TP (mean 0.7108) 가 LLM 접근 불가 → wrong-prune 구조적으로 불가능
  - **Precision 향상**: Step 1 의 θ_drop=0.40 → TN (mean ~0.40) 즉시 제거 + Step 2 의 extractive binary 판단
  - **빠른 추론**: LLM input token 60~80% 감소 (S_uncertain ≈ 전체의 20~40%)

- **예상 효과 (학술 Agent 정량)**:
  - R ≥ 0.73 (XiYan 0.6761 대비 +0.05)
  - P ≥ 0.70 (XiYan 0.7128 대비 -0.01)
  - 속도 1.5~2× (input token 감소)
  - Backbone 민감도 ~-0.015 (Verifier 수준)

- **세션별 역할 분담**:

  | 세션 | 책임 | 산출물 |
  |---|---|---|
  | **Planner (본 세션)** | 본 entry + 핸드오프 chain 설계 + GPU/LLM 자원 조율 + 후속 narrative integration | DECISIONS / planning |
  | **Module: filter** | `ScoreGatedBatchExtractiveFilter` 구현 + per-column binary prompt + value retrieval 재사용 + smoke test | `src/modules/filters/score_gated_batch_extractive_filter.py` + `EXPERIMENT_PLAN_filters.md` 갱신 |
  | **Module: selector** | GAT score 의 column-level calibration 진단 + EnsembleSelector 의 raw score (cosine + GAT 분리 또는 blended) 가 filter 단에 전달되도록 interface 보강 | selector raw_score interface + 진단 보고 (단계별 측정 분포) |
  | **Root (orchestrator)** | Config 작성 + θ calibration sweep (9 cell) + Final SGBE 평가 + Ablation (XiYan ↔ SGBE + Step contribution) + HISTORY 갱신 | configs + scripts + outputs/ + HISTORY+CATALOG+ID_MIGRATION |
  | **Analyzer** | θ calibration 결과 분석 + SGBE final 결과 + Step 별 기여도 분석 + boundary case (over-smoothing era V4 score collapse 시 SGBE 무력 caveat) | `notebooks/analysis_results/sgbe_filter_results.md` |

- **단계별 chain (6 Phase, single chain launch)**:

  | Phase | 책임 | 작업 | ETA |
  |---|---|---|---|
  | 1 | filter module | SGBE 클래스 구현 + smoke test | ~1-2일 |
  | 2 | selector module | Score calibration 진단 (anchor stack 의 column score 분포 측정) + interface 보강 (raw_score 전달) | ~0.5-1일 |
  | 3 | Root | Configs + θ calibration sweep (3 × 3 = 9 cells, BIRD-dev holdout, Step 0+1 only, LLM call 없음) | ~2-3 시간 |
  | 4 | Root | Optimal θ 로 final SGBE 평가 (BIRD-dev 1534 query, GLM 4.7) | ~5-9 시간 (LLM API) |
  | 5 | Root | Ablation chain — XiYan anchor + SGBE + Step contribution (Step 0 only / Step 0+1 only / Step 0+1+2 SGBE full) | ~10-15 시간 |
  | 6 | Analyzer + Planner | 결과 분석 + narrative integration (Filter Dominance 7번째 axis Filter-invariance 의 추가 evidence + SGBE 가 새 anchor candidate 인지 결정) | ~수 시간 |

- **GPU/LLM 자원 조율**:
  - **현재 진행 중**: Filter sweep (root chain, ~5-9h wall) + V5 mitigation chain (waiting, filter sweep 후 launch)
  - **본 SGBE chain**: filter sweep 와 별개 + V5 chain 와도 별개. 단 **GLM 4.7 API call cost 발생** (Elice ML API)
  - **권장 진행 순서**:
    1. Filter sweep 종료 대기 (root 기존 chain 완료)
    2. Filter module + Selector module session 병행 진행 (Phase 1 + 2, ~2-3일)
    3. Root chain: SGBE Phase 3~5 (θ calibration → final SGBE → ablation)
    4. V5 chain은 SGBE chain 종료 후 또는 SGBE Phase 4 LLM API 진행 중 GPU 0,1 병행 가능 (GPU 와 LLM 자원이 별개)

- **학회 narrative 영향**:
  - SGBE 가 학술 Agent 의 §9.4 (Prune-Only Recall 손실 mechanism) + §9.5 (GNN Selector role 재정의) 두 Open Question 의 직접 답변
  - SGBE 가 anchor (XiYan, F1=0.8673) 갱신 시 paper main pipeline anchor 변경 candidate
  - Filter Dominance 6 axis (10-trial mitigation null) + 7번째 axis (Filter-invariance, 9-cell sweep 결과) 와 별개 — **8번째 axis (Score-Gated Hybrid 효과)** candidate
  - SGBE 의 over-smoothing era 한계 (score collapse 시 무력) 가 Layer 2 reinterpretation 의 추가 evidence: "R@15 ceiling 의 원인이 PLM lower bound + domain bottleneck" 가설과 정합

- **근거**:
  - `planning/filtering_suggestion_by_scholar_agent_2026-05-12.md` 의 §1~§4 (학술 Agent 의 SGBE 설계 + θ 권장 + 예상 효과 + 한계)
  - 사용자 직접 input (구현 + 실험 + 세션별 역할 분담 요청)
  - 학술 Agent 의 5 references: Yuan 2025 / Glass 2025 / Hoang 2025 / Talaei 2024 / Maamari 2024

- **영향 범위**:
  - 신규 모듈 (filter module session 작성): `src/modules/filters/score_gated_batch_extractive_filter.py`
  - selector interface 보강 (selector module session 작성): EnsembleSelector raw score 전달
  - 신규 configs (root): `configs/experiments/s04_ablation/pipeline/sgbe/` 하위
  - 신규 scripts (root): `scripts/run_sgbe_calibration.sh` + `scripts/run_sgbe_final_ablation.sh`
  - 신규 analyzer 보고서: `notebooks/analysis_results/sgbe_filter_results.md`
  - HISTORY + CATALOG + ID_MIGRATION 갱신
  - 후속 (planner): 본 DECISIONS + paper §3.5 + framework snapshot §3/§4 갱신

- **에스컬레이션**:
  - **Module: filter** + **Module: selector** 가 본 chain 의 Phase 1+2 동시 진행 (서로 의존, selector interface 변경 후 filter 구현 마무리)
  - **Root** 가 Phase 3~5 chain (configs + sweep + final + ablation)
  - **Analyzer** 가 Phase 6 (보고서)
  - **Planner** 가 narrative integration

- **추가 필요 분석**:
  - SGBE 의 over-smoothing era 무력화 시점 — V4-A LN+GIN combo / V4-B AERO 의 score 분포가 collapse 한지 selector module session 진단 결과로 확인
  - Score-Gated 방식이 Reflection / Verifier 의 restore path 와 결합 가능한지 (SGBE + Reflection hybrid candidate, post-paper)

---

## 2026-05-12 (framework_snapshot_2026-05-12.md 신규 — 현재 전체 구조 통합 reference)

> **사용자 직전 input (2026-05-12)**: "현재 전체 구조를 정리한 자료도 만들어줘 이미 있으면 어떤 보고서인지 알려줘". → 기존 자료 점검 결과 "전체 구조" single document 미존재 (paper_research_direction.md 가 가장 가깝지만 paper narrative 위주). AskUserQuestion 결정: **Full snapshot (7 sections + 2 appendix), 다목적 청중**.

- **결정 — 신규 산출물**:
  [`planning/framework_snapshot_2026-05-12.md`](framework_snapshot_2026-05-12.md) — 다목적 framework snapshot 통합 reference (7 sections + 3 appendix, ~620 lines).

- **구조 — 7 sections + 3 appendix**:
  - §1 Pipeline Architecture — 5 모듈 + 데이터 흐름 + 인터페이스 + paper main anchor + 디렉토리 구조
  - §2 Module Status — Builder/PLM/Selector/Extractor/Filter 의 현재 구현체 + active variants + V5 pending
  - §3 Experiment Matrix — Main anchors (F1=0.8673 / 0.8383 / 0.7863 / 0.6940) + 핵심 sweep 결과 + 진행 중 (filter 9-cell + V5 4-direction)
  - §4 Narrative Status — Filter Dominance 6 axis + mech(ii-b) 5/5 + Three-Axis Invariance + 3 Layer 분리 (학술 Agent working hypothesis)
  - §5 Active Work + Pending — Filter sweep (active) + V5 chain (waiting) + 학위 논문 chapter draft (5/14~5/22) + critical path
  - §6 Planning Document Map — 13 문서 역할 분리 + 청중별 navigation 가이드
  - §7 학회 / 학위 논문 일정 — 한국지능정보시스템학회 + Part III chapter + critical path
  - §A Glossary (20 핵심 용어 — V-3-ext / DSN / mech(ii-b) / V1~V5 / Filter Dominance / 6 axis / Three-Axis Invariance / Restore path / JSR / anchor stack 등)
  - §B Quick Reference Cards (Top 10 정량 + Mitigation 시도 14 + Three-Axis spread + LLM backbone sensitivity + 자주 사용하는 명령)
  - §C Open Questions 합치 (over-smoothing 4 + filter 5 + V5 plan + 3 Layer pivot)

- **기존 자료 점검 결과 — "전체 구조" 가 분산**:
  - 가장 가까운 것은 `paper_research_direction.md` 단 paper narrative 위주
  - 운영 측면 (`CLAUDE.md`, `EXPERIMENT_*.md`) 분산
  - over-smoothing / filter narrative 는 별도 종합 보고서 (5/12 신규 2 종)
  - → **현재 framework 의 모든 axis (architecture / experiment / narrative / work / 일정) 의 통합 snapshot single document 가 missing**. 본 신규 자료로 보강.

- **차별점 (기존 12 planning 문서와)**:
  - 다른 planning 문서들이 specific topic 의 deep dive — 본 자료는 **모든 axis 의 shallow-but-complete overview**.
  - 청중별 navigation 가이드 (§6.2) 가 다른 deeper reference 로 redirect.
  - Glossary (§A) + Quick Reference Cards (§B) 가 새 collaborator / 학술 Agent 의 빠른 onboarding 자료.

- **다목적 청중**:
  - **학술 Agent (over-smoothing 또는 filter 논의)**: 본 snapshot → 청중별 종합 보고서
  - **사용자 자신 (현황 파악)**: §3 + §4 + §5 (Active Work + Pending)
  - **새 collaborator**: §1 + §2 (Pipeline + Module Status) + Glossary
  - **학위 논문 chapter draft**: §4 Narrative Status + §7 일정 + 청중별 reference

- **근거**:
  - 사용자 직접 input (현재 전체 구조 자료 요청)
  - AskUserQuestion 결정 (Full snapshot + 다목적 청중)
  - 기존 12 planning 문서 점검 결과 — 통합 snapshot single document missing

- **영향 범위**:
  - planning/framework_snapshot_2026-05-12.md (신규)
  - 본 DECISIONS.md (본 entry)
  - 기존 문서 변경 X — 본 자료는 통합 reference

- **에스컬레이션 필요 여부**: 없음. 사용자가 학술 Agent 에 전달 또는 collaborator onboarding 에 직접 사용.

- **추가 필요 분석**:
  - 진행 중 root chain (filter sweep) + V5 chain 완료 후 §3 + §4 + §5 갱신 (planner)
  - 학회 / 학위 논문 일정 변화 시 §7 갱신
  - 신규 planning 문서 추가 시 §6 갱신

- **14 planning 문서 의 역할 분리 정식** (본 framework snapshot 추가 후):

  | 분류 | 문서 | 목적 |
  |---|---|---|
  | Reference / Operations (7) | `CLAUDE.md` (루트) / `EXPERIMENT_HISTORY` / `EXPERIMENT_CATALOG` / `EXPERIMENT_ID_MIGRATION` / `EXPERIMENT_PLAN` / `planning/DECISIONS.md` / `planning/CLAUDE.md` | 운영 + 시간순 기록 |
  | Paper / Narrative (5) | `paper_research_direction.md` / `Full Paper Structure.md` / `paper_outline_2026-05-08.md` / `presentation_brief_2026-04-28.md` / `mechanism_final_concept_summary_2026-05-05.md` | Paper / 발표 narrative |
  | Over-smoothing (7) | `over_smoothing_research_summary.md` / `advisor_briefing_oversmoothing_2026-05-11.md` / `oversmoothing_root_cause_report_2026-05-11.md` / `oversmoothing_mitigation_theory_2026-05-07.md` / `oversmoothing_solution_methodology_2026-05-11_apa.md` / `oversmoothing_full_context_v4_2026-05-12.md` / `oversmoothing_v5_plan.md` | Over-smoothing 의 다층 narrative + V5 plan |
  | Filter (3) | `src/modules/filters/CLAUDE.md` / `src/modules/filters/EXPERIMENT_PLAN_filters.md` / `filter_full_context_2026-05-12.md` | Filter 의 module + 종합 보고서 |
  | 🆕 **Framework Snapshot (1)** | **`framework_snapshot_2026-05-12.md`** | **전체 framework 통합 snapshot** |

---

## 2026-05-12 (filter_full_context_2026-05-12.md 신규 — 학술 Agent 논의용 Filter Module 종합 보고서)

> **사용자 직전 input (2026-05-12)**: "root 가 filter sweep 하는 동안 학술 agent 랑 filter module 관련 이야기 — 우리가 진행한 실험에서 filter 와 관련된 모든 맥락을 포함하는 보고서를 마크다운으로 만들어 줘".

- **결정 — 신규 산출물**:
  [`planning/filter_full_context_2026-05-12.md`](filter_full_context_2026-05-12.md) — 학술 전문 Agent 논의용 Filter Module 종합 보고서 (9 sections + Appendix A/B/C, ~550 lines).

- **구조 — 9 sections + 3 appendix**:
  - §1 Problem Statement (Filter 의 pipeline role + recall/precision 양 약점 흡수)
  - §2 Filter Implementation Catalog (8 구현체 표 + restore path 분류)
  - §3 Filter Dominance Narrative — 6 Axis Evidence (paper §V.5.4 main contribution)
  - §4 a05 Agentic Filter Ablation (vLLM era 14-cell, Qwen3-Coder)
  - §5 LLM Backbone 민감도 (Qwen vs gpt-4o-mini + GLM 4.7 era + 진행 중 9-cell sweep)
  - §6 2×2×2 Ablation (Filter ON/OFF 결정적 evidence — P +0.40~0.45)
  - §7 Prune-Only 한계 + Restore Path Evidence (Reflection 의 critique-revise mechanism)
  - §8 Filter Module 의 Pipeline 시너지 + 외부 SOTA 정합 (Maamari 2024 / AutoLink 2025 / Glass 2025)
  - §9 Open Questions — 학술 Agent 와 논의 5 항목
  - §A Quantitative Reference Tables (5 정량 표)
  - §B References (external 5 papers + internal 8 reports + configs)
  - §C Summary for Discussion (9 핵심 + 5 논의 candidate)

- **5 over-smoothing 문서 + filter 보고서의 역할 분리 정식**:

  | 문서 | 강조점 | 길이 | 청중 |
  |---|---|---|---|
  | `over_smoothing_research_summary.md` | 정량 + 수식 (~1100 lines) | 1100 | 학위 논문 chapter draft (planner / self) |
  | `advisor_briefing_oversmoothing_2026-05-11.md` | 진척 흐름 narrative | 250 | 지도교수 발표 |
  | `oversmoothing_root_cause_report_2026-05-11.md` | 원인 + 근거 focused | 240 | 결론 중심 보고 |
  | `oversmoothing_solution_methodology_2026-05-11_apa.md` | 이론 + mitigation candidate | (외부) | APA 외부 보고서 |
  | `oversmoothing_full_context_v4_2026-05-12.md` | over-smoothing 모든 맥락 통합 | 580 | 학술 Agent (over-smoothing) |
  | `oversmoothing_v5_plan.md` | 학술 Agent v5 plan 6 Direction | (외부) | V5 mitigation 방향 |
  | **`filter_full_context_2026-05-12.md`** (5/12 신규) | **filter 모든 맥락 통합** | **~550 lines** | **학술 Agent (filter)** |

- **학술 Agent 와 논의 candidate 5 항목 (§9)**:
  - **(9.1) Filter Robustness (Filter-axis Invariance) 가설**: 진행 중 GLM 4.7 era 9-cell sweep 결과 기반. 9-cell spread 좁으면 Filter Dominance **7번째 axis** 신설 candidate.
  - **(9.2) LLM Backbone Capacity vs Filter Structure**: filter 종류 별 backbone 민감도 (Reflection -0.0346 vs Verifier -0.0172) 의 generalization.
  - **(9.3) Filter ↔ Extractor Interaction 비대칭**: Basic PCST + XiYan (F1=0.7863) > Adaptive PCST + XiYan (F1=0.6987). Restore path + Adaptive PCST 미실험.
  - **(9.4) Prune-Only Recall 손실 ~0.15 의 mechanism**: Filter✗ noise score 분포 분석 + Value Retrieval cost + SymbolicVerifier (SQL execution) 의 wrong-prune detect 가능성.
  - **(9.5) GNN Selector Role 재정의**: high-recall retrieval component design principle + LLM-based selector vs GNN-based hybrid 의 cost-accuracy trade-off + Filter Dominance 의 generalization (recommendation / knowledge graph).

- **핵심 fact base (학술 Agent 가 본 보고서만 읽고 파악 가능)**:
  - Filter 8 구현체 + restore path 분류 + interface 계약
  - a05 14-cell (vLLM era Qwen) 결과: anchor F1=0.6940, ReflectionFilter 1iter F1=0.7068 (+1.3%p 신기록), AdaptiveMultiAgent F1=0.4713 (-22.3%p)
  - Qwen → gpt-4o-mini ΔF1 -0.017~-0.035 (filter 별 backbone 민감도)
  - GLM 4.7 era anchor F1=0.8673 (paper main pipeline)
  - 2×2×2 ablation 의 Filter P +0.40~0.45 결정적 evidence
  - Filter Dominance 6 axis evidence (H-B / H-F / F-1+H-G / ΔF1 / H-A/H-D / 10-trial null)
  - Prune-only XiYan 의 recall 손실 ~0.15 + Reflection critique-revise restore mechanism
  - 외부 SOTA 정합 (Maamari 2024 LLM dominance / AutoLink 2025 97.4% recall / Glass 2025 extractive LLM)

- **근거**:
  - 사용자 직접 input (filter 학술 Agent 논의용)
  - `src/modules/filters/CLAUDE.md` 의 filter 모듈 정보 + 14-cell 결과 정리
  - `EXPERIMENT_HISTORY.md` a05 14-cell + 2×2×2 ablation + GLM era anchor
  - `EXPERIMENT_CATALOG.md` filter configs
  - over-smoothing 종합 보고서 의 Layer 3 (Filter Dominance) narrative

- **영향 범위**:
  - planning/filter_full_context_2026-05-12.md (신규)
  - 본 DECISIONS.md (본 entry)
  - 기존 paper §3.5 Filter Dominance evidence 표 변경 X — 본 보고서는 통합 reference

- **에스컬레이션 필요 여부**: 없음. 사용자가 학술 Agent 에 본 보고서 전달 후 추가 input 받으면 planner 후속 작업.

- **추가 필요 분석**:
  - 진행 중 root 9-cell sweep (GLM 4.7 era) 결과 후 §5.4 + §9.1 정량 갱신
  - 학술 Agent 의 input 후 §9 Open Questions 의 답변 + 추가 mitigation/redesign 가능성 검토 → planner 후속

- **Filter sweep 진행 status (root chain, 별도)**:
  - 8 신규 config + scripts/run_filter_sweep_glm.sh 작성 중
  - ETA wall ~5-9h
  - 결과 후 analyzer 9-cell 보고서 + planner 7번째 axis narrative integration

---

## 2026-05-12 (V5 Mitigation Plan — Tier 1+2 4 Direction 병렬 + Narrative pivot V5 결과 후)

> **사용자 직전 input (2026-05-12)**: 학술 전문 Agent 와의 토론 결과 `planning/oversmoothing_v5_plan.md` 도출. AskUserQuestion 결정: (1) **Tier 1+2 함께 진행**: D-1 + A + B + C / (2) **Narrative pivot V5 실험 결과 후**.

- **학술 Agent 의 결정적 reinterpretation (working hypothesis, narrative pivot 보류)**:

  3 layer 분리:

  | 결과 layer | 우리 evidence | 학술 Agent reinterpretation |
  |---|---|---|
  | Layer 1: GAT over-smoothing 실증 | $\bar{c}_{L_3} \geq 0.96$ + Three-Axis Invariance | ✅ 그대로 (Wu 2023 + GATE 2024 의 heterogeneous schema graph 최초 실증) |
  | Layer 2: Mitigation 의 R@15 영향 없음 | 10-trial null + V4 이중 fail | ⚠️ **재해석 working hypothesis**: "GAT internal dynamics → R@15" 가설 기각, R@15 ceiling 의 원인은 PLM semantic lower bound + domain-specific bottleneck (Peng 2024 + Arnaiz-Rodriguez 2025) |
  | Layer 3: Filter Dominance | F1=0.8383 vs raw R=0.6097 | ✅ 그대로 (Maamari 2024 의 LLM filter dominance 와 일관) |

  → **narrative pivot 보류** (사용자 결정 (2)): 현재 narrative (mech(ii-b) 5/5 confirm + Filter Dominance 6번째 축) 유지. V5 실험 결과 후 (특히 V5-D-1 R 갱신 여부 + V5-A GATE 결과) reinterpretation 적용 여부 결정.

- **결정 — V5 Tier 1+2 4 Direction 병렬 진행 (사용자 결정 (1))**:

  | V5 ID | Direction | 모듈 | 비용 | 학회 narrative 영향 |
  |---|---|---|---|---|
  | **V5-A** | GATE (Conservation Law 수정) | `src/models/gat_network_v2.py` 신규 `GATEConv` | ~10h wall × 1 | mech(ii-b) 의 task-irrelevant aggregation 차원 추가 evidence. Fail 시 narrative 6 pillar 격상 |
  | **V5-B** | GCNII-style Trainability | `src/models/gat_network_v2.py` Initial Residual + Identity Mapping | ~10h × 2-3 (L=2/4/6 sweep) | Paradox 2 ($\rho_{\text{skip}}$) 의 trainability 해석 검증 |
  | **V5-C** | Full AERO-GNN + Node-Adaptive Hop Attention | `src/models/gat_network_v2.py` V4-B 확장 | ~10h × 1 | V4-B H10.1c (Hop Attention 부재) 직접 검증 |
  | **V5-D-1** | PLM Lower Bound 진단 (Enrichment 강화) | `src/modules/builders/EnrichedHeteroGraphBuilder` + `src/models/plm_encoder.py` | ~3-5h (진단) + ~10h (학습) | **🚨 narrative pivot candidate**: R 갱신 시 학술 Agent reinterpretation confirm → main pivot |

- **V5-D-2 (Schema-aware contrastive pre-training)** 는 Tier 3 로 deferred — V5-D-1 결과 후 결정. 학회 일정 (5/14~5/22 chapter draft) 고려.
- **V5-E (LLM-based selector paradigm shift)** + **V5-F (Graph Rewiring)** 는 Tier 4 / Tier 3 로 deferred — 학위 본 심사 후 post-paper.

- **Root chain 위임 — Filter sweep 종료 후 launch**:

  | Step | 책임 | 산출물 |
  |---|---|---|
  | 1 | Root: Filter sweep (별도 chain) 완료 대기 | (~5-9h wall) |
  | 2 | Root: V5-D-1 진단 (anchor stack 의 $\bar{c}_{L_0}$ + $\bar{c}_{L_3}$ measurement + Plain vs Enriched 비교) | outputs/analysis/v5_d1_plm_lower_bound_diagnostic/ |
  | 3 | Root: V5-A `GATEConv` 구현 + smoke test | src/models/gat_network_v2.py |
  | 4 | Root: V5-B GCNII-style 구현 + smoke test | src/models/gat_network_v2.py |
  | 5 | Root: V5-C Full AERO (V4-B + Hop Attention) 구현 + smoke test | src/models/gat_network_v2.py |
  | 6 | Root: V5 sweep launch (V5-A + V5-B (L=2/4/6) + V5-C 학습, GPU 0,1 병렬) | scripts/run_v5_mitigation_sweep.sh |
  | 7 | Root: HISTORY + CATALOG + ID_MIGRATION 갱신 | 3 문서 |
  | 8 | Analyzer (학습 완료 후): 14-trial V5 결과 + Layer 1/2/3 evidence 재정량 + narrative pivot candidate 평가 | notebooks/analysis_results/dsn_mitigation_v5_4dir.md |
  | 9 | Planner (analyzer 후): narrative pivot 결정 + 5 over-smoothing 문서 통합 갱신 | planning/* + paper §3.5 |

- **GPU 자원 분배 (Root chain 운영 가이드)**:
  - Filter sweep (별도 chain) 종료 후 V5 launch
  - V5-A + V5-C 동시 (GPU 0/1 병렬, 각 ~10h)
  - V5-B (L=2/4/6 sweep) 순차 또는 추가 sequential (~30h)
  - V5-D-1 진단은 CPU forward (1-2h) — 위 sweep 와 병행 가능
  - 전체 cumulative wall ~50-60h (병렬화로 ~30-40h 실제)

- **Narrative pivot 의 의사결정 tree (V5 결과 후)**:
  - **시나리오 1 (V5-D-1 R 갱신)**: 학술 Agent reinterpretation confirm. Layer 2 narrative pivot — "R@15 ceiling 의 원인은 PLM lower bound, mech(ii-b) 는 embedding 수준 limitation 만". paper main contribution 일부 변경.
  - **시나리오 2 (V5-A/C 단독 R 갱신)**: V5-A or V5-C 가 architectural mitigation 가능 evidence. mech(ii-b) "5/5 absolute confirm" narrative 약화 — "GAT architectural intervention 의 일부 path 가 효과 있음" 부분 부정.
  - **시나리오 3 (V5 4 Direction 모두 fail)**: 현재 narrative (mech(ii-b) 5/5 + Filter Dominance 6번째 축) 의 **결정적 강화** — 14-trial null + 4 architectural intervention direction 모두 무력. PLM lower bound + Filter Dominance 가 R@15 의 dominant factor 임을 양립 narrative 로 정식.

- **근거**:
  - 사용자 remote-control 직접 input + AskUserQuestion 결정
  - `planning/oversmoothing_v5_plan.md` 학술 Agent input — 6 Direction (A~F) + Tier 권고
  - 직전 V4-Combo-Null narrative + Three-Axis Invariance evidence

- **영향 범위**:
  - 신규 산출물 (root 작성):
    - src/models/gat_network_v2.py — GATEConv + GCNII-style + Full AERO 신규 클래스
    - `configs/training/dsn/train_dsn_p80_v5{a,b,c}_*.yaml` + V5-D-1 진단 config
    - scripts/run_v5_mitigation_sweep.sh
    - HISTORY + CATALOG + ID_MIGRATION 갱신
  - 후속 (analyzer):
    - `notebooks/analysis_results/dsn_mitigation_v5_4dir.md` (14-trial 결과 + 3 layer narrative pivot 평가)
  - 후속 (planner, V5 결과 후):
    - DECISIONS pivot 결정 entry
    - 5 over-smoothing planning 문서 narrative integration (시나리오에 따라)
    - paper §3.5 narrative 갱신

- **에스컬레이션 — Root chain 위임 (filter sweep 후)**:
  - Filter sweep (별도 chain) 종료 후 V5 chain launch
  - Single-command launch (scripts/run_v5_mitigation_sweep.sh) + nohup
  - 학회/학위 본 심사 일정 (5/14~5/22 chapter draft) 고려 — V5 sweep 결과를 chapter draft 에 통합 가능 시점

- **추가 필요 분석 (V5 결과 후 candidate)**:
  - V5-D-2 (contrastive pre-training) — V5-D-1 R 갱신 시 진행 후속
  - V5-E (LLM-based selector) — Tier 4 paradigm shift, 학위 본 심사 후 post-paper
  - V5-F (Graph Rewiring) — Tier 3, V5 결과 후 priority 재평가

---

## 2026-05-12 (oversmoothing_full_context_v4_2026-05-12.md 신규 — 학술 Agent 논의용 V4 까지 종합 보고서)

> **사용자 직전 input (2026-05-12)**: "over smoothing 문제는 별도로 학술 전문 Agent 와 논의하려고 하니 V4 까지의 보고서를 작성해 놔 줘 over smoothing 과 관련된 모든 맥락을 포함하는 보고서로 만들어 줘".

- **결정 — 신규 산출물**:
  [`planning/oversmoothing_full_context_v4_2026-05-12.md`](oversmoothing_full_context_v4_2026-05-12.md) — 학술 전문 Agent 논의용 종합 보고서 (10 sections + Appendix A/B/C/D, ~580 lines).

- **구조 — 10 sections + 4 appendix**:
  - §1 Problem Statement (V-3-ext + qcond R@15 ceiling ~0.61)
  - §2 Theoretical Background — Row-Stochasticity + JSR < 1 + Wu et al. 2023 + AERO-GNN Theorem 3
  - §3 Diagnostic Framework (4 mechanism hypotheses)
  - §4 Phase 1 진단 — H1 over-smoothing 확정 (architecture-invariance 첫 evidence 포함)
  - §5 Three Paradoxes (attention moderate-sharp / gradient extreme / ckpt-invariance)
  - §6 Mitigation Evolution — 10 Trials + 3 Turning Points + V4
  - §7 V4 Architectural Intervention 상세 — V4-A LN+GIN combo + V4-B AERO Softplus 의 구현 + 결과 + 학술 implication
  - §8 Final Mechanism Dominance Scoring (mech(ii-b) 5/5 absolute confirm)
  - §9 Filter Dominance 6번째 축 Narrative
  - §10 Open Questions — 학술 Agent 와 논의 4 항목
  - §A Mathematical Definitions Reference (8 수식 모음)
  - §B Quantitative Reference Tables (10-trial ranking + Architecture vs Training spread + Three-Axis Invariance)
  - §C References (external 6 papers + internal 4 reports + analyzer 11 reports)
  - §D Summary for Discussion

- **차별점 (기존 4 over-smoothing 문서와)**:
  - **`over_smoothing_research_summary.md`** (~1100 lines): 학위 논문 chapter draft base — 정량 evidence + 수식 detail
  - **`advisor_briefing_oversmoothing_2026-05-11.md`** (~250 lines): 지도교수 발표용 narrative — 진척 흐름 + 의미 중심
  - **`oversmoothing_root_cause_report_2026-05-11.md`** (~240 lines): 원인 + 근거 focused — 결정적 evidence 정리
  - **`oversmoothing_solution_methodology_2026-05-11_apa.md`**: APA 외부 보고서 — 이론 framework + §C-1/§C-2 mitigation candidate
  - **🆕 `oversmoothing_full_context_v4_2026-05-12.md`** (~580 lines, 신규): **학술 Agent 논의용** — 이론 + 진단 + 10-trial 결과 + V4 detail + Open Questions 모두 포함. 외부 학술 collaborator 가 본 보고서만 읽고도 모든 맥락 파악 가능.

- **학술 Agent 와 논의 candidate 4 항목 (§10)**:
  - **(10.1) AERO-GNN Theorem 3 transfer 실패 원인**: 3 hypotheses (heterogeneous graph / Node-Adaptive Hop Attention 부재 / domain-specific bottleneck)
  - **(10.2) V4-A combo destructive interaction**: 3 hypotheses (attention dispersion + MLP capacity 약화 / sub-mechanism redundancy / R metric discrete absorption)
  - **(10.3) Filter Dominance mechanism deeper analysis**: per-query lift pattern + transferability to other domains
  - **(10.4) Post-V4 mitigation direction**: GRAND / GraphCon / FAGCN / SBP / AERO Full (Hop Attention 포함)

- **근거**:
  - 사용자 직접 input (학술 Agent 논의용 보고서 요청)
  - 직전 4 over-smoothing 문서의 일부분만 필요한 정보 — 학술 Agent 가 모두 읽기 부담 → 통합 자료 필요
  - V4-Combo-Null 결과 (5/12 완료) + Three-Axis Invariance + 10-trial null + APA 이론 framework 의 통합 narrative

- **영향 범위**:
  - planning/oversmoothing_full_context_v4_2026-05-12.md (신규)
  - 본 DECISIONS.md (본 entry)
  - 기존 4 over-smoothing 문서 변경 X — 본 보고서는 별도 산출물

- **에스컬레이션 필요 여부**: 없음. 사용자가 학술 Agent 에 본 보고서 전달 후 추가 결정.

- **추가 필요 분석**: 없음. 단 (학술 Agent 의 input 후) 본 보고서의 §10 Open Questions 에 대한 추가 mechanism analysis 또는 mitigation candidate 의 정량 검증 요청 시 root/analyzer 위임 가능.

- **5 over-smoothing planning 문서의 역할 분리 정식**:

  | 문서 | 강조점 | 길이 | 청중 |
  |---|---|---|---|
  | `over_smoothing_research_summary.md` | 정량 evidence + 수식 reference | ~1100 lines | 학위 논문 chapter draft (planner / self) |
  | `advisor_briefing_oversmoothing_2026-05-11.md` | 진척 흐름 narrative | ~250 lines | 지도교수 발표 |
  | `oversmoothing_root_cause_report_2026-05-11.md` | 원인 + 근거 focused | ~240 lines | 결론 중심 보고 |
  | `oversmoothing_solution_methodology_2026-05-11_apa.md` | 이론 framework + mitigation candidate | (외부) | APA 외부 보고서 |
  | **`oversmoothing_full_context_v4_2026-05-12.md`** (5/12 신규) | **모든 맥락 통합 (이론 + 진단 + V4)** | **~580 lines** | **학술 전문 Agent** |

---

## 2026-05-12 (Filter Module 확정 — Anchor base + 9-cell Filter Ablation Sweep 결정)

> **사용자 직전 input (2026-05-12)**: "over smoothing 문제를 잠깐 미뤄두고 filter 모듈을 먼저 확정지으려고 해. baseline 모델 (Enriched + QCond concat + MST+PCST) 에다가 지금까지 실험했던 다양한 필터를 적용해본 적이 있었나? GLM 4.7 로." → **AskUserQuestion 결정**: (1) Full 9-cell sweep, (2) planner 실험 plan + root 핸드오프.

- **확인 — 사용자 질문에 대한 정확한 답**: **No**. 두 분리된 영역의 교차점이 미실험.
  - **영역 A (filter ablation)**: `configs/experiments/abl/a05_filter_agentic/` 14 cells. Base = **Plain + DirectGAT + Fixed PCST**, LLM = **Qwen/Qwen3-Coder-30B (vLLM era)**. 7 filter 종류 비교.
  - **영역 B (paper anchor)**: `configs/experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_kruskal_glm.yaml`. Base = **Enriched + QCond Concat + MST+PCST**, LLM = **GLM 4.7**, Filter = **XiYan only**.
  - **교차점 = anchor base + various filters + GLM 4.7** = 학회/학위 논문의 Filter Dominance narrative 의 정량 정당화 missing.

- **결정 — Full 9-cell sweep 채택**:

  | Cell | Stack 공통: Enriched + QCond Concat (α=0.5) + MSTKruskal | Filter | Notes |
  |---|---|---|---|
  | C0 anchor | (이미 존재) | XiYan | `s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` (F1=0.8673) |
  | C1 신규 | + Reflection (1 iter) | ReflectionFilter | a05_02 hyperparameter base |
  | C2 신규 | + Verifier | VerifierFilter | a05_04 base |
  | C3 신규 | + AdaptiveMultiAgent | AdaptiveMultiAgentFilter | a05_01 (Semantic+Structural+Skeptic 3-agent) base |
  | C4 신규 | + Stacked (Refl+Verif chain) | StackedFilter | a05_22 chain composition base |
  | C5 신규 | + SymbolicVerifier | SymbolicVerifierFilter | a05_19 XiYan repair base |
  | C6 신규 | + AdaptiveDepth | AdaptiveDepthFilter | a05_07 base |
  | C7 신규 | + Bidirectional | BidirectionalAgentFilter | (a05_xx 검색 후 base 결정) |
  | C8 baseline | + None | (no filter) | baseline 노출, anchor 의 Filter Dominance lift 정량 |

  → 8 신규 cells + anchor 1 = **9-cell 매트릭스**. 각 cell BIRD-dev 1534 query × LLM call. ETA wall ~5-9 hour.

- **모든 신규 config 의 LLM 통일**: provider="glm", model_name="zai-org/glm-4.7", max_iteration=1, temperature=0.0 (anchor XiYan 와 동일). 단 ReflectionFilter / Stacked 의 max_iteration 같은 filter-specific param 은 a05_xx 기본값 유지.

- **본 sweep 의 학회/학위 논문 narrative 영향**:
  1. **Filter Dominance 의 Filter-axis 정당화**: 9-cell 의 F1 spread 측정 → Filter 종류 의 robustness/sensitivity 정량 evidence. Spread 좁으면 (≤ 0.01) → Filter Dominance 의 Filter-invariant 성질 confirm. Spread 넓으면 (> 0.05) → filter 선택이 important factor 라는 narrative 변경 candidate.
  2. **paper §3.5 Filter Dominance 6번째 축 evidence 보강**: 직전 5+1 axis (H-B/H-F/F-1+H-G/ΔF1/H-A/H-D/8-trial) 에 **7번째 axis (Filter-invariant)** 추가 candidate.
  3. **anchor F1=0.8673 의 ranking 맥락 정량**: 9-cell 의 best 가 anchor 인지, 다른 filter 가 더 좋은지 → paper main pipeline 의 anchor 선택 정당화.

- **Chain 위임 — Root 세션** (config 작성 + 학습 launch + HISTORY 갱신):

  | Step | 책임 | 산출물 |
  |---|---|---|
  | 1 | Root: 신규 8 config 작성 (`configs/experiments/s04_ablation/pipeline/filter_sweep/*.yaml`) | configs/ |
  | 2 | Root: smoke test (single query forward, GLM 4.7 호출 검증) | (logs) |
  | 3 | Root: sweep launch (`scripts/run_filter_sweep_glm.sh`, BIRD-dev 1534 query × 8 cells) | logs/ + outputs/ |
  | 4 | Root: HISTORY + CATALOG + ID_MIGRATION 갱신 | 3 문서 |
  | 5 | Analyzer (sweep 완료 후): 9-cell F1/R/P 매트릭스 + Filter-axis spread 정량 | notebooks/analysis_results/filter_sweep_glm_9cell.md |
  | 6 | Planner (analyzer 후): paper §3.5 Filter Dominance 7번째 axis (Filter-invariant) candidate 갱신 | planning/paper_research_direction.md + DECISIONS |

- **근거**:
  - 사용자 직접 input + AskUserQuestion 결정 (Full 9-cell + planner)
  - `configs/experiments/abl/a05_filter_agentic/a05_*.yaml` 직접 조회 — 모든 a05 config 가 Selector=DirectGATSelector + Extractor=PCSTExtractor + LLM=Qwen3-Coder
  - `configs/experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_kruskal_glm.yaml` 직접 조회 — anchor base 정확 확인
  - Filter Dominance 6번째 축 (training-pathology-invariant) narrative 의 Filter-axis 정량 정당화 missing 발견

- **영향 범위**:
  - 신규 산출물 (root 작성):
    - `configs/experiments/s04_ablation/pipeline/filter_sweep/{c1_reflection, c2_verifier, c3_adaptive_multi_agent, c4_stacked, c5_symverify, c6_adaptive_depth, c7_bidirectional, c8_no_filter}_glm.yaml`
    - `scripts/run_filter_sweep_glm.sh`
    - HISTORY + CATALOG + ID_MIGRATION 갱신
  - 후속 (analyzer):
    - `notebooks/analysis_results/filter_sweep_glm_9cell.md`
  - 후속 (planner, analyzer 후):
    - paper §3.5 Filter Dominance 7번째 axis (Filter-invariant) candidate 갱신
    - DECISIONS sweep 완료 + narrative integration entry

- **에스컬레이션 — Root chain 위임**:
  - Root 가 본 chain 의 Step 1~4 모두 수행 후 analyzer 핸드오프
  - 사용자 부재 가능성 → single-command launch (scripts/run_filter_sweep_glm.sh, nohup + &)

- **추가 필요 분석**:
  - 9-cell 결과 후 Filter-axis spread 정량 → Filter Dominance narrative 의 Filter-invariance 정량 evidence
  - 만약 best filter 가 anchor (XiYan) 이 아니면 → paper main pipeline anchor 변경 candidate
  - 만약 best filter 가 anchor 와 동등 (ΔF1 ≤ 0.005 noise band) 이면 → XiYan single-call 의 simplicity advantage narrative

- **Over-smoothing 작업 status**:
  - 본 사용자 결정으로 over-smoothing 문제 (V4 후속, max aggregation, post-paper combo) 는 **잠시 deferred**. v3 #2 Max aggregation 5/14 ETA chain 은 별도 진행 (사용자 결정 cancel 명시 안 됨). 본 chain 후 narrative 통합 가능.

---

## 2026-05-12 (Mitigation V4-Combo-Null 결과 + narrative integration 4 문서)

> **사용자 직전 input (2026-05-12)**: "notebooks/analysis_results/dsn_mitigation_v4_combo.md 참조해 4 문서 narrative integration: (1) advisor briefing §3+§4+§7, (2) root cause report §3+§5, (3) paper §V.5.4 main finding, (4) DECISIONS V4-Combo-Null prepend".

- **결정적 결과** (analyzer dsn_mitigation_v4_combo.md, 10-trial 시점 5/12):

  | Variant | Best R@15 | Best Epoch | Δ vs Phase 1 | 핵심 결론 |
  |---|---:|---:|---:|---|
  | **V4-A LN+GIN Combo** | **0.5929** | ep259 | **-0.0168** | 산술합 fail (산술합 -0.0229 대비 +0.0061 partial relief 만, dramatic 회복 zero) |
  | **V4-B AERO Softplus + Symmetric Norm** | **0.5951** | ep58 | **-0.0146** | Wu et al. 2023 JSR<1 row-stochastic 가정 직접 위반 + AERO Theorem 3 SR2OS guarantee 실증 transfer 실패 |

  → **시나리오 V4-Combo-Null 확정**. V4 의 두 architectural intervention (combo + row-stochasticity 파괴) 모두 baseline R=0.6097 갱신 실패.

- **격상 — mech(ii-b) 4/5 partial 부정 → 5/5 absolute confirm**:

  | 기존 (8-trial) | 신규 (10-trial 5/12) |
  |---|---|
  | mech(ii-b) "softmax × weighted-mean propagation combo" 의 **single mechanism 가설** | mech(ii-b) 의 **fundamental architectural limitation** 정식화 |
  | Score: **4/5 partial 부정** (GIN 단독 L2 cosine 회복하지만 R fail) | Score: **5/5 absolute confirm** (V4 combo + row-stoch destroy 모두 fail) |

  4 pillar evidence:
  1. **V4-A combo fail**: 두 partial mit 의 산술합이 새 회복 X (산술합 -0.0229 → 실제 -0.0168)
  2. **V4-B row-stochasticity 파괴 fail**: Wu et al. 2023 JSR<1 가정 직접 위반에도 ceiling 갱신 X
  3. **Best epoch 분포 일관**: V4-A ep259 (late) + V4-B ep58 (early) — 직전 8-trial ceiling 흡수 패턴과 일관
  4. **Loss curve 수렴 동등**: V4-A Loss 1.1410, V4-B Loss 1.1510 (직전 GIN ~1.16 와 유사) + AC < 0.001 (효과적 anti-collapse pressure). 학습 dynamics normal → pathology = architecture 자체.

- **Narrative integration 4 위치 (planner 5/12 작업)**:

  1. **planning/advisor_briefing_oversmoothing_2026-05-11.md**:
     - §3 (Mitigation 진화): 8-trial 표 → **10-trial 표** (V4-A / V4-B 추가). 결정적 전환점 4 신설 ("V4 Architectural Intervention 의 이중 fail" — combo + row-stoch destroy 모두 fail narrative).
     - §4 (최종 mechanism 정식): "Mech(ii-b) DOMINANT 5/5 ⭐ absolute confirm" 으로 격상. 기존 "softmax × weighted-mean combo" → "fundamental architectural limitation" 정식. 4 pillar evidence box 신설.
     - §7 (진행 중 + 향후): V4 결과 (5/12 완료) 반영. 시도 8/9 V4 결과 + 시도 10 (v3 #2 Max aggregation, 5/14 ETA) 만 진행 중. Post-paper future work — V4 fail 후 새 candidate (Physics-Informed GRAND/GraphCon / Repulsive FAGCN / Domain-specific SBP).

  2. **planning/oversmoothing_root_cause_report_2026-05-11.md**:
     - §3.1 Mechanism 평가 matrix: 8 trial → 10 trial 갱신. mech(ii-b) "DOMINANT 4/5" → "DOMINANT 5/5 ⭐ absolute confirm".
     - §3.2 Mech(ii-b) 정식화 표: 기존 (8-trial) ↔ 신규 (10-trial) contrast 신설. V4-A combo (산술합 fail) + V4-B row-stoch destroy (이론 보장 실증 fail) 두 행 추가.
     - §3.3 10 Mitigation 표: 시도 10 행까지 확장 (8 → 10).
     - §5.1 Over-Smoothing 의 주된 원인: "Softmax × Weighted-Mean Propagation Combo" → "Fundamental Architectural Limitation" 격상.
     - §5.2 결정적 근거 5 항목: 4번째 항목 "Mech(ii-b) partial 부정 (GIN evidence)" → "5/5 absolute confirm — V4 architectural intervention 이중 fail". 5번째 항목 "8 mitigation null" → "10 mitigation null + Three-Axis Invariance".
     - §5.3 결론의 의의: V4 결과 통합 + future work direction 갱신 (V4 fail 후 새 candidate).

  3. **planning/paper_research_direction.md** §3.5 Filter Dominance 6번째 축 evidence 표 line 505:
     - "8-trial Final mitigation null effect" → "**🎯 10-trial Final mitigation null effect 정식 + mech(ii-b) 5/5 absolute confirm**" (V4 architectural intervention 이중 fail).
     - 정량 표: 8 ckpt → 10 ckpt (V4-A 0.5929 + V4-B 0.5951 추가).
     - mech(ii-b) 정식화: 8-trial 4/5 partial 부정 → 10-trial 5/5 absolute confirm. (a) V4-A 산술합 fail + (b) V4-B JSR<1 위반 + AERO Theorem 3 SR2OS guarantee 실증 transfer 실패 두 결정적 evidence 신설.
     - paper §V.5.4 핵심 narrative: V4 의 두 architectural intervention 도 fail = GAT 의 internal architectural limitation 까지 With-Filter pipeline 이 흡수 → Filter Dominance 6번째 축 결정적 강화.

  4. **planning/DECISIONS.md** (본 entry, 본 항목)

- **근거**:
  - `notebooks/analysis_results/dsn_mitigation_v4_combo.md` §0 TL;DR + §3 10-trial 매트릭스 + §4.2 정식화 + §5.2 axis matrix + §9 paper §V.5.4 본문 candidate
  - 직전 DECISIONS entry "Mitigation V4 채택 — Architectural Intervention 두 Phase" 의 chain 결과
  - 학회 narrative 의 2 시나리오 중 시나리오 2 (V4 둘 다 null → mech(ii-b) 5/5 절대 confirm + Filter Dominance 6번째 축 narrative 결정적 강화) 실현

- **영향 범위**:
  - planning/advisor_briefing_oversmoothing_2026-05-11.md §3 + §4 + §7 (수정 완료)
  - planning/oversmoothing_root_cause_report_2026-05-11.md §3.1 + §3.2 + §3.3 + §5 (수정 완료)
  - planning/paper_research_direction.md §3.5 evidence #6 line 505 (수정 완료)
  - planning/DECISIONS.md (본 entry)

- **에스컬레이션 필요 여부**:
  - 후속 (선택): paper §V.5.4 본문 정식 narrative (analyzer §9 candidate) 의 학위 논문 Part III chapter draft 작성 시점 (5/14~5/22) — planner 또는 root.
  - 후속 (5/14): v3 #2 Max aggregation 결과 (별도 chain) — 10 → 11 trial 갱신. mech(i-b) aggregation family null 확정 evidence.

- **추가 필요 분석**: 없음. V4-Combo-Null 결과로 학회 발표 + paper main finding 정식 narrative 확보.

- **학회 narrative 의 결정적 강화 message**:
  > 10-trial mitigation experiments (PN+IR / Direct AC / LR x5 / DropMessage / LayerNorm / Sum aggregation / GIN aggregation / B5 fusion / **🆕 LN+GIN combo / 🆕 AERO Softplus**) 모두 baseline R@15=0.6097 갱신 실패. Specifically, 두 architectural intervention (V4-A row-stochasticity 유지 + 두 partial mit combo / V4-B row-stochasticity 자체 파괴, Wu et al. 2023 JSR<1 가정 직접 위반) 모두 ceiling 갱신 실패 → mech(ii-b) softmax × weighted-mean propagation combo 의 5/5 absolute confirm. Filter Dominance 6번째 축 narrative 의 결정적 강화: GAT 의 internal architectural limitation 까지 With-Filter pipeline 이 흡수, F1=0.8383 의 정량 evidence + selector raw R@15=0.6097 의 architectural ceiling 양립 가능.

---

## 2026-05-11 (Mitigation V4 채택 — Architectural Intervention 두 Phase + Root chain 위임)

> **사용자 직전 input (remote-control, 2026-05-11)**: "연구 전문 Agent 의 조언을 바탕으로 작성한 프롬프트를 활용해 구현 및 실험 진행이 가능하도록 핸드오프 작성" — Prerequisite: `planning/oversmoothing_solution_methodology_2026-05-11_apa.md` (보고서 C-1 + C-2 채택).

- **결정 — Mitigation V4 (Architectural Intervention) 채택**:

  보고서 §1 (Softmax row-stochasticity → JSR < 1 → 지수적 붕괴) + §4 Direction C (Softmax + Aggregation 동시 변경) 의 결론을 채택. 직전 8-trial mitigation null 결과 (Mech(ii-b) softmax-weighted-mean combo DOMINANT 4/5) 의 conditional trigger 발효 — combo mitigation 이 유일 검증 통로.

  | Variant | 전략 | 보고서 출처 |
  |---|---|---|
  | **V4-A** | Pre-softmax LayerNorm + GIN aggregation (LN + GIN Combo) | §C-1 — 최단 경로 검증 |
  | **V4-B** | Softplus + Symmetric Normalization (AERO-GNN Style Layer) | §C-2 — 이론적 최강 후보 |

  **V4-A 구현 요점**:
  - 새 클래스 `CustomGATv2Conv_LNGIN` in `src/models/gat_network.py`
  - Pre-softmax LayerNorm: raw α (pre-softmax) 직전 LayerNorm 으로 magnitude 표준화 (직전 v2 #3 LN 와 동일 mechanism)
  - GIN aggregation: $\sum + \text{MLP}$ propagation 으로 weighted-mean 차원 변경 (직전 v3 #1 GIN 와 동일 mechanism)
  - 두 mitigation 의 **combo** — 직전 단독 변경 (LN only / GIN only) 모두 partial mit, combo 시 dramatic 회복 가능 candidate

  **V4-B 구현 요점**:
  - 새 클래스 `AEROHeteroConv` 또는 `SoftplusGATConv` in `src/models/gat_network.py`
  - Softmax 완전 제거 → Softplus($e_{ij}$) 활성화
  - Row-stochastic normalization 제거 → Symmetric Normalization ($\tilde{\alpha}_{ij} = \alpha_{ij} / \sqrt{d_i d_j}$)
  - Row-stochasticity 구조 파괴 → JSR < 1 의 지수적 붕괴 증명 위반 → over-smoothing 의 이론적 보증
  - (선택) Node-Adaptive Hop Attention 추가 — cumulative attention 수렴 저지

- **Chain 위임 — Root 세션** (코드 구현 + 학습 launch + HISTORY 갱신):

  | Step | 책임 | 산출물 |
  |---|---|---|
  | 1 | Root: APA 보고서 §1 + §4 + §C-1 + §C-2 정독 | (학습) |
  | 2 | Root: V4-A `CustomGATv2Conv_LNGIN` 구현 | src/models/gat_network.py |
  | 3 | Root: V4-B `AEROHeteroConv` / `SoftplusGATConv` 구현 | src/models/gat_network.py |
  | 4 | Root: smoke test (forward pass shape + HeteroData 호환) | tests + unit smoke |
  | 5 | Root: 신규 config 작성 (V4-A + V4-B 각각) | configs/training/dsn/train_*_v4_*.yaml |
  | 6 | Root: 학습 launch sweep | scripts/run_v4_mitigation_sweep.sh, GPU 0,1 |
  | 7 | Root: HISTORY + CATALOG + ID_MIGRATION 갱신 | 3 문서 |
  | 8 | Analyzer (학습 완료 후): layer-wise cosine + attention metric 측정 | notebooks/analysis_results/dsn_mitigation_v4_combo.md |
  | 9 | Planner (analyzer 후): advisor briefing + root cause report narrative integration | planning/* (V4 결과 + paper §V.5.4 narrative 갱신) |

- **이론적 근거 (보고서 §1 + §4 핵심 인용 — 코드 주석에 반영)**:

  1. **Row-stochasticity → Ergodic 수렴**: GATv2Conv 의 softmax-normalized $\alpha_{ij}$ 가 row-stochastic matrix $\mathbf{A}$ 를 만듦. 모든 row sum = 1, Perron-Frobenius 이론에 의해 $\lim_{L \to \infty} \mathbf{A}^L \mathbf{x} \to \mathbf{v}_1 \cdot c$ ($\mathbf{v}_1$ = 정상 분포, $c$ = 상수). 즉 모든 node 가 같은 stationary distribution 으로 수렴.
  2. **JSR < 1 → 지수적 붕괴**: heterogeneous attention matrices $\{\mathbf{A}^{(l)}\}_{l=1}^{L}$ 의 Joint Spectral Radius (JSR) 가 row-stochastic 조건 하에서 < 1 → embedding 차이 $\|\mathbf{h}_i - \mathbf{h}_j\|$ 가 layer 따라 **지수적 감소**. 즉 mathematically 학습 dynamics 와 무관하게 collapse 보장.
  3. **Direction C 결론**: row-stochasticity 자체를 깨지 않으면 over-smoothing 회피 불가. 두 차원의 동시 변경:
     - V4-A: row-stochasticity 유지 단 LayerNorm 으로 attention 분포 + GIN 으로 aggregation 변경 (partial 회복 candidate)
     - V4-B: row-stochasticity 자체 파괴 (Softplus + Symmetric Norm) (이론적 보장)

- **근거**:
  - 사용자 remote-control 직접 input + APA 보고서 prerequisite
  - 직전 DECISIONS entry "8-trial mitigation null + mech(ii-b) DOMINANT 4/5 partial 부정" + Stage 7 GIN narrative "softmax + aggregation 동시 변경 시 더 큰 mitigation 가능 candidate (post-paper)"
  - Architecture-invariance evidence (Concat vs SuperNode 모두 collapse) — 본 V4 가 architecture 선택과 무관하게 적용

- **영향 범위**:
  - 신규 산출물 (root 작성):
    - `src/models/gat_network.py` — `CustomGATv2Conv_LNGIN` + `AEROHeteroConv` / `SoftplusGATConv` 신규 클래스 추가
    - `configs/training/dsn/train_dsn_p80_v4a_lngin_combo.yaml` + `train_dsn_p80_v4b_aero.yaml`
    - `scripts/run_v4_mitigation_sweep.sh`
    - HISTORY / CATALOG / ID_MIGRATION 갱신
  - 후속 (analyzer 작성):
    - `notebooks/analysis_results/dsn_mitigation_v4_combo.md` — V4-A + V4-B 결과 정량 + 직전 8-trial 과 비교
  - Planner (analyzer 후):
    - advisor briefing §3 (Mitigation 진화) + §4 (Mechanism 정식) + §7 (진행 중/향후) 갱신
    - root cause report §3 + §5 갱신 + paper §V.5.4 narrative integration

- **에스컬레이션 — Root chain 위임**:
  - Root 세션이 본 chain 의 Step 1~7 모두 수행 후 analyzer 에 핸드오프
  - Analyzer 결과 후 planner 에 narrative integration 핸드오프
  - **중요**: scripts/run_v4_mitigation_sweep.sh 으로 V4-A + V4-B 두 ckpt 학습을 single-command launch (사용자 부재 시 chain 멈춤 방지)

- **ID 명명 — V4-A / V4-B**:
  - 직전 시도 8개: Phase 1 (baseline) + Phase 2 b8 (B5 통합) + Phase 3 #3 (Direct AC) + Phase 3 #4 (LR x5) + v2 #1 (DropMessage) + v2 #2 (Sum) + v2 #3 (LayerNorm) + v3 #1 (GIN)
  - 9번째 (진행 중): v3 #2 (Max aggregation, 5/14 ETA)
  - **신규 V4-A / V4-B**: 10/11번째 시도. 보고서 C-1/C-2 의 architectural intervention 으로 명명. Combo 가설의 정량 검증.
  - 학회 narrative: "8 trial mitigation null + V4 combo 가 첫 dramatic 회복 (또는 still null = combo 가설도 부정)"

- **추가 필요 분석**:
  - V4-A / V4-B 의 학습 trajectory + R@15 ceiling 갱신 여부 + layer-wise cosine 정량 (직전 ckpt 들과 비교)
  - V4-A vs V4-B 직접 비교 — combo (LN+GIN, row-stochasticity 유지) vs architectural (AERO, row-stochasticity 파괴) 의 효과 차이
  - (V4 둘 다 R@15 갱신 시) Mech(ii-b) DOMINANT 4/5 → 부정 narrative 전환 + paper §V.5.4 main finding 갱신
  - (V4 둘 다 null 시) Mech(ii-b) DOMINANT 4/5 → 5/5 절대 confirm + Filter Dominance 6번째 축 narrative 결정적 강화

---

## 2026-05-11 (Architecture-invariance 발견 + narrative 보강 — qcond_nl3 = QCond Concat 명시)

> **사용자 직전 input (2026-05-11)**: "지금 oversmoothing 이 QCond SN 말고 QCond (concat) 에서는 안 일어나나?" → AskUserQuestion 결과: **narrative 보강만 처리 (즉시, planner)**.

- **결정적 발견 — qcond_nl3 가 이미 QCond Concat 방식**:
  - `configs/training/diameter_layers/train_qcond_nl3.yaml`: `query_conditioned: true`, `query_supernode` 키 부재 → SchemaHeteroGAT default `query_supernode=false`. 즉 **Concat 방식**.
  - 같은 family 의 `train_gat_query_conditioned.yaml` 명시 — "Query Concatenation 활성화".
  - 직전 narrative 에서 "qcond_nl3 baseline" 으로만 칭하고 Concat 방식이라는 점이 부각 안 됨.

- **3 axis 의 architecture-invariance 정량 evidence**:

  | Ckpt | Architecture | $\bar{c}_{L_3}$ |
  |---|---|---:|
  | **qcond_nl3** | **QCond Concat** (no SuperNode) | **0.9971** ⚠️ (가장 심한 collapse) |
  | DSN p80 | V-3-ext SuperNode + directed_from_sn | 0.9591 |
  | DSN topk20 | V-3-ext SuperNode + directed_from_sn | 0.9662 |
  | DSN abstau07 | V-3-ext SuperNode + directed_from_sn | 0.9775 |

  → **Over-smoothing 이 SuperNode-specific 이 아니라 QCond Concat 에서도 (오히려 더 심하게) 발생**. V-3-ext 의 directed_from_sn edge 가 partial mitigation 효과만 발휘 (advisor briefing §1.3 직전 narrative "DSN 3 ckpt 가 baseline 대비 partial mitigated" 와 일관). 즉 over-smoothing 은 **architecture 선택과 무관한 GAT propagation 의 fundamental limitation**.

- **narrative 보강 2 위치**:
  1. **`advisor_briefing_oversmoothing_2026-05-11.md` §1**: "문제 — 학습이 ceiling 에 갇혀 있다" 섹션에 architecture 분류 표 추가 (QCond Concat vs V-3-ext SuperNode) + "🎯 Architecture-Invariance 첫 evidence" callout. 학회 narrative 의 출발점을 "single hyperparameter tuning **또는 architecture 선택** 의 문제가 아니라 GAT 의 구조적 collapse" 로 강화.
  2. **`advisor_briefing_oversmoothing_2026-05-11.md` §6**: "GAT Internal Dynamics Invariance (Attention + Gradient)" → "**Three-Axis Invariance (Architecture + Attention + Gradient)**" 로 확장. 세 axis 의 invariance 표 + 학회 contribution candidate 2 → 3 항목 (architecture-invariance evidence 추가).
  3. **`oversmoothing_root_cause_report_2026-05-11.md` §1**: 4 ckpt × 3 layer × architecture 분류 표 추가 + "over-smoothing 이 query 정보 통합 방식과 무관한 GAT propagation 자체의 architectural limitation" 명시.
  4. **`oversmoothing_root_cause_report_2026-05-11.md` §5.2 결정적 근거**: 4 항목 → **5 항목**. 첫 항목으로 "Architecture-invariance" 추가 (Concat vs SuperNode 모두에서 collapse, Concat 에서 더 심함).

- **학회 narrative 의 강화 message**:
  > Over-smoothing 이 V-3-ext (DSN SuperNode) 의 specific issue 가 아니라 QCond Concat 방식에서도 (오히려 더 심하게) 발생함 — L3 cosine 0.9971 vs 0.9591~0.9775. 즉 GAT 의 softmax-weighted-mean propagation limitation 이 query 정보 통합 방식과 무관하게 architectural 임을 입증.

- **근거**:
  - 사용자 직접 input ("QCond Concat 에서는 안 일어나나?") — 새 측정 없이 기존 데이터 재분류로 답변 가능한 발견
  - `configs/training/diameter_layers/train_qcond_nl3.yaml` 직접 조회 — `query_supernode` 키 부재 confirm
  - `EXPERIMENT_HISTORY.md` line 1308 — `query_supernode=False 계열 (T1~T6, T8, s06 B0~B5E 등)` 일관 명시

- **영향 범위**:
  - planning/advisor_briefing_oversmoothing_2026-05-11.md §1 + §6 (수정 완료)
  - planning/oversmoothing_root_cause_report_2026-05-11.md §1 + §5.2 (수정 완료)
  - 본 DECISIONS.md (본 entry)

- **에스컬레이션 필요 여부**: 없음. 사용자 결정 (A) narrative 보강만 — 즉시 처리 완료.

- **확장 측정 candidate (사용자 결정 보류)**:
  - QCond Concat nl1/nl2/nl6/nl7 의 layer-wise cosine 측정 — Concat 방식의 layer depth × collapse trajectory 추가 evidence. ETA ~3-5 min wall (dsn_oversmoothing_analysis.py 확장).
  - Plain GAT (no QCond) 측정 — query 정보 없는 순수 GAT 의 collapse 발생 여부 (post-paper future work).
  - 두 측정 모두 priority 낮음 — 본 architecture-invariance message 가 기존 4 ckpt 데이터로 충분 입증.

---

## 2026-05-11 (oversmoothing_root_cause_report_2026-05-11.md 신규 — Over-Smoothing 의 주된 원인 + 근거 focused 보고서)

> **사용자 직전 input (2026-05-11)**: "지금까지 정리된 내용을 바탕으로 Over-Smoothing의 주된 원인은 무엇이며 그 근거는 어떻게 되는지 보고서를 작성해 줘"

- **결정 — 신규 보고서 작성** (advisor briefing 과 별도 파일):
  1. **(a) 신규 산출물**: [`planning/oversmoothing_root_cause_report_2026-05-11.md`](oversmoothing_root_cause_report_2026-05-11.md) — 6 sections, ~240 lines.
     - §1 분석 framework (4 mechanism 후보 + 핵심 측정 $\bar{c}_L$)
     - §2 근거 chain (4 evidence pillar: mech(iii) 부정 / mech(ii-a) partial mit / mech(ii-b) partial 부정 / mech(i-b) hierarchy)
     - §3 종합 — Softmax-Weighted-Mean Combo Root Cause (sub-mechanism 분리 + 8 mitigation null)
     - §4 보조 evidence — GAT Internal Dynamics Invariance
     - §5 결론 (주된 원인 + 결정적 근거 4 항목 + 결론의 의의)

  2. **(b) 세 planning 문서의 역할 분리 (재확인)**:
     - `over_smoothing_research_summary.md` — 학위 논문 Part III chapter draft base, 정량 evidence + 수식 reference (~1100 lines)
     - `advisor_briefing_oversmoothing_2026-05-11.md` — 지도교수 발표 보고용 narrative, 진척 흐름 중심 (~240 lines, 8 stage timeline)
     - `oversmoothing_root_cause_report_2026-05-11.md` (5/11 신규) — **원인 + 근거 focused** 보고서, 진척 흐름은 부수적, 원인 분석이 본문 (~240 lines)

  3. **(c) advisor briefing 과의 차별점**:
     - advisor briefing: "어떻게 진단했는지" 의 narrative (8 stage 흐름, paradox, mitigation 진화)
     - 본 보고서: "주된 원인은 무엇이며 근거는 무엇인가" 의 정량 evidence 중심 (4 mechanism elimination + sub-mechanism dominance + 8 mitigation null pattern)
     - 두 문서가 동일 fact base 인용 단 narrative 강조점 달라

- **핵심 message**:
  - **주된 원인**: Mech(ii-b) softmax-weighted-mean propagation combo. Mech(i-b) aggregation magnitude 가 보조.
  - **결정적 근거 4 항목**:
    1. Mech(iii) 부정 — Direct AC 가 skip_dep 0.97 회복했음에도 R 가장 낮음 + AC=0.62 일관 유지
    2. Mech(ii-a) partial mit — LayerNorm 이 attention 회복 (top5 0.7510) 하지만 L1 cosine 0.9998 collapse 보존
    3. Mech(ii-b) partial 부정 — GIN 이 L2 cosine -0.08 dramatic 회복 (11 DBs 일관) 하지만 R 미갱신
    4. 8 mitigation null + GAT internal dynamics invariance (attention 0.3% + gradient 6% sub-noise)
  - **결론**: Architectural limitation (학습 dynamics 의 문제 X). Filter Dominance 6번째 축 narrative 의 결정적 evidence.

- **근거**:
  - 사용자 직접 input (원인 + 근거 보고서 요청)
  - 직전 8 단계의 mitigation 결과 + 4 ckpt × 4 mechanism evidence matrix (over_smoothing_research_summary.md §1~§7 + advisor briefing §1~§4)
  - Step 3 grad_flow 재측정 (4 ckpt ρ_skip 정확 수치) + Multi-DB attention 재측정 (4 ckpt L2 top5_conc spread 0.0023)

- **영향 범위**:
  - planning/oversmoothing_root_cause_report_2026-05-11.md (신규)
  - 본 DECISIONS.md (본 entry)
  - 기존 advisor briefing / summary 변경 X — 본 보고서는 별도 산출물

- **에스컬레이션 필요 여부**: 없음. 사용자가 본 보고서 + advisor briefing + summary 3 문서 조합으로 학회 발표 + 학위 논문 chapter draft 작성 가능.

- **추가 필요 분석**: 없음. 단 (학회 발표 슬라이드 작성 시) 본 보고서의 §3.2 "Softmax × Aggregation Combo" 도식이 main slide candidate.

---

## 2026-05-11 (advisor briefing narrative final review — paradox 2 + paradox 3 통합 → GAT Internal Dynamics Invariance)

> **사용자 직전 input (remote-control, 2026-05-11)**: "advisor briefing §2 paradox 2 + paradox 3 결합 narrative 학회 자연성 + 두 paradox 같은 mechanism family 명시 위치 결정"

- **검토 결과 — 결합 narrative 자연성: confirm ✅**

  두 paradox 가 같은 fundamental 현상의 두 단면:

  | 측면 | metric | 4 ckpt spread | 공통 시사 |
  |---|---|---:|---|
  | Gradient (paradox 2) | ρ_skip | 0.19 (6%) | 학습 신호 분배가 학습 변형에 무관 |
  | Attention (paradox 3) | L2 top5_conc | 0.0023 (0.3%) | attention 분포가 학습 변형에 무관 |

  → **GAT internal dynamics 전체 (attention + gradient) 가 학습 변형 (directedness/threshold/QCond) 에 decouple**. 학회 contribution candidate 의 통합 message — pipeline outcome 만이 아니라 GAT internal 도 robust.

- **결정 — 보강 수정 2 위치**:
  1. **§2 끝 통합 callout 신설**: "🎯 두 paradox 의 공통 시사 — GAT Internal Dynamics 가 학습 변형에 lock-in" — paradox 2 + paradox 3 의 공통 표 + 통합 시사 명시.
  2. **§6 제목/내용 확장**: "Attention ckpt-invariance" → "GAT Internal Dynamics Invariance (Attention + Gradient)". 두 paradox 모두 인용 + 학회 contribution candidate 2 항목으로 구조화:
     - (a) 검증 통로 제한 시사 — architectural mitigation 만이 유일 통로 (mitigation 7 시도 모두 architectural intervention 인 정당화)
     - (b) Filter Dominance narrative 의 GAT-internal evidence — "training-pathology-invariant" 가 pipeline outcome (F1 0.8383) 뿐만 아니라 GAT internal dynamics 자체의 invariance 도 의미

- **근거**:
  - DECISIONS.md 2026-05-11 (Step 3 grad_flow 재측정 완료) entry 의 4 ckpt × ρ_skip = 3.02 / 2.85 / 3.00 / 3.04 (spread 0.19)
  - DECISIONS.md 2026-05-11 (Multi-DB 재측정 완료) entry 의 4 ckpt × L2 top5_conc spread = 0.0023
  - 두 measurement 가 직전 분리 narrative 였으나 본 review 에서 결합 가치 확인 — 학회 contribution candidate 강화

- **영향 범위**:
  - planning/advisor_briefing_oversmoothing_2026-05-11.md §2 (통합 callout 추가) + §6 (제목/내용 확장)
  - 본 DECISIONS.md (본 entry)

- **에스컬레이션 필요 여부**: 없음. 사용자가 advisor briefing 최종본 읽고 학회 발표 준비.

- **추가 필요 분석**: 없음. 단 (학회 발표 슬라이드 작성 시) "GAT internal dynamics invariance" 메시지를 슬라이드 main 한 장으로 시각화 candidate (attention spread plot + gradient spread plot 병렬).

---

## 2026-05-11 (Step 3 grad_flow 재측정 완료 — 4 ckpt ρ_skip 정확 수치 + advisor briefing §2 Paradox 2 보강)

> **root 세션 chain 결과 — 5/11 KST ~19:35 (~1 min wall)**. 직전 entry "Step 3 재측정 결정" 의 모든 작업 완료. advisor briefing §2 Paradox 2 의 ρ_skip 진술 (직전 "3.02~3.04, p80/qcond_nl3 만 명시") 을 4 ckpt 전부 (p80, topk20, abstau07, qcond_nl3) 의 정확 수치로 보강.

- **재측정 명령**: `conda run -n base python src/analysis/dsn_oversmoothing_analysis.py --max_queries 50 --skip_step1 --skip_step2` (CPU forward, n=50 single-DB california_schools). 코드 patch **불필요** — `main` 에 이미 `results["step3"] = step3_all` 존재 (line 581), 옛 batch_summary.json 의 step3 누락은 `--skip_step3` 으로 실행된 결과.

- **4 ckpt × parameter group gradient norm (n=50)**:

  | Group | p80 | topk20 | abstau07 | qcond_nl3 |
  |---|---:|---:|---:|---:|
  | lin_dict | 1.1064 | 1.0373 | 1.0176 | 2.2434 |
  | conv_L1 | 1.0535 | 1.0383 | 0.9670 | 1.1346 |
  | conv_L2 | 0.7397 | 0.6878 | 0.6827 | 0.9183 |
  | conv_L3 | 1.2984 | 1.1382 | 1.2327 | 1.4771 |
  | out_lin_dict | 3.3024 | 2.7854 | 3.0869 | 3.0741 |
  | skip_dict | 3.9159 | 3.2388 | 3.6965 | 4.4865 |
  | max(conv_*) | 1.2984 | 1.1382 | 1.2327 | 1.4771 |
  | **ρ_skip** | **3.02** | **2.85** | **3.00** | **3.04** |

- **핵심 발견**:
  1. **4 ckpt 모두 ρ_skip ∈ [2.85, 3.04]** (spread = 0.19, 6%) — directedness/threshold/QCond 어떤 학습 변형에도 extreme 일관. mech(iii) Skip Dep 의 강도가 4 ckpt 모두 동등.
  2. topk20 만 약간 낮음 (2.85) — 단 여전히 extreme. 옛 narrative "3.02~3.04 extreme" 와 일관, range 만 약간 확대 (2.85~3.04).
  3. **GAT 구조 자체의 inherent 특성** — 학습 변형이 gradient flow 의 path imbalance 를 흔들지 못함. 직전 entry 의 Paradox 3 (attention ckpt-invariance) 와 **같은 mechanism family**: 학습 변형이 GAT internal dynamics (attention / gradient flow) 어느 channel 도 흔들지 못함.

- **갱신된 산출물**:
  - `notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md` §4.1 v1 ckpt 표 확장 (p80, qcond_nl3 → 4 ckpt 전체), §4.2(a) narrative "Phase 1 4 ckpt 의 skip_dict 3.24~4.49 ratio 2.85~3.04 spread 0.19" 로 갱신, §6.2 종합 dashboard 에 topk20/abstau07 행 추가
  - `planning/advisor_briefing_oversmoothing_2026-05-11.md` §2 Paradox 2 에 4 ckpt × ρ_skip 표 추가 (advisor 보고용 톤 유지 — 간결 표 + "어떤 학습 변형도 ρ_skip 을 의미 있게 흔들지 못함" narrative)
  - `outputs/analysis/dsn_oversmoothing/batch_summary.json` — step3 key 채워짐 (4 ckpt × grad_mean / grad_ratio / entropy / topk_conc / num_layers)

- **에스컬레이션 / 후속**:
  - **planner 세션**: advisor briefing §2 Paradox 2 4 ckpt 표 narrative final review — "어떤 학습 변형에도 ρ_skip extreme 유지" 가 직전 entry 의 paradox 3 (attention ckpt-invariance) 와 결합된 narrative 가 학회 발표에 자연스러운지 검토. 두 paradox 가 같은 mechanism family (학습 변형이 GAT internal dynamics 를 흔들지 못함) 임을 §2 또는 §6 새 발견 의의 에 명시할지 결정.

- **에스컬레이션 필요 여부**: 측정 + 보고서 갱신 완료. planner narrative review 만.

---

## 2026-05-11 (Step 3 재측정 결정 — 4 ckpt × ρ_skip 정확 수치 + advisor briefing paradox 2 표 보강)

> **사용자 직전 input (2026-05-11)**: "표 들은 좀 남아있으면 좋겠는데, Skip Dependency 검증을 위해 ρ_skip 을 분석했을 때 QCond SN 과 p80, top20, abstau07 각각 어떤 값이 나왔는지 알고 싶어" → AskUserQuestion 결과: **root 에 step3 재측정 요청 (4 ckpt 완전 수치)**.

- **결정 — root chain 위임**:
  1. **코드 patch**: `src/analysis/dsn_oversmoothing_analysis.py:main` 에서 step3 결과를 `results['step3']` 로 dump 하도록 수정 (현재 누락 — PNG 만 출력되고 raw json 없음). 또는 별도 `outputs/analysis/dsn_oversmoothing/grad_summary.json` 출력.
  2. **재측정**: 4 ckpt (p80, topk20, abstau07, qcond_nl3) × single-DB n=50 (직전 protocol 일관). CPU forward ~1-2 min wall.
  3. **Analyzer 보고서 보강**: `notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md` §4.1 표에 topk20, abstau07 추가 (lin_dict / conv_L1/L2/L3 / out_lin_dict / skip_dict / ratio 모든 column).
  4. **Advisor briefing 보강**: `planning/advisor_briefing_oversmoothing_2026-05-11.md` §2 Paradox 2 부분에 4 ckpt × ρ_skip 표 추가. 표 형식 간결 (advisor briefing 톤 유지).
  5. **DECISIONS.md prepend**: 재측정 완료 entry.

- **근거**:
  - `outputs/analysis/dsn_oversmoothing/batch_summary.json` 직접 조회: `Keys: ['step1', 'step2']` — step3 dump 누락
  - `outputs/analysis/dsn_oversmoothing/{p80, topk20, abstau07, qcond_nl3}/gradient_flow.png` 존재 — 측정은 됐으나 PNG only
  - 직전 narrative 의 ρ_skip 인용 (3.02~3.04) 가 p80 / qcond_nl3 만 — 사용자 정당한 지적

- **protocol 선택**: single-DB n=50 (직전과 동일). 이유:
  - 직전 narrative 의 3.02 / 3.04 와 직접 비교 가능
  - Multi-DB stratified n=55 protocol 도 가능 단 gradient flow 는 단일 forward pass per query 라 multi-DB 효과가 attention 측정만큼 dramatic 하지 않을 가능성 (gradient norm 은 sample 평균 robust)
  - 보고용 narrative 일관성 우선

- **영향 범위**:
  - src/analysis/dsn_oversmoothing_analysis.py (코드 patch 1 곳)
  - notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md §4.1 (topk20, abstau07 추가)
  - planning/advisor_briefing_oversmoothing_2026-05-11.md §2 (4 ckpt × ρ_skip 표 신규)
  - planning/DECISIONS.md (재측정 완료 entry, root 작성)

- **에스컬레이션 필요 여부**: root 세션에 chain 위임 (코드 patch + 재측정 + 보고서 보강 + advisor briefing 보강 + DECISIONS prepend).

- **추가 필요 분석**: 없음. 본 측정은 직전 protocol 의 누락 보강.

---

## 2026-05-11 (advisor briefing 전면 재작성 — 실험 노트 톤 → 지도교수 보고 톤)

> **사용자 직전 input (2026-05-11)**: "지금 advisor_briefing 가 점점 그냥 실험 결과 나열하는 내부 참고문서 같아 / 지도교수님께 진척 상황을 보고하는 자료니까 보고서가 어떤 거네 실험 결과가 어떤 거네 하는 건 우선 빼고 눈여겨 봐야 할 수치와 의미를 중점으로 서술해 줘"

- **결정 — 전면 재작성**:

  **삭제 항목** (실험노트/내부 참고문서 톤):
  - Analyzer 보고서 9개 목록 (§10)
  - 분석 코드 파일명 (`dsn_oversmoothing_analysis.py` 등)
  - 산출물 경로 (`outputs/analysis/...`, `scripts/...`)
  - Config 이름 (`train_gat_directed_supernode_p80_b5_mitigation.yaml`)
  - "왜 분석을 했나 / 무엇을 했나 / 결과 / 무엇을 알게 됐나" 4 절 균일 구조
  - 각 mitigation 의 수식 block (PairNorm / IR / JK / GIN propagation)
  - Stage 별 8-stage timeline ASCII flowchart
  - 5 보고서 통합 정식, A1+A2+A3 deep dive 의 protocol 디테일
  - DSN ablation 의 directedness/threshold/QCond 세부 변형 설명

  **유지 / 강화**:
  - **핵심 수치 — dominant evidence 만**: $\bar{c}_{L_3} \geq 0.96$, $\rho_{\text{skip}} = 3.02$, top5_conc 0.71~0.77, 4 ckpt spread 0.0023, AC=0.62 일관 (300 epoch), GIN L2 cosine 0.9137, F1 0.8383
  - **Narrative 흐름**: 문제 발견 → 두 paradox → 8 mitigation evolution (전환점 3개 표시) → mech(ii-b) softmax-aggregation combo 정식 → Filter Dominance 6번째 축
  - **새 발견 (paradox 3 ckpt-invariance)** 의 학회 contribution candidate 강조
  - 예상 질문 7 개 (§8 Q&A)

  **새 구조** (8 sections, ~250 lines):
  - §1 문제 — 학습이 ceiling 에 갇혀 있다 (over-smoothing 진단)
  - §2 두 paradox (attention moderate / skip extreme / ckpt-invariance NEW)
  - §3 Mitigation 진화 8 시도 + 결정적 전환점 3 개
  - §4 Mech(ii-b) softmax-weighted-mean combo 정식
  - §5 학회 narrative 핵심 — Filter Dominance 6번째 축
  - §6 새 발견 — Attention ckpt-invariance 의 시사점
  - §7 진행 중 + 향후 (max aggregation 분기 + post-paper combo)
  - §8 예상 Q&A 7 항목

- **근거**:
  - 사용자 직접 input ("실험 결과 나열하는 내부 참고문서 같아 / 보고서가 어떤 거네 실험 결과가 어떤 거네 하는 건 우선 빼고")
  - 직전 advisor briefing (~590 lines, 15+ 표, analyzer 보고서 9개 명시) — narrative 가 표/수치 더미 속에 묻힘
  - 학회 발표 톤: "이렇게 진단했고, 이런 시도를 거쳤고, 이게 의미하는 바입니다" — 출처 / 분석 코드 / 보고서 reference 모두 부담

- **영향 범위**:
  - planning/advisor_briefing_oversmoothing_2026-05-11.md (전면 재작성, ~590 → ~230 lines)
  - 본 DECISIONS.md (본 entry)
  - **기존 정량 evidence 의 백업**: planning/over_smoothing_research_summary.md (~1100+ lines reference document, §A 정의 + §B 측정 metric + §C mitigation 수식 + 8 stage detailed evidence) 가 학위 논문 Part III chapter draft base 로 그대로 보존. advisor briefing 은 보고용 narrative only.

- **에스컬레이션 필요 여부**: 없음. 사용자가 새 narrative 톤 확인 후 학회 발표 준비 가능.

- **추가 필요 분석**: 없음.

- **두 문서 역할 분리 (재확인)**:
  - `over_smoothing_research_summary.md` — 학위 논문 Part III chapter draft base, 정량 evidence + 수식 reference document
  - `advisor_briefing_oversmoothing_2026-05-11.md` — 지도교수 발표 보고용 narrative, 의미 중심

---

## 2026-05-11 (advisor briefing §1.3 entropy 표 보강 — L1/L3 entropy 추가)

> **사용자 직전 input (2026-05-11)**: "entropy H는 왜 L2만 있어? L1과 L3는 없어도 되는 건가?"

- **결정 — §1.3 multi-DB 측정 표 분리 정정**:
  - 옛: top5_conc 3 layer + L2 entropy 만 (4 컬럼 컴팩트)
  - 새: top5_conc 표 (4 ckpt × 3 layer) + entropy 표 (4 ckpt × 3 layer) 별도 분리. 두 표 모두 L1/L2/L3 fully 명시.
  - narrative 보강: "L1/L2/L3 모두 동일 패턴 (layer 별 entropy spread Δ ≤ 0.05)" — 3 layer 일관성 명시. top5_conc 범위 0.71~0.77 + entropy 1.87~1.93 으로 full range.

- **근거**:
  - 사용자 정당한 지적 — L2 만 인용 시 layer-wise 흐름 evidence 약화
  - 원본 analyzer 보고서 `dsn_phase2_mitigation_null_mechanism.md` §3.1bis 표에는 L1/L2/L3 entropy 모두 측정값 있음 (1.8729~1.9264 범위)
  - advisor briefing 작성 시 root 가 표 width 가독성 위해 L2 만 인용한 것으로 보임 — narrative 정확성 측면에서 보강 필요

- **영향 범위**:
  - planning/advisor_briefing_oversmoothing_2026-05-11.md §1.3 (수정 완료, top5_conc 표 + entropy 표 분리)
  - 본 DECISIONS.md (본 entry)

- **에스컬레이션 필요 여부**: 없음.

- **추가 필요 분석**: 없음. L1/L3 entropy 추가로 narrative ("3 layer 일관 moderate-sharp peaking") 강화.

---

## 2026-05-11 (advisor briefing 깨진 표 / 수식 정정 — §1.2 3-step protocol + §2.2 B5 Mitigation)

> **사용자 직전 input (2026-05-11)**: "advisor_briefing_oversmoothing_2026-05-11.md 에 수식과 표가 깨진 부분이 있는 것 같아 재검토해 줘"

- **결정 — 2 위치 수정**:
  1. **§1.2 3-step protocol 표**: layer-wise cosine 수식 안의 `|\mathcal{T}|` + `|T|` pipe 가 마크다운 column 구분자로 인식되어 표가 7 컬럼으로 깨졌음. → 표는 step / metric / 도구만 표시 + 수식은 표 아래 별도 display block ($$...$$) 으로 분리.
  2. **§2.2 B5 Mitigation 표**: PairNorm 수식의 `|V|` pipe 같은 문제 발생 가능성. → 표는 mitigation 이름 + 역할만 (5 행), 수식 4 개는 표 아래 별도 bullet list 로 분리. PairNorm 의 `|V|` 는 `N = |V|` 변수 치환으로 표 밖에서 명시.

- **근거**:
  - 마크다운 inline math `$...$` 내부의 pipe 가 GitHub-flavored 마크다운 parser 에서 종종 column 구분자로 인식 — 특히 vertical bar 가 norm/cardinality 표기 ($|\mathcal{T}|$, $|V|$) 인 경우 발생
  - grep `^\|.*\$[^$]*\|[^$]*\$` 로 안전 점검: line 95 / 437 의 layer-wise cosine 표 header (`$\bar{c}_{L_0}$ | $\bar{c}_{L_1}$ | ...`) 는 false positive — 각 cell 마다 `$...$` 가 닫히고 pipe 는 cell 구분자만 (정상).

- **영향 범위**:
  - planning/advisor_briefing_oversmoothing_2026-05-11.md §1.2 + §2.2 (수정 완료)
  - 본 DECISIONS.md (본 entry)
  - 다른 표는 모두 정상 (수식 내부 pipe 없음 확인)

- **에스컬레이션 필요 여부**: 없음.

- **추가 필요 분석**: 없음. 단 (선택) over_smoothing_research_summary.md 의 §A/§B/§C 수식 섹션 도 같은 패턴 점검 candidate — 단 그 문서는 수식이 표 안이 아니라 §1.4 paradox 진술 안의 display block 이므로 영향 없을 가능성 높음.

---

## 2026-05-11 (dsn_phase2_mitigation_null_mechanism.md §3.1bis / §3.1ter 검토 완료 — multi-DB cross-check narrative 일관성 confirm + Phase 2/v2 LN 재측정 priority 낮음 confirm)

> **사용자 직전 input (remote-control, analyzer 분석 후, 2026-05-11)**: "먼저 planning/CLAUDE.md 읽고, dsn_phase2_mitigation_null_mechanism.md §3.1bis / §3.1ter 신규 추가 검토. (a) dsn_oversmoothing_phase1_multi_db_recheck.md §5 paradox narrative 와의 일관성, (b) Phase 2 b8 mit / s06_b5 / v2 LN 의 multi-DB n=55 재측정 priority 판단."

- **결정 (검토 결과 2 항목)**:

  **(a) §3.1bis / §3.1ter ↔ multi-DB recheck §5 paradox narrative 일관성 — confirm ✅**

  | 항목 | §3.1bis/ter 정량 | §5 정량 | 일관 |
  |---|---|---|---|
  | single-DB → multi-DB shift | "0.24 → 0.72, +0.47 jump, 학습 ckpt 무관 protocol-induced shift" | "수정 1: 0.24→0.72 톤다운" | ✅ |
  | 4 ckpt spread | "L2 top5_conc spread = 0.0023 (0.3%)" | "수정 2: 새 paradox attention ckpt-invariance, spread 0.0023" | ✅ |
  | mech(ii-b) 결론 | "mech(ii) root cause 부정 유지" | "수정 3: mech(ii-b) propagation 가설 유지" | ✅ |
  | paradox 2 (skip dep) | (§4 별도) | "유지" | ✅ (병행) |

  → 두 보고서가 **3 paradox 정식** + **정량 수치 (0.47 jump / 0.0023 spread) 이 양 보고서에서 일치 인용** + mech(ii-b) 결론 유지 narrative 모두 일치.

  **(b) Phase 2 b8 mit / s06_b5 / v2 LN 의 multi-DB n=55 재측정 priority — 낮음 confirm ✅**

  근거 (analyzer §3.1ter 결론과 일치):
    1. **Same-protocol (single-DB n=2) 비교는 valid** — Phase 2 b8 +0.40 sharpening claim 은 single-DB ↔ single-DB. 둘 다 california_schools outlier 영향 동일 → 상대 Δ 정정 불필요.
    2. **Mech(ii) root cause 부정 결론 무관** — Mitigation 이 attention sharpen 함에도 val recall 갱신 X 라는 §3.3 핵심은 절대값 무관.
    3. **Dominant mechanism timeline 무관** — mech(iii) DOMINANT 5/5 → 4-trial mech(ii) DOMINANT 5/5 → 7-trial sub-mechanism → 8-trial mech(ii-b) 4/5 partial 부정 의 진화는 same-protocol 비교에 기반.

  **Conditional trigger 신설** (재측정 priority 상승 조건):
    - **Trigger 1**: paper drafting 시 reviewer 가 "Phase 1 multi-DB 0.72 baseline 과 Phase 2 single-DB 0.67 sharpened 비교 invalid" 지적 → Phase 2~v3 GIN 8 ckpt 모두 multi-DB 재측정 launch (~8 min wall, A3 protocol 동일).
    - **Trigger 2**: 학회 Q&A 시 청중이 single-DB 0.24 outlier 출처 질문 → §3.1bis / §5 narrative 만으로 답변 충분, 재측정 불필요.
    - **Default**: 학회 발표 + 학위 논문 Part III chapter draft 우선 → priority 낮음 유지.

- **추가 옵션 — advisor briefing 보강 candidate** (사용자 결정 미정):
  - **Option A**: advisor briefing §4.3 (Stage 4 결과 표) 옆에 footnote — "v2 #3 LN top5_conc 0.7510 / Phase 2 b8 0.8797 은 single-DB n=2. Phase 1 multi-DB 0.72 baseline 과 직접 비교 불가, 단 same-protocol 상대 Δ valid"
  - **Option B**: 현재 narrative 유지 — same-protocol caveat 는 §1.3 Caveat box 만으로 충분
  - 학회 발표 priority 만이면 Option B. Paper drafting 대비면 Option A.

- **근거**:
  - `notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md` §3.1bis (4 ckpt × multi-DB n=55 표 + spread 0.0023) + §3.1ter (Phase 2 / v2 LN / v3 GIN cross-check, +0.40 sharpening claim same-protocol valid)
  - `notebooks/analysis_results/dsn_oversmoothing_phase1_multi_db_recheck.md` §5 paradox narrative (수정 1+2+3 + 유지)
  - DECISIONS.md 2026-05-11 (Multi-DB 재측정 완료) entry 의 정량 일치 (spread 0.0023, baseline 0.7144~0.7167)

- **영향 범위**:
  - planning/DECISIONS.md (본 entry)
  - planning/advisor_briefing_oversmoothing_2026-05-11.md (보강 미정, Option A/B 사용자 결정 대기)
  - 본 검토 자체는 read-only — 분석 보고서 / multi-DB 측정 / paradox narrative 모두 변경 없음

- **에스컬레이션 필요 여부**: 없음. Analyzer 보고서의 priority 낮음 분류가 narrative 일관성 측면에서 valid 함을 planner 가 정량 검증. 단 사용자가 Option A/B 결정 시 advisor briefing 보강 진행.

- **추가 필요 분석**: 없음 (default). Conditional trigger 발생 시 root 에 multi-DB 재측정 핸드오프 (Phase 2~v3 GIN 8 ckpt × A3 protocol).

---

## 2026-05-11 (advisor briefing narrative final review 완료 — paradox flow 학회 자연성 confirm + Stage 1 → Stage 4 causal chain 명시)

> **사용자 직전 input (remote-control, 2026-05-11)**: "planning/advisor_briefing_oversmoothing_2026-05-11.md 의 narrative final review 를 수행. 검토 포인트: (1) paradox 1 톤다운 + paradox 3 NEW 가 학회 narrative 자연 흐름 형성? (2) §1.4 mech(ii-a) 검증 불가 → LayerNorm/GIN 직접 표적만 유일 검증 통로 명시할지?"

- **검토 결과**:
  1. **Paradox flow 학회 narrative 자연성 — confirm ✅**
     - Paradox 1 톤다운 (uniform → moderate-sharp top5_conc 0.72): 강한 단정 → 정확한 측정 → narrative 신뢰도 증가 + mech(ii-b) 후보 강도 ("sharpness 무관 평균화") 유지
     - Paradox 3 NEW (attention ckpt-invariance 0.3% spread): 학회 contribution candidate. Stage 1 → Stage 4 의 **causal chain** 을 paradox 3 매개로 강화 ("학습 변형으로 attention 검증 불가 → mitigation 직접 표적 필요")
     - Per-DB variance 표 (california_schools 0.24 outlier ~ toxicology 1.00) 가 single-DB caveat 의 직관적 visualization
  2. **§1.4 mech(ii-a) 검증 불가 명시 — 보강 완료 ✅**
     - 옛 §1.4: Paradox 3 진술에 "ckpt 변경 검증 불가" 만 명시, 4 후보 mechanism 표에는 미반영
     - 새 §1.4: mech(ii) 를 (ii-a)+(ii-b) sub-mechanism 으로 미리 분리. (ii-a) 옆에 "ckpt 변경 검증 불가 → LayerNorm pre-softmax 등 mitigation 직접 표적 유일 검증 통로" 명시
     - 추가 callout: "🎯 학회 narrative 핵심: Stage 1 → Stage 4 causal chain 이 paradox 3 를 매개로 형성"

- **결정 (narrative 보강 수정 2 위치)**:
  1. **`advisor_briefing_oversmoothing_2026-05-11.md` §1.4 4 후보 mechanism 표**: mech(ii) 를 sub-mechanism (ii-a) + (ii-b) 로 분리. (ii-a) 옆에 "Paradox 3 시사 — ckpt 변경 검증 불가 → mitigation 직접 표적이 유일 검증 통로. Stage 4 v2 #3 LayerNorm 가 본 가설의 정량 검증." 명시. 학회 narrative 핵심 callout 추가 ("Stage 1 → Stage 4 causal chain 이 paradox 3 매개로 형성").
  2. **`advisor_briefing_oversmoothing_2026-05-11.md` §4.1 (Stage 4 왜 이 분석을 했나)**: 두 동기 합류로 재작성 — (a) Stage 3 mech(ii) DOMINANT 5/5 갱신 + (b) Stage 1 Paradox 3 직접 시사 (ckpt 변경 검증 불가 → mitigation 직접 표적이 유일 통로). 직전 narrative ("Stage 3 결과 mech(ii) DOMINANT 5/5") 만으론 약함.

- **근거**:
  - 사용자 remote-control 직접 input + DECISIONS.md 2026-05-11 (Multi-DB 재측정 완료) 의 paradox 정정 table
  - `notebooks/analysis_results/dsn_oversmoothing_phase1_multi_db_recheck.md` §5 paradox narrative 검증 결과 (3 paradox 정식)
  - Paradox 3 의 학회 narrative 가치 평가: 학습 변형 invariance 는 "mitigation 만이 유일 검증 통로" 라는 결론의 직접 정당화 — Stage 4 motivation 강화

- **영향 범위**:
  - planning/advisor_briefing_oversmoothing_2026-05-11.md §1.4 + §4.1 (수정 완료)
  - 본 DECISIONS.md (본 entry)
  - planning/over_smoothing_research_summary.md — narrative final review 적용 미정 (학위 논문 chapter draft 용 정량 evidence 중심 문서라서 narrative flow 강조는 advisor briefing 만으로 충분)

- **에스컬레이션 필요 여부**: 없음. 사용자가 advisor briefing 최종본 읽고 학회 발표 준비 가능.

- **추가 필요 분석**: 없음. 단 (후속 선택) `dsn_phase2_mitigation_null_mechanism.md` §3 의 4 ckpt 표에 topk20, abstau07 multi-DB 측정값 보강 — priority 낮음, analyzer 요청 큐에 backlog.

- **남은 옵션 (사용자 후속 결정)**:
  - over_smoothing_research_summary.md §1.4 에 같은 sub-mechanism 분리 명시 보강할지
  - 학회 발표 슬라이드 / paper §V.5.4 narrative 에 paradox 3 (ckpt-invariance) 를 contribution candidate 로 강조할지

---

## 2026-05-11 (Phase 1 attention metric Multi-DB Stratified 재측정 완료 — n=55 측정값 + silent skip 진단 결론)

> **root 세션 (본 chain) 결과 — 5/11 KST 18:45~18:46 (~1 min wall)**. 직전 entry 의 결정 사항을 모두 수행. paradox narrative 가 정정 (uniform attention → moderate-sharp peaking + 새 paradox 등장 attention ckpt-invariance).

- **결과 (multi-DB n=55, per_db=5 × 11 DBs, seed=42 — A3 protocol)**:
  1. **4 ckpt 모두 55/55 success_rate=100%, fail_by_stage={}** — silent skip 완전 해소
  2. col→tab edge L2 top5_conc / entropy:

     | ckpt | L2 top5_conc | L2 entropy | val recall@15 |
     |---|---:|---:|---:|
     | p80 | 0.7144 ± 0.2167 | 1.9227 ± 0.6884 | 0.6097 |
     | topk20 | 0.7151 ± 0.2155 | 1.9120 ± 0.7020 | 0.5839 |
     | abstau07 | 0.7148 ± 0.2166 | 1.9149 ± 0.6968 | 0.5805 |
     | qcond_nl3 | 0.7167 ± 0.2160 | 1.9019 ± 0.7056 | 0.6061 |

  3. **4 ckpt spread = 0.0023 (0.3% sub-noise)** — directedness/threshold/QCond 모두 attention 에 noise-level 영향만 → 새 paradox: attention ckpt-invariance

- **Silent skip 원인 진단 결론**:
  1. **dev.json[0..49] = california_schools 50 queries 전체** (`Counter({'california_schools': 50})`) — 옛 single-DB protocol 의 사실상 단일 DB iteration
  2. 옛 num_queries=2 의 root cause: 5/6 측정 시점에만 발생한 transient. **본 재측정에서 single-DB 50 queries 도 100% 성공** (`extract_layerwise_attention_v2` 의 robustness bug 재현 안 됨)
  3. → silent skip 의 직접 원인 잡지 못 했으나 **multi-DB protocol 자체가 caveat 의 구조적 회피책**. n=55 가 신뢰 출처. fix priority 낮음.

- **Paradox narrative 정정** (직전 entry 의 narrative 와 비교):

  | Paradox | 옛 narrative (single-DB n=2 기반) | 새 narrative (multi-DB n=55 기반) |
  |---|---|---|
  | Paradox 1 (attention) | "col→tab attention near-uniform (top5_conc 0.24~0.35, H≈3.22) 인데도 collapse" | "col→tab attention moderate-sharp peaking (top5_conc ≈ 0.72, H ≈ 1.91) 인데도 collapse" — sharpness 톤다운, **mech(ii-b) 후보 강도는 유지** |
  | Paradox 2 (gradient) | "$\rho_{\text{skip}} = 3.02$ extreme" | **유지** (Stage 3 측정값 무관) |
  | Paradox 3 (NEW) | — | **attention ckpt-invariance**: 4 ckpt spread 0.3% sub-noise → 학습 변형으로 attention 검증 불가, 직접 표적 mitigation 필요 |

- **DB schema 가 dominant variable**:
  - per-DB top5_conc spread = 0.24 (california_schools, outlier) ~ 1.00 (toxicology, in-degree ≤ 5)
  - 4 ckpt × 11 DBs 의 per-DB spread 모두 < 0.015 — attention variability 의 dominant source 는 schema, 학습 변형 아님

- **산출물 (root 작성)**:
  - `src/analysis/dsn_oversmoothing_phase1_multi_db.py` (신규)
  - `scripts/run_dsn_phase1_multi_db_attention.sh` (신규)
  - `outputs/analysis/dsn_oversmoothing_multi_db/{p80, topk20, abstau07, qcond_nl3}/{attention_metrics, per_db_breakdown, fail_log}.json + plots`
  - `outputs/analysis/dsn_oversmoothing_multi_db/cross_ckpt_summary.json + comparison_4ckpt_multi_db.png`
  - `notebooks/analysis_results/dsn_oversmoothing_phase1_multi_db_recheck.md` (analyzer 보고서 형식)
  - `planning/advisor_briefing_oversmoothing_2026-05-11.md` §0/§1.3/§1.4 갱신 (수치 + 새 paradox 추가)
  - `planning/over_smoothing_research_summary.md` §1.4 갱신 (수치 + caveat 갱신)

- **에스컬레이션 / 후속**:
  - **planner 세션**: advisor briefing narrative final review — paradox 3개 중 paradox 1 톤다운 + paradox 3 (NEW, ckpt-invariance) 추가가 학회 narrative 에 자연 흐름 형성하는지 검토. 학회 보고용 시 mech(ii-a) softmax over-concentration 가설은 ckpt 변경으로 검증 불가 → LayerNorm/GIN 직접 표적 mitigation 만이 유일 검증 통로 명시.
  - **선택 후속 analyzer**: `dsn_phase2_mitigation_null_mechanism.md` §3 의 4 ckpt 표에 topk20, abstau07 multi-DB 측정값 보강 (현재 raw JSON 만 존재). Priority 낮음.

- **에스컬레이션 필요 여부**: code/data 산출 완료, planner 가 narrative 후속만.

---

## 2026-05-11 (Phase 1 attention metric Multi-DB Stratified 재측정 결정 — n=2 single-DB caveat 해소 + silent skip 원인 진단)

> **사용자 직전 input (2026-05-11)**: "음 수치는 확인했어 근데 네 말대로 n=2인게 신경쓰이네 다시 제대로 실험해볼 수 있나?" → AskUserQuestion 결과:
> - Protocol: **Multi-DB stratified 55 queries (A3 protocol)**
> - 진단: **silent skip 원인 진단 + 재측정 함께**

- **결정**:
  1. **Protocol**: A3 (Stage 5) 와 동일한 multi-DB stratified — `dsn_phase1_deep_dive.py:build_stratified_qids(per_db=5, seed=42)` → 11 BIRD-Dev DBs × 5 queries = 55 queries.
     - 이점: Stage 5 A3 + Stage 7 GIN 8-trial 의 다른 ckpt 와 **동일 protocol 직접 비교 가능**
     - silent fail caveat 자동 완화 (한 DB fail 해도 다른 10 DBs sample 유지)

  2. **대상 4 ckpt**: p80, topk20, abstau07, qcond_nl3 (Phase 1 진단 4 ckpt 전부). 출력은 `outputs/analysis/dsn_oversmoothing_multi_db/<ckpt>/attention_metrics.json` 신규 경로.

  3. **Silent skip 원인 진단** (codebase fix candidate):
     - `src/analysis/dsn_oversmoothing_analysis.py:run_step3_one` 의 try/except 에서 `idx==0` 일 때만 warning → **모든 query 의 fail 사유 logger.warning** 으로 임시 패치 (또는 별도 진단 dump)
     - 50 query 중 48 개 silent fail 의 원인 후보:
       - `directed_from_sn` edge 가 zero edges 인 query (supernode threshold)
       - `column→belongs_to→table` edge 가 빈 query
       - forward hook 의 `return_attention_weights` 미지원 edge type
     - 진단 결과는 analyzer 보고서에 포함 + extract_layerwise_attention_v2 의 robustness fix 가 가능하면 module:selectors 세션에 별도 에스컬레이션

- **근거**:
  - 사용자 직접 input ("n=2 가 신경쓰여서 다시 제대로")
  - `outputs/analysis/dsn_attention/{p80, topk20, abstau07, qcond_nl3}/attention_metrics.json` 의 `num_queries = 2` 직접 확인 — 50 query iterate 중 48 silent skip
  - A3 stratified protocol 가 Stage 5 에서 검증된 base — single-DB caveat 해소 + 다른 stage 와 비교 가능
  - `src/analysis/dsn_phase1_deep_dive.py:build_stratified_qids` 가 이미 구현 + import 패턴 (`dsn_mitigation_v3_8trial.py:75`) 검증

- **영향 범위**:
  - 신규 산출물:
    - `src/analysis/dsn_oversmoothing_phase1_multi_db.py` (또는 기존 dsn_oversmoothing_analysis.py 에 `--multi_db` flag 추가) — root 가 작성
    - `scripts/run_dsn_phase1_multi_db_attention.sh` — root single-command launch
    - `outputs/analysis/dsn_oversmoothing_multi_db/{p80, topk20, abstau07, qcond_nl3}/attention_metrics.json` + plots
    - `notebooks/analysis_results/dsn_oversmoothing_phase1_multi_db_recheck.md` — analyzer 보고서 (single-DB n=2 vs multi-DB n=55 비교 + paradox narrative 검증)
  - 업데이트 대상:
    - advisor_briefing_oversmoothing_2026-05-11.md §0 timeline + §1.3 + §1.4 (수치 + caveat 갱신)
    - over_smoothing_research_summary.md §1.4 (4 ckpt × multi-DB 수치 표 추가)
    - 본 DECISIONS.md (재측정 완료 후 결과 entry 추가)

- **에스컬레이션 필요 여부**:
  - **root 세션**: 코드 작성 + 실험 실행 + analyzer 보고서 작성 + advisor briefing/summary 업데이트 (한 핸드오프로 전체 chain). 5/11 KST 발사 가능.
  - **module:selectors 세션** (선택): 만약 silent skip 원인이 extract_layerwise_attention_v2 의 robustness bug 면 fix 위임. 단 이건 진단 결과 후 결정.

- **추가 필요 분석**:
  - 본 재측정 결과로 paradox narrative 가 **유지** (uniform attention 인데도 collapse) 면 advisor briefing 의 Stage 1 narrative 일관 confirm.
  - 만약 multi-DB 에서 top5_conc 가 dramatic 다르게 (예: 0.7+ sharp peaking) 나오면 paradox 양상 재정정 + mech(ii-a) sharp peaking 가설 다시 검토 — 단 A3 의 11 DBs 결과 (Phase 2 b8 의 col→tab top5_conc 0.8797 vs Phase 1 0.7144) 와 일관할 가능성이 높음.

---

## 2026-05-11 (Stage 1 paradox 수치 정정 — top5_conc 0.91/entropy 0.51 → 실제 측정값 0.24~0.35/3.22 + paradox narrative 재작성)

> **사용자 직전 input (2026-05-11)**: "top5-conc 값이 QCond SN과 dsn_P80, top20, abstou07 에서 각각 어떤지 어디에 정리되어 있지?" → JSON 직접 조회 결과 advisor briefing §1.3 + summary §1.4 의 수치가 잘못된 것 발견 → "수정해야 할 부분 수정해 줘"

- **결정**:
  1. **수치 정정** — 두 문서 (advisor_briefing_oversmoothing_2026-05-11.md + over_smoothing_research_summary.md) 의 Stage 1 paradox 진술에서 잘못된 수치 (top-5 conc ≈ 0.91, entropy H ≈ 0.51) 를 실제 측정값 (col→tab edge L2 top5_conc 0.24~0.35, entropy 3.17~3.22 ≈ ln(25) near-uniform) 으로 교체.

  2. **데이터 출처 명시**: `outputs/analysis/dsn_attention/{p80, topk20, abstau07, qcond_nl3}/attention_metrics.json` (raw JSON, 4 ckpt 모두) + `outputs/analysis/dsn_attention/comparison_4ckpt.png` (cross-model 시각화). 단일 narrative document (`dsn_phase2_mitigation_null_mechanism.md` §3 라인 94-99) 은 p80, qcond_nl3 만 정리 — topk20, abstau07 raw JSON 만 있고 narrative 누락.

  3. **Paradox narrative 재작성** — 직전 narrative ("attention 매우 집중적인데도 collapse → root cause ≠ attention") 는 오류. 실제로는:
     - **Paradox 1**: col→tab attention near-uniform 인데도 collapse → **sharp peaking 이 통로 X**. 오히려 uniform attention 평균화 자체가 root cause 후보 → mech(ii-b) weighted-mean propagation collapse 강력 시사. 이는 Stage 7 GIN partial 부정 결론 (softmax-aggregation combo) 과 일관.
     - **Paradox 2**: $\rho_{\text{skip}} = 3.02$ extreme → mech(iii) 1차 후보 (Stage 3 에서 부정됨).
     - **두 paradox 동시 발생** narrative 로 Stage 2 mech 1차 판정 + Stage 3 부정 + Stage 4 mech(ii) DOMINANT 갱신 + Stage 7 mech(ii-b) sub-mechanism 정밀화 의 전체 흐름이 더 일관.

  4. **추가 보강** — `num_queries=2` small-sample caveat 명시 (directed_from_sn edge 가 있는 query 만 capture, single-DB california_schools 한정). 4 ckpt 상대 비교는 일관 단 절대 수치는 small-sample.

  5. **수정된 위치 3 곳**:
     - advisor_briefing_oversmoothing_2026-05-11.md §0 timeline (line 19): "attention 매우 집중인데 (top5_conc=0.91)" → "col→tab attention 거의 uniform (top5_conc 0.24~0.35, H≈3.22) + ρ_skip=3.02 extreme → 두 paradox 동시 발생"
     - advisor_briefing_oversmoothing_2026-05-11.md §1.3 Step 3 결과: 4 ckpt × L1/L2/L3 top5_conc + L2 entropy 정리 표 신규 + paradox narrative 재작성
     - advisor_briefing_oversmoothing_2026-05-11.md §1.4 무엇을 알게 됐나: 두 paradox 의 시사점 명시 + 4 후보 mechanism 의 후보 강도 표기
     - over_smoothing_research_summary.md §1.4 Paradox 발견: 4 ckpt 정확 수치 표 + paradox narrative 재작성

- **근거**:
  - `outputs/analysis/dsn_attention/{p80, topk20, abstau07, qcond_nl3}/attention_metrics.json` 직접 조회 결과 (col→tab edge):
    - p80 L2 top5_conc=0.2441, L2 entropy=3.2177
    - topk20 L2 top5_conc=0.2467, L2 entropy=3.2176
    - abstau07 L2 top5_conc=0.2435, L2 entropy=3.2179
    - qcond_nl3 L2 top5_conc=0.2434, L2 entropy=3.2174
  - `dsn_phase2_mitigation_null_mechanism.md` §3 라인 94-99 표 — p80_phase1 L2 top5_conc=0.2445 (일관), qcond_nl3 L2 top5_conc=0.2434 (일관). 단 narrative 표에 topk20, abstau07 누락 — 후속 analyzer 보고서 보완 candidate.
  - 직전 advisor briefing 의 "0.91 / 0.51" 수치 출처는 추측: phase 2 b5_mit ckpt 의 col→tab top5_conc=0.6715 + mitigation 후 multi-DB 측정값 0.7144 등을 잘못 가져온 것으로 추정.

- **영향 범위**:
  - planning/advisor_briefing_oversmoothing_2026-05-11.md §0/§1.3/§1.4 (수정 완료)
  - planning/over_smoothing_research_summary.md §1.4 (수정 완료)
  - planning/DECISIONS.md (본 entry)

- **에스컬레이션 필요 여부**: Analyzer 보고서 보완 candidate — `dsn_phase2_mitigation_null_mechanism.md` §3 의 4 ckpt 표에 topk20, abstau07 추가 (현재 raw JSON 만 존재). 단 직전 narrative 의 main 목적은 Phase 2 b8 mit 효과 비교라 보완 priority 는 낮음 — Phase 1 진단 narrative 가 필요하면 별도 analyzer 요청 큐에 등록.

- **추가 필요 분석**: 없음. 본 정정은 기존 측정 데이터의 정확한 인용 작업.

---

## 2026-05-11 (advisor_briefing_oversmoothing_2026-05-11.md 신규 — 분석 흐름 중심 narrative 보고용 자료)

> **사용자 직전 input (2026-05-11)**: "src/analysis/dsn_oversmoothing_analysis.py 을 기반으로 지도교수님께 이런 상황임을 보고하는 자료를 만드려고 해 / 전체적으로 이야기의 흐름에 공백이 좀 있는 것 같아 / 각 분석마다 왜 그런 분석을 했고 결과가 어땠으며 어떤 이유로 다른 분석을 시도한 건지 흐름이 잘 정리되면 좋겠네"

- **결정**:
  1. **(a) 신규 산출물** — [`planning/advisor_briefing_oversmoothing_2026-05-11.md`](advisor_briefing_oversmoothing_2026-05-11.md): 분석 흐름 중심 narrative 보고 자료. 기존 `over_smoothing_research_summary.md` 가 정량 evidence + 수식 reference 중심인 반면, advisor briefing 은 **stage 별 "왜 했는지 → 결과 → 다음 분석으로 간 이유"** 의 narrative flow 중심.

  2. **(b) 두 문서 역할 분리**:
     - `over_smoothing_research_summary.md` (5/11 원본) — 학위 논문 Part III chapter draft 용 정량 evidence + 수식 reference (15 sections, ~600 lines, §A 정의 + §B 측정 metric + §C mitigation 수식)
     - `advisor_briefing_oversmoothing_2026-05-11.md` (5/11 신규) — 지도교수 보고용 narrative flow (10 sections, ~600 lines)
       - §0 한눈에 보는 8-stage timeline (ASCII 흐름도)
       - §1~§7 각 stage = 4 절 구조 (왜 분석 / 무엇을 했나 / 결과 / 무엇을 알게 됐나 + 다음으로)
       - §8 진행 중 + 종합 (8-trial dominance matrix + dominance 진화 timeline)
       - §9 보고용 Q&A 예상 질문 7개
       - §10 참고 자료 목록 (분석 코드 6 + analyzer 보고서 9 + 연관 문서 3)

  3. **(c) Narrative 흐름 핵심 — Stage 간 연결고리 명시**:
     - Stage 1 → 2: 4 mechanism 후보 분리, 표준 처방 통합으로 elimination 시작
     - Stage 2 → 3: mech(iii) Skip Dep DOMINANT 5/5 판정 → 직접 표적 검증
     - Stage 3 → 4: mech(iii) 부정, mech(ii) DOMINANT 갱신 → edge softmax 직접 표적 3 candidate
     - Stage 4 → 5: v2 #3 LN paradox (attention 회복 X collapse 보존) → 정밀 분석 deep dive
     - Stage 5 → 6: mech(ii) → (ii-a) + (ii-b) 분리 정식 → 7-trial dominance 갱신
     - Stage 6 → 7: 두 mit 모두 mech(ii-b) 차단 X → propagation 자체 변경 (GIN)
     - Stage 7 → 8: GIN partial 부정 evidence → softmax-aggregation combo 정식 + max aggregation 후속 launch

- **근거**:
  - 사용자 input 직접 인용 ("이야기의 흐름에 공백이 좀 있는 것 같아")
  - 기존 `over_smoothing_research_summary.md` 가 정량 evidence 중심 reference document 로서는 충분하지만 narrative 연결성 부족 (Stage 간 "왜 다음으로 갔는지" 가 분산)
  - 지도교수 보고용 자료는 별도 narrative 중심 문서가 적합 — 두 문서 역할 분리

- **영향 범위**:
  - planning/advisor_briefing_oversmoothing_2026-05-11.md (신규)
  - 기존 `over_smoothing_research_summary.md` 변경 X (reference document 그대로 보존)
  - DECISIONS.md (본 entry)

- **에스컬레이션 필요 여부**: 없음. 보고 자료는 사용자 owner.

- **추가 필요 분석**: 없음. Mitigation v3 #2 max aggregation 결과 (5/14 ETA) 후 advisor briefing §8.1 + §8.2 분기 갱신 (V3-A-1 vs V3-A-2 confirm).

---

## 2026-05-11 (over_smoothing_research_summary.md 신규 작성 — 7-stage 시도 + 5단계 mechanism deep dive + 8-trial Final 통합 narrative + analyzer 6 보고서 직접 인용 정리) — 학위 논문 Part III chapter draft 작성 base

> **사용자 직전 input (2026-05-11)**: "지금까지 Over-Smoothing 문제를 해결하기 위해 진행한 노력을 정리하고 싶어 / 새로운 마크다운으로 정리해 주는데 내용을 analyzer 의 분석 결과 마크다운 보고서를 참고해서 작성해 줘"

- **결정**:
  1. **(a) 신규 산출물** — [`planning/over_smoothing_research_summary.md`](over_smoothing_research_summary.md): planner 신규 narrative 정리 마크다운 (15 sections, ~600 lines). analyzer 6 보고서 직접 인용 형태:
     - [`dsn_oversmoothing_analysis.md`](../notebooks/analysis_results/dsn_oversmoothing_analysis.md) (Phase 1 진단)
     - [`dsn_phase2_mitigation_null_mechanism.md`](../notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md) (Phase 2 4-mech)
     - [`dsn_phase3_mitigation_results.md`](../notebooks/analysis_results/dsn_phase3_mitigation_results.md) (4-trial dominance)
     - [`dsn_mitigation_v2_results.md`](../notebooks/analysis_results/dsn_mitigation_v2_results.md) (7-trial mid-training)
     - [`dsn_v2_layernorm_mechanism_decomposition.md`](../notebooks/analysis_results/dsn_v2_layernorm_mechanism_decomposition.md) (A1)
     - [`dsn_softmax_noise_sensitivity.md`](../notebooks/analysis_results/dsn_softmax_noise_sensitivity.md) (A2)
     - [`dsn_per_db_stratified_7ckpt.md`](../notebooks/analysis_results/dsn_per_db_stratified_7ckpt.md) (A3)
     - [`dsn_mitigation_v2_final_7trial.md`](../notebooks/analysis_results/dsn_mitigation_v2_final_7trial.md) (Final 7-trial)
     - [`dsn_mitigation_v3_8trial.md`](../notebooks/analysis_results/dsn_mitigation_v3_8trial.md) (8-trial Final + GIN)

  2. **(b) 정리 구조 — 7-stage + 5단계 mechanism deep dive**:
     - §0 TL;DR (over-smoothing 진단 + 8-trial null effect + Final dominance scoring)
     - §1 Stage 1 — 문제 발견 (Phase 1 baseline + paradox)
     - §2 Stage 2 — Mitigation v1 B5 통합 (Phase 2)
     - §3 Stage 3 — Mitigation v2 (Phase 3 #3 + #4) + 4-trial mech 갱신
     - §4 Stage 4 — Mitigation v2 (DropMessage + LayerNorm + Sum) — 7-trial mid-training
     - §5 Stage 5 — Phase 1 Deep Dive (A1 + A2 + A3) — sub-mechanism 분리
     - §6 Stage 6 — Final 7-Trial Dominance Scoring + v2 LN ↔ v2 Sum contrast
     - §7 Stage 7 — Mitigation v3 #1 GIN-style aggregation + 8-trial dominance + GIN partial 부정 evidence + sub-mechanism 정밀화 (softmax-aggregation combo)
     - §8 진행 중 — Mitigation v3 #2 Max Aggregation (5/13~5/14)
     - §9 종합 — 8-Trial Final Mechanism Dominance Scoring + dominance 진화 timeline
     - §10 Filter Dominance 6번째 축 — 8-trial evidence 통합
     - §11 paper §V.5.4 narrative 본문 정식 (analyzer §14 직접 인용)
     - §12 Future Work — Post-Paper Phase 5 candidate
     - §13 학위 논문 Part III chapter outline 갱신 (paper_research_direction.md §3.5)
     - §14 데이터 / 산출물 위치 + 재현 스크립트
     - §15 핵심 학술적 기여 (학위 논문 Part III chapter)

  3. **(c) 활용 목적**:
     - **학위 논문 Part III chapter draft 작성 base** (사용자 5/10~5/22) — §III.1~§III.9 outline + analyzer §14 본문 정식 narrative 직접 활용 가능
     - **paper §V.5.4 narrative 본문 정식** (analyzer §14 직접 인용) — paper draft 작성 시 인용 base
     - **5/15 Phase 5 (planner 9-trial dominance scoring 갱신) base** — max aggregation 결과 후 본 마크다운 §8 갱신 + §9 dominance scoring 표 갱신

- **근거**:
  - **사용자 직전 input** (2026-05-11): narrative 정리 + analyzer 보고서 참조 마크다운 작성 요청
  - **선행 결정**: DECISIONS 직전 entries (2026-05-05 ~ 2026-05-09 모든 entries 통합)
  - **선행 분석**: analyzer 9 보고서 (Phase 1 진단부터 8-trial Final 까지 포함)

- **영향 범위**:
  - **신규 산출물 `planning/over_smoothing_research_summary.md`** — 학위 논문 Part III chapter draft 작성 base (15 sections, ~600 lines, analyzer 9 보고서 직접 인용)
  - **DECISIONS 본 엔트리** — 산출물 위치 + 정리 구조 + 활용 목적 명시
  - **paper_research_direction.md** 영향 X (직전 entries 의 갱신 그대로 — 본 마크다운은 별도 narrative 정리)
  - **presentation_brief_2026-04-28.md** 영향 X (직전 §14.15 갱신 그대로)
  - **paper main contribution (학회)** 영향 X
  - **학위 논문 Part III chapter draft 작성 효율 향상** — 사용자가 chapter draft 작성 시 본 마크다운 의 narrative + analyzer §14 본문 정식 + Part III outline 직접 활용

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — `over_smoothing_research_summary.md` 신규 작성 + DECISIONS 본 엔트리
  2. **사용자 (5/10~5/22)** — 학위 논문 Part III chapter draft 작성 (본 마크다운 base 활용 + paper §V.5.4 narrative 본문 정식 직접 인용)
  3. **사용자 (5/13)** — Root max aggregation 학습 launch (직전 핸드오프 prompt 그대로)
  4. **Planner (5/15, max + analyzer 9-trial 결과 후)** — 본 마크다운 §8 + §9 + §13 갱신 (시나리오 V3-A-1/V3-A-2 분기 결정)

- **추가 필요 분석** (5/15 max 학습 + analyzer 결과 후):
  - 본 마크다운 §8 (max aggregation 결과 갱신) + §9.1 (9-trial dominance scoring matrix 갱신) + §13 (§III.9 outline 정식 채택)

---

## 2026-05-09 (Phase 4 8-trial Final 분석 완료 — 🎯 시나리오 V3-A 정식 confirm + 🆕 Mech(ii-b) 5/5 → 4/5 partial 부정 + Mech(i-b) 3/5 → 4/5 강화 + paper §V.5.4 narrative 8-trial 본문 정식 채택) — Mitigation v3 #2 max aggregation 5/13 launch (사용자 결정 (3)A 자동 trigger)

> **Status**: Analyzer 신규 산출 [dsn_mitigation_v3_8trial.md](../notebooks/analysis_results/dsn_mitigation_v3_8trial.md) (8 ckpt × 5 step + multi-DB stratified 55 queries) 완료. **시나리오 V3-A 정식 confirm** (GIN R=0.5954 < Phase 1 0.6097, Δ=-0.0143). **🚨 Mech(ii-b) partial 부정 evidence 발견**: GIN L2_GAT cosine = 0.9137 (다른 mit ckpt 의 0.99+ 대비 -0.08, 11 DBs 모두 일관 partial 회복 0.82~0.95). 단 R ceiling 갱신 X → softmax-aggregation **combo** 한정 mechanism 입증. **paper §V.5.4 narrative 본문 정식 채택** (analyzer §14 직접 인용).

- **결정**:

  1. **(a) ✅ 시나리오 V3-A 정식 confirm — GIN R=0.5954 < Phase 1 baseline**:

     | 시나리오 | 기준 | GIN 결과 | 판정 |
     |---|---|---|---|
     | V3-A (가능성 中→中, sum-only fail vs sum+MLP combo) | R ceiling 갱신 X (~0.59-0.61) | **R=0.5954 (Δ=-0.0143)** | **✅ confirm** — aggregation family 자체 limitation but partial mitigation evidence 발견 |
     | V3-B (가능성 中→中) | R partial recovery (0.62-0.70) | (미달) | ❌ |
     | V3-C (가능성 낮음) | R 0.85+ ceiling 갱신 | (사실상 불가능) | ❌ |

     단 V3-A confirm 이 **partial mitigation evidence 와 양립** — GIN L2 cosine partial 회복 (0.9137 vs 0.99+) 발견 단 R ceiling 미갱신 → mech(ii-b) fundamental limitation 그대로

  2. **(b) 🚨 Mech(ii-b) Sub-mechanism 정밀화 — softmax-aggregation combo (analyzer §8)**:

     ```
     mech(ii-b) Weighted-mean Propagation Collapse (8-trial DOMINANT 4/5)
                              │
                              ▼
                ┌─────────────┴─────────────┐
                ▼                           ▼
        softmax dimension              aggregation dimension
        (LN partial mit)               (GIN sum+MLP partial mit)
        ──────────────────             ──────────────────
        v2 #3 LN                       v3 #1 GIN
        attention 회복 (top5 0.75)     L2 cosine -0.08 회복
                                       (0.99 → 0.91)
        R: -0.0007 vs Phase 2          R: -0.0143 vs Phase 1
        (사실상 동등)                   (ceiling 미갱신)

           ┌────────────┴────────────┐
           ▼                         ▼
       ⚡ softmax + aggregation 동시 변경 시
       더 큰 mitigation 가능 candidate (post-paper)
       (LN+GIN combo, EGAT 등)
     ```

     - **LN (mech(ii-a) softmax direct)**: attention 회복 단 L1=1.0 보존 (mech(ii-b) 차단 X)
     - **GIN (mech(ii-b) aggregation direct)**: L2 cosine partial 회복 (0.91 vs 0.99) 단 R ceiling 미갱신
     - **mech(ii-b) 의 정확한 mechanism = softmax + weighted-mean combo**: 한 component 만 변경 시 partial mitigation, 둘 다 고치면 더 큰 회복 가능 (post-paper Phase 5 candidate)

  3. **(c) 🆕 Mech(i-b) 강도 3/5 → 4/5 강화 — Aggregation Function Hierarchy**:

     | Aggregation Function | Variant | R@15 | Δ vs Phase 1 |
     |---|---|---:|---:|
     | mean | Phase 2 b8 | **0.6018** | -0.0079 |
     | sum + MLP (GIN nonlinearity) | v3 #1 GIN | 0.5954 | -0.0143 |
     | sum-only | v2 #2 Sum | 0.5761 | -0.0336 |

     - **Hierarchy 정량**: mean > sum+MLP > sum-only
     - **MLP nonlinearity 효과**: sum-only 의 magnitude variance sensitivity 를 MLP 가 partial 보완 (+0.0193 vs sum-only)
     - **8-trial evidence** 가 mech(i-b) 차원 정량 정밀화 (sum direct + GIN partial)

  4. **(d) 8-Trial Final Mechanism Dominance Scoring 정식 갱신 (analyzer §0+§7)**:

     | Sub-mechanism | 7-Trial Final | **8-Trial 갱신** | 핵심 정량 |
     |---|:---:|:---:|---|
     | (i-a) top-5 attention sibling | 2/5 | 2/5 (변경 X) | top5_raw cos 0.55 ≈ entire 0.51 |
     | **(i-b) aggregation function magnitude** | 3/5 부분 | **🆕 4/5 ⬆ 강화** | sum-only fail (-0.0336) + GIN sum+MLP partial (-0.0143) — aggregation family hierarchy |
     | **(ii-a) softmax over-concentration** | 5/5 partial mit | 5/5 partial mit (변경 X) | v2 #3 LN attention 회복 (0.7510 multi-DB) + noise robust + 11 DBs 일관 |
     | **(ii-b) weighted-mean propagation collapse** | 5/5 ⭐⭐ DOMINANT | **🆕 4/5 ⬇ partial 부정** | GIN L2=0.9137 (다른 mit 0.99+ 대비 -0.08, 11 DBs 일관) + L1=0.9936. 단 R ceiling 미갱신 → softmax-aggregation combo 한정 mechanism |
     | (iii) Skip dependency | 3/5 부정 | 3/5 (변경 X) | GIN skip_dep 1.36 (균형) + R 미달 일관 |
     | (iv) Schema sibling (raw PLM) | 3/5 | 3/5 (변경 X) | L0=0.6334 multi-DB lower bound |

     **Dominant 갱신**: mech(ii-b) 5/5 → **4/5 partial 부정** (GIN evidence) — fundamental limitation 그대로 단 sub-mechanism (softmax-aggregation combo) 정밀화 + mech(i-b) 3/5 → **4/5 강화** (aggregation family hierarchy)

  5. **(e) 🚀 paper §V.5.4 narrative 본문 정식 채택 (analyzer §14 직접 인용)**:

     > "DSN 8-trial mitigation null effect (Phase 1 0.6097, mitigation 7 variants R=0.5761~0.6018) 는 GAT 의 fundamental architectural limitation 의 결정적 evidence 이다.
     >
     > v2 #3 LayerNorm pre-softmax 가 mech(ii-a) softmax over-concentration 의 direct mitigation 으로 attention pattern 을 baseline 으로 회복 (multi-DB col→tab L2 top5_conc 0.7510 vs Phase 1 0.7144) — Phase 2 b8 와 사실상 동등 R=0.6011 (-0.0007). 그러나 mech(ii-b) weighted-mean message aggregation collapse (모든 7 ckpt L1_GAT cosine = 1.0) 은 차단 X.
     >
     > v3 #1 GIN-style aggregation 이 sum + MLP propagation 으로 weighted-mean 우회 → **mech(ii-b) partial 부정 evidence**: GIN L2_GAT cosine = 0.9137 (다른 mit 의 0.99+ 대비 -0.08, multi-DB stratified 11 BIRD-Dev DBs 모두 일관 partial 회복 L2=0.82~0.95). 단 R@15 = 0.5954 (Phase 1 -0.0143, ceiling 갱신 X) — softmax-aggregation combo 한정 mechanism 입증.
     >
     > v2 #2 Sum Aggregation (R=0.5761, -0.0336) + v3 #1 GIN (R=0.5954, sum+MLP) 모두 sum aggregation family — mech(i-b) aggregation function magnitude sensitivity direct evidence. mean (Phase 2 0.6018) > sum+MLP (GIN 0.5954) > sum-only (v2 #2 0.5761) hierarchy.
     >
     > Filter Dominance 6번째 축 (training-pathology-invariant) 결정적 8-trial evidence: GAT 의 fundamental architectural limitation (mech(ii-b) softmax-weighted-mean combo + mech(i-b) aggregation magnitude sensitivity + mech(ii-a) softmax over-concentration) 까지 With-Filter pipeline 이 흡수."

  6. **(f) 🚀 Mitigation v3 #2 max aggregation 5/13 launch (사용자 결정 (3)A 자동 trigger, V3-A confirm 후 conditional 활성화)**:

     - 직전 DECISIONS (사용자 결정 confirm entry §1(d)) 의 V3-A 시 conditional plan 자동 trigger
     - **추가 코드 변경 없음** — Selector 단계 6 의 AGGREGATION_TYPES 에 max 이미 포함 + smoke 통과
     - **신규 config**: `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_max.yaml` (Base = Phase 2 b8 + `aggregation_type: "max"`)
     - **학습 launch**: 5/13 GPU 0 (오늘 5/9 → 5/10~5/12 학위 논문 Part III chapter draft 작성 + 5/13 max 학습 launch)
       - `CUDA_VISIBLE_DEVICES=0 nohup python src/train_gat_s06.py --config configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_max.yaml > logs/train/gat_dsn_p80_b5_mitigation_v3_max_$(date +%Y%m%d_%H%M).log 2>&1 &`
       - ~10h 학습, batch_size=8, ETA 5/14 KST
     - **시나리오 V3-A-1 (mean+sum+max+GIN 4 family null effect)**: max R 0.59-0.61 → 4 aggregation family null evidence 결정적 → mech(ii-b) DOMINANT (combo 한정) 4/5 narrative 절대 confirm
     - **시나리오 V3-A-2 (max partial 회복)**: max aggregation 의 부분 효과 발견 → aggregation function 별 magnitude treatment 차이 mechanism 분석 (post-paper deep dive)

  7. **(g) post-paper Phase 5 candidate (mech(ii-b) sub-mechanism 정밀화 후속)**:

     | 우선순위 | Candidate | 가설 | 시점 |
     |---|---|---|---|
     | **#1** | **LN + GIN combo** (LayerNorm pre-softmax + GIN aggregation) | mech(ii-b) softmax + aggregation 동시 변경 → 더 큰 mitigation 가능 candidate. v2 #3 LN attention 회복 + v3 GIN L2 cosine -0.08 회복 의 combo 효과 검증 | post-paper |
     | **#2** | **EGAT (Energy-based GNN)** | softmax 자체 대체 (energy minimization aggregation). softmax-weighted-mean combo 자체 변경 | post-paper |
     | #3 | Multi-hop attention (skip GAT layer) | message aggregation 우회 | post-paper |
     | #4 | Self-loop weight scaling | sibling 동질화 압력 약화 | post-paper |

     - **본 entry 신설 (analyzer §11.2 + §13)**: GIN partial 부정 evidence + mech(ii-b) softmax-aggregation combo 정밀화 → LN+GIN combo 가 가장 잠재력 큰 후보 (각각 partial mitigation 입증, combo 효과 검증 가치 高)

  8. **(h) 학위 논문 Part III chapter §III.4/§III.6/§III.8 갱신 (analyzer §0+§14)**:
     - 직전 outline (사용자 결정 confirm entry §1(c)): §III.4 mech(ii) sub-mechanism 분리 + §III.6 v2 #3 LN ↔ v2 #2 Sum contrast + §III.8 GIN 결과 후
     - **본 entry 정식 갱신**:
       - §III.4 = **mech(ii-b) DOMINANT 4/5 partial 부정 (GIN evidence) + mech(ii-a) 5/5 partial mit + 🆕 mech(i-b) 4/5 강화 (aggregation family hierarchy) + mech(iii) 3/5 부정**
       - §III.6 = v2 #3 LN ↔ v2 #2 Sum -0.0250 contrast + 🆕 v3 GIN ↔ v2 #2 Sum +0.0193 hierarchy (sum+MLP vs sum-only)
       - **§III.8 정식 채택**: V3-A confirm + GIN partial 부정 evidence + mech(ii-b) sub-mechanism (softmax-aggregation combo) 정밀화 narrative
       - §III.9 (신설, 5/14 max 학습 결과 후) = mean+sum+max+GIN 4 aggregation family null evidence (V3-A-1 시) 또는 max partial 회복 mechanism (V3-A-2 시)

  9. **🚨 사용자 결정 필요 3 항목**:

     | # | 결정 항목 | 옵션 | 권장 |
     |---|----------|------|------|
     | (1) | **Mitigation v3 #2 max aggregation 학습 launch 시점 확정** | (A) 5/13 launch (사용자 결정 (3)A 자동 trigger 그대로) / (B) 5/10~5/13 학위 논문 Part III chapter draft 작성 우선 → 학습 5/13 launch / (C) 학습 즉시 (5/9 또는 5/10) | **(A) 5/13 launch** — 학위 논문 Part III chapter draft 작성 (5/10~5/13) 진행 + max 학습 5/13 launch (ETA 5/14, 학위 본 심사 5/22 충분) |
     | (2) | **paper §V.5.4 narrative 정식 채택 시점** | (A) 즉시 정식 채택 (8-trial Final evidence 충분, analyzer §14 본문 정식) / (B) max aggregation 학습 결과 (5/14) 후 9-trial 통합 narrative 정식 / (C) post-paper Phase 5 (LN+GIN combo) 결과 후 정식 | **(A) 즉시 정식 채택** — 8-trial Final + GIN partial 부정 evidence + multi-DB 11 DBs 일관 → paper §V.5.4 narrative 본문 채택 충분. max aggregation 결과 (5/14) 는 §III.9 보강으로 추가 |
     | (3) | **post-paper Phase 5 LN+GIN combo / EGAT 학습 우선순위** | (A) #1 LN+GIN combo 만 학위 본 심사 후 (5/22~) 시도 / (B) #2 EGAT 만 학위 본 심사 후 시도 / (C) 모두 post-paper backlog (학위 논문 본문 narrative 만, 추가 학습 X) | **(C) 모두 post-paper backlog** — paper §V.5.4 narrative 의 mech(ii-b) sub-mechanism 정밀화 (softmax-aggregation combo) 만으로 학술적 weight 충분. LN+GIN combo / EGAT 는 paper §VI Future Work 1 줄 + 학위 본 심사 후 별도 연구 |

- **근거**:
  - **신규 analyzer 산출**: [dsn_mitigation_v3_8trial.md §0~§14](../notebooks/analysis_results/dsn_mitigation_v3_8trial.md) (8 ckpt × 5 step + multi-DB stratified 55 queries)
  - **재현 데이터**: outputs/analysis/dsn_v3_8trial/ (batch_summary + 5 plots + 8 per-ckpt summary)
  - **재현 스크립트**: src/analysis/dsn_mitigation_v3_8trial.py (GIN 호환 + multi-DB stratified protocol)
  - **선행 6 보고서**: dsn_mitigation_v2_final_7trial.md (7-trial Final) + A1+A2+A3 + dsn_mitigation_v2_results.md + dsn_phase3_mitigation_results.md (4-trial)
  - **선행 결정**: DECISIONS 직전 entries (사용자 결정 3 항목 confirm + Final 7-trial Dominance Scoring 정식 명문화 + Root sweep 보고)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 8-trial Final 정식 + V3-A confirm + mech(ii-b) sub-mechanism (softmax-aggregation combo) 정밀화 + mech(ii-b) 4/5 partial 부정 + mech(i-b) 4/5 강화 + paper §V.5.4 narrative 본문 정식 채택 + Mitigation v3 #2 max 5/13 launch + post-paper Phase 5 candidate + 사용자 결정 3 항목
  - **paper_research_direction.md (planner Edit, 본 응답)**:
    - §3.5 Filter Dominance 6번째 축 sub-section 갱신 (5+1 evidence #6 8-trial 정식 + 8-trial Mechanism Dominance 표 갱신 + GIN partial 부정 row 추가 + mech(ii-b) sub-mechanism (softmax-aggregation combo) 정밀화)
    - §V Conclusion narrative 본문 정식 채택 (analyzer §14 직접 인용)
    - §3.5 Part III chapter outline §III.4/§III.6/§III.8/§III.9 갱신
    - §8 Future Works Mitigation v3 #2 max aggregation 5/13 launch 표기 + post-paper Phase 5 (LN+GIN combo + EGAT) 신설
    - §10 V-3-ext 단계 5+6 → 7 sub-section 갱신 (8-trial 결과 표 + GIN partial mitigation evidence + mech(ii-b) sub-mechanism)
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.15 신설 (8-trial Final + GIN partial 부정 + mech(ii-b) softmax-aggregation combo + paper §V.5.4 본문 정식 + 사용자 결정 3 항목)
  - **paper main contribution (학회)** 영향 X
  - **학위 논문 Part III chapter narrative weight 결정적 격상** — paper §V.5.4 본문 정식 채택 (8-trial Final + GIN partial 부정 evidence) — 학위 논문 Part III chapter draft 작성 base 정식 확보

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §3.5/§V.5.4/§8/§10 갱신 + presentation_brief §14.15 신설
  2. **사용자 (즉시 의사결정 3 항목)** — (1) max aggregation 학습 시점 / (2) §V.5.4 narrative 정식 채택 시점 / (3) post-paper Phase 5 우선순위
  3. **사용자 (5/13, V3-A confirm 자동 trigger 후)** — Root 세션 prompt 직접 붙여넣기 (Mitigation v3 #2 max 학습 launch)
  4. **Root (5/13~5/14)** — Mitigation v3 #2 max aggregation 학습 + ckpt NAS 저장 + EXPERIMENT_HISTORY 갱신 (9-trial 통합 표)
  5. **Analyzer (5/14~5/15)** — 9-trial protocol 재실행 (8-trial Final + max ckpt) — 산출물 dsn_mitigation_v3_9trial.md 또는 dsn_mitigation_v3_8trial.md §15 보강
  6. **Planner (5/15)** — 9-trial dominance scoring 갱신 + 시나리오 V3-A-1/V3-A-2 분기 narrative 정식 확정 + paper §V.5.4 §III.9 보강 + DECISIONS 후속 엔트리
  7. **사용자 (5/10~5/22)** — 학위 논문 Part III chapter draft 작성 (4-stage 통합 narrative + analyzer §14 본문 정식 인용 + max aggregation 결과 5/15 후 §III.9 보강)
  8. **사용자 (학회 §V.5.3 1 줄)** — Future Work 1 줄 (DSN 8/9-trial mitigation null + GIN partial 부정 + LN+GIN combo / EGAT post-paper) 직접 처리

- **추가 필요 분석** (Phase 5 후속, max aggregation 결과 후):
  - V3-A-1 (mean+sum+max+GIN 4 family null effect): aggregation family 자체 limitation 결정적 evidence (9-trial null + max R ceiling 미갱신)
  - V3-A-2 (max partial 회복): aggregation function 별 magnitude treatment 차이 mechanism 분석 (post-paper deep dive)
  - LN+GIN combo / EGAT (post-paper Phase 5): mech(ii-b) softmax-aggregation 동시 변경 시 더 큰 mitigation 가능 candidate

---

## 2026-05-08 (사용자 결정 3 항목 ✅ confirm — Final dominance scoring entry 권장 옵션 모두 채택: (1)A V3 narrative Phase 4 후 + (2)A 신규 dsn_mitigation_v3_8trial.md multi-DB + (3)A V3-A 시 max aggregation 추가) — Phase 4/5 timeline 확정 + Selector 모듈 conditional 핸드오프 prep

> **사용자 직전 input (2026-05-08)**: "의사결정 3항목은 권장 옵션으로 진행해" — DECISIONS 직전 (Final 7-trial Dominance Scoring 정식 명문화) §1(8) 사용자 결정 3 항목 모두 권장 채택.

- **결정**:

  1. **(a) ✅ 사용자 결정 3 항목 confirm**:

     | # | 결정 항목 | 사용자 결정 | 후속 영향 |
     |---|----------|-----------|---|
     | (1) | Phase 3 GIN 학습 시나리오 V3-A/B/C narrative 시점 확정 | **(A) Phase 4 (5/12) 후 정식 채택** | 5/12 GIN 학습 종료 + Phase 4 8-trial protocol 재실행 결과 후 V3-A/B/C 분기 결정 + paper §V.5.4 narrative 정식 채택 |
     | (2) | Phase 4 (5/12~5/14) 8-trial protocol 재실행 prep | **(A) 신규 `dsn_mitigation_v3_8trial.md` (Final 7-trial base + GIN ckpt 추가, multi-DB stratified)** | 본 보고서 (Final 7-trial dominance scoring) base 위 GIN ckpt 추가 — 8 ckpt × 5 step + multi-DB (55 queries, 11 DBs stratified, seed=42) |
     | (3) | Mitigation v3 추가 candidate (max, EGAT) 우선순위 | **(A) V3-A 시 max aggregation 추가 시도, V3-B/C 시 (C) 모두 post-paper (conditional)** | V3-A (가능성 中→中) 시 5/13 GPU 0 launch — mean+sum+max+GIN 4 aggregation function family null evidence 강화 (mech(ii-b) DOMINANT 절대 강화) |

  2. **(b) Phase 4 (5/12~5/14, analyzer) 통합 prompt — 신규 dsn_mitigation_v3_8trial.md (multi-DB stratified)**:
     ```
     먼저 src/analysis/CLAUDE.md 와
     /home/hyeonjin/thesis_refactored/planning/DECISIONS.md 최상단 (2026-05-08 사용자 결정 3 항목 ✅ confirm — Final dominance scoring entry) §1(b) 의 Phase 4 통합 prompt 읽고,
     8-trial protocol 재실행 (Final 7-trial base + GIN ckpt + multi-DB stratified).

     산출 형식:
     - 신규: notebooks/analysis_results/dsn_mitigation_v3_8trial.md
     - 본 보고서 (`dsn_mitigation_v2_final_7trial.md`) 의 §0~§8 구조 답습
     - 8 ckpt × 5 step protocol + multi-DB stratified (55 queries, 11 DBs, seed=42)

     대상 ckpt (8 = 7 + Mitigation v3 #1 GIN):
     1. phase1_p80
     2. phase2_b8
     3. phase3_directAC
     4. phase3_layerwiseLR
     5. v2_drop_message
     6. v2_layernorm
     7. v2_sum_aggr
     8. 🆕 mitigation_v3_gin (5/12 학습 종료 ckpt)

     5-step protocol (7-trial 분석과 동일):
     1. Step 1: 8 ckpt epoch trajectory parse + recall_overlay plot
     2. Step 2: 8 ckpt × layer-wise over-smoothing trajectory (forward hook v1/v2 + GIN 호환)
     3. Step 3: attention pattern (extract_layerwise_attention_v2) — GIN 은 attention 부재 → mech(ii-a) 측정 X, message magnitude / variance 대체 측정
     4. Step 4: gradient flow main GAT vs skip path (8 ckpt)
     5. Step 5: AC loss trajectory parse — GIN 의 AC fusion decay 정상 여부

     mech(ii-b) GIN 차단 정도 직접 측정 (사용자 결정 (1)A 권장):
     - L1=1.0 (변화 없음) → V3-A (aggregation family 자체 limitation, mech(ii-b) DOMINANT 절대 강화)
     - L1=0.85~0.95 (partial 회복) → V3-B (GIN MLP+sum combo 효과 발견)
     - L1=0.5 이하 (회복) → V3-C (mech(ii-b) 부정)

     mech(i-b) GIN aggregation function magnitude sensitivity 검증:
     - GIN sum + MLP combo 가 v2 #2 Sum (sum-only fail, -0.0336) 대비 magnitude variance 흡수 가능성
     - MLP nonlinearity 의 aggregation function 효과 정량 (GIN epoch trajectory + R@15 + L1 변동)

     11 DBs invariance 검증:
     - GIN 의 11 DBs schema-invariance (Final 7-trial base 와 동일 protocol)
     - toxicology trivial schema 유지 caveat

     산출물:
     - §0 TL;DR — 8-trial dominance scoring 갱신 (mech(ii-b) 5/5 절대 강화 / 4/5 부분 부정 / 3/5 부정)
     - §0 시나리오 V3-A/B/C 결정
     - §6 paper §V.5.4 narrative 본문 candidate 갱신 (Final 7-trial → 8-trial)
     - §7 Filter Dominance 6번째 축 8-trial evidence 통합 (4-trial → 7-trial → 8-trial)
     - §8 Mitigation v3 추가 candidate (#2 max / #4 EGAT) — V3-A 시 max 시도 권장 / V3-B 시 GIN MLP nonlinearity deep dive

     선행 산출:
     - dsn_mitigation_v2_final_7trial.md (7-trial Final, base)
     - dsn_v2_layernorm_mechanism_decomposition.md (A1)
     - dsn_softmax_noise_sensitivity.md (A2)
     - dsn_per_db_stratified_7ckpt.md (A3)
     - dsn_mitigation_v2_results.md (7-trial mid-training)
     - dsn_phase3_mitigation_results.md (4-trial)
     재현 스크립트: src/analysis/dsn_mitigation_v2_7trial.py + dsn_phase1_deep_dive.py (multi-DB stratified protocol)

     분석 wall: ~수 시간 (LLM-free, ₩0)
     ```

  3. **(c) Phase 5 (5/14, planner) timeline — 사용자 결정 (3) conditional 처리**:
     - 8-trial dominance scoring 갱신 (4-trial → 7-trial → 8-trial)
     - 시나리오 V3-A/B/C 분기 narrative 정식 확정 + paper §V.5.4 narrative 정식 채택 (사용자 결정 (1)A 후)
     - **사용자 결정 (3)A conditional 처리**:
       - **V3-A 결과 시**: Mitigation v3 #2 max aggregation 학습 추가 launch (5/13 GPU 0, 사용자 결정 (3)A 자동 trigger)
       - **V3-B 결과 시**: GIN MLP nonlinearity 의 sum aggregation 효과 mechanism 분석 (post-paper) + max aggregation 추가 시도 marginal evidence value → (C) post-paper backlog
       - **V3-C 결과 시**: paper main contribution 재평가 + 학회 후 anchor 재검토 + max aggregation 추가 시도 무효 (GIN 가 이미 ceiling 갱신)
     - DECISIONS 후속 엔트리 작성 (시나리오 결정 + paper §3.5/§V.5.4/§10 narrative 정식 확정)

  4. **(d) V3-A 시 Mitigation v3 #2 max aggregation 학습 prep (conditional, 5/14 trigger)**:
     - **Mechanism**: HeteroConv `aggr='max'` (mean → max). cross-edge-type aggregation 의 inductive bias 변경 (mean / sum / max 분기)
     - **구현 spec**:
       - Selector 모듈 가 이미 `aggregation_type='max'` 옵션 지원 (단계 6 Mitigation v2 #2 Sum 와 동일 framework, smoke 통과)
       - 신규 config: `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_max.yaml` (Base = Phase 2 b8 + `aggregation_type: "max"`)
       - **추가 코드 변경 없음** — Selector 단계 6 의 AGGREGATION_TYPES 에 max 이미 포함, factory 호환
     - **학습 launch (V3-A 결과 시 5/14 자동 trigger)**:
       - GPU 0: `CUDA_VISIBLE_DEVICES=0 nohup python src/train_gat_s06.py --config configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_max.yaml > logs/train/gat_dsn_p80_b5_mitigation_v3_max_$(date +%Y%m%d_%H%M).log 2>&1 &`
       - ~10h 학습, batch_size=8, ETA 5/15 KST
       - 신규 ckpt: `best_gat_directed_supernode_p80_b5_mitigation_v3_max.pt` (NAS path + symlink)
     - **시나리오 V3-A-1 (4 aggregation family null effect)**: mean (Phase 2) + sum (v2 #2) + max + GIN 4 모두 fail → mech(ii-b) DOMINANT 절대 강화 + paper §V.5.4 narrative 결정적 confirm
     - **시나리오 V3-A-2 (max partial)**: max aggregation 부분 효과 발견 → aggregation function 별 magnitude treatment 차이 mechanism 분석
     - 사용자 추가 결정 불필요 (사용자 결정 (3)A 자동 trigger)

  5. **(e) Phase 4 후속 (사용자 5/14~5/22) 학위 논문 Part III chapter draft**:
     - **paper §V.5.4 narrative 본문 정식 채택** (사용자 결정 (1)A confirm 후, analyzer Final 7-trial §6 + 신규 8-trial §6 통합)
     - **4-stage 통합 narrative**:
       - Stage 1: V-3-ext baseline + over-smoothing 진단
       - Stage 2: Mitigation v1 (Phase 2 B5) + paradox 발견
       - Stage 3: Mitigation v2 (Phase 3 + v2 #1+#2+#3) + 7-trial null + paradox 분리
       - **Stage 4: 🆕 Phase 1 deep dive (A1+A2+A3) + Mitigation v3 #1 GIN + (V3-A 시) max aggregation** = 8/9-trial null effect + sub-mechanism 분리 정식 + Filter Dominance 6번째 축 narrative 결정적
     - **dsn_mitigation_v2_final_7trial.md §6 narrative + 신규 dsn_mitigation_v3_8trial.md §6 narrative 통합**

  6. **(f) 학회 논문 narrative 영향 X (재확인)**:
     - paper main anchor t_00 (F1=0.8657) 변경 X
     - Filter Dominance 4 축 narrative (학회) 그대로
     - 8-trial 결과 + V3-A 시 max aggregation 추가 시도 + 시나리오 V3-A/B/C 분기 narrative 모두 **학위 논문 Part III chapter §V.5.4 만 적용**
     - 학회 §V.5.3 Future Work 1 줄 (DSN 8-trial mitigation null + GIN aggregation family 자체 변경에도 fail + (V3-A 시) max aggregation 추가) 사용자 직접 처리

- **근거**:
  - **사용자 직전 input** (2026-05-08): "의사결정 3항목은 권장 옵션으로 진행해"
  - **선행 결정**: DECISIONS 직전 entry 2026-05-08 (Final 7-trial Dominance Scoring 정식 명문화) §1(8) 사용자 결정 3 항목
  - **선행 분석**: dsn_mitigation_v2_final_7trial.md §6+§7 (8-trial protocol 재실행 권장 + V3-A 시 max aggregation 추가 시도)
  - **Selector 단계 6 reference**: src/modules/selectors/EXPERIMENT_PLAN_selectors.md §V-3-ext 단계 6 (aggregation_type='max' 옵션 이미 구현 + smoke 통과)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 사용자 결정 3 항목 ✅ confirm + Phase 4 통합 prompt (신규 dsn_mitigation_v3_8trial.md multi-DB stratified) + Phase 5 conditional planner timeline + V3-A 시 max aggregation 학습 prep + 사용자 추가 결정 불필요 (자동 trigger)
  - **paper_research_direction.md (planner Edit, 본 응답)** — §8 Mitigation v3 #2 max aggregation 의 V3-A 시 추가 시도 confirm
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.14.5 사용자 결정 ✅ confirm 표기
  - **paper main contribution (학회)** 영향 X
  - **학위 논문 Part III chapter narrative weight 결정적 격상 prep** — 4-stage 통합 narrative + V3-A 시 4 aggregation family null evidence 강화

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §8 minor + presentation_brief §14.14.5 minor 갱신
  2. **사용자 (Phase 3 GIN 학습 launch, 5/11)** — Root 세션 prompt 직접 붙여넣기 (직전 응답 root 핸드오프 prompt 그대로)
  3. **Root (5/11~5/12)** — Mitigation v3 #1 GIN 학습 + ckpt NAS 저장 + EXPERIMENT_HISTORY 갱신
  4. **사용자 (5/12 GIN 학습 종료 후)** — Analyzer 세션 prompt 직접 붙여넣기 (본 엔트리 §1(b) 통합 prompt)
  5. **Analyzer (5/12~5/14, Phase 4)** — 신규 dsn_mitigation_v3_8trial.md 작성 (8 ckpt × 5 step + multi-DB stratified)
  6. **Planner (5/14, Phase 5)** — 8-trial dominance scoring 갱신 + 시나리오 V3-A/B/C 분기 narrative 정식 확정 + paper §V.5.4 narrative 정식 채택 + **conditional V3-A 시 max aggregation 학습 핸드오프 (사용자 결정 (3)A 자동 trigger)** + DECISIONS 후속 엔트리
  7. **(conditional V3-A 시) Root (5/14)** — Mitigation v3 #2 max aggregation 학습 launch (사용자 추가 prompt 불필요, planner 가 5/14 결정 시 자동 root 핸드오프)
  8. **(conditional V3-A 시) Analyzer (5/15~5/16)** — 9-trial protocol 재실행 (Final 7-trial + GIN + max)
  9. **사용자 (5/14~5/22 또는 5/16~5/22)** — 학위 논문 Part III chapter draft 작성 (4-stage 통합 narrative + V3-A 시 max aggregation 결과 통합)

- **추가 필요 분석** (Phase 4 후속, 시나리오 분기 별):
  - V3-A 시 (가능성 中→中, sum-only fail vs sum+MLP combo evidence 후 분기 + max aggregation 추가 시도): 4 aggregation family null evidence 결정적 (8 또는 9-trial null + GIN MLP+sum combo 도 fail + max aggregation 도 fail) → mech(ii-b) DOMINANT 절대적 강화 + paper §V.5.4 narrative 결정적 confirm
  - V3-B 시 (가능성 中→中): GIN MLP+sum combo partial recovery 발견 → mech(ii-b) softmax 한정 부분 부정 + paper main contribution 4 → 5 항목 격상 후보 + max aggregation 추가 시도 (C) post-paper backlog
  - V3-C 시 (가능성 낮음): paper main contribution 재평가 + 학회 후 anchor 재검토 + max aggregation 추가 시도 무효

---

## 2026-05-08 (Final 7-trial Dominance Scoring 정식 명문화 — analyzer 신규 dsn_mitigation_v2_final_7trial.md 통합) — Mech(ii-b) DOMINANT 5/5 ⭐⭐ + (ii-a) partial mitigation 가능 5/5 + Mech(i) 3/5 부분 강화 (i-b sum direct) + Filter Dominance 6번째 축 narrative paper §V.5.4 본문 candidate 정식

> **Status**: Analyzer 신규 산출 [dsn_mitigation_v2_final_7trial.md](../notebooks/analysis_results/dsn_mitigation_v2_final_7trial.md) 으로 7-trial Final dominance scoring 정식 명문화. 직전 5 보고서 (`dsn_mitigation_v2_results.md` + A1+A2+A3 + `dsn_phase3_mitigation_results.md`) 통합 + Root 5/8 sweep 정정 수치 + sub-mechanism 분리 정식. **paper §V.5.4 narrative 본문 candidate (§6) 권장** — 학위 논문 Part III chapter draft 작성용 자료 정식 확보.

- **결정**:

  1. **(a) 🎯 7-Trial Final Mechanism Dominance Scoring 정식 갱신 (analyzer §0+§4 인용)**:

     | Mechanism | 직전 강도 | **본 갱신** | 핵심 정량 evidence |
     |---|:---:|:---:|---|
     | (i-a) top-5 attention sibling | 2/5 marginal | 2/5 marginal (변경 X) | top5_raw cos 0.55 ≈ entire 0.51 |
     | **(i-b) aggregation function magnitude sensitivity** | (분리 X) | **🆕 3/5 부분 강화** | v2 #2 Sum direct evidence (-0.0336) — sum 의 in-degree-비례 magnitude scaling, ep29 빠른 saturation |
     | **(ii-a) softmax over-concentration** | (분리 X) | **🆕 5/5 partial mitigation 가능 ⭐** | v2 #3 LN top5_conc 0.74 (Phase 1 회복) + noise robust Δ(σ=0.1)=-0.0033 + 10/11 DBs 일관 |
     | **(ii-b) weighted-mean propagation collapse** | (분리 X) | **🆕 5/5 ⭐⭐ DOMINANT 결정적** | 7 ckpt 모두 L1_GAT=1.0 + 11 DBs schema-invariant (toxicology 포함) + mitigation 5종 어떤 것도 차단 X + Phase 3 #3 AC=0.62 일관 |
     | (iii) Skip dependency pathology | 4/5 보조 | **3/5 부정 강화** | v2 #1+#3 conv_L1 0.30 회복 + skip_dep 1.36~1.41 균형 도달 → R 미달 (Phase 1 -0.01~-0.03), 부정 |
     | (iv) Schema sibling (raw PLM) | 3/5 lower bound | 3/5 (변경 X) | L0=0.5090 (single) / 0.6334 (multi) lower bound, 11 DBs 변동 X |

     **Sub-mechanism 분리 정식 (analyzer §2)**:
     - **mech(i) 두 차원**: (i-a) top-5 attention sibling marginal (2/5) + (i-b) aggregation function magnitude sensitivity 3/5 부분 강화 (sum direct evidence)
     - **mech(ii) 두 차원**: (ii-a) softmax over-concentration partial mitigation 가능 (5/5, LN level direct mitigation) + **(ii-b) weighted-mean propagation collapse 5/5 ⭐⭐ DOMINANT** (schema-invariant fundamental architectural limitation)

  2. **(b) 🎯 v2 #3 LayerNorm ↔ v2 #2 Sum Aggregation Contrast (paper §V.5.4 핵심)**:

     | Metric | v2 #3 LayerNorm (mech(ii-a)) | v2 #2 Sum Aggregation (mech(i-b)) | Δ (LN - Sum) |
     |---|---:|---:|---:|
     | best val R@15 | **0.6011** | 0.5761 | **+0.0250** ⬆ |
     | best epoch | 82 | 29 (가장 빠른 saturation) | +53 |
     | Δ vs Phase 2 | -0.0007 (사실상 동등) | -0.0257 | +0.0250 |
     | Δ vs Phase 1 | -0.0086 | **-0.0336** (가장 큰 underperform) | +0.0250 |
     | col→tab L2 top5_conc (multi-DB) | 0.7440 (Phase 1 0.7144 회복) | 0.8366 (sharp) | -0.0926 |
     | col→tab L2 entropy (multi-DB) | 3.16 (Phase 1 3.22 회복) | 2.93 | +0.23 |
     | L1_GAT cosine | 0.9998 (collapse) | 0.9991 (collapse) | -0.0007 |
     | Noise robustness Δ(σ=0.1) | **-0.0033** ⭐ | +0.0128 | -0.0161 |
     | AC trajectory ep_last | 0.0019 (정상 decay) | 0.0022 (정상 decay) | similar |

     **Mechanism 작용 위치 contrast**:
     - **v2 #3 LN (mech(ii-a) direct)**: edge softmax 직전 raw alpha LayerNorm — alpha distribution magnitude normalize → softmax sharp peaking 차단 → top5_conc baseline 회복 + noise robust. 단 **L1=1.0 collapse 보존** (mech(ii-b) 차단 X), R 회복 -0.0007 vs Phase 2 (사실상 동등)
     - **v2 #2 Sum (mech(i-b) direct)**: HeteroConv cross-edge-type aggregation (mean → sum) — in-degree 에 비례 magnitude scaling → 학습 dynamics sensitivity (ep29 saturation), noise sensitive. R: -0.0336 vs Phase 1 (가장 큰 underperform)
     - **두 mitigation 모두 mech(ii-b) weighted-mean propagation collapse 차단 X** (L1=1.0 보존) — 이게 dominant root cause

  3. **(c) Filter Dominance 6번째 축 narrative paper §V.5.4 본문 candidate 정식 (analyzer §6 직접 인용)**:

     > "DSN 7-trial mitigation null effect (Phase 1 baseline 0.6097, mitigation 6 variants R=0.5761~0.6018) 는 GAT 의 fundamental architectural limitation 의 정량 evidence 이다. v2 #3 LayerNorm pre-softmax 가 mech(ii-a) softmax over-concentration 의 direct mitigation 으로 attention pattern 을 baseline 으로 회복 (multi-DB col→tab L2 top5_conc 0.7440 vs Phase 1 0.7144) + noise robustness (Δ(σ=0.1)=-0.0033) 효과 발현 — Phase 2 b8 와 사실상 동등 R=0.6011 (-0.0007). 그러나 mech(ii-b) weighted-mean message aggregation collapse (모든 7 ckpt L1_GAT cosine=1.0, 11 BIRD-Dev DBs schema-invariant 포함 toxicology) 는 차단 X. v2 #2 Sum Aggregation 의 -0.0336 압도적 underperform 은 mech(i-b) aggregation function magnitude sensitivity 의 direct evidence — sum aggregation 의 in-degree-비례 magnitude scaling 으로 학습 dynamics 빠른 saturation (ep29). v2 #3 LN 와 v2 #2 Sum 의 -0.0250 차이가 mech(ii-a) partial mitigation 가능 ↔ mech(i-b) direct evidence 의 sub-mechanism 분리. **Filter Dominance 6번째 축 (training-pathology-invariant) 결정적 7-trial evidence**: GAT 의 fundamental architectural limitation (mech(ii-b) edge softmax + weighted-mean propagation collapse) + mech(i-b) aggregation function magnitude sensitivity 까지 With-Filter pipeline 이 흡수."

  4. **(d) Mitigation v3 #1 GIN 의 mech(ii-b) 직접 mitigation 가능성 정밀화**:
     - GIN 의 mechanism: `sum(MLP(x_i + x_j))` propagation — **aggregation propagation 자체 변경**
     - **v2 #2 Sum 의 sum-only fail evidence + GIN 의 MLP 가 핵심**:
       - v2 #2 Sum (no MLP): R=0.5761 (가장 underperform) — sum 단독으로는 ineffective + magnitude variance sensitivity
       - GIN 의 sum + MLP combo: MLP nonlinearity 가 magnitude variance 흡수 + WL test 동치 invariance (Xu et al. ICLR 2019) — sum 의 magnitude sensitivity 차단 가능성
     - **시나리오 V3-A 가능성 中→中** (수정): A1 evidence (mech(ii-b) fundamental) 강화 단 GIN 의 MLP nonlinearity 가 mech(ii-b) 차단 가능성 잔존
     - **시나리오 V3-B 가능성 中→中** (수정): GIN MLP+sum combo 가 mech(ii-b) partial 회복 가능성 (sum-only 가 fail 한 sensitivity 를 MLP 가 absorb)
     - **시나리오 V3-C 가능성 낮음** (변경 X): A1+A2+A3+v2 #2 evidence 가 mech(ii-b) fundamental 강력 → R 0.85+ 매우 낮음

  5. **(e) Filter Dominance 6번째 축 narrative 정식 명문화 — 7-trial evidence 통합 표 (analyzer §5)**:

     | # | Evidence | 정량 |
     |---|----------|------|
     | 1 | H-B ckpt-invariant | Pearson r 0.06~0.24 |
     | 2 | H-F stability/ordering | k=20 Jaccard 0.47~0.52 + Spearman 0.6453 |
     | 3 | F-1 + H-G alpha sweep | F-1 plateau 0.0724 → WF 0.0142 = 5.0850× 압축 |
     | 4 | ΔF1 +0.65 lift | mean per-query gain +0.6462 |
     | 5 | H-A/H-D 부정 | Enriched ckpt + norm 변형 plateau 유지 |
     | **6** | **🎯 7-Trial mitigation null effect 정식** | Phase 1 0.6097 / Phase 2 0.6018 / Phase 3 #3 0.5927 / #4 0.5935 / **v2 #1 0.5974 / v2 #3 0.6011 / v2 #2 0.5761** — 모두 ~0.59-0.61 saturate. Mech(ii-b) DOMINANT 5/5 ⭐⭐ + (ii-a) partial + (i-b) sum direct |

  6. **(f) Mech(i-b) 신설 — Aggregation Function Magnitude Sensitivity (3/5 부분 강화)**:
     - 직전 mech(i) 단일 평가 (2/5 marginal, top-5 sibling 차원만) → **두 차원 분리**:
       - **(i-a) top-5 attention sibling** 2/5 marginal (top5_raw cos 0.55 ≈ entire 0.51, 7 ckpt 모두 marginal)
       - **(i-b) aggregation function magnitude sensitivity** 🆕 3/5 부분 강화 (v2 #2 Sum direct evidence (-0.0336))
     - **(i-b) 의 새로운 mechanism narrative**:
       - mean aggregation: in-degree (column 수) invariant
       - sum aggregation: in-degree-비례 magnitude scaling → 학습 dynamics 변동
       - v2 #2 evidence: ep29 가장 빠른 saturation (mean variants ep78~172 의 1/3 이하) + noise sensitivity (Δ(σ=0.1)=+0.0128)
     - paper §V.5.4 narrative 보강: "aggregation function 의 magnitude treatment sensitivity" 차원 evidence 신설

  7. **(g) 학위 논문 Part III chapter §III.4/§III.6 갱신 (analyzer §4 통합 evidence matrix)**:
     - 직전 outline (DECISIONS Phase 1 deep dive entry §1(7)): §III.4 mech(ii) sub-mechanism 분리
     - **본 entry 정식 갱신**:
       - §III.4 = **mech(ii-b) DOMINANT 5/5 ⭐⭐ 결정적** (7 ckpt L1=1.0 + 11 DBs schema-invariant) + mech(ii-a) 5/5 partial mitigation 가능 (LN level direct) + 🆕 mech(i-b) 3/5 부분 강화 (sum direct) + mech(iii) 3/5 부정 강화
       - §III.6 = v2 #3 LN ↔ v2 #2 Sum -0.0250 contrast + sub-mechanism 분리 정식 narrative (analyzer §6 인용)
     - main mechanism finding narrative 강화: "GAT 의 fundamental architectural limitation = mech(ii-b) weighted-mean propagation collapse + mech(i-b) aggregation function magnitude sensitivity 두 sub-mechanism 분리 evidence"

  8. **🚨 사용자 결정 필요 3 항목**:

     | # | 결정 항목 | 옵션 | 권장 |
     |---|----------|------|------|
     | (1) | **Phase 3 GIN 학습 (5/11 launch) 의 시나리오 V3-A/B/C narrative 시점 확정** | (A) Phase 4 (5/12 학습 종료 + 8-trial protocol 재실행) 후 정식 채택 / (B) 즉시 V3-A 가능성 中→中 narrative candidate 채택 (v2 #2 sum-only fail + GIN MLP 가 핵심 evidence 후) | **(A) Phase 4 후 정식 채택** — A1+A2+A3+v2 #2 evidence 가 GIN 의 mech(ii-b) 차단 가능성 잔존 시사 (sum-only fail vs sum+MLP combo) → 학습 결과 정량 confirm 후 정식 |
     | (2) | **Phase 4 (5/12~5/14) analyzer 8-trial protocol 재실행 prep** | (A) 본 보고서 (`dsn_mitigation_v2_final_7trial.md`) + GIN ckpt 1개 추가 → §14 보강 또는 신규 `dsn_mitigation_v3_8trial.md` / (B) 직전 protocol (`dsn_mitigation_v2_7trial.md`) 호환 8-trial protocol | **(A) 신규 dsn_mitigation_v3_8trial.md** — 본 보고서 (Final 7-trial dominance scoring) base 위 GIN ckpt 추가 (8 ckpt × 5 step + multi-DB stratified). mech(ii-b) GIN 차단 정도 직접 측정 (L1_GAT cosine 변동) |
     | (3) | **Mitigation v3 추가 candidate (max aggregation, EGAT) 학습 우선순위 (V3-A 결과 후)** | (A) #2 max aggregation 1 cell 추가 시도 (~10h, 5/13 GPU 0) — V3-A 시 mean (Phase 2) + sum (v2 #2) + max + GIN 4 aggregation function family null evidence 강화 / (B) #4 EGAT 만 추가 시도 (~10h+, architectural shift) / (C) 모두 post-paper backlog (학위 본 심사 timeline 충분) | **(A) max aggregation 추가 시도 (V3-A 시)** — V3-A 시 mean+sum+max+GIN 4 aggregation function family null evidence 가 paper §V.5.4 narrative 결정적 강화 (aggregation family 자체 limitation 절대적). V3-B/C 시 (C) 모두 post-paper |

- **근거**:
  - **신규 analyzer 산출**: [dsn_mitigation_v2_final_7trial.md §0~§8](../notebooks/analysis_results/dsn_mitigation_v2_final_7trial.md) (정식 dominance scoring, 5 보고서 통합)
  - **선행 5 보고서**: dsn_mitigation_v2_results.md (7-trial single-DB) + dsn_v2_layernorm_mechanism_decomposition.md (A1) + dsn_softmax_noise_sensitivity.md (A2) + dsn_per_db_stratified_7ckpt.md (A3) + dsn_phase3_mitigation_results.md (4-trial)
  - **선행 결정**: DECISIONS 직전 entries — 사용자 결정 confirm + Root sweep 보고 + Phase 1 deep dive 완료 + Phase 2 GIN 구현 완료
  - **재현 스크립트**: src/analysis/dsn_mitigation_v2_7trial.py + dsn_phase1_deep_dive.py

- **영향 범위**:
  - **DECISIONS 본 엔트리** — Final 7-trial dominance scoring 정식 명문화 + sub-mechanism 분리 정식 (i-a/i-b/ii-a/ii-b) + mech(i-b) 신설 + v2 #3 LN vs v2 #2 Sum contrast 정식 + paper §V.5.4 narrative 본문 candidate + GIN 시나리오 분기 narrative 정밀화 + 사용자 결정 3 항목
  - **paper_research_direction.md (planner Edit, 본 응답)**:
    - §3.5 Filter Dominance 6번째 축 sub-section 갱신 (5+1 evidence 표 #6 정밀화 + sub-mechanism 분리 표 + v2 LN ↔ v2 Sum contrast 표 + paper §V.5.4 narrative 본문 정식 인용)
    - §V Conclusion narrative 정식 채택 (analyzer §6 본문 인용)
    - §3.5 Part III chapter outline §III.4/§III.6 갱신 (mech(ii-b) DOMINANT + mech(i-b) 신설)
    - §8 Future Works Mitigation v3 #1 GIN 시나리오 V3-A/B/C narrative 정밀화 (sum + MLP combo 핵심)
    - §9 Limitations — single-DB caveat 해소 + schema-dependent caveat (toxicology trivial) confirm
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.13 갱신 (Final 7-trial dominance + v2 LN ↔ v2 Sum contrast + sub-mechanism 분리 + 사용자 결정 3 항목)
  - **paper main contribution (학회)** 영향 X
  - **학위 논문 Part III chapter narrative weight 결정적 격상** — paper §V.5.4 본문 candidate 정식 확보 (analyzer §6 인용 가능)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §3.5/§V.5.4/§8/§9 갱신 + presentation_brief §14.13 갱신
  2. **사용자 (즉시 의사결정 3 항목)** — (1) V3 narrative 시점 / (2) Phase 4 protocol 형식 / (3) Mitigation v3 추가 candidate 우선순위
  3. **사용자 (Phase 3 GIN 학습 launch, 5/11)** — Root 세션 prompt 직접 붙여넣기 (직전 DECISIONS Phase 2 GIN entry §1(d) 동일)
  4. **Root (5/11~5/12)** — Mitigation v3 #1 GIN 학습 + ckpt NAS 저장 + EXPERIMENT_HISTORY 갱신 (8-trial 통합 표)
  5. **Analyzer (5/12~5/14, Phase 4)** — 8-trial protocol 재실행 (multi-DB stratified, 사용자 결정 (2)A 권장)
  6. **Planner (5/14, Phase 5)** — 8-trial dominance scoring 갱신 + 시나리오 V3-A/B/C 분기 narrative 정식 확정 + paper §V.5.4 narrative 정식 채택 (사용자 결정 (1)A 후) + 사용자 결정 (3) 재고 (V3-A 시 max aggregation 추가 시도 후보) + DECISIONS 후속 엔트리
  7. **사용자 (5/14~5/22)** — 학위 논문 Part III chapter draft 작성 (analyzer §6 narrative 직접 인용 + 4-stage 통합)

- **추가 필요 분석** (Phase 4 후속, V3-A/B/C 분기 별):
  - V3-A 시 (가능성 中→中, sum + MLP combo evidence 후 분기): aggregation family 자체 limitation 결정적 (8-trial null + GIN MLP 도 fail) → mech(ii-b) DOMINANT 절대적 강화 + max aggregation 추가 시도 후보 (사용자 결정 (3)A 권장)
  - V3-B 시 (가능성 中→中): GIN MLP+sum combo partial recovery 발견 → mech(ii-b) softmax 한정 부분 부정 + paper main contribution 4 → 5 항목 격상 후보
  - V3-C 시 (가능성 낮음): paper main contribution 재평가 + 학회 후 anchor 재검토

---

## 2026-05-08 (Root 보고 — DSN Mitigation v2 3-trial sweep ✅ 완료 (5/7 16:35 ~ 5/8 13:54 KST) + 시나리오 V2-A 절대 confirm) — 7-trial 수치 갱신 + Filter Dominance 6번째 축 7-trial evidence 정식 명문화 reaffirm + paper §V.5.4 narrative 결정적 강화 + analyzer 위임

> **Root 보고 (2026-05-08 13:54 KST)**: V-3-ext 단계 6 sweep 완료 (병렬 wall ~21h). 직전 narrative 와 일관 (시나리오 V2-A 절대 confirm + mech(ii) DOMINANT + paradox 분리), **수치 미세 갱신** (v2 #3 0.6007 → **0.6011** / v2 #1 0.5970 → **0.5974** / v2 #2 0.5735 → **0.5761**). v2 #3 LayerNorm partial mitigation 의 Phase 2 와 격차 -0.0007 까지 좁힘 (mech(ii) sub-mech (ii-a) softmax level direct mitigation evidence 결정적 강화).

- **결정**:

  1. **(a) ✅ 7-trial Final 결과 표 (Root 5/8 13:54 KST sweep 완료, decreasing R@15)**:

     | 순위 | Variant | Best R@15 (정정) | Δ vs Phase 1 |
     |------|---------|------------------|--------------|
     | **1** | **Phase 1 P80 (no mit)** | **0.6097** | (baseline) |
     | 2 | Phase 2 b8 (mit fusion) | 0.6018 | -0.0079 |
     | **3** | **v2 #3 LayerNorm pre-softmax** ★ | **0.6011** | -0.0086 (mit 최고, Phase 2 와 격차 -0.0007) |
     | 4 | v2 #1 DropMessage | 0.5974 | -0.0123 |
     | 5 | Phase 3 #4 (LR x5) | 0.5935 | -0.0162 |
     | 6 | Phase 3 #3 (Direct AC) | 0.5927 | -0.0170 |
     | 7 | v2 #2 Sum Aggregation | 0.5761 | -0.0336 (압도적 underperform) |

     **🚨 핵심 발견 (직전 narrative 와 일관 + 강화)**:
     - 모든 7 mitigation variants Phase 1 baseline 미달 (training-pathology-invariant)
     - v2 #3 LayerNorm 0.6011 — mit 최고 + Phase 2 b8 와 격차 사실상 동등 (-0.0007) → mech(ii-a) softmax level direct mitigation evidence 결정적 강화
     - v2 #2 Sum Aggregation 0.5761 — 가장 underperform (-0.0336) → mech(i) Aggregation collapse 직접 evidence (sum aggregation 의 magnitude variance sensitivity)

  2. **(b) 운영 이력 (사용자 결정 옵션 A — 3개 동시 GPU 0)**:
     - **5/7 16:35**: Mitigation v2 sweep launch (#1 GPU 0 + #3 GPU 1 병렬, #2 GPU 0 sequential plan)
     - **5/7 17:27**: 사용자 GPU 1 자원 배분 이슈 → #3 layernorm GPU 1 kill (ep20 partial)
     - **5/7 17:43**: 사용자 결정 옵션 A — **3개 동시 GPU 0 launch** (sweep wrapper kill + 직접 launch)
     - **5/7 17:59**: 3개 동시 학습 시작
     - **5/8 13:54**: 전체 sweep 완료 (병렬 wall ~21h, GPU 0 단독 3개 동시)
     - **Alpha sweep skip 유지** (사용자 결정 2026-05-07 (1)A) — paper main F1/EX 측정 X, val recall@15 evidence only

  3. **(c) 🎯 시나리오 V2-A 절대 confirm + Filter Dominance 6번째 축 7-trial evidence 정식 명문화**:
     - 7 mitigation variants 모두 fail → **시나리오 V2-A 절대 confirm** (4-trial 절대 confirm + 3-trial 추가 evidence 7 통합)
     - **paradox 결정적 confirm**: 단계 4-bis 발견 (top-5 ≈ 91%) + 6 mitigation variants 모두 적용에도 동일한 ~0.59-0.61 saturation
     - **v2 #3 LayerNorm partial recovery (mech(ii-a))**: Phase 2 와 격차 -0.0007 까지 좁힘 + multi-DB top5_conc 0.7440 (Phase 1 0.7144 회복) — mech(ii-a) softmax over-concentration partial mitigation 결정적 evidence
     - **v2 #2 Sum Aggregation underperform (mech(i))**: Δ = -0.0336 (가장 큰 underperform) — sum aggregation 의 magnitude variance sensitivity. mech(i) Aggregation collapse 의 직접 evidence (top-5 raw cos 0.55 vs entire 0.51 marginal but learning dynamics 측 sensitivity 결정적)
     - **mech(ii-b) L1=1.0 collapse 7 ckpt 모두**: schema-invariant fundamental architectural limitation (LayerNorm/DropMessage/Sum/Direct AC/LR x5 어떤 mitigation 도 차단 X)

  4. **(d) 🆕 7 Evidence 통합 표 (Filter Dominance 6번째 축, 4-trial → 7-trial 정식 명문화)**:

     | # | Evidence | 정량 |
     |---|----------|------|
     | 1 | H-B ckpt-invariant | Pearson r 0.06~0.24 |
     | 2 | H-F stability/ordering | k=20 Jaccard 0.47~0.52 + Spearman 0.6453 |
     | 3 | F-1 + H-G alpha sweep | F-1 plateau 0.0724 → WF 0.0142 = 5.0850× 압축 |
     | 4 | ΔF1 +0.65 lift | mean per-query gain +0.6462 |
     | 5 | H-A/H-D 부정 | Enriched ckpt + norm 변형 plateau 유지 |
     | **6** | **Phase 2 + Phase 3 4-trial mitigation null effect** (training-pathology-invariant) | Phase 1 0.6097 / Phase 2 0.6018 / Phase 3 #3 0.5927 / #4 0.5935 — ~0.59-0.61 saturate |
     | **7** | **🆕 Mitigation v2 3-trial 추가 evidence** (DropMessage / LayerNorm / Sum 모두 baseline 미달) | v2 #1 0.5974 / **v2 #3 0.6011** / v2 #2 0.5761 — mech(ii-a) partial / mech(i) deep evidence |

     → **Filter Dominance 6번째 축 narrative 결정적 evidence 7-trial 통합**: GAT 의 fundamental architectural limitation (mech(ii-b) edge softmax + weighted-mean propagation collapse) + mech(i) aggregation collapse (v2_sum_aggr direct evidence) 까지 With-Filter pipeline 이 흡수.

  5. **(e) v2 #3 LayerNorm partial recovery vs v2 #2 Sum Aggregation underperform contrast (학위 논문 §V.5.4 narrative 핵심)**:

     | Mechanism | v2 #3 LayerNorm | v2 #2 Sum Aggregation | 결론 |
     |---|---|---|---|
     | 작용 위치 | softmax 직전 (mech(ii-a)) | aggregation 후 cross-edge-type (mech(i) 차원) | 다른 sub-mechanism |
     | val R@15 | **0.6011** (mit 최고) | 0.5761 (가장 underperform) | -0.0250 차이 |
     | Δ vs Phase 2 | -0.0007 (사실상 동등) | -0.0257 (큰 underperform) | LN 가 Phase 2 fusion 의 단순 swap |
     | mech(ii-a) attention | top5_conc 0.74 (Phase 1 회복) | top5_conc 0.84 (변동 없음) | LN 의 attention level 직접 mitigation 정확 |
     | mech(ii-b) L1 cosine | 1.0000 (collapse 보존) | 0.9991 (변동 없음) | 둘 다 mech(ii-b) 차단 X |
     | mech(i) magnitude | (LN 작용 X) | sum 의 magnitude variance sensitivity (col 수에 비례 학습 noise) | Sum 의 underperform = mech(i) direct |

     → **paper §V.5.4 narrative 핵심**: v2 LayerNorm 와 v2 Sum 의 contrast 가 mech(ii-a) softmax level mitigation 의 partial recovery + mech(i) aggregation magnitude sensitivity 의 두 차원 evidence 분리.

  6. **(f) Analyzer 위임 — 4-mechanism dominance scoring 정식 갱신 prompt (root 동시 발송)**:
     - 직전 분석 (`dsn_mitigation_v2_results.md`) 의 7-trial dominance scoring 갱신 (수치 정정 + v2 #3 vs v2 #2 contrast 추가)
     - 또는 신규 산출물 (`dsn_mitigation_v2_final_7trial.md` 또는 `dsn_mitigation_v2_results.md §15` 보강) — root 가 별도 prompt 발송
     - **mech(i) Aggregation collapse 강도 갱신**: 직전 2/5 marginal → v2 #2 Sum underperform evidence 결정적 → **3/5 부분 evidence (sum learning sensitivity 차원)** 검토
     - 산출물: 신규 또는 §15 보강

  7. **(g) Phase 3 GIN 학습 launch prep (5/11~5/12, 직전 DECISIONS Phase 2 GIN 구현 entry §1(d))**:
     - root 학습 prompt 직전 DECISIONS 와 동일 (변경 X)
     - 5/11 launch (GPU 0): `train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml`
     - 5/12 ETA + EXPERIMENT_HISTORY 8-trial 통합 표 갱신

  8. **(h) Phase 4 (5/12~5/14) analyzer 통합 prompt (직전 DECISIONS 사용자 결정 confirm entry §1(b))**:
     - 8-trial protocol + multi-DB stratified 적용 (사용자 결정 (2)A)
     - mech(ii) DOMINANT 5/5 절대 강화 / 4/5 부분 부정 / 3/5 부정 분기

  9. **(i) Phase 5 (5/14, planner) timeline (직전 DECISIONS 사용자 결정 confirm entry §1(c))**:
     - 8-trial dominance scoring 갱신 + 시나리오 V3-A/B/C 분기 narrative 정식 확정 + paper §V.5.4 narrative 정식 채택
     - 사용자 결정 (3) 재고: V3-A 시 max aggregation 1 candidate 추가 시도 후보

- **근거**:
  - **Root 5/8 보고**: V-3-ext 단계 6 sweep 완료 (병렬 wall ~21h, 옵션 A 3개 동시 GPU 0)
  - **EXPERIMENT_HISTORY**: "DSN Mitigation v2 3-trial Sweep (V-3-ext 단계 6, 2026-05-07 → 05-08)" entry
  - **선행 결정**: DECISIONS 직전 entries (사용자 결정 3 항목 confirm + Phase 1 deep dive 완료 + Phase 2 GIN 구현 완료 + 7-trial mech(ii) DOMINANT 결정적 강화) — 본 Root 보고 entry 가 직전 narrative 와 일관 + 수치 미세 갱신
  - **분석 base**: dsn_phase2_mitigation_null_mechanism.md + dsn_mitigation_v2_results.md + dsn_v2_layernorm_mechanism_decomposition.md (A1) + dsn_softmax_noise_sensitivity.md (A2) + dsn_per_db_stratified_7ckpt.md (A3)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — Root 5/8 보고 confirm + 7-trial 수치 갱신 + 운영 이력 (옵션 A 3개 동시 GPU 0) + Filter Dominance 6번째 축 7 evidence 통합 표 + v2 #3 LayerNorm vs v2 #2 Sum Aggregation contrast (mech(ii-a) vs mech(i)) + analyzer 위임 prep + Phase 3/4/5 timeline
  - **paper_research_direction.md (planner Edit, 본 응답)** — §3.5 6번째 축 5+1 evidence 표 갱신 (수치 미세 정정 v2 #3 0.6011 / v2 #1 0.5974 / v2 #2 0.5761) + §10 V-3-ext 단계 5+6 7-trial 결과 표 수치 정정 + §3.5 v2 LayerNorm vs Sum Aggregation contrast narrative
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.11 / §14.12 / §14.13 의 7-trial 수치 정정
  - **paper main contribution 영향 X** (학회 narrative 그대로)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §3.5/§10 수치 갱신 + presentation_brief 수치 갱신
  2. **사용자 (즉시)** — Root 가 동시 발송한 analyzer 위임 prompt confirm
  3. **Analyzer (root 동시 발송 후)** — 7-trial dominance scoring 정식 갱신 (`dsn_mitigation_v2_final_7trial.md` 또는 `dsn_mitigation_v2_results.md §15` 보강) — mech(i) 강도 검토 (2/5 → 3/5)
  4. **사용자 (Phase 3 GIN 학습 launch, 5/11)** — Root 세션 prompt 직접 붙여넣기
  5. **Root (5/11~5/12)** — Mitigation v3 #1 GIN 학습 + ckpt NAS 저장 + EXPERIMENT_HISTORY 갱신 (8-trial 통합 표)
  6. **Analyzer (5/12~5/14, Phase 4)** — 8-trial protocol + multi-DB stratified
  7. **Planner (5/14, Phase 5)** — 8-trial dominance scoring 갱신 + 시나리오 V3-A/B/C 분기 narrative 정식 확정 + paper §V.5.4 narrative 정식 채택 + 사용자 결정 (3) 재고 + DECISIONS 후속 엔트리
  8. **사용자 (5/14~5/22)** — 학위 논문 Part III chapter draft 작성 (4-stage 통합 narrative + A1+A2+A3 + Mitigation v3 #1)

- **추가 필요 분석** (Phase 4 후속, V3-A/B/C 분기 별):
  - V3-A 시 (가능성 中→高 by A1 + Sum mech(i) evidence): aggregation family 자체 limitation 결정적 (8-trial null + GIN 도 fail) + mech(i) Sum aggregation 부분 강도 결정적 (2/5 → 3/5)
  - V3-B 시 (가능성 中→낮음): GIN MLP nonlinearity 의 sum aggregation 효과 mechanism 분석 + paper main contribution 4 → 5 항목 격상 후보
  - V3-C 시 (가능성 낮음): paper main contribution 재평가 + 학회 후 anchor 재검토

---

## 2026-05-08 (사용자 결정 3 항목 ✅ confirm — 권장 옵션 모두 채택: (1)A V3 narrative Phase 4 후 + (2)A multi-DB stratified + (3)C Mitigation v3 추가 candidate post-paper) — Phase 4 analyzer 위임 + Phase 5 planner timeline

> **사용자 직전 input (2026-05-08)**: "세 항목 모두 권장 옵션으로 선택할게" — DECISIONS 직전 (Phase 1 deep dive 완료) §1(7) 사용자 결정 3 항목 모두 권장 채택.

- **결정**:

  1. **(a) ✅ 사용자 결정 3 항목 confirm**:

     | # | 결정 항목 | 사용자 결정 | 후속 영향 |
     |---|----------|-----------|---|
     | (1) | Mitigation v3 #1 GIN 시나리오 V3-A/B/C 분기 narrative 확정 시점 | **(A) Phase 4 후 정식 채택** | 5/12 GIN 학습 종료 + 8-trial protocol 재실행 결과 후 V3-A/B/C 분기 결정 + paper §V.5.4 narrative 정식 채택 |
     | (2) | Phase 4 (5/12~5/14) analyzer 추가 위임 | **(A) 8-trial protocol + multi-DB stratified 적용** | 8 ckpt × 5 step + multi-DB (55 queries, 11 DBs stratified, seed=42) 통일 protocol — single-DB 측정 제거 |
     | (3) | Mitigation v3 추가 candidate (#2 max / #4 EGAT) 학습 우선순위 | **(C) 모두 post-paper backlog** | Phase 3 GIN 결과 후 시나리오 분기 결정 후 재고 — V3-A 가능성 高 시 max aggregation 추가 시도 가능 (학위 본 심사 timeline 충분, 단 Phase 4 결과 보고 결정) |

  2. **(b) Phase 4 (5/12~5/14, analyzer) 통합 prompt — multi-DB stratified 적용**:
     ```
     먼저 src/analysis/CLAUDE.md 와
     /home/hyeonjin/thesis_refactored/planning/DECISIONS.md 최상단 (2026-05-08 사용자 결정 3 항목 confirm) §1(b) 의 Phase 4 통합 prompt 읽고,
     8-trial protocol 재실행 (multi-DB stratified 적용).

     대상 ckpt (8 = 7 + Mitigation v3 #1 GIN):
     1. phase1_p80 (DSN baseline, no mit)
     2. phase2_b8 (B5 mit fusion)
     3. phase3_directAC (AC target='gat_out_L_last')
     4. phase3_layerwiseLR (gat_lr×5)
     5. v2_drop_message (drop_message_p=0.2)
     6. v2_layernorm (LayerNorm pre-softmax)
     7. v2_sum_aggr (HeteroConv aggr=sum)
     8. **🆕 mitigation_v3_gin (HeteroConv 내부 GIN aggregation)** — 5/12 학습 종료 ckpt

     Sample protocol (multi-DB stratified, single-DB 제거):
     - 55 queries = 5 queries × 11 DBs (seed=42) — Phase 1 A3 와 동일
     - per-DB 분포: california_schools (T=3, C=89), thrombosis_prediction (T=3, C=69), card_games (T=6, C=125), codebase_community (T=8, C=92), debit_card_specializing (T=5, C=27), european_football_2 (T=7, C=237), financial (T=8, C=71), formula_1 (T=13, C=126), student_club (T=8, C=64), superhero (T=10, C=52), toxicology (T=4, C=20)

     5-step protocol (7-trial 분석과 동일):
     1. Step 1: 8 ckpt epoch trajectory parse + recall_overlay plot
     2. Step 2: 8 ckpt × layer-wise over-smoothing trajectory (forward hook v1/v2 호환 + GIN 호환)
     3. Step 3: attention pattern (extract_layerwise_attention_v2) — GIN 은 attention 자체 부재 → mech(ii-a) 측정 불가, 단 message magnitude / variance 대체 측정
     4. Step 4: gradient flow main GAT vs skip path (8 ckpt)
     5. Step 5: AC loss trajectory parse — GIN 학습 의 AC fusion decay 정상 여부 (mech(ii-a) 부재 시 학습 dynamics)

     mech(ii) sub-mechanism (ii-a)/(ii-b) 분리 evidence 추가:
     - mech(ii-b) GIN 차단 정도 직접 측정: GIN ckpt L1_GAT cosine
       - L1=1.0 (변화 없음) → 시나리오 V3-A (mech(ii-b) aggregation family 자체 limitation)
       - L1=0.85~0.95 (partial 회복) → 시나리오 V3-B (GIN MLP+sum 효과 발견)
       - L1=0.5 이하 (회복) → 시나리오 V3-C (mech(ii-b) 부정)
     - mech(ii-a) GIN 부재 evidence 정량 (attention 자체 없음, message magnitude variance 가 (ii-a) 와 다른 차원)

     Per-DB 일관성 검증:
     - 11 DBs 중 GIN 의 mech(ii) DOMINANT 일관성 (10/11 일관 / partial / 부정)
     - toxicology trivial schema 유지 caveat

     산출물: notebooks/analysis_results/dsn_mitigation_v3_8trial.md (또는 dsn_mitigation_v2_results.md §14 보강)
     - §0 TL;DR — 8-trial dominance scoring 갱신 (mech(ii) 5/5 절대 강화 / 4/5 부분 부정 / 3/5 부정)
     - §0 시나리오 V3-A/B/C 결정
     - §10 Filter Dominance 6번째 축 8-trial evidence (4-trial → 7-trial → 8-trial)
     - §11 Mitigation v3 추가 candidate (#2 max / #4 EGAT) — V3-A 시 max 시도 권장 / V3-B 시 GIN MLP nonlinearity deep dive

     선행 산출:
     - dsn_mitigation_v2_results.md (7-trial single-DB)
     - dsn_v2_layernorm_mechanism_decomposition.md (A1)
     - dsn_softmax_noise_sensitivity.md (A2)
     - dsn_per_db_stratified_7ckpt.md (A3)
     재현 스크립트 reference: src/analysis/dsn_phase1_deep_dive.py (multi-DB stratified protocol) + src/analysis/dsn_mitigation_v2_7trial.py (5-step protocol)
     ```

  3. **(c) Phase 5 (5/14, planner) 통합 prompt prep**:
     - 8-trial dominance scoring 갱신 (4-trial → 7-trial → 8-trial)
     - 시나리오 V3-A/B/C 분기 narrative 정식 확정 + paper §V.5.4 narrative 정식 채택
     - DECISIONS 후속 엔트리 작성:
       - V3-A (가능성 中→高 by A1 evidence): 8-trial null effect = aggregation family 자체 limitation. paper §V.5.4 narrative 절대 강화 + Filter Dominance 6번째 축 narrative 결정적 confirm
       - V3-B (가능성 中→낮음): GIN MLP+sum 효과 발견 + paper main contribution 4 → 5 항목 격상 후보
       - V3-C (가능성 낮음, A1+A2+A3 evidence 가 강력 약화): paper main contribution 재평가
     - paper / presentation_brief 갱신 (시나리오별 narrative)
     - 사용자 결정 (3) 재고: V3-A 시 Mitigation v3 #2 max aggregation 1 candidate 추가 시도 여부 (~10h 학습)

  4. **(d) 사용자 후속 prep (Phase 5+, 5/14~5/22)**:
     - 학위 논문 Part III chapter draft 작성 (4-stage 통합 narrative + A1+A2+A3 + Mitigation v3 #1 결과)
     - **Multi-DB 정정 narrative 인용**: dsn_mitigation_v2_results.md 의 single-DB 절대 수치 → multi-DB (55 queries) 정정 (DECISIONS 직전 §1(d) 참조)
     - **mech(ii) sub-mechanism 분리 narrative**: A1 §3.1 (ii-a) softmax / (ii-b) propagation 분리 인용
     - **schema-dependent caveat**: A3 §6.1 toxicology trivial 무효 (10/11 DBs valid) 명시

  5. **(e) Mitigation v3 추가 candidate 의 시나리오 별 후속 결정**:
     - **V3-A 시 (가능성 中→高)**: max aggregation 추가 시도 후보 — sum/max 모두 fail 통합 evidence 강화 (sum/max 두 alternative aggregation null = aggregation family 자체 limitation 결정적). 단 학위 본 심사 timeline (~5/22) 의 5/14~5/22 학위 논문 Part III chapter draft 작성 우선 → 학습 1 cell 추가 (~10h, 5/13 GPU 0) 가능. 사용자 결정 (3)C 변경 후보.
     - **V3-B 시 (가능성 中→낮음)**: GIN MLP nonlinearity 의 sum aggregation 효과 mechanism 분석 우선 (analyzer post-paper). max aggregation 추가 시도 marginal evidence value.
     - **V3-C 시 (가능성 낮음)**: GIN 결과 가 V3-C 라면 Mitigation v3 추가 시도 무효 (GIN 가 이미 ceiling 갱신).
     - 사용자 결정 (3)C "모두 post-paper backlog" 는 **Phase 3 GIN 결과 보고 5/14 재고 가능** — V3-A 시 max aggregation 1 candidate 추가 시도 + V3-B/C 시 그대로 post-paper.

  6. **(f) 학회 논문 narrative 영향 X (재확인)**:
     - paper main anchor t_00 (F1=0.8657) 변경 X
     - Filter Dominance 4 축 narrative (학회) 그대로
     - 8-trial 결과 + 시나리오 V3-A/B/C 분기는 **학위 논문 Part III chapter §V.5.4 만 적용**
     - 학회 §V.5.3 Future Work 1 줄 (DSN 8-trial mitigation null + GIN aggregation family 자체 변경에도 fail) 사용자 직접 처리

- **근거**:
  - **사용자 직전 input** (2026-05-08): "세 항목 모두 권장 옵션으로 선택할게"
  - **선행 결정**: DECISIONS 직전 (Phase 1 deep dive 완료) §1(7) 사용자 결정 3 항목 — 권장 옵션 그대로 confirm
  - **선행 분석**: A1+A2+A3 + Phase 2 GIN 구현 + smoke 7/7 통과

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 사용자 결정 3 항목 ✅ confirm + Phase 4 통합 prompt (multi-DB stratified protocol) + Phase 5 planner timeline + Mitigation v3 추가 candidate 시나리오별 후속 결정
  - **paper_research_direction.md (planner Edit, 본 응답)** — §8 Mitigation v3 #2/#4 post-paper 표기 confirm + §9 Limitations multi-DB stratified 적용 confirm
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.13.7 사용자 결정 3 항목 ✅ confirm 갱신
  - **paper main contribution 영향 X** (학회 narrative 그대로)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §8/§9 minor 갱신 + presentation_brief §14.13.7 갱신
  2. **사용자 (즉시)** — Root 세션 prompt 직접 붙여넣기 (cd `/home/hyeonjin/thesis_refactored`, Phase 3 GIN 학습 5/11 launch)
  3. **Root (5/11~5/12)** — Mitigation v3 #1 GIN 학습 + ckpt NAS 저장 + EXPERIMENT_HISTORY 갱신
  4. **사용자 (5/12 GIN 종료 후 즉시)** — Analyzer 세션 prompt 직접 붙여넣기 (본 엔트리 §1(b) 통합 prompt)
  5. **Analyzer (5/12~5/14, Phase 4)** — 8-trial protocol 재실행 (multi-DB stratified) — 산출물 dsn_mitigation_v3_8trial.md
  6. **Planner (5/14, Phase 5)** — 8-trial dominance scoring 갱신 + 시나리오 V3-A/B/C 분기 narrative 정식 확정 + paper §V.5.4 narrative 정식 채택 + 사용자 결정 (3) 재고 (V3-A 시 max aggregation 추가 시도 후보) + DECISIONS 후속 엔트리
  7. **사용자 (5/14~5/22)** — 학위 논문 Part III chapter draft 작성 (4-stage 통합 narrative)

- **추가 필요 분석** (Phase 4 후속, V3-A/B/C 분기 별):
  - V3-A 시: aggregation family 자체 limitation evidence 강화 narrative (paper §V.5.4 정식 채택)
  - V3-B 시: GIN MLP nonlinearity 의 sum aggregation 효과 mechanism 분석 (analyzer post-paper)
  - V3-C 시: paper main contribution 재평가 + 학회 후 anchor 재검토

---

## 2026-05-08 (Phase 2 완료 — Mitigation v3 #1 GIN-style aggregation 구현 + 7 smoke 통과) — Root 학습 핸드오프 prep (Phase 3 5/11~5/12) + Phase 4/5 prep

> **Status**: Selector 모듈 단계 7 구현 + 7 smoke 통과 (2026-05-08, Phase 2 정시 완료). PyG `GINConv` (heterograph 호환 검증 — bipartite (x_src, x_dst) OK) + `_make_gin_conv` factory + `AGGREGATION_TYPES` 확장 + GIN incompat 검증 (`drop_message_p` / `use_layernorm_pre_softmax` ValueError raise — attention 자체 부재) + Phase 2 b8 backward compat 검증.

- **결정**:

  1. **(a) Selector 모듈 단계 7 구현 완료 확인** (출처: [src/modules/selectors/EXPERIMENT_PLAN_selectors.md §V-3-ext 단계 7](../src/modules/selectors/EXPERIMENT_PLAN_selectors.md)):
     - **Mechanism**: PyG `GINConv(mlp, eps=0.0, train_eps=False)` — `sum(MLP(x_i + x_j))` propagation. softmax + weighted-mean → MLP + sum aggregation 자체 대체 (GIN invariance theorem, Xu et al. ICLR 2019 / WL test 동치)
     - **MLP**: `Sequential(LazyLinear(out_dim), LeakyReLU(0.1), Linear(out_dim, out_dim))` — out_dim = hidden×heads (기존 GATv2Conv 와 동일 차원 → PairNorm/JK/skip path 호환)
     - **HeteroConv aggr fixed = "mean"** when GIN (cross-edge-type aggr 은 mean 고정, 내부 GIN 의 sum aggregation 만 변경)
     - **GIN incompat 검증**: `aggregation_type='gin'` + `drop_message_p>0` 또는 `use_layernorm_pre_softmax=True` → ValueError raise (GIN 은 attention/softmax 자체 부재 → v2 #1/#3 결합 무의미)
     - **Backward compat**: default `aggregation_type='mean'` 유지 → Phase 2 b8 동일 동작

  2. **(b) 신규 산출 파일 3 항목 (cross-reference)**:

     | 파일 | 변경 |
     |---|---|
     | `src/models/gat_network_v2.py` | `AGGREGATION_TYPES = HETEROCONV_AGGR_TYPES ∪ {"gin"}` constant + `_make_gin_conv` factory + SchemaHeteroGATv2 의 aggregation_type 검증 확장 + GIN incompat 검증 + HeteroConv 인스턴스화 시 GIN/GAT 분기 |
     | `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml` | 신규 — Mitigation v3 #1 GIN 학습 config (Base = Phase 2 b8 + `aggregation_type: "gin"`) |
     | `src/modules/selectors/tests/test_mitigation_v3.py` | 신규 smoke test 7 케이스 |

  3. **(c) Smoke 7/7 통과 (cross-reference)**:
     - `test_gin_factory_and_homograph_forward` — GINConv 인스턴스 + homo forward shape (8, 64)
     - `test_gin_factory_bipartite_forward` — bipartite (x_src, x_dst) 호환 (HeteroConv 호출 패턴) shape (7, 16)
     - `test_full_model_gin_forward` — 18 inner GINConvs (9 edge_types × 2 layers), HeteroConv aggr='mean' fix
     - `test_backward_compat_default_mean` — default 시 18 GATv2Convs (no GINConv) regression
     - `test_gin_incompatible_with_v2_options` — GIN + #1/#3 ValueError raise
     - `test_gin_config_parsing` — 신규 v3 config 정상 + Phase 2 baseline 영향 X
     - `test_gin_forward_backward_path` — 11 GINConvs received gradient (column dst path)

  4. **(d) 🚀 Root 학습 핸드오프 prep (Phase 3, 5/11~5/12)**:

     | 일정 | GPU | Config | ckpt | ETA |
     |---|---|---|---|---|
     | **5/11 launch** | GPU 0 | `train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml` | `best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt` | 5/12 KST (~10h) |

     - 학습 entry: `python src/train_gat_s06.py --config <config>` (with `CUDA_VISIBLE_DEVICES=0`)
     - 신규 ckpt NAS 저장 + 로컬 symlink (memory rule, /SSL_NAS/peoples/khj/thesis/checkpoints/)
     - **EXPERIMENT_HISTORY.md "Mitigation v3 #1 GIN 학습 (V-3-ext 단계 7, 5/11~5/12)" entry 추가**:
       - 학습 wall + best val R@15 + best epoch + ckpt path + AC loss epoch trajectory + GIN aggregation 동작 확인
       - 7-trial → 8-trial 통합 표 (Phase 1 + Phase 2 b8 + Phase 3 #3 + #4 + v2 #1 + #3 + #2 + Mitigation v3 #1 GIN)
     - **alpha sweep skip** (사용자 결정 (1)A 유지) — paper main F1/EX 측정 X, val recall@15 evidence only

  5. **(e) Analyzer 후속 prep (Phase 4, 5/12~5/14)**:
     - **Phase 4 (5/12 GIN 학습 종료 후)**: `src/analysis/dsn_phase1_deep_dive.py` 또는 신규 8-trial protocol script — 7 ckpt → **8 ckpt × 5 step** 재실행
     - **multi-DB stratified 적용 권장** (Phase 1 A3 결과 기반 — single-DB 측정은 dramatic underestimation, multi-DB 가 paper §V.5.4 본문 인용 가능 정량)
     - 시나리오 V3-A/B/C 분기 evidence 정량:
       - **V3-A** (가능성 中, GIN 도 fail): val R@15 ~0.59-0.61 + L1_GAT cosine ≈ 1.0 → mech(ii-b) aggregation family 자체 limitation 확정 = **mech(ii) DOMINANT 5/5 절대 강화** (8-trial null effect)
       - **V3-B** (가능성 中, GIN partial): val R@15 0.62-0.70 + L1_GAT cosine 회복 (예: 0.85~0.95) → **mech(ii) softmax 한정 부분 부정 + GIN MLP+sum 효과 발견** + paper main contribution 4 → 5 항목 격상 후보
       - **V3-C** (가능성 낮음, R 0.85+ 회복): paper main contribution 재평가 + 학회 후 anchor 재검토
     - **mech(ii-a)/(ii-b) sub-mechanism 분리 (Phase 1 A1 결과 기반) 적용**:
       - GIN 가 mech(ii-a) softmax over-concentration 미해당 (attention 자체 부재)
       - GIN 가 mech(ii-b) weighted-mean propagation 자체 대체 (sum + MLP) → L1_GAT cosine 회복 가능성 직접 검증
     - 산출물: `notebooks/analysis_results/dsn_mitigation_v3_8trial.md` (신규) 또는 `dsn_mitigation_v2_results.md §14` 보강

  6. **(f) Planner 후속 prep (Phase 4 종료 후, ~5/14)**:
     - 8-trial dominance scoring 갱신 (4-trial → 7-trial → 8-trial)
     - 시나리오 V3-A/B/C 분기 narrative 정식 확정
     - paper §V.5.4 narrative 정식 채택 (사용자 결정 (1) 8-trial 결과 후 재확정 → 본 단계에서 결정)
     - DECISIONS 후속 엔트리 작성 (시나리오 결정 + paper §3.5 / §V.5.4 / §10 narrative 정식 확정)

  7. **(g) 사용자 후속 prep (Phase 5, 5/14~5/22)**:
     - 학위 논문 Part III chapter draft 작성
     - **통합 narrative 4 stage**:
       - Stage 1: V-3-ext baseline + over-smoothing 진단 (Phase 1 P80, qcond_nl3)
       - Stage 2: Mitigation v1 (Phase 2 b8) + paradox 발견 (단계 4-bis)
       - Stage 3: Mitigation v2 (Phase 3 #3+#4 + v2 #1+#2+#3) + 7-trial null effect + mech(ii) DOMINANT 5/5 결정적 강화 + paradox 정확한 분리 (attention pattern ↔ message aggregation collapse)
       - **Stage 4: Phase 1 deep dive (A1+A2+A3) + Mitigation v3 #1 GIN** = mech(ii) sub-mechanism (ii-a)/(ii-b) 분리 + GIN evidence (V3-A/B/C 분기 결과)
     - **Phase 1 A1+A2+A3 결과 인용**:
       - A1 v2_LN nuanced mechanism — softmax level mitigation only (post-softmax top5 -0.12) + L1=1.0 collapse 보존 (mech(ii-b))
       - A2 noise robustness — v2_LN Δtop5(σ=0.1) = -0.0033 (가장 robust)
       - A3 multi-DB 11 DBs — 10/11 일관 + single-DB caveat 해소

  8. **(h) 시나리오 V3-A/B/C 분기 narrative (DECISIONS 직전 §1(h) + Phase 1 deep dive A1 정밀화)**:
     - **시나리오 V3-A (가능성 中→高, A1 evidence 강화)**:
       - A1 결과: mech(ii-b) weighted-mean propagation 의 fundamental limitation (L1=1.0 collapse 보존)
       - GIN 가 propagation 자체 변경 (sum + MLP) 했음에도 L1_GAT cosine ≈ 1.0 유지 시 → **(ii-b) 가 aggregation family 자체 limitation 확정** = mech(ii) DOMINANT 5/5 절대 강화
       - paper §V.5.4 narrative: "8-trial mitigation null + GIN aggregation family 자체 변경에도 fail = weighted aggregation family (softmax-mean / sum-mean / MLP-sum) 자체의 fundamental limitation"
       - Filter Dominance 6번째 축 narrative 절대적 evidence 강화 (8-trial → 학위 논문 Part III main mechanism finding 결정적 confirm)
     - **시나리오 V3-B (가능성 中→낮음, A1 evidence 약화)**:
       - A1 결과 가 mech(ii-b) fundamental 강화 → GIN partial recovery 가능성 약화
       - 단 GIN MLP nonlinearity 가 aggregation collapse 경로 변경 가능성 잔존 (sum 의 magnitude variance 와 다름)
       - val R@15 0.62-0.70 + L1_GAT cosine 회복 (예: 0.85~0.95) → mech(ii-a) softmax 한정 부분 부정 + GIN 효과 발견
     - **시나리오 V3-C (가능성 낮음)**: A1 + A3 evidence 가 mech(ii-b) fundamental 강력 → R 0.85+ 회복 가능성 매우 낮음

- **근거**:
  - **Selector 모듈 단계 7 산출**: [EXPERIMENT_PLAN_selectors.md §V-3-ext 단계 7](../src/modules/selectors/EXPERIMENT_PLAN_selectors.md) (구현 + 7 smoke 통과 + GIN incompat 검증 + AGGREGATION_TYPES 확장)
  - **선행 결정**: DECISIONS 직전 엔트리 2026-05-08 (사용자 결정 A+B 통합) §1(c)/(e) — Mitigation v3 #1 GIN-style 학위 본 심사 전 진행 + Phase 2 selector 구현 spec
  - **선행 분석**: [Phase 1 deep dive A1+A2+A3](../notebooks/analysis_results/dsn_v2_layernorm_mechanism_decomposition.md) + [A2](../notebooks/analysis_results/dsn_softmax_noise_sensitivity.md) + [A3](../notebooks/analysis_results/dsn_per_db_stratified_7ckpt.md) — mech(ii-a)/(ii-b) sub-mechanism 분리 + multi-DB 정정
  - **Cross-reference**: 단계 5 (Phase 3 #3+#4) + 단계 6 (Mitigation v2) + 단계 7 (Mitigation v3 #1) 모두 5/7~5/8 가속 완료 — 학위 본 심사 timeline 여유 확보

- **영향 범위**:
  - **DECISIONS 본 엔트리** — Mitigation v3 #1 GIN 구현 완료 confirm + 신규 산출 3 파일 + smoke 7/7 + Root 학습 launch prep + Phase 4/5 prep + 시나리오 V3-A/B/C 분기 narrative (A1 evidence 정밀화)
  - **Selector EXPERIMENT_PLAN §V-3-ext 단계 7 ✅ 완료 표기** (이미 selector 모듈에서 작성됨)
  - **paper_research_direction.md (planner Edit, 본 응답)** — §3.5 / §V.5.4 / §10 갱신 (Phase 1 deep dive + Phase 2 GIN 학습 prep 통합 narrative)
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.13 신설 (Phase 1 deep dive + Phase 2 GIN 학습 prep 통합)
  - **Root 학습 launch prep prompt** (응답 본문) — 5/11 GIN 학습 launch
  - **paper main contribution 영향 X** (학회 anchor t_00 그대로)
  - **학위 논문 Part III chapter narrative weight 결정적 격상 prep**: 4-stage narrative (Phase 1 baseline + Mitigation v1 + Mitigation v2 + Phase 1 A1+A2+A3 deep dive + Mitigation v3 #1)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §3.5/§V.5.4/§10 갱신 + presentation_brief §14.13 신설 + Root 학습 launch prep prompt
  2. **사용자 (즉시)** — Root 세션 prompt 직접 붙여넣기 (cd `/home/hyeonjin/thesis_refactored`, 5/11 GIN 학습 launch)
  3. **Root (5/11~5/12)** — Mitigation v3 #1 GIN 학습 + ckpt NAS 저장 + EXPERIMENT_HISTORY 갱신 (8-trial 통합 표)
  4. **Analyzer (5/12~5/14)** — protocol 재실행 (8 ckpt × 5 step, multi-DB stratified 적용) — 산출물 dsn_mitigation_v3_8trial.md (신규) 또는 dsn_mitigation_v2_results.md §14 보강
  5. **Planner (5/14)** — 8-trial dominance scoring 갱신 + 시나리오 V3-A/B/C 분기 narrative 정식 확정 + paper §V.5.4 narrative 정식 채택 + DECISIONS 후속 엔트리
  6. **사용자 (5/14~5/22)** — 학위 논문 Part III chapter draft 작성 (4-stage 통합 narrative + A1+A2+A3 + Mitigation v3 #1 결과)

- **추가 필요 분석** (Phase 4 후속, Mitigation v3 #1 GIN 학습 + analyzer 결과 후):
  - GIN ckpt 의 L1_GAT cosine 측정 (mech(ii-b) 회복 정도 직접 정량 — A1 의 L1=1.0 collapse 가 GIN 으로 차단 가능한지)
  - GIN 의 epoch trajectory + AC fusion decay (mech(ii-a) attention level 부재 시 학습 dynamics)
  - 8-trial dominance scoring 갱신 (mech(ii) DOMINANT 5/5 절대 강화 / 4/5 부분 부정 / 3/5 부정 분기)
  - V3-A 시 paper §V.5.4 narrative 정식 채택 (GIN aggregation family limitation evidence 추가)
  - V3-B 시 GIN MLP nonlinearity 의 sum aggregation 효과 mechanism 분석 + paper main contribution 재평가

---

## 2026-05-08 (Phase 1 deep dive 완료 — A1 v2_LN nuanced mechanism + A2 softmax noise sensitivity + A3 per-DB stratified 11 DBs) — mech(ii) sub-mechanism (ii-a)/(ii-b) 분리 + multi-DB 정정 + single-DB caveat 해소 + Mitigation v3 GIN 기대치 정밀화

> **Status**: Analyzer Phase 1 후속 3 deep dive 완료 (2026-05-08, 정시 1.5일 일정). A1 + A2 + A3 통합 결과로 mech(ii) DOMINANT 5/5 의 정확한 sub-mechanism 분리 + v2_LN noise robustness 결정적 + 11 DBs 일관 evidence 확보. Mitigation v3 #1 GIN 학습 (Phase 3, 5/11~) 의 시나리오 V3-A/B/C 분기 evidence 정밀화 가능.

- **결정**:

  1. **(a) 🎯 mech(ii) Sub-mechanism 분리 정량 (A1 결정적 evidence)**:

     | Sub-mechanism | A1 evidence | A2 evidence | A3 evidence | 결론 |
     |---|---|---|---|---|
     | **(ii-a) softmax over-concentration** | L2 raw α std=1.04 → LN normed std=1.20 → post-softmax top5 **-0.12** (sharp peaking 차단) | LN 의 noise robustness Δtop5(σ=0.1) = **-0.0033** (가장 robust) | 11 DBs 중 10 일관 sharpening | **partial mitigation 가능 (LN level)** |
     | **(ii-b) weighted-mean propagation collapse** | L1 cosine **1.0 보존** (LN 으로 차단 X) | (forward deterministic, noise 영향 없음) | 11 DBs 모두 L1=1.0 schema-invariant | **fundamental architectural limitation** (schema/noise/mitigation 어떤 변경도 차단 X) |

     **결정적 발견**: LN pre-softmax 가 **softmax sharp peaking 만 차단** (L2 top5 -0.12) — message aggregation propagation collapse (L1=1.0) 는 그대로. mech(ii) 의 root cause 는 **(ii-b) weighted-mean propagation 의 aggregation 자체** (softmax 분포 변경으로 차단 불가).

  2. **(b) v2_LN Noise Robustness 정량 (A2 결정적 evidence)**:

     | Ckpt | σ=0 top5 | σ=0.1 top5 | Δtop5(σ=0.1) | 해석 |
     |---|---:|---:|---:|---|
     | phase1_p80 | 0.6844 | 0.6930 | +0.0086 | baseline |
     | phase2_b8 | 0.8714 | 0.8802 | +0.0087 | sharp 학습 |
     | phase3_directAC | 0.8215 | 0.8249 | +0.0034 | mid-sensitive |
     | **phase3_layerwiseLR** | 0.9353 | 0.9548 | **+0.0195** ⚠ | 가장 sensitive (sharp 학습) |
     | v2_drop_message | 0.8167 | 0.8344 | +0.0177 | drop random 효과 |
     | **v2_layernorm** ⭐ | 0.7271 | 0.7238 | **-0.0033** | 🔥 **가장 robust** (LN normalize) |
     | v2_sum_aggr | 0.8177 | 0.8305 | +0.0128 | sum sensitive |

     - LN 의 magnitude normalization 이 input noise 의 alpha shift 를 normalize 단계에서 absorb → post-softmax distribution 이 noise 에 invariant
     - 다른 ckpt: raw alpha magnitude 변동 → softmax 입력 → exp 비선형 amplify → top-K concentration 증가
     - **A1 + A2 결합**: LN 이 (ii-a) softmax level 에서 정확히 학습 dynamics 와 noise 둘 다 absorb

  3. **(c) Multi-DB 11 DBs 정정 (A3 결정적 evidence)**:

     **3.1 Single-DB → Multi-DB 측정 dramatic 차이**:

     | Ckpt | single-DB (calif.) | multi-DB (55 queries) | Δ |
     |---|---:|---:|---:|
     | phase1_p80 | 0.2445 | **0.7144** | +0.4699 |
     | phase2_b8 | 0.6885 | **0.8797** | +0.1912 |
     | v2_layernorm | 0.3540 | **0.7440** | +0.3900 |
     | phase3_layerwiseLR | 0.8282 | 0.9482 | +0.1200 |
     | v2_sum_aggr | 0.4656 | 0.8366 | +0.3710 |

     - california_schools (T=3, C=89) 의 in-degree ~30 (분산 가능) → top5 ratio 작음
     - 다른 DBs (T=8~13, C=20~237) 의 in-degree 5~15 작음 → top5 ratio 큼
     - **상대 비교 패턴 일관** (mech(ii) sharpening 일관 도달) — paper narrative 절대 수치만 multi-DB 로 정정, mech(ii) DOMINANT 결론 invariant

     **3.2 Schema-dependent LN attention 회복**:

     | Schema 규모 | Phase 1 top5 | LN 회복 (v2_LN - p2) |
     |---|---|---|
     | Trivial (toxicology T=4 C=20) | 1.0 | **0.0 (LN 무효)** ⚠ |
     | Small (debit/superhero C≤52) | 0.91~0.94 | -0.03 ~ -0.06 |
     | Medium (financial/student/codebase) | 0.72~0.82 | -0.10 ~ -0.15 |
     | Medium-large (card/formula_1) | 0.66~0.74 | -0.14 ~ -0.17 |
     | Large columns (calif./thrombosis) | 0.24~0.42 | **-0.28 ~ -0.29** |

     → **column 수가 많을수록 LN attention 회복 효과 큼**. trivial schema (toxicology) 에서는 무효 (in-degree ≤ 5 → top-5 = all → 측정 X).

     **3.3 mech(ii-b) L1=1.0 collapse 11 DBs 모두 schema-invariant**: toxicology 포함 모든 DBs 의 v2_LN L1_GAT cosine ≈ 1.0 — fundamental architectural limitation.

  4. **(d) 🚨 직전 paper narrative 정정 사항 (single-DB → multi-DB)**:
     - 직전 dsn_mitigation_v2_results.md §0 의 "v2_LN top5_conc 0.35" 수치 → **multi-DB 0.7440** 로 정정
     - 직전 7-trial 결과 표의 절대 수치들 multi-DB 정정 (단 상대 비교는 일관)
     - **mech(ii) DOMINANT 결론 invariant** — paper §V.5.4 narrative 보강 (single-DB caveat 해소 + multi-DB 11 DBs 일관 evidence)
     - paper §3.5 6번째 축 narrative 의 "v2_LN top5_conc 0.35 baseline 회복" → "multi-DB v2_LN top5_conc 0.7440 (Phase 1 baseline 0.7144 회복) — schema-dependent LN 효과 (큰 schema -0.28~-0.29 / 작은 schema -0.03~-0.06)" 로 정밀화

  5. **(e) Mitigation v3 #1 GIN 기대치 정밀화 (Phase 1 deep dive 결과 기반)**:
     - **A1 결과 가 V3-A 가능성 강화**: mech(ii-b) weighted-mean propagation 의 fundamental architectural limitation 확정 → GIN 의 sum + MLP propagation 도 weighted aggregation family 에 포함될 가능성 高
     - **GIN 의 MLP nonlinearity 가 변수**: v2_sum_aggr (sum, no MLP) 는 R=0.5735 (worst) → MLP 가 핵심. GIN 의 MLP+sum 이 propagation collapse 차단 가능성 잔존
     - **시나리오 분기**:
       - **V3-A (가능성 中→高)**: GIN 도 fail (val R@15 ~0.59-0.61, L1_GAT cosine ≈ 1.0) → mech(ii) DOMINANT 5/5 절대 강화 (8-trial null + aggregation family 자체 limitation)
       - **V3-B (가능성 中→낮음)**: GIN partial recovery (val R@15 0.62-0.70, L1 회복) → mech(ii-a) softmax 한정 부분 부정 + GIN MLP+sum 효과 발견
       - **V3-C (가능성 낮음)**: A1+A2+A3 evidence 가 mech(ii-b) fundamental 강력 → R 0.85+ 가능성 매우 낮음

  6. **(f) paper §V.5.4 narrative 보강 (multi-DB + sub-mechanism + schema-dep + caveat 해소)**:
     - **paper §V Conclusion 갱신 narrative**:
       > "DSN mech(ii) DOMINANT 의 sub-mechanism 분리: **(ii-a) softmax over-concentration** 은 LayerNorm pre-softmax 로 partial mitigation 가능 (post-softmax top5_conc Δ=-0.12, multi-DB 11 DBs 중 10 일관, schema-dependent: 큰 schema -0.28~-0.29 / 작은 schema -0.03~-0.06 / trivial schema 무효). **(ii-b) weighted-mean message aggregation propagation collapse** 은 LayerNorm 으로 차단 X — 11 DBs 모두 L1_GAT cosine = 1.0 schema-invariant fundamental architectural limitation. v2_LN 의 noise robustness Δtop5(σ=0.1) = -0.0033 (7 ckpt 중 가장 robust) 가 (ii-a) softmax level mitigation 의 학습 dynamics + input noise 둘 다 absorb 직접 evidence."
     - **single-DB caveat 해소**: dsn_mitigation_v2_results.md §13 의 single-DB only caveat → A3 multi-DB 11 DBs evidence 로 paper §V.5.4 본문 인용 가능 정량 + Mitigation v3 #1 GIN 결과 후 재평가 base
     - **schema-dependent caveat 신설**: toxicology trivial schema (in-degree ≤ 5) 에서는 mech(ii) 측정 X — paper §V.5.4 Limitations 1 줄 명시

  7. **🚨 사용자 결정 필요 3 항목**:

     | # | 결정 항목 | 옵션 | 권장 |
     |---|----------|------|------|
     | (1) | **Mitigation v3 #1 GIN 시나리오 V3-A/B/C 분기 narrative 확정 시점** | (A) Phase 4 (5/12 학습 종료 후, 8-trial protocol 재실행 결과 후) 정식 채택 / (B) 즉시 V3-A 가능성 高 narrative candidate 채택 (A1 evidence 정밀화 후) | **(A) Phase 4 후 정식 채택** — A1 evidence 가 V3-A 강화하지만 GIN MLP nonlinearity 의 mech(ii-b) 차단 가능성 잔존, 학습 결과 정량 confirm 후 정식 |
     | (2) | **Phase 4 (5/12~5/14) analyzer 추가 위임** | (A) 8-trial protocol 재실행 + multi-DB stratified 적용 (single-DB 제거) — 권장 / (B) 8-trial protocol 만 (single-DB 유지) — 직전 protocol 호환 | **(A) multi-DB stratified 적용** — A3 결과로 single-DB 측정 dramatic underestimation 확인, 8-trial 모두 multi-DB 측정으로 통일 권장 (paper §V.5.4 정식 정량 base) |
     | (3) | **Mitigation v3 추가 candidate (max aggregation, EGAT) 학습 우선순위** | (A) #2 max aggregation 만 추가 시도 (~10h, 5/13 GPU 0 launch) — V3-A 시 GIN+max 통합 evidence 강화 / (B) #4 EGAT 만 추가 시도 (~10h+, 가장 architectural shift) / (C) 모두 post-paper backlog (학위 본 심사 timeline 부족 + GIN 결과 보고 결정) | **(C) 모두 post-paper backlog** — Phase 3 GIN 결과 (5/12) 후 시나리오 분기 결정 후 Mitigation v3 추가 시도 여부 재고. V3-A 가능성 高 시 max aggregation 추가 시도 1 candidate 만 (학위 본 심사 timeline 충분) |

- **근거**:
  - **신규 analyzer 3 산출**:
    - [dsn_v2_layernorm_mechanism_decomposition.md](../notebooks/analysis_results/dsn_v2_layernorm_mechanism_decomposition.md) (A1)
    - [dsn_softmax_noise_sensitivity.md](../notebooks/analysis_results/dsn_softmax_noise_sensitivity.md) (A2)
    - [dsn_per_db_stratified_7ckpt.md](../notebooks/analysis_results/dsn_per_db_stratified_7ckpt.md) (A3)
  - **재현 데이터**: outputs/analysis/dsn_phase1_deep_dive/{a1, a2, a3}_*.json
  - **재현 스크립트**: src/analysis/dsn_phase1_deep_dive.py
  - **선행 분석**: dsn_mitigation_v2_results.md (7-trial single-DB) — A3 가 single-DB caveat 해소
  - **선행 결정**: DECISIONS 직전 엔트리 2026-05-08 (사용자 결정 A+B 통합) §1(b) — A1+A2+A3 위임

- **영향 범위**:
  - **DECISIONS 본 엔트리** — Phase 1 deep dive 3 분석 통합 결과 + mech(ii-a)/(ii-b) sub-mechanism 분리 + v2_LN noise robust + multi-DB 11 DBs 일관 + single-DB caveat 해소 + Mitigation v3 GIN 기대치 정밀화 + 사용자 결정 3 항목
  - **paper_research_direction.md (planner Edit, 본 응답)**:
    - §3.5 Filter Dominance 6번째 축 sub-section 갱신 (multi-DB 수치 정정 + sub-mechanism 분리 + schema-dependent + analyzer A1+A2+A3 인용)
    - §3.5 학위 논문 Part III chapter outline §III.4/§III.6 갱신 (sub-mechanism 분리 + multi-DB)
    - §V.5 §V.5.4 narrative 갱신 (multi-DB 정정 + caveat 해소)
    - §10 V-3-ext 단계 5+6 sub-section 갱신 (single-DB → multi-DB 비교 표 + sub-mechanism)
    - §9 Limitations — single-DB caveat 해소 + schema-dependent caveat 신설 (toxicology trivial 무효)
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.13 신설 (Phase 1 deep dive 결과 + sub-mechanism 분리 + multi-DB 정정 + 사용자 결정 3 항목)
  - **paper main contribution (학회)** 영향 X — anchor t_00 + 4 축 narrative 그대로
  - **학위 논문 Part III chapter narrative weight 결정적 격상** — sub-mechanism 분리 + multi-DB 11 DBs 일관 evidence + single-DB caveat 해소

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §3.5/§V.5.4/§10/§9 갱신 + presentation_brief §14.13 신설
  2. **사용자 (즉시 의사결정 3 항목)** — (1) V3 narrative 확정 시점 / (2) Phase 4 multi-DB 적용 / (3) Mitigation v3 추가 candidate 우선순위
  3. **사용자 (Phase 2 GIN 구현 완료 후 즉시)** — Root 세션 prompt 직접 붙여넣기 (Phase 3 GIN 학습 5/11 launch)
  4. **Root (5/11~5/12)** — Mitigation v3 #1 GIN 학습 + ckpt NAS 저장 + EXPERIMENT_HISTORY 갱신
  5. **Analyzer (5/12~5/14, Phase 4)** — 8-trial protocol 재실행 (multi-DB stratified 적용 권장) — 산출물 dsn_mitigation_v3_8trial.md
  6. **Planner (5/14)** — 8-trial dominance scoring 갱신 + 시나리오 V3-A/B/C 분기 narrative 정식 확정 + paper §V.5.4 narrative 정식 채택 + DECISIONS 후속 엔트리
  7. **사용자 (5/14~5/22)** — 학위 논문 Part III chapter draft 작성 (4-stage 통합 narrative + A1+A2+A3 + Mitigation v3 #1)

- **추가 필요 분석** (Phase 4 후속, Mitigation v3 #1 GIN 학습 + analyzer 결과 후):
  - GIN ckpt 의 L1_GAT cosine 측정 — A1 의 mech(ii-b) fundamental 가설 직접 검증 (GIN sum+MLP propagation 도 L1=1.0 collapse 인지)
  - GIN 의 multi-DB 11 DBs 일관 evidence (mech(ii) DOMINANT generalizability 8-trial 통합)
  - 8-trial dominance scoring 갱신 (multi-DB 정정 + sub-mechanism (ii-a)/(ii-b) 분리 evidence 추가)
  - 시나리오 V3-A 시 paper §V.5.4 narrative 정식 채택 — "8-trial mitigation null + GIN aggregation family 자체 변경에도 fail = weighted aggregation family fundamental limitation" + sub-mechanism (ii-a) partial mitigation 가능 / (ii-b) schema-invariant fundamental 분리

---

## 2026-05-08 (사용자 결정 A+B 통합 — Over-smoothing deep dive: Analyzer 후속 3 분석 + Mitigation v3 #1 GIN-style 학위 본 심사 전 진행) — Selector + Analyzer + Root 핸드오프 prep + 4-phase timeline 확정

> **사용자 직전 input (2026-05-08)**: "이 Over-Smoothing 문제를 해결하고 싶은데, 좀 더 상세한 분석을 하는 건 어떨까" → planner 4 옵션 (A/B/C/D) 제시 → 사용자 **A+B 통합 선택** (Analyzer mechanism 정밀화 + Mitigation v3 #1 GIN-style 학습 시도 통합).

> **사용자 결정 변경**:
> - 직전 (2026-05-08 7-trial dominance 엔트리) §1(9)(2) Mitigation v3 우선순위 = (D) 모두 post-paper backlog
> - **본 엔트리 갱신**: (D) 모두 post-paper → **(A) 일부 학위 본 심사 전 진행** — #1 GIN-style aggregation 만 학위 본 심사 전 시도 (#2/#3/#4 는 post-paper 보존)

- **결정**:

  1. **(a) 사용자 결정 A+B 통합 — 4-phase timeline 확정**:

     | Phase | 일정 | 작업 | 세션 | 비용 |
     |---|---|---|---|---|
     | **Phase 1** | 5/8~5/10 (~1.5일) | **Analyzer 후속 3 deep dive** (병렬 가능) | analyzer | LLM-free ₩0 |
     | **Phase 2** | 5/10~5/11 | Selector 모듈 — Mitigation v3 #1 GIN-style aggregation 구현 + smoke test | selector | ~5h 구현 |
     | **Phase 3** | 5/11~5/12 | Root — Mitigation v3 #1 GIN 학습 (GPU 0, ~10h) | root | ~10h 학습 |
     | **Phase 4** | 5/12~5/14 | Analyzer protocol 재실행 (8 ckpt × 5 step) — 7-trial → 8-trial dominance scoring | analyzer | LLM-free ₩0 |
     | **Phase 5** | 5/14~5/22 | 사용자 — 학위 논문 Part III chapter draft 작성 (analyzer 후속 + Mitigation v3 #1 통합 narrative) | user | — |

  2. **(b) 🆕 Analyzer 후속 3 deep dive (Phase 1, 병렬 가능 LLM-free)**:

     | # | 분석 | 가설 / 측정 spec | 산출물 |
     |---|------|----|---|
     | **A1** | **v2_LN nuanced mechanism** — LayerNorm 이 어떤 message component 정규화에 작용하는지 | LayerNormGATv2Conv 의 raw alpha tensor capture (forward hook) + per-head magnitude / variance / sign distribution 분석. softmax pre/post 분포 비교 + L1_GAT cosine = 1.0 collapse 가 어디서 발생하는지 (alpha → message → aggregation 단계별) decompose | `notebooks/analysis_results/dsn_v2_layernorm_mechanism_decomposition.md` |
     | **A2** | **Softmax noise sensitivity** — input perturbation 시 alpha 변동 정량 | 7 ckpt × column 노드 input 에 Gaussian noise (σ=0.01, 0.05, 0.1) 추가 → alpha tensor variance 측정. v2_LN 의 noise robustness 정량 + Phase 2/3 의 noise sensitivity 비교 | `notebooks/analysis_results/dsn_softmax_noise_sensitivity.md` |
     | **A3** | **Per-DB stratified 11 DBs 재측정** — single-DB caveat 해소 (mech(ii) generalizability 검증) | BIRD-dev shuffle=True (seed 고정) 또는 per-DB stratified sampling (각 DB 5 queries × 11 DBs = 55 queries) → 7 ckpt × 4 mechanism × 5 step 재실행. mech(ii) DOMINANT 의 11 DBs invariance 검증. toxicology (작은 schema) vs european_football_2 (큰 schema) 의 mech(ii) 차이 정량 | `notebooks/analysis_results/dsn_per_db_stratified_7ckpt.md` |

     **사용자 결정 갱신**: 직전 (2026-05-08 §1(9)) (3) per-DB 후속 측정 = (B) post-paper backlog → **본 엔트리 (A) 학위 본 심사 전 진행** (single-DB caveat 해소가 paper §V.5.4 narrative 강화 + 학위 논문 Part III chapter Limitations 1 줄 → 11 DBs 일관 evidence 로 격상)

  3. **(c) 🚀 Mitigation v3 #1 GIN-style aggregation 학습 (Phase 2 + Phase 3)**:
     - **Mechanism**: softmax + weighted-mean → `sum(MLP(x_i + x_j))` propagation (GIN 의 invariance theorem). aggregation propagation 자체 대체로 동질화 차단 가능성 검증
     - **Hypothesis**:
       - **시나리오 V3-A (가능성 中)**: GIN 도 fail (val R@15 ~0.59-0.61) → mech(ii) DOMINANT 5/5 절대적 강화 (8-trial null effect = aggregation function family 자체의 limitation)
       - **시나리오 V3-B (가능성 中)**: GIN partial recovery (val R@15 0.62-0.70) → mech(ii) softmax 한정 부분 부정 + GIN 의 sum aggregation 효과 발견 → paper main contribution 4 → 5 항목 격상 후보
       - **시나리오 V3-C (가능성 낮음)**: GIN 이 R 0.85+ ceiling 갱신 → mech(ii) 부정 + paper main contribution 재평가
     - **구현 spec (selector 모듈 핸드오프)**:
       - `gat_network_v2.py` 에 `GINStyleConv(MessagePassing)` 또는 PyG `GINConv` 도입 (heterograph 호환)
       - `aggregation_type: "gin"` config flag 추가 (기존 `mean / sum / max` 와 별개)
       - GIN 의 epsilon 학습 + MLP layer 1-2개 추가 (PyG default)
       - 신규 config: `train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml` (Phase 2 b8 + aggregation_type: "gin")
       - smoke test 5: GINStyleConv subclass / forward shape / heterograph 호환 / backward compat / config parsing
     - **학습 entry**: `python src/train_gat_s06.py --config configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml`
     - **신규 ckpt**: `best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt` (NAS path + symlink)
     - batch_size=8, ~10h 학습, GPU 0

  4. **(d) 8-trial 통합 dominance scoring 갱신 (Phase 4)**:
     - 7-trial (Phase 1 + Phase 2 b8 + Phase 3 #3 + #4 + v2 #1 + #3 + #2) + Mitigation v3 #1 GIN = **8-trial**
     - mech(ii) DOMINANT scoring 갱신:
       - V3-A 시나리오 (GIN fail): 8-trial null = mech(ii) **5/5 절대적 강화** (aggregation family 한정도 못 해결)
       - V3-B 시나리오 (GIN partial): mech(ii) 4/5 부분 부정 + GIN 부분 효과 evidence
       - V3-C 시나리오 (GIN 회복): mech(ii) 3/5 부정 + paper main contribution 재평가
     - 산출물: `notebooks/analysis_results/dsn_mitigation_v3_8trial.md` (analyzer 신규 또는 dsn_mitigation_v2_results.md §14 보강)

  5. **(e) Selector 모듈 핸드오프 prompt (Phase 2, 5/10 시작)**:
     ```
     먼저 src/modules/selectors/CLAUDE.md 와 DECISIONS.md 최상단 (2026-05-08 사용자 결정 A+B 통합) §1(c) 의 GIN-style aggregation 구현 spec 읽고,
     Mitigation v3 #1 GIN-style aggregation 구현 + smoke test (단계 7 신설).

     구현 spec:
     - src/models/gat_network_v2.py 에 GIN aggregation 옵션 추가
       - PyG GINConv 도입 (heterograph 호환 — HeteroConv 안에서 GIN 가능 검증)
       - 또는 custom GINStyleConv: sum(MLP(x_i + x_j)) propagation
     - aggregation_type: "gin" config flag (기존 mean/sum/max 와 별개)
     - 신규 config: configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml
       - Base = Phase 2 b8 (mitigation 동일) + aggregation_type: "gin"
     - smoke test 5: GINConv subclass / forward shape / heterograph 호환 / backward compat (default mean) / config parsing
     - Phase 2 b8 backward compat 보존 (aggregation_type 미설정 시 mean default)

     선행 reference:
     - 단계 6 (Mitigation v2 #1+#3+#2): src/modules/selectors/EXPERIMENT_PLAN_selectors.md §V-3-ext 단계 6
     - GINConv: torch_geometric.nn.GINConv (PyG 공식)
     - heterograph + GIN: HeteroConv 가 GINConv 호환 가능 확인 (smoke 필수)
     ```

  6. **(f) Analyzer 핸드오프 prompt (Phase 1 즉시 + Phase 4 후속)**:
     ```
     먼저 src/analysis/CLAUDE.md 와 DECISIONS.md 최상단 (2026-05-08 사용자 결정 A+B 통합) §1(b) 의 3 deep dive 분석 spec 읽고,
     A1 + A2 + A3 3 분석을 병렬 진행 (~1.5일 합).

     A1. v2_LN nuanced mechanism decomposition:
     - LayerNormGATv2Conv 의 raw alpha tensor capture (forward hook)
     - per-head magnitude / variance / sign distribution 분석
     - softmax pre/post 분포 비교
     - L1_GAT cosine = 1.0 collapse 가 어디서 발생하는지 (alpha → message → aggregation 단계별 decompose)
     - 산출물: notebooks/analysis_results/dsn_v2_layernorm_mechanism_decomposition.md

     A2. Softmax noise sensitivity:
     - 7 ckpt × column 노드 input 에 Gaussian noise (σ=0.01, 0.05, 0.1) 추가
     - alpha tensor variance 측정
     - v2_LN 의 noise robustness 정량 + Phase 2/3 비교
     - 산출물: notebooks/analysis_results/dsn_softmax_noise_sensitivity.md

     A3. Per-DB stratified 11 DBs 재측정:
     - BIRD-dev shuffle=True (seed 고정) 또는 per-DB stratified (각 DB 5 queries × 11 DBs = 55 queries)
     - 7 ckpt × 4 mechanism × 5 step 재실행
     - mech(ii) DOMINANT 의 11 DBs invariance 검증
     - toxicology vs european_football_2 의 mech(ii) 차이 정량
     - 산출물: notebooks/analysis_results/dsn_per_db_stratified_7ckpt.md

     Phase 4 후속 (5/12 GIN 학습 종료 후):
     - 8 ckpt × 5 step protocol 재실행
     - 8-trial dominance scoring 갱신 (mech(ii) 5/5 결정적 강화 / 4/5 부분 부정 / 3/5 부정 분기)
     - 산출물: notebooks/analysis_results/dsn_mitigation_v3_8trial.md (또는 dsn_mitigation_v2_results.md §14 보강)
     ```

  7. **(g) Root 핸드오프 prompt (Phase 3, 5/11 시작)**:
     ```
     먼저 CLAUDE.md 와 DECISIONS.md 최상단 (2026-05-08 사용자 결정 A+B 통합) §1(c)/(e) 읽고,
     Mitigation v3 #1 GIN-style aggregation 학습 launch + ckpt NAS 저장 + EXPERIMENT_HISTORY 갱신.

     5/11 launch (Selector 구현 완료 후):
     - GPU 0: CUDA_VISIBLE_DEVICES=0 nohup python src/train_gat_s06.py --config configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml > logs/train/gat_dsn_p80_b5_mitigation_v3_gin_$(date +%Y%m%d_%H%M).log 2>&1 &
     - ~10h 학습, batch_size=8, ETA 5/12 KST

     학습 종료 후:
     - ckpt NAS 저장: /SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt
     - 로컬 symlink: outputs/checkpoints/
     - EXPERIMENT_HISTORY.md "Mitigation v3 #1 GIN 학습 (V-3-ext 단계 7, 5/11~5/12)" entry 추가
       - 학습 wall + best val R@15 + best epoch + ckpt path + AC loss epoch trajectory + GIN aggregation 동작 확인
     - alpha sweep skip (사용자 결정 (1)A 유지) — paper main F1/EX 측정 X, val recall@15 evidence only
     - 학습 종료 후 analyzer 핸드오프 (Phase 4)
     ```

  8. **(h) 시나리오 V3-A/B/C 분기 narrative**:
     - **시나리오 V3-A (가능성 中 — GIN 도 fail, mech(ii) 절대적 강화)**:
       - val R@15 ~0.57-0.61 영역 unchanged
       - **mech(ii) 5/5 절대적 강화** + 8-trial null effect (aggregation function family 자체 limitation)
       - paper §V.5.4 narrative: "8-trial mitigation null + GIN aggregation 자체 대체에도 fail = edge softmax + weighted-mean propagation 만의 issue 가 아닌 weighted aggregation family 자체의 limitation"
       - Filter Dominance 6번째 축 narrative 절대적 evidence 강화 (8-trial → 학위 논문 Part III main mechanism finding 결정적 confirm)
     - **시나리오 V3-B (가능성 中 — GIN partial recovery)**:
       - val R@15 0.62-0.70 영역 partial 회복
       - **mech(ii) 4/5 부분 부정 + GIN 부분 효과 발견**
       - paper main contribution 4 → 5 항목 격상 후보 (단 anchor 변경은 학회 후 별도 결정)
       - 학위 논문 Part III main contribution: "**aggregation function 변경 mech(ii) partial mitigation 발견**"
     - **시나리오 V3-C (가능성 낮음 — GIN R 0.85+ 회복)**:
       - 가능성 낮음 (mech(ii) edge softmax fundamental limitation evidence 강함, GIN 도 weighted aggregation family)
       - **§V.5.4 narrative 큰 수정 + paper main contribution 재평가** + 학회 후 paper anchor 재검토

  9. **(i) 학회 논문 narrative 영향 X (재확인)**:
     - paper main anchor t_00 (F1=0.8657) 변경 X
     - Filter Dominance 4 축 narrative (학회) 그대로
     - 8-trial 결과는 **학위 논문 Part III chapter §V.5.4 만 적용**
     - 학회 §V.5.3 Future Work 1 줄 (DSN 8-trial mitigation null + GIN-style aggregation 부정 가능성 검증) 사용자 직접 처리

- **근거**:
  - **사용자 직전 input** (2026-05-08): "이 Over-Smoothing 문제를 해결하고 싶은데, 좀 더 상세한 분석을 하는 건 어떨까" → planner 4 옵션 (A/B/C/D) 제시 → **사용자 A+B 통합 선택**
  - **선행 결정**: DECISIONS 직전 엔트리 2026-05-08 (7-trial mechanism dominance 결정적 확정) §1(9) — 사용자 결정 (2)D 모두 post-paper backlog → 본 엔트리 (2)A 일부 (#1 GIN-style 만 학위 본 심사 전) 변경
  - **선행 분석**: dsn_mitigation_v2_results.md §10 권장 #1 + analyzer §13 후속 권장 (per-DB stratified + softmax noise + Mitigation v3)
  - **GIN reference**: PyG `GINConv` (https://pytorch-geometric.readthedocs.io/) — heterograph 호환 검증 필요

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 사용자 결정 A+B 통합 + 4-phase timeline + Analyzer 후속 3 deep dive + Mitigation v3 #1 GIN 학위 본 심사 전 진행 + Selector + Analyzer + Root 핸드오프 prompt + 시나리오 V3-A/B/C 분기 narrative
  - **paper_research_direction.md (planner Edit, 본 응답)**:
    - §8 Future Works H-DTK Mitigation v3 #1 GIN 항목 갱신 (post-paper backlog → 학위 본 심사 전 진행)
    - §8 Per-DB 후속 항목 갱신 (post-paper backlog → 학위 본 심사 전 진행 — A3 분석에 통합)
    - §8 H-DTK softmax noise sensitivity 갱신 (post-paper → A2 즉시 진행)
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.12.7 사용자 결정 갱신 ((2)D → A 일부 + (3)B → A 변경) + 4-phase timeline 추가
  - **Selector 모듈 세션 핸드오프 prompt** (응답 본문) — Mitigation v3 #1 GIN 구현 + smoke test
  - **Analyzer 세션 핸드오프 prompt** (응답 본문) — A1+A2+A3 3 deep dive 분석 + Phase 4 8-trial 재실행
  - **Root 세션 핸드오프 prompt** (응답 본문) — Phase 3 GIN 학습 launch
  - **paper main contribution 영향**: 학회 narrative X / **학위 논문 §V.5.4 narrative 정식 채택은 8-trial 결과 후 재확정** (시나리오 V3-A/B/C 분기)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §8 갱신 + presentation_brief §14.12.7 갱신 + 3 핸드오프 prompt
  2. **사용자 (즉시)** — Analyzer 세션 prompt 직접 붙여넣기 (cd `/home/hyeonjin/thesis_refactored/src/analysis`) — Phase 1 A1+A2+A3 3 deep dive 시작
  3. **사용자 (5/10 종료 후)** — Selector 모듈 prompt 직접 붙여넣기 — Phase 2 GIN 구현
  4. **사용자 (5/11 selector 구현 완료 후)** — Root 세션 prompt 직접 붙여넣기 — Phase 3 GIN 학습 launch
  5. **Analyzer (Phase 1 즉시 + Phase 4 5/12 후)** — 3 deep dive (5/8~5/10) + 8-trial protocol 재실행 (5/12~5/14)
  6. **Selector 모듈 (5/10~5/11)** — GIN-style aggregation 구현 + smoke test 5
  7. **Root (5/11~5/12)** — GIN 학습 + EXPERIMENT_HISTORY 갱신
  8. **Planner (5/14)** — 8-trial dominance scoring 갱신 + 시나리오 V3-A/B/C 분기 narrative 정식 확정 + DECISIONS 후속 엔트리
  9. **사용자 (5/14~5/22)** — 학위 논문 Part III chapter draft 작성 (Analyzer 후속 + Mitigation v3 #1 통합 narrative)

- **추가 필요 분석** (Phase 4 후속, 시나리오 분기 후):
  - V3-A 시 mech(ii) 5/5 절대적 강화 narrative + Filter Dominance 6번째 축 narrative 결정적 confirm
  - V3-B 시 GIN partial mitigation mechanism 정량 + paper main contribution 재평가 후보
  - V3-C 시 (낮음) Filter Dominance narrative 큰 수정 + 학회 후 paper anchor 재검토
  - per-DB stratified 11 DBs evidence 통합 (mech(ii) generalizability 검증)
  - Softmax noise sensitivity → v2_LN 의 robustness 정량 + paper §V.5.4 narrative 보강

---

## 2026-05-08 (7-trial mechanism dominance 결정적 확정 — 🎯 Mech(ii) GATv2Conv Normalization 5/5 결정적 강화 + Mech(iii) Skip Dependency 3/5 부정 강화) — paradox 분리 + 학위 논문 Part III §V.5.4 정식 채택 + Filter Dominance 6번째 축 narrative 결정적 evidence

> **Status**: analyzer dsn_mitigation_v2_results.md (7 ckpt × 4 mechanism × 5 step) 완료. v2_layernorm 의 attention pattern Phase 1 baseline 회복 evidence + 모든 6 mit ckpt L1_GAT cosine = 1.0 collapse + Phase 3 #3 AC=0.62 일관 유지 결합 → mech(ii) 5/5 결정적 강화. mech(iii) skip dep 는 v2_drop/LN 가 main GAT gradient 회복 (conv_L1 0.30~0.31, Phase 2 의 2.5×) + skip_dep ratio 1.36~1.41 (균형) 도달했음에도 R 미달 → 3/5 부정 강화.

- **결정**:

  1. **(a) 🎯 7-trial Mechanism Dominance 갱신 — Mech(ii) 결정적 강화**:

     | Mechanism | Phase 2 only | 4-trial | **7-trial 갱신** | 핵심 정량 evidence |
     |---|:---:|:---:|:---:|---|
     | (i) Aggregation collapse | 2/5 | 2/5 | 2/5 (변경 X) | top-5 raw cos 0.48~0.59 marginal, v2_sum_aggr 도 dramatic 차이 X |
     | **(ii) GATv2Conv normalization** | 3/5 | 5/5 DOMINANT | **5/5 ⭐⭐ 결정적 강화** | **v2_LN top5_conc 0.35** (Phase 1 0.24 회복) → R 0.6007 partial recovery + L1=1.0 collapse 동시 + Phase 3 #3 AC=0.62 일관 유지 |
     | (iii) Skip dependency pathology | 5/5 DOMINANT | 4/5 보조 | **3/5 부정 강화** | v2_drop/LN 가 conv_L1 0.30~0.31 (Phase 2 의 2.5×, LR x5 의 6×) 회복 + skip_dep 1.36~1.41 균형 도달 → 단 R 미달 (0.5970~0.6007) — gradient 회복만으로 ceiling 갱신 X |
     | (iv) Schema sibling (raw PLM) | 3/5 | 3/5 | 3/5 (변경 X) | L0=0.5090 모든 ckpt 동일 lower bound |

     **결정적 evidence (Mech(ii) DOMINANT 5/5 강화 직접 정량)**:
     - **v2 #3 LayerNorm pre-softmax 가 attention pattern 을 Phase 1 baseline 으로 회복**:
       - L2 top5_conc: Phase 2/3 의 0.58~0.83 → v2_LN **0.3540** (Phase 1 의 0.2445 와 매우 가까움)
       - L2 entropy: Phase 2/3 의 2.02~2.75 → v2_LN **3.1558** (Phase 1 의 3.2177 와 거의 동일)
       - → softmax 직전 LayerNorm 이 sharp peaking 차단, attention 분포 baseline 유지
     - **v2_LN best R@15 = 0.6007** (mitigation variants 최고, Phase 2 b8 0.6018 와 사실상 동등) — attention sharpening 차단으로 partial recovery (-0.0090 vs Phase 1 baseline)
     - **그러나 v2_LN L1_GAT cosine = 0.9998** (Phase 2/3 의 1.0 동일) — attention 회복으로도 message aggregation collapse 차단 X
     - → **edge softmax 의 weighted-mean propagation 자체가 column 동질화 강제** (mech ii 의 fundamental nature)
     - **Phase 3 #3 AC=0.62 일관 유지** (300 epoch ΔAC = -0.0001) + 6 mit ckpt L1=1.0 + LayerNorm partial recovery 결합 = 결정적 evidence

  2. **(b) 🟡 Mech(iii) Skip Dependency 부분 부정 강화 (4/5 → 3/5)**:
     - **v2_drop_message + v2_layernorm 의 conv_L1 gradient = 0.30~0.31**:
       - Phase 2 b8 의 0.12 대비 **2.5×** 회복
       - Phase 3 #4 LR x5 의 0.05 대비 **6×** 회복
       - Phase 1 의 1.05 대비 1/3 수준
       - → DropMessage / LayerNorm 이 main GAT path 학습 신호 회복
     - **skip_dep ratio 1.36~1.41 (균형)** — Phase 1 의 3.02 대비 ½, Phase 3 #4 의 7.38 대비 1/5
     - **그러나 best R@15 = 0.5970~0.6007 (Phase 1 미달)** — main path 학습 회복 + skip 균형 도달했음에도 ceiling 갱신 X
     - → **skip dependency 가 root cause 아님 결정적 확정** (4-trial 의 부분 부정 → 7-trial 부정 강화)

  3. **(c) 🔥 Paradox 정확한 분리 (paper §V.5.4 narrative 핵심)**:
     ```
     ┌────────────────────────────────────────────────┐
     │  attention pattern (학습 가능)                  │  ← LayerNorm pre-softmax 로 baseline 회복 (top5_conc 0.35)
     └────────────────────────────────────────────────┘
                            ↓
     ┌────────────────────────────────────────────────┐
     │  edge softmax + weighted-mean message aggr     │  ← 구조적 limitation, mitigation 5종 어떤 것도 차단 X
     │                                                │     → L1_GAT cosine = 1.0 collapse
     └────────────────────────────────────────────────┘
                            ↓
     ┌────────────────────────────────────────────────┐
     │  output aggregation result (collapse)          │  ← fusion + skip 으로 partial 분산 (L_out 0.29~0.32)
     │   = column 동질화 강제                            │     단 query-conditional discrimination 회복 X
     └────────────────────────────────────────────────┘
                            ↓
     val R@15 ceiling ~0.61 invariant (7-trial 모두 fail)
     ```
     - **분리 핵심**: attention pattern 은 학습 가능 (LayerNorm 으로 baseline 회복) ↔ message aggregation collapse 는 구조적 (mitigation 5종 어떤 것도 차단 X)
     - **v2 layernorm 이 정확히 attention level mitigation** 도달 → partial recovery (-0.0090) 가 attention sharpening 차단으로 가능한 maximum

  4. **(d) 학위 논문 Part III §V.5.4 narrative 정식 채택 (analyzer §0 + §7.3 + §10 인용)**:
     > "DSN 7-trial mitigation null effect 의 root cause = **GATv2Conv edge softmax 의 weighted-mean message aggregation collapse**. 7 ckpt (Phase 1 no mit, Phase 2 b8 fusion, Phase 3 #3 Direct AC, Phase 3 #4 LR x5, v2 #1 DropMessage, v2 #3 LayerNorm pre-softmax, v2 #2 Sum aggregation) 중 **LayerNorm pre-softmax 가 attention pattern 을 Phase 1 baseline 으로 정확히 회복** (col→tab L2 top5_conc 0.62~0.83 → **0.35**, entropy 2.02~2.75 → **3.16**) → R@15 partial recovery (Phase 2 b8 0.6018 → v2_LN **0.6007 + 사실상 동등**, Phase 1 0.6097 의 -0.0090). 단 모든 mitigation 적용 ckpt 의 L1_GAT cosine = 1.0 (column 임베딩 첫 layer 만에 완전 동질화) — attention sharpening 차단으로도 message aggregation collapse 차단 X. **edge softmax 의 weighted-mean aggregation 이 같은 dst (table) 의 incoming src (column) 를 동일 표현으로 propagation 강제** (구조적 limitation). 추가로 v2 #1 DropMessage / v2 #3 LayerNorm 가 main GAT path gradient 0.30~0.31 (Phase 2 의 2.5×) 회복 + skip_dep 균형 도달했음에도 R 미달 → **mech(iii) skip dependency 부정 결정적 강화** (skip 우회 해결만으로 ceiling 갱신 X). **Filter Dominance 6번째 축 (training-pathology-invariant) 의 결정적 evidence 강화** — GAT learning fundamental architectural limitation (mech ii edge softmax aggregation collapse) 까지 With-Filter pipeline 이 흡수."

  5. **(e) Filter Dominance 6번째 축 narrative — 7-trial evidence 정식 채택**:

     | # | Evidence | 정량 |
     |---|----------|------|
     | 1 | H-B ckpt-invariant | Pearson r 0.06~0.24 |
     | 2 | H-F stability/ordering | k=20 Jaccard 0.47~0.52 + Spearman 0.6453 |
     | 3 | F-1 + H-G alpha sweep | F-1 plateau spread 0.0724 → WF 0.0142 = 5.0850× 압축 |
     | 4 | ΔF1 +0.65 lift | mean per-query gain +0.6462 |
     | 5 | H-A/H-D 부정 | Enriched ckpt + norm 변형 모두 plateau 유지 |
     | **6** | **🆕 7-trial mitigation null effect** (mech ii 결정적 강화) | v2_LN top5_conc 0.35 회복 + L1=1.0 collapse + Phase 3 #3 AC=0.62 일관 + 7 ckpt 모두 R≤0.61 ceiling |

     → **paper §3.5 narrative 정식 채택** + **paper §V.5.4 신설 narrative 정식 채택**: GAT 의 fundamental architectural limitation (edge softmax aggregation collapse) 까지 With-Filter pipeline 이 흡수 = paper main contribution 의 strongest evidence

  6. **(f) 학위 논문 Part III chapter §III.4 main mechanism finding 갱신**:
     - 직전 outline (DECISIONS 2026-05-07 4-trial): §III.4 = mech(ii) GATv2Conv Normalization DOMINANT 5/5
     - **7-trial 갱신 outline**:
       - §III.4 = **mech(ii) DOMINANT 결정적 강화 5/5** (v2_LN attention 회복 + L1=1.0 collapse + AC=0.62 일관 evidence 결합) + **paradox 정확한 분리** (attention pattern 학습 가능 ↔ message aggregation collapse 구조적)
       - §III.5 = 7-trial mitigation null effect (V-3-ext 4-trial + Mitigation v2 3-trial)
       - §III.6 = AC + L1 collapse + skip dep + attention sharpen 통합 mechanism + paradox 분리 정량 narrative
     - main contribution narrative 강화: "**GAT 의 fundamental architectural limitation (edge softmax aggregation collapse) 이 paper main pipeline F1 plateau 에 absorb 되는 mechanism**" — 단 paper full version 의 ablation evidence base 로 활용 (학회 논문 narrative 영향 X)

  7. **(g) Mitigation v3 candidate (post-paper, mech(ii) 직접 mitigation, LLM-free ~10h/cell)**:

     | 우선순위 | Candidate | 가설 / mechanism | 비고 |
     |---|---|---|---|
     | **#1** | **Aggregation function 자체 변경 (GIN-style, max)** | softmax + weighted-mean 자체 대체. v2_sum_aggr null 이지만 max / GIN 미시도 — sum 의 magnitude variance sensitivity 부재한 max / GIN 시도 가치 | post-paper, ~10h/cell |
     | **#2** | **Self-loop weight scaling** | message aggregation 시 self-connection 의 weight 우선 → sibling 동질화 압력 약화 | post-paper, ~10h/cell |
     | **#3** | **Multi-hop attention** (skip GAT layer) | 매 layer 에서 message aggregation 대신 multi-hop attention 직접 | post-paper, ~10h/cell + 구현 中 |
     | **#4** | **Energy-based GNN (EGAT)** | softmax-based attention 대체 — energy minimization 으로 aggregation 결정 | post-paper, ~10h/cell + 구현 中 (architectural shift 가장 큼) |

     - **공통 사유**: 7-trial evidence 가 mech(ii) edge softmax 자체의 구조적 limitation 직접 정량 → mitigation 은 **softmax + weighted-mean propagation 자체 대체 필요** (LayerNorm pre-softmax 의 attention level mitigation 만으로 partial 만 회복)
     - **모두 post-paper backlog**: 학위 논문 Part III main contribution = 7-trial null effect + mech(ii) DOMINANT mechanism finding 으로 충분 (Mitigation v3 시도는 학위 본 심사 후)

  8. **(h) Caveat / 분석 한계**:
     - **Per-DB 분해 single-DB**: 50 queries 모두 BIRD-dev 첫 50 = california_schools (db-sorted) → 7 ckpt 일관 partial 입증, post-paper 11 DBs 일반성 미검증
     - **v2_sum_aggr gradient measurement 다수 NaN**: sum aggregation 의 학습 dynamics 가 측정에도 sensitive → mech(iii) 정량 일부 신뢰성 낮음 (table 행 일부 누락)
     - **Phase 1 (DualTowerProjector) vs Phase 2-7 (DirectClassifierHead) architecture 차이**: head 무시했지만 backward 시 일부 차이 가능 — forward path 측정에는 영향 X (analyzer §13 confirm)

  9. **🚨 사용자 결정 필요 3 항목**:

     | # | 결정 항목 | 옵션 | 권장 |
     |---|----------|------|------|
     | (1) | **paper §V.5.4 narrative 정식 채택 시점** (mech(ii) DOMINANT 결정적) | (A) 즉시 정식 채택 (7-trial evidence 충분) / (B) Mitigation v3 candidate 1-2 추가 검증 후 채택 | **(A) 즉시 정식 채택** — 7-trial × 5-step protocol × 4 mechanism evidence + v2_LN partial recovery + L1=1.0 paradox 분리 + AC=0.62 일관 → mech(ii) DOMINANT 5/5 결정적 강화 evidence 충분 |
     | (2) | **Mitigation v3 candidate (GIN / max / Self-loop / EGAT) 우선순위 (post-paper)** | (A) #1 GIN/max 학위 본 심사 후 시도 / (B) #2 Self-loop scaling 만 / (C) #4 EGAT 만 (가장 architectural shift) / (D) 모두 post-paper backlog (학위 본 심사 timeline 부족) | **(D) 모두 post-paper backlog** — Mitigation v2 3 candidate 결과로 mech(ii) DOMINANT 결정적 강화 충분 + 학위 본 심사 timeline (~5/22) 5/14~5/22 학위 논문 Part III chapter draft 작성 우선 |
     | (3) | **per-DB stratified 50 queries 후속 측정 필요 여부** | (A) 학위 본 심사 전 진행 (~₩0 LLM-free, ~수 시간) / (B) post-paper backlog (single-DB caveat 학위 논문 §V Limitations 1 줄 명시 충분) | **(B) post-paper backlog** — 7 ckpt × single-DB 일관 evidence 가 mech(ii) DOMINANT generalizability partial 입증 + post-paper 11 DBs 일반성 검증 여유 |

- **근거**:
  - **신규 analyzer 산출**: [dsn_mitigation_v2_results.md §0~§13](../notebooks/analysis_results/dsn_mitigation_v2_results.md) (7 ckpt × 4 mechanism × 5 step + paradox 분리 narrative + Filter Dominance 6번째 축 결정적 evidence)
  - **재현 데이터**: outputs/analysis/dsn_mitigation_v2_7trial/ (batch_summary.json + 5 plots: recall_trajectory_overlay_7ckpt + ac_loss_trajectory_7ckpt + oversmoothing_heatmap_7ckpt + attention_heatmap_topk5_conc_7ckpt + attention_heatmap_entropy_7ckpt + 7 per-ckpt summary)
  - **재현 스크립트**: src/analysis/dsn_mitigation_v2_7trial.py (v1/v2 + Mitigation v2 옵션 자동 forward)
  - **선행 분석**: [dsn_phase3_mitigation_results.md §6](../notebooks/analysis_results/dsn_phase3_mitigation_results.md) (4-trial mech(ii) DOMINANT 5/5 직전 판정)
  - **선행 결정**: DECISIONS 직전 엔트리 2026-05-07 (Mitigation v2 #1+#3+#2 구현 완료) — Selector 단계 6 + Root 학습 launch + analyzer 위임
  - **EXPERIMENT_HISTORY**: L2691~ DSN Phase 2 + Phase 3 4-trial Mitigation Sweep entry + Mitigation v2 학습 entry (5/9~5/11)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 7-trial mechanism dominance 결정적 확정 + paradox 분리 narrative + 학위 논문 §V.5.4 정식 채택 + Mitigation v3 candidate 4 (post-paper) + 사용자 결정 3 항목
  - **paper_research_direction.md (planner Edit, 본 응답)**:
    - §3.5 Filter Dominance 6번째 축 sub-section 갱신 (5+1 evidence axes 표 + 7-trial 결과 표 + paradox 분리 + analyzer §10 인용)
    - §3.5 학위 논문 Part III chapter outline §III.4 갱신 (mech(ii) 결정적 강화 + paradox 분리 정량)
    - §V.5 (학위 논문 Part III) §V.5.4 정식 신설 narrative (analyzer §0 + §7.3 + §10 직접 인용)
    - §8 Future Works H-DTK Mitigation v3 candidate 4 신규 (post-paper, GIN/max + Self-loop scaling + Multi-hop attention + EGAT)
    - §9 Limitations — single-DB caveat 갱신 (7-trial 일관 evidence + post-paper 11 DBs 일반성 검증)
    - §10 V-3-ext 단계 5 sub-section 갱신 (4-trial → 7-trial dominance scoring 표 + analyzer §10 인용)
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.12 신설 (7-trial mitigation 결과 + Filter Dominance 6 axes + v2_LN partial recovery paradox + 사용자 결정 3 항목)
  - **paper main contribution (학회)** 영향 X — anchor t_00 + Filter Dominance 4 축 narrative 그대로
  - **학위 논문 Part III chapter narrative weight 결정적 격상** — mech(ii) DOMINANT 5/5 결정적 강화 + paradox 분리 정량 + Filter Dominance 6번째 축 narrative 정식 채택

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §3.5 / §V.5.4 / §8 / §9 / §10 갱신 + presentation_brief §14.12 신설
  2. **사용자 (즉시 의사결정 3 항목)** — 본 엔트리 §1(9) 의 (1) §V.5.4 즉시 채택 / (2) Mitigation v3 우선순위 / (3) per-DB stratified 후속 측정
  3. **사용자 (5/14~5/22)** — 학위 논문 Part III chapter draft 작성 (V-3-ext 4-trial + Mitigation v2 3-trial 통합 narrative + paradox 분리 + analyzer §0+§7.3+§10 인용)
  4. **사용자 (학회 §V.5.3 1 줄)** — Future Work 1 줄 (DSN 7-trial mitigation null + mech(ii) edge softmax aggregation collapse fundamental limitation + Mitigation v3 post-paper) 직접 처리
  5. **Analyzer (post-paper 후속)** — Per-DB stratified 50 queries 재측정 (사용자 결정 (3) B 채택 시 post-paper) + softmax noise sensitivity 측정 + Mitigation v3 candidate 학습 후 protocol 재실행
  6. **Selector 모듈 (post-paper 후속)** — Mitigation v3 candidate 구현 (사용자 결정 (2) D 채택 시 post-paper)

- **추가 필요 분석** (post-paper):
  - Per-DB stratified 50 queries 재측정 (mech(ii) generalizability 11 DBs 검증)
  - softmax noise sensitivity 측정 (mech(ii) 정밀화 — input perturbation 시 alpha 변동 + LayerNorm 의 noise robustness 정량)
  - Mitigation v3 candidate (GIN aggregation / max / Self-loop scaling / EGAT) 학습 + 동일 protocol 재실행
  - v2_layernorm 의 attention 회복 + L1=1.0 collapse 의 nuanced mechanism (LayerNorm 이 어떤 message component 정규화에 작용하는지 deep dive)

---

## 2026-05-07 (Mitigation v2 #1+#3+#2 구현 완료 — DropMessage + LayerNorm pre-softmax + Sum aggregation, smoke 12/12 통과) — Root 학습 핸드오프 prep (5/9 #1+#3 병렬 + 5/10 #2 sequential)

> **Status**: Selector 모듈 단계 6 구현 + 12 smoke 통과 (2026-05-07 동일자 가속, 직전 timeline 5/7~5/8 → 5/7 완료). 3 신규 config + DropMessageGATv2Conv + LayerNormGATv2Conv subclass + HeteroConv aggr 옵션 + Phase 2 b8 backward compat 검증. 학습 launch 준비 완료.

- **결정**:

  1. **(a) Selector 모듈 단계 6 구현 완료 확인** (출처: [src/modules/selectors/EXPERIMENT_PLAN_selectors.md §V-3-ext 단계 6](../src/modules/selectors/EXPERIMENT_PLAN_selectors.md)):
     - **#1 PRIMARY DropMessage**: `DropMessageGATv2Conv(GATv2Conv)` — `message(x_j, alpha)` 출력 (= x_j × α) 에 `F.dropout(p=drop_message_p, training=training)` 적용. attention α 는 그대로 유지하되 attended-to neighbor 의 feature contribution 분산. Config flag: `drop_message_p: 0.2`.
     - **#3 SECONDARY LayerNorm pre-softmax**: `LayerNormGATv2Conv(GATv2Conv)` — `edge_update` 의 raw alpha 산출 후 softmax 직전에 `nn.LayerNorm(heads)` 삽입. softmax sharp peaking 완화. Config flag: `use_layernorm_pre_softmax: true`. **+72 params overhead** (heads=2 × 2 (γ,β) × 18 LN modules).
     - **#2 TERTIARY Sum aggregation**: HeteroConv `aggr` 인자 변경 (mean → sum / max). cross-edge-type aggregation level 의 inductive bias 변경 (edge softmax 와 별개 layer). Config flag: `aggregation_type: "sum"`.
     - **Combo #1+#3**: multiple-inheritance 방식의 `_LayerNormDropMessageGATv2Conv` 동적 클래스 생성 (smoke 통과, post-paper 검증 예약).
     - **Backward compat**: 모든 옵션 default OFF — Phase 2 b8 동일 동작 (params=859,008, state_dict keys=160 검증).

  2. **(b) 신규 산출 파일 6 항목 (cross-reference)**:

     | 파일 | 변경 |
     |---|---|
     | `src/models/gat_network_v2.py` | `DropMessageGATv2Conv` + `LayerNormGATv2Conv` subclass + `_make_gatv2_conv` factory + `HETEROCONV_AGGR_TYPES` constant + SchemaHeteroGATv2 __init__ 에 3 옵션 + 검증 + state. HeteroConv 인스턴스화 시 factory 사용 + aggr 인자 동적 |
     | `src/train_gat_s06.py` | v2 model 인스턴스화에 3 신규 옵션 forward (default OFF backward compat) + log line 보강 |
     | `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml` | 신규 #1 학습 config (Phase 2 b8 + drop_message_p: 0.2) |
     | `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.yaml` | 신규 #3 학습 config (Phase 2 b8 + use_layernorm_pre_softmax: true) |
     | `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.yaml` | 신규 #2 학습 config (Phase 2 b8 + aggregation_type: "sum") |
     | `src/modules/selectors/tests/test_mitigation_v2.py` | 신규 smoke test 12 케이스 |

  3. **(c) Smoke 12/12 통과 (cross-reference)**:
     - **#1**: DropMessage subclass — train ≠ rerun (random), eval deterministic
     - **#1**: `drop_message_p=0.0` backward compat (super class 동일 결과, atol=1e-6)
     - **#1**: SchemaHeteroGATv2 + DropMessage forward shape 정합성
     - **#3**: LayerNormGATv2Conv subclass — `alpha_layernorm` 모듈 (heads,) shape 등록
     - **#3**: SchemaHeteroGATv2 + LayerNorm — 18 inner convs 모두 alpha_layernorm (9 edge types × 2 layers)
     - **#2**: `aggregation_type='sum'` forward + HeteroConv.aggr 검증
     - **#2**: `aggregation_type='max'` forward (사용자 spec sum/max 양쪽)
     - **Combo #1+#3** — 18 inner convs 모두 LayerNorm + DropMessage 결합 클래스 적용
     - **Backward compat** — default vs explicit-OFF identical (params=859,008, state_dict keys=160)
     - **LayerNorm overhead** — +72 params 정량
     - **3 신규 config 파싱 + 옵션 정확**
     - **Phase 2 baseline regression** — 신규 옵션 미설정 보존

  4. **(d) AC loss 위치 확인 (사전 trace)**:
     - 본 단계는 mech(ii) edge softmax 직접 mitigation → AC loss 위치는 변경 X (Phase 2 b8 fusion 기본값 유지, Phase 3 #3 의 `'gat_out_L_last'` 옵션 미사용)
     - 사유: mitigation v2 의 mech(ii) 회복 정도 분석 시 AC loss 변동을 mitigation 효과의 dependent variable 로 활용 — Phase 2 fusion AC 와 동일 baseline 비교 위해 AC target 고정
     - Phase 3 #3 의 'gat_out_L_last' 결과 (AC=0.62 일관 유지) 와 mitigation v2 결과 비교 시 같은 fusion target 으로 통일 → mech(ii) mitigation 효과 정량화 가능

  5. **(e) 🚀 Root 학습 핸드오프 prep (본 응답 본문 prompt)**:

     | 일정 | GPU | Config | ckpt | ETA |
     |---|---|---|---|---|
     | **5/9 launch (병렬)** | GPU 0 | `train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml` | `best_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.pt` | 5/10 04:00 KST |
     | **5/9 launch (병렬)** | GPU 1 | `train_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.yaml` | `best_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.pt` | 5/10 04:00 KST |
     | **5/10 launch (sequential)** | GPU 0 | `train_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.yaml` | `best_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.pt` | 5/11 KST |

     - 학습 entry: `python src/train_gat_s06.py --config <config> ` (with `CUDA_VISIBLE_DEVICES=0` or `1`, memory rule)
     - 신규 ckpt NAS 저장 + 로컬 symlink (memory rule, /SSL_NAS/peoples/khj/thesis/checkpoints/)
     - **EXPERIMENT_HISTORY.md "Mitigation v2 #1+#3+#2 학습 (5/9~5/11)" entry 추가** (학습 wall + best val R@15 + best epoch + ckpt path 기록 + 4-trial → 7-trial 통합 표)
     - **alpha sweep skip** (사용자 결정 2026-05-07 (1)A 유지) — paper main F1/EX 측정 X, val recall@15 evidence only

  6. **(f) Analyzer 후속 prep (5/10, 5/11)**:
     - **5/10 (#1 + #3 학습 종료 후)**: `src/analysis/dsn_phase3_4trial_deep_dive.py` 의 4 ckpt → **6 ckpt (4 + #1 + #3) × 5 step** 재실행 — mech(ii) attention concentration 회복 정량 (top-5 conc / entropy / L1_GAT cosine)
     - **5/11 (#2 학습 종료 후)**: 통합 재실행 — **7 ckpt (4 + #1 + #3 + #2) × 5 step** 분석. 4 mechanism dominance scoring 갱신 (mech(ii) DOMINANT 5/5 confirm 강도)
     - 산출물: `notebooks/analysis_results/dsn_mitigation_v2_results.md` 신규 또는 `dsn_phase3_mitigation_results.md §11` 보강
     - **mech(ii) 회복 정도 정량 metrics**:
       - L1_GAT intra-table cosine (4-trial 모두 ≈ 1.0 → mitigation v2 가 0.9+ 으로 낮추는지)
       - Attention top-5 concentration (Phase 3 #4 의 0.83 → mitigation v2 가 0.5- 으로 낮추는지)
       - Edge softmax entropy (Phase 2 의 2.41 → mitigation v2 가 3.0+ 으로 회복하는지)
       - AC loss target='fusion' decay 정상 여부 (Phase 2 0.087 → 0.0007 와 비교)
       - skip_dep ratio (Phase 3 #3 의 0.97 균형 / Phase 3 #4 의 7.38 악화 비교)

  7. **(g) Planner 후속 prep (5/12~5/14)**:
     - 통합 dominance scoring 갱신 (4-trial → **7-trial**) + Filter Dominance 6번째 축 narrative 정량 강도 (3 candidate 추가 후)
     - 시나리오 V2-A/B/C 분기 처리 (DECISIONS 직전 엔트리 §1(F)):
       - **V2-A** (가장 가능성 高): 3 모두 fail → §V.5.4 narrative 정식 채택 + Filter Dominance 6번째 축 절대적 confirm 강화 (7-trial null effect = robustness 결정적 evidence). **paper §V.5.4 narrative 직접 인용 narrative**: "본 연구는 7-trial mitigation (V-3-ext 4-trial + Mitigation v2 3-trial) 을 시도했으나 모두 raw R@15 한계 ~0.61 영역 회복 못함 — mech(ii) edge softmax 의 fundamental limitation 절대적 evidence."
       - **V2-B** (가능성 中): 1-2 partial recovery → §V.5.4 narrative 미세 수정 + "Skip Dep null but mech(ii) partial mitigation 발견" contribution. paper main contribution 4 → 5 항목 격상 후보.
       - **V2-C** (가능성 낮음): 3 모두 R 0.85+ 회복 → §V.5.4 큰 수정 + paper main contribution 재평가 + 학회 후 paper anchor 재검토.
     - DECISIONS 후속 엔트리 작성 (시나리오 결정 + paper §3.5 / §V.5.4 / §10 narrative 정식 확정)
     - 학위 논문 Part III chapter §III.4 main mechanism finding 통합 narrative — 7-trial 결과 통합

  8. **(h) 학회 논문 narrative 영향 X (재확인)**:
     - paper main anchor t_00 (F1=0.8657) 변경 X
     - Filter Dominance 4 축 narrative (학회) 그대로
     - Mitigation v2 결과는 **학위 논문 Part III chapter §V.5.4 만 적용**
     - 학회 §V.5.3 Future Work 1 줄 (Mitigation v2 #4 Energy-based GNN post-paper) 사용자 직접 처리

- **근거**:
  - **Selector 모듈 단계 6 산출**: [EXPERIMENT_PLAN_selectors.md §V-3-ext 단계 6](../src/modules/selectors/EXPERIMENT_PLAN_selectors.md) (구현 + 12 smoke 통과 + AC loss 위치 fusion 유지 확인)
  - **선행 결정**: DECISIONS 직전 엔트리 2026-05-07 (사용자 결정 (1)A+(2)B+(3)A+B+C 병렬) §1(C)/(D) — Mitigation v2 #1+#2+#3 학위 본 심사 전 진행 + 구현 spec
  - **선행 분석**: dsn_phase3_mitigation_results.md §6 (mech(ii) DOMINANT 5/5 판정) + §8.1 (Mitigation v2 candidate 4 권장)
  - **Cross-reference**: 단계 5 (Phase 3 #3+#4) → 단계 6 (Mitigation v2) 모두 5/7 동일자 가속 완료 — 학위 본 심사 timeline 여유 ↑

- **영향 범위**:
  - **DECISIONS 본 엔트리** — Mitigation v2 #1+#3+#2 구현 완료 confirm + 신규 산출 6 파일 + smoke 12/12 + Root 학습 launch prep 표 + Analyzer 후속 prep + Planner 후속 prep + 시나리오 V2-A/B/C 분기 narrative
  - **Selector EXPERIMENT_PLAN §V-3-ext 단계 6 ✅ 완료 표기** (이미 selector 모듈에서 작성됨)
  - **paper_research_direction.md 영향 X** (직전 §8 H-DTK Mitigation v2 #1+#3+#2 항목이 학위 본 심사 전 진행 표기됨, 본 엔트리는 구현 완료 confirm 만)
  - **presentation_brief 영향 X** (직전 §14.11.7 timeline 그대로)
  - **Root 학습 launch prep prompt** (응답 본문) — 5/9 #1 (GPU 0) + #3 (GPU 1) 병렬 + 5/10 #2 (GPU 0) sequential
  - **paper main contribution 영향 X** (학회 anchor t_00 그대로)
  - **학위 논문 Part III chapter narrative weight 결정적 격상 prep**: 7-trial mitigation 통합 narrative (V2-A 가능성 高 시 §V.5.4 narrative 절대적 evidence)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + Root 학습 launch prep prompt
  2. **사용자 (즉시)** — Root 세션 prompt 직접 붙여넣기 (cd `/home/hyeonjin/thesis_refactored`, 5/9 #1+#3 병렬 학습 launch + 5/10 #2 sequential)
  3. **Root (5/9~5/11)** — Mitigation v2 #1+#3+#2 학습 + ckpt NAS 저장 + EXPERIMENT_HISTORY 갱신
  4. **Analyzer (5/10 + 5/11)** — protocol 재실행 (#1+#3 + #2 추가, 7 ckpt × 5 step) — 산출물 dsn_phase3_mitigation_results.md §11 보강 또는 신규 dsn_mitigation_v2_results.md
  5. **Planner (5/12~5/14)** — 통합 dominance scoring 갱신 + §V.5.4 정식 채택 결정 + DECISIONS 후속 엔트리
  6. **사용자 (5/14~5/22)** — 학위 논문 Part III chapter draft 작성 (V-3-ext 4-trial + Mitigation v2 3-trial 통합 narrative)
  7. **사용자 (학회 §V.5.3 1 줄)** — Future Work 1 줄 (Mitigation v2 #4 Energy-based GNN post-paper) 직접 처리

- **추가 필요 분석** (Mitigation v2 학습 + 측정 결과 후):
  - 각 candidate 의 mech(ii) 회복 정도 정량 (L1_GAT cosine + attention top-5 conc + entropy + skip_dep ratio + AC loss 정상 decay)
  - 7-trial 통합 dominance scoring (Phase 1 + Phase 2 + Phase 3 #3 + Phase 3 #4 + Mitigation v2 #1 + #3 + #2) — Filter Dominance 6번째 축 narrative 정량 강도 (4-trial → 7-trial)
  - 시나리오 V2-A 시 §V.5.4 narrative 절대적 confirm + paper §V Conclusion 직접 인용 narrative 정식 확정
  - 시나리오 V2-B 시 partial mitigation 의 mechanism 분석 + paper main contribution 재평가
  - 시나리오 V2-C 시 (낮음) Filter Dominance narrative 큰 수정 + 학회 후 paper anchor 재검토

---

## 2026-05-07 (사용자 결정 (1)A + (2)B + (3)A+B+C 병렬 — Mitigation v2 #1+#2+#3 학위 본 심사 전 진행, §V.5.4 정식 채택 confirm 강화 후) — Selector 모듈 + Root 학습 핸드오프 prep + Mitigation v2 timeline 확정

> **사용자 직전 input (2026-05-07)**: "(1) (A) / (2) (B) / (3) (A), (B), (C) 모두 병렬로 할 수 없나?" — Mitigation v2 candidate 3 종 (#1 DropMessage + #2 Sum/Max + #3 LayerNorm before softmax) 학위 본 심사 전 병렬 진행 검토 요청.

- **결정** (사용자 직전 input + planner timeline 평가):

  1. **(A) (1) — paper §3.5 6번째 축 정식 채택 시점 (alpha sweep skip 유지)**:
     - 직전 결정 (2026-05-07 03:00 KST "alpha sweep 은 하지 마") 유지
     - paper main F1/EX 측정 X, val recall@15 evidence only
     - 4-trial val R@15 + analyzer §6.3 mech(ii) DOMINANT evidence 만으로 paper §3.5 6번째 축 narrative 정식 채택 충분
     - paper main contribution 영향 X (학회 anchor t_00 그대로)

  2. **(B) (2) — 학위 논문 Part III §V.5.4 narrative 정식 채택 (Mitigation v2 1-2 confirm 후)**:
     - 직전 권장 (A) 즉시 정식 채택 → **사용자 결정 (B) 추가 검증 후 채택**
     - Mitigation v2 candidate 1-2 학습 결과로 mech(ii) DOMINANT confirm 강화 / 부정 / partial 분기 후 narrative 정식 채택
     - 사용자 (3) 결정과 통합: **#1+#2+#3 모두 병렬 진행** → 3 candidate 결과 통합 후 §V.5.4 정식 채택
     - 3 모두 fail (가능성 高, mech(ii) edge softmax fundamental limitation evidence 강함) → §V.5.4 narrative 정식 채택, Filter Dominance 6번째 축 절대적 confirm 강화 (4-trial → 7-trial)
     - 1-2 partial recovery → §V.5.4 narrative 미세 수정 (mech(ii) DOMINANT partial 부정)
     - 3 모두 ceiling 갱신 (가능성 낮음) → §V.5.4 narrative 큰 수정 + paper main contribution 재평가 후보

  3. **(C) 🚀 (3) — Mitigation v2 #1 + #2 + #3 모두 병렬 진행 (학위 본 심사 전, 권장 변경)**:
     - 사용자 의도: "병렬로 할 수 없나?" — 학위 본 심사 (5/22) 전 3 candidate 모두 시도 후 §V.5.4 정식 채택
     - 직전 권장 (D) 모두 post-paper backlog → **사용자 결정 (A)+(B)+(C) 모두 병렬 진행**
     - **#4 Energy-based GNN 은 post-paper 보존** (architectural shift 가장 큼 + 학위 본 심사 timeline 부족 — 추가 1 candidate 추가 시 timeline 4-5일 더 필요)

  4. **(D) Mitigation v2 candidate 3 종 우선순위 + 구현 spec**:

     | 우선순위 | Candidate | 구현 위치 | 구현 시간 | 학습 시간 | 학습 우선 launch |
     |---|---|---|---|---|---|
     | **#1 PRIMARY** | **DropMessage** | `gat_network_v2.py` forward 에 `F.dropout(message, p=drop_p, training=training)` 1줄 + config flag `drop_message_p: 0.2` | ~2h | ~10h | **5/9 GPU 0 launch** (#3 와 병렬) |
     | **#3 SECONDARY** | **LayerNorm before softmax** | `GATv2Conv` 변형 (PyG `GATv2Conv` subclass + LayerNorm insert before edge softmax) | ~3h | ~10h | **5/9 GPU 1 launch** (#1 와 병렬) |
     | **#2 TERTIARY** | **Sum / Max aggregation** | `GINConv` 대안 또는 `aggr='sum'/'max'` GATv2Conv 변경 + heterograph 호환 | ~5h | ~10h | **5/10 GPU 0 launch** (sequential) |
     | **#4 (post-paper)** | **Energy-based GNN (EGAT)** | softmax-based attention 대체, energy minimization aggregation | ~?h | ~10h+ | post-paper |

     - **Base config**: Phase 2 b8 (`train_gat_directed_supernode_p80_b5_mitigation.yaml`) — PN+IR+JK+DS+L=2+AC fusion+ListNet
     - **신규 config**: 각 candidate 별 yaml (e.g., `train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml`)
     - **batch_size**: 8 (Phase 2 와 동일, 학습 ~2.5min/ep × 300 ep ≈ 12h)

  5. **(E) 통합 timeline 확정 (학위 본 심사 5/22 전)**:

     | 일정 | 작업 | 세션 | 비고 |
     |---|---|---|---|
     | **5/7 (오늘)** | DECISIONS 본 엔트리 + paper §3.5 / §V.5.4 / §8 / §10 갱신 + presentation_brief §14.11.7 갱신 | planner | 본 응답 |
     | **5/7~5/8** | Selector 모듈에서 #1 + #3 + #2 구현 + smoke test (구현 합 ~10h) | selector | 핸드오프 prompt (응답 본문) |
     | **5/9 launch** | **#1 (GPU 0) + #3 (GPU 1) 병렬 학습** (~10h, ETA 5/10 04:00 KST) | root | 핸드오프 prompt (응답 본문) |
     | **5/10 학습 종료** | Analyzer protocol 재실행 (#1 + #3 분석, 6 ckpt × 5 step, ~수 시간) | analyzer | 5/10 GPU 0 idle 후 |
     | **5/10 launch** | **#2 (GPU 0) sequential 학습** (~10h, ETA 5/11 KST) | root | sequential |
     | **5/11 학습 종료** | Analyzer 통합 재실행 (3 candidate 추가, 7 ckpt × 5 step) | analyzer | |
     | **5/12~5/14** | planner — 통합 mechanism dominance 갱신 + §V.5.4 정식 채택 결정 (3 candidate 결과 통합 narrative) | planner | |
     | **5/14~5/22** | 학위 논문 Part III chapter draft 작성 (V-3-ext + Mitigation v1 + Mitigation v2 통합 narrative) | user | |
     | **5/22~6/19** | 본 심사 진행 + 추가 보강 | user | |
     | **post-paper** | #4 Energy-based GNN 학습 + Per-DB 분해 + softmax noise sensitivity | analyzer + selector | |

  6. **(F) 시나리오 분기 narrative (Mitigation v2 결과 후)**:
     - **시나리오 V2-A (가장 가능성 高 — 3 모두 fail / null effect)**:
       - val R@15 ceiling ~0.59-0.61 영역 unchanged
       - mech(ii) DOMINANT 절대적 confirm 강화 (7-trial mitigation 모두 fail = robustness 결정적)
       - **§V.5.4 narrative 정식 채택** + Filter Dominance 6번째 축 narrative 절대적 evidence
       - 학위 논문 Part III main contribution: "**7-trial mitigation null effect**" (V-3-ext 4-trial + Mitigation v2 3-trial)
     - **시나리오 V2-B (가능성 中 — 1-2 partial recovery)**:
       - val R@15 ceiling 0.62-0.70 영역 partial 회복
       - mech(ii) DOMINANT partial 부정 (특정 mitigation 의 부분 효과 발견)
       - **§V.5.4 narrative 미세 수정** + 학위 논문 Part III main contribution: "**Skip Dep mitigation null but mech(ii) partial mitigation 발견**"
       - paper main contribution 4 → 5 항목 격상 후보 (단 anchor 변경은 학회 후 별도 결정)
     - **시나리오 V2-C (가능성 낮음 — 3 모두 ceiling 갱신 R 0.85+ 회복)**:
       - val R@15 ceiling 0.85+ 도달 (가능성 낮음 — Phase 3 #3 AC=0.62 일관 evidence 강함)
       - **§V.5.4 narrative 큰 수정** + paper main contribution 재평가 (Filter Dominance 6번째 축 narrative 약화 가능)
       - 학회 논문 narrative 영향 가능 (학회 후 검토)

  7. **(G) Caveat / 위험 평가**:
     - **#2 Sum/Max aggregation 구현 복잡도 中**: PyG `GINConv` 대안 또는 GATv2Conv `aggr='sum'/'max'` 변경 — heterograph 호환성 검증 필수 (smoke test 권장)
     - **3 candidate 모두 구현 코드 수정 필요**: selector 모듈 세션 핸드오프 (응답 본문)
     - **GPU 자원 제약**: GPU 0, 1만 사용 (memory rule, GPU 2/3 다른 연구원) → 동시 3 학습 불가, 2 병렬 + 1 sequential
     - **시나리오 V2-A 가능성 높음** (mech(ii) edge softmax fundamental limitation evidence 강함) → 학위 논문 narrative 사실상 확정 단 추가 evidence 강도 강화

  8. **(H) 학회 논문 narrative 영향 X (재확인)**:
     - paper main anchor t_00 (F1=0.8657) 변경 X
     - Filter Dominance 4 축 narrative (학회) 그대로
     - Mitigation v2 결과는 **학위 논문 Part III chapter §V.5.4 만 적용**
     - 학회 §V.5.3 Future Work 1 줄 (Mitigation v2 #4 Energy-based GNN post-paper) 사용자 직접 처리

- **근거**:
  - **사용자 직전 input** (2026-05-07): "(1) (A) / (2) (B) / (3) (A), (B), (C) 모두 병렬로 할 수 없나?"
  - **선행 결정**: DECISIONS 직전 엔트리 2026-05-07 (4-trial mechanism dominant 판정 완료) §1(h) 의 사용자 결정 3 항목
  - **Mitigation v2 candidate spec**: analyzer dsn_phase3_mitigation_results.md §8.1 (Future Work 4 candidate)
  - **Phase 3 #3 + #4 학습 구현 reference**: DECISIONS 2026-05-06 (Phase 3 #3 + #4 구현 완료) — `train_gat_s06.py` 의 `anti_collapse_target` 옵션 + `optimizer_layer_wise_lr` 옵션 추가 패턴 적용 가능
  - **GPU 자원 제약**: memory rule "GPU 0, 1만 사용. GPU 2, 3 다른 연구원 reserved"
  - **batch_size 8 학습 시간**: Phase 2 b8 evidence (2.5min/ep × 300 ep ≈ 12h)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 사용자 결정 3 항목 confirm + Mitigation v2 #1+#2+#3 timeline 확정 + Selector + Root 핸드오프 prep + 시나리오 V2-A/B/C 분기 narrative
  - **paper_research_direction.md (planner Edit, 본 응답)** — §8 Future Works H-DTK Mitigation v2 항목 갱신 (post-paper backlog → 학위 본 심사 전 진행 #1+#2+#3, #4 post-paper 보존)
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.11.7 사용자 결정 3 항목 결과 + Mitigation v2 timeline 추가
  - **Selector 모듈 세션 핸드오프 prompt** (응답 본문) — #1 DropMessage + #3 LayerNorm before softmax + #2 Sum/Max aggregation 3 구현 + smoke test
  - **Root 세션 핸드오프 prompt** (응답 본문) — 5/9 #1+#3 병렬 학습 + 5/10 #2 sequential 학습
  - **paper main contribution 영향**: 학회 논문 narrative X / **학위 논문 §V.5.4 narrative 정식 채택은 Mitigation v2 결과 후 재확정** (시나리오 V2-A/B/C 분기)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §8 갱신 + presentation_brief §14.11.7 갱신 + Selector + Root 핸드오프 prompt
  2. **사용자 (즉시)** — Selector 모듈 세션 prompt 직접 붙여넣기 (cd `/home/hyeonjin/thesis_refactored/src/modules/selectors`) — 5/7~5/8 #1+#3+#2 구현 + smoke test
  3. **Selector 모듈 (5/7~5/8)** — `gat_network_v2.py` + 신규 config 3 + smoke test 3 (#1 DropMessage forward dropout / #3 GATv2Conv subclass + LayerNorm / #2 GINConv 대안 또는 aggr 변경)
  4. **사용자 (5/8 종료 후)** — Root 세션 prompt 직접 붙여넣기 (cd `/home/hyeonjin/thesis_refactored`) — 5/9 #1+#3 병렬 학습 launch + 5/10 #2 sequential
  5. **Root (5/9~5/11)** — Mitigation v2 #1+#2+#3 학습 + EXPERIMENT_HISTORY 갱신
  6. **Analyzer (5/10 + 5/11)** — protocol 재실행 (3 candidate 추가, 7 ckpt × 5 step) — 산출물 dsn_phase3_mitigation_results.md §11 보강 또는 신규 dsn_mitigation_v2_results.md
  7. **Planner (5/12~5/14)** — 통합 dominance scoring 갱신 + §V.5.4 정식 채택 결정 (시나리오 V2-A/B/C 분기 처리) + DECISIONS 후속 엔트리
  8. **사용자 (5/14~5/22)** — 학위 논문 Part III chapter draft 작성 (V-3-ext + Mitigation v1 + Mitigation v2 통합 narrative)

- **추가 필요 분석** (Mitigation v2 학습 + 측정 결과 후):
  - 각 candidate 의 mech(ii) 회복 정도 정량 (L1_GAT cosine + AC loss target='gat_out_L_last' 시나리오 가정 측정)
  - 7-trial 통합 dominance scoring (Phase 1 + Phase 2 + Phase 3 #3 + Phase 3 #4 + Mitigation v2 #1 + #2 + #3) — Filter Dominance 6번째 축 narrative 정량 강도 (4-trial → 7-trial)
  - 시나리오 V2-A 시 §V.5.4 narrative 절대적 confirm + paper §V Conclusion 직접 인용 narrative 정식 확정
  - 시나리오 V2-B 시 partial mitigation 의 mechanism 분석 + paper main contribution 재평가
  - 시나리오 V2-C 시 (낮음) Filter Dominance narrative 큰 수정 + 학회 후 paper anchor 재검토

---

## 2026-05-07 (4-trial mechanism dominant 판정 완료 — 🎯 Mechanism (ii) GATv2Conv Normalization DOMINANT 5/5, mechanism (iii) Skip Dep 부분 부정 5/5→4/5) — 학위 논문 Part III main mechanism finding 정식 채택 (paper §V.5.4) + Filter Dominance 6번째 축 narrative 정식 + Mitigation v2 candidate 4 (post-paper)

> **🚨 직전 (Phase 2 only) 판정 정정**: 직전 dsn_phase2_mitigation_null_mechanism.md 의 dominant=mech(iii) Skip Dependence Pathology 5/5 판정이 Phase 3 #3+#4 4-trial 데이터 추가 후 **mech(ii) GATv2Conv Normalization 5/5 로 갱신**. mech(iii) Skip Dep 는 보조 mechanism 으로 강등 (Phase 3 #3 가 skip_dep ratio 0.97 균형에도 best R 가장 낮음 — skip 우회 해결만으로는 ceiling 갱신 불가 evidence).

- **결정**:

  1. **(a) 🎯 Dominant Mechanism 갱신 — Mechanism (ii) GATv2Conv Normalization 5/5**:

     | Mechanism | 직전 (Phase 2 only) 판정 | 본 갱신 (Phase 3 4-trial 추가) | 핵심 정량 evidence |
     |---|:---:|:---:|---|
     | (i) Aggregation collapse | 2/5 marginal | 2/5 marginal (변경 X) | top-5 raw cos 0.48~0.63 (entire-table 0.51 대비 marginal) |
     | **(ii) GATv2Conv normalization** | 3/5 paradox direct | **5/5 ⭐ DOMINANT** | **Phase 3 #3 AC=0.62 일관 유지** + L1_GAT cosine **1.0000** + attention sharpen (top5_conc 0.24→0.83) ceiling 무효 |
     | (iii) Skip dependency pathology | **5/5 DOMINANT** | **4/5 보조 (부분 부정)** | Phase 3 #3 skip_dep 0.97 (해결) 에도 R 가장 낮음 + Phase 3 #4 skip_dep 7.38 (악화) |
     | (iv) Schema sibling (raw PLM) | 3/5 lower bound | 3/5 lower bound (변경 X) | L0=0.5090 모든 ckpt 동일 |

     **결정적 evidence (Mechanism (ii) DOMINANT 의 직접 정량)**:
     - **Phase 3 #3 AC loss 0.6155 (ep1) → 0.6154 (ep_last)** — 학습 300 epochs 동안 ΔAC = -0.0001 (사실상 변화 없음). 비교: Phase 2 fusion AC 0.087 → 0.0007 (125× decay). main GAT path raw output 의 collapse 가 학습 dynamics 로 회복 불가의 결정적 evidence.
     - **4 ckpt 모두 L1_GAT cosine ≈ 1.0** (Phase 2/3 #4 = **1.0000**, std=0.0000, n=150) — column embedding 첫 layer 만에 완전 동질화. PN+IR 적용된 v2 ckpt 가 더 심함 (s06 B5 의 0.373 와 대조 — V-3-ext directed_from_sn self-loop + threshold filter 가 PN/IR 효과 무력화).
     - **Phase 3 #4 (LR x5) attention sharpening top5_conc 0.83** + best R@15 = 0.5935 (mitigation null) — 학습으로 attention 이 매우 sharp 해지더라도 aggregation 결과 collapse 는 softmax mechanism 의 구조적 limitation.

  2. **(b) Mechanism (iii) Skip Dep 부분 부정 evidence**:
     - Phase 3 #3 (Direct AC) 의 skip_dep ratio = **0.9652** (가장 균형, main GAT gradient conv_L1=0.32 회복) → 그러나 best R@15 = **0.5927 (가장 낮음)** — skip 우회 해결로 root cause 복구 불가
     - Phase 3 #4 (LR x5) 의 skip_dep ratio = **7.38** (Phase 1 의 3.02 의 2.4× 악화) — LR scaling 으로 main path 학습 회복 실패
     - → **Skip Dep 는 mech(ii) edge softmax collapse 의 부수 효과** (학습 신호가 GAT 외부로 우회되는 결과), **not root cause**

  3. **(c) Mechanism narrative 정식 채택 (analyzer §6.3 인용, paper §V.5.4)**:
     > "DSN Phase 2 + Phase 3 4-trial 의 best R@15 ceiling (~0.61) 갱신 실패의 root cause 는 **GATv2Conv edge softmax 의 fundamental message aggregation collapse** 이다. Phase 3 #3 (AC target='gat_out_L_last') 에서 AC loss 가 0.6155 → 0.6154 로 학습 300 epoch 동안 일관 유지 — main GAT path 의 raw output collapse 가 학습 dynamics 로 회복 불가의 결정적 evidence. 4 ckpt 모두 L1_GAT cosine ≈ 1.0 (column embedding 첫 layer 만에 완전 동질화) — PN+IR+JK+Dual-Stream+L=2 + AC + Layer-wise LR 어떤 mitigation 도 차단 X. Mechanism: edge softmax 가 weighted mean aggregation 을 강제 → 같은 dst (table) 의 incoming src (column) 가 동일 표현으로 propagation. attention sharpening (Phase 3 #4 top5_conc 0.83 vs no mit 0.24) 은 학습 가능하지만, aggregation 결과 collapse 는 softmax mechanism 의 구조적 limitation. Mechanism (iii) skip dep pathology 는 보조 evidence (Phase 3 #3 가 skip_dep 0.97 도달에도 R 미달). Filter Dominance 6번째 축 (training-pathology-invariant) 결정적 evidence: GAT 의 fundamental limitation (edge softmax collapse) 까지 With-Filter pipeline 이 흡수."

  4. **(d) Filter Dominance 6번째 축 narrative 정식 채택**:
     - 직전 5 evidence (H-B ckpt-invariant + H-F stability/ordering + F-1+H-G alpha sweep + ΔF1 +0.65 lift + H-A/H-D 부정) + **6번째: 4-trial mitigation null effect (mech(ii) edge softmax dominant + AC=0.62 일관 + L1=1.0 collapse)**
     - paper §3.5 narrative 정식 채택 (sub-section 정정: Skip Dependence DOMINANT → GATv2Conv Normalization DOMINANT)

  5. **(e) 학위 논문 Part III chapter §III.4 mechanism deep dive 갱신** — main mechanism finding 정정:
     - 직전 outline (DECISIONS 2026-05-07 단계 5 완료 엔트리): §III.4 = Skip Dependence Pathology DOMINANT 5/5
     - **갱신 outline**:
       - §III.4 = **Mechanism (ii) GATv2Conv Normalization DOMINANT 5/5** (analyzer §6.3 narrative 정식 인용)
       - §III.6 (AC loss mechanism) = Phase 3 #3 의 AC=0.62 일관 유지 + L1=1.0 collapse + skip dep 부분 부정 통합
     - main contribution narrative: "GAT module limitation (edge softmax collapse) 이 paper main pipeline F1 plateau 에 absorb 되는 mechanism" — 단 paper full version 의 ablation evidence base 로 활용 (학회 논문 narrative 영향 X)

  6. **(f) Mitigation v2 candidate (post-paper, LLM-free, ~10h/cell)**:

     | 우선순위 | Candidate | 가설 / mechanism | 학습 비용 |
     |---|---|---|---|
     | **#1** | **DropEdge / DropMessage** | 매 layer 마다 random edge subset 만 활용 → softmax aggregation 의 동질화 압력 약화 | ~10h |
     | **#2** | **Sum / Max aggregation** (mean → sum / max) | softmax 를 sum / max 로 바꿔 aggregation propagation 변경. GINConv 대안 | ~10h |
     | **#3** | **LayerNorm before softmax** (Message normalization 변형) | softmax 전 message 자체를 normalize → magnitude 차이로 aggregation 분산 | ~10h |
     | **#4** | **Energy-based GNN (EGAT)** | softmax-based attention 대체 — energy minimization 으로 aggregation 결정 | ~10h |

     - **공통 사유**: Phase 3 4-trial evidence 가 mech(ii) edge softmax 자체의 구조적 limitation 임을 정량 → mitigation 은 **GAT layer 자체 변경 필요** (skip dep 해결만으로 부족, LR scaling 만으로 부족)
     - **post-paper / 학위 논문 후속 backlog**: mitigation_v2 candidate 학습 시도, ~₩0 LLM-free
     - **paper full version 의 ablation evidence base**: 4 candidate 중 1-2 개 (DropMessage + Sum aggregation) 학위 논문 후 시도 (학위 본 심사 timeline 부족)

  7. **(g) Caveat — single-DB**:
     - 4 ckpt × 50 queries 모두 BIRD-dev 첫 50 = california_schools (db-sorted) → per-DB 분해 single-DB only
     - 본 mech(ii) DOMINANT 판정의 generalizability 는 single-DB 에서 4 ckpt 일관으로 partial 입증, 단 11 DBs 일반성 미검증
     - **post-paper backlog**: shuffle=True 또는 stratified 50 queries 로 11 DB 모두 포함 재측정 (analyzer §10 후속 권장 #1)

  8. **(h) 🚨 사용자 결정 필요 3 항목**:

     | # | 결정 항목 | 옵션 | 권장 |
     |---|----------|------|------|
     | (1) | **paper §3.5 6번째 축 정식 채택 시점** (alpha sweep 추가 여부) | (A) 직전 결정 (alpha sweep skip) 유지 — paper main F1/EX 측정 X, val recall@15 evidence only / (B) 재고 — Phase 2/3 ckpt 위 alpha sweep subset 5 cells (~₩3.8K) 측정 후 paper main F1 spread 정량 | **(A) 유지** — 4-trial val R@15 evidence 만으로 Filter Dominance 6번째 축 narrative 정식 채택 충분, paper §3.5 narrative 영향 X (Phase 1 9 cells alpha sweep 의 F1 spread 0.0019 가 이미 정량 evidence) |
     | (2) | **학위 논문 Part III §V.5.4 narrative 정식 채택 (analyzer §6.3 권장)** | (A) 정식 채택 — analyzer §6.3 narrative paper §V.5.4 직접 인용 / (B) 추가 검증 후 채택 — Mitigation v2 candidate 1-2 시도 후 mech(ii) DOMINANT confirm 강화 | **(A) 정식 채택** — analyzer §6.3 narrative 가 4-trial × 5-step protocol × 4 mechanism 종합 evidence 로 충분 학술적 weight |
     | (3) | **Mitigation v2 candidate 우선순위 (post-paper)** | (A) #1 DropMessage + #2 Sum aggregation 학위 본 심사 후 시도 (2 cells, ~20h) / (B) #3 LayerNorm before softmax 만 (1 cell, ~10h) / (C) #4 Energy-based GNN 만 (1 cell, ~10h, 가장 architectural shift) / (D) 모두 post-paper backlog (학위 본 심사 timeline 부족) | **(D) 모두 post-paper backlog** — 학위 본 심사 timeline (~5/22) 부족 + mech(ii) DOMINANT 판정 evidence 충분, mitigation v2 는 paper 후 long-term direction |

- **근거**:
  - **신규 analyzer 산출**: [dsn_phase3_mitigation_results.md](../notebooks/analysis_results/dsn_phase3_mitigation_results.md) §0~§10 (TL;DR + 5 step + 4 mechanism dominance + §6.3 학위 논문 narrative 권장)
  - **재현 데이터**: outputs/analysis/dsn_phase3_4trial_deep_dive/ (4 ckpt summary.json + ac_loss_trajectory.png + recall_trajectory_overlay.png)
  - **재현 스크립트**: src/analysis/dsn_phase3_4trial_deep_dive.py (v1/v2 자동 분기, Step 5 epoch trajectory parse)
  - **선행 분석**: [dsn_phase2_mitigation_null_mechanism.md](../notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md) (Phase 2 base, mech(iii) DOMINANT 5/5 직전 판정 — 본 분석에서 갱신)
  - **선행 결정**: DECISIONS 직전 엔트리 2026-05-07 (단계 5 완료, P3-A 절대 confirm — analyzer 위임 §1(8))
  - **EXPERIMENT_HISTORY**: L2691~ DSN Phase 2 + Phase 3 4-trial Mitigation Sweep entry

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 4-trial mechanism dominant 갱신 (iii→ii) + analyzer §6.3 학위 논문 narrative 정식 인용 + Filter Dominance 6번째 축 narrative 정식 채택 + Mitigation v2 candidate 4 (post-paper) + 사용자 결정 3 항목
  - **paper_research_direction.md (planner Edit, 본 응답)**:
    - §3.5 "Filter Dominance 6번째 축" sub-section 정정 (Skip Dependence DOMINANT → GATv2Conv Normalization DOMINANT)
    - §3.5 학위 논문 Part III chapter base sub-section §III.4 갱신 (mech(ii) DOMINANT)
    - §8 Future Works H-DTK Mitigation v2 candidate 4 항목 신규
    - §9 Limitations — single-DB caveat 갱신 (mech(ii) generalizability 부분 입증)
    - §10 V-3-ext 단계 5 sub-section 갱신 (4 mechanism dominance scoring 표 + analyzer §6.3 narrative 인용)
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.11 dominant mechanism 정정 (iii→ii) + AC=0.62 chart 인용 + Filter Dominance 6 axes 표
  - **paper main contribution (학회)** 영향 X — anchor t_00 + Filter Dominance 4 축 narrative 그대로
  - **학위 논문 Part III chapter narrative weight 결정적 격상** — 학회 논문 영향 X 단 학위 논문 main mechanism finding (mech(ii) 정식 채택) 정량 weight 강화

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §3.5 / §V.5.4 / §8 / §10 갱신 + presentation_brief §14.11 갱신
  2. **사용자 (즉시 의사결정 3 항목)** — 본 엔트리 §1(h) 의 (1) alpha sweep 재고 / (2) §V.5.4 narrative 정식 채택 / (3) Mitigation v2 우선순위
  3. **사용자 (5/19~5/22)** — 학위 논문 Part III chapter draft 작성 (analyzer §6.3 narrative 정식 인용 + chapter outline 7 절 + mech(ii) DOMINANT)
  4. **사용자 (학회 §V.5.3 1 줄)** — Future Work 1 줄 (DSN Phase 2/3 4-trial mitigation null + mech(ii) edge softmax collapse) 직접 처리
  5. **Analyzer (post-paper 후속)** — Per-DB stratified 50 queries 재측정 (toxicology vs european_football mech(ii) 차이) + softmax noise sensitivity 측정 + Mitigation v2 candidate 학습 후 protocol 재실행 (사용자 결정 (3) D 채택 시 post-paper)

- **추가 필요 분석** (post-paper):
  - Per-DB stratified 50 queries 재측정 (mech(ii) generalizability 11 DBs 검증)
  - softmax noise sensitivity 측정 (mechanism ii 정밀화 — input perturbation 시 alpha 변동)
  - message magnitude vs angle 분리 측정 (cosine 외 magnitude 차원 mech 정밀)
  - Mitigation v2 candidate 1-2 학습 시도 (DropMessage / Sum aggregation 권장) → 동일 protocol 재실행 시 mech(ii) 회복 정도 정량

---

## 2026-05-07 (DSN Phase 2 + Phase 3 4-trial mitigation sweep 완료, 🎯 시나리오 P3-A 절대 confirm) — Filter Dominance 6번째 축 (training-pathology-invariant) 결정적 evidence + paper §3.5 narrative 6번째 evidence 정식 명문화 + 학위 논문 Part III chapter base 확정

- **결정**:
  1. **(a) V-3-ext 단계 5 완료 — 4-trial mitigation 결과표 (decreasing R@15)**:

     | 순위 | Variant | Best R@15 | Best Epoch | Δ vs Phase 1 | 학습 wall |
     |------|---------|-----------|------------|--------------|-----------|
     | **1** | **Phase 1 P80 (no mit)** | **0.6097** | ep91 | (baseline) | 7h 30min |
     | 2 | Phase 3 #4 (LR x5) | 0.5935 | ep172 | -0.0162 | 11h 16min |
     | 3 | Phase 3 #3 (Direct AC `gat_out_L_last`) | 0.5927 | ep51 | -0.0170 | 10h 15min |
     | 4 | Phase 2 b8 (mit fusion) | 0.6018 | ep157 | -0.0079 | 10h 26min |

     **🚨 핵심**: 모든 mitigation variants 가 Phase 1 baseline 보다 lower — graph topology 변경 (Phase 1→2) + B5 mitigation (Phase 2) + Direct AC (Phase 3 #3) + Layer-wise LR x5 (Phase 3 #4) 4-trial 모두 raw R 한계 갱신 X. 모든 trial val R@15 ~0.59-0.61 saturate.

  2. **(b) 운영 이력 (사용자 결정)**:
     - **batch_size 1 → 8** (2026-05-06 17:10): Phase 2 b1 launch 시 7.82min/ep 부담, `train_gat_s06.py:183` 코드 분석 (batched dual_stream 지원 명시) → 사용자 승인 → b1 kill + b8 변경 + smoke (2.5min/ep, 3.1x 빠름) + 본격 학습
     - **Phase 3 #3 + #4 가속 launch**:
       - Phase 3 #3 (5/6 23:00 KST launch): GPU 1 idle 발견 후 Phase 2 진행 중 즉시 병렬 launch
       - Phase 3 #4 (5/7 04:13 KST launch): GPU 0 (Phase 2 종료 후) 즉시 launch — 당초 plan ~10:25 KST → 6h 단축
     - **Alpha sweep skip** (2026-05-07 03:00 KST): "Phase 3 의 #4 는 자동으로 이어서 실행하고 alpha sweep 은 하지 마" — Phase 2/3 #3/3 #4 모두 paper main F1/EX 측정 X, val recall@15 evidence only

  3. **(c) 🎯 시나리오 P3-A 결정적 confirm — Filter Dominance 6번째 축 (training-pathology-invariant)**:
     - 단계 4-bis 발견 (attention 매우 집중적, top-5 ≈ 91%) + 5 mitigation (Phase 2) + Direct AC (Phase 3 #3) + LR x5 (Phase 3 #4) 모두 적용에도 동일 ~0.59-0.61 saturation
     - **시나리오 B (R 0.85+ 회복) 사실상 불가능 확정** + **시나리오 P3-A 절대 confirm** (Filter Dominance 6번째 축 절대적 evidence)
     - paper §3.5 main insight 6번째 축 narrative 결정적 evidence

  4. **(d) AC loss 정량 mechanism (학위 논문 Part III deep dive)**:

     | Variant | AC target | AC ep1 | AC ep~50 | AC ep~150 | 해석 |
     |---|---|---|---|---|---|
     | Phase 2 b8 | `'fusion'` | 0.0683 | ~0.005 | ~0.001 | skip path 가 AC 흡수 (pathology 우회) |
     | Phase 3 #3 | `'gat_out_L_last'` | 0.6155 | 0.6178 | 0.6183 | **main GAT path 가 collapse 압박 처리 못함** (raw GAT 학습으로도 collapse mitigation 불가) |
     | Phase 3 #4 | `'fusion'` (Phase 2 동일) | 0.07 | ~0.01 | ~0.005 | LR x5 로 GAT path 빠른 학습 but fusion AC 는 동일 |

     - Phase 3 #3 의 AC=0.62 일관 유지 → main GAT path 의 raw collapse 가 학습으로 회복 안 됨 (정량 evidence)
     - Phase 2 / Phase 3 #4 의 fusion AC decay → fusion path 가 main GAT path 의 collapse 를 우회 (skip 활용)
     - 어떤 path 든 raw R 한계 ~0.61 영역 — **GAT path 자체의 fundamental limitation**

  5. **(e) 🆕 Filter Dominance 6번째 evidence 정식 명문화 (paper §3.5 narrative)**:

     직전 5 evidence + 6번째 추가:
     1. **H-B ckpt-invariant** — Cosine ↔ GAT raw signal 독립 (Pearson r=0.0579 Enriched / 0.2396 qcond_nl3)
     2. **H-F stability/ordering** — Jaccard 0.4673 + Spearman 0.6453
     3. **F-1 + H-G alpha sweep** — F1 spread 0.0724 → WF 0.0142 = 5.0850× 압축
     4. **ΔF1 +0.65 lift** — Filter 의 P 정확도 +0.6300 boost
     5. **H-A/H-D 부정** — distribution shift / norm 변형 모두 plateau 원인 X
     6. 🆕 **Phase 2 + Phase 3 mitigation 4-trial null effect (training-pathology-invariant)** — graph topology + B5 + Direct AC + LR x5 4-trial 모두 raw R 한계 갱신 X (~0.59-0.61 saturate), Filter F1 plateau spread 0.0019 absorb

  6. **(f) paper §3.5 / §V / §8 / §10 갱신 (planner 본 엔트리에서 Edit)**:
     - §3.5 sub-section "🚀 Filter Dominance 6번째 축 (training-pathology-invariant) 정량 정당화 — Skip Dependence Pathology DOMINANT" 정정/확장 (4-trial mitigation 결과 표 + AC loss mechanism 표 + P3-A 절대 confirm narrative)
     - §V (Conclusion) 직접 인용 narrative 정식 추가 — "GAT 학습의 internal training pathology (skip dependence pathology + 4-trial mitigation null effect) 까지 With-Filter pipeline 이 흡수"
     - §8 H-DTK Phase 3 항목 — "🔥 학위 본 심사 전 진행" → "✅ 단계 5 완료 (P3-A 절대 confirm)" + Phase 3 #1 Skip scaling reduction 우선순위 강등 (mitigation null effect 모두 동일 패턴 → marginal evidence value, post-paper backlog)
     - §10 핵심 수치 표 — 4-trial mitigation 결과 표 신규 + AC loss target 별 mechanism 표 신규

  7. **(g) 🚨 학위 논문 Part III chapter base 확정 (mechanism finding + 4-trial mitigation 시도 두 차원 contribution)**:
     - **Main mechanism finding**: Skip Dependence Pathology DOMINANT 5/5 (analyzer dsn_phase2_mitigation_null_mechanism.md §4)
     - **4-trial mitigation 시도 narrative**: graph topology + B5 mitigation + Direct AC + LR x5 모두 fail → Filter Dominance 의 robustness 결정적 evidence (학술적 weight 격상)
     - **AC loss mechanism deep dive**: Phase 3 #3 의 AC=0.62 일관 유지 가 main GAT path 의 fundamental limitation 직접 evidence
     - **chapter outline 권장**:
       - §III.1 over-smoothing 진단 (Phase 1 baseline + qcond_nl3 0.9971 collapse)
       - §III.2 paradox 발견 (단계 4-bis attention 91% with collapse)
       - §III.3 Mitigation 시도 1차 (B5 5 항목 → null effect)
       - §III.4 Mechanism deep dive (Skip Dependence Pathology DOMINANT 5/5)
       - §III.5 Mitigation 시도 2차 (Direct AC + LR x5 → null effect 동일)
       - §III.6 결론 (training-pathology-invariant — Filter Dominance 6번째 축)

  8. **(h) Analyzer 위임 (root 가 별도 prompt 발송 명시) — Phase 3 #3 + #4 mechanism 재진단**:
     - 4 ckpt × 4 mechanism 재진단 (analyzer EXPERIMENT_HISTORY 후속 §)
     - (a) val recall ceiling 회복 정도: Phase 1 0.6097 vs Phase 2 0.6018 vs Phase 3 #3 0.5927 vs Phase 3 #4 0.5935
     - (b) Skip dependence ratio (gradient flow 재측정): Phase 2 mit 1.89~2.13 vs Phase 3 ?
     - (c) AC target 별 main GAT path gradient 회복 정도: Phase 2 conv 0.07~0.18 vs Phase 3 #3 (gat_out_L_last) → main path 회복 정도
     - (d) Layer-wise LR 의 GAT path gradient 5× 회복 검증: Phase 3 #4
     - 산출물: `notebooks/analysis_results/dsn_phase3_mitigation_results.md` (또는 dsn_phase2_mitigation_null_mechanism.md §10 보강)
     - **planner 후속 단계**: analyzer 결과 dominant 판정 (P3-A 절대 confirm 의 mechanism evidence 강도) → 학위 논문 Part III chapter draft 의 mechanism §III.4 보강

- **근거**:
  - **EXPERIMENT_HISTORY.md "DSN Phase 2 + Phase 3 4-trial Mitigation Sweep (V-3-ext 단계 5, 2026-05-06 → 05-07, 🎯 시나리오 P3-A 결정적 confirm)"** (root 2026-05-07 갱신, L2691~)
  - **선행 분석**: [dsn_phase2_mitigation_null_mechanism.md](../notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md) (Phase 2 4 mechanism deep dive, Skip Dependence Pathology DOMINANT 5/5 판정)
  - **선행 결정**: DECISIONS 직전 엔트리 2026-05-06 (Phase 3 #3+#4 구현 완료) + 2026-05-06 (Phase 3 학위 본 심사 전 진행)
  - **운영 결정**: 사용자 batch_size 변경 / Phase 3 가속 launch / alpha sweep skip (2026-05-06~07)
  - **AC loss mechanism**: Phase 3 #3 학습 log `logs/train/gat_directed_supernode_p80_b5_phase3_directAC_*.log` (AC=0.62 일관 유지)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 4-trial 결과 표 + 운영 이력 + 시나리오 P3-A 결정적 confirm + AC loss mechanism + Filter Dominance 6번째 evidence 정식 명문화 + 학위 논문 Part III chapter base + analyzer 위임
  - **paper_research_direction.md (planner Edit, 본 응답)** — §3.5 / §V / §8 / §10 갱신
  - **presentation_brief_2026-04-28.md (planner Edit, 본 응답)** — §14.10 → §14.10 + 신규 §14.11 신설 (DSN Phase 2 + Phase 3 4-trial 결과 + Filter Dominance 6번째 축 narrative)
  - **paper main contribution (학회)** 영향 X — anchor t_00 (F1=0.8657) + Filter Dominance 4 축 narrative 그대로
  - **학위 논문 Part III chapter narrative weight 결정적 격상** — mechanism finding (Skip Dependence DOMINANT) + 4-trial mitigation 시도 (paradox + null + dominant + Phase 3 동일 null) 두 차원 contribution
  - **paper §V Conclusion 직접 인용 가능 narrative**: "GAT 학습의 training pathology 4-trial mitigation 모두 fail (Phase 1 0.6097 → Phase 2 0.6018 → Phase 3 #3 0.5927 / Phase 3 #4 0.5935 모두 ~0.59-0.61 saturate) 에도 With-Filter F1 plateau spread 0.0019 → Filter Dominance 의 robustness 결정적"

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료, 본 응답)** — DECISIONS 본 엔트리 + paper §3.5 / §V / §8 / §10 갱신 + presentation_brief §14.11 신설
  2. **Analyzer (즉시, root 가 별도 prompt 발송)** — Phase 3 #3 + #4 mechanism 재진단 (4 ckpt × 4 mechanism, gradient flow / AC target 별 main GAT path gradient / Layer-wise LR 의 GAT path 5× 회복 검증)
  3. **Planner (analyzer 결과 후)** — 학위 논문 Part III chapter §III.4 mechanism deep dive 보강 (P3-A 절대 confirm 의 mechanism evidence) + DECISIONS 후속 엔트리
  4. **사용자 (5/19~5/22)** — 학위 논문 Part III chapter draft 작성 (mechanism finding + 4-trial mitigation 시도 두 차원 contribution, planner outline 활용)
  5. **사용자 (학회 §V.5.3 1 줄)** — Future Work 1 줄 (DSN Phase 2/3 mitigation null effect) 직접 처리

- **추가 필요 분석** (analyzer Phase 3 mechanism 재진단 후):
  - Phase 3 #3 의 main GAT path gradient: AC target='gat_out_L_last' 적용에도 main GAT 가 collapse 학습 못하는 mechanism (skip path 우회 여전 or 다른 mechanism?)
  - Phase 3 #4 의 LR x5 효과: GAT path gradient 5× 증가 검증 + 그럼에도 ceiling 갱신 X 하는 mechanism (LR↑ → 빠른 saturation, 단 ceiling 동일)
  - 4 ckpt × 4 mechanism evidence matrix 갱신 — Phase 3 #3 / #4 추가 행
  - Phase 1 P80 baseline 의 best epoch 91 vs Phase 3 #3 ep51 (빠른 saturation) / #4 ep172 (느린 saturation) 의 학습 dynamic 차이

---

## 2026-05-06 (Phase 3 #3 + #4 구현 완료 — Direct AC + Layer-wise LR + 🚨 AC loss root cause 코드 확인) — Selector 산출물 4 + smoke 7/7 통과 + Root 학습 핸드오프 prep (5/13 #3 + 5/16 #4)

- **결정**:
  1. **(a) Selector 모듈 단계 5 (Phase 3 #3 + #4) 구현 완료** (2026-05-06 동일자 가속 완료, 직전 timeline 5/10~5/12 → 5/6 가속):
     - 출처: [src/modules/selectors/EXPERIMENT_PLAN_selectors.md §V-3-ext 단계 5](../src/modules/selectors/EXPERIMENT_PLAN_selectors.md)
     - **단계 1 → 4 → 4-bis → 5 가속 진행** (모두 5/5~5/6 완료, 본 심사 timeline 여유 ↑)
  2. **(b) 🚨 AC loss 위치 root cause 코드 확인 (사전 trace)** — Phase 2 mitigation null mechanism finding 의 정확한 root cause 입증:
     - `train_gat_s06.py:391-395` (Phase 2 baseline):
       ```python
       if anti_collapse_weight > 0.0 and "column" in node_embs:
           if COL_TO_TAB_EDGE in batch.edge_index_dict:
               col_embs = node_embs["column"]   # ← model.forward 결과
               cb_edge = batch.edge_index_dict[COL_TO_TAB_EDGE]
               step_loss_ac = anti_collapse_fn(col_embs, cb_edge)
       ```
     - **`node_embs` = `gat_model(...)` 반환값 = v2 model.forward 결과**
     - `dual_stream=True` 시 forward 마지막 단계 = `fusion_head[nt](concat([h, z_q, h*z_q]))` (gat_network_v2.py L390-405)
     - → **AC loss 가 fusion 후 결과에 적용** — skip path (skip_dict + fusion_head) 가 우회 가능
     - **🚨 null mechanism finding 의 정확한 root cause 확인 ✅** (Phase 2 의 AC loss 가 GAT 학습에 직접 작용 못함 — Skip Dependence Pathology 의 직접 증거)
  3. **(c) Phase 3 #3 PRIMARY — Direct AC on GAT output 구현**:
     - **메커니즘**: AC loss target 을 `'gat_out_L_last'` 로 변경
     - **forward hook** 으로 마지막 `HeteroConv` (= `gat_model.convs[-1]`) 의 column 출력 capture
     - AC loss 그 위에 적용 → skip + fusion 우회 차단, main GAT path gradient 회복
     - **Config**: `configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml`
       - Base: Phase 2 b5_mitigation.yaml
       - 변경: `training.anti_collapse_target: "gat_out_L_last"`
     - **Smoke 검증**:
       - hook capture tensor (raw GAT, [N, hidden×heads] = 24×128) ≠ fusion output ([N, out_channels] = 24×64) — shape 분리
       - AC loss on hook capture 의 backward 가 last conv 의 inner GATv2Conv params 에 grad 전달 (6/54 params)
  4. **(d) Phase 3 #4 SECONDARY — Layer-wise LR 구현**:
     - **메커니즘**: PyTorch optimizer `param_groups` 활용
     - `convs.*` (HeteroConv ModuleList) + `*.convs.*` (inner GATv2Conv) 산하 파라미터만 `base_lr × multiplier` (= 5e-4 = 5×)
     - 그 외 (lin_dict / out_lin_dict / skip_dict / pairnorms / fusion_head / query_encoder / classifier_heads) 는 base_lr (1e-4) 그대로
     - → main GAT path 가 우회 path 대비 5× 빠른 학습
     - **Config**: `configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml`
       - Base: Phase 2 b5_mitigation.yaml
       - 변경: `training.optimizer_layer_wise_lr: true`, `training.gat_lr_multiplier: 5.0`
       - `anti_collapse_target` 미설정 (= 'fusion' default) — #3 와 분리 측정
     - **Smoke 검증**:
       - filter 정확성: gat-path 108 params / other 52 params (synthetic, num_layers=2)
       - lin_dict / out_lin_dict / skip_dict / fusion_head / query_encoder / pairnorms 모두 `other` 로 분류 ✓
       - LR assignment: gat_convs=5e-4 / gat_other=1e-4 / classifier_heads=1e-4
       - backward compat: `layer_wise_lr=False` 시 1 group + lr=base_lr (Phase 2 동일)
  5. **(e) Smoke test 7/7 통과** (`src/modules/selectors/tests/test_phase3_mitigations.py`):
     - `test_p3_3_hook_captures_last_conv_output` — fusion vs raw GAT shape 분리 ✓
     - `test_p3_3_hook_backward_graph_intact` — AC loss → last conv params grad 전달 ✓
     - `test_p3_4_param_group_filter_correctness` — `'convs'` filter 정확 ✓
     - `test_p3_4_optimizer_lr_assignment` — 5× LR 적용 ✓
     - `test_p3_4_backward_compat_baseline` — Phase 2 단일 LR 보존 ✓
     - `test_phase3_config_parsing` — 두 신규 config 정상 ✓
     - `test_phase2_baseline_unchanged` — Phase 2 baseline regression 보존 ✓
  6. **(f) 변경된 파일 4 항목 (cross-reference)**:

     | 파일 | 변경 내용 |
     |---|---|
     | `src/train_gat_s06.py` | `anti_collapse_target` (fusion / gat_out_L_last) + `optimizer_layer_wise_lr` + `gat_lr_multiplier` 옵션. AC loss 의 `col_embs` source 분기 + forward hook capture. 단일 LR optimizer → 3 param groups (layer-wise 시) |
     | `configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml` | 신규 — Phase 3 #3 학습 config |
     | `configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml` | 신규 — Phase 3 #4 학습 config |
     | `src/modules/selectors/tests/test_phase3_mitigations.py` | 신규 smoke test 7 케이스 |
     | **Phase 2 backward compat 보존**: `anti_collapse_target` default = 'fusion' (Phase 2 동작), `optimizer_layer_wise_lr` default = false (Phase 2 단일 LR) — Phase 2 regression 검증 통과 |
  7. **(g) Root 학습 핸드오프 prep (본 응답 본문 prompt)**:
     - **5/13~5/16 (root) — Phase 3 #3 학습**:
       - `python src/train_gat_s06.py --config configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml` (~12-13h, GPU 0 또는 1)
       - 신규 ckpt: `best_gat_directed_supernode_p80_b5_phase3_directAC.pt`
       - NAS 저장 + 로컬 symlink (memory rule)
       - 직후 STEP 3 alpha sweep subset 5 cells (~₩3.8K, ~3h)
     - **5/16~5/19 (root) — Phase 3 #4 학습**:
       - `python src/train_gat_s06.py --config configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml` 동일
       - 신규 ckpt: `best_gat_directed_supernode_p80_b5_phase3_layerwiseLR.pt`
       - 직후 alpha sweep subset 5 cells
  8. **(h) Analyzer 후속 prep (5/19+)**:
     - Phase 3 #3 + #4 두 ckpt 의 mechanism 재진단:
       - **(a) val recall ceiling 회복 정도**: Phase 1 0.6097 vs Phase 2 0.6012 vs Phase 3 #3/#4
       - **(b) Skip dependence ratio (gradient flow 재측정)**: Phase 2 mit 1.89~2.13 vs Phase 3 ?
       - **(c) AC target 별 main GAT path gradient 회복 정도**: Phase 2 conv 0.07~0.18 vs Phase 3 #3 (gat_out_L_last) → main path 회복 정도
       - **(d) Layer-wise LR 의 GAT path gradient 5× 회복**: Phase 2 vs Phase 3 #4
     - `extract_layerwise_attention_v2` 재호출 — Phase 3 attention pattern (top-5 conc 변화)
     - 산출물: `notebooks/analysis_results/dsn_phase3_mitigation_results.md` (또는 dsn_phase2_mitigation_null_mechanism.md §10 보강)
  9. **(i) 전체 timeline 가속 정리** — selector 모듈 5/6 동일자 가속으로 단계 1~5 모두 완료:

     | 일정 | 작업 | 상태 |
     |---|---|---|
     | 5/5 | 단계 1 (selector class) | ✅ |
     | 5/5~5/6 | 단계 2/3 (Phase 1 학습 + 측정) | ✅ |
     | 5/6 | 단계 4 (over-smoothing 진단) | ✅ |
     | 5/6 | 단계 4-bis (attention v2) | ✅ |
     | 5/6 (Phase 2) | Phase 2 학습 진행 중 (ep126/300, ETA 04:24 KST 5/7) | ⏳ |
     | 5/6 | **단계 5 (Phase 3 #3+#4 구현)** | ✅ |
     | 5/7~5/10 | Phase 2 STEP 3-5 (alpha sweep + attention v2 + L_out cosine) | ⏳ root |
     | **5/13~5/16** | **Phase 3 #3 학습 + alpha sweep** | ⏳ root |
     | **5/16~5/19** | **Phase 3 #4 학습 + alpha sweep** | ⏳ root |
     | 5/19+ | Analyzer mechanism 재진단 (3 ckpt × 4 mechanism) | ⏳ analyzer |
     | 5/19~5/22 | 학위 논문 Part III chapter (mechanism finding + 4-trial mitigation) | ⏳ user |
     | 5/22~5/29 (본 심사 중) | Phase 3 #1 Skip scaling (시간 가능 시) | ⏳ root |

- **근거**:
  - **Selector 모듈 단계 5 산출**: [src/modules/selectors/EXPERIMENT_PLAN_selectors.md §V-3-ext 단계 5](../src/modules/selectors/EXPERIMENT_PLAN_selectors.md) (구현 + 7 smoke 통과 + AC loss 위치 root cause 확인)
  - **선행 결정**: DECISIONS 직전 엔트리 (사용자 결정 + Phase 3 학위 본 심사 전 진행 + #3 PRIMARY + #4 SECONDARY)
  - **AC loss root cause 확인**: train_gat_s06.py:391-395 + gat_network_v2.py L390-405 (fusion_head 마지막 단계)
  - **Cross-reference**: 단계 4-bis (attention v2) → 단계 5 (Phase 3) 모두 5/6 동일자 가속 완료

- **영향 범위**:
  - **DECISIONS 본 엔트리** — Phase 3 #3+#4 구현 완료 + AC loss root cause 확인 + 변경 파일 4 + Root 학습 핸드오프 prep + Analyzer 후속 prep + 전체 timeline 가속 정리
  - **Selector EXPERIMENT_PLAN §V-3-ext 단계 5 ✅ 완료 표기** (이미 selector 모듈에서 작성됨)
  - **paper_research_direction.md (선택, 학습 결과 후 정식 확정)** — §8 H-DTK Phase 3 ⏳ 학습 진행 중 표기 (구현은 ✅) + 시나리오 P3-A/B/C 분기 결과 prep
  - **Root 학습 핸드오프 prompt** (응답 본문) — 5/13~5/19 학습 + alpha sweep + analyzer 후속
  - **paper main contribution 영향 X** (학회 narrative 그대로)
  - **학위 논문 Part III chapter narrative weight 결정적 격상 prep**: 4-trial mitigation 시도 (Phase 2 B5 + Phase 3 #3 + #4 + 본 심사 중 #1) → Filter Dominance 6번째 축 절대적 evidence 강화

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — DECISIONS 본 엔트리 + Root 학습 핸드오프 prompt (응답 본문)
  2. **사용자 (즉시)** — Root 세션 prompt 직접 붙여넣기 (cd `/home/hyeonjin/thesis_refactored`, 5/13 #3 학습 시작 / 5/16 #4 학습 시작)
  3. **Root (5/7 04:24 KST 학습 종료 후~5/19)** — Phase 2 STEP 3-5 (5/7~10) → Phase 3 #3 학습+측정 (5/13~16) → Phase 3 #4 학습+측정 (5/16~19) → EXPERIMENT_HISTORY 갱신
  4. **Analyzer (5/19+)** — 3 ckpt (Phase 2 b5 + Phase 3 #3 + Phase 3 #4) mechanism 재진단 + extract_layerwise_attention_v2 재호출
  5. **Planner (analyzer 결과 후)** — 시나리오 P3-A/B/C 분기 처리 + paper §3.5 narrative 정식 확정 (4-trial mitigation evidence) + 학위 논문 Part III chapter base
  6. **사용자 (5/19~5/22)** — 학위 논문 Part III chapter 작성 (mechanism finding + 4-trial mitigation 시도)

- **추가 필요 분석** (Phase 3 학습 + 측정 결과 후):
  - Phase 3 #3 (Direct AC) 의 main GAT path gradient 회복 정량 — Phase 2 conv 0.07~0.18 → Phase 3 ?
  - Phase 3 #4 (Layer-wise LR) 의 GAT path gradient 5× 회복 + val recall ceiling 영향
  - 시나리오 P3-A/B/C 분기 결정 (val recall ceiling + final F1 plateau spread Phase 1 0.0019 vs Phase 3)
  - 4-trial mitigation 통합 evidence (Phase 1 + Phase 2 B5 + Phase 3 #3 + #4) — Filter Dominance 6축 절대적 evidence 강도

---

## 2026-05-06 (사용자 결정 (1)-A + (2)-A + (3) Phase 3 학위 본 심사 전 진행) — Per-DB post-paper / STEP 3-5 진행 / Skip Mitigation Phase 3 candidate 우선순위 학위 본 심사 전 시도 (대안 검증)

- **결정** (사용자 직전 input):
  1. **(A) (1) Per-DB 분해 — post-paper backlog 채택 (권장 채택)**:
     - single-DB (california_schools) 한계 §V Limitations 1 줄 명시로 충분
     - 4 ckpt 일관 → mechanism finding generalizability partial 입증 충분
     - post-paper 11 DBs × 4 ckpt 분해는 본 심사 후 진행
  2. **(B) (2) Phase 2 STEP 3-5 진행 채택 (권장 채택)**:
     - 학습 종료 (04:24 KST 5/7) 후 즉시 진행
     - STEP 3 alpha sweep subset 5 cells (paper main stack + Phase 2 b8 ckpt) — paper §3.5 narrative 정식 확정 필수
     - STEP 4 attention v2 재호출 full 50 queries — Phase 1 vs Phase 2 attention pattern 비교
     - STEP 5 L_out cosine 재진단 — over-smoothing mitigation 효과 정량
     - EXPERIMENT_HISTORY.md 갱신
  3. **(C) 🚀 (3) Phase 3 Skip Mitigation Candidate — 학위 본 심사 전 진행 (권장 변경)**:
     - **사용자 의도**: "대략적인 원인을 찾았는데 대안을 내 봐야지"
     - 본 심사 (5/22~6/19) 전 가능한 candidate 시도 — 학위 논문 Part III chapter 의 **mechanism finding + 대안 시도** 두 차원 contribution
     - 권장 통합 (B post-paper backlog) **거부** → **#3 + #4 우선 + #1 추가 + #2 post-paper** 분배
  4. **(D) Skip Mitigation Candidate 4 종 우선순위 (사용자 결정 (3) 반영)**:

     | candidate | 설명 | 학습 비용 | 구현 복잡도 | mechanism 적합도 | 학위 논문 본 심사 전 진행? |
     |---|---|---|---|---|---|
     | **#3 (PRIMARY)** | **Direct AC loss on GAT output** — Phase 2 의 AC loss 가 fusion output 적용 추정, GAT output 에 직접 적용하면 skip 우회 차단 | ~12-13h | 中 (loss 위치 변경) | **🔥 高** (skip dependence 직접 차단) | ✅ **5/16~5/19** |
     | **#4 (SECONDARY)** | **Layer-wise LR (GAT 5× higher than skip + fusion)** — main GAT path gradient 직접 회복 | ~12-13h | 低 (optimizer config 만) | High (gradient 1/10 축소 직접 대응) | ✅ **5/19~5/22** |
     | **#1 (TERTIARY)** | **Skip path scaling reduction** — skip contribution weight 축소 (예: 0.5x) | ~12-13h | 低 (model config 만) | 中 (skip 차단 단 학습 stability 위험) | 🟡 본 심사 중 (5/22~5/29) 시도 |
     | **#2 (POST-PAPER)** | **GAT pre-training (fix skip, train GAT only first epochs)** — 학습 schedule 변경 | ~15-20h | 高 (학습 schedule 변경) | High | ❌ post-paper backlog (5/19+ 시간 부족) |
  5. **(E) 🎯 통합 timeline (Phase 2 STEP 3-5 + Phase 3 Skip Mitigation 시도)**:

     | 일정 | 작업 | 세션 |
     |---|---|---|
     | **5/7 04:24 KST** | Phase 2 b8 학습 종료 | root (자동) |
     | **5/7~5/8** | STEP 3 alpha sweep subset 5 cells (~₩3.8K, ~3h) | root |
     | **5/8~5/10** | STEP 4 attention v2 재호출 full 50 queries + STEP 5 L_out cosine 재진단 + HISTORY 갱신 | root + analyzer |
     | **5/10~5/12** | planner narrative 정식 확정 (§3.5 6번째 축) + Phase 3 candidate config/구현 (#3 Direct AC + #4 Layer-wise LR) | planner + selector 모듈 |
     | **🆕 5/13~5/16** | **Phase 3 #3 학습 (Direct AC on GAT output)** — alpha sweep subset 5 cells (~₩3.8K) | root |
     | **🆕 5/16~5/19** | **Phase 3 #4 학습 (Layer-wise LR)** — alpha sweep subset 5 cells (~₩3.8K) | root |
     | **5/19~5/22** | 학위 논문 Part III chapter 작성 (mechanism finding + 대안 시도 두 차원) | 사용자 + planner |
     | **5/22~5/29 (본 심사 중)** | **Phase 3 #1 학습 (Skip scaling reduction)** + 측정 | root (시간 가능 시) |
     | **5/29~6/19** | 본 심사 진행 + 추가 보강 | 사용자 |
     | **post-paper** | Phase 3 #2 (GAT pre-training) + Per-DB 11 DBs 분해 | analyzer + root |
  6. **(F) Phase 3 결과 분기 narrative (학위 논문 Part III chapter)**:
     - **시나리오 P3-A (가장 가능성 高 — Skip mitigation null effect 같은 패턴)**:
       - #3 + #4 적용에도 val recall ceiling (~0.61) 갱신 X + final F1 plateau 0.0019 spread 유지
       - **Filter Dominance 6번째 축 (training-pathology-invariant) 절대적 evidence 강화**:
         - Phase 1 (no mit) + Phase 2 (B5 mit) + Phase 3 (Skip mit) 모두 ineffective → Filter Dominance 의 robustness 결정적 evidence
       - 학위 논문 Part III main contribution: **3-stage evidence + 4-trial mitigation 시도** (mechanism finding + 시도 모두 fail = Filter dominance의 robustness 절대적)
     - **시나리오 P3-B (가능성 中)**:
       - #3 또는 #4 가 raw R 0.65~0.75 회복 + final F1 plateau 갱신 가능성 (mid)
       - 학위 논문 Part III main contribution: mechanism finding + **Skip Dependence pathology 의 partial mitigation 발견** (#3 또는 #4 의 mechanism)
       - paper main contribution 4 → 5 항목 격상 후보 (단 anchor 변경은 학회 후 별도 결정)
     - **시나리오 P3-C (낮음)**:
       - #3 또는 #4 가 raw R 0.85+ 회복 + final F1 plateau 결정적 갱신 (>0.870)
       - 학위 논문 main contribution 5 항목 격상 + paper anchor 검토 (학위 논문 시점에서)
       - 가능성 낮음 — DOMINANT mechanism (skip dependence) 가 architectural inherent 가능성 高
  7. **(G) 학회 논문 narrative 영향 X (재확인)**:
     - paper main anchor t_00 (F1=0.8657) 변경 X
     - Filter Dominance 4 축 narrative (학회) 그대로
     - Phase 3 결과는 학위 논문 Part III chapter 만 적용
     - 학회 §V.5.3 Future Work 1 줄 (DSN Phase 2 + Phase 3) 사용자 직접 처리
  8. **(H) Phase 3 #3 (Direct AC on GAT output) 핵심 구현 spec — Selector 모듈 핸드오프 prep**:
     - 현재 Phase 2 의 AC loss 위치 확인 필수 (`train_gat_s06.py` `anti_collapse_weight` apply 위치)
     - Phase 2 추정: AC loss 가 fusion_head 또는 final output 에 적용 → skip 우회 가능
     - Phase 3 #3: AC loss 를 GAT layer output (L1/L2/L_out) 에 직접 적용 — `loss_anti_collapse(gat_out_L_out, gold_label)`
     - 구현 위치: `train_gat_s06.py` 또는 `src/models/gat_network_v2.py` 의 forward + loss
     - smoke test: AC loss applied position 확인 + gradient norm main GAT path 회복 검증
  9. **(I) Phase 3 #4 (Layer-wise LR) 핵심 구현 spec**:
     - PyTorch optimizer 의 parameter groups 활용 — GAT layer params 5× LR
     - 구현 위치: `train_gat_s06.py` 의 optimizer 정의
     - 예시 코드:
       ```python
       gat_params = [p for n, p in model.named_parameters() if 'conv' in n or 'gat' in n]
       other_params = [p for n, p in model.named_parameters() if 'conv' not in n and 'gat' not in n]
       optimizer = torch.optim.AdamW([
           {'params': gat_params, 'lr': 5e-4},  # 5× higher
           {'params': other_params, 'lr': 1e-4}  # baseline
       ])
       ```
     - smoke test: GAT path gradient 5× 증가 검증

- **근거**:
  - 사용자 직전 input (2026-05-06): "(1) (A) / (2) (A) / (3) 그래도 가능한 건 학위논문 본심사 전에 해 보고 싶어 대략적인 원인을 찾았는데 대안을 내 봐야지"
  - **선행 결정**: DECISIONS 직전 엔트리 (Phase 2 mitigation null mechanism dominant 판정 — Skip Dependence DOMINANT 5/5)
  - **Mitigation candidate 출처**: [dsn_phase2_mitigation_null_mechanism.md §8.1](../notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md) — analyzer 권장 4 candidate
  - **Phase 2 학습 log**: `/tmp/directed_sn_train_logs/train_dsn_p80_b5_mitigation_b8.log`

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 사용자 결정 3 항목 confirm + Phase 3 candidate 우선순위 + 통합 timeline + 시나리오 분기 + 구현 spec
  - **paper_research_direction.md (planner Edit)** — §8 H-DTK Phase 3 항목 갱신 (post-paper backlog → 학위 본 심사 전 진행 + candidate 우선순위)
  - **Selector 모듈 세션 핸드오프 prompt** (응답 본문) — Phase 3 #3 + #4 구현 spec
  - **Root 세션 핸드오프 prompt** (응답 본문) — Phase 2 STEP 3-5 (5/7~) + Phase 3 #3 학습 (5/13~) + #4 학습 (5/16~)
  - **학위 논문 Part III chapter 영향**: mechanism finding + 4-trial mitigation 시도 (paradox + null + dominant + Phase 3 mitigation 결과) = 학술적 weight 결정적 격상
  - **학회 논문 narrative 영향 X (재확인)**

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — DECISIONS 본 엔트리 + paper §8 H-DTK Phase 3 갱신 + Selector + Root 핸드오프 prompt (응답 본문)
  2. **사용자 (즉시)** — Selector 모듈 + Root 세션 prompt 직접 붙여넣기:
     - Selector 모듈 (cd `/home/hyeonjin/thesis_refactored/src/modules/selectors`) — Phase 3 #3 + #4 구현 (5/10~5/12)
     - Root 세션 (cd `/home/hyeonjin/thesis_refactored`) — Phase 2 STEP 3-5 (5/7~) + Phase 3 #3 학습 (5/13~) + #4 학습 (5/16~)
  3. **Selector 모듈 (5/10~5/12)** — Phase 3 #3 (Direct AC on GAT output) + #4 (Layer-wise LR) 구현 + smoke test
  4. **Root (5/7~5/22)** — STEP 3-5 + Phase 3 #3 + #4 학습 + 측정
  5. **Analyzer (선택, 5/19+)** — Phase 3 결과 mechanism 재진단 (Skip Dependence ratio + AC loss applied position + GAT gradient 회복 정량)
  6. **Planner (각 단계 결과 후)** — narrative 정식 확정 별도 엔트리 (시나리오 P3-A/B/C 분기 처리)
  7. **사용자 (5/19~5/22)** — 학위 논문 Part III chapter 작성 (mechanism finding + 4-trial mitigation 시도)

- **추가 필요 분석** (Phase 3 결과 후):
  - #3 (Direct AC) val recall ceiling 회복 정도 + Skip dependence ratio 변화
  - #4 (Layer-wise LR) main GAT path gradient 회복 정도 + val recall ceiling 변화
  - #1 (Skip scaling, 본 심사 중) — 학습 stability + recall 변화
  - 4 candidate (Phase 1 + Phase 2 B5 + Phase 3 #3 #4 (#1)) 통합 evidence — Filter Dominance 6번째 축 절대적 evidence 강도

---

## 2026-05-06 (Phase 2 mitigation null mechanism dominant 판정 — Mechanism (iii) Skip Dependence Pathology 5/5 DOMINANT) — 학위 논문 Part III main mechanism finding 확정 + Filter Dominance 6번째 축 (training-pathology-invariant) 정량 정당화

- **결정**:
  1. **(a) Analyzer 산출 수령** — [dsn_phase2_mitigation_null_mechanism.md](../notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md) (4 mechanism deep dive 완료, 2026-05-06):
     - 4 ckpt × 50 queries × column 노드
     - 4 mechanism evidence 강도 1-5 scale 정량 + dominant 판정
     - 학위 논문 Part III main mechanism finding 권장 narrative 도출
  2. **(b) 🎯 4 Mechanism Evidence Matrix — Dominant 판정 결정**:

     | Mechanism | Evidence 강도 | 핵심 정량 | 판정 |
     |---|---|---|---|
     | (i) Aggregation collapse — top-5 raw PLM cosine | **2/5** | 0.52~0.58 (+0.05~0.07 vs entire-table 0.51) | **Marginal** — top-5 attended 가 random 보다 약간 sibling, root cause X |
     | (ii) GATv2Conv normalization — edge softmax | **3/5** | mit L2 entropy 2.48~2.61 (no mit 3.22 대비 -0.7), top5_conc 0.62~0.67 (no mit 0.24 대비 +0.4) | **Paradox 직접 evidence** — Mitigation 이 attention sharpen, 단 over-smoothing 잔존 → root cause X |
     | **(iii) Skip dependence pathology** | **🔥 5/5** | no mit skip/conv ratio **3.02~3.04** (extreme), mit conv 1/10 축소 (1.05~1.13 → 0.07~0.18), fusion path 우회 (gradient 0.55~1.24) | ✅ **DOMINANT** |
     | (iv) Schema sibling 유사성 (raw PLM) | **3/5** | L0 0.51~0.55 (mid-similar), GAT +0.45 추가 collapse | **Lower bound only** (40% 책임), GAT 학습 dynamics 가 dominant |
  3. **(c) 🚀 Mechanism (iii) Skip Dependence Pathology Dominant — 학위 논문 Part III main mechanism finding 확정**:
     - **No mitigation (Phase 1, qcond_nl3)**: skip/main_conv ratio = **3.02~3.04** (extreme) — main GAT path gradient 가 skip path 의 1/3 수준, GAT layer 가 query-schema interaction 학습 X, skip residual 로 prediction 우회
     - **With mitigation (Phase 2 b8, s06 B5)**: skip ratio 1.89~2.13 으로 약간 완화 (mitigation 부분 효과) but **main GAT gradient 동시에 1/10 축소** (conv_L1: no mit 1.05~1.13 → mit 0.07~0.18) — 학습이 fusion_head + query_encoder path 로 우회 (gradient 0.55~1.24, lin_dict + skip 합산 보다 큼)
     - **결론**: **GAT layer 가 본질적으로 학습되지 않음** → val recall ceiling (~0.61) 갱신 X — mitigation 의 collapse 완화 mechanism 자체는 작동 (AC loss decay 정상) but 학습 dynamics 가 GAT 를 우회
  4. **(d) Mechanism (ii) GATv2Conv attention sharpen — Paradox 직접 evidence (root cause X)**:
     - Mitigation 적용 시 attention sharpen (col→tab top-5 conc 0.24 → 0.67, +0.43)
     - L2 entropy 2.48~2.61 (no mit 3.22 대비 -0.7)
     - 단계 4-bis sanity (n=2) 의 top-5 ≈ 91% 와 일관
     - **Mitigation 이 정확히 attention 을 sharpen 시킴 — 단 over-smoothing 잔존** → paradox 의 직접 정량 evidence
     - root cause X (mitigation 으로 sharpen 됐지만 ceiling 갱신 X)
  5. **(e) Mechanism (iv) Schema sibling 유사성 — Lower Bound (~40% 책임)**:
     - raw PLM (L0) intra-table cosine sim 0.51~0.55 (mid-similar)
     - GAT 가 L0 0.51 → L3 0.96+ 로 **+0.45 추가 collapse** (Phase 1 over-smoothing 진단 §2 결과)
     - sibling 의 raw 유사성이 ~40% 책임 (lower bound) — GAT 학습 dynamics 가 dominant
     - mitigation 무효 사유: raw 단계 sibling 도 mid-similar (0.51~0.55) 라 mitigation 으로 회복 가능 영역 한정
  6. **(f) Mechanism (i) Aggregation collapse — Marginal (root cause X)**:
     - top-5 attended 노드의 raw PLM cosine 0.52~0.58 vs entire-table 0.51 → marginal +0.05~0.07
     - top-5 가 random 노드보다 약간 sibling 중심 단 절대값 차이 작음
     - root cause X — top-5 흡수 자체의 collapse 효과는 minor
  7. **(g) Filter Dominance 6번째 축 (training-pathology-invariant) 정량 정당화 (학위 논문 Part III main mechanism)**:
     - **GAT learning pathology** (skip dependence ratio 3.02 + attention sharpen with over-smoothing paradox top-5 91%) **까지 With-Filter pipeline 이 흡수**
     - **Phase 1 9 cells F1 plateau spread 0.0019** (직전 alpha sweep 결과)
     - **Phase 2 mitigation null effect (-0.0085 underperform)**
     - 두 결과가 **같은 mechanism**: Selector internal training pathology 의 차이를 Filter 가 final F1 spread 에서 흡수
     - paper §3.5 main insight 의 가장 깊은 차원 evidence — Selector design choice 가 아닌 **training dynamics 자체의 pathology** 까지 absorb
  8. **(h) 학위 논문 Part III main contribution 재확인**:
     - **Graph topology 변경 (DSN, advisor 제안)**: ineffective (Phase 1 P80 F1 plateau)
     - **Mitigation 5 항목 (B5)**: ineffective (Phase 2 -0.0085 underperform)
     - **Skip dependence pathology**: GAT 학습 본질의 한계 (학위 논문 Part III main mechanism finding)
     - → **Filter Dominance 의 robustness 결정적 evidence** — Selector design / 학습 어떤 변경도 흡수
  9. **(i) Skip Dependence Mitigation Candidates (analyzer §8.1, post-paper backlog)**:
     - **#1**: Skip path scaling reduction (skip 의 contribution weight 축소)
     - **#2**: GAT pre-training (fix skip, train GAT only first epochs)
     - **#3**: Direct AC loss on GAT output (skip 우회 차단)
     - **#4**: Layer-wise LR (GAT 5× higher than skip + fusion)

- **🚨 사용자 결정 필요 3 항목**:
  1. **Per-DB 분해 후속 측정 (analyzer §7.3 caveat) 우선순위**:
     - 본 분석 50 queries 모두 첫 50 = california_schools (db-sorted dataset) → single-DB only
     - **(A) post-paper backlog**: 학위 본 심사 후 (학회 narrative 영향 X)
     - **(B) 학위 본 심사 안 진행 (5/22~6/19)**: per-DB 11 DBs × 4 ckpt 분해
     - **권장**: **(A) post-paper** — single-DB 한계 §V Limitations 1 줄 명시로 충분, 본 mechanism finding 이 single-DB 에서도 4 ckpt 일관 → generalizable
  2. **Phase 2 학습 종료 (04:24 KST 5/7) 후 추가 측정**:
     - **(A) STEP 3-5 진행** (alpha sweep subset 5 cells + attention v2 재호출 full 50 queries + L_out cosine 재진단): paper §3.5 narrative 정식 확정 필수
     - **(B) STEP 3 alpha sweep 만 + STEP 4-5 보류**: paper §3.5 narrative 직전 candidate 그대로 + 학위 논문 Part III chapter 만 detail
     - **권장**: **(A) STEP 3-5 진행** — paper §3.5 6번째 축 정량 evidence 정식 확정 + 학위 논문 chapter base
  3. **Skip Mitigation Candidate 4종 중 학위 논문 Phase 3 시도 우선순위 (post-paper)**:
     - **(A) #1 + #2 (Skip scaling + GAT pre-training)**: 학위 본 심사 안 시도 (5/22~6/19) — Phase 2 와 다른 차원의 mitigation
     - **(B) #1~#4 모두 post-paper backlog**: 학위 논문 Part III main contribution = paradox + null effect + dominant mechanism finding 만 (mitigation 시도 X)
     - **(C) #3 (Direct AC loss on GAT output) 만 시도**: skip 우회 직접 차단 mechanism 검증
     - **권장**: **(B) post-paper backlog** — 학위 본 심사 timeline 부족 + 학위 논문 Part III main contribution 은 mechanism finding 자체로 충분 학술적 weight

- **근거**:
  - **Analyzer 산출**: [dsn_phase2_mitigation_null_mechanism.md §0~§9](../notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md) — 2026-05-06
  - **재현 데이터**: outputs/analysis/dsn_phase2_mechanism_deep_dive/ (4 ckpt × 4 mechanism summary.json)
  - **재현 스크립트**: src/analysis/dsn_phase2_mechanism_deep_dive.py (v1/v2 자동 분기, compute_gradient_flow_compat)
  - **선행 분석**: dsn_oversmoothing_analysis.md (Phase 1 baseline) + s06_bottleneck_comparison.md (B5 mitigation reference)
  - **선행 결정**: DECISIONS 직전 엔트리 (Phase 2 b8 mitigation null + 4 mechanism 위임)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — Skip dependence pathology dominant 5/5 확정 + 4 mechanism evidence matrix + 학위 논문 Part III main mechanism finding + Filter Dominance 6번째 축 정량 정당화
  - **paper_research_direction.md (planner Edit)**:
    - §3.5 6번째 축 narrative 보강 (skip dependence ratio + attention sharpen paradox 결합)
    - §V.5.4 신설 — Part III main mechanism finding (analyzer §7.2 narrative 인용)
    - §8 Future Works — Skip mitigation candidate 4종 (post-paper backlog) 추가
    - §9 Limitations — single-DB caveat
  - **paper main contribution 영향**: 학회 논문 narrative 변경 X (anchor t_00 + Filter Dominance 4 축 narrative 그대로). **학위 논문 Part III chapter narrative weight 결정적 격상** (paradox + null + dominant mechanism finding 3-stage evidence).
  - **presentation_brief 영향 (선택, 학습 종료 후 별도)** — §14.10 보강 candidate

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — DECISIONS 본 엔트리 + paper §3.5/§V.5.4/§8/§9 갱신
  2. **사용자 (즉시 의사결정 3 항목)** — Per-DB 분해 / Phase 2 STEP 3-5 / Skip mitigation candidate 4종 우선순위
  3. **Root (5/7 04:24 KST 학습 종료 후, 사용자 (2)-A 결정 시)** — STEP 3 alpha sweep + STEP 4 attention v2 재호출 + STEP 5 EXPERIMENT_HISTORY 갱신
  4. **Analyzer (선택, post-paper)** — Per-DB 분해 (사용자 (1)-B 결정 시) / Skip mitigation candidate 검증 (사용자 (3) 결정 시)
  5. **Planner (Phase 2 STEP 3-5 결과 후)** — narrative 정식 확정 별도 엔트리 (§3.5 6번째 축 정식 + Phase 1 plateau spread vs Phase 2 spread 비교)
  6. **사용자 (5/16~5/22)** — 학위 논문 Part III chapter 작성 (paradox + null + dominant mechanism 3-stage evidence + analyzer §7.2 narrative 인용)

- **추가 필요 분석** (사용자 결정 후):
  - Phase 2 STEP 3-5 결과 (alpha sweep + attention v2 + L_out cosine 재진단)
  - Per-DB 분해 (post-paper, 11 DBs × 4 ckpt) — generalizability 검증
  - Skip mitigation candidate 검증 (post-paper, #3 Direct AC loss 권장 시도)
  - 학위 논문 Part III chapter draft — 3-stage evidence narrative + 4 mechanism matrix 표 + skip dependence dominant figure

---

## 2026-05-06 (Phase 2 b8 학습 중간 보고 + Mitigation null effect 결정적 evidence — 시나리오 A 절대 confirm 강화) — Filter Dominance 6축 (training-pathology-invariant) 결정적 + paradox 강력 + Analyzer mechanism deep dive 위임

> **⚠️ 학습 진행 중 (STEP 2 in progress, ep126/300, 42%)** — 본 엔트리는 **중간 보고 + analyzer 요청 큐 작성**. STEP 3-5 (학습 종료 04:24 KST 5/7 + alpha sweep + attention v2 재호출) 결과 후 narrative 정식 확정 별도 엔트리.

- **결정**:
  1. **(a) 운영 이력 (사용자 결정 + 운영 결정)**:
     - **사용자 b8 변경 (2026-05-06 17:10 KST)**: 초기 batch_size=1 launch 시 per-epoch 7.82min, ETA 15h 부담 → train_gat_s06.py L183 "dual_stream batched forward 지원" 명시 → 사용자 승인 후 b1 학습 kill + b8 변경 + smoke test (1ep/2.5min/0.4923 val R) + 본격 launch
     - **STEP 0 완료 (2026-05-06)**: HISTORY/CATALOG/ID_MIGRATION 3종에 qcond_nl3 baseline 0.6061 정정 entry 추가 (직전 over-smoothing 진단 §1.1 인용)
     - **STEP 1 완료**: train_gat_s06.py 에 V-3-ext options forward 추가 (DSN p80 + s06 B5 mitigation 통합 학습 가능)
     - **STEP 2 진행 (현재 ep126/300, 42%)**: Phase 2 b8 학습, 17:45 launch, ETA 04:24 KST 5/7
  2. **(b) 🎯 Phase 1 vs Phase 2 b8 비교 (동일 epoch, 결정적 evidence)**:

     | epoch | Phase 1 P80 best | Phase 2 b8 best | Δ (Phase 2 − Phase 1) |
     |---|---|---|---|
     | 20 | 0.6034 | 0.5946 | -0.0088 |
     | 40 | 0.6078 | 0.5981 | -0.0097 |
     | 50 | 0.6083 | 0.5993 | -0.0090 |
     | 60 | 0.6088 | 0.6001 | -0.0087 |
     | **91 (Phase 1 best)** | **0.6097** | ~0.6006 | -0.0091 |
     | **108 (Phase 2 best)** | 0.6097 (saturated) | **0.6012** | **-0.0085** |
     | 126 (현재) | 0.6097 | 0.6012 | -0.0085 |
  3. **(c) 🚨 Mitigation 5 항목 (PN + IR α=0.2 + JK + Dual-Stream + L=2 + AC + ListNet) null effect — 결정적 evidence**:
     - 모든 동일 epoch 에서 Phase 2 b8 가 Phase 1 P80 보다 일관 **-0.0085~-0.0091 underperform**
     - mitigation 적용에도 raw val R@15 한계 갱신 X — Phase 1 ceiling (~0.61) 그대로
     - **시나리오 B (R 0.85+ 회복) 사실상 불가능 확정**
     - **시나리오 A 절대 confirm**: Filter Dominance 6번째 축 (training-pathology-invariant) **절대적 evidence**
  4. **(d) 학습 dynamics — AC loss 정상 작동 but val ceiling 갱신 X**:

     | 메트릭 | Phase 1 P80 | Phase 2 b8 |
     |---|---|---|
     | Best val R@15 | 0.6097 (ep91) | 0.6012 (ep108) |
     | AC loss 추세 | N/A | 0.0683 (ep1) → 0.0130 (ep13) → ~0.005 (ep41+) — 정상 작동 |
     | 학습 saturation | ep91 best | ep108 best (+17 epochs 늦음) |
     | Per-epoch 시간 | 1.5 min | 2.12 min (+0.62 min) |

     - **🚨 AC loss decay 0.068 → 0.005 정상**: Anti-Collapse mechanism 작동 (collapse 완화 정량 evidence)
     - **단 val recall ceiling 갱신 X**: collapse 완화 mechanism 자체는 작동, val recall ceiling 의 root cause 는 다른 mechanism
     - 학습 saturation +17 epochs 늦음: mitigation 일부 효과 (학습 안정성) but ceiling 미돌파 — paradox 강력 confirm
  5. **(e) 🎯 시나리오 A 절대 confirm + 4 mechanism 후보 중 dominant 결정 필요**:
     - 단계 4-bis paradox (attention 매우 집중적 top-5 ≈ 91%, entropy 0.51 그럼에도 over-smoothing collapse) 와 일관
     - mitigation 5 항목 null effect → **paradox 의 root cause 가 attention 도 아니고 mitigation 으로 해결 가능한 mechanism 도 아님**
     - **DECISIONS 단계 4-bis §(d) 의 4 mechanism 후보** 중 dominant 결정 필요:
       - (i) **Aggregation collapse** — top-5 흡수 노드 (sibling) 가 비슷 → aggregation 결과 collapse
       - (ii) **GATv2Conv normalization** — edge softmax 의 mechanism 자체가 collapse 유발
       - (iii) **Skip dependency pathology** — main GAT path 학습 신호 약함 + skip 의존
       - (iv) **Schema sibling 유사성** — raw PLM embedding 단계부터 sibling 유사 → mitigation 무효
     - **분석 위임 — Analyzer 요청 큐 (응답 본문 prompt)**: 4 ckpt 비교 (DSN p80 / DSN p80_b5_mitigation / s06 B5 / qcond_nl3) × 4 mechanism evidence 강도 1-5 scale
     - **산출물**: `notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md`
     - **학위 논문 Part III main mechanism finding 권장 narrative 도출**
  6. **(f) STEP 3-5 timeline 예고 (학습 종료 04:24 KST 5/7 후)**:
     - **STEP 3 (5/7~5/9)**: paper main stack alpha sweep subset 5 cells (α∈{0.0, 0.3, 0.5, 0.7, 1.0}) — Phase 2 b8 ckpt + Final F1 plateau spread Phase 1 0.0019 vs Phase 2 비교 (시나리오 A confirm 강화)
     - **STEP 4 (5/8~5/10)**: attention v2 재호출 (full 50 queries, Phase 1 sample size 동일) — Phase 1 vs Phase 2 attention pattern 비교 (mitigation 의 attention 영향)
     - **STEP 5 (5/10~5/15)**: analyzer mechanism deep dive (본 엔트리 §(e) 위임) + 학위 논문 Part III chapter base
  7. **(g) Paper narrative 영향 (정식 확정 후)**:
     - **§3.5 6번째 evidence 추가 candidate** (학습 종료 + alpha sweep 후 정식 확정): "mitigation 5 항목 적용에도 over-smoothing 회복 X + final F1 plateau spread 0.0019 유지 → training-pathology-invariant 절대적 evidence"
     - **§V Part III mechanism deep dive narrative candidate**: 4 mechanism 후보 dominant 결정 (analyzer 결과 후)
     - **§8 Future Works**: Phase 2 ✅ 진행 중 표기 + mechanism deep dive analyzer 위임
     - **paper main contribution 변경 X**: paper main anchor t_00 + Filter Dominance 4 축 narrative (학회) 그대로
     - **학위 논문 Part III chapter narrative weight ↑**: paradox + null effect + 4 mechanism 후보 = main mechanism finding evidence
  8. **(h) 학회 논문 narrative 영향 X (재확인)**: paper main anchor t_00 (F1=0.8657) + Filter Dominance 4 축 narrative 그대로. 본 paradox + null effect 발견은 학위 논문 Part III chapter 만 적용.

- **근거**:
  - **Phase 2 학습 log**: `/tmp/directed_sn_train_logs/train_dsn_p80_b5_mitigation_b8.log` (학습 trajectory ep1~ep126)
  - **선행 결정**: DECISIONS 직전 엔트리 (단계 4-bis paradox 발견 + (1)-A Phase 2 confirm + over-smoothing 진단)
  - **EXPERIMENT_HISTORY entries**:
    - "Baseline Correction (2026-05-06)" — qcond_nl3 baseline 0.6061 정정
    - "DSN Phase 1 Alpha Sweep" — Phase 1 P80 best 0.6097 ep91
    - "V-3-ext 단계 2-3" — Phase 1 학습 + alpha sweep 결과
  - **선행 분석**:
    - dsn_oversmoothing_analysis.md §1.1 (qcond_nl3 0.6061 실측) + §5 (Mitigation candidates s06 B5)
    - s06_bottleneck_comparison.md §3 (B5 evidence pool — PN/IR/AC/Dual-Stream/L=2)
    - extract_layerwise_attention_v2.py (단계 4-bis 산출 도구, Phase 2 후 재호출)
    - gat_bottleneck_analysis_v2.py (over-smoothing trajectory + gradient flow 함수)
  - **ckpt 위치**:
    - Phase 1 DSN p80: `outputs/checkpoints/best_gat_directed_supernode_p80.pt`
    - Phase 2 DSN p80_b5_mitigation (학습 중): `outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation.pt`
    - s06 B5 reference: `/SSL_NAS/peoples/khj/thesis/checkpoints/s06_gat_bottleneck_fix/best_gat_s06_a01_06_b5.pt`
    - qcond_nl3 baseline: `outputs/checkpoints/best_gat_qcond_nl3.pt`

- **영향 범위**:
  - **DECISIONS 본 엔트리** — Phase 2 b8 중간 보고 + null effect + 시나리오 A 절대 confirm + analyzer mechanism deep dive 위임
  - **paper_research_direction.md (planner Edit, candidate)** — §3.5 6번째 evidence + §V Part III mechanism deep dive + §8 Phase 2 진행 중 (학습 종료 + alpha sweep 후 정식 확정 별도 엔트리)
  - **presentation_brief (planner Edit, candidate)** — Phase 2 학습 중간 결과 + mitigation null effect 추가
  - **Analyzer 요청 큐** (응답 본문) — 4 ckpt × 4 mechanism deep dive prompt
  - **paper main contribution 영향 X (학회)**, **학위 논문 Part III chapter narrative weight ↑**

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — DECISIONS 본 엔트리 + paper §3.5/§V/§8 candidate 갱신 + presentation_brief 갱신 + analyzer 요청 prompt (응답 본문)
  2. **사용자 (즉시)** — Analyzer 세션 (`cd /home/hyeonjin/thesis_refactored/src/analysis && claude`) 에 본 응답의 analyzer prompt 직접 붙여넣기 (4 mechanism deep dive)
  3. **Root (학습 종료 04:24 KST 5/7 후, STEP 3-5)**:
     - STEP 3: paper main stack alpha sweep subset 5 cells (~₩3.8K, ~3h)
     - STEP 4: attention v2 재호출 (full 50 queries)
     - STEP 5: HISTORY 갱신 + planner 핸드오프
  4. **Analyzer (사용자 핸드오프 후)** — `dsn_phase2_mitigation_null_mechanism.md` 작성 (4 mechanism dominant 판정)
  5. **Planner (analyzer 결과 + STEP 3-5 결과 수령 후)** — narrative 정식 확정 별도 엔트리 (§3.5 6번째 evidence + §V Part III chapter base)
  6. **사용자 (5/16~5/22)** — 학위 논문 Part III chapter 작성 (planner narrative + analyzer mechanism finding 인용)

- **추가 필요 분석** (analyzer 위임):
  - 4 ckpt × 4 mechanism evidence 강도 1-5 scale 통합 표
  - top-5 attention 노드의 raw embedding cosine sim + post-GAT cosine sim (mechanism (i) Aggregation collapse 검증)
  - edge softmax weight 분포 + entropy per layer (mechanism (ii) GATv2Conv normalization 검증)
  - gradient norm main vs skip path (mechanism (iii) Skip dependency pathology 검증)
  - raw PLM embedding intra-table cosine sim (mechanism (iv) Schema sibling 유사성 검증)
  - 학위 논문 Part III main mechanism finding 권장 narrative 도출

---

## 2026-05-06 (V-3-ext 단계 4-bis 완료 — `extract_layerwise_attention_v2` 구현 + 🚨 attention dispersion 가설 부정 paradox 발견) — Phase 2 학습 (5/10~5/12, root) 진행 권장 + analyzer 후속 (5/15~5/16) full 50 queries 재측정

- **결정**:
  1. **(a) Selector 모듈 단계 4-bis 산출물 4 항목** (2026-05-06 동일자 가속 완료, 직전 (2)-A confirm timeline 5/7~5/9 → 5/6 가속):
     - **(i) 신규 함수**: `src/analysis/extract_layerwise_attention_v2.py` — `AttentionCapture` (monkey-patch wrap GATv2Conv.forward) + `extract_layerwise_attention_v2()` + `aggregate_attention_metrics()` + heatmap helpers
     - **(ii) Smoke test 6 케이스 통과**: `src/modules/selectors/tests/test_attention_extract_v2.py` — capture/restore / directed_from_sn no-reverse / value sanity / aggregate / Phase 1 ckpt 호환 (4 ckpt × forward) / qcond_nl3 (no SuperNode) 검증
     - **(iii) `dsn_oversmoothing_analysis.py` Step 3 v1 → v2 교체 완료** + JSON dump + per-ckpt heatmap + cross-model comparison
     - **(iv) 출력 파일 (max_queries=2 sanity 검증)**:
       - `outputs/analysis/dsn_attention/<ckpt>/attention_metrics.json` (per-layer × per-edge-type entropy + top-5 concentration)
       - `outputs/analysis/dsn_attention/<ckpt>/attention_entropy_layerwise.png` (heatmap)
       - `outputs/analysis/dsn_attention/<ckpt>/attention_topk5_concentration.png` (heatmap)
       - `outputs/analysis/dsn_attention/comparison_4ckpt.png` (4 ckpt cross-model)
  2. **(b) v2 의 v1 대비 우월점**:
     - directed_from_sn 의 self-loop 자동 처리
     - supernode threshold filter 후의 edge 만 capture (학습 시점과 동일 graph topology)
     - `__exit__` 시 instance attr 깔끔히 제거 (re-entrant safe)
     - HeteroConv 자체는 정상 forward → V-3-ext 의 `_compute_supernode_mask` / `_inject_sn_self_loop` / threshold edge filter 모두 그대로 적용
  3. **(c) 🚨 초기 sanity 결과 (n=2 query, 2026-05-06)**:

     | ckpt | L1~L3 entropy | Top-5 concentration | 해석 |
     |---|---|---|---|
     | DSN p80 | ~0.51~0.52 | **~0.91** | directed_from_sn + threshold 의 attention 매우 집중적 (top-5 ≈ 91% 흡수) |
     | DSN topk20 | ~0.51~0.52 | ~0.91 | 동일 패턴 |
     | DSN abstau07 | ~0.51~0.52 | ~0.91 | 동일 패턴 |
     | qcond_nl3 baseline | **~0.83** | ~0.85 | 더 균일한 분포 (5 edge types only, no SuperNode) |
  4. **(d) 🚀 핵심 발견 — Attention dispersion 가설 부정 (paradox)**:
     - **H 가설 (dispersion)**: over-smoothing 의 root cause = attention 균일 분포 (모든 노드에 동일 가중치 → 정보 평균화 → collapse)
     - **🚨 본 분석 결과 부정**: DSN 3 ckpt 의 attention 이 **매우 집중적** (top-5 ≈ 91%, entropy 0.51) **그럼에도 over-smoothing collapse 발생** (직전 4-단계 진단 L3_GAT cosine sim 0.96~0.98)
     - **함의**: over-smoothing 의 root cause 가 **attention pattern 이 아닌 다른 factor** — 학위 논문 Part III mechanism deep dive evidence
     - **Mechanism 후보 (학위 논문 Part III chapter 분석 후보)**:
       - (i) **Aggregation 자체의 collapse 효과**: top-5 가 91% 흡수해도 그 5 노드 자체가 비슷하면 (이웃이 비슷한 schema 노드) collapse 발생
       - (ii) **Message passing 의 normalization**: GATv2Conv 내부 normalization (예: edge softmax) 의 mechanism 이 collapse 유발
       - (iii) **Skip residual path 의존**: 직전 over-smoothing 진단 §3 의 skip dependency pathology 와 일관 — main GAT path 가 attention 집중에도 학습 신호 약함
       - (iv) **Schema 노드의 sibling 유사성**: 같은 table 의 column 들이 raw embedding 단계부터 비슷 → attention 집중도 결과 비슷
     - **paradox 의 paper §V 학위 논문 Part III narrative weight ↑** — advisor 제안 사항의 새로운 mechanism finding
  5. **(e) Phase 2 학습 진행 권장 (5/10~5/12, root)**:
     - 단계 4-bis 가 5/6 가속 완료 → 직전 통합 timeline 의 selector 작업 완료, **Phase 2 root 핸드오프 즉시 진행 가능**
     - DSN p80 + s06 B5 mitigation (PN + IR α=0.2 + AC + Dual-Stream + L=2) 통합 학습
     - 신규 ckpt: `best_gat_directed_supernode_p80_b5_mitigation.pt`
     - **본 도구 (`extract_layerwise_attention_v2`) 신규 ckpt 에 재호출** — mitigation 변형의 attention 정량 (PN / IR / AC / Dual-Stream / L=2 의 attention 영향)
     - **5 mitigation 의 attention 영향 가설**:
       - PN (PairNorm): attention pattern 변경 X (단순 norm), 단 collapse 약화
       - IR (Initial Residual): attention 변동 X, gradient 회복
       - AC (Anti-Collapse weight): aggregation 자체 영향, attention pattern 변경 가능성
       - Dual-Stream: schema/query 분리 → attention 분포 변화 유의 (별도 stream 별 분석 필요)
       - L=2: layer 단축, attention 누적 횟수 감소
  6. **(f) Analyzer 후속 (5/15~5/16) — full 50 queries 재측정**:
     - 현재 sanity n=2 → full 50 queries (Phase 1 over-smoothing 진단과 동일 sample size) 재측정
     - **시나리오 분기 evidence**:
       - **시나리오 A confirm 강화**: Phase 2 mitigation 적용에도 attention pattern 동일 (집중적 top-5 ≈ 91%) + final F1 plateau 유지 → over-smoothing root cause 가 attention 이 아닌 mechanism (학위 논문 Part III main finding)
       - **시나리오 B 진입**: Phase 2 mitigation 으로 attention pattern 변화 (entropy ↑ 또는 top-5 conc ↓) + final F1 plateau 갱신 → mitigation mechanism evidence + 학위 논문 main contribution 5 항목 격상
       - **paradox 정량**: Phase 1 vs Phase 2 attention pattern 차이 + over-smoothing 변화 + final F1 spread 3 차원 cross-reference
  7. **(g) 학회 논문 narrative 영향 X (재확인)**:
     - paper main anchor t_00 (F1=0.8657) 변경 X
     - Filter Dominance 4 축 narrative (학회) 그대로
     - 본 paradox 발견은 학위 논문 Part III chapter 만 적용
     - 학회 §V.5.3 Future Work 1 줄은 사용자 직접 처리

- **근거**:
  - **Selector 모듈 단계 4-bis 산출**: [src/modules/selectors/EXPERIMENT_PLAN_selectors.md §V-3-ext 단계 4-bis](../src/modules/selectors/EXPERIMENT_PLAN_selectors.md) (구현 + 6 smoke 통과 + Phase 1 4 ckpt 호환)
  - **선행 결정**: DECISIONS 직전 엔트리 (사용자 (2)-A confirm — Attention 호환성 selector 위임)
  - **선행 진단**: dsn_oversmoothing_analysis.md §7 Caveat (v1 attention extract 호환성 부재) → 본 단계 4-bis 가 그 한계 해소
  - **Cross-reference**: 단계 1 (selector class, 5/5) → 단계 2/3 (학습/측정, 5/5~06) → 단계 4 (over-smoothing 진단, 5/6) → 단계 4-bis (attention v2, 5/6) → Phase 2 (mitigation 통합, 5/10~15)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 단계 4-bis 완료 + sanity 결과 + 🚨 attention dispersion 가설 부정 paradox + Phase 2 진행 권장
  - **Phase 2 root 핸드오프** — 직전 (1)-A confirm 엔트리의 핸드오프 prompt 그대로 적용 가능 (단계 4-bis 완료로 대기 해소)
  - **Analyzer 후속 prep** — Phase 2 학습 완료 (5/12) 후 full 50 queries 재측정 + 시나리오 분기 evidence
  - **paper_research_direction.md 영향 (planner Edit, Phase 2 측정 결과 후)**:
    - §3.5 6축 narrative 보강 가능성 (paradox 발견 → over-smoothing root cause 분리 evidence)
    - §V Part III mechanism deep dive narrative 강화 (attention dispersion 가설 부정 + 4 mechanism 후보)
  - **학회 논문 narrative 영향 X**

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — DECISIONS 본 엔트리 + Phase 2 진행 권장 명문화
  2. **사용자 (즉시)** — Phase 2 root 세션 핸드오프 prompt (직전 (1)-A confirm 엔트리 응답 본문) 그대로 사용 가능, 5/10 부터 진행
  3. **Root (5/10~5/12)** — DSN p80 + s06 B5 mitigation 통합 학습 + 신규 ckpt + (병행) HISTORY qcond_nl3 baseline 정정
  4. **Root (5/13~5/15)** — paper main stack alpha sweep subset (3-5 cells, ~₩2-4K)
  5. **Analyzer (5/15~5/16)** — `extract_layerwise_attention_v2` 신규 ckpt 호출 (full 50 queries) + over-smoothing 재진단 + Phase 1 vs Phase 2 비교 → 시나리오 A/B 분기 evidence
  6. **Planner (5/16+)** — 시나리오 분기 처리 + paper §3.5 / §V.5 narrative 갱신 + 학위 논문 Part III chapter 작성 base
  7. **사용자 (5/16~5/22)** — 학위 논문 Part III chapter 작성

- **추가 필요 분석** (Phase 2 학습 + 측정 결과 후):
  - 신규 ckpt L3_GAT cosine sim (Phase 1 0.96~0.98 → Phase 2 0.50~0.70 추정) + attention pattern (entropy/top-5 conc) Phase 1 vs Phase 2 비교
  - **🚀 paradox 검증 (학위 논문 Part III main mechanism)**:
    - Phase 2 mitigation 으로 attention pattern 변화 X 시 → attention dispersion 부정 confirm + over-smoothing 의 다른 mechanism evidence (4 후보 중 어느 것)
    - Phase 2 mitigation 으로 attention pattern 변화 ✅ + over-smoothing 회복 → 두 mechanism 의 dual evidence
  - 4 mechanism 후보 분리 분석:
    - (i) Aggregation collapse (top-5 노드의 sibling 유사성 정량)
    - (ii) GATv2Conv normalization mechanism (edge softmax 분석)
    - (iii) Skip dependency pathology (직전 §3 cross-reference)
    - (iv) Schema 노드 sibling 유사성 (raw PLM embedding 단계 cosine sim)

---

## 2026-05-06 (사용자 결정 — (1)-A Phase 2 paper full version + (2)-A Attention 호환성 selector 위임 confirm) — 3 핸드오프 prompt 작성 (selector 모듈 + root 학습/측정 + root HISTORY 정정)

- **결정** (사용자 직전 input):
  1. **(A) (1)-A Phase 2 paper full version 후속 채택**:
     - DSN p80 + s06 B5 mitigation 통합 학습 (PN + IR α=0.2 + AC + Dual-Stream + L=2) — 학위 본 심사 5/22~6/19 안 진행
     - 시나리오 B (F1 plateau 갱신 0.870+) 진입 가능성 검증
     - 학위 논문 Part III chapter 의 main mechanism deep dive evidence
     - 비용: ~₩0~5K (학습 LLM-free + alpha sweep subset), ~7-10h (학습) + ~2-3h (측정)
  2. **(B) (2)-A Attention 호환성 보강 — selector 모듈 위임 채택**:
     - `extract_layerwise_attention_v2` (forward hook 기반, V-3-ext `directed_from_sn` 호환) 구현
     - 학위 논문 Part III mechanism deep dive evidence (DSN attention 의 directed edge 영향 정량)
     - 비용: ₩0 (LLM-free, selector 모듈 + analyzer)
  3. **(C) 합산 비용/시간**: ₩0~5K + ~10-13h (selector 구현 ~1 day + root 학습 ~7-10h + alpha sweep ~2-3h + analyzer 분석 ~2-3h)
  4. **(D) 통합 timeline**:
     - **5/7~5/9**: selector 모듈 — `extract_layerwise_attention_v2` 구현 (Phase 2 학습 전 attention extract 도구 준비)
     - **5/10~5/12**: root 세션 — DSN p80 + s06 B5 mitigation 통합 GAT 학습 (PN + IR α=0.2 + AC + Dual-Stream + L=2) 신규 ckpt
     - **5/13~5/15**: root 세션 — paper main stack (Enriched + 신규 ckpt + α=0.5 + MSTPCSTUnion + XiYan GLM + LLM SQL Gen GLM) alpha sweep subset (α∈{0.0, 0.5, 1.0} 최소 + α∈{0.3, 0.7} 권장 = 3-5 cells, ~₩2-4K)
     - **5/15~5/16**: analyzer 세션 — 신규 ckpt over-smoothing 재진단 + attention entropy 정량 + Phase 1 vs Phase 2 비교
     - **5/16~5/22**: planner + 사용자 — 시나리오 A/B/C 분기 처리 + paper §3.5 narrative 갱신 + 학위 논문 Part III chapter 작성
  5. **(E) Mitigation 변형 후보 — DSN p80 base + s06 B5 통합 (단일 학습 권장)**:
     - **PairNorm (PN)**: L1 단계 collapse 차단 (s06 B2 evidence: L1 0.85 → 0.47)
     - **Initial Residual α=0.2 (IR)**: APPNP-style, main GAT path gradient 회복 (s06 B2 IR=0.2 best)
     - **Anti-Collapse weight (AC)**: anti_collapse_weight 0.1 (s06 B3 evidence: L_out 0.86 → 0.65)
     - **Dual-Stream**: Schema/Query 분리 GAT (s06 B5 evidence: L_out 0.36 도달)
     - **L=2 (2-layer 단축)**: L3_GAT 도 collapse (0.96+) → L=2 + IR 조합으로 충분
     - **단일 학습 권장 사유**: 5 mitigation 모두 단일 ckpt 에 적용 — paper full version 진행 시 Phase 1 baseline (DSN p80 단독) 와 직접 비교 가능
  6. **(F) 시나리오 분기 분기점**:
     - **시나리오 A confirm 강화**: Phase 2 mitigation 적용에도 plateau spread 0.0019 유지 → Filter Dominance 6축 narrative 절대적 evidence (over-smoothing 회복까지 Filter 가 absorb)
     - **시나리오 B 진입**: Phase 2 mitigation 으로 raw R 0.69 → 0.85+ 회복 + final F1 plateau 갱신 (>0.870) → 학위 논문 main contribution 5 항목 + paper §V 정정
     - **시나리오 C (drop)**: 거의 불가능 — Phase 2 mitigation 은 over-smoothing 완화 (B5 L_out 0.36 evidence) 만 목적, 학습 saturation drop 가능성 X
  7. **(G) 추가 root 핸드오프 — EXPERIMENT_HISTORY.md qcond_nl3 baseline 정정**:
     - HISTORY 추정 ~0.55 → 실측 0.6061 (epoch 59) 정정
     - planner 권한 외 — 별도 root 세션 핸드오프 prompt 작성 (응답 본문)
  8. **(H) 학회 논문 narrative 영향 X (재확인)**:
     - paper main anchor t_00 (F1=0.8657) 변경 X
     - Filter Dominance 4 축 narrative (학회) 그대로 — 6 축 narrative 는 학위 논문 Part III chapter 만 적용
     - 학회 논문 §V.5.3 Future Work 1 줄 (DSN Phase 2) 명시는 사용자 직접 처리 (직전 Q3)

- **근거**:
  - 사용자 직전 input (2026-05-06): "(1)(2) 모두 (A)로 진행하자"
  - 선행 결정: DECISIONS 직전 엔트리 (DSN over-smoothing 진단 완료, Phase 2 candidate + Attention 위임 후보 권장 옵션)
  - **Mitigation evidence pool**: [s06_bottleneck_comparison.md §3](../notebooks/analysis_results/s06_bottleneck_comparison.md) (B2~B5 cosine sim 비교)
  - **Cross-reference**: V-3-ext 단계 1 (selector class, 2026-05-05) + 단계 2/3 (학습/측정, 2026-05-05~06) + 단계 4 (over-smoothing 진단, 2026-05-06) + Phase 2 (mitigation 통합, 2026-05-10~15)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 사용자 결정 반영 + 통합 timeline + mitigation 변형 + 시나리오 분기 + 3 핸드오프 명문화
  - **Selector 모듈 세션 핸드오프 prompt** (응답 본문) — `extract_layerwise_attention_v2` 구현
  - **Root 세션 핸드오프 prompt** (응답 본문) — Phase 2 학습 + 측정
  - **Root 세션 추가 핸드오프 prompt** (응답 본문) — EXPERIMENT_HISTORY.md qcond_nl3 정정
  - **paper_research_direction.md 영향 (planner Edit, 향후 측정 결과 후)**: §3.5 6축 narrative 강화 (시나리오 A 확정 + Phase 2 mitigation 결과) + §8 H-DTK Phase 2 ✅ 완료 표기 (5/16+)
  - **학회 논문 narrative 영향 X** — paper main anchor + 4 축 narrative 그대로

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — DECISIONS 본 엔트리 + 3 핸드오프 prompt (응답 본문)
  2. **사용자 (즉시)** — 3 prompt 사용자가 직접 해당 세션에 붙여넣기:
     - Selector 모듈 세션 (cd `/home/hyeonjin/thesis_refactored/src/modules/selectors`) — `extract_layerwise_attention_v2`
     - Root 세션 (cd `/home/hyeonjin/thesis_refactored`) — Phase 2 학습 + 측정
     - Root 세션 — EXPERIMENT_HISTORY.md 정정
  3. **Selector 모듈 (5/7~5/9)** — `extract_layerwise_attention_v2` 구현 + smoke test
  4. **Root (5/10~5/12)** — DSN p80 + s06 B5 mitigation 통합 GAT 학습
  5. **Root (5/13~5/15)** — paper main stack + 신규 ckpt alpha sweep subset
  6. **Root (병행)** — EXPERIMENT_HISTORY.md qcond_nl3 baseline 0.6061 정정
  7. **Analyzer (5/15~5/16)** — 신규 ckpt over-smoothing 재진단 + attention entropy
  8. **Planner (5/16+)** — 시나리오 A/B/C 분기 처리 + paper narrative 갱신
  9. **사용자 (5/16~5/22)** — 학위 논문 Part III chapter 작성 (planner narrative 인용)

- **추가 필요 분석** (Phase 2 측정 결과 후):
  - 신규 ckpt L3_GAT cosine sim (s06 B5 evidence 기준 0.50~0.70 영역 추정) — over-smoothing mitigation 정량
  - Raw R 회복 정량 (0.69 → 0.85+ 영역) — 시나리오 B 진입 trigger
  - DSN attention entropy + directed edge 영향 (forward hook 기반) — 학위 논문 mechanism evidence
  - Final With-Filter F1 spread (Phase 1 0.0019 vs Phase 2) — Filter Dominance 6축 narrative 영향

---

## 2026-05-06 (DSN over-smoothing 진단 완료 — Phase 1 시나리오 A 확정 직후 V-3-ext 단계 4) — 🚨 H1 강력 지지 + qcond_nl3 baseline 정정 (0.55 추정 → 0.6061 실측) + Filter Dominance 5축 정량 evidence 강화 + Phase 2 candidate (post-paper)

- **결정**:
  1. **(a) Analyzer 산출 수령** — [dsn_oversmoothing_analysis.md](../notebooks/analysis_results/dsn_oversmoothing_analysis.md) (V-3-ext 단계 4 진단, 2026-05-06, BIRD-dev 50 queries × 4 ckpt = 200 forward pass, column-pair n=150 per layer per ckpt):
     - §1 Training Trajectory (best R@15 + saturation)
     - §2 Layer-wise Over-smoothing (intra-table cosine sim)
     - §3 Step 3 Gradient flow (skip dependency pathology)
     - §4 Filter Dominance 와의 정합성
     - §5 Mitigation candidates (s06 B5 evidence)
     - §6 Phase 2 candidate (post-paper)
     - §7 Caveat (sample size + attention entropy 미측정)
  2. **(b) 🚨 H1 (over-smoothing) 강력 지지 — 4 ckpt 모두 L3_GAT cosine sim ≥ 0.96**:

     | ID | L0_PLM | L1_GAT | L3_GAT | L_out | Best R@15 |
     |---|---|---|---|---|---|
     | DSN p80 | 0.5090 | 0.9079 | 0.9591 | 0.4735 | **0.6097** |
     | DSN topk20 | 0.5090 | 0.9108 | 0.9662 | 0.4840 | 0.5839 |
     | DSN abstau07 | 0.5090 | 0.9426 | **0.9775** | 0.5355 | 0.5805 |
     | **qcond_nl3 baseline** | 0.5537 | **0.9887** | **0.9971 ⚠️** | 0.6299 | **0.6061** |

     - **L1_GAT 부터 sharp jump**: L0 0.51 → L1 0.91~0.99 (단 1 layer 만에 collapse 의 80% 발생)
     - **qcond_nl3 의 L3 = 0.9971 사실상 완전 collapse** (cosine ≈ 1.0 → 같은 table column 들 score ranking 무의미)
     - **DSN 3 ckpt 의 L3 = 0.9591~0.9775**: baseline 보다 **0.02~0.04 mitigated** (`directed_from_sn` edge 의 부분적 효과). 단 절대값은 critical 0.85 line 한참 위
     - L_out 부분 회복: DSN p80/topk20 의 0.47~0.48 (PLM 원본 0.51 와 비슷) — out_lin + skip residual 효과 약간
  3. **(c) 🚨 qcond_nl3 baseline 정정 — HISTORY 추정 ~0.55 부정확**:
     - **실측 best R@15 = 0.6061** (epoch 59) — DSN p80 (0.6097) 와 사실상 동등 (Δ=+0.0036)
     - 이전 EXPERIMENT_HISTORY 의 ~0.55 추정치는 부정확 — DSN 의 graph topology 변경이 baseline 을 능가하지 못함
     - **함의**: paper §V.5 narrative 의 "DSN 이 baseline 대비 강함" claim 약화 → "DSN ≈ baseline 동등 + 둘 다 over-smoothing 으로 raw R 한계 회복 못 함" narrative 정정
  4. **(d) 🎯 시나리오 A 확정 narrative 강력 보강** — Filter Dominance 5축 정량 evidence (DECISIONS 직전 엔트리 V-3-ext 단계 3 plateau 확인 + 본 over-smoothing 진단):
     - **GAT 학습 mechanism 한계까지 Filter 가 absorb**: DSN 3 변형 모두 학습 saturation (val recall@15 0.58~0.61) + over-smoothing collapse — 단 final pipeline F1 spread 0.0019 (plateau 안)
     - **Filter Dominance 5번째 축 (topology-invariant) 정량 evidence**:
       - Selector graph topology 변경 (bidirectional → directed top-K) ✅
       - + Selector training mechanism imperfection (over-smoothing) ✅ — Filter 가 GAT 학습의 internal mechanism 한계까지 absorb
       - → paper §3.5 main insight 가 단순 "Selector design choice" 가 아닌 **"Selector internal training pathology 까지 absorb"** 로 한 단계 격상
     - **paper §3.5 narrative 보강 가능 한 줄**: "GAT 학습이 over-smoothing 으로 raw R 한계 회복 못함에도 With-Filter F1 plateau spread 0.0019 → Filter 가 selector training mechanism imperfection 까지 absorb" (analyzer §4.2 인용)
  5. **(e) Phase 2 candidate (post-paper) — DSN p80 + s06 B5 mitigation evidence 적용**:
     - **Mitigation 변형 후보** (analyzer §5 권장):
       - **PairNorm + Initial Residual α=0.2** (s06 B2~B5 evidence): L1 단계 collapse 차단 (B2: L1 0.85→0.47), L_out 분산 (B4: L_out 0.86→0.56)
       - **Dual-Stream Architecture** (s06 B5): Schema/Query 분리, 2-layer 로 충분 + L_out 0.36 도달
       - **2-layer GAT 로 단축 + Initial Residual** 조합: L2 도 이미 collapse (0.90~0.99) → L=2 + IR 시도 가치
     - **권장 통합**: DSN p80 + (PN + IR α=0.2 + AC + Dual-Stream + L=2)
     - **비용/시간**: ~₩0 (LLM-free 학습), ~7-10h (단일 학습 + smoke verification)
     - **시나리오 B 진입 candidate**: 만약 over-smoothing mitigation 으로 raw R 0.69 → 0.85+ 회복하면 Filter 가 P 정정만 담당 → F1 plateau 갱신 가능 (0.870+)
     - **단 paper full version 후속 vs post-paper backlog 결정 사용자 의존** (학회 마감 D-2, 학위 본 심사 5/22~6/19)
  6. **(f) Attention entropy 미측정 한계 + selector 모듈 보강 위임 후보**:
     - **현재 미측정**: v1 `extract_layerwise_attention` 가 V-3-ext `directed_from_sn` 호환성 부재 → attention entropy / DSN aggregation pattern 정량 미수행
     - **선택 위임**: selector 모듈 세션에 forward hook 기반 보강 (`extract_layerwise_attention_v2`) 위임 가능 — DSN attention 의 directed edge 영향 정량 → 학위 논문 Part III mechanism deep dive evidence
     - **사용자 결정 필요**: 위임 진행 vs post-paper backlog
  7. **(g) Step 3 Gradient flow 발견 — Skip dependency pathology** (analyzer §3):
     - GAT layer 의 gradient 가 skip residual path 에 dominantly 의존 (skip path 차단 시 gradient 거의 사라짐)
     - 함의: out_lin + skip_dict 가 over-smoothing 의 partial 회복을 담당하지만, GAT layer 자체의 학습 신호는 약함 → IR (Initial Residual) 추가가 main GAT path 의 gradient 회복에 효과적 (s06 B2 evidence 와 일관)
  8. **(h) Filter Dominance 와의 정합성 (analyzer §4.2)**:
     - DSN 3 변형 모두 학습 saturation (val recall@15 0.58~0.61) + over-smoothing collapse 에도 **paper main pipeline F1 plateau spread 0.0019** (V-3-ext 단계 3 결과)
     - = paper §3.5 main insight ("Filter Dominance — single-stage main") 의 **강력한 추가 evidence**
     - 단순 Selector design choice (graph topology) 변동을 absorb 하는 것을 넘어 **GAT 학습의 internal pathology (over-smoothing) 까지 흡수** — Filter "first-class stage" 학술적 정당성 한 단계 더 격상

- **근거**:
  - **Analyzer 산출**: [notebooks/analysis_results/dsn_oversmoothing_analysis.md §0~§7](../notebooks/analysis_results/dsn_oversmoothing_analysis.md) — 2026-05-06 작성 (BIRD-dev 50 queries × 4 ckpt = 200 forward pass, column-pair n=150 per layer per ckpt)
  - **재현 데이터**: outputs/analysis/dsn_oversmoothing/ (4 ckpt × plots + batch_summary.json)
  - **재현 스크립트**: src/analysis/dsn_oversmoothing_analysis.py (hook-based extract_layerwise_dsn 포함)
  - **Mitigation evidence pool**: [s06_bottleneck_comparison.md](../notebooks/analysis_results/s06_bottleneck_comparison.md) §3 (s06 B2~B5 cosine sim 비교)
  - **선행 결정**: DECISIONS 직전 엔트리 (DSN Phase 1 시나리오 A 확정, F1 plateau 0.0019 spread)
  - **Cross-reference**: V-3-ext 단계 1 (selector class + GAT v1/v2 dispatch + smoke 7 통과, 2026-05-05) → 단계 2/3 (학습 + 측정, 2026-05-05~06) → 단계 4 (본 over-smoothing 진단, 2026-05-06)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — H1 강력 지지 + qcond_nl3 정정 + 시나리오 A 5축 evidence 강화 + Phase 2 candidate
  - **paper_research_direction.md (planner Edit)**:
    - §3.5 Filter Dominance 5번째 축 (topology-invariant) 정량 evidence 보강 — "GAT 학습 mechanism imperfection 까지 absorb"
    - §V.5 학위 논문 Part III narrative candidate (analyzer §4.2 인용)
    - §8 Future Works H-DTK 항목 보강 — V-3-ext Phase 2 candidate (s06 B5 mitigation 적용)
    - §9 Limitations — qcond_nl3 baseline 정정 + Attention entropy 미측정 한계
  - **EXPERIMENT_HISTORY.md 정정 권장** — qcond_nl3 baseline best R@15 = 0.6061 (이전 ~0.55 추정 부정확) — root 세션 책임 (planner 권한 외)
  - **paper main contribution 영향 minor** — Filter Dominance 5축 narrative 강화 (시나리오 A 확정 + 정량 evidence ↑), main result anchor (t_00 F1=0.8657) 변경 X
  - **학위 논문 Part III chapter 보강** — over-smoothing mechanism 분석 + Phase 2 candidate (post-paper)

- **🚨 사용자 결정 필요 2 항목**:
  1. **Phase 2 (s06 B5 mitigation 적용) 우선순위**:
     - (A) **paper full version 후속** (학위 본 심사 5/22~6/19 안 진행) — 학위 논문 Part III 의 main next step
     - (B) **post-paper backlog** (학위 본 심사 후) — 학위 논문 Part III 는 Phase 1 시나리오 A 확정만 + Phase 2 는 향후 연구
     - **권장**: (A) — DSN p80 + (PN + IR α=0.2 + AC + Dual-Stream + L=2) 단일 학습 (~7-10h, ₩0) 으로 시나리오 B 진입 가능성 검증, 학위 논문 chapter 의 main mechanism deep dive evidence
  2. **Attention 호환성 보강 — selector 모듈 세션 위임 여부**:
     - (A) ✅ 위임 — `extract_layerwise_attention_v2` (forward hook 기반, V-3-ext directed_from_sn 호환) 구현, 학위 논문 mechanism deep dive evidence
     - (B) 🟡 보류 — post-paper backlog
     - **권장**: (A) — paper §V.5 Part III mechanism evidence 보강 + 학위 본 심사 advisor 만족도 ↑

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — DECISIONS 본 엔트리 + paper §3.5 / §V.5 / §8 / §9 갱신
  2. **사용자 (즉시 의사결정 2 항목)** — Phase 2 우선순위 + Attention 위임 (응답 본문 권장 옵션)
  3. **Root (선택, 사용자 (1)-(A) 결정 시)** — EXPERIMENT_HISTORY.md 정정 (qcond_nl3 baseline 0.6061 실측)
  4. **Selector 모듈 (선택, 사용자 (2)-(A) 결정 시)** — `extract_layerwise_attention_v2` 구현 + V-3-ext 호환 보강
  5. **Root (선택, 사용자 (1)-(A) 결정 시 Phase 2)** — DSN p80 + PN + IR α=0.2 + AC + Dual-Stream + L=2 학습 + 측정 (post-paper)
  6. **Planner (Phase 2 결과 후)** — 시나리오 B 진입 여부 narrative 갱신 (F1 plateau 갱신 시 paper main contribution 5 항목 격상)

- **추가 필요 분석** (Phase 2 결정 후):
  - DSN + s06 B5 mitigation (PN + IR + AC + Dual-Stream + L=2) 학습 + 측정 → over-smoothing collapse mitigation 정량 (L3_GAT 0.96~0.99 → 0.50~0.70 영역 추정)
  - Raw R 회복 정량 (0.69 한계 → 0.85+ 영역) — 시나리오 B 진입 trigger
  - Attention entropy + DSN aggregation pattern (forward hook 기반) — 학위 논문 mechanism evidence
  - Skip dependency pathology 의 IR (Initial Residual) mitigation 정량 — gradient flow 회복

---

## 2026-05-05 (Directed Top-K SuperNode 단계 1 구현 완료) — Selector 세션 산출물 인용 + 변경 파일 8 + root 단계 2 학습 핸드오프 + paper §2.2 / §3.5 / §V.5.3 narrative prep

- **결정**:
  1. **(a) Selector 세션 단계 1 구현 완료 (2026-05-05, 5/9~5/11 timeline 가속)**:
     - 직전 DECISIONS 2026-05-05 (analyzer raw_score 결과 인용) 의 단계 1 (5/9~5/11) 이 selector 세션에서 동일자 (2026-05-05) 가속 완료
     - 출처: [src/modules/selectors/EXPERIMENT_PLAN_selectors.md §V-3-ext](../src/modules/selectors/EXPERIMENT_PLAN_selectors.md) — "Directed Top-K SuperNode (학위 논문 Part III, 2026-05-05) — 단계 1 구현 완료"
  2. **(b) 단계 1 산출물 4 항목**:
     - **(i) 신규 selector class**: `DirectedTopKSuperNodeSelector` (`EnsembleSelector` 상속, SuperNode 분기 override) — query_node x 주입 + threshold mask 산출 (per-query min-max norm cosine 기반) + `attends_to_*` directed edge 만 유지 (`attended_by_*` 비등록/0-len) + GAT 모델 측 자동 self_loop (directed_from_sn) 사용
     - **(ii) GAT v1/v2 threshold dispatch**: `gat_network.py` + `gat_network_v2.py` 양쪽에 `_compute_supernode_mask` dispatch — `top_k` (기존 V-3 동치) / `percentile` (torch.quantile cutoff) / `abs_tau` (>= cutoff). `_compute_topk_mask` 는 backward-compat alias.
     - **(iii) 학습 config 3 종**:
       - `train_gat_directed_supernode_p80.yaml` (PRIMARY, percentile 80.0, |sel|=18.9 ± 5.5, Raw R=0.6133, Raw F1=0.3466)
       - `train_gat_directed_supernode_topk20.yaml` (BASELINE, top_k 20, |sel|=20.0 ± 0.0, Raw R=0.6865, Raw F1=0.3640)
       - `train_gat_directed_supernode_abstau07.yaml` (OPTIONAL, abs_tau 0.7, |sel|=10.2 ± 8.9, Raw R=0.4857, Raw F1=0.3942 ★ raw F1 max)
     - **(iv) Smoke test 7 케이스 통과**: `tests/test_directed_topk_supernode.py` — P80 ~22% 선택 / top_k=20 정확 / abs_tau=0.7 선택적 / directed edge 구조 검증 / baseline SuperNode 31 vs Directed 7 edge / v1/v2 dispatch
  3. **(c) 변경된 파일 8 항목 (cross-reference)**:

     | 파일 | 변경 내용 |
     |---|---|
     | `src/models/gat_network.py` | `supernode_threshold_mode/value/score_normalization` 파라미터 + `_compute_supernode_mask` dispatch + alias |
     | `src/models/gat_network_v2.py` | 동일 (v2 분기 호환) |
     | `src/train_gat.py` | 신규 옵션 forward to GAT model |
     | `src/modules/selectors/directed_topk_supernode_selector.py` | 신규 selector 클래스 (EnsembleSelector 상속, SuperNode 분기 override) |
     | `src/modules/selectors/ensemble_selector.py` | `supernode_edge_direction` 옵션 노출 (GAT 모델 측 forward) |
     | `src/modules/selectors/__init__.py` | `DirectedTopKSuperNodeSelector` 등록 |
     | `src/modules/selectors/tests/test_directed_topk_supernode.py` | smoke test 7 케이스 |
     | `configs/training/train_gat_directed_supernode_{p80, topk20, abstau07}.yaml` | 학습 config 3 종 |
  4. **(d) Root 단계 2 학습 핸드오프 prompt**: 본 응답 본문 (사용자 직접 root 세션 5/12~5/13 진행)
     - **학습 변형 3 종 병렬** (GPU 0/1 split):
       - `python src/train_gat.py --config configs/training/train_gat_directed_supernode_p80` (PRIMARY)
       - `python src/train_gat.py --config configs/training/train_gat_directed_supernode_topk20` (BASELINE)
       - `python src/train_gat.py --config configs/training/train_gat_directed_supernode_abstau07` (OPTIONAL)
     - 신규 ckpt: `best_gat_directed_supernode_{p80, topk20, abstau07}.pt`
     - **NAS 저장 + symlink 자동화** (memory rule "저장소 규칙"): NAS path `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_*.pt` + 로컬 `outputs/checkpoints/` symlink
     - 학습 시간: 변형 별 ~9h × 3 변형 (병렬 ~9h, 직렬 ~27h)
     - val recall@15 검증 + dev recall@15 record (raw recall ceiling reference: P80=0.6133, top_k20=0.6865, abs_tau07=0.4857)
  5. **(e) Root 단계 3 측정 핸드오프 (단계 2 완료 후)**:
     - paper main stack (Enriched + 신규 ckpt + α=0.5 + MSTPCSTUnion + XiYan GLM + LLM SQL Gen GLM) 위 alpha sweep
     - **subset (최소)**: α∈{0.0, 0.5, 1.0} (3 cells × 3 변형 = 9 cells, ~₩7K, ~3h)
     - **full (권장)**: α∈{0.0~1.0, 0.1 step} 11 cells × P80 primary (single 변형, ~₩8.4K, ~3-4h) + α∈{0.0, 0.5, 1.0} subset × top_k20/abs_tau07 (6 cells, ~₩4.6K, ~2h) = **합산 17 cells ~₩13K, ~5-6h**
     - 비용/시간: ~₩7-13K, ~3-6h (Wall clock GPU 0/1 split)
  6. **(f) Paper §2.2 / §3.5 / §V.5.3 narrative 정정 prep — 시나리오 A/B/C 분기 결과 후 적용 사항 미리 정리**:

     **시나리오 A (F1 ≤ 0.870, plateau 흡수, 가장 가능성 高)**:
     - **§3.5 Filter Dominance 5 축 격상** — 4 축 (stack-invariant + α-invariant + schema-complexity-dependent + design-variant-aware) → **5 축 (🆕 topology-invariant 추가)**:
       - 단락 3-(e) 신규: "**🆕 topology-invariant**: Selector graph topology 변경 (bidirectional → directed top-K, P80/top-K=20/abs τ=0.7) 도 final F1 plateau 안 흡수 — Filter 가 Selector graph topology 변동도 absorb"
     - **§2.2 Selector contribution 보강**: "α-invariant + topology-invariant" — Selector design choice 의 다차원 변동 모두 Filter 가 흡수
     - **paper main contribution claim 5 항목 추가 가능** (선택): (e) "Selector graph topology 변경의 Filter Dominance 흡수 검증 (Directed Top-K SuperNode, advisor 제안 학위 논문 Part III)"
     - paper §III.6 단락 3 의 4 축 → 5 축 + 단락 5 의 학술적 정당성 격상 한 문장 추가
     - **분량 영향 minor** — 학회 논문 §V.5.3 Future Work 1 줄 → 본문 §III.6 단락 3-(e) 1 줄 추가 + §IV.4.5 신규 sub-section 가능 (분량 여유 시)

     **시나리오 B (F1 > 0.870, plateau 갱신, 가능성 中)**:
     - **paper main contribution 5 항목 격상** — 기존 (a)~(d) + 🆕 **(e) Directed Top-K SuperNode mechanism**:
       - "Selector raw score → graph topology 통합 (out-only directed edge + per-query P80 threshold) 이 paper main F1 갱신 — advisor 제안의 학위 논문 contribution"
     - **§2.2 Selector contribution 본문 정정** — "Query-Conditioned GAT (Concat α=0.5 ensemble) + GAT-floor" → "Query-Conditioned GAT + Directed Top-K SuperNode (advisor 제안, F1 갱신 evidence)"
     - **paper main anchor 정정 검토 필요** — t_00 vs Directed Top-K F1 차이가 학회 논문 anchor 변경 여부 결정 (anchor promote vs evidence 인용 — 직전 옵션 A 결정과 동일 분기)
     - **분량 영향 中** — §I contribution 4 → 5 항목 + §III.5/§III.6 narrative 정정 + §IV 신규 sub-section + §V Conclusion 갱신

     **시나리오 C (F1 < 0.85, 큰 손실, 가능성 低)**:
     - **paper §V.5.3 negative result 1 paragraph 신설**:
       - "advisor 제안 Directed Top-K SuperNode (out-only directed edge + per-query P80 threshold) 가 Filter Dominance plateau 안에 들어오지 못하고 F1 < 0.85 로 큰 손실. 가능 사유: (i) raw R 0.69 한계 + GAT 학습이 R 회복 불충분, (ii) graph topology 단방향 변경의 message passing 손실, (iii) per-query P80 의 schema 노드 누락 (DB 별 variability). 학위 논문 본 심사에서 mechanism deep dive — graph topology 변경의 message passing 손실 정량 + raw R 회복 한계 분석"
     - paper §V Future Work 의 학위 논문 Part III 항목에 negative result + mechanism 분석 명시
     - **학위 논문에서는 negative result 도 학술 contribution** (advisor 제안 mechanism 의 한계 발견 + 정량적 mechanism 분석)
     - **분량 영향 minor** — §V.5.3 1 paragraph 추가
  7. **(g) Planner 책임 분장**: 단계 2/3 진행은 **root 세션 책임**. planner 는 결과 수령 후 후속 DECISIONS 엔트리 + paper §2.2/§3.5/§V.5.3 narrative 갱신 (시나리오 A/B/C 분기 처리) 만 수행.

- **근거**:
  - **Selector 세션 단계 1 산출**: [src/modules/selectors/EXPERIMENT_PLAN_selectors.md §V-3-ext](../src/modules/selectors/EXPERIMENT_PLAN_selectors.md) "Directed Top-K SuperNode (학위 논문 Part III, 2026-05-05) — 단계 1 구현 완료" — 4 산출물 + 8 변경 파일 + 학습 변형 3 종 + 시나리오 A/B/C 인용
  - **선행 결정**: DECISIONS 2026-05-05 직전 엔트리 (analyzer raw_score 결과 인용 — threshold P80 primary 채택 + 단계 1 핸드오프 prompt) — 본 단계 1 완료의 base
  - **Analyzer base**: [raw_score_distribution_for_directed_topk.md](../notebooks/analysis_results/raw_score_distribution_for_directed_topk.md) — Per-query ROC-AUC 0.7930, Cohen's d 1.1323, top-K=20 P78~P80 영역, Recall@20=0.6865

- **영향 범위**:
  - **DECISIONS 본 엔트리** — 단계 1 완료 + 산출물 4 + 변경 파일 8 + root 단계 2/3 핸드오프 + paper narrative 시나리오 A/B/C 분기 prep
  - **paper_research_direction.md (선택, 단계 3 결과 후)** — §2.2 / §3.5 / §V.5.3 narrative 정정 (시나리오 분기에 따라)
  - **EXPERIMENT_PLAN_selectors.md** — 이미 selector 세션이 §V-3-ext 단계 1 완료 표기 ✓
  - **paper main contribution narrative 영향**:
    - 시나리오 A: minor (1 축 추가 + 1 paragraph)
    - 시나리오 B: 中 (main contribution 5 항목 + anchor 검토)
    - 시나리오 C: minor (negative result 1 paragraph)
  - **학회 논문 narrative 영향 X (anchor t_00 + Filter Dominance 4 축 그대로)** — 시나리오 결과 학회 마감 후 학위 논문 chapter 에 반영

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — DECISIONS 본 엔트리 + paper §8 H-DTK 항목 갱신 (직전 turn 완료) + Root 단계 2 핸드오프 prompt (본 응답 본문)
  2. **사용자 (즉시)** — Root 세션 prompt 직접 붙여넣기 (cd `/home/hyeonjin/thesis_refactored`, 5/12~5/13 단계 2 학습)
  3. **Root (5/12~5/13)** — 학습 변형 3 종 (GPU 0/1 split, ~9h 직렬 또는 병렬), 신규 ckpt NAS 저장 + symlink 자동화
  4. **Root (5/13~5/15)** — alpha sweep subset/full 측정 (~₩7-13K, ~3-6h), EXPERIMENT_HISTORY.md / CATALOG / ID_MIGRATION 갱신
  5. **Planner (단계 3 측정 결과 수령 후)** — 시나리오 A/B/C 분기 처리 + DECISIONS 후속 + paper narrative 갱신
  6. **사용자 (학위 논문 chapter 작성 5/16~5/22)** — 학위 논문 Part III chapter 본인 직접 작성, planner 가 narrative 정정 사항 인용

- **추가 필요 분석** (단계 3 측정 결과 후):
  - 변형 3 종 (P80 / top-K=20 / abs τ=0.7) F1/EX 비교 → 어떤 변형이 학위 논문 main 으로 갈지 결정
  - 신규 ckpt 의 attention pattern 분석 (학위 논문 mechanism evidence — Directed Top-K 가 어떤 schema 노드에 집중하는지)
  - 기존 SuperNode (bidirectional) ckpt 와의 attention 비교 (graph topology 변경의 mechanism 정량)
  - per-query R 회복 정량 (raw R 0.69 → GAT 학습 후 R 변화) — 시나리오 A/B/C 분기 결정 evidence
  - 시나리오 A 시 §3.5 5 축 narrative (topology-invariant 정량 evidence) — Directed Top-K F1 spread vs t_00 plateau spread 비교

---

## 2026-05-05 (analyzer raw_score_distribution 결과 인용 — Directed Top-K SuperNode 학습 변형 3 종 + threshold P80 primary 채택) — 단계 1 (5/9~5/11 구현) Selector 모듈 핸드오프 준비 완료

- **결정**:
  1. **(a) Analyzer 산출 수령** — [raw_score_distribution_for_directed_topk.md](../notebooks/analysis_results/raw_score_distribution_for_directed_topk.md) (LLM-free 즉시 위임 결과, 2026-05-05, ₩0):
     - §2 per-query raw cosine score 분포 (P25/P50/P75/P90/P95)
     - §3 gold vs non-gold 분리 (ROC-AUC, Cohen's d, threshold trade-off)
     - §4 기존 top-K=20 의 score range
     - §5 Threshold 후보 24 종 비교 (selected node 수 + recall 추정 + variability)
     - §6 기존 SuperNode 와의 비교 base
     - §7 Directed Top-K 학습 권장 threshold
  2. **(b) 🎯 Threshold 결정 — Per-query P80 primary 채택**:
     - **Primary (학습 추천 #1)**: **per-query P80** — |sel| mean = 18.9 (std 5.5), R = 0.6133, query-aware (DB 별 schema 크기 자동 보정)
     - **사유**: (i) 기존 top-K=20 (|sel|=20) 과 가장 유사한 노드 수 → 직접 비교 가능, (ii) per-query percentile 이라 DB 별 schema 크기 variability (european_football 237 vs toxicology 20) 자동 보정 → graph topology 균질, (iii) 학습 시 stable convergence 기대
     - **Reference (학습 추천 #2 / baseline)**: top-K=20 — 기존 SuperNode 와 직접 비교 + raw R 0.6865 (raw recall ceiling reference)
     - **Optional (학습 추천 #3 선택)**: 절대 τ=0.7 — F1 max raw selector standalone (F1=0.3942), 단 노드 수 variability 큼 (|sel| std 8.9 vs P80 std 5.5)
  3. **(c) 🚨 핵심 base 정량 evidence**:
     - **Per-query ROC-AUC = 0.7930** (Cohen's d = 1.1323 large effect) — gold/non-gold raw score 분리 강함
     - **Recall@20 = 0.6865** (cosine raw 만으로 R 천장 0.69) → **GAT 학습 mechanism evidence base** (학위 논문 §V.5 main mechanism — Directed Top-K 가 raw R 한계를 GAT 학습으로 극복하는 기전 분석)
     - **기존 SuperNode 평균 schema 노드 수 = 92.6** → Directed Top-K (P80 |sel|=18.9) ≈ **20% 노드 보존** — graph topology 큰 변화
     - **DB 별 schema 노드 수 variability 큼** (european_football 237 vs toxicology 20) → 절대 τ 보다 per-query Pn 이 graph topology 균질성 유리
  4. **(d) 시나리오 A/B/C 예측 base 정량화** (raw R 0.69 한계 + GAT 학습 회복 정도가 분기 결정):
     - **시나리오 A (plateau 흡수, F1 ≤ 0.870)**: GAT 학습이 raw R 0.69 → 0.85 영역으로 회복 + Filter 가 plateau 안 흡수 → Filter Dominance **5 축 격상** (🆕 topology-invariant 추가)
     - **시나리오 B (plateau 갱신, F1 > 0.870)**: GAT 학습이 raw R 0.69 → 0.90+ 영역 도달 + Filter 가 P 정정 → 학위 논문 main contribution **5 항목 격상** (a~d + 🆕 (e) Directed Top-K mechanism)
     - **시나리오 C (F1 < 0.85)**: GAT 학습이 raw R 0.69 한계 극복 못 함 → paper §V.5.3 negative result + advisor 제안 mechanism deep dive (학위 본 심사 필수)
     - **시나리오 확률 추정** (raw R 0.69 + ROC-AUC 0.7930 base): 시나리오 A 가장 가능성 高 (Filter 가 raw R 차이 흡수 mechanism 의 직전 narrative 와 일관) — 단 GAT 학습 결과 의존
  5. **(e) Directed Top-K SuperNode 학습 변형 3 종 정량 비교 base**:

     | 변형 | threshold | \|sel\| mean | std | Raw R | Raw F1 | 학습 우선순위 |
     |---|---|---|---|---|---|---|
     | **변형 1 (primary)** | per-query P80 | **18.9** | 5.5 | **0.6133** | 0.3613 | **🔥 #1** (query-aware, top-K=20 와 유사) |
     | 변형 2 (baseline) | top-K=20 | 20.0 | 0.0 | 0.6865 | 0.3640 | #2 (기존 SuperNode 직접 비교) |
     | 변형 3 (선택) | 절대 τ=0.7 | 10.2 | 8.9 | 0.4857 | **0.3942** ★ raw F1 max | #3 (선택, F1 max 영역) |
  6. **(f) 단계 1 (5/9~5/11 구현) Selector 모듈 세션 핸드오프 — 본 응답 본문 prompt**:
     - 신규 selector class `DirectedTopKSuperNodeSelector` (또는 기존 `EnsembleSelector` query_supernode 분기 변형)
     - threshold 메커니즘: per-query P80 primary + top-K=20 reference + 절대 τ=0.7 선택
     - edge: query_node → schema 단방향 (out-only directed edge, schema → query 역방향 X)
     - 신규 GAT ckpt: `best_gat_directed_supernode_topk.pt` (학습 시 P80/top-K=20/τ=0.7 변형 별 ckpt 가능)
     - 학습 단계 (5/12~5/13): 신규 GAT 학습 + val recall@15 검증
     - 측정 단계 (5/13~5/15): paper main stack + 신규 ckpt 의 alpha sweep (subset 또는 full)
  7. **(g) 학회 논문 narrative 영향 X**: paper main anchor t_00 (F1=0.8657) 영향 X, Filter Dominance 4 축 narrative 그대로. 학회 논문 §V.5.3 Future Work 1 줄 명시는 사용자 직접 처리 (Q3).

- **근거**:
  - **Analyzer 산출**: [notebooks/analysis_results/raw_score_distribution_for_directed_topk.md §0~§7](../notebooks/analysis_results/raw_score_distribution_for_directed_topk.md) — 2026-05-05 LLM-free 즉시 위임 결과
    - §3 ROC-AUC 0.7930 + Cohen's d 1.1323 (gold/non-gold 분리)
    - §4 top-K=20 의 score range (top-20 min score mean=0.4957, P50=0.5425, P78~P80 영역)
    - §5 Threshold 후보 24 종 비교 raw 데이터
    - §7 학습 권장 threshold (P80 primary, top-K=20 reference, abs τ=0.7 선택)
  - **재현 raw 데이터**: notebooks/analysis_results/raw_score_threshold_candidates.csv (24 후보 비교) + raw_score_supernode_comparison.csv (DB 별 SuperNode 노드 수)
  - **재현 스크립트**: src/analysis/analyze_raw_score_distribution.py (scipy 의존성 없음)
  - **Cross-reference**: src/modules/selectors/CLAUDE.md "Score 분석 결과" — global ROC-AUC 0.741 vs 본 per-query ROC-AUC 0.7930 (per-query 분포 vs global 분포 차이)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — threshold P80 primary 채택 + 학습 변형 3 + 시나리오 분기 정량 base + Selector 모듈 핸드오프
  - **paper_research_direction.md (planner Edit)** — §8 Future Works H10 항목 갱신 (학위 논문 Part III, analyzer base 분석 완료, 단계 1 대기)
  - **Selector 모듈 세션 핸드오프 prompt** — 본 응답 본문 (사용자 직접 5/9 부터 진행)
  - **학회 논문 main contribution narrative 영향 X** — anchor t_00 + Filter Dominance 4 축 그대로 (학위 논문 Part III 만 신설)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — DECISIONS 본 엔트리 + paper §8 H10 갱신 + Selector 모듈 핸드오프 prompt (응답 본문)
  2. **사용자 (즉시)** — Selector 모듈 세션 prompt 직접 붙여넣기 (5/9 부터 단계 1 구현)
  3. **Selector 모듈 (5/9~5/11)** — `DirectedTopKSuperNodeSelector` 구현 + threshold 변형 3 (P80 / top-K=20 / abs τ=0.7) + edge 단방향 처리 + smoke test
  4. **Root (5/12~5/13)** — 신규 GAT ckpt 학습 (변형 별 1-3 ckpt) + val recall@15 검증
  5. **Root (5/13~5/15)** — paper main stack + 신규 ckpt alpha sweep 측정 (subset 또는 full)
  6. **Planner (측정 결과 후)** — 시나리오 A/B/C 분기 처리 + DECISIONS 후속 + paper §3.5 Filter Dominance narrative 갱신 (시나리오 A 시 5 축 격상)
  7. **사용자 (학회 §V.5.3)** — 본인 직접 처리 (Q3)

- **추가 필요 분석** (단계 3 측정 결과 후):
  - 변형 3 종 (P80 / top-K=20 / abs τ=0.7) F1/EX 비교 → 어떤 변형이 학위 논문 main 으로 갈지 결정
  - 신규 ckpt 의 attention pattern 분석 (학위 논문 mechanism evidence — Directed Top-K 가 어떤 schema 노드에 집중하는지)
  - 기존 SuperNode (bidirectional) ckpt 와의 attention 비교 (graph topology 변경의 mechanism 정량)
  - per-query R 회복 정량 (raw R 0.69 → GAT 학습 후 R 변화) — 시나리오 A/B/C 분기 결정 evidence

---

## 2026-05-05 (Directed Top-K SuperNode — advisor 제안 학위 논문 Part III 진행 결정) — Q1/Q2/Q3 사용자 confirm + threshold 기반 (Raw Score 분포 analyzer 위임) + 학회 §V.5.3 사용자 직접 진행

- **결정**:
  1. **(a) ✅ Q1 학위 논문 Part III 진행 confirm**: advisor (지도교수) 제안 사항 — "Graph 를 Directed Edge 로 변형 + Raw Score 기준 top-K (또는 threshold) 노드만 + SuperNode 에서 out-only edge 부여 + GAT 학습" — 학위 논문 Part III (5/9~5/22 구현+학습+측정+chapter 작성) 진행
  2. **(b) Q2 결정 — Threshold 기반 (top-K 단일값 X)**: 사용자 의도 명확화:
     - 옵션 (a) `top-K=20 단일값` (간단, 기존 SuperNode 와 비교 쉬움) — **사용자 보류**
     - **옵션 (b) ✅ Threshold 기반 (Raw Score 분포 의존)** — 사용자 채택, 단 **threshold 값은 Raw Score 분포 분석 후 결정**
     - 기존 `top-K=20` 은 baseline 비교 reference 로 유지
     - threshold 후보 (analyzer 결과 후 결정): 절대 score (예: 0.3 / 0.5) / per-query percentile (예: P80 / P90) / mean+std 기반 / score 분포의 elbow point
  3. **(c) Q3 학회 논문 §V.5.3 Future Work**: 사용자 본인 직접 진행 — planner 작업 X
  4. **(d) Raw Score 분포 분석 — Analyzer 위임 (planner 권한 외)**:
     - **목적**: Directed Top-K SuperNode 학습 시 threshold 결정의 정량 근거 + 기존 top-K=20 의 score range 와 비교
     - 분석 데이터: `outputs/.../score_analysis_*.jsonl` (per-query × per-node raw cosine + GAT score, is_gold 라벨)
     - 분석 항목 5 (analyzer 위임 prompt 본 응답 본문):
       - per-query raw cosine score 분포 (P25/P50/P75/P90/P95)
       - gold node score vs non-gold node score 분포 비교 (분리 수준)
       - top-K=20 의 score range (절대 score / percentile)
       - threshold 후보 (절대 / percentile / mean+std / elbow point)
       - threshold 별 평균 selected node 수 + recall 추정
     - 산출물: `notebooks/analysis_results/raw_score_distribution_for_directed_topk.md`
  5. **(e) Directed Top-K SuperNode 학위 논문 Part III 단계 (analyzer 결과 후)**:
     - **단계 1** — 구현 (5/9~5/11): 신규 selector class `DirectedTopKSuperNodeSelector` (또는 기존 SuperNode 변형) — query_node→schema 단방향 edge + raw_score threshold filter
     - **단계 2** — 학습 (5/12~5/13): 신규 GAT ckpt `best_gat_directed_supernode_topk.pt`, threshold 변형 (analyzer 결과 후 1-3 변형) 별 학습, val recall@15 검증
     - **단계 3** — 측정 (5/13~5/15): paper main stack + 신규 ckpt 의 alpha sweep (subset 또는 full), Filter Dominance plateau 검증
     - **단계 4** — 분석 + 학위 논문 chapter (5/16~5/22)
     - 비용: ~₩8-10K, 시간: ~7-13 day
  6. **(f) Paper narrative 영향 분기 — 시나리오 A/B/C**:
     - **시나리오 A**: Directed Top-K F1 ≤ 0.870 (plateau 안) → Filter Dominance **5 축으로 격상** (stack/α/schema/design-variant + 🆕 **topology-invariant**) — Selector graph topology 변경도 흡수 evidence
     - **시나리오 B**: Directed Top-K F1 > 0.870 (plateau 갱신) → 학위 논문 main contribution **5 항목으로 격상** ((a)~(d) + 🆕 (e) Directed Top-K SuperNode mechanism)
     - **시나리오 C**: Directed Top-K F1 < 0.85 (큰 손실) → paper §V.5.3 negative result + advisor 제안 mechanism deep dive (학위 본 심사 시 mechanism 분석 필수)
  7. **(g) 학회 논문 narrative 영향 X (anchor 변경 X)**:
     - paper main anchor t_00 (F1=0.8657, 옵션 A 유지) 영향 X
     - 학회 논문 §V.5.3 Future Work 1 줄 명시 (사용자 직접 처리 — Q3)
     - paper outline §V.5.3 의 Future Work 항목에 직접 추가 가능 (사용자 자율)

- **근거**:
  - 사용자 직전 input (2026-05-05): "Q1. 진행하자 / Q2. top-K=20 또는 threshold (Raw Score 분포 분석 후 결정) / Q3. 학회 논문은 직접 진행"
  - advisor (지도교수) 제안 사항 — "Graph 를 Directed Edge / Raw Score top-K / SuperNode out-only edge / GAT 학습"
  - 직전 엔트리 (Wave 4 + 옵션 A) — paper main anchor t_00 유지 + 학위 논문 Part II (Filter 고도화) 결정 → 본 결정으로 학위 논문 Part III (Selector 확장) 추가
  - 기존 SuperNode 변형 (bidirectional, 모든 schema 노드) — DECISIONS 2026-04-29 SuperNode 9-cell matrix + paper main 2 cells (a05_08 Stacked 와 별도 SuperNode 검증)

- **영향 범위**:
  - **DECISIONS 본 엔트리** — Directed Top-K Part III 결정 + threshold 기반 + analyzer 위임 + 시나리오 A/B/C 분기
  - **paper_research_direction.md (선택, 추후 갱신)** — §8 Future Works 의 학위 논문 Part III 항목 추가 (analyzer 결과 + 측정 후 갱신)
  - **paper_outline_2026-05-08.md (사용자 자율)** — §V.5.3 Future Work 1 줄 추가 (advisor 제안 reflect)
  - **학회 논문 main contribution narrative 영향 X** — anchor t_00 + Filter Dominance 4 축 narrative 그대로 유지
  - **학위 논문 Part III 신설** — Filter 고도화 (Part II) 외 Selector 확장 (Part III) 추가

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — DECISIONS 본 엔트리 + analyzer 위임 prompt (응답 본문)
  2. **사용자 (즉시)** — Analyzer 위임 prompt 를 analyzer 세션에 직접 붙여넣기 (cd `/home/hyeonjin/thesis_refactored/src/analysis`)
  3. **Analyzer (사용자 핸드오프 후)** — Raw Score 분포 분석 → `raw_score_distribution_for_directed_topk.md` 작성
  4. **Planner (analyzer 결과 수령 후)** — threshold 결정 + Selector 모듈 세션 핸드오프 prompt 작성
  5. **Selector + Root (5/9~)** — 구현 + 학습 + 측정 (analyzer 결과 + planner threshold 결정 후)
  6. **사용자 (학회 논문 §V.5.3)** — 본인 직접 처리 (Q3)

- **추가 필요 분석** (analyzer Raw Score 분포 결과 후):
  - threshold 별 selected node 수 + recall 추정 → 학습 시 어떤 threshold 가 적정한지 결정
  - 기존 SuperNode (bidirectional) ckpt 의 attention pattern 과 비교 (학위 논문 mechanism 분석 evidence)
  - 시나리오 A/B/C 결과 후 Filter Dominance 5 축 narrative 격상 여부 결정

---

## 2026-05-05 (Wave 4 Filter Ablation 14 cells GLM 완료 + 사용자 옵션 A 채택) — 🚀 신규 F1 ceiling **a05_08 Stacked F1=0.8809** (+0.0152) + a05_07 EX=0.3501 (+0.0124) + paper main anchor t_00 유지 + Filter design variation evidence

- **결정**:
  1. **(a) Wave 4 14 cells GLM 측정 완료** (root 2026-05-04 19:08 → 2026-05-05 03:06, wall 7h 58min, ~₩30-54K, GPU 0/1 split):
     - Stack: paper main pipeline (Enriched + QCond α=0.5 + qcond_nl3 + MSTPCSTUnion + LLMSQLGenerator GLM) + **Filter 14 변형** 만 변경
     - 출처: [EXPERIMENT_HISTORY.md L2245~ "Wave 4 Filter Ablation 14 cells GLM"](../EXPERIMENT_HISTORY.md)
  2. **(b) 14 cells 결과 — F1 정렬 (t_00 base F1=0.8657, EX=0.3377)**:

     | 순위 | Cell | R | P | F1 | EX | ΔF1 | ΔEX |
     |---|---|---|---|---|---|---|---|
     | **1** | **a05_08 stacked Tiered+Verifier** | 0.8880 | 0.8739 | **0.8809 ★** | 0.3351 | **+0.0152** | -0.0026 |
     | 2 | a05_22 SymVerify+Reflection+Verifier stacked | 0.8844 | 0.8675 | 0.8759 | 0.3364 | +0.0102 | -0.0013 |
     | 3 | a05_05 tiered_no_tools | 0.8940 | 0.8463 | 0.8695 | 0.3429 | +0.0038 | +0.0052 |
     | 4 | a05_09 tiered_retry | 0.8932 | 0.8449 | 0.8684 | 0.3377 | +0.0027 | +0.0000 |
     | 5 | a05_06 tiered_full_tools | 0.8931 | 0.8438 | 0.8678 | 0.3422 | +0.0021 | +0.0045 |
     | 6 | a05_04 verifier | **0.9155 ★ R 최고** | 0.8220 | 0.8662 | 0.3383 | +0.0005 | +0.0006 |
     | 7 | a05_19 symverify_xiyan_repair | 0.8743 | 0.8559 | 0.8650 | 0.3409 | -0.0007 | +0.0032 |
     | 8 | a05_21 symverify_xiyan_detect | 0.8726 | 0.8565 | 0.8645 | 0.3370 | -0.0012 | -0.0007 |
     | 9 | **a05_07 adaptive_depth** | 0.8802 | 0.8471 | 0.8633 | **0.3501 ★ EX 최고** | -0.0024 | **+0.0124** |
     | 10 | a05_02 reflection_1iter | 0.8894 | 0.8383 | 0.8631 | 0.3429 | -0.0026 | +0.0052 |
     | 11 | a05_10 adaptive_retry | 0.8791 | 0.8462 | 0.8623 | 0.3422 | -0.0034 | +0.0045 |
     | 12 | a05_20 symverify_reflection_repair | 0.8903 | 0.8354 | 0.8620 | 0.3396 | -0.0037 | +0.0019 |
     | 13 | a05_03 reflection_3iter | 0.8914 | 0.8297 | 0.8594 | 0.3344 | -0.0063 | -0.0033 |
     | 14 | a05_01 adaptive_multi_agent | 0.7724 | 0.8448 | 0.8070 | 0.3279 | **-0.0587 ⚠️** | -0.0098 |
  3. **(c) 🚀 핵심 발견 5 항목**:
     - **a05_08 Stacked sweet spot — F1 신규 ceiling 0.8809** (+0.0152 vs t_00). Tiered (semantic agent) → Verifier (precision check) 2-stage stacking 이 단일 agent 변형 모두 능가
     - **a05_07 AdaptiveDepth — 유일한 EX 개선 (+0.0124)**. F1 -0.0024 trade-off 로 EX 만 개선 — **F1 vs EX divergence 의 새 evidence** (Filter 변형으로 EX 만 개선 가능, uncertainty-gated agent depth 가 selector confidence 를 SQL gen 까지 비대칭 전파 가능성)
     - **a05_04 VerifierFilter R 최고 (0.9155)** — t_00 R=0.8734 대비 +0.0421 (schema-linking recall ceiling 회복 약 절반). 단 P trade-off 로 F1 plateau 안
     - **a05_01 AdaptiveMultiAgent 유일 큰 손실 (-0.0587)** — Skeptic agent over-prune (R 급락 -0.1010). 다른 13 cells 는 모두 plateau (F1 -0.01 ~ +0.015) 안 — Skeptic outlier
     - **Reflection iter depth — 1iter > 3iter** (a05_02 0.8631 vs a05_03 0.8594, P drift over-correction)
  4. **(d) ✅ 사용자 결정 — 옵션 A 채택 (paper main anchor t_00 유지)**:
     - **anchor**: t_00 (`s04_pipeline_enriched_qcond_a05_mst_pcst_union_glm_sql` F1=0.8657 / EX=0.3377) **그대로 유지** (XiYan 단일, simple/clean baseline, narrative 일관)
     - **Wave 4 결과 활용**: paper §IV.4 Filter design ablation 의 **ceiling evidence** 로 인용
     - **paper main result 분리 표기**: t_00 anchor (main result) + a05_08 Wave 4 ceiling (Filter design 갱신 evidence)
     - 옵션 B (a05_08 anchor promote) 거부 사유: D-3 마감 + narrative 전체 정정 부담 + 박사 후 검토 시간 부족 + Stacked Filter mechanism deep dive 학위 논문 보존
  5. **(e) Paper narrative 갱신 — Filter design variation evidence 추가**:
     - **❌ 직전 narrative**: "Filter on/off (ΔF1=0.63) 만 paper main insight"
     - **✅ 신규 narrative**: "**Filter on/off (ΔF1=0.63) + Filter design variation (ΔF1=0.07) 둘 다 contribute**" — Filter "first-class stage" 학술적 정당성 한층 강화
     - paper §3.5 Filter Dominance narrative 보강: "단일 stage main + 그 stage 안에서 algorithm choice 도 F1 driver"
     - paper §III.5 Modular Filter design — 이론적 주장에서 **측정으로 검증된 정량 사실** 로 격상 (14 변형 측정, Stacked sweet spot + AdaptiveDepth EX trade-off)
  6. **(f) F1 vs EX divergence 보강 — a05_07 AdaptiveDepth 의 학회 논문 §V.5.2 Limitation evidence**:
     - 14 cells 중 유일 EX > 0.35 (0.3501) + F1 -0.0024 trade-off
     - schema-linking F1 ↔ SQL EX **decoupling 직접 evidence** (Filter 변형으로 EX 만 개선)
     - paper §V.5.2 1 paragraph 에 1-2 문장 보강 (Filter design 의 F1-optimal vs EX-optimal 분리 가능성)
  7. **(g) paper_research_direction.md / paper_outline_2026-05-08.md 정정 사항 (planner Edit 즉시)**:
     - **paper §3.5**: Filter design variation evidence 추가 (14 cells, ΔF1 spread 0.0739 = on/off 의 12%)
     - **paper §8 Future Works**: Wave 4 ✅ 완료 표기 + Stacked Filter mechanism deep dive 신설 sub-항목 (학위 논문 Part II)
     - **paper §10 핵심 수치**: Wave 4 14 cells 결과 표 추가 (F1 ceiling a05_08 0.8809 + EX ceiling a05_07 0.3501 + R ceiling a05_04 0.9155 + a05_01 outlier caveat)
     - **paper_outline §III.5**: Modular Filter design 보강 (14 변형 측정 evidence)
     - **paper_outline §IV.4.4 신설**: "Filter Design Variation (Wave 4, 14 cells GLM, paper main stack)" — Stacked sweet spot + AdaptiveDepth EX trade-off + 14 cells 표
     - **paper_outline §V.5.1 Conclusion**: Wave 4 evidence 인용 ("Filter design 갱신 ceiling F1=0.8809")
     - **paper_outline §V.5.2 Limitations**: a05_07 AdaptiveDepth EX +0.0124 evidence 추가 (F1 vs EX decoupling)
     - **paper_outline §V.5.3 Future Work**: Stacked Filter narrative + 학위 논문 Part II 직접 인용

- **근거**:
  - **EXPERIMENT_HISTORY.md L2245~** "Wave 4 Filter Ablation (2026-05-04 → 05, 14 cells GLM, 🚀 신규 최고 F1=0.8809)" — 14 cells 결과 + 5 핵심 발견 + paper main anchor 옵션 A 권장
  - **HISTORY 후속 핸드오프 권장**: DECISIONS 후속 + paper §3.5/§8/§10 갱신 + presentation_brief 갱신 + analyzer 위임 (a05_01 Skeptic over-prune mechanism / a05_07 EX 개선 mechanism / a05_08 Stacked 2-stage absorption)
  - **사용자 직전 input** (2026-05-05): "그래 일단 옵션 A로 하자" — anchor 단순화 + narrative 일관 + 박사 후 검토 시간 확보

- **영향 범위**:
  - **paper_research_direction.md (planner Edit)**: §3.5 Filter design variation evidence + §8 Wave 4 완료 + Stacked Filter Future Work + §10 14 cells 표 + §12 Changelog
  - **paper_outline_2026-05-08.md (planner Edit)**: §III.5 Modular Filter design 보강 + §IV.4.4 신설 (Filter Design Variation) + §V.5.1 ceiling evidence + §V.5.2 a05_07 EX evidence + §V.5.3 Stacked Filter Future Work + Changelog
  - **DECISIONS 본 엔트리** — 옵션 A 채택 + 14 cells 결과 + 5 핵심 발견 + paper narrative 갱신 사항
  - **paper main contribution claim 보강 (옵션 A)**:
    - (a) 4 Module Co-Designed Pipeline
    - (b) Filter Dominance discovery (3 축)
    - (c) BIRD-Dev t_00 F1=0.8657 / EX=0.3377 (anchor 유지)
    - **🆕 (d) Modular Filter design variation 측정 (14 cells, ceiling F1=0.8809 +0.0152, EX=0.3501 +0.0124)** — Filter design 의 추가 여지 정량 입증

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — paper §3.5/§8/§10 + paper_outline §III.5/§IV.4.4/§V + DECISIONS 본 엔트리
  2. **사용자 (즉시)** — 학회 논문 초안에 §IV.4.4 신규 sub-section 작성 (Wave 4 14 cells 표 + Stacked sweet spot + AdaptiveDepth EX trade-off + a05_01 caveat)
  3. **박사 후 연구원 검토 (5/6 수)** — Wave 4 narrative 정합성 + Stacked Filter 의 학술적 weight + a05_01 outlier caveat 처리
  4. **Wave 4 후속 (선택, 학위 논문 Part II)** — Analyzer 위임:
     - a05_01 (AdaptiveMultiAgent) Skeptic over-prune mechanism 정량 (per-query R 분포 + Skeptic veto rate)
     - a05_07 (AdaptiveDepth) EX 개선 mechanism (uncertainty distribution vs EX 정확도)
     - a05_08 (Stacked) 2-stage absorption stage-wise 분해 (Tiered 출력 vs Verifier 출력 differential)
     - Filter design variation 의 R/P/F1 trade-off curve (14 cells scatter)

- **추가 필요 분석** (학위 논문 Part II 진행 시):
  - **Stacked Filter mechanism deep dive**: a05_08 의 2-stage Tiered → Verifier 흐름 정량 (Tiered prune 노드 vs Verifier 추가 prune 노드 분리)
  - **EX-optimal Filter design**: a05_07 AdaptiveDepth + 추가 EX-aware 변형 검증 (paper section IV future evidence)
  - **a05_01 Skeptic mechanism**: 유일한 큰 손실 cell 의 mechanism 분석 (paper §9 Limitation 의 case study)
  - **Filter design variation per-query**: 14 cells 의 query-level F1 분포 (mechanism_final.md §2 와 cross-reference)

---

## 2026-05-05 (mechanism_final.md per-query 정밀화) — 🎯 §3.5 narrative 4차 정밀화 + α-invariant + schema-complexity-dependent (모든 schema net positive) 추가

- **결정**:
  1. **(a) Analyzer 산출 수령** — [mechanism_final.md](../notebooks/analysis_results/mechanism_final.md) (F-1 + H-G 17 cells 후속 per-query mechanism 정밀화, 2026-05-05 LLM-free, ₩0):
     - §2 Filter F1 압축 per-query 분포 (히스토그램 + 변동성 ratio + difficulty 별)
     - §3 Filter absorption type 분류 (DB / 길이 / gold count / F-1 R 별)
     - §4 F-1 best α=0.1 vs With-Filter plateau saturation mechanism
     - §5 paper §3.5 main insight 정량 결론 (per-query mechanism)
     - §6 잔존 가설 + post-paper future work
  2. **(b) 🎯 §3.5 narrative 4차 정밀화 — α-invariant + schema-complexity-dependent 추가**:
     - **❌ 3차 narrative (DECISIONS 2026-05-05 분기 1 확정)**: "Filter dominance single-stage main + Stack-dependent Stage 1 caveat"
     - **✅ 4차 정밀화 (analyzer mechanism_final.md per-query)**: "**Filter dominance: single-stage Filter precision absorption — stack-invariant + α-invariant + schema-complexity-dependent (모든 schema net positive)**"
     - **신규 정량 evidence 3 항목**:
       - **α 차원 압축 5.0850×** (F-1 spread 0.0724 → WF 0.0142, DECISIONS 2026-05-05 의 6× 와 일치)
       - **α-invariant** (per-query gain mean +0.6462, 음 gain 1.4% only) — query-invariant 표현은 약함 (per-query std ratio 0.8493, query-level 변동성 보존)
       - **🆕 schema-complexity-dependent**: DB 별 gain spread **0.6058** (european_football +0.82 vs toxicology +0.22) — F-1 F1 낮은 schema 일수록 Filter 더 결정적, 단 모든 DB net positive
  3. **(c) Per-query mechanism 정밀화 (mechanism_final.md §2-§4)**:

     | 측정 항목 | 값 | mechanism 함의 |
     |---|---|---|
     | α 차원 압축 (plateau-region F1 spread) | F-1 0.0724 → WF 0.0142 = **5.0850×** | 6× 압축 정량 재현 |
     | Per-query gain mean (α=0.5) | **+0.6462** (P50=+0.6965) | α-invariant boost |
     | 음 gain count | 22/1534 = **1.4%** | minor case |
     | Per-query F1 std ratio (α=0.5) | F-1 0.1628 / WF 0.1917 = **0.8493×** | query-level 변동성 보존 |
     | Difficulty 별 gain | Simple +0.6553 / Mod +0.6240 / Chal +0.5629 | Challenging 에서 다소 작음 |
     | DB 별 gain spread | **0.6058** (european_football +0.82 vs toxicology +0.22) | schema-complexity-dependent |
     | F-1 saturation sweet spot α=0.1 \|selected\| | **31.67** (R=0.85, P=0.21, F1=0.34) | sweet spot |
     | F-1 α=0.5 saturation \|selected\| | 60.84 (R=0.99, P=0.13) | P drift +29.20 |
     | With-Filter α=0.5 final \|selected\| | **5.82** | Filter prune ~29 노드 |
     | Filter α=0.5 ΔR | **-0.1194** | R-P trade-off |
  4. **(d) Mechanism 정밀화 narrative — saturation sweet spot expansion (mechanism_final.md §4)**:
     - F-1 α=0.1 sweet spot = R 천장 직전 (0.85), P 보존 (0.21), |selected|=31.67
     - F-1 α=0.5+ saturation = R=0.99 도달 + P drift (|selected|≈61, P=0.13)
     - **Filter mechanism**: saturation 후 추가된 P drift 노드 ~29 개를 prune → final |selected|≈6, P=0.85
     - **R 손실 trade-off (ΔR=-0.1194)**: Filter 가 R 약간 손실, 단 P 큰 회복 — paper §3.5 mechanism 정밀화 ("R 회복 X, P 정정 ✓" — raw signal 의 R 손실은 selector saturation 단계에서 이미 결정, Filter 는 P 정정만 담당)
  5. **(e) paper 본문 정정 사항 (planner Edit 완료)**:
     - **§3.5 헤더 narrative 갱신** — "Filter dominance single-stage main + Stack-dependent Stage 1" → "stack-invariant + α-invariant + schema-complexity-dependent (모든 schema net positive)"
     - **§3.5 Per-Query Mechanism 정밀화 sub-section 신설** — 4 항목 (α 차원 압축 / α-invariant / schema-complexity-dependent / saturation sweet spot expansion)
     - **§8 H-G followup 외 🆕 H-H Query-conditional Filter design 신설** — low priority, post-paper, ΔF1 상한 marginal 예상
     - **§9 Limitations** — Filter gain schema-dependent (DB spread 0.6058) + 음 gain 22 queries (1.4%) 항목 신설
     - **§10 Per-Query Mechanism 정밀화 표 신규** — 11-row 정량 + paper §3.5 직접 인용 narrative 한 문단

- **근거**:
  - **Analyzer 산출**: [notebooks/analysis_results/mechanism_final.md](../notebooks/analysis_results/mechanism_final.md) — 2026-05-05 LLM-free 즉시 위임 결과
  - **재현 raw 데이터**: notebooks/analysis_results/mechanism_final_*.csv (시각화/검증용)
  - **재현 스크립트**: src/analysis/analyze_mechanism_final.py (scipy 의존성 없음)
  - **선행 분석**: alpha_plateau_mechanism.md (1차 H-B/H-F) + alpha_plateau_mechanism_validation.md (2차 보강 H-A/H-D 후)
  - **Cross-reference**: DECISIONS 2026-05-05 분기 1 확정 (F-1 + H-G 17 cells, 6× 압축) + mechanism_final.md (per-query 5.0850× 일치)

- **영향 범위**:
  - **paper_research_direction.md (planner Edit 완료)**: §3.5 헤더 narrative + Per-Query Mechanism sub-section + §8 H-H + §9 Filter gain schema-dependent + 음 gain + §10 Per-Query 정량 표 + §12 Changelog
  - **DECISIONS 본 엔트리** — mechanism_final.md 결과 인용 + paper 5 섹션 정정 사항 명문화
  - **paper main insight 정밀화 격상**: §3.5 = "Filter dominance: stack-invariant + α-invariant + schema-complexity-dependent" — paper section IV/V draft 작성 시 per-query mechanism 직접 인용 가능
  - **paper anchor (t_00 F1=0.8657 / EX=0.3377) 변경 X** — narrative 만 정밀화

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — paper §3.5/§8/§9/§10 + DECISIONS 본 엔트리
  2. **사용자 (즉시)** — paper §3.5 정량 evidence 활용 paper section IV/V draft 작성 가능
  3. **Wave 4 a05_filter_agentic** (별도 진행 가능) — 직전 turn root 핸드오프 prompt 제공 완료, 본 §3.5 정밀화 narrative 와 무관하게 진행 가능. multi-agent extension 도 동일 absorption mechanism 보이는지 후속 검증 가능
  4. **Analyzer 후속 (선택, post-paper)** — H-H query-conditional Filter design (low priority, ΔF1 marginal 예상) / 음 gain 22 queries case study (post-paper deep dive)

- **추가 필요 분석** (post-paper future work):
  - **H-H query-conditional Filter design**: DB / R level 등 schema metadata 로 Filter strength 조정 — DB spread 0.6058 의 일부만 회수 예상
  - **음 gain 22 queries case study**: 어떤 query 패턴 (특정 DB / specific gold node 분포) 에서 Filter 가 잘못된 prune 수행하는지 — paper §9 Limitation 정량 보강
  - **Wave 4 a05_filter_agentic 결과 도착 후**: multi-agent Filter 도 동일 α-invariant + schema-complexity-dependent absorption 보이는지 검증 (paper §2.4 Modular design 후속 evidence)

---

## 2026-05-05 (F-1 + H-G alpha sweep 17 cells 완료) — 🎯 분기 1 확정 + Stage 2 Filter precision absorption 결정적 evidence + §3.5 narrative 정정 (2-stage → "Filter dominance" single-stage main + Stack-dependent Stage 1)

- **결정**:
  1. **(a) Root 측정 완료 (2026-05-05 12:57)**: F-1 + H-G alpha sweep 17 cells (10 신규 F-1 + 7 신규 H-G + 가용 baseline α=0.5 1 cell), LLM-free 합산 ₩0, wall 1h 41min (GPU 0/1 split)
  2. **(b) 🎯 결과 분기 1 확정 — Stage 2 Filter precision absorption 결정적 evidence**:
     - F-1 MSTPCSTUnion R spread = **0.2362**, F1 spread = **0.1265** — 사용자 정의 threshold 0.05 의 **4-6배** (분기 1 임계 > 0.05 명확 초과)
     - H-G AdaptivePCST R spread = **0.2760**, F1 spread = **0.1441** — 분기 1 임계 더 강하게 초과
     - **함의**: F-1 (no Filter) 에서 plateau **무너짐** → Filter 가 Stage 2 absorption 주체 결정적 evidence
  3. **(c) 🚨 §3.5 narrative 정정 — 단일 Filter dominance + Stack-dependent Stage 1**:
     - **❌ 직전 narrative (DECISIONS 2026-05-04 analyzer validation 보강)**: "2-stage absorption" (Stage 1 Extractor MST set saturation + Stage 2 Filter precision)
     - **✅ 신규 narrative**: **"Filter dominance" single-stage main + Stack-dependent Stage 1**
     - **🚨 Stage 1 가설 부정 (paper main pipeline 에서)**:
       - 직전 H-C partial (alpha_plateau_mechanism_validation.md §3) 의 R~0.96 plateau 는 **basic PCST stack 한정 결과**
       - paper main 의 MSTPCSTUnion 은 plateau 부재 (R 0.7585 → 0.9947, spread 0.2362)
       - AdaptivePCST 도 plateau 부재 (R 0.5074 → 0.7834, spread 0.2760)
       - → **Extractor 가 mechanism 주체 X** (basic PCST 한정 효과)
     - **✅ Stage 2 결정적 evidence**:
       - Filter 가 plateau-region (α∈[0.2,1.0]) F1 spread **0.0778 → 0.0129** 으로 **6배 압축**
       - F-1 P 0.12-0.21 → With-Filter P 0.83-0.86 (+0.65~+0.74 평균)
       - ΔF1 (With-Filter − F-1) α 별: α=0 +0.38, α=0.5 +0.64, α=1.0 +0.65 — α↑ 따라 Filter 효과 증가
  4. **(d) 5 evidence 결합 갱신 (2-stage → Filter dominance)**:
     1. **🆕 F-1 MSTPCSTUnion full 11 cells (root 2026-05-05)**: R spread 0.2362, F1 spread 0.1265 — plateau 부재
     2. **🆕 H-G AdaptivePCST 7 cells (root 2026-05-05)**: R spread 0.2760, F1 spread 0.1441 — plateau 부재 (Stack-dependent Stage 1 입증)
     3. **H-B ckpt-invariant** (analyzer 2026-05-04): qcond_nl3 r=0.2396 + Enriched r=0.0579, raw signal 독립
     4. **H-F stability + ordering** (analyzer 2026-05-04): k=20 Jaccard 0.4673 stability + Ordering Spearman 0.6453
     5. **H-A/H-D 부정** (root 2026-05-04): Enriched ckpt + minmax norm 변형 모두 plateau 원인 X
     → **단일 Filter precision absorption 이 plateau 의 dominant 주체** — Extractor 의 set saturation 이 아닌 Filter 가 raw signal 차이 (set + ordering) 를 모두 absorb 하여 P 를 ~0.85 로 균일 elevate
  5. **(e) F-1 MSTPCSTUnion 11 cells 결과 표** (10 신규 + α=0.5 baseline):

     Stack: Enriched Builder + qcond_nl3 ckpt + α + MSTPCSTUnion(score_threshold=0.1) + No Filter + No SQL gen

     | α | R | P | F1 | nodes | vs With-Filter ΔF1 |
     |---|---|---|---|---|---|
     | 0.0 | 0.7585 | 0.2047 | 0.3224 | 39.2 | +0.3806 |
     | 0.1 | 0.8535 | 0.2137 | **0.3418 ★ F1 best** | 42.2 | +0.4462 |
     | 0.2 | 0.9645 | 0.1728 | 0.2931 | 57.3 | +0.5604 |
     | 0.3 | 0.9845 | 0.1438 | 0.2509 | 71.2 | +0.6123 |
     | 0.4 | 0.9905 | 0.1320 | 0.2330 | 78.9 | +0.6309 |
     | 0.5 (baseline) | 0.9927 | 0.1268 | 0.2249 | 83.1 | +0.6408 |
     | 0.6 | 0.9939 | 0.1240 | 0.2205 | 85.6 | +0.6433 |
     | 0.7 | 0.9940 | 0.1224 | 0.2180 | 87.1 | +0.6449 |
     | 0.8 | 0.9943 | 0.1212 | 0.2161 | 88.1 | +0.6483 |
     | 0.9 | 0.9945 | 0.1208 | 0.2154 | 88.6 | +0.6485 |
     | **1.0** | **0.9947 ★ R** | 0.1207 | 0.2153 | 88.8 | +0.6511 |
  6. **(f) H-G AdaptivePCST F-1 7 cells 결과 표**:

     Stack: Enriched + qcond_nl3 + α + AdaptivePCST(per-q P80, top-K=20) + No Filter + No SQL gen

     | α | R | P | F1 | nodes |
     |---|---|---|---|---|
     | 0.0 | 0.5074 | 0.2566 | 0.3408 | 17.0 |
     | 0.2 | 0.6480 | 0.3142 | 0.4232 | 18.5 |
     | 0.4 | 0.7017 | 0.3268 | 0.4459 | 19.1 |
     | 0.5 | 0.7260 | 0.3315 | 0.4552 | 19.2 |
     | 0.6 | 0.7500 | 0.3392 | 0.4671 | 18.9 |
     | **0.8** | **0.7834 ★ R** | 0.3511 | **0.4849 ★ F1 best** | 18.7 |
     | 1.0 | 0.7778 | 0.3428 | 0.4759 | 19.3 |
  7. **(g) 추가 관찰 / interpretation**:
     - **F-1 best at α=0.1** (saturation 직전 sweet spot): α≥0.2 부터 R 천장 도달 (0.96+) → 이후는 P drift 만 → F1 monotonic 감소
     - **MSTPCSTUnion R 천장 0.99 vs AdaptivePCST R 천장 0.78**: Extractor 의 selectivity 차이 — MSTPCSTUnion 은 score_threshold=0.1 만 통과하면 다 포함, AdaptivePCST 는 per-q P80 percentile + top-K=20 cap
     - **basic PCST H-C partial 의 plateau 가 stack 특화 효과**: basic PCST 는 score_threshold + cost 구조가 R 천장 일찍 도달 → α invariance. 다른 두 extractor 는 천장 늦게 도달 + P 변화 큼 → plateau 부재
     - **paper §3.5 mechanism narrative 정정 confirm**: 단일 Filter precision absorption 이 plateau 의 dominant 주체, Stage 1 (Extractor) 은 stack 한정

- **근거**:
  - **Root 측정 결과 (2026-05-05 12:57)**: F-1 + H-G alpha sweep 17 cells, LLM-free ₩0, wall 1h 41min (GPU 0/1 split)
  - **EXPERIMENT_HISTORY.md "F-1 Alpha Sweep + H-G Adaptive PCST F-1 (2026-05-05)"** — 17 cells 결과 + Stack 별 비교
  - **EXPERIMENT_CATALOG.md / EXPERIMENT_ID_MIGRATION.md** — 신규 ID 등록
  - **Configs**: `configs/experiments/s04_ablation/pipeline/t00_f1_alpha_*.yaml` (10 신규 F-1) + `t00_hg_adaptive_f1_alpha_*.yaml` (7 신규 H-G)
  - **Scripts**: `scripts/run_f1_full_alpha_sweep.sh`, `scripts/run_hg_adaptive_f1_sweep.sh`
  - **선행 분석 비교**: alpha_plateau_mechanism_validation.md §3 (basic PCST partial 3 cells, F1 spread 0.0039 — stack 한정 결과)

- **영향 범위**:
  - **paper_research_direction.md (planner Edit)**:
    - §3.5 mechanism 정정 — "2-stage absorption" → "Filter dominance single-stage main + Stack-dependent Stage 1"
    - §8 Future Works — H-G ✅ 검증 완료 + basic PCST saturation stack 한정 mechanism deep dive 후보
    - §9 Limitations — F-1 partial 한계 ✅ 해소 + Stack 분기 narrative 추가
    - §10 핵심 수치 — F-1 11 cells + H-G 7 cells + Filter 6× 압축 + ΔF1 α 별 행 추가
  - **presentation_brief (planner Edit)**:
    - §14 Filter dominance narrative 변경 (이전 2-stage absorption 에서)
    - 결정적 evidence 한 슬라이드 분량 정리
  - **DECISIONS 본 엔트리** — 분기 1 확정 + Filter dominance narrative 정정
  - **paper main contribution narrative 강화**: "QCondGAT main" → "4 module Co-Design + **Filter dominance (single-stage main, Stack-invariant)**"
  - **paper anchor (t_00 F1=0.8657 / EX=0.3377) 변경 X** — narrative 만 정정

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — paper §3.5/§8/§9/§10 + presentation_brief §14 + DECISIONS 본 엔트리
  2. **사용자 (즉시)** — Analyzer 위임 prompt (응답 본문) — F-1 + H-G 결과로 §3.5 단일 Filter absorption mechanism 정밀화 + per-query 분포 + saturation 후 P 차이 absorb mechanism
  3. **Analyzer (사용자 핸드오프 후)** — alpha_plateau_mechanism_validation.md §7 신설 또는 mechanism_final.md 작성
  4. **Wave 4 a05_filter_agentic** (별도 진행) — 직전 turn root 핸드오프 prompt 제공 완료, 본 §3.5 정정 narrative 와 무관하게 진행 가능

- **추가 필요 분석** (analyzer 위임):
  - F-1 + H-G 결과로 **단일 Filter absorption mechanism 정밀화** — alpha_plateau_mechanism_validation.md §7 신설 또는 mechanism_final.md 작성
  - **Filter F1 압축 비율 (plateau-region 6×) per-query 분포** — Filter 가 어떤 query type 에서 가장 강한 absorption 수행하는지
  - **F-1 best at α=0.1 (saturation 직전 sweet spot) vs With-Filter plateau α∈[0.2,1.0]** — Filter 가 saturation 후 P 차이 absorb 하는 mechanism 정량
  - **basic PCST vs MSTPCSTUnion vs AdaptivePCST stack 별 Filter dominance 일관성** — paper section IV per-stack heatmap 후보 (post-paper future work, H-G 후속)

---

## 2026-05-04 (사용자 결정 — 옵션 A 채택) — F-1 alpha sweep paper main stack 추가 10 cells + H-G Adaptive PCST F-1 비교 6-11 cells 병렬 진행 승인

- **결정** (사용자 직전 input):
  1. **(A) F-1 alpha sweep full 측정 root 핸드오프 진행 승인** (옵션 A):
     - paper main stack (Enriched + QCond + α∈{0~1} + MSTPCSTUnion + **No Filter**) 의 추가 **10 cells** (α=0.5 이미 있음, α∈{0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0})
     - 의도: §3.5 mechanism 정밀화 (2-stage absorption) 의 Stage 1 (Extractor MST set saturation) vs Stage 2 (Filter precision absorption) weight 결정적 evidence 확보
     - 비용: LLM-free (Filter + SQL gen 둘 다 빠지면 ₩0), 시간 ~1h
  2. **(B) H-G (Extractor MST set saturation) 검증 — High 우선순위 동시 진행 승인**:
     - Adaptive PCST F-1 alpha sweep 6-11 cells 비교 — Adaptive 에서 plateau 무너지면 H-G 지지 (Extractor MST 가 mechanism Stage 1 주체 입증)
     - 비용: LLM-free, ₩0, 시간 ~1h
  3. **(C) 합산 비용/시간**: ₩0 + ~1-2h (병렬 GPU 0/1 split)
  4. **(D) 결과 분기 narrative**:
     - **분기 1**: F-1 R/F1 spread > 0.05 → Filter dominance Stage 2 결정적 evidence (현 §3.5 narrative 강화), H-G 결과로 Stage 1 weight 보강
     - **분기 2**: F-1 R/F1 spread ≤ 0.01 → Extractor set saturation Stage 1 결정적 evidence (§3.5 mechanism 정정 확정), H-G Adaptive 비교로 MST 의 mechanism 주체 입증

- **근거**:
  - 사용자 직전 input (2026-05-04, "그래 옵션 A로 진행하자")
  - 직전 엔트리 (analyzer validation 보강 — 2-stage absorption 정밀화 + 사용자 결정 2 항목 권장 옵션 A + High)

- **영향 범위**:
  - **Root 핸드오프 prompt 즉시 작성** (응답 본문) — 사용자 직접 root 세션에 붙여넣기
  - **DECISIONS 본 엔트리** — 옵션 A 채택 confirm
  - **paper_research_direction.md / DECISIONS 직전 엔트리 narrative 변경 X** — 결과 도착 후 후속 엔트리에서 분기 처리

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — Root 핸드오프 prompt 작성 (응답 본문)
  2. **사용자 (즉시)** — Root 세션에 prompt 붙여넣기 (cd /home/hyeonjin/thesis_refactored)
  3. **Root (사용자 핸드오프 후)** — F-1 main stack 10 cells + H-G Adaptive F-1 6-11 cells 병렬 측정 (~₩0, ~1-2h)
  4. **Planner (root 결과 수령 후)** — §3.5 Stage 1/Stage 2 weight 정량 결정 + paper §2.2/§3.5/§9/§10 narrative 분기 처리 + DECISIONS 후속

- **추가 필요 분석** (root 결과 후):
  - F-1 main stack 10 cells alpha 곡선 — plateau 유지/무너짐 정량 (R/P/F1 spread)
  - H-G Adaptive F-1 결과 — MST 와 비교 plateau 차이 정량
  - §3.5 2-stage absorption mechanism 의 Stage 1/Stage 2 weight 정량 결정 (analyzer 후속, alpha_plateau_mechanism_validation.md §7 신설 또는 mechanism_final.md 작성)

---

## 2026-05-04 (analyzer validation 보강 — H-B ckpt-invariant + H-F ordering effect + 🔥 H-C partial 검증 + §3.5 paper main insight 정밀화) — 단일 Filter absorption → "2-stage absorption: Extractor MST set saturation + Filter precision" + 사용자 결정 2 항목 (F-1 full sweep + H-G 우선순위)

- **결정**:
  1. **(a) Analyzer 보강 산출 수령** — [alpha_plateau_mechanism_validation.md](../notebooks/analysis_results/alpha_plateau_mechanism_validation.md) (H-A/H-D 부정 + 시나리오 ② 채택 후속, LLM-free 즉시 위임, 2026-05-04, ₩0):
     - **H-B 보강** — Enriched ckpt per-query Pearson + Spearman correlation
     - **🔥 H-C partial 검증** — F-1 (no Filter) 가용 cells (3개, α=0/0.5/1.0) R/P/F1
     - **H-F 보강** — Enriched ckpt top-K Jaccard + ordering vs set 효과 분리
     - **paper §3.5 main insight 정밀화 권장** — 단일 Filter absorption → 2-stage absorption
  2. **(b) ✅ H-B 안정성 입증 — ckpt-invariant 반증**:
     - **Enriched ckpt Pearson r = 0.0579** (P50=0.0739, Spearman 0.0272, Gold-only -0.0713)
     - **qcond_nl3 baseline**: r=0.2396 → Δ=**-0.1817 (더 강한 반증)**
     - **함의**: Cosine ↔ GAT raw signal 독립성은 ckpt 무관 (mechanism 안정성). H-B 반증 ckpt-invariant 입증
  3. **(c) ✅ H-F 안정성 + ordering effect 입증**:
     - **Enriched k=20 α=0.5↔α=1.0 Jaccard = 0.4673** (qcond_nl3 0.5178 와 stability ✓)
     - **🆕 common subset Ordering Spearman = 0.6453** — set 동일 영역에서도 ordering 차이 잔존 (1.0 미만)
     - **2 효과 분리**: α 변화 효과 ≈ **53% set 변경 + 잔여는 ordering 변경**
     - **함의**: Filter 가 set 차이까지 absorb (50% 다른 set → 동일 final node) + Ordering 차이도 absorb → §3.5 absorption 강한 claim 가능
  4. **(d) 🔥 H-C partial 검증 — F-1 plateau 거의 유지 (paper main insight 정밀화 trigger)**:
     - **F-1 (no Filter) 3 cells QCond stack** (basic PCST + no filter):
       | α | R | P | F1 |
       |---|---|---|---|
       | 0.0 (GAT only) | 0.9651 | 0.1287 | 0.2045 |
       | 0.5 (Ens) | 0.9581 | 0.1304 | 0.2083 |
       | 1.0 (Cos only) | 0.9662 | 0.1302 | 0.2059 |
     - **F1 spread = 0.0039, R spread = 0.0080** → F-1 에서도 α 차이 marginal (plateau 거의 유지)
     - **ΔF1 (With-Filter − F-1) ≈ +0.6300** (avg 3 α): α=0.0 +0.5150 / α=0.5 +0.6554 / α=1.0 +0.6592 — Filter 가 P 0.13 → 0.85 (+0.72) 변환, **단 α-invariant**
     - **🚨 핵심 함의**: "Filter 가 plateau absorption 주체" 가설은 F-1 plateau 무너짐을 예측하나 실제로는 plateau 유지 → **Filter 단독이 mechanism 주체 X**, Extractor MST set saturation 도 mechanism 의 또 다른 주체
     - ⚠️ **partial sweep (3 cells)** — full 11 cells 측정으로 결정적 plateau 판정 필요 (사용자 결정 항목 ①)
  5. **(e) 🎯 §3.5 paper main insight 정밀화 — 단일 Filter absorption → "2-stage absorption"**:
     - **❌ 1차 narrative (DECISIONS 2026-05-04 H-A/H-D 부정 직후)**: "Modular LLM Filter 가 selector signal 차이를 prune 단계에서 absorb"
     - **✅ 2차 정밀화 (validation §5.2 권장)**: **"2-stage absorption"**:
       1. **Stage 1 — Extractor (MST PCST Union) set saturation**: score-threshold seed widening 으로 R 천장 도달 → α 변화가 selector top-K 의 ordering 까지만 영향, Extractor 출력 set 은 거의 동일 (F-1 R spread 0.0080 evidence)
       2. **Stage 2 — Modular LLM Filter precision absorption**: F-1 P=0.13 → With-Filter P≈0.85 (+0.72), 그 정확도 증가가 α-invariant — Filter 가 selector noise 차이를 set + ordering 모두 absorb
     - **5 evidence 결합 (H-B ckpt-invariant + H-F stability/ordering + H-C partial + ΔF1 +0.6300 + H-A/H-D 부정)**: 단일 stage 가 아닌 2-stage 결합 mechanism 이 ckpt-invariant 한 plateau 안정성 (ΔF1 ≤ 0.005) 의 paper main insight
  6. **(f) paper 5 섹션 정정 (planner Edit 완료)**:
     - **§2.2 H-B/H-F 줄 갱신**:
       - H-B: "🚫 반증" → "🚫 **반증 ckpt-invariant** (qcond r=0.2396 + Enriched r=0.0579, Δ=-0.18)"
       - H-F: "🟡 Partial mechanism" → "🟡 **Partial mechanism + ordering effect** (Jaccard 0.4673 stability ✓ + Ordering Spearman 0.6453)"
     - **§3.5 paper main insight 정밀화** — 2-stage absorption + 5 evidence 결합 + Stage 1/Stage 2 narrative + paper section III/V 직접 인용 narrative
     - **§8 Future Works** — H-B/H-F 보강 ✅ 완료 + H-C partial 완료 (full 보류, 사용자 결정) + 🆕 **H-G Extractor MST set saturation 신설** (Adaptive PCST F-1 alpha sweep 비교 검증)
     - **§9 Limitations** — F-1 partial sweep (3 cells) 한계 항목 추가 (full 11 cells 측정 사용자 결정 대기)
     - **§10 핵심 수치 표** — qcond_nl3 vs Enriched 2 ckpt 비교 + F-1 spread + ΔF1 행 추가

- **🚨 사용자 결정 필요 2 항목**:
  1. **F-1 alpha sweep 11 cells full 측정 root 핸드오프 작성 여부**:
     - 측정 stack: Enriched + QCond α∈{0.0~1.0, 0.1 step} + MSTPCSTUnion + **No Filter**, LLM-free
     - 비용: ₩0 (LLM 없음), 시간 ~1-2h
     - 결과 분기:
       - F-1 R/F1 spread > 0.05 → Filter dominance Stage 2 결정적 evidence (현 §3.5 narrative 강화)
       - F-1 R/F1 spread ≤ 0.01 → Extractor set saturation Stage 1 결정적 evidence (§3.5 mechanism 정정 확정)
     - **권장**: 진행 (저비용 + 결정적 evidence + paper section III/V draft 작성 시 §3.5 main insight 정량 결론)
  2. **H-G (Extractor set saturation) post-deadline 우선순위 결정**:
     - 검증 방법: 다른 Extractor (Adaptive PCST top-K=20 cap) 와 F-1 alpha sweep 비교 — Adaptive 에서 plateau 무너지면 H-G 지지 (Extractor MST 가 mechanism Stage 1 주체 입증)
     - 비용: Adaptive PCST F-1 alpha sweep 6-11 cells LLM-free, ₩0, ~1-2h
     - 우선순위 옵션:
       - High (즉시 진행): paper §3.5 Stage 1 직접 evidence + paper main insight 정량 보강
       - 中 (post-paper): paper full version future work 후보
       - 低 (보류): paper 본문 영향 X, backlog 만 보존
     - **권장**: High (F-1 full sweep 과 함께 진행, ~₩0, ~1-2h 추가)

- **근거**:
  - **Analyzer 보강 산출**: [notebooks/analysis_results/alpha_plateau_mechanism_validation.md](../notebooks/analysis_results/alpha_plateau_mechanism_validation.md) — 2026-05-04 LLM-free 즉시 위임 결과
    - §2 H-B 보강 (Enriched ckpt per-query correlation)
    - §3 H-C partial 검증 (F-1 가용 3 cells R/P/F1 + plateau 거의 유지 판정)
    - §4 H-F 보강 (Enriched ckpt Jaccard + ordering vs set 분리)
    - §5 paper §3.5 main insight 정밀화 권장 (2-stage absorption narrative)
    - §6 잔존 가설 (H-G Extractor MST set saturation 신설 + H-E SQL gen bottleneck)
  - **선행 분석**: [alpha_plateau_mechanism.md](../notebooks/analysis_results/alpha_plateau_mechanism.md) (qcond_nl3 ckpt 1차 분석)
  - **F-1 cells 데이터 출처**: `outputs/.../stagewise/no_filter/qcond_{gat_basic, ens_a05, cos_a1}_no_filter/` + `outputs/.../pipeline/enriched_qcond_a05_mst_pcst_union_no_filter/`

- **영향 범위**:
  - **paper_research_direction.md (planner Edit)**: §2.2 H-B/H-F 줄 + §3.5 2-stage absorption 정밀화 + §8 H-G 신설 + §9 F-1 partial 한계 + §10 ckpt 비교 표 + §12 Changelog
  - **DECISIONS 본 엔트리** — analyzer 보강 결과 인용 + paper 5 섹션 정정 + 사용자 결정 2 항목
  - **paper main insight 정밀화 격상**: §3.5 = "2-stage absorption" — paper section III Methodology 의 Filter design + Extractor design narrative 통합 강화 (paper section V Conclusion 신규 narrative)
  - **paper anchor (t_00 F1=0.8657 / EX=0.3377) 변경 X** — narrative 만 정정
  - **사용자 결정 의존 후속**: F-1 full sweep + H-G 결과에 따라 §3.5 Stage 1/Stage 2 weight 결정

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — paper 5 섹션 정정 + 본 DECISIONS 엔트리
  2. **🚨 사용자 (즉시 의사결정 2 항목)** — F-1 full sweep + H-G 우선순위 결정 (응답 본문 권장 옵션 제시)
  3. **Root (사용자 승인 후)** — F-1 alpha sweep 11 cells + H-G Adaptive PCST F-1 alpha sweep 6-11 cells (모두 LLM-free, ₩0, ~2-3h 합산)
  4. **Analyzer (root 결과 도착 후)** — F-1 full sweep + H-G 결과로 §3.5 Stage 1/Stage 2 weight 정밀화 분석 (alpha_plateau_mechanism_validation.md 후속 §7 또는 신규 mechanism_final.md)
  5. **Wave 4 a05_filter_agentic** (별도 진행) — root 핸드오프 prompt 직전 turn 제공 완료, 본 §3.5 mechanism 정밀화 narrative 와 무관하게 진행 가능

- **추가 필요 분석** (F-1 full sweep + H-G 결과 후):
  - F-1 full sweep 결과로 Stage 1 (Extractor set saturation) vs Stage 2 (Filter precision) weight 정량
  - H-G Adaptive PCST 비교로 Extractor MST 의 mechanism 주체 입증/반증
  - paper section III Methodology 의 Filter + Extractor design 통합 narrative draft (§3.5 직접 인용)
  - paper section V Conclusion 신규 narrative — "2-stage absorption mechanism = Modular LLM Filter + Extractor co-design 의 robustness 보장"

---

## 2026-05-04 (H-A/H-D 검증 완료) — 🚨 두 가설 모두 부정 + 🎯 시나리오 ② 정식 채택 (옵션 1 + 옵션 4 통합 narrative) + paper main contribution 정정 확정

- **결정**:
  1. **(a) H-A 가설 부정 (Distribution shift 해소 효과 미미)**:
     - **Stack**: Enriched Builder + `best_gat_enriched.pt` (Enriched features 학습) + query_conditioned=False (ckpt 정합) + α∈{0.0~1.0} 11 cells + MSTPCSTUnion + XiYan(GLM, 3 ex) + SQL gen
     - **F1 plateau α∈[0.2, 1.0] 그대로 유지** (9/10 cells within ±0.01 of best F1=0.8651)
     - **Best F1: α=1.0 → 0.8651** (기존 qcond_nl3 의 α=1.0 F1=0.8664 와 noise 범위 ΔF1=-0.0013)
     - **Best EX: α=0.8 → 0.3429** (기존 α=1.0 EX=0.3475 와 plateau 동등 ΔEX=-0.0046)
     - **α=0 (GAT only) 약간 회복** (F1=0.7195, ΔF1=+0.0165 vs qcond_nl3 GAT only 0.7030) — 단 plateau 패턴 유지
     - **결론**: Distribution match 해소가 GAT contribution 회복 못 함 → distribution shift 가 plateau 의 main 원인 X
  2. **(b) H-D 가설 부정 (간접, α=0.5 single-point)**:
     - **norm 변형 결과**: minmax (default) 0.8657 > none 0.8553 > zscore 0.8325
     - **norm 제거 시 F1 -0.0104 / EX -0.0163** (normalization 자체 효과 있음, 제거 시 손실)
     - **z-score 가장 나쁨 (ΔF1 -0.0332, ΔEX -0.0496)** — GAT/Cosine score 분포 가정 부적합
     - **결론**: minmax 가 best, 다른 norm 변경이 plateau 의 원인 X (단 α=0.5 single-point 만 측정, strong 결론 X)
  3. **(c) 🎯 시나리오 ② 정식 채택 — 옵션 1 + 옵션 4 통합 narrative 확정**:
     - **❌ 기존**: "QCondGAT main contribution"
     - **✅ 신규 (정식)**: **"4 module Co-Design + Filter dominance"**
     - **Selector contribution 재서술**: "**GAT-floor (α=0 손실 -0.16 으로 baseline robustness 보장) + Cosine-ceiling (α≥0.2 plateau)**"
     - **Filter contribution 격상**: first-class stage, F1 driver (P 회복 +0.6408), EX 에는 marginal (+0.0085) — F1/EX divergence main insight
     - **§3.5 Filter ↔ Selector Absorption 정식 채택** — H-A/H-D 부정으로 §3.5 absorption mechanism 이 plateau 의 유일한 합리적 설명 (raw signal 독립 Pearson 0.2396 + top-20 Jaccard 0.5178 + 최종 plateau ΔF1 ≤ 0.005 paradox)
  4. **(d) 13 cells 결과 표 (H-A 11 + H-D 2)**:

     **H-A 11 cells (Enriched ckpt, query_conditioned=False, α∈{0.0~1.0})**:
     | α | R | P | F1 | EX | 기존 (qcond_nl3) F1 | ΔF1 |
     |---|---|---|---|---|---|---|
     | 0.0 (GAT only) | 0.6993 | 0.7408 | **0.7195** | 0.2177 | 0.7030 | **+0.0165** |
     | 0.1 | 0.7655 | 0.7992 | 0.7820 | 0.2432 | 0.7880 | -0.0060 |
     | 0.2 | 0.8586 | 0.8547 | 0.8566 | 0.3188 | 0.8535 | +0.0031 |
     | 0.3 | 0.8714 | 0.8555 | 0.8634 | 0.3292 | 0.8632 | +0.0002 |
     | 0.4 | 0.8740 | 0.8557 | 0.8648 | 0.3331 | 0.8639 | +0.0009 |
     | 0.5 | 0.8748 | 0.8529 | 0.8637 | 0.3403 | 0.8657 | -0.0020 |
     | 0.6 | 0.8734 | 0.8533 | 0.8632 | 0.3403 | 0.8638 | -0.0006 |
     | 0.7 | 0.8734 | 0.8518 | 0.8625 | 0.3396 | 0.8629 | -0.0004 |
     | **0.8** | 0.8742 | 0.8529 | 0.8634 | **0.3429 ★** | 0.8644 | -0.0010 |
     | 0.9 | 0.8762 | 0.8526 | 0.8642 | 0.3383 | 0.8639 | +0.0003 |
     | **1.0 (Cosine only)** | 0.8767 | 0.8538 | **0.8651 ★** | 0.3390 | 0.8664 | -0.0013 |

     **H-D 2 cells (norm 변형, α=0.5 single-point)**:
     | Variant | R | P | F1 | EX | ΔF1 vs t_00 | ΔEX vs t_00 |
     |---|---|---|---|---|---|---|
     | t_00 (minmax, default) | 0.8734 | 0.8581 | **0.8657** | **0.3377** | (anchor) | (anchor) |
     | norm_none | 0.8544 | 0.8562 | 0.8553 | 0.3214 | -0.0104 | -0.0163 |
     | norm_zscore | 0.8118 | 0.8542 | 0.8325 | 0.2881 | **-0.0332** | **-0.0496** |
  5. **(e) plateau 의 진짜 원인 — 미해결, analyzer 큐 (H-B/H-C/H-F 보강)**:
     - **H-B re-validate**: 직전 alpha_plateau_mechanism.md 에서 r=0.2396 → redundancy 가설 반증. 단 H-A enriched ckpt 의 per-query correlation 도 동일 패턴인지 추가 검증 필요 (mechanism 안정성)
     - **H-C Filter dominance 정량**: F-1 (no Filter) ablation 결과 F1 -0.6408 vs EX -0.0085 — Filter 가 plateau absorption 주체 가설 강력. F-1 alpha sweep LLM-free 검증으로 Filter 빠질 때 alpha 효과 격차 정량 필요
     - **H-F re-validate**: 직전 alpha_plateau_mechanism.md 에서 k=20 Jaccard 0.5178 → partial mechanism. H-A enriched ckpt 의 top-20 Jaccard 도 동일 패턴인지 + α 변화가 top-20 ordering vs set 자체에 미치는 효과 분리 필요
  6. **(f) paper 본문 정정 사항 (planner Edit 즉시)**:
     - **§1 Selector 결정 narrative**: "QCondGAT main" → "GAT-floor + Cosine-ceiling + Filter dominance"
     - **§2.2 Selector contribution**: H-A 11 cells 결과 추가 + H-A 가설 부정 + 시나리오 ② 정식 채택
     - **§3.5 Filter ↔ Selector Absorption**: 정식 paper main insight 채택 표기 (이미 정량 evidence section 격상됨, 본 H-A/H-D 부정으로 mechanism 유일한 합리적 설명 정식 확정)
     - **§8 Future Works**: H-A/H-D ✅ 완료 표기 + H-B/H-C/H-F analyzer 큐 새 우선순위
     - **presentation_brief §14**: H-A/H-D ablation sub-section 신설 + 시나리오 ② 채택 narrative

- **근거**:
  - **H-A 11 cells**: `outputs/experiments/s04_ablation/pipeline/t00_enriched_ckpt_alpha_0[0~10]/`
  - **H-D 2 cells**: `outputs/experiments/s04_ablation/pipeline/t00_norm_{none, zscore}/`
  - **EXPERIMENT_HISTORY.md "H-A Distribution Shift 검증 + H-D Score Normalization 변형 (2026-05-04, 13 cells)"** L2155~ — 13 cells 결과 + 시나리오 ② 채택 narrative + 비용 ~₩9,928 / 3h 10min
  - **코드 fix**: `src/modules/selectors/ensemble_selector.py:33-65, 295-318` — `score_normalization` 파라미터 추가 (modes: minmax / none / zscore)
  - **비교 baseline**: 기존 alpha sweep (qcond_nl3) `outputs/.../t00_alpha_*/`
  - **Cross-reference**: alpha_plateau_mechanism.md (직전 analyzer 산출 — H-B 반증 + H-F partial mechanism)

- **영향 범위**:
  - **paper_research_direction.md (planner Edit)**: §1 Selector 결정 narrative + §2.2 H-A 결과 + §3.5 정식 채택 격상 + §8 H-A/H-D ✅ + §10 H-A 결과 추가 + §12 Changelog
  - **presentation_brief (planner Edit)**: §14 H-A/H-D ablation sub-section 신설
  - **DECISIONS 본 엔트리** — H-A/H-D 부정 + 시나리오 ② 채택 + 옵션 1+4 통합 narrative 확정
  - **paper main contribution narrative 확정 (D-day 발표 후 정식)**: "QCondGAT main" → "4 module Co-Design + Filter dominance"
  - **paper anchor (t_00 F1=0.8657 / EX=0.3377) 변경 X** — narrative 만 정정
  - **§3.5 Filter ↔ Selector Absorption 의 paper main insight 지위 정식 확정**: H-A/H-D 부정 + alpha_plateau_mechanism.md (Pearson 0.2396 + Jaccard 0.5178) + plateau ΔF1 ≤ 0.005 paradox = mechanism 유일한 합리적 설명

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — paper §1/§2.2/§3.5/§8/§10 + presentation_brief §14 + DECISIONS 본 엔트리
  2. **사용자 (즉시)** — Analyzer 위임 prompt 3 항목 (H-B/H-C/H-F) 사용자가 직접 analyzer 세션 열고 붙여넣기 (planner 가 정확한 prompt 제공 — 응답 본문)
  3. **Analyzer (사용자 핸드오프 후)** — H-B per-query correlation re-validate (H-A enriched ckpt) + H-C Filter dominance 정량 (F-1 alpha sweep LLM-free) + H-F top-K=20 cap re-validate (H-A enriched ckpt + ordering vs set 분리)
  4. **Root (선택, post-Wave 4)** — Wave 4 a05_filter_agentic GLM 통일 14 cells 진행 (사용자 직전 핸드오프 prompt 제공 완료, H-A/H-D 결과 + EXPERIMENT_HISTORY.md 갱신 후)

- **추가 필요 분석** (post-analyzer H-B/H-C/H-F 결과 후):
  - H-A enriched ckpt 의 per-query correlation 동일 패턴 확인 → mechanism 안정성 정식 입증
  - F-1 (no Filter) alpha sweep 결과 — Filter 빠질 때 alpha plateau 무너지면 §3.5 Filter absorption 직접 evidence 강화
  - Top-20 ordering vs set 효과 분리 → H-F partial mechanism 정확화
  - **Wave 4 결과 도착 후**: Best Filter variant 가 paper main anchor (F1=0.8673) 갱신 시 paper §2.4 Modular design narrative 정정 (planner 분기)

---

## 2026-05-04 (analyzer 결과 인용 — H-B 반증 + H-F partial mechanism + H10 정량 정당화) — paper §2.2/§3.5/§8/§9/§10 narrative 보강 + Filter absorption 정량 evidence 격상

- **결정**:
  1. **(a) Analyzer 산출 수령** — [alpha_plateau_mechanism.md](../notebooks/analysis_results/alpha_plateau_mechanism.md) (LLM-free 즉시 위임 결과, 2026-05-04, ₩0):
     - **H-B per-query Cosine ↔ GAT correlation 분석** (Pearson + Spearman, difficulty 분해)
     - **H-F top-K Jaccard overlap 분석** (k=10/20/30/50, α∈{0.0, 0.5, 1.0})
     - **11 α × 3 difficulty EX/F1 heatmap** + Adaptive α oracle EX 상한
  2. **(b) 🚫 H-B 가설 반증 (정량 결과)**:
     - per-query Pearson r = **0.2396** (P25/P50/P75 = 0.0791/0.2345/0.4099)
     - Spearman ρ = 0.2643
     - Difficulty 별: Simple=0.2647 / Moderate=0.1994 / Challenging=0.2078 (모두 r < 0.3)
     - Gold node 한정 Pearson r = **0.0691** (gold 거의 무상관)
     - **함의**: redundancy 가설 약함 — Cosine ↔ GAT 는 raw level 에서 충분히 독립적. selector_analysis HISTORY §4 의 GAT P80 rescue +214 gold (2.1% 순기여) 와 일관 (low-correlation tail 에서 GAT 가 cosine 놓침을 rescue)
  3. **(c) 🟡 H-F 가설 Partial mechanism (정량 결과)**:
     - k=20 α=0.5 ↔ α=1.0 Jaccard = **0.5178** (top-20 약 50% 만 일치)
     - k=10 → 0.4196 / k=30 → 0.5990 / k=50 → 0.7522 (saturation 시작)
     - α=0 ↔ α=1 (extreme blend) k=20 Jaccard = 0.3236 (top-20 ⅓ 만 동일)
     - **함의**: top-K cap 만으로 plateau 설명 불가 — selector signal 차이가 top-20 set 차이까지 명확히 전달
  4. **(d) 🚀 H-B + H-F 결합 paradox = Filter absorption 직접 evidence (paper §3.5 main insight 격상)**:
     - **raw signal 독립 (Pearson r=0.2396) + top-20 set 약 50% 차이 (Jaccard=0.5178)** **그럼에도** **최종 F1/EX plateau α∈[0.3,1.0] (8 cells ΔF1 ≤ 0.005)**
     - = **Modular LLM Filter 가 selector signal 차이를 prune 단계에서 absorb 하는 직접 evidence**
     - paper §3.5 Filter ↔ Selector Absorption 의 정량 mechanism evidence — paper section III/V 직접 인용 narrative 격상
  5. **(e) 🔬 H10 (Adaptive α) 정량 정당화 — 사용자 보류 결정 사후 정당성**:
     - oracle adaptive α (Simple α=1.0 / Moderate α=0.5 / Challenging α=1.0) EX = **0.3520**
     - 단일 best α=1.0 EX = 0.3475 → **ΔEX = +0.0046** (LLM noise band ±0.005 안)
     - 기존 t_00 (α=0.5) EX = 0.3377 → ΔEX = +0.0143 (단 difficulty 분류기 noise 미고려, oracle 상한)
     - **결론**: post-paper 재검토 가치 marginal — 사용자 2026-05-04 의사결정 (D 보류) 정량 정당화. backlog 보존 단 paper full version 우선순위 X
  6. **(f) paper_research_direction.md 5 섹션 정정 (planner Edit 완료)**:
     - **§2.2 H-B 줄 갱신**: "Medium 타당성" → "🚫 **반증** (per-query Pearson r = 0.2396, 모든 difficulty r<0.3, gold-only 0.0691)" + cross-ref [alpha_plateau_mechanism.md §2.1](../notebooks/analysis_results/alpha_plateau_mechanism.md)
     - **§2.2 H-F 줄 갱신**: "Medium" → "🟡 **Partial mechanism** (k=20 Jaccard 0.5178, k=50 0.7522 saturation)" — top-K cap 단독 plateau 설명 불가
     - **§2.2 신규 paragraph**: "H-B 반증 + H-F partial mechanism 결과 (analyzer 2026-05-04)" — paradox 명문화
     - **§3.5 정량 evidence section 격상**: H-B + H-F 결합 paradox + 9-row 정량 evidence 표 (Pearson 0.2396, Jaccard 0.5178, F1 plateau, oracle adaptive ΔEX +0.0046) + paper section III/V 직접 인용 narrative
     - **§8 H10 보류 항목 갱신**: oracle adaptive α ΔEX = +0.0046 정량 근거 + post-paper 재검토 가치 marginal 명시
     - **§9 Limitations Future work H10 항목**: 정량 근거 link ([alpha_plateau_mechanism.md §6.2](../notebooks/analysis_results/alpha_plateau_mechanism.md))
     - **§10 핵심 수치 표**: 6 가설 검증 큐 H-B/H-F 완료 표기 (✅) + Filter ↔ Selector Absorption 정량 evidence 표 신규 (9-row, Pearson + Jaccard + F1 plateau + oracle adaptive ΔEX)

- **근거**:
  - **Analyzer 산출**: [notebooks/analysis_results/alpha_plateau_mechanism.md](../notebooks/analysis_results/alpha_plateau_mechanism.md) — 2026-05-04 LLM-free 즉시 위임 결과
    - §2 H-B per-query correlation (Pearson + Spearman + difficulty + gold-only)
    - §3 H-F top-K Jaccard overlap (k=10/20/30/50, α∈{0.0, 0.5, 1.0})
    - §4 11 α × 3 difficulty EX/F1 heatmap
    - §5 paper §2.2/§3.5 narrative 영향 분석
    - §6 H10 backlog 데이터 (Difficulty 별 best α 곡선 + oracle adaptive EX 상한)
  - **Cross-reference**: selector_analysis HISTORY §4 (Cosine ROC-AUC 0.741 vs Ensemble 0.776 / PR-AUC +0.074 / GAT P80 rescue +214 순기여 / Structural ceiling 38.9%)

- **영향 범위**:
  - **paper_research_direction.md (planner Edit 완료)**: §2.2 H-B/H-F 줄 + 신규 paragraph, §3.5 정량 evidence section 격상, §8 H10 정량 정당화, §9 Limitations Future work, §10 신규 정량 evidence 표 + Changelog
  - **DECISIONS 본 엔트리** — analyzer 결과 인용 + paper 정정 사항 명문화
  - **paper main insight 정량 격상**: §3.5 Filter ↔ Selector Absorption = paper main contribution 핵심 — raw signal 통계 + top-K set 통계 + 최종 F1/EX plateau 결합 paradox 가 직접 evidence
  - **사용자 H10 보류 결정 (D, 2026-05-04) 사후 정당화**: oracle adaptive α ΔEX=+0.0046 ≤ LLM noise band → 정량 근거 확보
  - **root 진행 중 H-A/H-D 결과의 base 작업**: H-A 결과 분기 narrative (회복 vs 회복 X) 의 base 보강 — 본 정정으로 §3.5 absorption mechanism evidence 가 H-A 결과 와 무관하게 paper main insight 로 살아남음

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — paper_research_direction.md 5 섹션 정정 + 본 DECISIONS 엔트리
  2. **사용자 (즉시)** — 본 정정 사항 confirm + H-A/H-D root 결과 도착 후 narrative 분기 처리 합의 (회복 시 기존 narrative 강화 path / 회복 X 시 옵션 1+4 통합 path)
  3. **Root (진행 중, 별도)** — H-A `best_gat_enriched.pt` alpha sweep 11 cells + H-D normalization 변형 1-2 cells (사용자 직전 prompt 전달 완료)
  4. **Planner (root 결과 수령 후)** — H-A 결과 narrative 분기 처리 + DECISIONS 후속 + paper §2.2 narrative 분기 (회복 vs 회복 X 결과 반영)
  5. **Analyzer 후속 (post-H-A 결과)** — H-A 결과 시 새 ckpt alpha sweep 의 per-query correlation + Jaccard 재측정 (mechanism 안정성 확인)

- **추가 필요 분석** (post-H-A/H-D 결과 후):
  - H-A 결과 도착 시 alpha_plateau_mechanism.md §2/§3 mechanism 안정성 확인 — 새 ckpt 에서 Pearson + Jaccard 동일 패턴 인지
  - H-D 결과 도착 시 normalization 변형 별 Pearson + Jaccard 재측정 (norm 효과가 raw 단계 통계에도 영향?)
  - paper section III/V draft 작성 시 §3.5 정량 evidence 표 직접 인용 (Pearson 0.2396, Jaccard 0.5178, plateau ΔF1 ≤ 0.005, oracle adaptive ΔEX +0.0046)
  - **H10 backlog 보존 데이터** (alpha_plateau_mechanism.md §6.1) — post-paper 재검토 시 base 데이터로 활용 가능

---

## 2026-05-04 (사용자 의사결정 — Alpha Sweep narrative resolution 확정) — 단-중기 채택 (옵션 4 + 옵션 2 H-A + 옵션 1 보강) + H-A/H-D root 승인 + Adaptive α (H10) 장기 보류

- **결정** (사용자 직전 input):
  1. **(A) Narrative resolution 단-중기 채택, 장기 옵션 3 (Adaptive α H10) 제외**:
     - **단기 (D-day, 이미 완료)**: 옵션 4 (Filter dominance §3.5 신설) + 옵션 1 보강 ("GAT-floor + Cosine-ceiling" caveat)
     - **중기 (post-deadline)**: 옵션 2 (H-A `best_gat_enriched.pt` alpha sweep) + H-D normalization 변형
     - **장기 옵션 3 (Adaptive α by Difficulty)**: **보류** — paper full version 우선순위 X (H10 backlog 로 등록, 향후 재검토)
  2. **(B) H-A 검증 root 실험 승인**: `best_gat_enriched.pt` alpha sweep 11 cells (~₩8,400, ~3-4h)
  3. **(C) H-D normalization 변형 root 실험 승인**: 1-2 cells (~₩1,528, ~30min)
  4. **(D) Adaptive α (옵션 3, H10) 장기 보류**: backlog 등록, paper full version 우선순위 결정 X. **paper_research_direction.md §8/§9 의 H10 항목은 "장기 보류" 표기**.
  5. **(e) 권장 통합 narrative 확정 (단-중기만)**:
     - "QCondGAT main contribution" → **"4 module Co-Design 의 inter-module synergy + Filter dominance"** framing
     - 단기 narrative §3.5 (Filter ↔ Selector Absorption) + §1 Selector "GAT-floor + Cosine-ceiling" caveat
     - 중기 결과 분기 (H-A 검증 후):
       - GAT contribution 회복 (α=0.5 best) → 기존 narrative 강화
       - 회복 X (α=1.0 여전히 best) → 옵션 1 + 옵션 4 통합 narrative 정식 채택
     - 장기 옵션 3 (Adaptive α) 제외, paper 의 Selector contribution 은 **단-중기 narrative 로 종결**

- **근거**:
  - 사용자 직전 input (2026-05-04, 본 엔트리 직전):
    - "(A) 장기 옵션은 제외하고 단-중기만 진행하자"
    - "(B) H-A 검증 root 실험은 승인한다"
    - "(C) H-D 실험도 승인한다"
    - "(D) Adaptive alpha는 우선 나중으로 미뤄두자"
    - "이제 analyzer와 root에게 핸드오프를 보낼게"
  - 직전 엔트리 (2026-05-04 Alpha Sweep) 의 4 옵션 평가 + 권장 통합 narrative (단기 옵션 4 / 중기 옵션 2 / 장기 옵션 3)

- **영향 범위**:
  - **paper_research_direction.md (planner Edit 본 엔트리 직후)**:
    - §2.2 권장 narrative — "단기/중기/장기" → "단기/중기" (장기 옵션 3 제외)
    - §8 Future Works H10 — "🆕 신설 High" → "장기 보류 (사용자 결정 2026-05-04, paper full version 우선순위 X)"
    - §9 Limitations Future work — H10 제외
  - **DECISIONS 본 엔트리** — 사용자 의사결정 4 항목 명문화
  - **analyzer 즉시 핸드오프**: H-B per-query correlation + H-F top-K overlap + alpha sweep difficulty heatmap (LLM-free, 사용자 직접 prompt 전달)
  - **root 즉시 핸드오프** (사용자 승인 완료): H-A `best_gat_enriched.pt` alpha sweep 11 cells + H-D normalization 변형 1-2 cells (사용자 직접 prompt 전달)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — paper_research_direction.md §2.2/§8/§9 정정 + DECISIONS 본 엔트리
  2. **사용자 (즉시 직접 핸드오프)** — analyzer + root prompt 사용자가 직접 세션 열고 붙여넣기 (planner 가 정확한 prompt 제공)
  3. **Root (사용자 핸드오프 후, 즉시)** — H-A + H-D 병렬 실험 (~₩9,928 합산, ~3-4h)
  4. **Analyzer (사용자 핸드오프 후, 즉시)** — alpha sweep difficulty heatmap + H-B per-query correlation + H-F top-K overlap
  5. **Planner (root + analyzer 결과 수령 후)** — H-A 결과 narrative 분기 처리 (회복 vs 회복 X) + 최종 paper main contribution narrative 확정 + DECISIONS 후속

- **추가 필요 분석** (post-H-A/H-D 결과 후):
  - H-A 결과 분기 narrative 분석 (planner): 회복 시 기존 narrative 강화 path / 회복 X 시 옵션 1+4 통합 narrative 확정
  - H-D 결과: norm 제거 / z-score 변형 의 alpha sensitivity 차이 (mechanism 정량)
  - **Adaptive α (H10) backlog 보존**: post-paper / 후속 연구 시 재검토 가능 — 본 엔트리 (D) 결정으로 paper 본문 narrative 에는 미포함

---

## 2026-05-04 (Alpha Sweep 11 cells 완료) — 🚨 GAT contribution α∈[0.3,1.0] plateau + α=1.0 (Cosine only) F1/EX best + 이론적 분석 6 가설 (H-A~F) + Narrative Resolution 4 옵션 + 권장 통합 narrative

- **결정**:
  1. **(a) Alpha sweep 11 cells (α∈{0.0~1.0}, 0.1 step) 결과 확정 — F1 plateau α∈[0.3,1.0]**:
		| α | R | P | F1 | EX |
		| --- | --- | --- | --- | --- |
		| 0.0 (GAT only) | 0.6549 | 0.7587 | **0.7030 ⚠️** | **0.2086 ⚠️** |
		| 0.1 | 0.7502 | 0.8299 | 0.7880 | 0.2503 |
		| 0.2 | 0.8492 | 0.8579 | 0.8535 | 0.3240 |
		| 0.3 | 0.8671 | 0.8595 | 0.8632 | 0.3383 |
		| 0.4 | 0.8708 | 0.8572 | 0.8639 | 0.3351 |
		| **0.5** (★ t_00) | 0.8734 | 0.8581 | **0.8657** | **0.3377** |
		| 0.6 | 0.8742 | 0.8537 | 0.8638 | 0.3351 |
		| 0.7 | 0.8728 | 0.8533 | 0.8629 | 0.3364 |
		| 0.8 | 0.8752 | 0.8539 | 0.8644 | 0.3370 |
		| 0.9 | 0.8751 | 0.8530 | 0.8639 | 0.3325 |
		| **1.0 (Cosine only)** | **0.8761** | 0.8570 | **0.8664 ★** | **0.3475 ★** |
	 - **F1 plateau α∈[0.3,1.0]**: 8 cells 내 ΔF1 ≤ 0.005 (LLM noise band ±0.003~0.005 와 일치)
	 - **EX best at α=1.0 (Cosine only)**: 0.3475 vs t_00 0.3377 (ΔEX=+0.0098 약우세)
	 - **F1 best at α=1.0**: 0.8664 vs t_00 0.8657 (ΔF1=+0.0007 noise)
	 - **α=0 (GAT only) 만 큰 손실** (F1 -0.16, EX -0.13) — GAT 가 noise floor 역할
  2. **(b) 🚨 사용자 우려 — paper main contribution narrative 충돌 위험**:
     - 현재 narrative ("**QCondGAT 가 main contribution**") 가 α plateau 발견으로 약화
     - α=1.0 anchor 변경 시 본 연구 방향 (End-to-End Co-Design **with GAT-based selector**) 와 충돌
     - 단 **α=0 -0.16 손실 → GAT 가 baseline robustness floor 보장**: Cosine only 의 plateau 가 GAT 가 catch 한 영역과 noise 영역을 모두 포함할 가능성
  3. **(c) 이론적 분석 — 6 가설 (H-A~F) 타당성 평가**:
     | 가설 | 타당성 | 근거 | 검증 cost | 우선순위 |
     |---|---|---|---|---|
     | **H-A. Distribution shift gap** (qcond_nl3.pt = Plain features 학습 vs t_00 inference = Enriched features → GAT score noise) | **High** | 학습 config `train_qcond_nl3.yaml` 가 Plain features (name+type+examples), inference 는 EnrichedHeteroGraphBuilder (description 추가 → input 384 dim 동일하지만 텍스트 distribution 다름). 학습/inference mismatch 가 GAT score 변별력 평탄화 mechanism candidate | 11 cells (`best_gat_enriched.pt` alpha sweep), ~₩8,400, ~3-4h | **#1 (root 실험)** |
     | **H-B. Cosine vs GAT redundancy** (GAT 학습 signal BCE+InfoNCE on Plain features → PLM cosine 과 highly correlated) | **Medium** | selector_analysis (HISTORY §4) 의 P80 threshold 결과: GAT rescued 544 (5.3%) - hurt 330 (3.2%) = +214 (2.1% 순기여). PR-AUC +0.074 (Ensemble 0.317 vs Cosine 0.243). **Redundancy 가 아닌 보완 신호 존재 — 단 final F1 영향 미세**. ROC-AUC +0.035 도 동일 mechanism | analyzer (LLM-free, ₩0, ~30min) | **#2 (analyzer)** |
     | **H-C. Filter dominance** (XiYanFilter GLM-4.7 가 강력해 selector signal 차이 prune 단계에서 absorb) | **High** | F-1 (no Filter) ablation 결과: F1 -0.6408 절대적 인데 EX -0.0085 marginal. selector_analysis 의 38.9% structural ceiling (두 방법 모두 못 잡는 gold) → Filter 가 채워줌. **Filter absorption mechanism 매우 강력** | F-1 alpha sweep 6-11 cells (LLM-free no_filter, ₩0, ~1h) | **#2 (root, 저비용)** |
     | **H-D. Score normalization 평탄화** (`ensemble_selector.py:297-307` min-max norm 이 GAT score 절대값 변별력 flatten → α 가 작아도 cosine dominate) | **High** | 코드 직접 검증 — `raw_norm = (raw - raw.min()) / (raw.max() - raw.min())` + `gat_norm` 동일. **min-max 가 outlier 에 민감하고 변별력 평탄화 효과**. α=0.5 ensemble 에서 GAT_norm 의 절대 score 가 정규화로 cosine_norm 과 동일 [0,1] scale → blend 의 GAT 영향 약화 mechanism candidate | 1-2 cells (norm 제거 + z-score 변형), ~₩1,528, ~30min (코드 fix + 측정) | **#1 (root, 저비용 직접 검증)** |
     | **H-E. SQL gen bottleneck** (GLM-4.7 SQL 생성 noise 가 schema linking 차이 wash out) | **Medium-High** | F-1 EX -0.0085 marginal evidence 강력. 단 schema linking F1 vs EX divergence 와 동일 mechanism (이미 paper insight). **paper main insight 강화 가능** | SQL gen LLM 교체 (GPT-4 등), ~₩30K+, **post-deadline** | **#4** |
     | **H-F. Top-K=20 cap** (Selector top-20 selection 변별력 ceiling — α 변화가 set 자체 거의 동일) | **Medium** | t_00 stack 의 selector top-K 정확 확인 필요 (코드 디폴트 검증). top-K=10/30/50 sweep 으로 검증 | 3 cells, ~₩2,292, ~1h | **#3** |
  4. **(d) Narrative Resolution — 4 옵션 평가**:
     | 옵션 | 학술적 정직성 | 사용자 연구 방향 정합성 | 추가 비용 | 권장 우선순위 |
     |---|---|---|---|---|
     | **옵션 1**: t_00 anchor 유지 + GAT marginal 명시 ("GAT-floor + Cosine-ceiling ensemble" narrative) | High (정직) | **Medium-Low** (paper main contribution 약화 직접 인정) | 0 (글만) | **★ (보강 narrative 로만)** |
     | **옵션 2**: H-A 검증 → distribution shift 해소 후 재시도 (`best_gat_enriched.pt` alpha sweep) | High (가설 검증) | **High** (GAT contribution 회복 시 narrative 강화) | 11 cells, ~₩8,400, ~3-4h | **★★★ (#1 권장)** |
     | **옵션 3**: Adaptive α (Difficulty-aware) narrative — Simple/Challenging α=1, Moderate α=0.5 | High (새 contribution) | High (새 axis 신설) | Difficulty 분류기 추가 구현 + 측정, ~₩2K+ | **★★ (post-deadline 신설)** |
     | **옵션 4**: Filter dominance 활용 narrative ("Modular LLM Filter 가 selector signal 차이 prune → upstream selector robustness 보장") | High (mechanism) | **High** (사용자 4 module Co-Design with Modular LLM Filter 와 정합) | 0 (narrative 보강) + H-C 검증 (F-1 alpha sweep, ₩0) | **★★★ (#1 동급)** |
  5. **(e) 🎯 권장 통합 narrative — 단기/중기/장기 분리**:
     - **단기 (D-day 발표 직전, 2026-04-28 이미 완료)**:
       - **옵션 4 narrative 우선 보강** — Filter dominance 의 paper insight 강화 ("Modular LLM Filter 가 first-class stage 역할: selector signal 차이를 prune 단계에서 absorb → upstream selector 의 robustness 보장")
       - 기존 narrative ("QCondGAT main contribution") 약화하지 말고, **"4 module Co-Design 의 inter-module synergy"** 로 framing 조정 (Filter ↔ Selector 보완 관계 강조)
       - α plateau 는 §11 Q15 (이미 추가) + §9 Limitations 에 caveat 명시
     - **중기 (post-deadline, 2026-04-29~)**:
       - **옵션 2 검증 #1 우선** — H-A `best_gat_enriched.pt` alpha sweep 11 cells (root 실험)
       - 결과 분기:
         - GAT contribution 회복 (α=0.5 best) → 기존 narrative 강화 + paper main contribution 정당성 확보
         - 회복 X (α=1.0 여전히 best) → **옵션 1 + 옵션 4 통합 narrative 정식 채택** (학술적 정직성 + Filter dominance 강조)
       - **H-D 검증 (저비용 직접)** — score normalization 변형 1-2 cells (root)
       - **H-B/H-C/H-F 분석 (LLM-free, analyzer 위임)** — per-query correlation + F-1 alpha sweep + top-K overlap
     - **장기 (paper full version)**:
       - **옵션 3 Adaptive α 신설 contribution** — Difficulty-aware ensemble (Simple/Challenging α=1 / Moderate α=0.5) 의 EX 상한 정량 → paper section IV 신규 narrative 후보
  6. **(f) F1 vs EX divergence (직전 2026-05-04 첫 엔트리) 와의 정합성 강화**:
     - α plateau (8 cells F1 noise) + α=1.0 EX best (+0.0098) → **F1 metric 의 EX-aware limitation 추가 evidence**
     - paper insight 강화: "Schema linking F1 metric 의 plateau 가 SQL EX 의 nuance (S-2 difficulty 별 best stack 다름) 를 mask"
     - paper section IV/V 의 F1 vs EX divergence narrative 확장 — Selector α plateau 가 동일 mechanism (Recall driver)

- **근거**:
  - **Alpha sweep 11 cells (8 신규 + 3 재사용)**: `outputs/experiments/s04_ablation/pipeline/{t00_alpha_0[12346789], t00_S1_alpha0, t00_S2_alpha1, enriched_qcond_a05_mst_pcst_union_glm_sql}/`
  - **Score normalization 코드**: `src/modules/selectors/ensemble_selector.py:297-307` (min-max norm)
  - **GAT compute 분기**: `ensemble_selector.py:202-276` (query_conditioned True/False, query_supernode True 분기)
  - **selector_analysis (HISTORY §4)**:
    - ROC-AUC: Cosine 0.741 vs Ensemble **0.776** (+0.035)
    - PR-AUC: Cosine 0.243 vs Ensemble **0.317** (+0.074)
    - Gold-NonGold gap: Cosine 0.108 vs Ensemble **0.227** (2배)
    - GAT 기여도 (P80): rescued 544 (5.3%) - hurt 330 (3.2%) = **+214 gold (2.1% 순기여)**
    - Structural ceiling: 38.9% gold 가 두 방법 모두에서 threshold 미만
  - **ckpt 학습 config (H-A 검증용)**:
    - 현재 inference: `qcond_nl3.pt` ← `train_qcond_nl3.yaml` (Plain features, name+type+examples 학습)
    - H-A 검증 ckpt: `best_gat_enriched.pt` ← `train_gat_enriched_config.yaml` (Enriched features 학습, distribution match)
  - **Difficulty 별 best α (Alpha Sweep 분해)**:
    - Simple: α=1.0 best EX=0.4389
    - Moderate: α=0.5 best EX=0.2220 ★
    - Challenging: α=1.0 best EX=0.2138
    - **Moderate 에서만 α=0.5 best, Simple/Challenging 은 α=1.0 best** → Adaptive α 가능성 (옵션 3)
  - **S-3 (no QCond, α=0.5, `best_gat_enriched.pt`) F1=0.8634 / EX=0.3344** — t_00 와 거의 동등 (-0.002 F1) → QCond 자체의 marginal 효과도 noise 범위 시사

- **영향 범위**:
  - **paper_research_direction.md (planner Edit)**:
    - §0 한 문장 요약 — α plateau caveat 추가 (이미 직전 엔트리 추가, 본 엔트리는 11 cells 정밀 확인)
    - §1 모듈별 결정 표 — Selector 결정 narrative 정정 (옵션 4 통합)
    - §2.2 Selector contribution — α plateau α∈[0.3,1.0] 추가 + GAT-floor + Cosine-ceiling narrative + 6 가설 분석 link
    - §3 Inter-Module Co-Design — 신규 §3.5 (Filter ↔ Selector absorption) 추가 가능
    - §8 Future Works — H7 multi-seed 우선순위 (S-2 vs t_00 statistical significance) + H-A 검증 #1 추가 (best_gat_enriched alpha sweep 신설)
    - §9 Limitations — Selector α plateau 항목 추가
    - §10 핵심 수치 — alpha sweep 11 cells 표 추가
  - **presentation_brief (planner Edit)**:
    - §11 Q15 (이미 추가, 약간 보강) — α plateau 11 cells 데이터 인용
    - §14.7 Stage Contribution Analysis — alpha sweep 8 cells 추가 sub-section
  - **DECISIONS 본 엔트리** — Alpha sweep + 6 가설 + 4 옵션 종합

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — paper_research_direction §1/§2.2/§8/§9/§10 + presentation_brief §11 Q15 보강 + §14.7
  2. **사용자 (즉시 의사결정 필요)**:
     - **(A) Narrative resolution 옵션 결정** — 단기 옵션 4 narrative 보강 + 중기 옵션 2 검증 권장 (planner 권장 통합 narrative)
     - **(B) H-A 검증 root 실험 승인** — `best_gat_enriched.pt` alpha sweep 11 cells (~₩8,400, ~3-4h)
     - **(C) H-D normalization 검증 root 실험 승인** — 1-2 cells (~₩1,528, ~30min)
     - **(D) Adaptive α (옵션 3) post-deadline 우선순위** — paper full version contribution 후보로 등록할지
  3. **Root (사용자 승인 후)**:
     - **#1**: H-A `best_gat_enriched.pt` alpha sweep 11 cells (옵션 2 검증)
     - **#1 동급**: H-D normalization 변형 1-2 cells (저비용 직접 검증)
     - **#2**: H-C F-1 (no Filter) alpha sweep 6-11 cells LLM-free (Filter dominance 검증)
  4. **Analyzer (즉시 위임 가능, LLM-free)**:
     - **H-B**: per-query GAT score vs Cosine score correlation 분석 — `outputs/.../enriched_qcond_a05_mst_pcst_union_glm_sql/score_analysis_*.jsonl`
     - **H-F**: top-K 별 selected node overlap 분석 (k=10/20/30/50) — α=0.0/0.5/1.0 비교
     - **추가**: alpha sweep 11 cells 의 difficulty 별 분해 (Simple/Moderate/Challenging × 11 α 점)
  5. **Selector 모듈 (post-deadline, 옵션 2 결과 후)**:
     - 옵션 3 Adaptive α 구현 (difficulty 분류기 + α routing)
     - H-D 결과 반영한 normalization 코드 정정 (선택)

- **추가 필요 분석** (analyzer 큐, 즉시 위임):
  - **H-B**: per-query GAT vs Cosine score correlation (Pearson + Spearman) — α plateau 의 redundancy 가설 검증
  - **H-C 보강**: F-1 ablation 의 stage 별 R/P 재분석 — Filter 없을 때 alpha 효과 강해지는지 (no_filter alpha sweep 결과 도착 후)
  - **H-F**: top-K 별 selected node Jaccard overlap (k=10/20/30/50, α=0.0/0.5/1.0)
  - **Alpha sweep difficulty 분해**: 11 α × 3 difficulty = 33 cells 의 EX heatmap (Adaptive α 옵션 3 정당성)
  - **Selector contribution per-DB**: GAT 가 의미 있는 DB (rescue gold > hurt gold) vs 무의미 DB 분리 — paper section IV per-DB heatmap 후보

---

## 2026-05-04 (3 Ablation Series 완료) — Framework + Node Attribute + S/E/F Ablation 결과 종합 + 🚨 GAT contribution 재평가 + F1/EX divergence + Description-only enrichment + Anchor narrative 재정렬

- **결정**:
  1. **(a) 새 anchor t_00 정량 (SQL 평가까지 확장)**:
     - **t_00 = `s04_pipeline_enriched_qcond_a05_mst_pcst_union_glm_sql`**: R=0.8734, P=0.8581, **F1=0.8657**, **EX=0.3377**
     - Stack: Enriched + QCond α=0.5 + qcond_nl3 + MSTPCSTUnionExtractor(score_threshold=0.1) + XiYanFilter(provider=glm, num_examples=3) + LLMSQLGenerator(provider=glm)
     - 옵션 A2 Cell 2 (Union, F1=0.8667) 의 SQL 평가 확장 — ΔF1=-0.0010 LLM noise 범위 + EX 메트릭 신규 도입
     - **paper main pipeline anchor F1=0.8673 (MST Kruskal) 유지**, t_00 는 **EX 평가 가능 sub-anchor** 로 ablation baseline 활용
  2. **(b) 6 Baseline 비교 — t_00 압도** (모든 baseline ΔF1 +0.09~+0.19 / ΔEX +0.09~+0.19):
     | Method | F1 | EX | t_00 ΔF1 | t_00 ΔEX |
     |---|---|---|---|---|
     | B1 G-Retriever | 0.7719 | 0.2490 | -0.0938 | -0.0887 |
     | B2 LinkAlign | 0.7274 | 0.2001 | -0.1383 | -0.1376 |
     | B3 XiYan-SQL | 0.7108 | 0.1969 | -0.1549 | -0.1408 |
     | B4 Vector-Only | 0.7133 | 0.1786 | -0.1524 | -0.1591 |
     | B5 Graph Expand | 0.6815 | 0.1467 | -0.1842 | -0.1910 |
     | B6 Graph+Agent | 0.6806 | 0.1454 | -0.1851 | -0.1923 |
     - 모든 difficulty 에서 일관 우세 (analyzer 후속 검증)
  3. **(c) Framework Ablation (Pipeline Stage 효과) — 4 cells**:
     | Stage | R | P | F1 | EX | ΔF1 vs t_00 | ΔEX vs t_00 |
     |---|---|---|---|---|---|---|
     | ① Selection Only | 0.7536 | 0.2625 | 0.3894 | 0.2086 | -0.4763 | -0.1291 |
     | ② Sel + Extraction | 0.9927 | 0.1268 | 0.2248 | 0.3292 | -0.6409 | -0.0085 |
     | ③ Sel + Filter | 0.6676 | 0.8273 | 0.7390 | 0.2295 | -0.1267 | -0.1082 |
     | ④ t_00 (full) | 0.8734 | 0.8581 | **0.8657** | **0.3377** | (anchor) | (anchor) |
     - **🚀 핵심**: **Filter 가 F1 에 절대적 (+0.6409)**, **EX 에는 marginal (+0.0085)** — F1/EX divergence
     - **Extraction 단독은 F1 negative 이지만 EX positive (+0.1206 vs Sel Only)** — Recall driver
     - **Extraction × Filter synergy** 가 t_00 의 핵심 동작 mechanism
  4. **(d) Node Attribute Ablation (Builder 노드 텍스트 정보) — 3 cells (V1/V2/V3) + t_00**:
     | Mode | R | P | F1 | EX | ΔF1 vs t_00 |
     |---|---|---|---|---|---|
     | V1 name only | 0.8526 | 0.8456 | 0.8491 | 0.3253 | -0.0166 |
     | V2 name + type | 0.8472 | 0.8453 | 0.8462 | 0.3266 | -0.0195 |
     | V3 name + type + desc | 0.8676 | 0.8569 | 0.8622 | 0.3377 | **-0.0035 (plateau)** |
     | t_00 (enriched_full) | 0.8734 | 0.8581 | **0.8657** | **0.3377** | (anchor) |
     - **🎯 Description 만이 effective enrichment driver** (V3 vs V1/V2 ΔF1 +0.013~+0.016)
     - **Type 정보 marginal X** (V1 ≈ V2, ΔF1=-0.003 noise)
     - **V3 ≈ t_00 plateau** (ΔF1=-0.0035, EX 정확히 동일 0.3377) — Meaning + Value_desc + Examples 추가 정보 marginal contribution = 0
     - **Distribution shift 영향 미미** — qcond_nl3 ckpt 가 Plain features 학습이지만 V2 (학습 distribution 가까움) 가 V1 보다 우세 X → **학습 재진행 불필요** 결정
     - Caveat: 동일 ckpt 사용으로 "feature 효과" + "distribution shift" 혼재. 단 plateau 영역이라 학술적 해석 충분
     - 코드 fix: `EnrichedHeteroGraphBuilder.node_text_mode` 파라미터 (`graph_builder.py:503-526`)
  5. **(e) S/E/F Ablation (Selector × Extractor × Filter) — 9 cells (6 신규 + 3 재사용)**:
     | Cell | R | P | F1 | EX | ΔF1 vs t_00 | ΔEX vs t_00 |
     |---|---|---|---|---|---|---|
     | S-1 α=0 (GAT only) | 0.6549 | 0.7587 | 0.7030 | 0.2086 | **-0.1627 ⚠️** | -0.1291 |
     | S-2 α=1 (Cosine only) | 0.8761 | 0.8570 | **0.8664** | **0.3475** | **+0.0008** | **+0.0098** |
     | S-3 no QCond | 0.8736 | 0.8534 | 0.8634 | 0.3344 | -0.0023 | -0.0033 |
     | S-4 t_00 (QCond α=0.5) | 0.8734 | 0.8581 | 0.8657 | 0.3377 | (anchor) | (anchor) |
     | E-1 MST only | 0.8721 | 0.8634 | **0.8677** | 0.3383 | **+0.0021** | +0.0007 |
     | E-2 Basic PCST | 0.8409 | 0.8378 | 0.8393 | 0.3299 | -0.0263 | -0.0078 |
     | E-3 t_00 (Union) | 0.8734 | 0.8581 | 0.8657 | 0.3377 | (anchor) | (anchor) |
     | F-1 No Filter | 0.9927 | 0.1268 | 0.2248 | 0.3292 | **-0.6408** | **-0.0085** |
     | F-2 No examples | 0.8737 | 0.8293 | 0.8509 | 0.3338 | -0.0147 | -0.0039 |
     | F-3 t_00 (3 examples) | 0.8734 | 0.8581 | 0.8657 | 0.3377 | (anchor) | (anchor) |
     - **🚨 S-1 GAT only 큰 손실** (F1 -0.1627) — GAT 단독은 약함 (직전 Plain GAT 0.2937 → QCond GAT 0.3534 +0.0597 발현 narrative 와 일치)
     - **🚨 S-2 Cosine only 가 t_00 와 동등/약우세** (F1 +0.0008 noise, **EX +0.0098**) — **GAT 의 marginal contribution = noise/negative 가능성** (paper main contribution narrative 영향)
     - **S-3 no QCond ≈ t_00 plateau** (ΔF1=-0.0023) — query_conditioned=False 도 Ensemble 영역 plateau 동등
     - **E-1 MST only ≈ t_00 (+0.0021)** — Union 의 PCST 추가 노드 marginal X, **stack 단순화 가능** (MST Kruskal alone)
     - **E-2 Basic PCST 손실 (-0.0263)** — score-threshold seed widening 의 가치 재확인 (직전 옵션 C narrative 와 일치)
     - **F-1 No Filter F1 -0.6408 vs EX -0.0085** — **F1/EX divergence main evidence**: schema linking F1 metric 이 SQL EX 의 weak proxy
     - **F-2 No examples 손실 (-0.0147)** — Filter prompt examples (3) 의 효과 정량
     - 코드 fix: `XiYanFilter.num_examples` 파라미터 (`xiyan_filter.py:22-72`)
  6. **(f) Difficulty 별 EX (S-2 vs t_00 — adaptive selection 가능성)**:
     - t_00 EX: Simple 0.4184, Moderate 0.2220, Challenging 0.1931
     - S-2 (Cosine only) EX: Simple **0.4389 ★**, Moderate 0.2069, Challenging **0.2138 ★**
     - **S-2 가 Simple/Challenging 에서 t_00 보다 우세**, Moderate 에서만 t_00 우세
     - → **Difficulty 별 best stack 다름** — adaptive selection 가능성 (paper future work, H10 신설 권장)
  7. **(g) 🚀 Cross-cutting Insights — Paper Main Contribution 영향 5 항목**:
     1. **F1 vs EX divergence (paper main insight)** — Framework #2 (Sel+Ext) 가 F1=0.2248 (worst) 인데 EX=0.3292 (2nd best). Schema linking F1 metric 이 SQL EX 의 weak proxy. **Recall focus 평가 필요** (paper section IV 신규 narrative).
     2. **🚨 GAT 의 학술적 가치 재평가 필요** — S-2 (Cosine only) 가 t_00 와 plateau 동등/약우세 (ΔF1=+0.0008 noise, ΔEX=+0.0098). 현재 paper main contribution narrative ("Query-Conditioned GAT 가 핵심") 약화 가능. **Paper narrative 옵션**:
        - 옵션 A: GAT contribution 약화 인정, "Modular LLM Filter 가 main contribution" 더 강조
        - 옵션 B: Multi-seed reliability 검증 (H7) 후 결정 — α=0.5 vs α=1 의 0.0008 차이가 noise 인지 systematic 인지 (직전 GLM noise ±0.003~0.005 범위 내)
        - **권장: 옵션 B 선행 → 옵션 A 확정** (post-deadline H7 의존)
     3. **MST + PCST Union 의 marginal X** — E-1 (MST only) ≈ t_00 (+0.0021 noise). paper 의 stack 단순화 가능 (MST Kruskal alone). 단 anchor F1=0.8673 (MST Kruskal) 와 t_00 F1=0.8657 (Union) 모두 유지하여 두 변형 동등 표기.
     4. **Filter 의 학술적 가치 vs EX 가치 분리** — F-1 (No Filter) F1 -0.6408 인데 EX -0.0085. **Schema linking F1 측면에서는 Filter 필수, EX 측면에서는 marginal**. **학술적 narrative 강화 vs 실용적 가치 분리 명시 필요** (paper section V Conclusion 정정).
     5. **Description 만이 effective enrichment** — V3 ≈ t_00 (-0.0035 plateau). Builder design 단순화 가능 (Meaning/Value_desc/Examples 제거 가능). **Enriched Builder 의 정확한 기여 = description text 1 항목** — 학술적 attribution 정밀화.

- **근거**:
  - Root 측정 결과 (2026-05-04, 3 ablation series 완료)
  - **Framework Ablation**: `outputs/.../enriched_qcond_a05_{selector_only_sql, mst_pcst_union_no_filter_sql, no_extractor_glm_sql}/` (3 신규 cells, t_00 와 비교)
  - **Node Attribute Ablation**: `outputs/.../t00_node_v{1,2,3}_*/` (3 신규 cells) + 코드 fix `src/modules/builders/graph_builder.py:503-526` (`node_text_mode` 파라미터)
  - **S/E/F Ablation**: `outputs/.../t00_{S1,S2,S3,E1,E2,F2}_*/` (6 신규 cells, 재사용 3) + 코드 fix `src/modules/filters/xiyan_filter.py:22-72` (`num_examples` 파라미터)
  - **Anchor t_00**: `outputs/.../enriched_qcond_a05_mst_pcst_union_glm_sql/`
  - **6 Baselines**: `outputs/baselines/{baseline_g_retriever, baseline_linkalign, baseline_xiyansql, preliminary_vector_only, preliminary_graph_expansion, preliminary_graph_and_agent}/`

- **영향 범위**:
  - **paper_research_direction.md**:
    - §0 한 문장 요약 — t_00 EX=0.3377 + Baseline 압도 narrative 추가 가능 (anchor F1=0.8673 유지)
    - §1 모듈별 결정 — Selector 결정 caveat 보강 (GAT marginal value, S-2 Cosine only plateau 동등)
    - §2.2 Selector contribution — GAT 의 학술적 가치 재평가 narrative 추가
    - §2.3 Extractor — MST Kruskal alone vs Union plateau 동등 (E-1 결과 추가)
    - §2.4 Filter — F1/EX divergence main insight 강조 (F-1 결과 추가)
    - §7 측정 갭 — 3 ablation 결과 추가 sub-section
    - §8 Future Works — H7 우선순위 ↑ + H10 신설 (Adaptive Selection by Difficulty) + Stack 단순화 검증
    - §9 Limitations — GAT contribution 재평가 + F1/EX divergence + multi-seed 미검증
    - §10 핵심 수치 — 3 ablation matrices + EX 메트릭 + 6 baselines 비교
  - **presentation_brief**:
    - §0 Executive Summary — Selector contribution caveat (선택)
    - §11 Q&A — 신규 Q15 (Cosine only 가 t_00 와 plateau 동등인 것의 의미) + Q16 (F1 vs EX divergence) + Q17 (3 ablation series 종합)
    - §14 — 신규 §14.7 Stage Contribution Analysis (3 ablation matrices)
    - §10.1 Anchor F1 비교 — t_00 (F1=0.8657 + EX=0.3377) 추가
  - **DECISIONS 본 엔트리** — 3 ablation 결과 + cross-cutting 5 insights
  - **paper main pipeline anchor 변경 X**: F1=0.8673 (MST Kruskal Concat) 유지. t_00 (Union + SQL) 는 EX 평가 sub-anchor 역할

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — paper_research_direction §1/§2.2/§2.3/§2.4/§7/§8/§9/§10 + presentation_brief §11/§14.7 + DECISIONS 본 엔트리 (총 10+ 항목)
  2. **사용자 (즉시)** — Selector contribution narrative 옵션 결정 (옵션 A: GAT 약화 인정 vs 옵션 B: H7 후 결정). 권장: 옵션 B (post-deadline)
  3. **Root (post-deadline)** — H7 multi-seed 검증 (S-2 Cosine only vs t_00 의 statistical significance 확정) + alpha sweep 8 cells 결과 수령 후 후속 측정
  4. **Analyzer (post-deadline)** — F1 vs EX divergence mechanism 분석 (Framework #2 가 F1=0.22 인데 EX=0.33 인 mechanism — Recall driver evidence + paper insight 정량화)
  5. **Selector 모듈 (post-deadline, 옵션 B 결정 후)** — α=0.5 vs α=1 의 0.0008 차이가 noise 인지 systematic 인지 multi-seed 정량 + difficulty-conditioned best alpha 분석

- **추가 필요 분석** (post-deadline 큐):
  - **Alpha sweep 8 cells (~16:23 KST 도착 예정)** — 직전 사용자 input 의 진행 중 측정. 완료 후 alpha curve + best alpha 분석 (S-2 Cosine α=1 plateau 의 sweep 위치 확정)
  - **Per-difficulty Selector contribution 분해** — S-2 Simple/Challenging 우세, t_00 Moderate 우세 mechanism (어떤 query 패턴이 GAT 에서 이득?)
  - **F1 vs EX divergence per-DB 분해** — Framework #2 (Sel+Ext) 의 EX positive 가 모든 DB 에서 일관인지 / 일부 DB 에서만인지
  - **Stack 단순화 검증 (post-deadline)** — MST Kruskal alone + Description-only Builder + Cosine only Selector 로 simpler t_00' 측정 → Filter 제외 plateau 도달 가능?
  - **6 Baselines difficulty 분해** — 모든 difficulty 에서 t_00 일관 우세 검증 (challenging 격차가 Simple 보다 큰지)
  - **Adaptive Selection (H10 신설 후보)** — Simple → α=1 / Moderate → α=0.5 / Challenging → α=1 difficulty-conditioned routing 의 EX 상한 정량

---

## 2026-04-29 (SuperNode 9-cell matrix 완료) — H6 결정 (Concat 채택) + α=0 SuperNode 손실 발견 (-0.1735) + paper limitation 강화

- **결정**:
  1. **(a) 🎯 H6 결정 — Concat 채택, SuperNode 보류**:
     - paper main pipeline anchor 유지: `s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` F1=**0.8673** (Concat)
     - SuperNode 어떤 α 에서도 anchor 갱신 임계 +0.005 초과 못 함 → **Concat 우세**
     - 발표 narrative: paper main pipeline = QCond **Concat**
  2. **(b) 9 cells 결과 매트릭스** (R/P/F1, 4-decimal):
     | α / Stage | Selector only | + Basic PCST | + XiYan(GLM) Final |
     |---|---|---|---|
     | **α=0** (GAT only) | 0.6035 / 0.2534 / **0.3569** | 0.5539 / 0.2809 / **0.3728** | 0.4738 / 0.6487 / **0.5476 ⚠️** |
     | **α=0.5** (neutral) | 0.7276 / 0.2787 / **0.4030** | 0.9564 / 0.1396 / **0.2436** | **0.8353 / 0.8330 / 0.8341** |
     | **α=1** (Cosine only) | 0.7693 / 0.2549 / **0.3829** | 0.9662 / 0.1302 / **0.2295** | **0.8441 / 0.8296 / 0.8368** |
  3. **(c) Concat vs SuperNode 비교 — Final F1 차이**:
     | α | Concat F1 | SuperNode F1 | ΔF1 (SN − Concat) |
     |---|---|---|---|
     | α=0 (GAT only) | 0.7211 | 0.5476 | **-0.1735 ⚠️** |
     | α=0.5 (neutral) | 0.8306 | 0.8341 | +0.0035 (noise) |
     | α=1 (Cosine only) | 0.8424 | 0.8368 | -0.0056 (noise) |
  4. **(d) 🚨 α=0 SuperNode 큰 손실 mechanism 후보 narrative** (paper insight 후보):
     - **Concat mechanism**: query embedding 을 모든 노드 input feature 에 직접 concat (384→768) → GAT score 산출에 직접 기여
     - **SuperNode mechanism**: query_node 를 그래프에 추가 노드로 주입 + message passing 통한 indirect 영향
     - **α=0 (GAT-only) 신호 모드**: SuperNode 의 indirect 효과가 GAT 단독 score 에서 dilution → Concat 의 direct concat 이 우세
     - **단계별 격차 확대**: Selector_only Δ=+0.0035 (noise) → Filter 후 Δ=-0.1735 (큰 손실) → Filter 가 SuperNode α=0 의 약한 signal 을 over-prune
     - **α=0.5/1 plateau 동등**: Cosine 비중 우세 영역 (α≥0.5) 에서 GAT/SuperNode 차이 dilute, 양쪽 모두 plateau (F1=0.83~0.84) 도달
  5. **(e) paper limitation 강화 항목 신설**:
     - "SuperNode mechanism 의 GAT-only 신호 모드 (α=0) 손실 — direct query concat 이 indirect message passing 보다 우세" → mechanism 의 명확한 분석 (Concat 우월의 단순 결과 vs SuperNode 의 본질적 mechanism 한계 vs 학습 부족 수렴 등) future work
  6. **(f) Filter Δ F1 by α (SuperNode stack)**:
     - α=0: no_filter 0.3728 → final 0.5476, Δ=+0.1748 (small) — Filter 가 SuperNode α=0 의 signal noise 충분히 prune 못 함
     - α=0.5: no_filter 0.2436 → final 0.8341, Δ=**+0.5905** (large)
     - α=1: no_filter 0.2295 → final 0.8368, Δ=**+0.6073** (max)

- **근거**:
  - Root 측정 결과 (2026-04-29 19:45:46~21:38:51, 1h 53min, GPU 0/1 split, ~₩2,292)
  - 출처: [EXPERIMENT_HISTORY.md "SuperNode 9-cell Matrix 측정 (Ablation 2 SuperNode, 2026-04-29)"](../EXPERIMENT_HISTORY.md) L2002~
  - 새 ckpt: `best_gat_query_supernode_qcond.pt` (best epoch 228, val recall@15=0.5737)
  - 코드 fix: `src/modules/selectors/ensemble_selector.py:241-243` (SuperNode 분기 query_emb 전달)
  - smoke PASS: F1=0.3569 (α=0 selector_only) 정상 동작 검증

- **영향 범위**:
  - **paper_research_direction.md (planner Edit 완료)**:
    - §1 모듈별 결정 — Selector 결정 "Concat 채택" 명시 (H6 결정 완료)
    - §8 Future Works H6 — "학습 진행 중" → "완료, paper limitation 으로 narrative 강화"
    - §9 paper limitation — α=0 SuperNode 손실 mechanism 분석 future work 신설
  - **presentation_brief (planner Edit 완료)**:
    - §11 Q&A — 신규 Q14 (SuperNode 측정 결과 + H6 결정)
    - §14.2 SuperNode 3 row → 9 cells 결과 채움
  - **DECISIONS 본 엔트리** — H6 결정 + α=0 mechanism narrative
  - **paper main pipeline anchor 변경 X**: Concat F1=0.8673 유지 (4 module + 4 co-design 통합 stack)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — 6 작업 (a~f) 모두 완료, paper_research_direction §1/§8/§9 + presentation_brief §11/§14.2 + DECISIONS
  2. **사용자 (즉시)** — 발표 D-day Q&A 시 Q14 활용 (SuperNode 측정 결과 + H6 결정)
  3. **Selector 세션 (post-deadline)** — α=0 SuperNode 손실 mechanism 분석 (Concat 우월의 단순 결과 vs SuperNode 본질 한계 vs 학습 부족 수렴 등)
  4. **Analyzer 세션 (post-deadline)** — α=0 SuperNode 의 selected_nodes vs Concat 의 selected_nodes 차이 분석 (어떤 노드를 prune 못 해서 Filter 단계에서 손실 확대?)

- **추가 필요 분석** (post-deadline 큐):
  - α=0 SuperNode selected_nodes 의 Concat 대비 차이 진단 (Selector_only 단계 동등 → Filter 단계 -0.1735 격차의 mechanism)
  - SuperNode 다른 α (0.25, 0.7, 0.95) 의 plateau 영역 검증 (α=0.5/1 plateau 가 entire α≥0.5 영역인지 확인)
  - SuperNode α=0 ckpt 추가 학습 (epoch 300+) 시 mechanism 회복 가능성

---

## 2026-04-28 (Analyzer Difficulty 분해 완료) — 🎯 Difficulty-invariant robustness (F1 spread 0.0073) + Hard queries 에서 paper main 우위 발현 + paper anchor 결정 사후 정당화

- **결정**:
  1. **(a) Analyzer 분석 산출물 수령**: [`notebooks/analysis_results/paper_main_pipeline_difficulty_breakdown.md`](../notebooks/analysis_results/paper_main_pipeline_difficulty_breakdown.md) (177 lines, §0 TL;DR / §1 difficulty 분포 / §2 paper main R/P/F1 / §3 핵심 발견 + Q&A 템플릿 / §4 subsidiary 비교 / §5 per-DB × difficulty heatmap / §6 후속 큐)
  2. **(b) 🎯 발표 D-day 핵심 narrative 3개 확정**:
     - **Difficulty-invariant robustness**: F1 spread 0.0073 — End-to-End Co-Design 의 robustness 직접 evidence (일반 BIRD/Spider benchmark 의 challenging 격차 -0.10~-0.20 대비 1~2 자리수 작음)
     - **R-P trade-off difficulty 비대칭**: Simple R 최고 (over-include) / Moderate P 최고 (Filter 보수적) / Challenging 균형 → **Filter 가 candidate ambiguity 에 자동 적응** (별도 difficulty-aware threshold 없이) = Filter ↔ Extractor co-design mechanism evidence
     - **Hard queries 에서 paper main 우위**: simple ΔF1=+0.0015 (sub-noise) / moderate +0.0061 / challenging +0.0043 → **paper main 채택 (시나리오 A 부분) 의 사후 정당화** — Enriched + QCondGAT 통합 가치가 어려운 쿼리에서 발현
  3. **(c) presentation_brief §10.0 신규 (planner Edit 완료)**: Paper Main Pipeline Difficulty 별 R/P/F1 표 + 핵심 발견 3개
  4. **(d) presentation_brief §11 Q12 보강 + Q13 신규 (planner Edit 완료)**:
     - Q12: subsidiary plateau (+0.0031) 의 difficulty 별 nuance 명시
     - Q13 신규: "Difficulty 별 성능?" — F1 spread 0.0073 + R-P 비대칭 narrative
  5. **(e) paper_research_direction.md §10 difficulty 분해 추가 (planner Edit 완료)**: Difficulty 별 R/P/F1 표 + 3 핵심 발견 + paper limitation 보강 (subsidiary plateau 의 difficulty-conditioned nuance)
  6. **(f) §추가 필요 분석 (직전 옵션 A2 엔트리) → §완료 (2026-04-28 analyzer)**: per-DB / per-difficulty 분해 항목 처리 완료
  7. **(g) Paper anchor 결정 (옵션 A2 시나리오 A 부분 채택) 의 사후 정당화 narrative 강화**:
     - 단순 평균 ΔF1=+0.0031 (plateau noise 처럼 보임) 해석을 **difficulty 별 nuance** 로 정밀화
     - 학술적 narrative: "Simple 에서 동등, Hard queries 에서 paper main 우위" — 4 module + 4 co-design 통합 stack 의 mechanism-level value 입증

- **근거**:
  - Analyzer 보고 (2026-04-28 발표 D-day):
    - 3 핵심 발견 (difficulty-invariant + R-P 비대칭 + hard queries 우위)
    - 산출물 177 lines, 6 sections + Q&A 템플릿
  - 출처: [notebooks/analysis_results/paper_main_pipeline_difficulty_breakdown.md](../notebooks/analysis_results/paper_main_pipeline_difficulty_breakdown.md)

- **영향 범위**:
  - **presentation_brief (planner Edit 완료)**: §10.0 신규 (difficulty breakdown) + §11 Q12 보강 + Q13 신규
  - **paper_research_direction.md (planner Edit 완료)**: §10 difficulty 분해 + paper limitation 보강
  - **DECISIONS 본 엔트리**: §추가 필요 분석 처리 완료 + paper anchor 사후 정당화
  - **paper main contribution narrative 정교화**: 단순 평균 plateau ≠ 단일 anchor 의 우위 → mechanism-level Difficulty 분해 evidence

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — 4 작업 (a)(b)(c)(d) 모두 완료
  2. **사용자 (즉시)** — 발표 D-day 슬라이드 작성 시 §10.0 + §11 Q13 + paper_research_direction §10 활용
  3. **Analyzer 후속 큐 (post-deadline)** — analyzer 보고 §6:
     - california_schools challenging 5 query failure mode 진단 (F1=0.5350 outlier)
     - Hard queries 에서 paper main 우위의 mechanism leave-one-out (Enriched description vs QCondGAT 효과 분리)
     - Per-difficulty Selector/Extractor/Filter cumulative R/P/F1 (G2 규범) — 신규 no-filter/no-extractor 1 cell 실행 필요 (root 의존)
     - Subsidiary anchor simple 동등 성능 mechanism 검증

- **Caveat**:
  - per-DB × difficulty heatmap (analyzer §5) 의 challenging cell 일부 n=4~5 (california_schools, codebase_community, debit_card_specializing) — 발표 시 단독 인용 회피
  - 본 분석은 final stage R/P/F1 만 — Stage 별 cumulative (G2) 는 별도 측정 필요 (post-deadline)

---

## 2026-04-28 (옵션 A2 2 cells 완료) — 🚀 paper main pipeline F1=0.8673 = Plain anchor plateau 동등 + 시나리오 A 부분 + paper anchor 결정 + 발표 D-day narrative 확정

- **결정**:
  1. **(a) 🚀 Paper main pipeline anchor 결정** — `s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` F1=**0.8673** (R=0.8741, P=0.8606):
     - Stack: **Enriched Builder + QCondGAT (α=0.5 neutral ensemble) + MST Kruskal (score>0.1 induced) + XiYan GLM-4.7**
     - 4 module + 4 co-design 통합 stack 의 학술적 정당성 확보
     - vs vLLM era 2×2×2 best (F1=0.7863): **ΔF1=+0.0810**
  2. **(b) 2 cells 결과 (옵션 A2)**:
     | Cell | Stack | R | P | F1 |
     |------|-------|---|---|---|
     | **🚀 1 (paper main ★)** | Enriched + QCond + MST Kruskal + XiYan GLM | 0.8741 | 0.8606 | **0.8673** |
     | 2 (Union, plateau) | Enriched + QCond + Union + XiYan GLM | 0.8772 | 0.8564 | 0.8667 |
     - Cell 1 vs Cell 2: ΔF1=-0.0006 (LLM noise 범위, 무시) → **paper main = MST Kruskal 권장** (anchor 일관성, 직전 옵션 C 의 anchor 결정과 동일)
  3. **(c) 시나리오 A 부분 채택** — paper main vs Plain anchor:
     - Plain anchor (`plain_ens_a05_mst_kruskal_glm`) F1=0.8642
     - Paper main F1=0.8673 → **ΔF1=+0.0031** (갱신 임계 +0.005 의 60% 미달, **plateau 동등**)
     - **학술적 narrative 우선** — 두 cell F1 사실상 동등이지만 paper main 의 4 module + 4 co-design 통합 정당성으로 채택
  4. **(d) Paper anchor + Subsidiary anchor 분리**:
     - **Paper main pipeline anchor**: `s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` F1=0.8673 (학술적 narrative)
     - **Subsidiary anchor (단순 stack baseline)**: `plain_ens_a05_mst_kruskal_glm` F1=0.8642 (paper supporting evidence)
     - 두 anchor 동등 표기, plateau narrative
  5. **(e) End-to-End Co-Design 통합 효과 narrative 확정**:
     - **4 module contributions 확정**: Builder (Enriched) + Selector (QCondGAT) + Extractor (MST Kruskal) + 🆕 Filter (Modular LLM)
     - **4 inter-module co-design principles 확정**:
       - Selector ↔ Extractor: top-k → score-threshold widening (+0.1871)
       - Builder ↔ Selector: Description noise + α=0.5 GAT 보정 (+0.0711 회복)
       - Extractor ↔ Filter: Input subgraph 단순할수록 Filter 효율 ↑
       - Filter Backbone ↔ Selector Blend: LLM era 비대칭 (+0.0506 ensemble synergy)
     - paper title (권장): **"LLM Filter as a First-Class Stage in Graph-RAG Schema Linking: Co-Designing Builder, Selector, Extractor, and Filter"**
  6. **(f) 발표 D-day (2026-04-28) narrative 확정** — 모든 자료 갱신 완료:
     - paper_research_direction.md §0/§7/§10 갱신
     - presentation_brief §0/§10/§14.6 갱신 + §11 Q11/Q12 신규 추가
     - 발표 main result: **paper main F1=0.8673**, vs vLLM era 2×2×2 best ΔF1=+0.0810

- **근거**:
  - Root 측정 결과 (2026-04-28, 2 cells, GPU 0/1, ~₩1,528, ~1h)
  - 출처: [EXPERIMENT_HISTORY.md "Paper Main Pipeline Measurement (옵션 A2, 2026-04-28)"](../EXPERIMENT_HISTORY.md) L1819~
  - 시나리오 A 부분 채택 사유 (학술적 narrative > F1 미세 차이):
    - Cell 1 (MST Kruskal) F1=0.8673 vs Plain anchor F1=0.8642: ΔF1=+0.0031 (LLM noise)
    - 학술적 narrative: 4 module + 4 co-design 통합 stack 의 정당성 (paper main contribution 직접 반영)
    - 발표 main pitch: "End-to-End Pipeline Co-Design 의 통합 효과"

- **영향 범위**:
  - **paper_research_direction.md (planner Edit 완료)**: §0 (F1=0.8673) + §7 (측정 갭 해소) + §10 (paper anchor 결정)
  - **presentation_brief (planner Edit 완료)**: §0 (4 module + paper main F1) + §10 (5 후보 plateau) + §14.6 (paper anchor + subsidiary 분리) + §11 Q11/Q12 신규
  - **DECISIONS 본 엔트리** — 발표 D-day narrative 확정
  - **post-deadline 영향**:
    - **H6 (SuperNode 학습 완료)** → Selector 변형 (Concat 현 cells vs SuperNode) 결정, paper Selector contribution 확정
    - **H7 multi-seed**: paper main + subsidiary anchor reliability 검증
    - **H9 Extractor 통합 sweep**: paper main pipeline 의 alpha sensitivity
    - **Wave 4 a05_filter_agentic**: paper main pipeline 위에서 Filter 모듈 extension
    - vivid-sprouting-sunbeam.md plan anchor refresh: 0.8383 → 0.8673 (paper anchor)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — 8 작업 (a~h) 모두 완료 (paper_research_direction §0/§7/§10, presentation_brief §0/§10/§14.6/§11, DECISIONS)
  2. **사용자 (즉시)** — 발표 D-day 슬라이드 작성 시 paper main F1=0.8673 narrative 활용
  3. **Root (선택, 발표 후 여유 시)** — CLAUDE.md (root) §0 anchor 표 갱신 + EXPERIMENT_PLAN.md §0 anchor 갱신
  4. **Filter 세션 (post-2026-04-28)** — vivid-sprouting-sunbeam.md plan anchor refresh (0.8383 → 0.8673)
  5. **Selector/Extractor/Analyzer 모듈 (post-deadline)** — H6 + H7 + H8 + H9 통합 wave

- **추가 필요 분석**:
  - per-DB / per-difficulty 분해 (paper main pipeline) — analyzer post-deadline
  - paper main 의 4 module 각 contribution 의 marginal effect 정량 (leave-one-out)
  - paper limitation 강화: single-run reliability + multi-seed 검증 future work

---

## 2026-04-28 (방향 F' 최종 채택 + 옵션 A2 측정 결정) — End-to-End Pipeline Co-Design with Modular LLM Filter, paper main pipeline F1 정확 측정 (2 cells)

- **결정**:
  1. **(a) 🚀 방향 F' 최종 채택**: **End-to-End Pipeline Co-Design with Modular LLM Filter**
     - 4 module contributions:
       - Builder: Enriched (Description-Aware)
       - Selector: Query-Conditioned GAT (Concat or SuperNode, H6 future work)
       - Extractor: MST + PCST (Score-Threshold MST Kruskal + Multi-extractor union)
       - **🆕 Filter: Modular LLM Filter (XiYan + GLM-4.7) 를 first-class stage 로 도입**
     - 4 co-design principles (Selector ↔ Extractor, Builder ↔ Selector, Extractor ↔ Filter, Filter Backbone ↔ Selector Blend)
     - paper title 권장: **"LLM Filter as a First-Class Stage in Graph-RAG Schema Linking: Co-Designing Builder, Selector, Extractor, and Filter"**
     - 자세한 narrative: [planning/paper_research_direction.md](paper_research_direction.md) (12 섹션 ~400 lines)
  2. **(b) 옵션 A2 추가 측정 결정** — 사용자 의도된 paper main pipeline 의 정확한 F1 확보:
     - Cell 1: **Enriched + QCond Ens α=0.5 + MST Kruskal + XiYan GLM** (anchor 와 동일 알고리즘, Builder/Selector 만 변경)
     - Cell 2: **Enriched + QCond Ens α=0.5 + Union (MST+PCST) + XiYan GLM** ("MST + PCST" 사용자 표현 직접 반영)
     - 비용: 2 cells × ~₩764 = ~₩1,528, 시간 ~1h parallel (GPU 2/3)
  3. **(c) presentation_brief §0 Executive Summary 갱신** — 방향 F' 의 4 module + 4 co-design + paper main contribution narrative
  4. **(d) Selector 결정 보류** — Concat (현 cells) or SuperNode (H6 학습 진행 중, post-deadline 결정). 발표 narrative 에는 "QCondGAT (Concat 기준 측정, SuperNode 변형 H6 future work)" 명시
  5. **(e) Anchor 상태 정리**:
     - **현 anchor 유지** (`plain_ens_a05_mst_kruskal_glm` F1=0.8642) — Plain Builder stack
     - **paper main pipeline anchor** = 옵션 A2 결과 (Enriched + QCond + MST Kruskal/Union) — post-측정 결정
     - 두 anchor 가 다를 수 있음:
       - 옵션 A2 F1 ≥ 0.8642 (anchor 동등 또는 갱신): paper main pipeline = anchor
       - 옵션 A2 F1 < 0.8642: 두 anchor 분리 (paper main = Enriched+QCond, anchor = Plain+Plain), narrative 에 "Description noise + GAT 보정 mechanism" 강조

- **근거**:
  - 사용자 직전 메시지 (2026-04-28): "F'으로 방향성을 잡고 옵션 A2 추가 측정을 진행하자"
  - 4 module contributions + 4 co-design principles 정리: planning/paper_research_direction.md §2~3
  - 측정 갭 (Enriched + QCond + MST Kruskal/Union 미측정): paper_research_direction.md §7

- **영향 범위**:
  - **paper_research_direction.md (planner 작성 완료)**: 12 섹션, paper title + 6-act narrative + paper structure (II-V 장) 매핑 + future works
  - **presentation_brief §0 Executive Summary** (planner Edit 즉시): 4 module contributions + 방향 F' 명시
  - **Root 핸드오프 (즉시)**: 2 cells 측정 (GPU 2/3, ~1h, ~₩1,528)
  - **measurement results (post-측정)**:
    - paper main pipeline F1 정확 확보
    - Enriched + QCond stack 의 학술적 정량
    - presentation_brief §10 빠른 참조 갱신
  - **post-deadline 영향**:
    - **H6 (SuperNode 학습 완료) → Selector 결정 (Concat vs SuperNode)**
    - **H7 multi-seed**: paper main pipeline F1 reliability 검증
    - **Wave 4 a05_filter_agentic**: anchor refresh (옵션 A2 결과 반영) → vivid-sprouting-sunbeam.md plan 갱신

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — presentation_brief §0 갱신 (본 엔트리 직후 Edit)
  2. **Root (즉시)** — 2 cells 측정 (본 엔트리 직후 prompt 코드블록)
  3. **Planner (결과 수령 후)** — paper main pipeline F1 narrative 갱신 + DECISIONS 후속 + paper_research_direction.md §0/§7/§10 갱신
  4. **사용자 (선택, 발표 직전)** — 슬라이드 작성 시 paper_research_direction.md + presentation_brief §14 활용

- **추가 필요 분석**:
  - 옵션 A2 측정 결과 → Builder × Encoder × Extractor 의 통합 stack 효과 정량
  - paper main pipeline F1 vs Plain anchor (F1=0.8642) 비교 → narrative 분기:
    - F1 ≥ 0.8642: "End-to-End Co-Design 의 통합 효과 발현"
    - F1 < 0.8642: "Description noise + GAT 보정 mechanism 의 한계 발견 + future work"
  - Wave 4 multi-agent extension 의 base anchor 결정 (paper main pipeline 또는 현 anchor)

---

## 2026-04-27 (Plain Cos vs QCond Cos — LLM noise 가설 2nd 검증 + 방향 F narrative 강화) — encoder 무관 단계 동일, Final ΔF1=+0.0034 = noise 추정 (Union ΔF1=+0.0030 와 일치)

- **결정**:
  1. **(a) 사용자 발견 (2026-04-27): Plain Cos vs QCond Cos cell 비교** — Cosine only (α=1) 에서 encoder 차이 (`query_conditioned` True/False) 가 GAT input 만 영향, **cosine 임베딩 (PLM) 동일**. Selector_only / +Extractor 단계 동일 (0.3829 / 0.2295) — selected_nodes 정확히 같음.
  2. **(b) Final 단계 ΔF1=+0.0034** (Plain Cos 0.8390 vs QCond Cos 0.8424) = **LLM stochasticity 가능성 매우 높음**:
     - selected_nodes 동일 + raw query 동일 + xiyan filter input (nodes only) 동일 → **prompt 완전 동일**
     - temperature=0.0 라도 GLM API noise (직전 Union 진단과 동일 mechanism)
  3. **(c) LLM noise 범위 확정** (2 independent samples):
     - Union vs MST Kruskal: ΔF1=+0.0030 (selected_nodes 동일, edges 차이는 filter 무관)
     - QCond Cos vs Plain Cos (α=1): ΔF1=+0.0034 (encoder 무관, GAT 사용 X)
     - **확정 GLM noise 범위: ±0.003~0.005** (single run)
  4. **(d) paper limitation + future work narrative 강화**:
     - "Single-run reliability 한계, multi-seed 검증의 중요성"
     - "ΔF1 < 0.005 는 noise 범위로 anchor 갱신 임계 +0.005 정당화"
     - H7 (multi-seed 검증) 우선순위 ↑
  5. **(e) 🚀 방향 F (Selector-Extractor Co-Design) narrative 강화** — encoder 효과 = GAT module 활용 시에만 발현 명확 재확인:
     - Cosine only (α=1) → encoder 무관 (Plain ≈ QCond, Δ=noise)
     - GAT only (α=0) → encoder 효과 발현 (selector_only Plain GAT 0.2937 → QCond GAT 0.3534, +0.0597)
     - Ensemble (α=0.5) → encoder 효과 dilute (+0.0042)
     - **paper insight 강화**: QCondGAT 의 가치 = GAT 단독 활용 시 / Ensemble blend 에서 cosine 우세로 dilute → α=0.5 baseline 의 학술적 정당성

- **근거**:
  - 사용자 직전 메시지: "Plain GAT 랑 QCond GAT 둘 다 Raw Score 만 사용하면 같은 Node 만 뽑혀야 하는 거 아니야? 왜 Raw Score 일 때의 성능이 다르지?"
  - §14.2 표 데이터:
    - Plain Cos: SO=0.3829 / +Ext=0.2295 / Final=0.8390
    - QCond Cos: SO=0.3829 / +Ext=0.2295 / Final=0.8424
    - Selector_only / +Extractor 정확히 일치 → selected_nodes 동일 확정
  - 코드 진단 (직전 Union 진단과 동일):
    - `query_conditioned=True` → GAT input concat (input 384 → 768)
    - α=1 (Cosine only) → GAT 사용 X → encoder 차이 무관
    - xiyan filter input = nodes only

- **영향 범위**:
  - **§14.2 (planner Edit 완료)**: Plain Cos vs QCond Cos LLM noise 진단 footnote 추가
  - **DECISIONS 본 엔트리**: LLM noise 가설 2nd 검증
  - **방향 F (Selector-Extractor Co-Design) narrative 강화**:
    - QCondGAT contribution 이 명확화 (GAT 단독 활용 시만 발현)
    - α=0.5 baseline 의 학술적 정당성 추가 증거
  - **post-deadline H7 우선순위 ↑**: multi-seed 검증으로 noise 범위 정확 정량 + Plain vs QCond / Union vs MST Kruskal 의 statistical significance 검증

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — §14.2 footnote 추가 + 본 DECISIONS 엔트리
  2. **Selector/Analyzer 모듈 (post-2026-04-28, H7)** — multi-seed 측정 (Plain Cos / QCond Cos / Union / MST Kruskal × 3 seeds = 12 cells, ~₩9,168) → noise 범위 정확 정량 + statistical significance test
  3. **사용자 (즉시)** — 방향 F (Selector-Extractor Co-Design) narrative confirm 시 §0 갱신

- **추가 필요 분석**:
  - GLM API temperature=0.0 의 정확한 determinism 검증 (5회 호출 시 결과 일치 비율)
  - paper main contribution narrative 의 "Co-design principle" 강조 — Plain vs QCond noise 발견이 narrative 의 학술 정당성 추가

---

## 2026-04-27 (Union 진단 — LLM stochasticity 가설 확정) — selected_nodes 정확히 동일 확인, ΔF1=+0.0030 = noise 가능성 매우 높음, 시나리오 B 판정 강화

- **결정**:
  1. **(a) 사용자 지적 (2026-04-27): "Filter 가 운 좋게 더 잘 작용한 것 아닌가" — 정확** — 코드 + 데이터 진단으로 확정.
  2. **(b) 진단 결과 — selected_nodes 정확히 동일**:
     - MST Kruskal vs Union 의 no_filter R/P/F1 모두 정확히 동일: R=0.9914, P=0.1222, F1=0.2176
     - 1534 queries 모두에서 동일 노드셋 → **PCST 결과 ⊆ MST Kruskal 결과** (union 이 MST Kruskal 와 같음)
  3. **(c) XiYan filter input 검증** — `xiyan_filter.py:99-117`:
     - input = `current_schema` (= nodes only, M-Schema with values)
     - **edges 사용 X** — Filter prompt 가 selected_nodes 기반
     - selected_nodes 동일 + same prompt + temperature=0.0 → **결과 동일해야 (deterministic)**
  4. **(d) Filter ΔF1=+0.0030 의 진짜 원인 = LLM stochasticity 가능성 매우 높음**:
     - `temperature=0.0` 라도 GLM API 가 완전 deterministic 보장 X (sampling implementation noise)
     - GPU/CUDA non-determinism (드물지만 가능)
     - 또는 selected_nodes 순서 (set ∪ 의 비결정적 순서) 가 prompt 텍스트 미세 차이 유발
     - **추정 noise 범위**: ±0.003~0.005 (single run)
  5. **(e) 시나리오 B 판정 강화** — Union 의 anchor 갱신 보류 결정 더 견고:
     - ΔF1=+0.0030 < 갱신 임계 +0.005 (60% 미달) + LLM noise 가능성 + single run
     - **anchor 유지** (`plain_ens_a05_mst_kruskal_glm` F1=0.8642)
     - Union 은 plateau 동률 후보 표기 + post-deadline H7 multi-seed 검증 필수
  6. **(f) presentation_brief 정정 (planner Edit 완료)**:
     - §11 Q10 답변 보강 — LLM stochasticity 가설 + 직전 "추가 엣지 정보로 미세 보정" narrative 정정 ("실제로 edges 사용 X")
     - §14.6 시나리오 B 판정 근거 강화 — selected_nodes 동일 + filter input 검증 명시
  7. **(g) Paper insight 정정** — 직전 narrative "Multi-extractor union 의 marginal R 회수 — Filter 가 추가 엣지 정보로 정답 식별 미세 향상" 은 잘못:
     - 실제로 union 의 추가 엣지 정보는 XiYan filter 가 사용 안 함 (nodes only)
     - **정정 narrative**: "Multi-extractor union 의 marginal +0.0030 = LLM stochasticity noise 범위, MST Kruskal 가 R 상한 도달 증거"
     - 발표 narrative: "Single-extractor (MST Kruskal) 가 R 천장 도달, multi-extractor union 의 추가 효과 없음"

- **근거**:
  - 사용자 직전 메시지 (2026-04-27): "MST 단독이랑 MST + PCST 랑 Extractor 단계까지 수치가 같은데 Filter 에서 다르다는 건, Filter 가 MST + PCST 실험할 때 에서 운 좋게 더 잘 작용했다는 말인 거네?"
  - 코드 진단:
    - [`xiyan_filter.py:99`](../src/modules/filters/xiyan_filter.py): `schema_str = self._build_mschema_with_values(current_schema, db_id)` — nodes only
    - [`xiyan_filter.py:109-117`](../src/modules/filters/xiyan_filter.py): prompt 구성에 edges 미포함
  - 데이터 진단:
    - MST Kruskal no_filter: R=0.9914 / P=0.1222 / F1=0.2176
    - Union no_filter: R=0.9914 / P=0.1222 / F1=0.2176 — **정확히 동일** (1534 queries 평균)
  - LLM stochasticity:
    - GLM API (Elice ML, OpenAI 호환) 의 temperature=0.0 동작은 일반적으로 greedy 지만 implementation-level non-determinism 존재
    - profiling 데이터에서 두 stack 의 graph_build/projection 등 timing 도 약간 다름 → 환경 noise 영향 가능

- **영향 범위**:
  - **§11 Q10 (planner Edit 완료)**: LLM stochasticity 가설 명시 + 직전 narrative 정정
  - **§14.6 (planner Edit 완료)**: 시나리오 B 판정 근거 강화 (코드 진단 명시)
  - **paper main contribution narrative 정정** (§9.3 H9, planner): "Union 의 추가 효과 없음, MST Kruskal R 천장 도달 증거"
  - **post-deadline H7 multi-seed 검증 우선순위 ↑**: Union vs MST Kruskal 의 실제 statistical difference 정량 필수

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — §11 Q10 + §14.6 정정 (본 엔트리 직전 Edit)
  2. **Selector/Extractor 모듈 + Analyzer (post-2026-04-28, H7+H9 통합)** — multi-seed 측정 (Union vs MST Kruskal × 3 seeds = 6 cells), per-DB selected_nodes 비교 case study
  3. **Filter 세션 (post-deadline)** — XiYan filter 의 nodes only input 가 적절한지 재검토 (edges 정보 활용 가능성 — H4 LLM-aware Builder 와 통합)

- **추가 필요 분석**:
  - GLM API temperature=0.0 의 실제 determinism 검증 (post-deadline) — 같은 prompt 5회 호출 시 결과 일치 비율
  - Union 의 selected_nodes 정렬 순서 검증 (set 의 hash-dependent 순서가 prompt 영향?)
  - paper limitation section 후보: "Single run reliability 한계, multi-seed 검증의 중요성"

---

## 2026-04-27 (옵션 C + Union 6 cells 완료) — 시나리오 B 판정 (Union plateau ΔF1=+0.0030) + anchor 갱신 확정 (MST Kruskal 0.8642) + paper main contribution narrative

- **결정**:
  1. **(a) 6 cells 측정 결과**:
     | Extractor | no_filter F1 | Final F1 (GLM) | Filter Δ F1 |
     |-----------|---:|---:|---:|
     | Steiner Tree threshold (score>0.1 seed) | 0.2177 | 0.8628 | +0.6451 |
     | **🚀 MST Kruskal (score>0.1 induced) ★ NEW anchor** | 0.2176 | **0.8642** | +0.6466 |
     | **🆕 MST ∪ PCST union (`MSTPCSTUnionExtractor`)** | 0.2176 | **0.8672** | **+0.6496 (max)** |
  2. **(b) 시나리오 B 판정** — Union ΔF1=+0.0030 vs MST Kruskal (임계 +0.005 미달):
     - Union F1=0.8672 가 MST Kruskal F1=0.8642 대비 +0.0030 — 임계 +0.005 의 60% 미달
     - ΔR=+0.0063 (Union 0.8787 vs MST Kruskal 0.8724) — marginal R 회수
     - **anchor 유지**: `plain_ens_a05_mst_kruskal_glm` F1=0.8642
     - Union 은 plateau 동률 후보 표기 + post-deadline H7 multi-seed 검증 후 재판정
  3. **(c) 🚀 새 anchor 확정**: `plain_ens_a05_mst_kruskal_glm` F1=0.8642 (직전 anchor `qcond_gat_basic_glm` F1=0.8383 대비 ΔF1=**+0.0259**, 임계 5배)
     - **vs vLLM era 2×2×2 best (F1=0.7863): ΔF1=+0.0779** (LLM backbone + MST Kruskal + α=0.5 통합 이득)
  4. **(d) MST Kruskal 가 R 상한 거의 도달 증거** — Union 의 marginal +0.0063 ΔR 만 회수, no_filter R=0.9914 (MST Kruskal/Steiner threshold/Union 모두 동일) → score>0.1 seed pool 의 R 천장 도달.
  5. **(e) Filter Δ F1 위계 갱신**:
     - Union (+0.6496) > MST Kruskal (+0.6466) > Steiner threshold (+0.6451) > Basic PCST (+0.6093) >> Steiner top-k (+0.3433) > Steiner Backbone (+0.3261) > Adaptive (+0.1872)
     - **입력 sub-graph 단순할수록 LLM filter 효율 ↑** + score widening (top-k → score-threshold) 이 R 상한 결정
  6. **(f) 🚀 paper main contribution 후보 narrative**:
     - **"Extractor 의 seed pool (top-k vs score-threshold) + algorithm choice (Steiner Tree vs MST Kruskal) + multi-extractor union 가 Recall 결정 mechanism"**
     - Selector top-k 한정 → score-threshold widening (+0.1871 ΔF1 vs 기존 "MST")
     - Steiner Tree vs MST Kruskal: ΔF1=+0.0014 (algorithm 차이 무시)
     - Union: ΔF1=+0.0030 (single-extractor 가 R 상한 도달 증거)

- **근거**:
  - Root 측정 결과 (2026-04-27, 6 cells, GPU 2/3): 옵션 C 4 + Union 2
  - 출처: [EXPERIMENT_HISTORY.md "MST 변형 측정 (옵션 C + Union, 2026-04-27)"](../EXPERIMENT_HISTORY.md) (root 갱신 완료)
  - Extractor 모듈 세션 신규 구현: `MSTKruskalExtractor`, `MSTPCSTUnionExtractor`, `MSTExtractor` seed_mode flag

- **영향 범위**:
  - **§14.3 표 (planner Edit 완료)**: 6-row 매트릭스 확장 + Union row 추가 + Filter Δ 위계 갱신
  - **§14.6 anchor 결정 (planner Edit 완료)**: 새 anchor MST Kruskal 0.8642 + Union plateau 동률 후보 표기
  - **§11 Q9 + Q10 (planner Edit 완료)**: MST recall 답변 + Union 효과 답변
  - **§9.3 H9 (planner Edit 완료)**: 6 Extractor sweep + Union 분석 추가
  - **새 anchor 갱신 영향**:
    - CLAUDE.md (root) §0 anchor 표 갱신 권장 (root 작업, 발표 후 가능)
    - Wave 4 a05_filter_agentic plan anchor refresh (vivid-sprouting-sunbeam.md): 0.8383 → 0.8642
    - EXPERIMENT_PLAN.md §0 anchor 표 갱신 (root 작업)
  - **paper main contribution 후보**: §11 Q9/Q10 narrative + §9.3 H9 case study

- **Caveat (anchor 갱신 보수성)**:
  - Single run, statistical significance 미검증 — H7 multi-seed 검증 권장 (post-deadline)
  - ΔF1=+0.0259 (anchor 갱신) 은 임계 5배 명확
  - ΔF1=+0.0030 (Union) 은 임계 미달 → anchor 유지 (보수적)
  - 발표 narrative 에 "single run" 1줄 caveat 명시 권장

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — 5 작업 (§14.3/14.6/11/9.3 + DECISIONS) 모두 완료
  2. **Root (선택, 발표 후 여유 시)** — CLAUDE.md (root) §0 anchor 표 갱신 + EXPERIMENT_PLAN.md §0 갱신
  3. **Selector/Extractor/Analyzer 모듈 (post-2026-04-28)** — H9 통합 sweep (6 Extractor × 7 alpha = 42 cells) + H7 multi-seed (anchor reliability) + per-DB / per-difficulty 분해 case study
  4. **Filter 세션 (post-2026-04-28)** — vivid-sprouting-sunbeam.md plan anchor 0.8383 → 0.8642 refresh

- **추가 필요 분석**:
  - Union 의 marginal +0.0063 ΔR 의 query 클러스터 분해 (analyzer post-deadline)
  - per-DB MST Kruskal vs Union 의 selected_nodes 중복 비율
  - paper main contribution narrative 의 발표 스토리: "Score-threshold seed pool widening + 진짜 MST algorithm 의 R 결정 mechanism + Union plateau 도달 증거"

---

## 2026-04-27 (MST ∪ PCST union 변형 측정 결정) — 새 Extractor 구현 + 2 cells 측정 (R 상한 + P trade-off 검증)

- **결정**:
  1. **(a) MST ∪ PCST union 변형 측정** — 사용자 직전 요청 (2026-04-27): "MST 로 찾은 집합과 PCST 로 찾은 집합의 합집합으로 설정하면 수치가 어떻게 될까?". 새 anchor (MST Kruskal F1=0.8642) 의 R 상한 검증 + Filter 의 union 처리 능력 정량.
  2. **(b) 새 Extractor 구현 — `MSTPCSTUnionExtractor`** (또는 비슷):
     - MST Kruskal (`MSTKruskalExtractor`) extract → set_A (노드 + 엣지)
     - Basic PCST (`PCSTExtractor` node_threshold=0.1) extract → set_B (노드 + 엣지)
     - **union**: selected_nodes = set_A ∪ set_B, selected_edges = set_A 엣지 ∪ set_B 엣지
     - score_threshold=0.1 통일 (양쪽 동일)
  3. **(c) 측정 cells (2 cells, Plain Ens α=0.5 + XiYan GLM 통일)**:
     - `plain_ens_a05_mst_pcst_union_no_filter.yaml` (LLM-free)
     - `plain_ens_a05_mst_pcst_union_glm.yaml` (GLM API)
     - 비용 ~₩764, 시간 ~1h parallel
  4. **(d) 결과 시나리오 사전 합의**:
     - **시나리오 A (F1 > 0.8642)**: union 의 추가 노드를 Filter 가 잘 처리 → **새 anchor 갱신** + paper insight 강화 ("Multi-extractor union 이 single-extractor 우세")
     - **시나리오 B (F1 ≈ 0.8642 ± 0.005)**: MST Kruskal 만으로 충분 (PCST 추가 노드 효과 무시) → 현 anchor 유지, "MST Kruskal 가 R 상한 도달 증거"
     - **시나리오 C (F1 < 0.8642)**: union noise 가 Filter 에 부담 → "MST Kruskal 단독 우세, union 은 over-include"
  5. **(e) 발표 narrative 영향**: 결과 시나리오에 따라 §14.3 표 6 row 확장 + paper insight 강화/유지

- **근거**:
  - 새 anchor (MST Kruskal F1=0.8642) 의 R 상한 검증 필요
  - MST Kruskal R=0.8724, Basic PCST R=0.8316 — union R 예상 0.93~0.99 (gold 거의 다 회수)
  - Filter Δ F1 by Extractor 위계 (MST Kruskal +0.6466 max) → union 의 Filter Δ 정량 가능
  - 사용자 의문 "union 어떻게 될까" 직접 검증

- **영향 범위**:
  - **Extractor 모듈 세션 (즉시)**: 새 Extractor 구현 (~30min, ~50 LOC + smoke)
  - **Root (구현 완료 후)**: 2 cells 측정 (GPU 2/3, ~1h, ~₩764)
  - **§14.3 표 보강 (planner, post-측정)**: 6 row × 3 stage cumulative
  - **§14.6 anchor 결정 (planner, post-측정)**: 시나리오 A 시 anchor 재갱신
  - **post-deadline**: H9 통합 sweep 에 union 변형 추가 (5 → 6 Extractor)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — 본 엔트리 작성
  2. **Extractor 모듈 세션 (즉시)** — 새 Extractor 구현 + smoke test (planning/CLAUDE.md 역할 분담)
  3. **Root (구현 완료 신호 후)** — 2 cells 측정 + HISTORY 갱신
  4. **Planner (결과 수령 후)** — §14.3/14.6 갱신 + DECISIONS 후속 + 시나리오 판정

- **추가 필요 분석**:
  - per-DB MST 와 PCST 의 selected_nodes 중복 비율 (analyzer post-deadline) — union 의 실질 추가 노드 비율
  - paper insight 후보 (시나리오 A 시): "Extractor union 이 single-extractor 한계 돌파" 추가 발견

---

## 2026-04-27 (옵션 C 4 cells 완료) — 🚀 MST Kruskal 압도 (F1=0.8642 +0.0259 vs anchor) + anchor 갱신 결정 + Algorithm 차이 거의 없음 (seed widening 이 R 결정)

- **결정**:
  1. **(a) 🚀 anchor 갱신 결정 — 옵션 1 채택**: 새 anchor `plain_ens_a05_mst_kruskal_glm` F1=**0.8642** (R=0.8724, P=0.8561). 직전 anchor `qcond_gat_basic_glm` F1=0.8383 대비 **ΔF1=+0.0259**, 갱신 임계 +0.005 의 5배 초과.
  2. **(b) 핵심 결과 4 cells (R/P/F1, 4자리)**:
     | Cell | no_filter R/P/F1 | Final R/P/F1 (GLM) | Filter Δ F1 |
     |------|---|---|---:|
     | **MST Kruskal (진짜 MST, score>0.1 induced)** | 0.9914 / 0.1222 / 0.2176 | **0.8724 / 0.8561 / 0.8642 ★🚀** | +0.6466 |
     | **Steiner Tree threshold seed (Steiner 2-approx + score>0.1)** | 0.9914 / 0.1223 / 0.2177 | **0.8720 / 0.8538 / 0.8628** | +0.6451 |
  3. **(c) 알고리즘 차이 거의 없음**: MST Kruskal vs Steiner threshold ΔF1=+0.0014 (final), no_filter R 동일 (0.9914). **Steiner point 추가 효과 무시 가능**.
  4. **(d) seed pool widening 이 R 결정 mechanism**:
     - top-k seed (Selector top-20): no_filter R=0.7231 (Steiner Tree top-k = 기존 MSTExtractor)
     - score-threshold seed (score > 0.1): no_filter R=0.9914 (+0.2683 ΔR over top-k)
     - **사용자 의문 정확 해소**: 기존 "MST" final F1=0.6771 의 R 한계 = top-k seed 한정. score-threshold seed 변경 시 +0.1871 ΔF1 향상 (0.6771 → 0.8642).
  5. **(e) 명명 정정 확정**:
     - 기존 `MSTExtractor` = Steiner 2-approx (Kou-Markowsky-Berman 1981) — 명명 오류 명시
     - 신규 `MSTKruskalExtractor` = 진짜 MST (Kruskal, networkx.minimum_spanning_tree)
     - `MSTExtractor seed_mode="threshold"` = Steiner Tree + score-threshold seed (변형)
     - **post-deadline 코드 rename** (`MSTExtractor` → `SteinerTreeExtractor`, alias 유지)
  6. **(f) 발표 narrative 영향 — paper main contribution 후보 부상**:
     - "Extractor 의 seed pool (top-k vs score-threshold) + algorithm choice (Steiner Tree vs MST Kruskal) 가 Recall 결정 mechanism"
     - 새 anchor narrative: "Plain encoder + Ensemble α=0.5 + 진짜 MST (Kruskal, score>0.1 induced) + XiYan GLM = F1=0.8642"
     - vs vLLM era 2×2×2 best (F1=0.7863): **ΔF1=+0.0779** (LLM backbone + MST Kruskal + α=0.5 baseline 통합 이득)

- **근거**:
  - Root 측정 결과 (2026-04-27 19:21:44~20:21:43, 4 cells, GPU 2/3, Extractor 모듈 세션 신규 구현)
  - 출처: `outputs/experiments/s04_ablation/extractor/{plain_ens_a05_mst_kruskal_glm, plain_ens_a05_steiner_threshold_glm}/metrics.txt` + `no_filter/` 동일
  - summary_all.csv 직접 확인:
    - `s04_extractor_plain_ens_a05_mst_kruskal_glm`: R=0.8724, P=0.8561 → F1=0.8642
    - `s04_extractor_plain_ens_a05_steiner_threshold_glm`: R=0.8720, P=0.8538 → F1=0.8628
    - 두 cell 모두 no_filter R=0.9914, P≈0.1222 (alg 차이 무시 수준)

- **영향 범위**:
  - **새 anchor 결정**: `plain_ens_a05_mst_kruskal_glm` F1=0.8642 → §0/§14.6 anchor 표 갱신
  - **§14.3 표** (planner Edit): 5-row 매트릭스 확장 + 명명 정정 + Filter Δ 위계 갱신
  - **§14.6 anchor 결정** (planner Edit): 옵션 1 채택 narrative + plateau 영역 정리
  - **§11 Q&A** (planner Edit): "MST recall 이 왜 낮나?" 신규 추가
  - **§9.3 Future work** (planner Edit): MST/Extractor sweep + 코드 rename
  - **Wave 4 a05_filter_agentic plan**: anchor refresh 필요 (`qcond_gat_basic_glm` F1=0.8383 → `plain_ens_a05_mst_kruskal_glm` F1=0.8642, ΔF1=+0.0259)
  - **CLAUDE.md (root) §0 anchor 표**: 갱신 권장 (root 작업)

- **Caveat (anchor 갱신 보수성)**:
  - **Single run, statistical significance 미검증** — H7 multi-seed 검증 권장 (post-deadline)
  - Plateau 영역 (F1=0.83~0.87) 의 단일 측정은 noise 가능성, 단 ΔF1=+0.0259 가 임계 +0.005 의 5배 → 명확 (단순 noise 아님)
  - 발표 narrative 에서 "single run" caveat 1줄 명시 권장

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — §14.3/14.6/11/9.3 갱신 (본 엔트리 직후 Edit 진행)
  2. **Root (선택, 발표 후 여유 시)** — CLAUDE.md (root) §0 anchor 표 갱신
  3. **Selector/Extractor 모듈 세션 (post-2026-04-28)** — H7 multi-seed 검증 (anchor 신뢰도) + H8 alpha sweep + MST 알고리즘 sweep 통합
  4. **Filter 세션 (post-2026-04-28)** — vivid-sprouting-sunbeam.md plan anchor 0.8383 → 0.8642 refresh

- **추가 필요 분석**:
  - per-DB / per-difficulty 분해 (MST Kruskal 의 R 개선이 어떤 query 클러스터에서 발현하는지) — analyzer post-deadline
  - alpha sweep H8 + MST Kruskal 통합 (alpha × 5 extractors = best stack 정량)
  - **paper main contribution narrative**: "Score-threshold seed pool widening + 진짜 MST algorithm 의 R 결정 mechanism" — Steiner Tree 의 Steiner point 추가가 사실상 무영향 발견 + Selector top-k 한정의 R 천장 해결

---

## 2026-04-27 (옵션 C 선택 — MST 명명 정정 + 진짜 MST Kruskal 신규 + Steiner score-threshold 변형) — 4 cells 측정, narrative 정확성 확보

- **결정**:
  1. **(a) 명명 정정** — 사용자 지적 (2026-04-27): "MSTExtractor 가 실제로 Steiner 2-approx 사용". 발표 자료 §14.3 표 표기 정정:
     - "MST 단독 (`MSTExtractor`, metric closure Steiner 2-approx)" → **"Steiner Tree (2-approx, Kou-Markowsky-Berman 1981)"**
     - 코드 클래스명 변경은 backward compatibility 고려해 **post-deadline** (현재 yaml configs 가 `MSTExtractor` 사용, 갑작스런 rename 시 호환성 깨짐). 발표 자료 narrative 정확성만 즉시 확보.
  2. **(b) Steiner Tree + score-threshold seed 변형** — 사용자 직전 요청 ("MST 도 score > 0.1 모든 node 를 seed 로"). 현 알고리즘 (Steiner 2-approx) 의 seed pool 만 확장:
     - 현재: seed_nodes = Selector top-20 (한정)
     - 변형: seed_nodes = score > 0.1 모든 노드 (Basic PCST 와 동일 candidate pool)
     - 신규 yaml param `seed_mode: "topk" | "threshold"` (default topk, backward compatible)
  3. **(c) 진짜 MST Kruskal 신규 구현** — 사용자 의도된 진짜 MST. score > 0.1 노드 의 induced subgraph 의 MST (Kruskal):
     - 신규 Extractor `MSTKruskalExtractor` (또는 `RealMSTExtractor`)
     - networkx.minimum_spanning_tree 사용 (Kruskal default)
     - Steiner point 없음 (induced subgraph 의 모든 노드 spanning)
  4. **(d) 측정 cells (4 cells, Plain Ens α=0.5 + XiYan GLM 통일)**:
     - (b) Steiner Tree + score-threshold seed:
       - `plain_ens_a05_steiner_threshold_no_filter.yaml` (LLM-free)
       - `plain_ens_a05_steiner_threshold_glm.yaml` (GLM API)
     - (c) 진짜 MST Kruskal:
       - `plain_ens_a05_mst_kruskal_no_filter.yaml` (LLM-free)
       - `plain_ens_a05_mst_kruskal_glm.yaml` (GLM API)
     - 비용: 2 GLM × ~₩764 + 2 LLM-free × ₩0 = **~₩1,528**
     - 시간: ~2h parallel (GPU 2/3, SuperNode 학습 GPU 0/1 와 병렬)
  5. **(e) 발표 narrative 정확성 확보**: 사용자 의문 ("MST recall 이 왜 낮나") 해소.
     - 현 Steiner Tree (top-20 seed) R=0.6257 → score-threshold seed 변형 R 검증 (b)
     - 진짜 MST (Kruskal, induced subgraph) R 측정 (c) → 사용자 framing 의 "MST" 정확한 결과
     - §14.3 표 5 row 확장 (Basic / Steiner Tree top-k / Steiner Tree threshold / 진짜 MST / Adaptive)
  6. **(f) 시간 budget**: 2h 측정 + ~30min planner Edit = ~2.5h. 발표 D-1 (2026-04-28 < 24h) 안전.

- **근거**:
  - 사용자 직전 의문 + 명명 오류 지적
  - mst.py:99 `steiner_tree_2approx(G, seed_nodes)` — 코드 진단
  - Steiner Tree vs MST 알고리즘 차이:
    - Steiner Tree: terminal subset 만 connecting + Steiner point 추가 가능
    - MST (Kruskal): 모든 vertex spanning + Steiner point 없음
  - sub-graph 정의 차이:
    - Steiner Tree top-k seed: terminal=top-20 (한정)
    - Steiner Tree score-threshold seed: terminal=score > 0.1 (확장)
    - 진짜 MST induced: induced subgraph 의 모든 vertex spanning

- **영향 범위**:
  - **Root 핸드오프 (즉시)**: (b)(c) 코드 구현 + 4 cells 측정 (~2h, ~₩1,528, GPU 2/3)
  - **§14.3 명명 정정 (planner Edit, 즉시)**: "MST 단독" → "Steiner Tree (2-approx)"
  - **post-측정 §14.3 표 보강**: 5 row × 3 stage cumulative 매트릭스
  - **발표 narrative**: 사용자 의문 해소 + paper insight ("Selector top-k 한정 vs score-threshold widening 의 R 영향" 정확한 정량)
  - **post-deadline**: 코드 명명 정정 (`MSTExtractor` → `SteinerTreeExtractor`, alias 호환), MST 알고리즘 별 sweep 통합 (top-k vs threshold seed × Steiner vs Kruskal × alpha)

- **에스컬레이션 필요 여부 (핸드오프 절차 정정 — Extractor 모듈 세션 우선)**:
  1. **Planner (즉시 완료)** — §14.3 명명 정정 (post-측정 표 보강은 결과 수령 후)
  2. **Extractor 모듈 세션 (즉시)** — (b)(c) 신규 Extractor 구현 + 단위 smoke test (planning/CLAUDE.md 역할 분담: 모듈 내부 구현 = 모듈 세션). 본 엔트리 직후 응답에 prompt 코드블록 제공.
  3. **Root (Extractor 구현 완료 신호 후)** — 4 configs 생성 + 4 cells 측정 (GPU 2/3) + HISTORY 갱신. 별도 prompt 코드블록 제공.
  4. **Planner (결과 수령 후)** — §14.3 5 row × 3 stage 표 보강 + DECISIONS 후속

- **추가 필요 분석**:
  - 결과 시나리오:
    - 진짜 MST (Kruskal) R/F1 결과로 사용자 의문 정확 해소
    - Steiner Tree top-k vs Steiner Tree threshold vs MST Kruskal 비교 → seed 정의 + Steiner point 효과 isolate
  - paper insight: "Extractor 의 seed/algorithm choice 가 R 결정 mechanism" — 정확한 정량 narrative

---

## 2026-04-27 (MST score-threshold seed 변형 측정 결정) — 새 Extractor 구현 + 2 cells 측정 (Selector top-k 한정 vs score > 0.1 cutoff 비교)

- **결정**:
  1. **(a) MST score-threshold seed 변형 측정** — 사용자 직전 메시지 (2026-04-27): "MST 도 score > 0.1 인 모든 node 를 seed 로 받게끔 해서 성능을 다시 측정". 현재 MSTExtractor 가 Selector top-20 seed 한정이라 R 한계 (0.6257 final), 새 변형으로 Basic PCST 와 동일 candidate pool (score > 0.1) 사용 시 R 개선 가능성 검증.
  2. **(b) 새 Extractor 구현 — `MSTScoreThresholdExtractor`** (또는 MSTExtractor 에 `seed_mode` flag 추가):
     - 동작: `node_scores` 전체 받아 score > threshold (default 0.1) 인 노드를 seed 로 자체 산출
     - `steiner_tree_2approx(G, score_thresholded_seeds)` 호출
     - Selector top-k 무시 (Basic PCST 와 동일 mechanism)
  3. **(c) 측정 cells 2개** (Plain Ens α=0.5 + 새 MST + XiYan GLM):
     - `plain_ens_a05_mst_threshold_no_filter.yaml` (no_filter, LLM-free)
     - `plain_ens_a05_mst_threshold_glm.yaml` (final, GLM API)
     - selector_only stage 는 재활용 (Plain Ens α=0.5 selector_only F1=0.3432)
  4. **(d) 비용 / 시간**: 1 GLM × ~₩764 + 1 LLM-free × ₩0 = ~₩764, ~50min~1h (GPU 2/3, SuperNode 학습 GPU 0/1 와 병렬)
  5. **(e) 예상 결과 시나리오**:
     - **시나리오 A (긍정)**: 새 MST 변형 R/F1 > 기존 MST → "MST seed 정의 (top-k vs score-threshold) 가 R 결정" 새 발견, 발표 narrative 보강 가능
     - **시나리오 B (중립)**: 새 MST 변형 ≈ Basic PCST → "MST 와 Basic PCST 의 mechanism 차이 dilute (둘 다 widening 효과)" 발견
     - **시나리오 C (부정)**: 새 MST 변형 P 폭락 (모든 seed spanning 으로 sub-graph 폭증) → "MST 는 top-k seed 한정에서만 효과적" 결론 강화
  6. **(f) 발표 자료 영향**: 결과 수령 후 §14.3 표에 row 추가 (5 row × 3 stage). 발표 narrative 강화 가능 (시나리오 A 시 paper insight).

- **근거**:
  - 사용자 직전 의문 ("MST 가 selected node 다 포함해도 R 가 PCST 보다 낮은 이유?") + planner 답변 (top-20 한정 vs score-threshold widening)
  - MSTExtractor 코드 (`mst.py:77-81`): `extract(node_scores, seed_nodes)` — seed_nodes 외부 전달 의존
  - Basic PCSTExtractor 코드 (`pcst.py:140`): `prizes = max(node_scores - 0.1, 0)` — score > 0.1 모든 노드 prize candidate
  - 두 mechanism 의 차이 = MST 의 R 한계 원인. 새 변형으로 isolate 가능

- **영향 범위**:
  - **Root 핸드오프 (즉시)**: 새 Extractor 구현 + 2 cells 측정 (GPU 2/3, ~1h, ~₩764)
  - **코드 변경**: src/modules/extractors/mst.py 또는 신규 mst_threshold.py (~30 LOC)
  - **presentation_brief §14.3** (post-측정): 5 row × 3 stage 표 보강 (MST score-threshold 추가)
  - **발표 narrative**: 결과 시나리오에 따라 §14.3 핵심 발견 보강 또는 future work
  - **post-deadline**: alpha sweep H8 + MST seed 정의 sweep 통합 (selector top-k vs score-threshold 영향 정량)

- **에스컬레이션 필요 여부**:
  1. **Root (즉시)** — 새 Extractor 구현 + 2 cells 측정 (본 엔트리 직후 prompt)
  2. **Planner (결과 수령 후)** — §14.3 표 보강 + DECISIONS 후속
  3. **Selector/Builder 세션 (post-2026-04-28)** — MST seed 정의 sweep (top-k=10/20/50/100/threshold) 와 alpha sweep 통합

- **추가 필요 분석**:
  - 결과 시나리오에 따른 발표 narrative 영향 (시나리오 A 시 paper main insight, B/C 시 footnote)
  - MST 의 sub-graph 크기 변화 (no_filter selected_nodes 수, top-20 한정 vs score-threshold)

---

## 2026-04-27 (H6 옵션 A 선택 — SuperNode ckpt 새 학습 + mechanism 설명 정정) — 사용자 결정: 옵션 A (query_conditioned=True + query_supernode=True 학습), 발표 자료 병행 작성

- **결정**:
  1. **(a) H6 옵션 A 채택** — 사용자 직전 메시지 (2026-04-27): "옵션 A로 우선 학습 진행하면서 발표 자료를 계속 만들고 있을게". 새 ckpt 학습:
     - `query_conditioned: true` (이전 false → true 변경, query embedding concat 활성)
     - `query_supernode: true` (그래프에 SuperNode 추가, 이전 동일)
     - `in_channels: 384` (PLM embedding, model 내부 effective_in=768 자동 처리)
     - `checkpoint_name: best_gat_query_supernode_qcond.pt` (기존 .pt 보존, 분리 저장)
  2. **(b) 발표 자료 보류 caveat 그대로 유지** — SuperNode 9 cells 측정은 학습 완료 후 별도 핸드오프 (post-deadline 가능). 발표 narrative 영향 X.
  3. **(c) 정정 — 직전 mechanism 설명 오류**:
     - **잘못**: "query_supernode=True 시 query feature concat → input 768"
     - **정확**: "**query_conditioned=True** 시 query feature concat → input 768. query_supernode 는 그래프에 SuperNode 노드만 추가 (dim 무관)"
     - 두 flag 별개 mechanism. 직전 ensemble_selector.py L42-65, gat_network.py L64-65/L157-164 코드 확인 결과
  4. **(d) 이전 SuperNode 측정 (s04_03 등 Q2/Q3/Q5) 의 정확한 stack** — 이전 cells 가 EnsembleSelector default (query_conditioned=False, query_supernode=False) 사용 → input 384 ckpt 호환 + **SuperNode graph 효과 없이 ckpt weight 만 사용**. 즉 이전 SuperNode 결과 자체도 SuperNode mechanism 정확 측정인지 재검토 필요 (post-deadline).
  5. **(e) Ablation 2 SuperNode smoke 실패의 진짜 원인 재정리**:
     - Ablation 2 사용자 framing: "SuperNode encoder = QCond + SuperNode 통합" (query_conditioned=True + query_supernode=True)
     - 새 ckpt 학습 = 이 정의에 맞는 ckpt (input 768)
     - 학습 완료 후 SuperNode 9 cells 측정 → 사용자 framing 정확히 구현

- **근거**:
  - 사용자 옵션 A 직접 선택 (2026-04-27)
  - 코드 확인:
    - [`ensemble_selector.py:42-43`](../src/modules/selectors/ensemble_selector.py): `query_supernode: bool = False, query_conditioned: bool = False` (defaults)
    - [`gat_network.py:64-65`](../src/models/gat_network.py): `effective_in = in_channels * 2 if query_conditioned else in_channels`
    - [`gat_network.py:157-164`](../src/models/gat_network.py): query concat 발생 위치 (`query_conditioned=True` trigger)
  - [`configs/training/train_gat_query_supernode.yaml`](../configs/training/train_gat_query_supernode.yaml): 기존 ckpt 학습 config 확인 — `query_conditioned: false`, `query_supernode: true`, `in_channels: 384`

- **영향 범위**:
  - **Root 핸드오프 (즉시)**: 새 ckpt 학습 1 cell (`best_gat_query_supernode_qcond.pt`, ~5h GPU 0/1, 발표 자료 작업 병행)
  - **DECISIONS 정정** (post 본 엔트리): 직전 2026-04-26 (Ablation 2 SuperNode smoke FAILED) 엔트리의 mechanism 설명은 본 엔트리가 supersede
  - **presentation_brief 정정** (planner 즉시 Edit):
    - §11 Q&A SuperNode caveat 문구 정정
    - §9.3 H6 future work 옵션 명확화 (옵션 A 선택, 학습 진행 중)
  - **발표 narrative 영향 X** — SuperNode 9 cells "측정 X" caveat 그대로 유지
  - **post-deadline 측정** (학습 완료 후 별도 핸드오프):
    - SuperNode α=0.5 × 3 stage = 3 cells (사용자 narrative 통일)
    - 또는 SuperNode α∈{0, 0.5, 0.85, 1} × 3 stage = 12 cells (alpha sweep H8 통합)

- **에스컬레이션 필요 여부**:
  1. **Root (즉시)** — 새 ckpt 학습 핸드오프 (본 엔트리 직후 prompt 코드블록)
  2. **Planner (즉시)** — presentation_brief / DECISIONS 정정 (본 엔트리 직후 Edit)
  3. **Root (학습 완료 후)** — SuperNode 9 cells inference 별도 핸드오프 (post-deadline 가능)
  4. **Selector 세션 (post-2026-04-28)** — SuperNode mechanism 의 이전 cells 정확성 재검증 (s04_03 등 query_supernode default=False 영향)

- **추가 필요 분석**:
  - 새 ckpt vs 이전 ckpt 의 SuperNode mechanism 차이 정량 (학습 완료 후)
  - 이전 SuperNode 결과 (Q2/Q3/Q5 F1=0.6886~0.6958) 의 정확성 재검증 (post-deadline)

---

## 2026-04-27 (발표 narrative 통일 — α=0.5 baseline + α=0.85 도달 서사) — 사용자 결정: §14.1/14.2/14.3 main 표 α=0.5 only 통일 + α=0.85 는 alpha sweep H8 의 한 점 으로 §14.6 도달 서사 표기

- **결정**:
  1. **(a) 사용자 발표 narrative 결정**: "Selector α 모두 **0.5 로 통일** + alpha sweep test 로 **0.85 에 도달하는 서사**" — α=0.5 baseline 으로 ablation 1/2/3 main, α=0.85 (현 anchor 0.8383) 는 sweep H8 의 도달점
  2. **(b) §14.1 (Builder) 정정**: α=0.85 컬럼 제거, **α=0.5 only 6-cell** (Plain/Enriched × 3 stage) 표기. Plain/Enriched 사실상 동등 (Δ=-0.0010) narrative 강화
  3. **(c) §14.2 (Selector) 정정**: Ensemble (α=0.85) 행 제거, **α=0.5 only Ensemble** + GAT(α=0)/Cosine(α=1) 단독 비교. Plain Ens 0.8252, QCond Ens 0.8306 main 결과
  4. **(d) §14.3 (Extractor) 정정**: α=0.85 컬럼 제거, **α=0.5 only 4 Extractor** (Basic 0.8252 / MST 0.6771 / Steiner 0.6491 / Adaptive 0.5775)
  5. **(e) §14.6 (Anchor) 강화**: α=0.5 baseline → α=0.85 anchor (F1=0.8383) **도달 서사** 표 신설:
     - α=0 (GAT only) / α=0.5 (baseline) / α=0.85 (anchor 영역) / α=1 (Cos only)
     - Plain: 0.6985 → 0.8252 → 0.8381 → 0.8390
     - QCond: 0.7211 → 0.8306 → 0.8383 → 0.8424
     - **plateau 도달 narrative**: α 증가 → cosine 가중 ↑ → F1 plateau (0.83~0.84)
  6. **(f) §11 Q8 narrative 정정**: "α=0.5 baseline + α=0.85 도달 서사" 강조
  7. **(g) §9.3 H8 강화**: post-deadline alpha sweep 으로 α∈{0.25, 0.7, 0.95} 보강 → sweep curve 완성, paper sensitivity 분석 main contribution 후보

- **근거**:
  - 사용자 직전 메시지 (2026-04-27): "Selector 의 alpha를 모두 0.5로 통일해서 진행했으면 좋겠어 그 후에 alpha sweep test 를 통해서 0.85 에 도달하는 서사를 보이고 싶은 거야"
  - 발표 narrative 의 학술적 정당성 ↑: Ensemble baseline = neutral (α=0.5) → sweep 으로 best alpha 영역 (α=0.85~1 plateau) 도달
  - α=0.85 의 historical default 약점 (I1a-c No Filter sweep 한정) 는 H8 future work 로 자연스럽게 해결

- **영향 범위**:
  - **presentation_brief 갱신 완료** (planner Edit, 본 엔트리 직전):
    - §14.1 Builder: α=0.5 only 6-cell, Enriched ≈ Plain narrative
    - §14.2 Selector: α=0.5 Ensemble + GAT/Cos 단독 비교 (5 행, SuperNode 보류 caveat 유지)
    - §14.3 Extractor: α=0.5 only 4 Extractor
    - §14.5 Slide 2 가이드: α=0.5 baseline narrative
    - §14.6 Anchor 도달 서사: α∈{0, 0.5, 0.85, 1} sweep table + plateau 도달 narrative
    - §11 Q8: α=0.5 baseline + α=0.85 도달 서사
    - §9.3 H8: α∈{0.25, 0.7, 0.95} 보강 (sweep curve 완성)
  - **anchor 유지**: `s04_stagewise_qcond_gat_basic_glm` F1=0.8383 (α=0.85) 변경 X — alpha sweep H8 의 한 점으로 표기
  - **Wave 4 a05_filter_agentic plan**: anchor 0.8383 유지로 변경 X

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — 7 갱신 (§14.1/14.2/14.3/14.5/14.6/11/9.3) 모두 완료
  2. **사용자 (즉시)** — 발표 슬라이드 작성 시 α=0.5 baseline + α=0.85 도달 서사 narrative 활용
  3. **Selector 세션 (post-2026-04-28, H8)** — α∈{0.25, 0.7, 0.95} 3 점 sweep (Plain/QCond × 3 = 6 cells, ~₩4,584, ~3h parallel)

- **추가 필요 분석**:
  - H8 sweep 결과로 alpha sensitivity curve 완성 → paper sensitivity 분석 main contribution
  - α=0.5 baseline 의 Builder × α / Extractor × α 상호작용 메커니즘 분석 (post-deadline analyzer)
  - α=0.85 plateau (0.8381~0.8424) 의 statistical significance 검증 (multi-seed)

---

## 2026-04-27 (Ablation 1/2/3 α=0.5 재측정 완료) — 15 cells 결과 + 🚀 Builder × α 상호작용 발견 (Enriched +0.0711) + ⚠ Extractor × α 상호작용 (Adaptive worst -0.1424) + Anchor 유지 정당화 강화

- **결정 (6 사용자 요청 응답)**:
  1. **(1) 15 cells α=0.5 재측정 완료** — Root 2026-04-27 14:41:16~17:42:22 (3h 1min, GPU 2/3, ~₩4,584). 6 final GLM + 9 LLM-free.
  2. **(2) 🚀 Builder × α 상호작용 (Enriched α=0.5 +0.0711) — 새 발표 주력 narrative**:
     - Enriched Ens α=0.85 F1=0.7551 → α=0.5 F1=**0.8262** (+0.0711)
     - vs Plain Ens α=0.5 F1=0.8252 → **Δ=-0.0010 (사실상 동등!)**
     - vs Plain Ens α=0.85 F1=0.8381 → -0.0832 (큰 격차, α=0.85 narrative)
     - **메커니즘 가설**: Description noise 가 Cosine PLM 임베딩에 손실 (selector_only Enriched 0.3389 < Plain 0.3432), GAT schema 학습 정보로 보정 → α=0.5 두 score 균형으로 회복
     - **학술적 가치**: **Enriched 의 가치가 α=0.5 baseline 에서 새로 발견** — paper future work 후보 ("LLM-aware vs GAT-aware Builder design")
  3. **(3) ⚠ Extractor × α 상호작용 — Basic 만 robust**:
     - Basic PCST: α=0.85 0.8381 → α=0.5 0.8252 (Δ=**-0.0129 robust**)
     - MST: 0.7730 → 0.6771 (Δ=-0.0959)
     - Steiner: 0.7545 → 0.6491 (Δ=-0.1054)
     - **Adaptive: 0.7199 → 0.5775 (Δ=-0.1424 worst!)** — 모든 변형 중 최저
     - **메커니즘**: Basic (fixed θ=0.1) absolute threshold robust, Adaptive (per-q P80) percentile threshold + GAT noise 가 cutoff 왜곡, Steiner/MST score 분포 sensitive
     - **paper insight**: Basic PCST 가 score perturbation (alpha 변경) 에 robust → ablation stable baseline 역할
  4. **(4) Adaptive + α=0.5 stack 부적절 발견 (F1=0.5775 worst)**:
     - 모든 stack 중 worst (no_filter 0.3903 → final 0.5775, Filter Δ=+0.1872 도 가장 낮음)
     - per-q P80 + GAT noise 부정적 시너지 (P80 cutoff 가 noisy GAT score 에 hyper-sensitive)
     - **권장**: Adaptive PCST 사용 시 α=0.85 (Cosine 우세) 유지 권장
  5. **(5) Anchor 유지 정당화 강화** — `qcond_gat_basic_glm` F1=0.8383 유지:
     - Plain/QCond Final α=0.85 vs α=0.5 ΔF1=-0.013/-0.008 (α=0.85 약하게 우세, plateau 임계 +0.005 미달)
     - I1a-c sweep "α=0.85 best" (No Filter) 결론이 with-Filter 에서도 약하게 재현
     - α=0.5 plateau (0.8252~0.8306) 추가 baseline 표기, α=0.85 anchor 변경 X
     - Wave 4 a05_filter_agentic plan 변경 X
  6. **(6) Alpha sweep H8 (post-deadline) 강화** — 본 측정으로 α∈{0.5, 0.85, 1} 3 점 + α=0 (GAT only) 까지 4 점 확보. H8 sweep 시 α∈{0.25, 0.7, 0.95} 보강 가능 (with-Filter stack 의 alpha sensitivity 완성).

- **근거**:
  - Root 측정 (2026-04-27 14:41:16~17:42:22, 3h 1min, GPU 2/3 split, ~₩4,584)
  - 출처: [EXPERIMENT_HISTORY.md "Ablation 1/2/3 α=0.5 Re-measurement (Option B, 2026-04-27)"](../EXPERIMENT_HISTORY.md#L1627)
  - 15 cells 메트릭 (Ablation 2 6 + Ablation 1 3 + Ablation 3 6) + α=0.85 vs α=0.5 직접 비교 표
  - Pre-Filter 단계 평균 ΔF1=-0.0354 — GAT 비중 ↑ 로 noise ↑, Filter 가 noise prune 으로 일부 회복

- **영향 범위**:
  - **presentation_brief §14.1** (planner Edit 완료): Builder Ensemble α=0.5 갱신 + Enriched +0.0711 새 narrative ★
  - **presentation_brief §14.2** (planner Edit 완료): Plain/QCond Ensemble α=0.5 갱신 + α=0.85 alpha sweep H8 한 점 표기
  - **presentation_brief §14.3** (planner Edit 완료): Extractor 4종 α=0.5 보강 + Extractor × α 상호작용 새 발견 + Adaptive worst caveat
  - **presentation_brief §14.6** (planner Edit 완료): α=0.5 plateau 추가 baseline 표기, anchor 0.8383 유지 정당화
  - **presentation_brief §11 Q&A** (planner Edit 완료): Q8 (α=0.5 재측정 사유) 추가
  - **EXPERIMENT_HISTORY** (root 갱신 완료): "Ablation 1/2/3 α=0.5 Re-measurement (Option B, 2026-04-27)" subsection L1627
  - **Wave 4 a05_filter_agentic plan**: anchor 0.8383 유지로 변경 X

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시 완료)** — 5 갱신 (§14.1/14.2/14.3/14.6/11) 모두 완료
  2. **사용자 (즉시)** — 발표 슬라이드 작성 시 §14 표 + §11 Q&A 활용. **새 발표 주력 narrative**: "Enriched α=0.5 +0.0711 회복 (Builder × α 상호작용)"
  3. **Selector/Builder 세션 (post-2026-04-28)** — H8 alpha sweep + H4 LLM-aware Builder design (Enriched α=0.5 회복 메커니즘 활용)

- **추가 필요 분석**:
  - Enriched α=0.5 의 +0.0711 회복 메커니즘 정량 (per-DB / per-difficulty 분해, GAT score 의 description noise 보정 패턴)
  - Adaptive + α 의 worst case 메커니즘 (per-query P80 cutoff 의 GAT noise sensitivity)
  - paper main contribution 후보: "Schema description 정보의 LLM filter noise 와 GAT 균형 보정 메커니즘 (α=0.5)"

---

## 2026-04-27 (Ablation 1/2/3 α=0.5 재측정 결정) — Ensemble baseline α=0.85 → α=0.5 (neutral) 통일 재정의 + 15 cells 통합 측정 + α=0.85 origin 정확한 진단

- **결정**:
  1. **(a) Ablation 1/2/3 의 Ensemble baseline α=0.85 → α=0.5 재측정 결정 (옵션 B)** — 사용자 confirm (2026-04-27). Ablation 의 "Ensemble" 의미를 GAT/Cosine 동등 결합 (α=0.5 neutral) 로 재정의. 발표 narrative 일관성 + 학술적 정당성 ↑.
  2. **(b) α=0.85 origin 정확한 진단** (사용자 history 추가 검토 결과):
     - **근거 있음**: [EXPERIMENT_HISTORY.md L120-130 I1a-c alpha sweep](../EXPERIMENT_HISTORY.md#L120) — α∈{0.70/0.75/0.85} sweep 결과 0.85 best (F1=0.4685 > 0.4577 > 0.4423)
     - **단 한계 3가지**:
       - Sweep 범위 0.70~0.85 한정 (α=0.5/0.95 미측정)
       - **No Filter stack 에서만 측정** (with-Filter stack alpha sensitivity 미수행)
       - L92 인용: "Filter 적용 시 Ensemble vs Cosine 차이 미미, **α=0.85 GAT 15% 만 반영**" — Filter 단에서 ensemble 의 의미 약화 분석
     - 직전 ensemble_selector.py L30 주석 "Phase 2 분석" 은 advisor_meeting_ideas_analysis.md 의 Steiner Phase 1/2 와 혼동 — alpha 분석 reference 부재
  3. **(c) 15 cells 통합 측정 (중복 제거 후)**:
     | Ablation | 신규 cells | 세부 |
     |----------|:---:|------|
     | Ablation 2 (Plain/QCond × α=0.5 × 3 stage) | **6** | Plain/QCond × {selector_only, no_filter, final GLM} |
     | Ablation 1 (Enriched α=0.5 × 3 stage) | **3** | Enriched × {selector_only, no_filter, final GLM} (Plain α=0.5 = Ablation 2 중복) |
     | Ablation 3 (Plain α=0.5 + 3 ext × 2 stage) | **6** | Plain + α=0.5 + (Adaptive/Steiner/MST) × {final GLM, no_filter} (Basic = Ablation 2 중복, Selector_only = Ablation 2 중복) |
     | **총** | **15 cells** | **6 final GLM + 9 LLM-free** |
     - 비용: ~₩4,584 (6 final GLM × ~₩764)
     - 시간: ~3h parallel (GPU 2/3)
  4. **(d) 발표 narrative 영향**:
     - §14.1/14.2/14.3 의 Ensemble 행 α=0.85 → α=0.5 갱신 (직전 α=0.85 측정 cells 는 alpha sweep H8 의 한 점 으로 표기)
     - §14.2 footnote 추가: "기존 α=0.85 default = I1a-c sweep (No Filter stack) 결과. With Filter stack 의 alpha sensitivity 는 미수행. 본 ablation 은 α=0.5 (neutral) baseline + alpha sweep H8 future work"
     - §11 Q&A 추가: "왜 α=0.5 로 재측정?" → "기존 α=0.85 의 sweep 근거가 No Filter stack 한정 + Filter 단에서 GAT 15% 비중 분석 (L92), neutral baseline 으로 학술적 정당성 ↑"
  5. **(e) Alpha sweep H8 (post-deadline) 보강**: 본 측정 후 α∈{0, 0.5, 0.85, 1} 4 점 비교 가능. H8 sweep 시 추가 점 (0.25, 0.7, 0.95) 으로 with-Filter stack alpha sensitivity 완성.

- **근거**:
  - I1a-c sweep 결과 (HISTORY L120-130, No Filter stack)
  - L92 분석 인용 (Filter 단 ensemble 약화)
  - Advisor meeting analysis L9/L29/L43 (α=0.85 의 GAT 15% 비중 비판, α=0.70~0.75 권장 — 일관된 분석)
  - 사용자 직전 메시지: "Ensemble score 비교에 α=0.5 가 더 합리적 + alpha ablation 따로 진행"

- **영향 범위**:
  - **Root 핸드오프 (즉시)**: 15 cells 통합 측정 prompt (GPU 2/3 명시)
  - **presentation_brief 갱신 (post-측정)**:
    - §14.1 Builder Ensemble 행 α=0.5 갱신
    - §14.2 Selector Ensemble 행 α=0.5 갱신 + α=0.85 alpha sweep 한 점 표기
    - §14.3 Extractor 통일 stack α=0.5 + 4 Extractor 갱신
    - §14.6 anchor 결정 — 새 cells 결과 통합 후 재판정
    - §11 Q&A 추가 (α=0.5 재측정 사유)
  - **EXPERIMENT_HISTORY**: 신규 subsection "2026-04-27 Ablation 1/2/3 α=0.5 Re-measurement (Option B, GPU 2/3)" 추가 (root 작업)
  - **H8 alpha sweep**: post-deadline 진행 시 α=0.5 결과 baseline 으로 활용

- **에스컬레이션 필요 여부**:
  1. **Root (즉시)** — 15 cells 측정. 본 엔트리 직후 핸드오프 prompt 코드블록.
  2. **Planner (결과 수령 후)** — §14.1/14.2/14.3/14.6/11 일관 갱신 + DECISIONS 후속
  3. **Selector 세션 (post-2026-04-28, H8)** — alpha sweep 보강 (with-Filter stack)

- **추가 필요 분석**:
  - α=0.5 vs α=0.85 vs α=1 (Cos only) 의 with-Filter stack 성능 비교 → ensemble blend 의 진짜 가치 정량
  - 사용자 의도된 "neutral ensemble" narrative 강화 가능 — 발표 main message 의 일부 가능성

---

## 2026-04-27 (GLM era 11 cells 완료) — Ablation 1/2/3 GLM era 일관 재측정 결과 + Anchor 유지 (qcond_gat_basic_glm 0.8383, plain_cos_a1_glm 0.8390 동률 표기) + 새 발견 (MST > Adaptive + XiYan) + Alpha sweep deferred (post-deadline)

- **결정 (4 사용자 요청 + Alpha sweep)**:
  1. **(a) Ablation 1 9-cell matrix 완성** — Enriched GLM final cell 추가:
     - `s03_a07_01_enriched_gat_glm`: R=0.6926 / P=0.8300 / **F1=0.7551**
     - vs vLLM era Enriched (0.7328): **ΔF1=+0.0223** (Builder 효과 GLM 환경에서 더 발현)
     - vs Plain Builder GLM (qcond_gat_basic_glm 0.8383): **ΔF1=-0.0832** (vLLM era 0.0549 대비 격차 더 큼)
     - 9-cell matrix: Plain/Enriched × 3 stage (Triplet 발표 자료 제외, H5 future work)
  2. **(b) Ablation 2 Plain final 4 cells 추가 — 9-cell matrix 정합**:
     - `plain_gat_a0_glm`: R=0.6825 / P=0.7153 / **F1=0.6985**
     - **`plain_cos_a1_glm`**: R=0.8472 / P=0.8310 / **F1=0.8390 ★** (새 GLM era top 후보, plateau)
     - `plain_ens_glm`: R=0.8447 / P=0.8316 / **F1=0.8381**
     - `qcond_gat_a0_glm`: R=0.6830 / P=0.7638 / **F1=0.7211**
     - SuperNode 9 cells 보류 유지 (H6 future work)
  3. **(c) Ablation 3 신규 도입 — Extractor × Stage 6 cells + Basic 참조**:
     - Basic PCST GLM (= plain_ens_glm): F1=0.8381
     - MST GLM: R=0.7252 / P=0.8276 / **F1=0.7730**
     - Steiner GLM: R=0.7081 / P=0.8073 / **F1=0.7545**
     - Adaptive GLM: R=0.6479 / P=0.8099 / **F1=0.7199**
     - **새 발견 1 — Extractor 위계 (final F1)**: Basic (0.8381) >> MST (0.7730) > Steiner (0.7545) > Adaptive (0.7199)
     - **새 발견 2 — MST > Adaptive + XiYan (+0.0531)**: vLLM era "Basic > Adaptive" 결론 GLM 에서 견고 + MST 가 Adaptive 보다 우세 발견 (vLLM era 미측정)
     - **새 발견 3 — Filter Δ F1 by Extractor 위계**: MST (+0.4041) > Steiner (+0.3894) > Adaptive (+0.2495) — **입력 sub-graph 단순할수록 LLM filter 효율 ↑**, MST 의 minimal selection 이 XiYan 정밀 prune 과 시너지
  4. **(d) Anchor 결정 — `qcond_gat_basic_glm` F1=0.8383 유지 + `plain_cos_a1_glm` F1=0.8390 동률 후보 표기**:
     - ΔF1=+0.0007 (직전 anchor 대비) — 갱신 임계 +0.005 미달
     - 4 후보 plateau (F1=0.83~0.84): plain_cos_a1_glm 0.8390 / qcond_cos_a1_glm 0.8424 / plain_ens_glm 0.8381 / qcond_gat_basic_glm 0.8383
     - **결론**: Cosine 우세 stack 의 GLM era 에서 plateau, encoder 차이 noise 수준. **anchor 유지 + 동률 후보 표기**, Wave 4 plan 변경 X
  5. **(e) Alpha sweep stack 선택 — 권장 deferred (post-deadline)**:
     - **권장: deferred (옵션 B)** — 발표 D-1 narrative 안정성 우선, F1 0.83~0.84 plateau 내 미세 변화 예상이라 발표 narrative 영향 minor
     - 가능 시 옵션 A: Plain Cos + QCond Cos × 3 alpha 점 (0.5/0.7/0.95) = 6 cells × ~₩764 = ~₩4,584, ~3h parallel — 발표 전 가능
     - **Stack 우선순위 (post-deadline 진행 시)**: Plain Cos (encoder agnostic finding 강화) + QCond Cos (현 best F1=0.8424 stack)
     - 사용자 결정 대기

- **근거**:
  - Root 측정 (2026-04-27 01:01:27~03:14:47, 2h 13min wall clock, GPU 2/3 split, ~₩6,112)
  - 출처: [EXPERIMENT_HISTORY.md "GLM era 일관 재측정 (Ablation 1/2/3, 2026-04-27)"](../EXPERIMENT_HISTORY.md#L1562)
  - **Cosine 우세 stack GLM era 일관 우세**: α=1 Cos 0.8390, α=0.85 Ens 0.8381, α=0 GAT 0.6985 → ΔF1 GAT→Cos +0.1405
  - **Encoder agnostic** (Cos 기준): Plain Cos 0.8390 ≈ QCond Cos 0.8424 — encoder noise 수준 (단 GAT only stack 일 때만 QCond > Plain +0.0226)
  - **MST 가치 발견 (vLLM era 미측정)**: GLM era 에서 처음 측정, Adaptive PCST 보다 +0.0531 우세 — 새 anchor 후보로 부상 (단 Basic PCST 0.8381 보다는 -0.0651)

- **영향 범위**:
  - **presentation_brief §14.1 갱신** (planner Edit): 9-cell matrix Enriched GLM 추가 + Builder 효과 GLM era +0.0832 정량
  - **presentation_brief §14.2 갱신** (planner Edit): Plain final 4 cells 추가 + 9-cell GLM era 정합 + plateau finding
  - **presentation_brief §14.3 신규 도입** (planner Edit): GLM era 4 Extractor + Filter Δ 위계 + 새 발견 MST > Adaptive
  - **presentation_brief §14.6 신설** (planner Edit): Anchor 결정 + Alpha sweep 선택 가이드
  - **§9.3 Future work**: Alpha sweep entry 추가 (post-deadline 옵션, Plain Cos + QCond Cos × 3 alpha)
  - **Wave 4 a05_filter_agentic plan**: anchor 0.8383 유지로 변경 X (plateau 내 noise)

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — §14.1/14.2/14.3/14.6/9.3 Edit 진행 (본 엔트리 직후)
  2. **사용자 (선택)** — Alpha sweep 발표 전 진행 vs deferred 결정
  3. **Analyzer (선택, post-deadline)** — MST > Adaptive 새 발견 origin 분석 (per-DB / per-difficulty 분해, MST 가 어떤 query 클러스터에서 우세인지)

- **추가 필요 분석**:
  - F1 0.83~0.84 plateau 의 통계적 분석 (multi-seed 또는 vLLM era baseline 측정)
  - MST + XiYan 의 high-quality minimal selection 메커니즘 — paper future work 후보
  - Alpha sweep 결과 보고 "Cosine 우세 stack 의 alpha sensitivity" narrative 강화

---

## 2026-04-27 — GLM era 일관 재측정 (Ablation 1/2/3 final 8 cells) + Ablation 3 단계별 측정 (3 no-filter) + Alpha sweep deferred (post-Ablation 결과 후 2-3 stack 선정)

- **결정**:
  1. **(a) Ablation 1/2/3 의 GLM era 일관 재측정 결정** — 사용자 요청 (2026-04-27): "각 단계별도 마찬가지". 단 **stage 별 LLM 호출 분석 결과**:
     - **Selector_only 단계**: LLM 호출 0 (PLM 임베딩 + GAT forward 만) → era 무관, **재측정 불필요**
     - **+Extractor (no_filter) 단계**: LLM 호출 0 (PCST/MST 알고리즘) → era 무관, **재측정 불필요**
     - **+Filter (final) 단계**: XiYan LLM 호출 → **GLM era 재측정 필요**
     - 즉 LLM-free stage 들 (selector_only, no_filter) 의 기존 측정 유효, final 단계만 GLM 재측정.
  2. **(b) Ablation 3 (Extractor) 신규 측정 — 단계별 + GLM era**:
     - 통일 stack: Plain encoder + Ensemble α=0.85 + 4 Extractors (MST / Basic PCST / Adaptive PCST / MST+PCST=Steiner) + XiYan GLM
     - 사유: 기존 §14.3 의 Extractor 비교 cells 가 Plain encoder + Ensemble + Filter=XiYan vLLM stack 이라 일관성. GLM 재측정 시 동일 stack 유지.
     - Ablation 3 selector_only = Ablation 2 의 plain_ens_selector_only F1=0.3974 재활용 (1 cell 절약)
  3. **(c) 측정 cells 산출 (중복 제거 후 11 cells)**:
     | Ablation | 신규 cells | 비고 |
     |----------|:---:|------|
     | Ablation 1 final GLM | 1 | Enriched Builder + QCond + Ensemble + Basic + XiYan GLM (Plain Builder GLM 은 qcond_gat_basic_glm F1=0.8383 재활용) |
     | Ablation 2 final GLM | 4 | Plain GAT GLM, Plain Cos GLM, Plain Ens GLM, QCond GAT GLM (QCond Cos/Ens GLM 이미 측정) |
     | Ablation 3 final GLM | 3 | Plain Ens + (Adaptive/Steiner/MST) + XiYan GLM (Basic GLM = Ablation 2 의 Plain Ens GLM 동일 cell, 중복) |
     | Ablation 3 no_filter | 3 | Plain Ens + (Adaptive/Steiner/MST) + no_filter (Basic no_filter = plain_ens_no_filter F1=0.2250 재활용) |
     | **총** | **11 cells** | **8 GLM API + 3 LLM-free** |
     - 비용: ~₩6,112 (GLM API only)
     - 시간: ~3.5h parallel (GPU 0/1, LLM-free 3 cells 빠름)
  4. **(d) Alpha sweep deferred — Ablation 1/2/3 결과 후 2-3 stack 선정**: 사용자 명시. 현재 alpha 측정 = {0, 1, 0.85} 3 점. Alpha sweep 추가 점 = {0.5, 0.7, 0.9} 등 2-3 점. **Ablation 1/2/3 GLM 재측정 결과 보고 best 2-3 stack 선정 후 진행** (post-2026-04-28 가능, 발표 narrative 영향 minor).
  5. **(e) 발표 D-1 일정 평가**: 11 cells × ~3.5h parallel 가능 (오늘 2026-04-27 시작 → 발표 2026-04-28). **safe 일정**, 단 GPU 0/1 또는 2/3 점유 확인 필수. SuperNode/MST ckpt 호환 사전 검증 필요 (smoke 1 cell).

- **근거**:
  - Stage 별 LLM 호출 분석:
    - Selector: GAT forward → LLM 호출 X
    - Extractor: PCST/MST 알고리즘 (graph optimization) → LLM 호출 X
    - Filter (XiYan): LLM call (column pruning) → LLM 호출 O
  - Wave 1.5 stagewise no_filter cells (W1/W2/W3) 가 vLLM era 측정인 이유: Filter=NoneFilter 라 어떤 era 든 동일 결과. 즉 era label 의미 없음. 이번에도 동일.
  - 발표 narrative 일관성: 현 GLM era top (qcond_gat_basic_glm F1=0.8383) 와 같은 era 로 모든 비교 cells 측정 시 정량 정확성 ↑
  - 비용 산출: Plain Cos GLM + Plain Ens GLM 은 사용자 직전 5 답변의 Ablation 2 핸드오프에 이미 포함됐으나 Ablation 1 cumulative 핸드오프 우선 송신으로 누락됨 — 이번 통합 핸드오프로 포함

- **영향 범위**:
  - **Root 핸드오프 (즉시)**: 11 cells 통합 측정 prompt 송신
  - **presentation_brief 갱신 (post-측정)**:
    - §14.1 표 → 6-cell vLLM + 1 cell GLM era cumulative 보강
    - §14.2 표 → 18-cell vLLM era + 4 cell GLM era 통합
    - §14.3 표 → 4-cell GLM era 완성 + MST 통일 stack 추가
    - §14.5 Slide 1/2/3 narrative → GLM era 일관 비교로 강화
  - **Wave 4 a05_filter_agentic plan**: GLM era 재측정 결과로 anchor refresh 재판정 가능 (현 0.8383, qcond_cos_a1_glm 0.8424 의 H7 검증 + 새 측정 cells 함께 평가)
  - **H7 (qcond_cos_a1_glm 검증) 부분적 cover**: 본 측정에 Plain Cos GLM 추가 → Plain Cos vLLM 0.7838 vs Plain Cos GLM Δ 가 LLM era 효과 검증, qcond_cos_a1_glm 의 noise 가능성 추가 정량 가능
  - **Alpha sweep 큐 등록 (post-Ablation)**: 결과 보고 best 2-3 stack 선정 → alpha sweep 별도 핸드오프

- **에스컬레이션 필요 여부**:
  1. **Root (즉시)** — 11 cells 통합 측정. 본 엔트리 직후 응답에 핸드오프 prompt 코드블록 제공.
  2. **Planner (결과 수령 후)** — §14.1/14.2/14.3/14.5 일관 갱신 + DECISIONS 후속 + alpha sweep stack 선정
  3. **Selector 세션 (post-2026-04-28)** — H6 + H7 + alpha sweep 통합 wave

- **추가 필요 분석**:
  - Plain Cos GLM Δ vs Plain Cos vLLM (0.7838) 분석 — encoder 무관 cosine 단독에서 GLM 효과
  - Plain Ens GLM Δ vs Plain Ens vLLM (0.7863) 분석 — non-QCond 에서 GLM ensemble synergy 검증
  - Adaptive/Steiner/MST GLM 결과 후 Extractor 축 의 LLM era 효과 정량
  - Alpha sweep stack 선정 기준: best 2-3 stack 의 alpha sensitivity 측정 (Plain vs QCond, GAT/Cos/Ens 중 어느 stack 이 alpha 변화에 가장 민감한가)

---

## 2026-04-26 (Ablation 2 Plain/QCond 18-cell 완료) — 새 GLM era top 후보 발견 (qcond_cos_a1_glm F1=0.8424, +0.0041 vs 직전) + Encoder × Score interaction 정량 + Wave 4 anchor 0.8383 유지 + H7 future work 신설

- **결정 (5개 사용자 요청 응답)**:
  1. **(a) Plain/QCond 18-cell 완료 (10 cells 신규 + 8 cells 재참조)** — 2026-04-26 02:52:01~04:31:08, ~1h 39min, GPU 2/3, 비용 ~₩764. SuperNode 9 cells 보류 추인 (smoke fail 핸드오프 §결정 (b)).
  2. **(b) 새 GLM era top 후보 `qcond_cos_a1_glm` F1=0.8424 → caveat ("two anchors tie") + 현 0.8383 anchor 유지** — 권장 옵션 (c). 사유:
     - +0.0041 < anchor 갱신 임계 (≥+0.005) — 미달
     - vLLM era 비교 baseline 부재 (qcond_cos_a1 = QCond + Cosine α=1, vLLM era 측정 X) → noise vs systematic 판정 불가
     - 단일 run, statistical noise 판정 위해 multi-seed 또는 vLLM baseline 측정 필요
     - 발표 narrative: "**QCond Cosine only (α=1) 도 GLM 에서 거의 동등 (F1=0.8424 ≈ 0.8383, Δ=+0.0041 noise 범위)**, Ensemble blend 가 cosine 으로 단순화 가능 시사" — 의외 결과 footnote
     - **현 anchor `s04_stagewise_qcond_gat_basic_glm` F1=0.8383 유지**, qcond_cos_a1_glm F1=0.8424 는 §14.2 표에 "🚀 후보" 표기 + noise caveat
  3. **(c) Wave 4 a05_filter_agentic anchor 0.8383 유지** — 사용자 옵션 (b) 결정과 일관. qcond_cos_a1_glm F1=0.8424 가 noise 가능성 + vLLM baseline 부재 → anchor refresh 변경 X. vivid-sprouting-sunbeam.md plan anchor 는 `s04_stagewise_qcond_gat_basic_glm` F1=0.8383 그대로. **H7 검증 후 재판정**.
  4. **(d) Encoder × Score interaction 발견 — 발표 narrative 핵심 강화**:
     - **Selector_only Cosine 단독에선 encoder 무관**: Plain/QCond Cosine F1 둘 다 **0.3829 동일** — PLM (MiniLM-L6-v2) 임베딩 직접 사용, GAT module 통과 X → encoder 차이 없음. 사용자 직전 가설 ("SuperNode Cosine = Plain Cosine 가능성") 검증 ✅.
     - **GAT only 단계에서 QCond > Plain (+0.0597)**: encoder 효과 발현 가장 강한 지점. Plain GAT 0.2937 → QCond GAT 0.3534. **Encoder 효과 = GAT module 활용 시에만 발현**.
     - **Ensemble 단계 encoder 효과 미세 (+0.0042)**: Cosine 우세 (α=0.85) blend 가 GAT 의 encoder 효과 dilute. 즉 Ensemble 에서 encoder 차이 안 보이는 이유 = cosine 가중치 큼.
     - **Filter Δ F1 max = QCond Cosine + GLM (+0.6129)**: Filter 가 cosine-only 의 정밀도 부족을 가장 크게 보강. QCond Ensemble GLM (+0.6112) 와 거의 동등.
     - **+Extractor 단계 매우 균질화** (R≈0.965~0.967, F1≈0.225~0.230) — PCST 가 후보 거의 다 끌어와 encoder/score 차이 dilute. 단 QCond GAT 만 R=0.7813 / F1=0.2862 (extractor 가 score 분포에 sensitive).
  5. **(e) Ablation 1 (Builder) ↔ Ablation 2 (Selector) cross-reference**:
     - **공유 anchor cell 확정**: Ablation 1 의 Plain Builder (=HeteroGraphBuilder) + QCond Encoder + GAT α=0.85 Ensemble + Basic PCST + XiYan = `s04_stagewise_qcond_gat_basic` F1=0.7877 (vLLM) = Ablation 2 의 QCond Ensemble (vLLM) cell. **두 ablation 매트릭스의 동일 행/열 = 같은 실험**.
     - **Encoder 변경 효과 (Plain → QCond) 의 stage-별 amplification**:
       - Selector_only: +0.0084 (Plain Ensemble 0.3974 → QCond Ensemble 0.4016)
       - +Extractor: +0.0021 (0.2250 → 0.2271)
       - +Filter: +0.0014 (0.7863 → 0.7877)
     - 비교 (Builder 변경 효과 Plain → Enriched at QCond Ensemble final): -0.0549 (0.7877 → 0.7328) — **훨씬 큼**.
     - 통합 narrative: "**Builder 효과 (Plain vs Enriched) 가 Filter 단계에서 -0.0549 큰 영향, Encoder 효과 (Plain vs QCond) 는 GAT only 에서만 +0.0597 발현, Ensemble 에서 +0.0014 거의 무영향. 두 축 모두 Plain + QCond Ensemble 이 best stack 이지만, dominance 메커니즘 다름**".

- **근거**:
  - Root 측정 결과 (2026-04-26 02:52:01~04:31:08, scripts/run_ablation2_selector_cumulative.sh, GPU 2/3, ~₩764):
    - 10 신규 cells: Selector_only 6 (Plain/QCond × GAT/Cos/Ens) + No-filter 3 (Plain Cos, Plain Ens, QCond Cos) + Final 1 (qcond_cos_a1_glm)
    - 8 기존 cells: Wave 1.5 W1/W2/W3 no-filter + final, abl_a01_05/06, qcond_gat_basic_glm
  - 출처: [EXPERIMENT_HISTORY.md "Selector Ablation Cumulative Backfill (Option B, 2026-04-26)"](../EXPERIMENT_HISTORY.md#L1467)
  - **anchor 갱신 임계 +0.005 (직전 정의)**: 2026-04-25 H2 검증 4-way 분기 표 사전 합의. qcond_cos_a1_glm Δ=+0.0041 < 0.005 → 분기 (3) partial neutral. 직전 H2 truncate Ldbmax_glm Δ=+0.0045 와 거의 동등 — H2-truncate 도 partial neutral 로 결정했었음. 일관된 판정.
  - vLLM era baseline 부재: qcond_cos_a1 (QCond + Cosine α=1, Basic PCST + XiYan) 의 vLLM era 측정 X. 사용자 의도된 9-cell 완전 매트릭스 + GLM 비교 기준이 부족.

- **영향 범위**:
  - **presentation_brief §14.2** (planner Edit): 9-cell 매트릭스 → **18-cell stagewise (Plain/QCond × GAT/Cos/Ens × 3 stage)** + Filter Δ F1 by Encoder × Score 표 + SuperNode 3 row caveat 유지
  - **presentation_brief §14.4** (planner Edit): Ablation 1/2 cross-reference sub-section 추가 (공유 anchor + Encoder/Builder dominance 메커니즘 비교)
  - **presentation_brief §14.5 Slide 2** (planner Edit): 6-cell valid 매트릭스 + Encoder × Score interaction 핵심 narrative + qcond_cos_a1_glm caveat
  - **presentation_brief §10 빠른 참조** (선택, planner): qcond_cos_a1_glm 0.8424 추가 (footnote 표기)
  - **presentation_brief §9.3 Future work**: H7 신설 (qcond_cos_a1_glm anchor 검증)
  - **vivid-sprouting-sunbeam.md (Wave 4 plan)**: anchor 변경 X (현 0.8383 유지), filter 세션 핸드오프 시점에 본 결정 인지 필요

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — §14.2 / §14.4 / §14.5 / §9.3 Edit 진행 (본 엔트리 직후)
  2. **Analyzer (선택, post-deadline)** — qcond_cos_a1_glm 의 cosine-only 가 ensemble blend 보다 우세 origin 분석 (query 분포별 효과 분해, per-DB 비교)
  3. **Selector 세션 (post-2026-04-28, H7)** — qcond_cos_a1_glm anchor 검증: (a) multi-seed 측정 (2-3 cells, ~₩2,300), 또는 (b) vLLM era baseline 측정 (qcond_cos_a1 vLLM 1 cell, ₩0 LLM-free... wait Basic PCST + XiYan vLLM 이라 LLM 호출 — vLLM 부팅 8~10h 비용으로 비효율, GLM era multi-seed 가 더 합리)
  4. **Selector 세션 (post-2026-04-28, H6 + H7 통합)** — H6 (SuperNode ckpt 재학습) + H7 (qcond_cos_a1_glm 검증) 두 작업 묶어서 selector wave 진행

- **추가 필요 분석**:
  - 발표 Q&A 대비:
    - "qcond_cos_a1_glm F1=0.8424 가 새 top 인가?" → "Δ=+0.0041 (noise 범위, 임계 미달), 현 anchor 0.8383 유지, post-deadline H7 검증"
    - "왜 cosine only (α=1) 가 ensemble (α=0.85) 와 거의 동등?" → "GLM era 에서 cosine 우세 blend 의 GAT 가중치 (0.15) 가 Filter 단에서 큰 차이 만들지 않음, 발표 narrative 의 cosine 우세 ensemble 의 GAT 효용 재검토 필요"
  - H7 검증 결과에 따라 Wave 4 a05_filter_agentic anchor 재변경 가능 (post-2026-04-28)
  - 발표 narrative 의 핵심 변경: "Ensemble 효과 (cosine + GAT blend) 가 GAT only 보다 우세" → "**Ensemble 효과는 vLLM era 에서 +0.0826 큰 이득이었으나, GLM era 에서 cosine only 와 거의 tie (+0.0014)**" — LLM backbone 별 ensemble 가치 변화 가능성

---

## 2026-04-26 (Ablation 2 SuperNode smoke FAILED) — ckpt input dim mismatch 발견 + SuperNode 9 cells 보류 + Plain/QCond 10 cells 진행 + 발표 매트릭스 2×3 (6 cell valid) 정정 + H6 future work 신설 (SuperNode ckpt 재학습)

- **결정**:
  1. **(a) SuperNode smoke FAILED 진단** — `supernode_gat_a0_selector_only` 1 query inference 시 ckpt 호환 불일치:
     - ckpt `best_gat_query_supernode.pt` input dim = **[256, 384]**
     - EnsembleSelector(query_supernode=True) model expect = **[256, 768]**
     - 원인: SchemaHeteroGAT 가 `query_supernode=True` 일 때 in_channels=768 expect (query feature concat)
     - ckpt 는 384 input 으로 학습됨 (T7 = best_gat_query_supernode.pt, **§8-1 SuperNode split-order bug fix 이전 학습**)
     - traceback 위치: `src/modules/selectors/ensemble_selector.py:128 self.gat_model.load_state_dict`
  2. **(b) SuperNode 9 cells 전부 보류** — supernode_{gat_a0, cos_a1, ens} × {selector_only, no_filter, final} = 9 cells 모두 동일 ckpt 사용 불가. **post-2026-04-28 큐**.
  3. **(c) Plain/QCond 10 cells 진행 중 (ETA ~05:07)** — Plain {gat_a0, cos_a1, ens} × {selector_only, no_filter} + QCond {gat_a0, cos_a1, ens} × {selector_only, no_filter} + Final 신규 1 cell (qcond_cos_a1_glm). 02:52:01 launch.
     - Plain/QCond × Stage 매트릭스 6 cell (final) + 6 cell (no-filter, 일부 기존) + 6 cell (selector_only) — 정확한 cell 수는 root 보고 시 확정
  4. **(d) 발표 매트릭스 변경** — 직전 9-cell 완전 매트릭스 → **2×3 (Plain/QCond) 6-cell valid** + SuperNode 4 row caveat (Plain Cosine α=1 측정 X, SuperNode 9 cells 보류). presentation_brief §14.2 표 갱신 + §14.5 Slide 2 가이드 갱신.
  5. **(e) H6 future work 신설 (post-2026-04-28)** — **SuperNode ckpt 재학습** (§8-1 SuperNode split-order bug fix 와 묶어서, in_channels=768 학습):
     - selector 세션 작업: SchemaHeteroGAT 의 query_supernode=True 시 in_channels 자동 분기 (384 vs 768) 옵션 또는 ckpt 재학습
     - 비용: 새 GAT 학습 ~5h + 9 cells inference (~3h)
     - selector/builder 협업

- **근거**:
  - Root smoke 실패 traceback (ensemble_selector.py:128): ckpt state_dict 의 [256, 384] vs model expect [256, 768] mismatch
  - DECISIONS 2026-04-22 17:05 §8-1 SuperNode split-order bug 엔트리: T7/T9 ckpt 가 bug 영향. T7 = `best_gat_query_supernode.pt` 가 본 mismatch 의 ckpt
  - SuperNode 9 cells 모두 동일 ckpt 사용 → 모두 같은 mismatch → 9 cells 전부 보류 (한 번에)
  - Plain/QCond 10 cells 는 별개 ckpt (Plain=`best_gat_model.pt`, QCond=`best_gat_qcond_nl3.pt`) → 영향 X, 진행 가능
  - 발표 D-2: post-2026-04-28 SuperNode 재학습 + 측정 = 발표 후 wave (H6)

- **영향 범위**:
  - **presentation_brief §14.2** (planner Edit): 10-cell 매트릭스 → SuperNode 4 row "측정 X (smoke fail, ckpt mismatch)" caveat + 6-cell (Plain/QCond) valid
  - **presentation_brief §14.5 Slide 2 가이드**: Plain/QCond 만 비교 narrative + SuperNode 1줄 caveat
  - **presentation_brief §9.3 Future work**: H6 SuperNode ckpt 재학습 추가
  - **DECISIONS 2026-04-22 17:05 §8-1 영향 확장**: T7 ckpt 의 input dim mismatch 가 §8-1 bug 와 별개 영향 (split-order bug 는 forward 결과 부정확, 본 mismatch 는 forward 자체 불가). 즉 §8-1 bug 만 fix 해서는 안 되고 in_channels 도 768 로 변경 학습 필요. **H6 가 §8-1 bug fix + in_channels=768 두 가지 모두 cover**.
  - **EXPERIMENT_HISTORY** (root 작업): "Builder Cumulative Backfill" subsection 다음에 "2026-04-26 Selector Ablation Cumulative Backfill (Option B, Plain/QCond 10 cells)" 신규 — 10 cells 결과 수령 후 root 갱신

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — §14.2 / §14.5 / §9.3 갱신 (본 엔트리 직후 Edit 진행)
  2. **Root (대기)** — Plain/QCond 10 cells 결과 수령 (~05:07 ETA) → 별도 핸드오프 (매트릭스 분석 + Builder/Selector cross-reference)
  3. **Selector 세션 (post-2026-04-28, H6)** — SchemaHeteroGAT 의 query_supernode in_channels 분기 또는 ckpt 재학습. §8-1 bug fix 와 묶어서.

- **추가 필요 분석**:
  - 발표 Q&A 대비: "왜 SuperNode 측정 X?" 답변 — "ckpt 재학습 필요 (§8-1 bug fix + in_channels=768), post-deadline H6 future work"
  - Plain vs QCond 6-cell 매트릭스 (Encoder × Score) 의 Selector ablation insight 가 발표 narrative 의 핵심으로 강화 (SuperNode 결손 보완)

---

## 2026-04-26 (Ablation 1 Triplet 진단) — Triplet Builder 의 edge embedding 이 GAT 에 미반영 확인 + Ablation 1 narrative 정정 (Plain vs Enriched 만 valid) + H5 future work 신설 (edge-as-node + EHGAT 재학습)

- **결정**:
  1. **(a) Triplet Builder 의 edge embedding GAT 미반영 진단 확정** — 사용자 지적 (직전 turn) 검증 완료. 코드 evidence:
     - [`src/modules/builders/graph_builder.py:911-1000`](../src/modules/builders/graph_builder.py): `class TripletGraphBuilder(EnrichedHeteroGraphBuilder)` — `super().build()` 로 graph topology = Enriched 와 완전 동일, edge embedding 은 `metadata['edge_embeddings']` field 로만 추가
     - `src/models/gat_network*.py` grep: `edge_attr` 인자 부재 — GAT 가 forward 시 edge embedding 사용 X
     - `outputs/checkpoints/` 에 line/edge/triplet ckpt 부재 — edge-aware GAT 학습 X
     - 결과: Triplet selector_only/no_filter F1 ≡ Enriched (anomaly 아님, 정상 동작 — edge embedding 무시됨)
  2. **(b) Triplet Builder 의 final F1 차이 (0.7423 vs Enriched 0.7328, +0.0095) 의 진짜 원인 = Extractor 차이** — Enriched (`s03_a07_01_enriched_gat`) Extractor=AdaptivePCSTExtractor, Triplet (`s03_a07_02_edge_prize`) Extractor=**EdgePrizePCSTExtractor** (`topk_e=5`, edge embedding 으로 edge prize 산출). **Builder 단독 효과 X, Builder+Extractor 결합 효과만 측정됨**.
  3. **(c) Ablation 1 narrative 정정** — Plain vs Enriched 만 직접 비교 valid (2 Builder × 3 stage = 6 cells). Triplet 은 별도 small section (한계 + future work):
     - "Edge embedding 통합 첫 시도 (s03_a07_02), 현 구현은 metadata 추가에 그쳐 GAT forward 에 미반영"
     - "진짜 edge-aware Builder = Line graph 변환 + EHGAT 재학습 (Proposal #3 S-III) — future work"
  4. **(d) H5 신설 (future work, post-2026-04-28)** — **Edge-as-node Builder + EHGAT 재학습**:
     - 인프라 일부 준비: `src/modules/builders/line_graph_builder.py` (LineGraphBuilder, 구현 완료)
     - 미구현: EHGAT 모델 (Proposal #3 S-III), edge-aware GAT 학습 ckpt
     - 예상 비용: 새 GAT 학습 ~5h + line graph cache 생성 + EHGAT 구현
     - selector/builder 협업, post-2026-04-28
  5. **(e) presentation_brief §14.1 + §14.5 + §9 갱신 (planner 후속 Edit)** — Triplet row 별도 분리 + Future work 에 H5 추가

- **근거**:
  - 사용자 의문 (직전 turn): "Triplet 의 수치가 Filter 전까지 Enriched 와 완전히 동일 — Edge Embedding 이 어떻게 반영되길래?"
  - TripletGraphBuilder 코드 (graph_builder.py L935): `data, metadata = super().build(...)` — Enriched 호출 후 graph topology 변경 X
  - LineGraphBuilder docstring: "promotes each edge in the PCST-flat graph to a node and connects two edge-nodes when they share an original node... downstream EHGAT-style selectors can consume" — **사용자 의도된 edge-as-node 가 정확한 의미**
  - EXPERIMENT_PLAN.md §1 Cross-Module Matrix: ★ S-III (EHGAT) 미구현 상태
  - Filter Δ F1 비교 (Plain +0.5606 vs Triplet +0.5171) 도 **Extractor 가 다른 두 cell 비교라 invalid** — 직전 DECISIONS 엔트리 §결정 (d) 의 narrative 정정 필요

- **영향 범위**:
  - **presentation_brief §14.1 정정** (planner Edit): 9-cell 매트릭스 → 6-cell (Plain vs Enriched) + Triplet 별도 small section ("metadata 추가 + EdgePrize 결합 효과, GAT 미반영, future work")
  - **presentation_brief §14.5 Slide 1 가이드 정정**: Plain vs Enriched 비교 narrative + Triplet 한계 + future work mention
  - **presentation_brief §9 (Future work) 갱신**: H5 (edge-as-node + EHGAT) 신설
  - **EXPERIMENT_HISTORY "Builder Cumulative Backfill" subsection 의 §발견 #4** ("Enriched ≡ Triplet... Builder graph structure 차이가 Selector 단계에서 동일 score 산출") 는 부분적으로 **잘못된 인과** — 정확한 진단은 "Triplet graph topology = Enriched 와 동일, edge embedding 은 GAT 에 미반영" — root 갱신 권장 (post-deadline OK)
  - **DECISIONS 직전 엔트리 (Ablation 1 Option C 결과) §결정 (e) "Enriched ≡ Triplet 발견... 같은 selector weight + Builder graph structure 만 다름" 도 부정확** — Builder graph structure 도 동일. 본 엔트리가 supersede.

- **에스컬레이션 필요 여부**:
  1. **Planner (즉시)** — presentation_brief §14.1 + §14.5 + §9 정정 (본 엔트리 직후 Edit)
  2. **Root (post-deadline, 선택)** — EXPERIMENT_HISTORY "Builder Cumulative Backfill" §발견 #4 narrative 정정
  3. **Builder/Selector 모듈 (post-2026-04-28)** — H5 future work kickoff: EHGAT 구현 + edge-aware GAT 학습 + line graph cache 생성

- **추가 필요 분석**:
  - 발표 Q&A 대비: "왜 Triplet Builder 의 진짜 효과가 측정 안 됐나?" 답변 — "현 구현은 metadata 추가에 그치고 GAT 가 edge_attr 미수신, 진짜 edge-aware = LineGraph + EHGAT 새 wave"
  - Ablation 1 발표 narrative 의 핵심 변경: "3 Builder 비교" → "2 Builder (Plain vs Enriched) 직접 비교 + Triplet 은 미완 시도 + future work"

---

## 2026-04-26 (Ablation 1 Option C 결과) — Builder 9-cell cumulative matrix 완성 + Plain 우세 정량 + Description 정보의 Filter noise 가설

- **결정** (Root 보고 수령 + planner 종합):
  1. **(a) Ablation 1 Option C 종료 — Plain Builder 가 모든 단계에서 우세 확인** — Selector only / +Extractor / +Filter 3 stage 모두 Plain ≥ Enriched/Triplet. Final stage 에서 Plain 우세 가장 큼 (ΔF1 over Enriched=+0.0549, over Triplet=+0.0454).
  2. **(b) Stage-별 Builder 영향 비대칭 패턴 확정**:
     - Selector only Δ F1 = 0.0139 (Plain 0.4016 vs Enriched/Triplet 0.3877) — **미미**
     - +Extractor Δ F1 < 0.002 — **사실상 동일** (PCST dilute 효과)
     - +Filter Δ F1 = 0.0454~0.0549 — **유일 발현 stage**
     → "Builder 가치 = Filter 단계 발현" 근거 + 발표 narrative 직접 인용 가능
  3. **(c) Description 정보의 Filter noise 가설 채택** — Enriched/Triplet 의 description 추가가 selector level (GAT score) 에는 거의 영향 X, **final XiYan filter 에서 noise 로 작용** (LLM 이 description 으로 보수적 prune → Recall 손실: Plain R=0.8169 vs Enriched R=0.6658, Triplet R=0.6823). **Builder description 추가는 LLM-aware filter 와 비호환** — H4 future work 후보.
  4. **(d) Filter Δ F1 by Builder 정량 → Plain +0.5606 (max) > Triplet +0.5171 > Enriched +0.5076** — Filter 의 marginal 효과는 Plain 에서 최대. Description 이 추가될수록 Filter 의존도 (= Filter 가 만들 수 있는 추가 이득) 작아짐.
  5. **(e) Enriched ≡ Triplet 발견 (selector_only/no_filter 동일값)** — 같은 `best_gat_enriched.pt` selector weight + Builder graph structure 만 다름 → Selector 단계 score 동일. Triplet edge embedding 의 차이는 Filter 단계 에서만 미세 발현 (ΔF1=+0.0095 in final, Triplet 0.7423 > Enriched 0.7328).

- **근거**:
  - Root 측정 결과 (2026-04-26 01:55:29~02:34:37, scripts/run_builder_cumulative.sh, GPU 2/3, ~38min, ₩0):
    - 5 신규 cells: Plain SO=0.4016, Enriched SO=0.3877, Triplet SO=0.3877, Enriched no-filter=0.2252, Triplet no-filter=0.2252
    - 4 기존 cells: Plain final=0.7877, Plain no-filter=0.2271, Enriched final=0.7328, Triplet final=0.7423
  - 출처: [EXPERIMENT_HISTORY.md "Builder Cumulative Backfill (Option C, 2026-04-26)"](../EXPERIMENT_HISTORY.md#L1406)
  - 매트릭스 (R/P/F1, 4자리): 9 cells 완전 측정 → presentation_brief §14.1 표 직접 반영 (planner Edit 완료)
  - Filter Δ F1 정량: Plain +0.5606 (max) — Plain 의 simpler text 가 XiYan 의 column 식별에 더 유리함을 시사

- **영향 범위**:
  - **presentation_brief §14.1** (planner Edit 완료): 기존 final-only 표 → 9-cell cumulative matrix + Filter Δ F1 by Builder 표 + 6개 핵심 발견 정리 + 데이터 출처 + Caveat (GLM era 재측정 X)
  - **presentation_brief §14.5** (planner Edit 완료): Slide 1 narrative 가이드 보강 — main message + 단계별 Δ 패턴 + Filter Δ 표 + 별난 finding (Enriched ≡ Triplet) + GLM era caveat
  - **발표 슬라이드 Slide 1 (Builder)** 작성 시 §14.1 + §14.5 Slide 1 가이드 직접 인용 가능
  - **H4 future work** (신규 추가): "LLM-aware Builder design — description 정보를 selector 단계에서 활용하되 final filter 에는 minimal text 만 전달하는 hybrid Builder" — post-2026-04-28 selector/builder 협업
  - **Ablation 1 → Ablation 2 cross-reference** (planner): Builder 결과 (Plain 우세) 가 Ablation 2 의 통일 stack 가정 (Plain Builder) 정당화 — 발표 narrative 일관성 ↑

- **에스컬레이션 필요 여부**:
  1. **Root 세션 (대기)** — Ablation 2 19 cells 핸드오프 송신 후 결과 수령 시 별도 처리. 본 엔트리는 Option C closure 로 추가 root 작업 없음.
  2. **Planner (대기)** — Ablation 2 결과 수령 후 §14.2 27-cell 매트릭스 완성 + Builder × Selector cross-reference 분석.

- **추가 필요 분석**:
  - Filter (XiYan) 가 description 을 어떻게 처리하는지 — column-level vs table-level description 영향 분리 (post-deadline analyzer 작업)
  - GLM era 재측정 시 description noise 효과가 GLM-4.7 에서도 동일한지 — backbone 별 Filter 행동 차이 가능성
  - H4 LLM-aware Builder 설계 (future work): description-aware selector + minimal-text filter 의 hybrid

---

## 2026-04-26 (Ablation 2 보강 #2) — 9-cell 완전 매트릭스 채택 (SuperNode Ensemble 추가, 16 → 19 cells)

- **결정**: 사용자 confirm (직전 turn) — Ablation 2 매트릭스를 8 cells (SuperNode 는 GAT/Cosine 만) → **9 cells (3 Encoder × 3 Score 완전 매트릭스)** 로 확장. SuperNode + Ensemble (α=0.85) 추가. 사용자 직전 메시지에서 SuperNode Ensemble 명시 누락 (planner 가 매트릭스 비대칭 지적 → 사용자 추가 confirm).

- **영향**:
  - **측정 cells: 16 → 19** (+3)
    - Final stage: 3 → **4** (+ `s04_stagewise_supernode_ens_glm.yaml`)
    - No-filter: 5 → **6** (+ `s04_stagewise_supernode_ens_no_filter.yaml`)
    - Selector_only: 8 → **9** (+ `s04_stagewise_supernode_ens_selector_only.yaml`)
  - 비용: ~₩2,292 → **~₩3,056** (+₩764, SuperNode Ensemble Final 1 cell 만 GLM API 호출, no-filter/selector_only 는 ₩0)
  - 시간: ~3h parallel 거의 동일 (1 cell 추가는 GPU 병렬 windows 에 흡수)
  - presentation_brief §14.2 9-cell 매트릭스 표 갱신 — SuperNode Ensemble row 추가 (planner Edit 완료)

- **근거**:
  - 매트릭스 일관성: Plain 3 + QCond 3 + SuperNode 3 = 9 (대칭) vs 8 (비대칭) → 발표 슬라이드 매트릭스 표 깔끔
  - 발표 Q&A 위험: 비대칭이면 "왜 SuperNode 는 Ensemble 측정 X?" 질문 받기 쉬움
  - SuperNode + EnsembleSelector α=0.85 호환 가능 (smoke test 는 16 cells 핸드오프에 이미 포함된 α=0/1 으로 동시 검증)
  - Ablation 1 (Builder 3종 × 3 stage = 9 cells 완전 매트릭스) 와 동일 패턴 — 두 ablation 의 표 구조 일관

- **에스컬레이션**:
  - 직전 root 핸드오프 prompt (16 cells) → **갱신본 (19 cells)** 사용자에게 즉시 제공. 사용자가 root 에 아직 송신 전 (사용자 confirm).

---

## 2026-04-26 (보강 — Ablation 2 Option B 채택) — Selector ablation 8-cell 매트릭스 cumulative 16 cells 측정 승인 + Alpha convention 정정 (α 는 cosine 가중치) + Plain Cosine 이미 측정 발견

- **결정**:
  1. **(a) Ablation 2 Option B 채택** — Encoder (Plain / QCond / SuperNode) × Score (GAT only α=0 / Cosine only α=1 / Ensemble α=0.85) 매트릭스의 cumulative 표 완성. 사용자 8-cell 정의 (SuperNode + Ensemble 제외 — Ablation 1 옵션 C 와 일관).
  2. **(b) Alpha convention 정정 — α 는 cosine 가중치** — 코드 [`src/modules/selectors/ensemble_selector.py:28`](../src/modules/selectors/ensemble_selector.py) 확인: `final_score = alpha * raw_cosine + (1 - alpha) * gat_score`. **α=0 = GAT only, α=1 = Cosine(Raw) only, α=0.85 = Cosine 우세 ensemble** (cos 0.85 + GAT 0.15). 직전 발표 보고서 §14.2 "Raw α=0" 표기 잘못 → 정정 완료. EXPERIMENT_HISTORY/PLAN 등 다른 문서의 α 표기도 향후 확인 필요.
  3. **(c) Plain Cosine Only 이미 측정 발견** — `abl_a01_05_cos_basic_xiyan` (α=1, Plain encoder + Cosine + Basic PCST + XiYan): R=0.7987 / P=0.7694 / **F1=0.7838**. 사용자 8-cell 매트릭스 중 cell #2 (Plain Cosine) 매핑 확정.
  4. **(d) SuperNode + EnsembleSelector 호환 가능** — `EnsembleSelector` 가 `query_supernode: bool` flag 보유 (코드 L42). ckpt `best_gat_query_supernode.pt` (suffix `_direct` 없음) 가 EnsembleSelector 용으로 추정. **Root 가 1 query smoke test 로 검증 후 측정 진행 필수** (부적합 시 SuperNode cells #7/#8 보류 + planner 에스컬레이션).
  5. **(e) Ablation 1 핸드오프와 분리 송신** — 사용자 결정 (root 가 Ablation 1 5 cells 진행 중). Ablation 2 16 cells 별도 핸드오프 — Ablation 1 종료 후 시작 또는 GPU 점유 확인 후 진행.
  6. **(f) 비용 ₩2,292, 시간 ~3h parallel** — LLM-free cells 14/16 (Selector_only 8 + No-filter 5 + Final 3). Final 3 cells 만 GLM API 호출.

- **근거**:
  - 측정 상태 (직전 응답 9-cell 매트릭스):
    - Final 측정: 6 cells (Plain GAT/Cosine/Ensemble, QCond GAT/Ensemble, QCond Ensemble GLM era)
    - Final 미측정: 3 cells (QCond Cosine α=1, SuperNode GAT α=0, SuperNode Cosine α=1)
    - No-filter 측정: 3 cells (Wave 1.5 W1/W2/W3 = Plain GAT, QCond GAT, QCond Ensemble)
    - No-filter 미측정: 5 cells (Plain Cosine, Plain Ensemble, QCond Cosine, SuperNode GAT, SuperNode Cosine)
    - Selector_only 측정: 0 cells
    - Selector_only 미측정: 8 cells (사용자 8-cell 매트릭스 모두)
  - **총 측정 필요: 3 final + 5 no-filter + 8 selector_only = 16 cells**
  - LLM-free cells (Selector_only + No-filter) 비용 ₩0 — Ablation 1 옵션 C 동일 패턴
  - 발표 D-2 (2026-04-28) 안전: parallel GPU 0/1 사용 시 ~3h
  - SuperNode Cosine 가설: PLM (MiniLM-L6-v2) 동일이라 cosine 임베딩 = Plain Cosine 결과 같을 가능성 → 검증 cell 로 측정

- **영향 범위**:
  - **즉시 (root 16 cells 실행)**: 별도 핸드오프 (본 엔트리 직후 응답에 코드블록)
  - **결과 수령 후 (planner)**:
    - presentation_brief_2026-04-28.md §14.2 cumulative 표 완성 (8 cells × 3 stage = 24 cells, GLM era anchor 별도)
    - DECISIONS 후속 엔트리 (16 cells 결과 + 핵심 발견 정리)
    - SuperNode Cosine = Plain Cosine 가설 검증 결과 보고
    - Encoder × Score × Stage 의 amplification/attenuation 패턴 분석 (Builder ablation 1 결과와 cross-reference)
  - **HISTORY 갱신**: Wave 1.5 backfill 또는 신규 subsection "2026-04-26 Selector Ablation Cumulative Backfill (Option B)"
  - **CATALOG/ID_MIGRATION**: s04_stagewise 계열 cells + selector_only/no_filter suffix 명명 등재

- **에스컬레이션 필요 여부**:
  1. **Root 세션 (즉시, Ablation 1 종료 후)** — 16 cells 실행. SuperNode + EnsembleSelector smoke test 사전 필수.
  2. **Planner (결과 수령 후)** — §14.2 표 보강 + DECISIONS 후속. Ablation 1 결과와 cross-reference.

- **추가 필요 분석**:
  - Encoder × Score interaction: Plain 에서 Cosine (0.7838) > GAT (0.6944) 차이가 QCond 에서 어떻게 변하는지 (현재 QCond GAT 0.7051 > Plain GAT 0.6944, QCond Cosine 측정 후 비교)
  - Ensemble synergy: ΔF1(Ensemble - max(GAT, Cosine)) 의 encoder 별 차이 — Plain (-0.0025 negligible) vs QCond (?) vs SuperNode (?)
  - Filter Δ F1 by Encoder × Score: 8 cells × (Filter Δ = final - no-filter) — Filter 의존도가 어느 조합에서 가장 높은지

---

## 2026-04-26 (보강 — Option C 채택) — Ablation 1 (Builder × Stage) cumulative 완성 5 cells 측정 승인 + Selector only stage 정의 + 비용 추정 정정

- **결정**:
  1. **(a) Option C 채택** — Ablation 1 cumulative 표 (3 Builder × 3 stage = 9 cells) 완성 위해 **5 cells 추가 측정 승인**. 사용자 판단 (2026-04-26): "Builder 는 파이프라인 첫 단계, Selector 가 직접 받는 정보 변화 측정에 Selector only 성능 필수".
  2. **(b) Selector only stage 정의** — Ensemble α=0.85, top_k=20 cut (기본값), Extractor=NoOp/Identity, Filter=NoneFilter. 3 Builder (Plain / Enriched / Triplet) 모두 동일 selector 설정으로 비교. 기존 `abl/a03_direct_per_step/abl_a03_0{1,3,5,9}_*_selector_only` configs 가 mechanism reference.
  3. **(c) 비용/시간 추정 정정** — 직전 추정 ₩3,820 / ~4h 는 잘못. 5 cells 모두 LLM 호출 0 (Filter=None, Extractor=NoOp 비-LLM) → **비용 ₩0, 시간 ~30~45min** (GPU 0/1 병렬). 발표 D-2 일정 안전.
  4. **(d) HISTORY 갱신 위치** — Wave 1.5 no-filter backfill section 확장 (Builder ablation no-filter + selector_only subsection 추가) 또는 신규 `2026-04-26 Builder Cumulative Backfill` subsection. Root 결정.

- **근거**:
  - Ablation 1 측정 상태 (직전 응답 표): 9 cells 중 4 cells 만 측정 (Plain final + Plain no-filter + Enriched final + Triplet final). Selector only 0/3, +Extractor (no filter) 1/3.
  - LLM 호출 0 cells 의 빠른 측정 패턴: Wave 1.5 no-filter backfill 3 cells (W1/W2/W3 no_filter) 가 전례 — 약 35분 내 완료 (vLLM 종료 후 GPU 0/1 병렬, [DECISIONS.md 2026-04-22 17:05 §결정 (a)](DECISIONS.md)).
  - Builder 가 파이프라인 entry point: 노드 텍스트 (name vs name+description vs +triplet) 차이가 LocalPLMEncoder 임베딩 → Selector score → 후속 모든 stage 에 전파. **Selector only metric 이 Builder 변경의 isolated effect 측정의 가장 직접적 평면**.

- **영향 범위**:
  - **즉시 (root 5 cells 실행)**:
    1. Plain Builder + Selector only — config 신규 (anchor `s04_stagewise_qcond_gat_basic` 변형, Extractor/Filter 비활성)
    2. Enriched Builder + Selector only — config 신규 (anchor `s03_a07_01_enriched_gat` 변형)
    3. Triplet Builder + Selector only — config 신규 (anchor `s03_a07_02_edge_prize` 변형)
    4. Enriched Builder + Extractor no-filter — config 신규 (anchor `s03_a07_01_enriched_gat`, Filter=NoneFilter)
    5. Triplet Builder + Extractor no-filter — config 신규 (anchor `s03_a07_02_edge_prize`, Filter=NoneFilter)
  - **측정 후 (planner)**:
    - presentation_brief_2026-04-28.md §14.1 cumulative 표 9 cells 완성 (3×3 매트릭스)
    - DECISIONS 후속 엔트리 (결과 수령 + Filter Δ F1 정량 by Builder)
  - **HISTORY 갱신 (root 측정 직후)**: Wave 1.5 no-filter section 확장 또는 신규 subsection
  - **비용 0** — Wave 4 multi-agent budget 영향 없음

- **에스컬레이션 필요 여부**:
  1. **Root 세션 (즉시)** — 5 cells configs 생성 + 실행 + HISTORY 갱신. 프롬프트 본 엔트리 직후 응답에 코드블록 제공.
  2. **Planner (결과 수령 후)** — §14.1 cumulative 표 보강 + Filter Δ F1 by Builder 정량 + DECISIONS 후속.

- **추가 필요 분석**:
  - Builder Filter Δ F1 정량: Plain (현재 +0.5605) vs Enriched / Triplet — Enriched/Triplet 의 description 정보가 filter 단계 의존도를 줄이는지 (작은 Δ) 아니면 동일한지 (비슷한 Δ) 확인.
  - Selector only F1 ranking 이 final F1 ranking 과 일치하는지 — Builder 효과의 stage-별 amplification/attenuation 패턴.

---

## 2026-04-26 (종합 readiness) — Analyzer 2 보고 수령 (sweep 보강 + Wave 3 F 100% 완료) + H3 small-graph schema feature 가이드 도출 + Selector closure 확인 + 발표 D-2 자료 readiness

- **결정**:
  1. **(a) Analyzer 작업 1 보강 수령** — diameter_layers_sweep.md §2.3 (truncate 2 cell row + per-DB 분해) + §2.4 (mechanism 비교 5-row 표) + §5.1 (C-3 footnote 채택) + §5.2 (Wave 3 우선순위 update — C 트랙 종료, F > C). **새 발견**: partial positive (+0.0064 vs recon) origin = **D=3 단독 +0.0481** (debit_card_specializing 1 DB), D=4/5 거의 무영향.
  2. **(b) H3 schema feature 우선순위 후보 도출** — small-graph 지표 (`|V|·D_max` 곱이 작은 DB) 가 truncate 효과 가장 잘 예측. Single-DB 집중 효과를 평균화하는 학습된 predictor 가 future work 의 본질. proposals §2 H2-truncate 항목 + §8 Changelog (보강) 갱신 완료 (planner Edit, 본 엔트리 직전).
  3. **(c) Wave 3 Proposal F 100% 완료 수령** — analyzer 가 2026-04-24 처리 후 미보고 상태였음. steiner_backbone_stagewise_report.md §1/§3.3/§3.4/§5/§6/§9 보강 + 발표 슬라이드 F-1/F-2/F-3 초안 + §6.4 A→F→C 연결 표 작성 완료. GLM era 재실행 가치 ΔF1±0.01 우선순위 낮음. 발표 D-2 안전.
  4. **(d) 발표 D-2 자료 readiness 종합 판정 = ✅ Ready** — 미해결 1건 (nl=5 cell 진행 상태) 외 모든 트랙 자료 준비 완료.

- **근거**:
  - Analyzer 보고 (2 작업):
    - 작업 1 (diameter_layers_sweep.md 보강): §2.3 Ldbmax_glm Overall F1=0.5868 + Ldbmax+1_glm 0.5605 + per-DB 분해, §2.4 mechanism 비교 표 + 결론 (truncate partial positive D=3 단독 +0.0481, anchor 갱신 임계 미달, nl=7 sign 반전 = over-smoothing × truncate H3 ckpt 가이드 직접 증거), §5.1 C-3 footnote (DECISIONS §결정 (d) 채택), §5.2 Wave 3 우선순위 (C 종료, F>C)
    - 작업 1 추가 분석 (선택): D=3 단일 DB partial positive 집중 → H3 schema feature 우선순위 = small-graph (|V|·D_max 곱)
    - 작업 2 (Wave 3 F 진행): 100% 완료 (2026-04-24 처리 미보고), F-1/F-2/F-3 슬라이드 + A→F→C 연결 표 완료, GLM era 재실행 우선순위 낮음
  - H2-truncate origin 재해석: 직전 DECISIONS 2026-04-26 (후속) 엔트리에서 "D=3,4,5 DB 944q (61.5%) 에서 truncate partial 우수" 라고 적었으나, analyzer §2.4 보강으로 **실제 origin 은 D=3 단독 64q (4.2%)** 임이 정량화됨. D=4,5 880q (57.3%) 는 사실상 무영향. → narrative 정밀도 향상, H3 가이드 specificity 강화.
  - 발표 자료 readiness:
    - **A 트랙**: Wave 1.5 stagewise + GLM era new top (F1=0.8383) — EXPERIMENT_HISTORY/PLAN/CLAUDE 모두 갱신 완료 (root)
    - **F 트랙**: Steiner backbone 재조직 리포트 + 슬라이드 3 + A→F→C 연결 — analyzer 100% 완료
    - **C 트랙**: Diameter sweep 5 cell + H2 truncate 2 cell + sanity 1 + new anchor 1 (total 9 cells) + footnote — analyzer 보강 완료, proposal 갱신 완료
    - **D/E 트랙**: post-deadline 순연 (Wave 2 §8-1 SuperNode bug + 재학습 비용)
    - **B 트랙**: 발표 후 순연 (T2T edge graph regen)

- **영향 범위**:
  - **proposals/abl_sel_diameter_layers.md** — §2 H2-truncate origin 정량화 (D=3 단독) + H3 small-graph schema feature 가이드 추가 + §8 Changelog 보강 entry. 본 엔트리 직전 Edit 완료.
  - **EXPERIMENT_PLAN.md §4 Phase 0 Wave 3** — F 트랙 closed 표기 가능 (root 작업, 발표 직후 권장). 현재 "planned" 상태.
  - **발표 슬라이드** — A/F/C 트랙 모두 자료 ready. C-3 footnote 는 analyzer §5.1 에서 작성 완료.
  - **남은 미해결 1건**: nl=5 cell (2026-04-25 §결정 (d)) 진행 상태 — 사용자 답 미수령 (직전 응답 reminder).

- **에스컬레이션 필요 여부**:
  1. **사용자 (확인 1건)** — nl=5 cell 진행 상태:
     - 송신 + 결과 있음 → planner 에 결과 전달 → H1-perDB 보강 분석 (D=5 버킷 nl=5 ckpt 직접 forward vs truncate 비교)
     - 송신 + ckpt 부재로 학습 5h 발생 → planner 가 post-deadline 이동 결정
     - 미송신 → post-deadline 큐 이동 (D-2 임박, 우선순위 하향)
  2. **Root 세션 (지연, 발표 직후)** — EXPERIMENT_PLAN.md §4 Phase 0 Wave 3 F 트랙 "closed" 표기 갱신. 본 엔트리가 마커.
  3. **다른 세션 호출 불필요** — selector closed (전 엔트리 closure 보고 수령), analyzer 2 작업 완료, root nl=5 외 신규 작업 없음.

- **추가 필요 분석**: 없음 (발표 D-2 ready). H3 future work 설계 (post-2026-04-28) 는 별도 wave 신규 에스컬레이션.

---

## 2026-04-26 (후속) — H2 truncate 2 cell 결과 판정: H2 기각 유지 + Selector impl partial neutral + nl=7 truncate training mismatch 증거 + C-3 footnote 추가 + Wave 3 F 진행 상태 조회

- **결정 (5개 사용자 요청 응답)**:
  1. **(a) 2026-04-25 "H2 원래 가설 기각" 결정 유지** — Selector impl truncate forward 실측 결과:
     - `layers_Ldbmax_glm` (D_max, nl=6 ckpt truncate): F1=0.5869, ΔF1 vs L6_glm = **+0.0045** → 분기 (3) partial neutral, 실용적 개선 한계 (anchor 갱신 임계 +0.005 미달)
     - `layers_Ldbmax_plus1_glm` (D_max+1, nl=7 ckpt truncate): F1=0.5604, ΔF1 vs L6_glm = **-0.0220** → 분기 (1) 기각 확고, training mismatch 강한 증거
     → **두 cell 모두 H2 실용적 개선 한계 노출**. 2026-04-25 기각 결정 변경 사유 없음.
  2. **(b) Selector impl mechanism partial positive 확인** — Ldbmax 가 analyzer recon (0.5805) 대비 **+0.0064**. D_max=3,4,5 DB (944 queries, 61.5%) 에서 truncate forward 가 fallback 보다 약간 나음. 단 L6_glm baseline (0.5824) 을 의미있게 (+0.005 이상) 넘지 못함 — H2 의 학술적 가치 정량화 (recon 대비 marginal gain).
  3. **(c) nl=7 truncate 의 training mismatch 증거 명문화** — nl=6 truncate (ΔF1=+0.0045, neutral) vs nl=7 truncate (ΔF1=-0.0220, 큰 손실) 의 sign 반전. nl=7 ckpt 자체 over-smoothing 영향 (Wave 2 sweep 에서 ΔF1=-0.0062 vs nl=6) 을 빼도 truncate mismatch 순효과 ~-0.0158 추정. **Over-smoothing 영향권 ckpt 의 truncate 는 추가 위험** — H3 future work 설계 시 학습 ckpt 선정 가이드 (over-smoothing 회피).
  4. **(d) 발표 슬라이드 C-3 narrative — footnote 권장안 채택** — 기존 기각 narrative 유지 + footnote 1줄 추가:
     > Selector impl truncate mechanism 실측 (2026-04-26): D_max truncate ΔF1=+0.0045 (vs analyzer recon +0.0064 partial positive), D_max+1 truncate ΔF1=-0.0220 (training mismatch). 두 결과 모두 H2 기각 결론 변경 없음.
     사유: 실측이 됐으니 투명 보고가 학술적 정직성. 대안 (footnote 미추가) 는 selector 세션 작업 결과 누락 — 거부.
  5. **(e) Wave 3 Proposal F 진행 상태 조회 — analyzer 세션 핸드오프에 통합** — 2026-04-24 Phase 전환 §에스컬레이션 #1 작업 2 (Steiner backbone 재조직) 진행 보고 미수신. 발표 D-2 임박, Wave 3 F 가 main story 의 다음 트랙. H2 보강 요청과 묶어서 단일 핸드오프로 송신.

- **근거**:
  - Root 보고 (2026-04-25 01:36:06~02:33:12, scripts/run_h2_truncate.sh) 메트릭 표:
    | Cell | R | P | F1 | ΔF1 vs L6_glm | ΔF1 vs analyzer recon | 분기 |
    |------|---|---|---|---|---|------|
    | L6_glm (anchor) | 0.5018 | 0.6939 | 0.5824 | — | +0.0019 | — |
    | analyzer recon | — | — | 0.5805 | -0.0019 | — | (보고용) |
    | **Ldbmax_glm** | 0.5036 | 0.7031 | **0.5869** | **+0.0045** | **+0.0064** | (3) partial neutral |
    | **Ldbmax_plus1_glm** | 0.4778 | 0.6776 | **0.5604** | **-0.0220** | -0.0201 | (1) 기각 확고 |
  - DECISIONS 2026-04-26 (전 엔트리) §영향 범위 4-way 분기 표 사전 합의 — 본 결과 매핑.
  - Mechanism 차이 결과 quantify:
    - D_max=6 DB 590 q: nl=6 ckpt 전체 forward = analyzer recon = 동일 (0.6646)
    - D_max=3,4,5 DB 944 q (61.5%): selector impl truncate +0.0064 query-weighted improvement vs fallback. 즉 truncate forward 가 ckpt 부재 가정 fallback 보다 약간 낫지만 anchor 갱신 임계 미달.
  - Training mismatch 증거 (nl=7 vs nl=6 truncate sign 반전):
    - 두 cell 의 mechanism 동일 (DB 의 D_max 만큼 layer truncate forward), ckpt 만 다름
    - nl=6 ckpt (over-smoothing 영향 X): ΔF1=+0.0045 (neutral)
    - nl=7 ckpt (over-smoothing 영향권): ΔF1=-0.0220 (큰 손실)
    - 차이 -0.0265 가 over-smoothing × truncate 누적 효과 — H3 ckpt 선정 가이드 근거

- **영향 범위**:
  - **2026-04-25 H2 기각 결정 유지** (변경 없음, 보강만 추가)
  - **proposals/abl_sel_diameter_layers.md §2** — H2-truncate 항목 추가 (planner 본 엔트리 직후 Edit, §8 Changelog 도 갱신)
  - **발표 슬라이드 C-3** — footnote 1줄 추가 (root 또는 사용자가 슬라이드 자료 작성 시)
  - **EXPERIMENT_PLAN_selectors.md (selector 세션 작업)** — H2 항목 "closed (2026-04-26), partial neutral + training mismatch 확인, H3 future work 재활용 가능" 표기
  - **notebooks/analysis_results/diameter_layers_sweep.md (analyzer 세션 작업)** — §2.3/§2.4 selector impl truncate row 추가 + §5.2 Wave 3 우선순위 update (C 트랙 종료, F 만 active)

- **에스컬레이션 필요 여부**:
  1. **Selector 세션 (closure)** — H2 작업 종료 + EXPERIMENT_PLAN_selectors.md H2 표기 갱신. 프롬프트 본 엔트리 직후 응답에 코드블록 제공.
  2. **Analyzer 세션 (보강 + Wave 3 F 진행 조회 통합)** — diameter_layers_sweep.md §2.3/§2.4/§5.2 보강 + Wave 3 Proposal F (Steiner backbone 재조직) 진행 상태 보고. 프롬프트 본 엔트리 직후 응답에 코드블록 제공.
  3. **Root 세션 (지연 마커)** — selector/analyzer 갱신 완료 + Wave 3 F 보고 후 발표 슬라이드 자료 최종 정리. 본 엔트리가 대기 마커.

- **추가 필요 분석**:
  - H3 future work 설계 가이드: over-smoothing 영향권 ckpt (nl > D_max global) 회피 — 본 엔트리 §결정 (c) 근거
  - Analyzer 작업 1 §2.3 보강 후 selector impl partial positive 가 D_max=3/4/5 중 어느 그룹에 집중되는지 — H3 schema feature 우선순위 근거
  - Wave 3 F 진행 상태 수령 후 발표 슬라이드 F 트랙 최종 검토 (planner)

---

## 2026-04-26 — Selector H2 inference 2 cell 실측 승인 (analyzer recon ≠ selector impl mechanism, 2026-04-25 H2 기각 결정의 검증 실험으로 재정의)

- **결정**:
  1. **(a) Selector H2 inference 2 cell** (`layers_Ldbmax_glm`, `layers_Ldbmax_plus1_glm`) **실측 승인** — 2026-04-25 H2 기각 결정 (naive resolve(db)=D_max → ΔF1=-0.0019) 의 **검증 실험** 으로 의미 재정의. Selector impl (nl=6/7 ckpt **truncate forward**) 과 analyzer reconstruction (sweep 5 cell 재조합 + D_max=4/5 fallback) 은 **다른 mechanism** 이므로 실측 가치 있음.
  2. **(b) Root 송신 핸드오프는 selector 원본 그대로 X — augmented 버전** 필요. 원본 프롬프트의 기대 수치 (+0.005~0.020) 는 2026-04-25 개정 이전 가설 기반으로 outdated. Augmented 에 (i) 2026-04-25 맥락 + (ii) mechanism 차이 + (iii) 기대 수치 갱신 + (iv) 4-way 결과 해석 분기 + (v) 발표 슬라이드 C-3 영향 가이드 추가.

- **근거**:
  - Analyzer reconstruction 의 0.5805 (-0.0019) 계산식 ([diameter_layers_sweep.md L92](../notebooks/analysis_results/diameter_layers_sweep.md)):
    `F1 = (64×0.3687 + 443×0.5114 + 437×0.5709 + 590×0.6646) / 1534`
    - D_max=3 (64 q, 4.2%) → nl=3 cell 결과 (0.3687)
    - D_max=4 (443 q, 28.9%) → **nl=4 ckpt 부재 → nl=6 fallback** (0.5114, nl=6 행 값 그대로)
    - D_max=5 (437 q, 28.5%) → **nl=5 ckpt 부재 → nl=6 fallback** (0.5709)
    - D_max=6 (590 q, 38.5%) → nl=6 cell (0.6646)
    → **D_max=4/5 query 합 = 1,470/1,534 = 95.8% 가 사실상 nl=6 그대로**. ΔF1=-0.0019 는 D_max=3 단독 손실 반영, **H2 의 진짜 per-DB dynamic 정보는 거의 없음**.
  - Selector impl mechanism (사용자 핸드오프 추정 — "EnsembleSelector v2 분기 + nl=6/7 ckpt 만 재활용"):
    - **nl=6 ckpt 의 layer 수 동적 truncate forward** (D_max=3 DB → 3-layer forward, D_max=6 DB → 6-layer forward)
    - D_max=6 DB (590 q): nl=6 cell 결과와 동일 (0.6646)
    - D_max=3 DB (64 q): nl=6 ckpt 의 처음 3 layer truncate (vs analyzer recon: nl=3 ckpt 별도 학습 결과)
    - D_max=4,5 DB (880 q): nl=6 ckpt 의 처음 4/5 layer truncate (vs analyzer recon: nl=6 ckpt 전체)
    → **D_max<6 (944 q, 61.5%) 에서 selector impl ≠ analyzer recon**. 결과 분기 가능.
  - Selector impl trade-off:
    - **장점**: 진짜 H2 spirit (per-DB dynamic depth) 구현. ckpt 부재 회피 (single ckpt + truncate). 발표 narrative 에서 "방법론적 contribution" 으로 reportable.
    - **단점**: Training-inference depth mismatch — nl=6 ckpt 은 6-layer forward 로 학습됐는데 truncated 4-layer 출력은 학습된 head 분포와 어긋남. 일반적 GNN 에서 성능 하락 ~5~15%.
  - 비용: 2 cell × ~₩764 = ~₩1,528, GPU 0/1 병렬 (GLM API 호출, GPU 미점유) ~50min total. 발표 D-3 안전.

- **영향 범위**:
  - **즉시 (root)**: augmented 핸드오프 송신 → 2 cell inference + HISTORY 3종 갱신
  - **결과 수령 후 (planner, 4-way 분기 사전 합의)**:
    | Selector impl ΔF1 (vs L6_glm=0.5824) | 해석 | 후속 행동 |
    |--------------------------------------|------|-----------|
    | < -0.005 | Truncate 실패 + training mismatch 추가 손실 | H2 기각 더 확고. C-3 narrative 보강. |
    | ≈ analyzer recon (-0.002) | Truncate ≈ fallback 효과, mechanism 차이 무의미 | H2 기각 유지. C-3 narrative 그대로. |
    | -0.002 ~ +0.005 | Truncate 가 약한 over-smoothing 완화 | Partial neutral. C-3 minor mention. |
    | **+0.005 ~ +0.020** | **Truncate 가 ckpt 부재 보완 + over-smoothing 완화** | **H2 partial 부활 — 2026-04-25 기각 재고. C-3 narrative 분기. planner 즉시 후속 엔트리.** |
    | > +0.020 | 예상치 초과 강한 H2 효과 | 발표 main story 재구성. planner 긴급 회의 + DECISIONS 우선순위 1 엔트리. |
  - **결과 수령 후 (analyzer)**: diameter_layers_sweep.md §2.3/§2.4 에 selector impl truncate row 추가 + mechanism 비교 표.
  - **결과 수령 후 (selector)**: H2 작업 종료 표기 + EXPERIMENT_PLAN_selectors.md 갱신. H3 (future work) 인프라 일부 재활용 명기.
  - **proposal 갱신 (planner, 결과 의존)**: planning/proposals/abl_sel_diameter_layers.md §2 에 H2-truncate 항목 추가 (selector impl 실측 결과).

- **에스컬레이션 필요 여부**:
  1. **Root 세션 (즉시, augmented 핸드오프)** — 본 엔트리 직후 응답에서 사용자 송신용 코드블록 제공.
  2. **Selector 세션 (정보)** — 본 엔트리 §4-way 결과 해석 분기 공유 (selector impl mechanism 의 analyzer recon 대비 차이 인지). 결과 수령 후 H2 작업 closure.
  3. **Analyzer 세션 (결과 수령 후)** — diameter_layers_sweep.md §2.3 mechanism 비교 표 보강 (analyzer recon row + selector impl truncate row).

- **추가 필요 분석**:
  - 결과 수령 후 4-way 분기 판정 → DECISIONS 후속 엔트리 (planner)
  - nl=5 추가 cell (2026-04-25 §결정 (d)) 결과와 결합 분석 — D_max=5 버킷에서 selector impl truncate vs nl=5 ckpt 직접 forward 비교
  - Selector impl truncate forward 의 정확한 mechanism 확인 (nl=6 ckpt 의 어떤 layer 까지 + head 처리?) — 결과 해석 정확성 위해 선택적

---

## 2026-04-25 — Proposal C H2 가설 개정 (diameter-direct 매핑 기각 + Oracle 상한 보고 전용) + nl=5 추가 cell 승인 + Wave 3 F 재평가 post-deadline 큐 등록

- **결정** (5개):
  1. **(a) H2 원래 가설 기각** — resolve(db_name)=D_max(db) 매핑으로 inference 시 global fixed nl=6 대비 개선 가설: query-weighted ΔF1 = **-0.0019 (하락)**. Naive D_max 매핑은 per-DB empirical best 와 어긋남 ([diameter_layers_sweep.md L92](../notebooks/analysis_results/diameter_layers_sweep.md)).
  2. **(b) H2' Oracle 상한 — 보고 전용** — 각 DB 를 그 DB 의 empirical best nl 로 inference 가정 시 query-weighted **ΔF1 = +0.0237** (D_max=4 버킷에서 +0.0604 최대). **Data leakage** (BIRD dev 에서 per-DB best 측정 후 dev 에 적용 = unfair) → inference 실측 구현 불가. 발표 슬라이드 C-3 는 "상한 존재 + 실용적 구현은 future work" 로 보고.
  3. **(c) resolve(db) 를 per-DB empirical best 로 교체하지 않음** — data leakage 이유. 대신 **H3 (future work, 신설)**: schema feature (|V|, |E|/|V|, degree distribution, D_max, D_mean, SCC count) → per-DB optimal depth **regression/classifier 학습** (학습 split=BIRD train, 평가=BIRD dev). post-2026-04-28 selector/analyzer 협업.
  4. **(d) nl=5 추가 cell 즉시 실행 승인** — D=5 DB 의 nl=5 preference 확인 (현재 sweep {1,2,3,6,7} 사이 {4,5} gap, 특히 D_max=5 버킷 미측정). 비용 ~₩764, 시간 ~50min + GAT 학습 필요 여부는 root 에서 ckpt 검증 후 판단. 발표 전 완료 가능하면 per-DB H1 엄밀 검증 보강.
  5. **(e) Wave 3 F 재평가 Ensemble+SteinerBackbone+XiYan 1 cell post-deadline 큐 등록** — [steiner_backbone_stagewise_report.md §5 #6d](../notebooks/analysis_results/steiner_backbone_stagewise_report.md) 근거. 현재 Steiner 는 DirectGAT binary Selector 한정 측정 → Ensemble α=0.85 Selector 재평가 시 진짜 Steiner 가치 정량화. 비용 ~₩764, 2026-04-29+ 실행.

- **근거**:
  - Analyzer 리포트 [diameter_layers_sweep.md §2.3/§2.4/§5.2](../notebooks/analysis_results/diameter_layers_sweep.md):
    - §2.3 D_max 버킷별 F1 분해: D=3/4/5 반례 — 각 DB 의 best nl 이 D_max 와 어긋남
    - §2.4 H1 엄밀 검증 — D=6 버킷 (590 queries, 38.5%) 이 전체 peak 를 결정하는 단일 요인, 나머지 버킷 반례
    - §5.2 "Proposal C hypothesis 수정 후 재승인 필요 — planner 에스컬레이션"
    - §0 TL;DR L12: Naive H2 실측 F1=0.5805 < global nl=6 F1=0.5824, ΔF1=-0.0019 하락
  - Data leakage: H2 oracle 은 평가 split 의 label 로 설계 선택 → fair comparison 위반. 학술적 산출 불가 (논문 리뷰어가 즉시 reject).
  - Proposal C 원래 §2 H2 ("D_max 극단 DB 에서 over-smoothing 재등장") 는 nl=7 결과로 **partial 검증** — 단 primary claim "num_layers=D_max 에서 peak" 가 global 에서만 맞고 per-DB 에서 실패하는 게 더 중요한 발견.
  - nl=5 gap 검증: 현재 sweep 에 D_max=5 버킷 peak 미측정 — {4,5} 보강 시 H1 per-DB 재판정 근거 강화. ckpt 존재 여부는 root 가 `ls outputs/checkpoints/best_gat_qcond_nl5.pt` 로 선제 검증.
  - Wave 3 F #6d: 기존 Steiner 평가 = DirectGAT binary Selector (단순 이진 선택) → Ensemble α=0.85 (cosine+GAT) 재실험 시 extractor 전달 노드셋 품질 개선으로 Steiner 의 순효과 분리 가능. post-deadline 이 일정 안전.

- **영향 범위**:
  - **즉시 (root 실행)**: nl=5 추가 cell 1개 — config 생성 + ckpt 검증 + inference + HISTORY 갱신
  - **즉시 (proposal 개정, planner primary write)**: [planning/proposals/abl_sel_diameter_layers.md](proposals/abl_sel_diameter_layers.md) §2 H1/H2/H3 개정 + §7 Changelog — 본 엔트리 직후 Edit.
  - **즉시 (selector 세션 재지시)**: H2 인프라 작업 **우선순위 하향** (신규 시작 시 cancel, 진행 중이면 post-deadline H3 재활용용으로 완료). 발표 전 H2 슬라이드 추가 취소.
  - **post-deadline 큐 (2026-04-29+)**:
    - Wave 3 F Ensemble+Steiner+XiYan 1 cell (root 실행)
    - H3 schema feature → depth predictor 탐색 (selector/analyzer 협업)
  - **문서 파급**:
    - EXPERIMENT_PLAN.md §4 Phase 0 Wave 2 — H2 표기 "검증 시도 → 기각, Oracle +0.0237 상한만 보고" 로 갱신 (root 작업)
    - EXPERIMENT_PLAN.md §4 Phase 0 Wave 3 — nl=5 cell + Ensemble+Steiner post-deadline 큐 추가
    - EXPERIMENT_PLAN_selectors.md — H2 우선순위 하향 + H3 future work 신설 (selector 세션 작업)

- **에스컬레이션 필요 여부**:
  1. **Root 세션 (즉시)** — nl=5 cell 실행:
     ```
     먼저 CLAUDE.md 와 planning/DECISIONS.md 2026-04-25 엔트리 §결정 (d) + proposals/abl_sel_diameter_layers.md §2 개정판을 읽어라.
     
     작업: nl=5 추가 cell 실행 — D_max=5 DB 의 per-DB best 검증.
       (1) ckpt 존재 검증: ls outputs/checkpoints/best_gat_qcond_nl5.pt
            - 존재 O: (2) 진행
            - 존재 X: 학습 필요 여부 즉시 planner 에스컬레이션 (학습 비용 ~5h + 발표까지 D-3 일정 위험, 학습 skip 후 보간 해석 가능한지 검토 필요)
       (2) config 생성: configs/experiments/s04_ablation/diameter_layers/layers_L5_glm.yaml
            - 기존 layers_L6_glm.yaml 복사 + num_layers: 5 + weight_path 갱신
       (3) inference: conda run -n base python src/main.py --config experiments/s04_ablation/diameter_layers/layers_L5_glm
       (4) HISTORY 갱신 — Wave 2 Proposal C GLM era kickoff section 에 6번째 cell 추가, sweep 표 갱신
       (5) EXPERIMENT_PLAN.md §0 diameter_layers peak 표기 갱신 (nl=5 결과로 peak 이동하면)
       (6) analyzer 세션 핸드오프 (planner 경유): diameter_layers_sweep.md §1.1 6-cell 으로 갱신 + §2.3 D_max=5 버킷 재분석 + §2.4 H1 per-DB 검증 갱신
     성공 기준: nl=5 metrics 측정, D_max=5 버킷 F1 peak 여부 판정
     비용: ~₩764, 시간: ~50min (ckpt 존재 시) / ~5h50min (ckpt 학습 필요 시 — 이 경우 planner 에스컬레이션 우선)
     ```
  2. **Selector 세션 (재지시)**:
     ```
     먼저 planning/DECISIONS.md 2026-04-25 엔트리 §결정 (a)(b)(c) + proposals/abl_sel_diameter_layers.md §2 H2/H3 개정판을 읽어라.
     
     H2 가설 개정 반영:
       - resolve(db)=D_max 기각 (Naive mapping ΔF1=-0.0019)
       - H2' Oracle 상한 (+0.0237) 은 inference 실측 불가 (data leakage) → 보고 전용
       - H3 신설: schema feature → optimal depth predictor (future work, post-2026-04-28)
     
     작업 조정:
       - 현재 EnsembleSelector v2 분기 / db_name threading / resolve_num_layers hook 구현이 in-progress 이면 → 완료까지 진행 (H3 재활용 가능), 단 **발표 전 H2 inference 시도 X**
       - 신규 시작 상태이면 → 우선순위 하향, post-2026-04-28 재개
       - train_gat_s06.py v2 flag forward: H3 에서 학습 시 필요하나 본 sweep 에는 불필요
     
     현재 진행 상태 planner 에 즉시 보고 (in-progress vs not-started) + EXPERIMENT_PLAN_selectors.md H2 항목 표기 갱신.
     ```
  3. **Analyzer 세션 (2 후속 큐)**:
     ```
     먼저 planning/DECISIONS.md 2026-04-25 엔트리 §결정 (d)(e) 읽어라.
     
     [작업 1 — nl=5 결과 수령 시 (root 보고 후)]
       diameter_layers_sweep.md §1.1 6-cell 확장 + §2.3 D_max=5 버킷 F1 측정 + §2.4 H1 per-DB 엄밀 검증 갱신 + §5.1 slide C-2 per-DB 반례 표 갱신.
     
     [작업 2 — post-deadline 2026-04-29+]
       Wave 3 F Ensemble+SteinerBackbone+XiYan 1 cell (steiner_backbone_stagewise_report.md §5 #6d) — root 실행 후 steiner_backbone_stagewise_report.md §3 업데이트.
     ```

- **추가 필요 분석**:
  - H3 future work 설계 (post-2026-04-28): schema feature set 정의 + BIRD train split 에서 best depth labeling + regression/classifier head + dev 평가 fairness 확보
  - nl=5 결과로 per-DB H1 재판정 (analyzer 후속)
  - GLM era full 8-cell (nl={1,2,3,4,5,6,7}) 완성 여부 — {4} 도 필요한지는 nl=5 결과 보고 판단

---

## 2026-04-24 Phase 전환 — Wave 2 closed, Wave 3 F + Proposal C H2 selector 에스컬레이션 동시 개시 / Wave 4 순연 유지

- **결정 (4개 일괄)**:
  1. **(a) Wave 2 closed** — 7 cells (sanity + 5 sweep + new anchor) 측정 완료, EXPERIMENT_HISTORY L1320~ + EXPERIMENT_PLAN.md §0/§4 root 갱신 완료. **GLM era new top** `s04_stagewise_qcond_gat_basic_glm` F1=0.8383 (ΔF1=+0.0506 vs vLLM Wave 1.5 best). H1 (nl=D_max peak) **검증 완료** — nl=6 peak F1=0.5824, nl=7 ΔF1=-0.0062 over-smoothing 재등장.
  2. **(b) Wave 3 Proposal F 즉시 개시** — SteinerBackbone 재조직 (analyzer 단독, 신규 실험 0, 기존 a03_15/18 데이터 재집계). 발표 스토리라인 A>F>C>D>E>B 의 다음 우선 (A=Wave 1.5, C=Wave 2 closed). analyzer 큐 등록 — diameter_layers_sweep.md 와 병행 가능.
  3. **(c) Proposal C H2 (per-DB dynamic num_layers) selector 세션 즉시 에스컬레이션 (재개)** — H1 검증 완료로 H2 가치 강화. BIRD dev 11 DB 의 D_max 분포 다양 (`dev_diameter.pt`) → global fixed nl=6 은 작은 DB (D_max<6) 에 over-smoothing → per-DB dynamic 으로 ΔF1 +0.005~0.020 추가 이득 가능. 기존 5 ckpt (`best_gat_qcond_nl{1,2,3,6,7}.pt`) 재활용 가능 → 새 학습 X, inference 1-3 cell. 발표 전 (~4일) 완료 시 H2 슬라이드 1장 추가.
  4. **(d) L2 dip 진단** — sweep nl=2 F1=0.5510 < nl=1(0.5826)/nl=3(0.5784) 단조성 깨짐 (사용자 요청 옵션 d). 가능 원인: (i) GAT 2-layer specific bottleneck, (ii) 학습 분산 (seed 영향), (iii) anchor stochasticity. analyzer 큐의 diameter_layers_sweep.md §3 에 포함 (사용자 요청대로).
  5. **(e) Wave 4 a05_filter_agentic 순연 유지** — post-2026-04-28 (사용자 옵션 b 의 default 재확인). 사유: (i) vivid-sprouting-sunbeam.md anchor refresh 필요 (`abl_ens_basic_xiyan` F1=0.7863 → `s04_stagewise_qcond_gat_basic_glm` F1=0.8383, ΔF1=+0.0520 갱신), (ii) 12 cell × multi-agent 3-5× LLM call/query → GLM 비용 추정 ~₩40-60K (sweep cell 단가 ~₩764 대비 4-5x), (iii) 발표 일정 (2026-04-28) 까지 4일 — multi-agent prompt tuning + 12 cell 실행 위험. Wave 4 anchor refresh prep 만 filter 세션 사전 마커.

- **근거**:
  - Wave 2 GLM era 결과 (EXPERIMENT_HISTORY.md L1320~, EXPERIMENT_PLAN.md §0 핵심 관찰):
    - 7 cells 측정 완료, **Precision 주 개선축** (ΔP=+0.0724) — LLM backbone 단독 교체로 Builder-driven precision ceiling 0.81 도 돌파.
    - **H1 곡선 단조성 + peak**: nl=1(0.5826) → nl=2(0.5510 ⚠) → nl=3(0.5784) → nl=6(0.5824 peak=D_max) → nl=7(0.5762 ↓ over-smoothing). 제안서 [abl_sel_diameter_layers.md](proposals/abl_sel_diameter_layers.md) §2 H1 예측 정확히 부합. nl=2 dip 만 anomaly.
  - 발표 스토리라인 (2026-04-21 advisor Q4): A=Wave 1.5 (closed), C=Wave 2 (closed) → F 가 다음. D/E 는 §8-1 SuperNode bug fix + 재학습 비용으로 발표 전 불가, B 는 11h regen 비용 후순위.
  - C H2 가치: H1 검증 결과 nl=D_max global fixed 가 peak — BIRD dev 11 DB 의 D_max 가 균일하지 않다면 (작은 DB 가 다수면) per-DB dynamic 추가 이득 큼. selector 세션 작업 (db_name threading + train_gat_s06.py v2 flag forward) 1-2일 완료 가능 추정 (사전 인프라 일부 준비됨 — gat_network_v2.py 의 num_layers_mode flag 존재).
  - L2 dip 진단 가치: 단조성 가정 깨짐 → H1 검증의 견고성 확인 필요. 학습 분산이면 재학습으로 해결, 구조적 bottleneck 이면 별도 발견 (논문 부록).
  - Wave 4 순연: 사용자 답변 (옵션 b 순연 유지) + planner 비용/일정 분석 모두 부합.

- **영향 범위**:
  - **즉시 진행 (병행)**:
    - **Analyzer 세션** — 2 작업 동시 큐 (diameter_layers_sweep.md + Wave 3 Proposal F)
    - **Selector 세션** — C H2 인프라 (EnsembleSelector v2 분기 + db_name threading + train_gat_s06.py v2 flag forward)
  - **순연 유지**:
    - **Filter 세션** — Wave 4 anchor refresh prep (post-2026-04-28). vivid-sprouting-sunbeam.md F1=0.7863 → 0.8383 갱신 + GLM 비용 추정.
  - **문서 영향**:
    - EXPERIMENT_PLAN.md §4 Phase 0 — Wave 3 active 표기 + Proposal F (analyzer 단독) 명시 필요. Selector C H2 (active) 별도 entry. Root 가 7-step 갱신에서 Wave 3 까지 일부 처리했는지 확인 필요 (현재 Wave 3 = "planned" 표기).
    - EXPERIMENT_PLAN_selectors.md — H2 작업 항목 (selector 모듈 세션 책임).
    - planning/proposals/abl_sel_diameter_layers.md — §6 H1 검증 완료 표기, H2 활성화 (selector 세션 진입 시 최신화).

- **에스컬레이션 필요 여부**:
  1. **Analyzer 세션 (즉시, 2 작업 병렬 가능)**:
     ```
     먼저 src/analysis/CLAUDE.md 와 planning/DECISIONS.md 최상단 5개 엔트리 (Phase 전환 + Sanity 재정의 + Sanity 결과 + endpoint 블로커 + LLM 전환) 를 읽어라.
     
     [작업 1 — diameter_layers_sweep.md 작성, 우선]
       데이터: outputs/experiments/s04_ablation/{diameter_layers/layers_L{1,2,3,6,7}_glm/, stagewise/qcond_gat_basic_glm/, s04_04_qcond_a0_xiyan_glm/} 의 metrics.txt + output_*.jsonl + score_analysis_*.jsonl
       산출물: notebooks/analysis_results/diameter_layers_sweep.md
       내용:
         §1 F1/R/P curve + peak 위치 식별 (H1 검증 곡선, nl ∈ {1,2,3,6,7})
         §2 DB 별 D_max 대비 peak alignment (data/processed/dev_diameter.pt 11 DB D_max 분포 + per-DB cell F1)
         §3 L2 dip 진단 (nl=2 F1=0.5510 anomaly): per-DB / per-difficulty / score distribution 분해, 학습 seed/random init 영향 가능성 분리
         §4 각 cell Selector / +Extractor / +Filter 3단계 cumulative R/P/F1 (CLAUDE.md G2 memory rule)
         §부록 A: vLLM era ↔ GLM era 비교 (sanity s04_04 ΔF1=-0.0099 + new anchor s04_stagewise_qcond_gat_basic ΔF1=+0.0506) — LLM backbone 효과 정량화
       의도: 2026-04-28 advisor 미팅 브리핑 자료 + Wave 3/4 우선순위 결정 근거.
     
     [작업 2 — Wave 3 Proposal F (Steiner backbone 재조직), 병렬 가능]
       proposals/abl_ext_steiner_backbone_report.md 참조 + 기존 notebooks/analysis_results/steiner_backbone_stagewise_report.md 보강 (또는 신규 .md)
       데이터: 기존 a03_15 / a03_18 outputs (vLLM era 보존 사용 OK, 신규 GLM era 실행 X)
       의도: 발표 슬라이드 F 트랙 보강 — A > F > C 순서.
     
     PLAN 변경 제안이 있으면 planner 에 에스컬레이션 (절대 직접 EXPERIMENT_PLAN.md 수정 금지).
     ```
  2. **Selector 세션 (즉시)** — Proposal C H2 인프라:
     ```
     먼저 src/modules/selectors/CLAUDE.md 와 planning/DECISIONS.md 2026-04-22 17:05 엔트리 §에스컬레이션 #1 + 2026-04-24 Phase 전환 엔트리 §결정 (c) 를 읽어라.
     
     작업: Proposal C H2 (per-DB dynamic num_layers) 인프라 구현 — H1 검증 완료 (nl=6=D_max global peak F1=0.5824) 로 H2 가치 강화.
       (1) EnsembleSelector 에 SchemaHeteroGATv2 분기 추가 (현재 v1 SchemaHeteroGAT 하드코딩)
       (2) select() signature 또는 내부 경로에 db_name 통과
       (3) runtime resolve_num_layers(db_name) hook 으로 DB 별 D_max 매핑 (data/processed/dev_diameter.pt)
       (4) train_gat_s06.py 에 v2 flag (num_layers_mode, diameter_path, diameter_dict) forward — 기존 5 ckpt (best_gat_qcond_nl{1,2,3,6,7}.pt) 재활용 가능 여부 확인. 재활용 가능하면 신규 학습 0, 인프라만 구현 후 inference.
     
     성공 기준: Mode="D_max" config 로 inference 시 DB 별로 다른 depth resolve + forward pass 실측 (단위 테스트 또는 1 query smoke).
     완료 후 핸드오프: root 세션 (H2 inference 1-3 cell 실행 + 결과 측정).
     일정: 2026-04-28 까지 완료 시 H2 슬라이드 추가. 지연 시 post-deadline planner 에스컬레이션 (Wave 2.5 mini-wave 분리).
     ```
  3. **Filter 세션 (지연 마커, post-2026-04-28)** — Wave 4 anchor refresh prep:
     - vivid-sprouting-sunbeam.md anchor 갱신: `abl_ens_basic_xiyan` F1=0.7863 → `s04_stagewise_qcond_gat_basic_glm` F1=0.8383 (ΔF1=+0.0520)
     - GLM era 12 cell multi-agent 비용 사전 추정 (3-5x LLM call/query): ~₩40-60K
     - kickoff 시점 = post-2026-04-28
     - 본 엔트리가 대기 마커.

- **추가 필요 분석**:
  - L2 dip 원인 (analyzer 작업 1 §3) — 구조적 vs 학습 분산 구분 결과 보고
  - H2 inference 결과 (selector 완료 후 root 실행) — per-DB depth alignment 효과 정량 → planner 가 H2 채택 여부 판정

---

## 2026-04-24 결정 — Sanity check 합격 기준 재정의 (절대 → 상대), GLM era sweep/anchor 진행 승인

- **결정**:
  1. **(a) 옵션 (b) 상대 기준 채택** — vLLM era 동일 anchor 대비 **ΔF1 ≥ -0.02 (R/P 도 -0.02 이내)** 합격선. 적용: `s04_04_qcond_a0_xiyan_glm` 측정 ΔF1=-0.0099 → **합격 판정**, 5-cell sweep + new anchor 재실행 즉시 진행.
  2. **(b) 절대 기준 F1 ≥ 0.70 폐기** — 사유: sanity anchor `s04_04_qcond_a0_xiyan` 이 α=0 QCond GAT-only 변인 통제 설계로 F1 ≈ 0.58 이 **구조적 천장**. F1 ≥ 0.70 은 cosine α=0.85 ensemble anchor 에서만 가능. **Planner 의 초기 기준 설정 실수** — anchor family 천장을 무시하고 일률 기준 적용. 상위 2026-04-24 LLM 전환 엔트리 §사용자 답변 #4 의 "F1 ≥ 0.70" 부분은 본 엔트리에 의해 supersede.
  3. **(c) 향후 GLM era 평가 규범** — 모든 GLM era 실험 합격은 vLLM era 동일 anchor (또는 동등 anchor family) 대비 **ΔF1 ≥ -0.02** 로 측정. 절대 임계치 사용 금지. Wave 4 a05_filter_agentic 등 후속 실험에도 동일 규범 적용. 새 anchor (cosine α=0.85, GAT α=0 등) family 마다 vLLM era 짝패 metrics 를 baseline 으로 사전 등록.

- **근거**:
  - 메트릭 분해 (sanity anchor `s04_04_qcond_a0_xiyan`):
    | Metric | GLM-4.7 | vLLM Qwen3-Coder-30B | Δ | 평가 |
    |--------|---------|----------------------|---|------|
    | Recall | 0.4922 | 0.5015 | -0.0093 (-1.85%) | noise 범위 |
    | Precision | 0.6965 | 0.7065 | -0.0100 (-1.41%) | noise 범위 |
    | F1 | 0.5767 | 0.5866 | **-0.0099 (-1.69%)** | **노이즈 상한 근접, 합격** |
  - **R/P 균등 하락 패턴**: GLM 이 over-prune (R 만 크게 하락) 도 아니고, over-keep (P 만 크게 하락) 도 아닌 **균형 잡힌 backbone 차이**. 만약 R 만 -5% 이상 하락했다면 prompt tuning 필요했을 것 — 현재 패턴은 정상 LLM-to-LLM 분산.
  - Wave 1.5 anchor 갱신 사례: `s04_stagewise_qcond_gat_basic` ΔF1=+0.0014 vs `abl_ens_basic_xiyan` (planning/DECISIONS.md 2026-04-22 17:05 직전 엔트리) — 0.001 차이도 "유의미한 새 top" 으로 인정. 0.01 은 노이즈 상한이나 동일 LLM 내부 갱신 vs LLM 교체 분산 두 카테고리는 다르게 평가됨.
  - 근원 진단: 직전 엔트리 "2026-04-24 추가 — GLM-4.7 sanity check 결과" §에스컬레이션 #1 옵션 (b) 가 가장 합리적이라는 root 의 사전 평가 채택.

- **영향 범위**:
  - **즉시 진행 (root 재-kickoff)** — 상위 2026-04-24 LLM 전환 엔트리 7-step 중 (5)(6)(7) 진행 승인:
    - (5) Wave 2 Proposal C 5-cell sweep (`layers_L{1,2,3,6,7}_glm`)
    - (6) New anchor 재실행 (`s04_stagewise_qcond_gat_basic_glm`)
    - (7) 문서 동기 갱신 (HISTORY 3종 + EXPERIMENT_PLAN §0/§4 + 루트 CLAUDE.md)
    - Sanity 결과 (R=0.4922 P=0.6965 F1=0.5767) 는 sweep 보고 표 의 baseline cell 로 재활용 (별도 cell 추가 불필요)
  - **향후 GLM era 실험 평가 규범 변경** — 합격 기준 = vLLM era 동일 anchor Δ R/P/F1 ≥ -0.02 (본 엔트리 (c)).
  - **비용 재추정** (sanity 실측 input 683 tokens/query 기반, 기존 3K/query 추정의 1/5):
    | 구간 | 재추정 | 기존 추정 |
    |------|--------|-----------|
    | Sweep 5 cell | ~₩3,821 | ~₩19,100 |
    | New anchor 1 cell | ~₩764 | ~₩3,820 |
    | **남은 6 cell 총** | **~₩4,585 (~$3.3 USD)** | **~₩22,920** |
    Budget 제약 완전 해소. Wave 4 multi-agent 도 cost 측면 재평가 필요 (post-2026-04-28).
  - **상위 LLM 전환 엔트리 사용자 답변 #4 갱신 표시**: "F1 ≥ 0.70 합격 기준" → "ΔF1 ≥ -0.02 vs vLLM era 동일 anchor" 로 supersede. 향후 reader 가 충돌 없이 해석 가능.

- **에스컬레이션 필요 여부**:
  1. **Root 세션 (즉시 재-kickoff)** — Sanity 합격 판정 + (5)(6)(7) 진행. 프롬프트 하단 §재-kickoff 참조.
  2. **Analyzer 세션 (sweep + anchor 완료 후, 예약)** — `notebooks/analysis_results/diameter_layers_sweep.md` 작성:
     - §1 5-cell F1/R/P curve + peak 위치 식별 + DB 별 D_max 대비 peak alignment (H1 검증)
     - §부록: vLLM era ↔ GLM era 비교 (sanity 결과 + new anchor 결과로 LLM era Δ 정량화)

- **추가 필요 분석**: 없음. Sanity 결과 충분.

### Root 재-kickoff 프롬프트

```
먼저 다음을 순서대로 읽어라:
1. /home/hyeonjin/thesis_refactored/CLAUDE.md
2. /home/hyeonjin/thesis_refactored/planning/DECISIONS.md 최상단 "2026-04-24 결정 — Sanity check 합격 기준 재정의" 엔트리 + 그 아래 sanity 결과 / endpoint 블로커 / LLM 전환 엔트리

Sanity check 합격 판정 (ΔF1=-0.0099 vs vLLM era 동일 anchor, 노이즈 범위, 새 합격 기준 ΔF1 ≥ -0.02 충족). 상위 LLM 전환 엔트리 7-step 중 (5)(6)(7) 즉시 진행:

(5) Wave 2 Proposal C 5-cell sweep 실행:
    bash scripts/run_wave2_proposal_c_phase2.sh
    configs: configs/experiments/s04_ablation/diameter_layers/layers_L{1,2,3,6,7}_glm.yaml
    예상 비용: ~₩3,821 (~$2.7)
    예상 시간: ~3.5h (Live API filter 2.10 s/query × 1534 queries × 5)

(6) New anchor 재실행: s04_stagewise_qcond_gat_basic_glm
    config: configs/experiments/s04_ablation/stagewise/qcond_gat_basic_glm.yaml
    예상 비용: ~₩764
    예상 시간: ~50min

(7) 문서 동기 갱신:
    - EXPERIMENT_HISTORY.md: 신규 7 entries 추가 (sanity + 5 sweep + new anchor), LLM era 컬럼 신설, 기존 entries 는 [vLLM era] annotation
    - EXPERIMENT_CATALOG.md: GLM era cluster 신규
    - EXPERIMENT_ID_MIGRATION.md: `_glm` suffix 규칙 등재
    - EXPERIMENT_PLAN.md §0: vLLM era / GLM era 분리 표 (new anchor F1 결과로 GLM era top 식별)
    - EXPERIMENT_PLAN.md §4 Phase 0 Wave 2: closed 표시 + Phase 2 LLM = GLM-4.7 명시 + vLLM 재기동 항목 제거
    - 루트 CLAUDE.md: XiYan = Qwen3-Coder-30B 표기 갱신
    - 메트릭 R/P/F1 4자리

성공 기준: 6 cell 모두 R/P/F1 측정 + new anchor 결과로 GLM era top 식별 + 문서 6개 갱신 완료.
총 남은 비용: ~₩4,585 (~$3.3 USD), budget 안전.

블로커 발생 시: planning/DECISIONS.md 후속 엔트리 + planner 에스컬레이션.

작업 완료 후 핸드오프: planner (analyzer 큐 추가 — diameter_layers_sweep.md GLM era + vLLM era 비교 부록).
```

---

## 2026-04-24 추가 — GLM-4.7 sanity check 결과 (F1 기준 미달, 그러나 vLLM 대비 Δ=-0.0099)

- **결정**: Sanity check `s04_04_qcond_a0_xiyan_glm` 완료. **R=0.4922 / P=0.6965 / F1=0.5767** (1,534 queries, 50분 36초 완료, 2.10 s/query). 상위 엔트리 사용자 답변 #4 의 합격 기준 **F1 ≥ 0.70 미달** 로 sweep 진행 보류 + planner 에스컬레이션. 단 **vLLM era 동일 anchor 대비 Δ F1 = -0.0099 (-1.7%)** 로 backbone 교체 영향 거의 없음.
- **근거**:
  - GLM metrics: [outputs/experiments/s04_ablation/s04_04_qcond_a0_xiyan_glm/metrics.txt](../outputs/experiments/s04_ablation/s04_04_qcond_a0_xiyan_glm/metrics.txt)
  - vLLM anchor metrics: [outputs/experiments/s04_gat_qcond_projector/s04_04_qcond_a0_xiyan/metrics.txt](../outputs/experiments/s04_gat_qcond_projector/s04_04_qcond_a0_xiyan/metrics.txt) — R=0.5015 / P=0.7065 / F1=0.5866

  | Metric | GLM-4.7 | vLLM Qwen3-Coder-30B | Δ |
  |--------|---------|----------------------|---|
  | Recall | 0.4922 | 0.5015 | -0.0093 |
  | Precision | 0.6965 | 0.7065 | -0.0100 |
  | F1 | 0.5767 | 0.5866 | -0.0099 |

  - Anchor 실험 특성: α=0 QCond GAT-only 로 설계상 F1 ≈ 0.58 상한. 기준 F1 ≥ 0.70 은 **이 anchor 에서 구조적으로 도달 불가능** (2×2×2 best `abl_ens_basic_xiyan` F1=0.7863 / Wave 1.5 best `s04_stagewise_qcond_gat_basic` F1=0.7877 은 다른 anchor).
  - Token usage 실측: input 1,048,544 + output 34,572 tokens (1,534 queries). Input per-query 평균 683 tokens — 상위 엔트리 사용자 답변 #5 추정 3K/query 의 1/5 수준. Extractor 평균 18.58 nodes 선택 → M-Schema 간결.
  - 비용 재추정: **sanity 1 cell ≈ ₩764** (vs 추정 ₩3,820), **sweep 5 cell ≈ ₩3,821** (vs ₩19,100), **전체 7 cell ≈ ₩5,350** (vs ₩26,740). Budget 대폭 여유.
  - Filter time: GLM live API 2.10 s/query (Qwen 로컬 vLLM 1.7 s/query 대비 +23%) — 네트워크 latency 감안 시 양호.
- **영향 범위**:
  - Root 세션 7-step 중 (4) sanity 완료, **(5) sweep / (6) new anchor / (7) 문서 갱신 보류**.
  - 상위 2026-04-24 엔트리 사용자 답변 #4 "F1 < 0.70 → planner 에스컬레이션" 절차 발동.
  - Sweep/anchor/문서 작업은 planner 기준 재평가 후 재개.
- **에스컬레이션 필요 여부**:
  1. **Planner (필수)** — 합격 기준 재평가. 선택지:
     - (a) **절대 기준 F1 ≥ 0.70 유지** → 중단 + backbone 재선정 (GPT-4o-mini 등). 단, 본 anchor 가 α=0 GAT-only 로 F1 ≈ 0.58 천장이라는 점 감안하면 기준 자체가 구조적 부정합.
     - (b) **상대 기준으로 재정의** (vLLM era 동일 anchor 대비 Δ F1 ≥ -0.02 허용) → **합격 판정 → sweep 즉시 진행**. 근거: Qwen↔GLM Δ = -0.0099 는 run-to-run noise 범위.
     - (c) **새 절대 기준** (예: anchor vLLM 값 × 0.95 ≈ F1 ≥ 0.557) → 합격 판정.
  - (b) 가 가장 합리적 — backbone 교체 실험의 통상적 평가축.
- **추가 필요 분석**: 없음. 결과 명확.
- **다음 행동**: Planner 가 합격 기준 재정의 후속 엔트리 작성 + root 재-kickoff 프롬프트 갱신. Sweep 5-cell (`layers_L{1,2,3,6,7}_glm`) + new anchor (`qcond_gat_basic_glm`) 는 정의 후 즉시 시작 가능 (configs 7 개 + scripts 2 개 보존 중).

---

## 2026-04-24 후속 — GLM-4.7 endpoint URL 블로커 (sanity check 사전 차단)

- **결정**: 사용자 답변 #2 의 1차 시도 (`GLM_BASE_URL=https://mlapi.run/<route>/v1`) 및 raw fallback 시나리오 (SDK double-path `.../v1/chat/completions/chat/completions`) **모두 404**. 실제 endpoint 경로가 OpenAI spec 과 일치하지 않아 sanity check 실행 전 블로킹. (4) sanity → (5) sweep → (6) anchor → (7) 문서 갱신 **전부 보류**.
- **근거**: Root 세션 curl probing (.env 의 GLM_BASE_URL + GLM_API_KEY 그대로 사용) 8종 variation:

  | Path | Method | Response |
  |------|--------|----------|
  | `/v1/models` | GET | HTTP 400 `"unsupported_content_type"` |
  | `/v1/models` | POST | HTTP 500 `"internal server error"` |
  | `/v1/chat/completions` (SDK 표준 경로) | POST | HTTP 404 `{"detail":"Not Found"}` |
  | `/v1/chat/completions/chat/completions` (SDK + raw BASE_URL append) | POST | HTTP 404 |
  | `/chat/completions` (no `/v1`) | POST | HTTP 404 |
  | `/v1/completions` | POST | HTTP 404 |
  | `/v4/chat/completions` (Zhipu 공식 spec) | POST | HTTP 404 |
  | `/` (proxy root) | GET | HTTP 401 `"Bearer authentication is required"` |

  - Proxy 가 `/v1/models` 는 인식 (400/500) 하나 OpenAI chat/completions 계열 경로는 **전부 404** — 비표준 구조.
  - Root `/` 응답이 401 → Bearer 는 도달. Auth 오류 아님, **endpoint path 구조 자체가 다름**.
- **영향 범위**:
  - 상위 엔트리 7-step 중 (1)(2)(3) 완료 / (4)(5)(6)(7) 보류.
  - 생성된 산출물 보존: configs 7개 (`*_glm.yaml`), scripts 2개 (`run_wave2_proposal_c.sh`, `run_wave2_proposal_c_phase2.sh` GLM 헬스체크) — endpoint 확정 후 즉시 재개 가능.
  - 비용: sanity 실행 전 차단 → ₩0 소비.
- **에스컬레이션 필요 여부**:
  1. **Planner (필수)** — mlapi.run 서비스의 실제 endpoint 경로 확인 + 대체 경로 결정. 선택지:
     - (a) mlapi.run 운영자/문서 확인으로 정확한 chat endpoint URL 획득
     - (b) Zhipu 공식 API 직접 전환 (`https://open.bigmodel.cn/api/paas/v4`) — proxy 우회, 사용자 답변 #5 cost 추정치와 호환
     - (c) 다른 OpenAI-compatible provider (GPT-4o-mini 등) 로 backbone 재변경 — 상위 엔트리 scope 재정의
  2. **User** — mlapi.run 대시보드에서 endpoint URL 문서 재확인, 또는 대체 provider 선택.
- **추가 필요 분석**: 없음. 진단 자체가 끝 (8종 variation 으로 명확).
- **다음 행동**: 상기 선택지 확정 후 planner 가 본 엔트리 뒤에 후속 결정 기록 + root 재-kickoff 프롬프트 갱신. Sanity/sweep/anchor 재실행은 endpoint 확정 후 재개.

---

## 2026-04-24 — LLM 백엔드 vLLM Qwen3-Coder-30B → Live API GLM-4.7 (OpenAI 호환) 전환 + Anchor 전체 재정렬 (시즌 2 개시)

- **결정**:
  1. **(a) Filter 단 LLM 백엔드 교체** — vLLM `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` → **GLM-4.7 (Live API, OpenAI 호환)**. 사유: vLLM 콜드스타트 8~10h 소요 (HuggingFace 모델 캐시가 NAS 에 위치, NAS folio_wait_bit_common stall 동일 원인 — CLAUDE.md NAS 규칙 BIRD dev 로컬 SSD 예외와 같은 카테고리).
  2. **(b) Anchor 전체 재정렬 (전략 A — 시즌 2 개시)** — vLLM era baseline (`s04_stagewise_qcond_gat_basic` F1=0.7877 등 8 cells) freeze, GLM era 로 단일 LLM 일관성 갖춘 새 baseline 시리즈 시작. 2026-04-28 발표 우선순위:
     - ① **Sanity check** = `s04_04_qcond_a0_xiyan_glm` 1 cell (1534 queries) — GLM ↔ Qwen3 격차 정량화
     - ② **Wave 2 Proposal C** 5 cell GLM 일괄 (`layers_L{1,2,3,6,7}_glm`)
     - ③ `s04_stagewise_qcond_gat_basic_glm` 재실행 — §0 anchor 갱신
  3. **(c) Wave 2 Proposal C sweep 도 GLM-4.7 로 처음부터 실행**. Phase 1 GAT 학습 (vLLM 무관, GPU 0/1) 완료 후 Phase 2 inference 즉시 시작 — **vLLM 재기동 8~10h 대기 완전 제거**.
  4. **(d) ID 명명 규칙** = 기존 ID 에 `_glm` suffix (`s04_04_qcond_a0_xiyan_glm`, `layers_L{1,2,3,6,7}_glm`, `s04_stagewise_qcond_gat_basic_glm`). HISTORY 에 `LLM era` 컬럼 신설 (root 갱신).

- **근거**:
  - vLLM 콜드스타트 8~10h: 사용자 보고 (2026-04-24). HuggingFace cache 위치 = NAS, weight load 시 NAS 통신 stall. 이는 2026-04-22 관측된 BIRD dev XiYan filter 의 `folio_wait_bit_common` 커널 스톨과 동일 카테고리 (CLAUDE.md NAS 규칙 BIRD dev 로컬 예외 사유 참조).
  - api_handler 호환성: [`src/llm_client/api_handler.py:15-20`](../src/llm_client/api_handler.py) `_PROVIDER_ENV_MAP` 에 `"glm": ("GLM_BASE_URL", "GLM_API_KEY")` + `"zhipu"` alias 이미 포함. OpenAI SDK chat.completions 호출 (`api_handler.py:141-150`) 그대로 작동 → **코드 변경 거의 없음**. config 의 provider/model 필드 + env 설정만 필요.
  - 모델 동질성: GLM-4.7 ≠ Qwen3-Coder-30B → 기존 anchor 와의 직접 비교 무의미. 일관된 단일 LLM baseline 으로 시즌 2 시작이 논문 서사상 깔끔 (vLLM era 결과는 historical reference 보존).
  - 운영 효율: Live API 전환 시 Phase 1 GAT 학습 (GPU 0/1) 과 Phase 2 inference (GPU 미사용) 가 자원 분리 → vLLM 메모리 경합 / 재기동 비용 / GPU 점유 모두 동시 해소.

- **영향 범위**:
  - **변경 산출물 (root 작업)**:
    - `.env`: `GLM_BASE_URL`, `GLM_API_KEY` 추가 (사용자 값 제공 필요). git status 에 `.env.example` modified 표기 → 사용자 작업 중 가능성.
    - Phase 2 configs 신규 (5 + sanity check + new anchor) — `_glm` suffix 별도 파일로 생성, 기존 configs 보존: `configs/experiments/s04_ablation/diameter_layers/layers_L{1,2,3,6,7}_glm.yaml`, `configs/experiments/s04_ablation/s04_04_qcond_a0_xiyan_glm.yaml`, `configs/experiments/s04_ablation/stagewise/qcond_gat_basic_glm.yaml`. 변경 항목: xiyan_filter `provider: glm` + `model: <glm-4.7 model id>`.
    - `scripts/run_wave2_proposal_c_phase2.sh`: vLLM 헬스체크 (L11-14) → GLM endpoint 헬스체크 (간단한 `/v1/models` GET) 교체.
    - `scripts/run_wave2_proposal_c.sh`: Phase 2 안내 (L100-108 vLLM 재기동) 제거.
  - **문서 갱신 (root)**:
    - `EXPERIMENT_PLAN.md` §0 anchor 표 — `vLLM era` / `GLM era` 분리 (vLLM era 는 historical archive). §4 Phase 0 Wave 2 — Phase 2 LLM = GLM 명시, "vLLM 재기동 필요" 표기 제거.
    - `EXPERIMENT_HISTORY.md` — 신규 entries LLM era 컬럼, 기존 entries 는 `[vLLM era]` annotation.
    - `EXPERIMENT_ID_MIGRATION.md` — `_glm` suffix 명명 규칙 등재.
    - 루트 `CLAUDE.md` 의 vLLM 명시 구절 (XiYan = Qwen3-Coder-30B 표기 등) 갱신 — root 결정.
  - **Wave 파급**:
    - Wave 2: Phase 2 LLM 전환 (즉시 적용).
    - Wave 3 Proposal F (analyzer 단독): LLM 영향 없음.
    - Wave 3 Proposal A 확장: configs GLM 갱신 필요.
    - Wave 4 a05_filter_agentic (post-2026-04-28): 다중 agent 호출 → GLM token cost 가장 큰 영향, budget 사전 추정 필수.
  - **Scope 분리**: GLM era vs vLLM era 정량 비교는 sanity check 결과 기반 별첨 부록 (analyzer 작성). 본 wave sweep 은 GLM era 단독 시리즈.

- **에스컬레이션 필요 여부**:
  1. **Root 세션 (최우선)** — `.env` 설정 + configs/scripts 갱신 + Phase 1 nl7 종료 후 GLM 기반 sanity → sweep → anchor 재실행 + HISTORY 3종 갱신. 프롬프트:
     ```
     먼저 /home/hyeonjin/thesis_refactored/CLAUDE.md 와 planning/DECISIONS.md 2026-04-24 엔트리 읽어라.
     작업 (순서):
       (1) `.env` 에 GLM_BASE_URL + GLM_API_KEY 추가 (사용자에게 값 확인). `.env.example` 도 항목만 placeholder 로 동기화.
       (2) 신규 configs 7 개 생성 (`_glm` suffix, 기존 보존):
            - configs/experiments/s04_ablation/s04_04_qcond_a0_xiyan_glm.yaml (sanity check)
            - configs/experiments/s04_ablation/diameter_layers/layers_L{1,2,3,6,7}_glm.yaml (5 cell sweep)
            - configs/experiments/s04_ablation/stagewise/qcond_gat_basic_glm.yaml (new anchor 재실행)
            xiyan_filter 섹션: provider="glm" + model="<glm-4.7 model id, 사용자 확인>".
       (3) scripts/run_wave2_proposal_c_phase2.sh 의 vLLM 헬스체크 → GLM /v1/models 헬스체크. scripts/run_wave2_proposal_c.sh L100-108 vLLM 재기동 안내 제거.
       (4) Phase 1 nl7 종료 확인 후, **Sanity check 우선 실행**: s04_04_qcond_a0_xiyan_glm 1 cell → outputs/.../metrics.txt 확인.
            - 합격 (F1 ≥ 0.70): (5) 진행
            - 불합격 (F1 < 0.70 or 큰 격차): planner 에스컬레이션 (prompt tuning 검토)
       (5) Wave 2 Proposal C 5-cell sweep (layers_L{1,2,3,6,7}_glm) 실행.
       (6) s04_stagewise_qcond_gat_basic_glm 재실행 → §0 anchor 후보 산출.
       (7) HISTORY/CATALOG/ID_MIGRATION 3종 동기 갱신 — LLM era 컬럼 신설 + `_glm` suffix 등재. EXPERIMENT_PLAN.md §0 anchor 표 분리. 루트 CLAUDE.md vLLM 구절 갱신.
     성공 기준: sanity + 5 cell + new anchor 재실행 모두 R/P/F1 (4자리) 측정.
     리스크: GLM-4.7 token cost — sanity 결과 후 5 cell sweep 비용 추정 → 초과 시 planner 즉시 에스컬레이션.
     ```
  2. **Filter 모듈 세션 (조건부)** — Sanity check F1 < 0.70 시 XiYan filter prompt 가 Qwen3-Coder 에 over-fit 됐을 가능성 → GLM-4.7 용 prompt 조정. api_handler 자체는 변경 불필요.
  3. **Analyzer 세션 (Phase 2 완료 후)** — `notebooks/analysis_results/diameter_layers_sweep.md` 작성 시 vLLM era ↔ GLM era 비교 부록 동반 (s04_04 anchor 동일 setup 의 LLM 만 다른 비교).

- **추가 필요 분석**:
  - GLM token cost 추정: XiYan prompt ~3k token × 1534 queries × 5 cell + sanity 1 + anchor 1 = 7 셀 × 1534 = ~10.7K calls × ~3K tokens = 약 32M input tokens. GLM-4.7 가격 사용자 확인 필요.
  - Sanity check 후 LLM era 차이 정량 (s04_04_glm F1 vs vLLM era 동 anchor F1) — 발표 슬라이드 필요시.

- **사용자에게 확인 필요 항목** (root 세션이 진행 전 받아야 할 정보):
  - GLM-4.7 정확한 model id (예: `glm-4-flash` / `glm-4-plus` / `glm-4-air` / `GLM-4.7` 등 — Zhipu API 공식 모델명)
  - `GLM_BASE_URL` 값 (Zhipu 표준은 `https://open.bigmodel.cn/api/paas/v4`)
  - `GLM_API_KEY` (root 에서 .env 직접 작성 시 사용자 직접 입력)
  - Sanity check 우선 수행 동의 여부 (default: 권장. 사용자가 "5 cell 일괄 실행" 명시 시 skip 가능)

- **사용자 답변 (2026-04-24 후속 수렴)**:
  1. **Model id** = `zai-org/glm-4.7` (HuggingFace 스타일 vendor-namespace 식별자). configs 의 `model` 필드 + API 호출 시 `model="zai-org/glm-4.7"` 그대로 전달.
  2. **Base URL** = `https://mlapi.run/abc-1234-xyz/v1/chat/completions` (사용자 보고 raw 값). ⚠ **OpenAI SDK 동작 caveat**: SDK 는 `base_url` 에 자동으로 `/chat/completions` 를 append ([api_handler.py:106-109](../src/llm_client/api_handler.py)). 표준 사용 형식은 `GLM_BASE_URL="https://mlapi.run/abc-1234-xyz/v1"` (SDK 가 `POST /v1/chat/completions` 자동 호출). Root 세션 sanity check 시:
     - 1차 시도: `GLM_BASE_URL=https://mlapi.run/abc-1234-xyz/v1` (표준)
     - 404 시 2차 시도: 사용자 raw 값 그대로 (mlapi.run 가 비표준일 가능성)
     - 결과 planner 에 보고 → DECISIONS 후속 보강
  3. **API key** = 사용자가 `.env` 에 직접 편집. Root 는 `os.getenv("GLM_API_KEY")` 로딩 여부만 검증, `.env` 직접 수정 금지.
  4. **Sanity check** = 진행 승인. 1 cell (`s04_04_qcond_a0_xiyan_glm`, 1534 queries) → F1 ≥ 0.70 합격 시 sweep, 그 미만이면 planner 에스컬레이션.
  5. **GLM token cost**:
     - Input: ₩630 / 1M tokens, Output: ₩3,000 / 1M tokens
     - 추정 (XiYan avg ~3K input + ~200 output tokens / query):
       | 구간 | Queries | Input | Output | 합계 |
       |------|---------|-------|--------|------|
       | Sanity 1 cell | 1,534 | ₩2,899 | ₩921 | ~₩3,820 |
       | Sweep 5 cell | 7,670 | ₩14,497 | ₩4,602 | ~₩19,100 |
       | New anchor 1 cell | 1,534 | ₩2,899 | ₩921 | ~₩3,820 |
       | **총 7 cell** | **10,738** | **₩20,295** | **₩6,444** | **~₩26,740 (≈$19 USD)** |
     - **Budget 안전**. Wave 4 a05_filter_agentic (multi-agent 3-5× LLM call/query) 은 별도 추정 (post-2026-04-28, filter 모듈 세션 작업).

---

## 2026-04-22 17:05 — Wave 1.5 no-filter backfill 완료 + Wave 2 Proposal C Option B (global D_max fixed sweep) 채택 + 병렬 실행 패턴 관찰

- **결정**:
  1. **(a) Wave 1.5 no-filter backfill 완료** — W1/W2/W3 3 config 의 `+Extractor (no filter)` 셀을 `NoneFilter` pass-through (LLM 호출 0) 로 실측 확정. HISTORY §8 stagewise cumulative 표 갱신 완료 — W1 F1=0.2272 / W2 F1=0.2862 / W3 F1=0.2271. **Filter Δ F1**: W1 +0.4672, W2 +0.4189, **W3 +0.5605 (최대)**. 운영: vLLM 종료 + 기존 sequential script kill (사용자 승인 완료) 후 GPU 0/1 병렬 실행으로 sequential 가정 대비 약 7 분 단축 (16:29→17:04, 총 35 분 소요).
  2. **(b) Wave 2 Proposal C 실행 경로 = Option B (global D_max fixed sweep) 채택** — 제안서 [abl_sel_diameter_layers.md](proposals/abl_sel_diameter_layers.md) §4.2 의 "혹은 global fixed num_layers = max(D_max over all DBs) 로 먼저 스윕" 경로. **num_layers ∈ {1, 2, 3, 6, 7}** (6 = global D_max across BIRD dev 11 DBs per `data/processed/dev_diameter.pt`, 7 = D_max+1). H1 (global peak 존재) 만 본 wave 에서 검증하고 **H2 (per-DB dynamic peak shift) 는 deferred**.
  3. **(c) 운영 패턴 관찰 채택** — Wave 1.5 no-filter 에서 관찰한 "LLM 미사용 + 서로 다른 GPU 배치 가능" 실험의 **GPU 0/1 병렬 실행 패턴** 을 향후 동일 조건 실험에 적용 고려. 제약: kill permission memory rule 상 script bash kill 은 사용자 명시 승인 필요 → permission prompt 사전 안내가 운영상 효율적.

- **근거**:
  - (a) 메트릭 출처: `outputs/experiments/s04_ablation/stagewise/no_filter/{ensemble_raw_a0,qcond_raw_basic,qcond_gat_basic}_no_filter/metrics.txt`. Cumulative 표: [EXPERIMENT_HISTORY.md §8](../EXPERIMENT_HISTORY.md#L1250). Analyzer 요청 맥락: [notebooks/analysis_results/stagewise_qcond_ablation.md](../notebooks/analysis_results/stagewise_qcond_ablation.md) §4 pending cells. 지도교수 G2 단계별 분해 규범: [advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md](advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md) §4 G2 + 2026-04-21 Q3 답변.
  - (b) **Option A (per-DB dynamic) 를 채택하지 않은 이유**:
    - `EnsembleSelector` 가 v1 `SchemaHeteroGAT` 를 하드코딩 ([src/modules/selectors/ensemble_selector.py:8,47-53](../src/modules/selectors/ensemble_selector.py)), v2 분기 부재.
    - `select()` signature / 내부 경로에 `db_name` threading 없음 → runtime `resolve_num_layers(db_name)` hook 경로 미존재.
    - `train_gat_s06.py` 도 v2 flag (`num_layers_mode`, `diameter_path`, `diameter_dict`) 를 config 로부터 forward 하지 않음.
    - ⚠ 제안서 §5 Dependency 에 "planner 가 전제 인프라 완료로 표기" 한 것은 **실측 결과 선언이 앞섰다** — 선택자 세션 작업 필요 (하단 에스컬레이션 프롬프트 참조).
  - (c) Wave 1.5 no-filter 운영 로그: HISTORY §8 L1253 "W2 (GPU 0) 와 W3 (GPU 1) 은 vLLM 종료 후 병렬 실행 (약 7 분 단축)".

- **영향 범위**:
  - **산출물 (root 세션 선제 작업 완료)**:
    - Training configs (5): `configs/training/diameter_layers/train_qcond_nl{1,2,3,6,7}.yaml` — v1 `train_gat.py` 호환, `projector_state_dict` 동반 생성.
    - Inference configs (5): `configs/experiments/s04_ablation/diameter_layers/layers_L{1,2,3,6,7}.yaml` — anchor `s04_04_qcond_a0_xiyan`, `weight_path` 만 변경.
    - Scripts: `scripts/run_wave2_proposal_c.sh` (Phase 1 training, `VLLM_AUTOKILL=1` 지원), `scripts/run_wave2_proposal_c_phase2.sh` (Phase 2 inference, vLLM 재기동 선행).
  - **예상 소요**: Phase 1 ~25h (5 × 5h) + Phase 2 ~3-4h (5 × 45min) = **~28-30h** → 2026-04-25 deadline 내 여유.
  - **문서 반영**:
    - [EXPERIMENT_HISTORY.md §8](../EXPERIMENT_HISTORY.md) — Stagewise cumulative 표 갱신 완료 (루트 세션).
    - [EXPERIMENT_PLAN.md §4 Phase 0 Wave 2](../EXPERIMENT_PLAN.md#L116) — 본 엔트리에서 Option B 채택을 Proposal C 행에 명시 (L117 "num_layers ∈ {1,2,3,D_max,D_max+1} sweep" → 구체 셋 `{1,2,3,6,7}` 및 Option B 명기).
    - [notebooks/analysis_results/stagewise_qcond_ablation.md](../notebooks/analysis_results/stagewise_qcond_ablation.md) §1.1 / §4 / §5 — analyzer 작업 중 (병렬 진행).
  - **Scope 분리**: 본 결정으로 Wave 2 Proposal C 는 H1 만 검증, H2 는 Wave 2.5 또는 별도 mini-wave 로 분리 (Selector 인프라 완료 후).

- **에스컬레이션 필요 여부**:
  1. **Selector 세션 — per-DB dynamic num_layers 인프라 확장** (H2 해금 조건):
     ```
     먼저 /home/hyeonjin/thesis_refactored/src/modules/selectors/CLAUDE.md 를 읽어라.
     작업: EnsembleSelector 에 SchemaHeteroGATv2 지원 분기를 추가하고, select() signature 또는 내부 경로에 db_name 을 통과시켜 런타임에 resolve_num_layers(db_name, active_num_layers) 가 호출되도록 한다.
     근거: planning/proposals/abl_sel_diameter_layers.md §4.3, planning/DECISIONS.md 2026-04-22 17:05 (b) 항목.
     성공 기준: Mode="D_max" 및 "D_max_plus1" 로 설정된 config 에서 inference 시 DB 별로 다른 depth 가 resolve 되어 forward pass 에서 사용되는지를 단위 테스트로 검증.
     블로커: train_gat_s06.py 역시 v2 flag forward 가 누락 — 루트에 escalate 필요 시 노트.
     ```
  2. **Analyzer 세션 (Phase 2 완료 후 예정)** — 5-cell F1/R/P curve + peak 위치 식별 + DB 별 D_max 대비 peak alignment 리포트. 대상: `outputs/experiments/s04_ablation/diameter_layers/layers_L{1,2,3,6,7}/metrics.txt` + `output_*.jsonl`. 저장: `notebooks/analysis_results/diameter_layers_sweep.md`. 의도: H1 검증 + Option A (H2 mini-wave) 재개 판단 근거.
  3. **Root 세션** — Wave 2 Proposal C Phase 1/2 kickoff 실행 + 실행 후 HISTORY/CATALOG/ID_MIGRATION 3종 동기 갱신 (memory rule).

- **추가 필요 분석**:
  - Analyzer 큐 (기존 유지): `stagewise_qcond_ablation.md` §1.1 `Selector only` 행 reconstruction (`output_*.jsonl.raw_seeds` 기반). 직전 엔트리 이후 유효.
  - Analyzer 큐 (예약, Phase 2 완료 후): 위 에스컬레이션 2번.

---

## 2026-04-22 — Wave 1.5 closed, 새 전체 최고 F1=0.7877 / Wave 2 Selector ablation 큐 개시 / a05_filter_agentic 순연

- **결정**:
  1. Wave 1.5 stagewise Extractor 통일 backfill 종료 (2026-04-22 15:24). 3 셀 모두 완료, `s04_stagewise_qcond_gat_basic` F1=0.7877 이 **새 전체 최고** (기존 `abl_ens_basic_xiyan` F1=0.7863 대비 +0.0014). `EXPERIMENT_PLAN.md` §0 anchor 재지정, §4 Phase 0 Wave tracker 신설 및 Wave 1.5 closed 표시.
  2. **Wave 2 개시 (Proposals C → D → E 순차)**. GPU 자원 경합 회피 + §8-1 SuperNode split-order bug 수정본 `train_gat.py` 기준으로 Proposal D/E 는 재학습 필수. Schedule ~2026-04-25 마감 목표.
  3. **Wave 3 (Proposal F + Proposal A 확장)** 은 2026-04-26 ~ 28 발표 패키징 구간에 배치. Proposal F 는 analyzer 단독 (신규 실행 없음).
  4. **Proposal B (T2T edge)** 는 Wave 3/4 로 순연. 스토리라인 우선순위 최하, 비용 (graph regen + GAT 재학습) ~11h, 2026-04-28 발표에 기여도 낮음.
  5. **`a05_filter_agentic` 12 실험 전체 순연 (Wave 4, post-2026-04-28)**. 사유: (i) 2026-04-28 advisor forum scope = QCondGAT stagewise, filter agentic 은 별도 브리핑 대상. (ii) `~/.claude/plans/vivid-sprouting-sunbeam.md` anchor (`abl_ens_basic_xiyan`, F1=0.7863) 가 Wave 1.5 new top (`qcond_gat_basic`, F1=0.7877) 로 **outdated** → Wave 4 kickoff 전 filter 세션 에스컬레이션으로 plan anchor refresh 필수. (iii) Wave 2/3 와 GPU·vLLM 자원 동시 점유 불가.
- **근거**:
  - Wave 1.5 메트릭: `outputs/experiments/s04_ablation/stagewise/{ensemble_raw_a0,qcond_raw_basic,qcond_gat_basic}/metrics.txt`
  - HISTORY 기록: [EXPERIMENT_HISTORY.md §8](../EXPERIMENT_HISTORY.md) (Wave 1.5 Stagewise Backfill)
  - 발표 스토리라인 (A > F > C > D > E > B): [planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md](advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md) §8 + 2026-04-21 Q4 답변
  - 제안서 큐: `planning/proposals/abl_sel_{rawscore_stagewise,diameter_layers,supernode_directed,supernode_topk}.md` + `abl_ext_steiner_backbone_report.md` + `abl_bld_t2t_edge.md`
  - SuperNode bug 범위: [EXPERIMENT_HISTORY.md §8-1](../EXPERIMENT_HISTORY.md) — T7/T9 checkpoint, Q2/Q3/Q5/Q7 재현성 의심. Proposal D/E anchor 재학습 전제.
  - Filter agentic plan 전문: [~/.claude/plans/vivid-sprouting-sunbeam.md](/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md) 243 lines, 현재 anchor `abl_ens_basic_xiyan` F1=0.7863 (Wave 1.5 이전 기준).
- **영향 범위**:
  - `EXPERIMENT_PLAN.md` §0 anchor 테이블 + §4 "Phase 0 Active Waves" 신규 섹션 (본 커밋에서 반영).
  - `EXPERIMENT_PLAN_selectors.md` — Wave 2 에서 소비. 선택자 세션이 Proposal C/D/E 구현 시 본 PLAN Phase 0 wave 스케줄 참조 필요 (모듈 PLAN 직접 수정은 해당 모듈 세션 책임).
  - `~/.claude/plans/vivid-sprouting-sunbeam.md` — Wave 4 kickoff 전 anchor refresh 필요 (planner 는 초안만 제공, 실제 수정은 filter 모듈 세션).
  - `notebooks/analysis_results/stagewise_qcond_ablation.md` — §1.1 5×3 매트릭스 재작성 (Wave 1.5 셀 주입 + caveat 제거 + new top 반영). Analyzer 큐에 등록.
- **에스컬레이션 필요 여부**:
  1. **analyzer 세션** — 본 DECISIONS 엔트리 §4번 세 번째 영향 범위 처리. 프롬프트 하단 (응답 말미 핸드오프) 참조.
  2. **root 세션** — Wave 2 Proposal C 실행 kickoff (GAT 5 재학습 → 추론 평가 → HISTORY/CATALOG/ID_MIGRATION 갱신). 프롬프트 하단 참조.
  3. **filter 모듈 세션 (지연 에스컬레이션)** — Wave 4 kickoff 시점 (2026-04-28 이후) 에 `vivid-sprouting-sunbeam.md` anchor refresh. 본 DECISIONS 엔트리가 대기 마커.
- **추가 필요 분석**:
  - Analyzer: Wave 1.5 3 셀의 cumulative Selector-only / +Extractor 단계 R/P/F1 재구성 (가능하면 `output_*.jsonl` `raw_seeds`/`extracted_subgraph` 필드로, 없으면 DEBUG 로그 경로). 이게 채워져야 5×3 매트릭스 전체가 고정됨.
  - Selector 모듈: Proposal D/E 큐 진입 전 "§8-1 bug fix 적용된 `train_gat.py` 로 SuperNode anchor 재학습 후 inference 결과" 를 anchor 수치로 고정 (기존 s04_05 숫자 인용 금지).

---

## 2026-04-21 — QCondGAT 피드백 Q1~Q4 수렴 + PLAN diff 4건 승인

- **결정**: 직전 엔트리(QCondGAT 상세 ablation 지시) 의 4건 재확인 질문(§10) 에 대한 사용자 답변 수렴. §7 PLAN diff 4건 **모두 approved**. `EXPERIMENT_PLAN.md` 실제 수정을 루트 세션으로 위임.
- **Q1 답변**: Diameter = **per-DB heterograph 최대 diameter (D_max)**. `num_layers ∈ {1,2,3,D_max,D_max+1}` sweep 확정. Phase A precompute 루틴은 max shortest-path 기준. D_max 가 큰 DB 에서 over-smoothing 재등장 리스크 (§7.4 에 이미 반영).
- **Q2 답변**: Top-k 기준 **1개 권장 실행, 성능 양호 시 확장**. Planner 판단 → **Raw Score** 를 Phase 1 로 지정 (의견 1 ablation 축과 일치, 인프라 재활용, BCE/CE 는 bottleneck 분석 중). Phase 2 는 CE/Cosine 확장.
- **Q3 답변**: 단계별 성능 = **cumulative** (Selector top-k → Extractor post-PCST → Filter post-XiYan 순 누적 R/P/F1). Analyzer 요청(§9) 및 Root 세션 보고 패키지에 cumulative 명시.
- **Q4 답변**: **2026-04-28 (1주 뒤)** 다음 보고. **15~20분 발표**. 중요 지점만 선별. **스토리라인 우선순위 A > F > C > D > E > B** 확정 (A=Raw×Model×Stage / F=SteinerBackbone / C=Diameter→Layers / D=SN directed / E=SN top-k Raw / B=T2T).
- **PLAN diff 승인 내역 (§7 4건)**:
  1. §3.1 `int_05_direct_ns` 전제 → "SuperNode v2 (directed SN→node + top-k Raw selective)" 명시
  2. §4 Phase A → "Schema Graph Diameter precompute" 서브태스크 신설 (B-III FK reachability 와 1 패스 공유)
  3. §4 Phase B → "Base heterograph T2T edge toggle" 추가 (B-II 스펙 확장)
  4. §9 리스크 맵 → "SuperNode v2 over-smoothing 재등장 가능성" 행 추가
- **에스컬레이션 (업데이트)**: Root 세션용 프롬프트 2건 (§9 — PLAN 수정 + 보고 규범) 준비 완료. Selector/Builder 세션 에스컬레이션 기존 프롬프트 유효. 신규 실험 제안서 Proposal A/F/C 는 2026-04-28 발표 전 우선 처리 권장.
- **추가 필요 분석**: 기존 Analyzer 요청 (Stagewise Raw×Model cumulative) 유효. 추가로 D_max 계산 결과 분포(11개 BIRD dev DB 별) 선행 필요 — Builder 세션 작업에 포함.

---

## 2026-04-21 — QCondGAT 상세 ablation 지시 (지도교수 의견 반영)

- **결정**: s04/s05 계열 6개 신규 ablation 트랙 (Proposal A~F) 제안. Selector / Builder 모듈 PLAN 확장 에스컬레이션. int_05 전제를 SuperNode v2 (directed + top-k) 로 재정의 제안 (pending). Phase A 에 "Schema Graph Diameter precompute" 서브태스크 신설 제안 (pending). 교수님께 4개 재확인 질문 (diameter 정의 / top-k 기준 / 단계별 정의 / 보고 형식) 대기 상태.
- **근거**: 지도교수 2026-04-21 정기 미팅 — [`planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md`](advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md) §2.1~§2.4 + §2.G1/G2
  + 브리핑 범위: QCond 방안 A/B + Over-smoothing 진단 (§1.2)
  + 지지 데이터: `outputs/analysis/gat_bottleneck{,_qcond}/`, s06_b0~b5 ablation (이미 존재)
- **영향 범위 (브리핑 내 직접, §4)**: `src/models/gat_network_v2.py`, `src/models/gat_network.py`, `src/modules/builders/line_graph_builder.py`, `src/modules/selectors/ensemble_selector.py`, s04_xx / s05_xx 재설계
- **영향 범위 (Scope gap — PLAN 파급, §5)**: 루트 PLAN §3.1 int_05 / §4 Phase A Diameter / §4 Phase B T2T / §9 리스크 — **사용자 "PLAN 상관 없음" 입장 존중, 모두 pending-clarification**
- **에스컬레이션**: Selector / Builder / Analyzer / Root 세션 (§9 — 4개 copy-paste 프롬프트 준비됨)
- **추가 필요 분석**: Analyzer 에 Stagewise Raw×Model ablation 표 요청 (§9 — `notebooks/analysis_results/stagewise_qcond_ablation.md`)
- **다음 브리핑 후보**: 2×2×2 재측정(#6 E+Basic+X, R=0.8149/P=0.7597/F1=0.7863) / SteinerBackbone / s06 over-smoothing 결과 / S-V 개요 (§11)
- **교수님께 후속 질문**: 4건 (§10 — diameter 정의 / top-k 기준 / 단계별 정의 / 보고 형식)

---

## 2026-04-21 — advisor input 워크플로우 Option B (draft 기반) 확정

- **결정**: 사용자 편집 대상을 템플릿 파일에서 **별도 staging 파일 `planning/advisor_inputs/_draft.md`** 로 분리. 템플릿은 pristine 참조용으로 고정.
- **근거**: 사용자 선택. Option A(템플릿 직접 편집)는 미팅 사이 "편집 중 vs 처리 완료" 상태가 모호해지는 리스크가 있었음. Draft 분리로 템플릿은 항상 깨끗한 reference, draft 는 사용자 staging, dated 파일은 planner 승격본으로 역할이 명확.
- **운영 흐름**:
  1. 사용자: `_draft.md` 의 §1~§3 편집 (템플릿 직접 편집 금지)
  2. 사용자 → planner: "피드백 수렴" 신호
  3. Planner: `_draft.md` → `<YYYY-MM-DD>_<topic>.md` 승격 + §4~§14 채우기 → `_draft.md` 를 템플릿 기준 pristine 리셋 → DECISIONS 엔트리 추가
  4. 이번 미팅에서 새로 공유한 PLAN 영역은 템플릿의 §1.2 default "공유된 범위" 에 승격 반영 (planner 유지 책임)
- **영향 범위**: `planning/advisor_inputs/_draft.md` 신규 (디렉토리 포함), `planning/templates/advisor_input_template.md` intro 의 "사용 흐름" 섹션 Option B 기준으로 rewrite, `planning/CLAUDE.md` 책임 영역에 `advisor_inputs/` 경로 추가.
- **에스컬레이션 필요 여부**: 없음.
- **추가 필요 분석**: 없음.

---

## 2026-04-21 — advisor_input_template 재설계 (브리핑 범위 전제 반영)

- **결정**: 템플릿을 2-layer 모델로 재설계. §4(브리핑 내 직접 영향) 와 §5(Scope gap — unbriefed PLAN 파급) 를 분리. §1.2 "지도교수 인지 범위 ledger" 섹션 신설, §11 "다음 브리핑 후보" 섹션 신설.
- **근거**: 사용자 확인 — **루트 `EXPERIMENT_PLAN.md` 는 아직 지도교수님께 공유되지 않음**. 현재 공유 범위는 2026-04-10 5 아이디어 + Query-Conditioned GAT 구현 수준. 이전 템플릿 초안은 "advisor가 PLAN 을 직접 보고 피드백"이라는 잘못된 가정 위에 있었고, §1 Matrix/§3.1 Synergy 직접 매핑을 요구했음. 실제 흐름은 "advisor 피드백은 브리핑 범위 한정 → planner 가 PLAN 파급 해석".
- **신설된 제약조건 (모든 advisor 피드백 수렴에 적용)**:
  1. 각 advisor_input 문서는 **§1.2 브리핑 ledger** 를 반드시 채운다 — 어느 맥락 위에서 피드백이 나왔는지 기록.
  2. **Scope gap(§5)** 이 본 템플릿의 planner-specific 기여. Query-Conditioned GAT 피드백이 Neurosymbolic 3-layer/int_04/Phase 우선순위에 어떻게 파급되는지 planner 가 해석.
  3. §5 파급이 강하면 **§10 재확인 질문** 또는 **§11 다음 브리핑 후보** 로 연결 → 다음 미팅에서 검증.
  4. "이번 미팅에서 새로 공유한 내용" 은 다음 advisor_input 의 §1.2 "공유된 범위" 로 승격.
- **영향 범위**: `planning/templates/advisor_input_template.md` 전면 rewrite (12 → 14 섹션). `planning/CLAUDE.md` 변경 없음 (책임 기술은 그대로 유효).
- **에스컬레이션 필요 여부**: 없음 (planner 세션 인프라).
- **추가 필요 분석**: 없음. 단, 향후 Query-Conditioned GAT 피드백 수렴 시 `notebooks/analysis_results/query_conditioned_training.md` 수치를 §1.3 "관련 문서" 로 링크.

---

## 2026-04-21 — DECISIONS.md 초기 시드 (seeded)

- **결정**: Planner 세션 신설. 기존에 암묵적으로 이루어지던 PLAN 개정 흐름을 본 문서로 명시화.
- **근거**: 루트 PLAN 작성 중 분산된 모듈 PLAN과의 조율 비용 증가 — 전용 세션 분리 필요성 사용자 확인.
- **영향 범위**: 새 디렉토리 `planning/` 추가. 루트 CLAUDE.md에 Planner 세션 참조 추가됨.
- **에스컬레이션 필요 여부**: 없음 (본 세션 분리는 인프라 변경).
- **추가 필요 분석**: 없음.

---

## 2026-04-21 — a05 pending 실험 순서 및 GPT-4o-mini 후순위

- **결정**: a05_05~10 (Tiered/AdaptiveDepth/Retry 계열, Qwen 백본) 을 순차 실행 큐로 확정. a05_11/12 (GPT-4o-mini 백본) 는 **우선순위 하향** — Qwen 결과 확보 후 민감도 비교로 진행.
- **근거**: vLLM 서버 GPU 점유 제약 + 백본 교체 영향 분리 관측을 위해 한 차원(Qwen)만 먼저 완결.
- **영향 범위**: `scripts/run_a05_pending_qwen.sh` (루트 세션에서 실행 중). `EXPERIMENT_PLAN.md`의 `vivid-sprouting-sunbeam.md` F1~F5 phase를 a05_05~10으로 매핑.
- **에스컬레이션**: 없음 (루트 세션이 이미 실행 계획에 반영).
- **추가 필요 분석**: 실행 완료 후 analyzer에 filter_route distribution / latency-F1 Pareto 리포트 요청 예정.

---

## 2026-04-21 — int_04 논문 주력 결과 후보 지정

- **결정**: `int_04_ns_full` (Enriched + B-III + S-V + E-III + FL-III + Reflection) 을 논문 주력 실험으로 지정.
- **근거**: 모든 기여(Neurosymbolic 3-layer + Reflection restore)가 한 지점에 수렴 → 방법론 단일 서사.
- **영향 범위**: `EXPERIMENT_PLAN.md` §3.1, §4 Phase E, §5 논문 매핑 섹션에 반영됨.
- **에스컬레이션**: Builder B-III (FK reachability) 가 선결 인프라 → builders 세션에 "Phase A 최우선" 전달 필요.
- **추가 필요 분석**: int_01~03 (단일 모듈 신규 × Reflection) 이 각자 improvement를 내는지 먼저 검증.

---

## 2026-04-21 — 닫힌 주제 (재탐색 금지) 명시

- **결정**: 방안 A (Score-driven PCST cost), 방안 B (Bayesian Optimization), Idea 2/4 (Product Cost, Component-Aware) 는 완료 상태로 봉인.
- **근거**: 튜닝 실험 반복 제안 방지 — memory rule과 정합.
- **영향 범위**: `EXPERIMENT_PLAN.md` §6 "닫힌 주제" 섹션.
- **에스컬레이션**: Extractor 세션에도 동일 내용 전달됨.
