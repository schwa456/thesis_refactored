# Full Framework Experiment Plan — 9 아키텍처 제안 통합 로드맵

> **이 문서의 역할**: 루트(오케스트레이터) 세션에서 관리하는 **전 모듈 통합 실험 계획**.
> 각 모듈 세션의 상세 계획은 하위 EXPERIMENT_PLAN을 참조 — 이 문서는 **의존성, 우선순위, 통합 실험, 논문 매핑** 에만 집중.
>
> - [src/modules/builders/EXPERIMENT_PLAN_builders.md](src/modules/builders/EXPERIMENT_PLAN_builders.md)
> - [src/modules/selectors/EXPERIMENT_PLAN_selectors.md](src/modules/selectors/EXPERIMENT_PLAN_selectors.md)
> - [src/modules/extractors/EXPERIMENT_PLAN_extractors.md](src/modules/extractors/EXPERIMENT_PLAN_extractors.md)
> - [src/modules/filters/EXPERIMENT_PLAN_filters.md](src/modules/filters/EXPERIMENT_PLAN_filters.md)
> - a05 filter agentic 세부: [~/.claude/plans/vivid-sprouting-sunbeam.md](~/.claude/plans/vivid-sprouting-sunbeam.md)

---

## 0. 현재 베이스라인 (모든 실험의 출발점)

### GLM era (2026-04-24 ~) — XiYan backbone = `zai-org/glm-4.7` via Elice ML API (OpenAI-compatible)

| 범주 | Anchor | R | P | F1 |
|------|--------|---|---|-----|
| **🚀 전체 최고 (GLM era, 2026-04-24)** | `s04_stagewise_qcond_gat_basic_glm` (QCond + GAT α=0.85 + Basic PCST + XiYan GLM-4.7) | **0.8438** | **0.8329** | **0.8383** |
| diameter_layers sweep peak | `abl_sel_diameter_layers_nl6_glm` (nl=6=global D_max) | 0.5018 | 0.6939 | 0.5824 |
| GLM sanity (α=0 GAT-only) | `s04_04_qcond_a0_xiyan_glm` | 0.4922 | 0.6965 | 0.5768 |

### vLLM era (2026-02 ~ 04-24) — XiYan backbone = `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` [archive]

| 범주 | Anchor | R | P | F1 |
|------|--------|---|---|-----|
| **Wave 1.5 최고** | `s04_stagewise_qcond_gat_basic` (QCond encoder + GAT α=0.85 + Basic PCST + XiYan) | **0.8169** | 0.7605 | **0.7877** |
| Ensemble (2×2×2 anchor) | `abl_ens_basic_xiyan` (Cosine+GAT α=0.85 + Basic PCST + XiYan) | 0.8149 | 0.7597 | 0.7863 |
| Ensemble + Enriched Builder | `abl_ens_enriched_xiyan` (E1, Enriched Builder) | 0.6658 | 0.8147 | 0.7327 |
| Ensemble + Triplet | `abl_ens_triplet_xiyan` (E2) | 0.6823 | 0.8139 | 0.7424 |
| Direct variant 최고 | `abl_a03_17` (SuperNode-Direct + Fixed PCST + XiYan) | 0.6761 | 0.7128 | 0.6940 |
| Filter 최고 | `abl_a05_02` (Reflection, anchor=a03_17) | 0.7320 | 0.6833 | 0.7068 |
| α=0 GAT-only (GLM sanity baseline) | `s04_04_qcond_a0_xiyan` | 0.5015 | 0.7065 | 0.5866 |

**핵심 관찰**:
- **🚀 GLM era 새 전체 최고** (2026-04-24): `s04_stagewise_qcond_gat_basic_glm` F1=0.8383 이 Wave 1.5 vLLM best 대비 **ΔF1=+0.0506** (ΔR=+0.0269, ΔP=+0.0724). **Precision 주 개선축** — LLM backbone 단독 교체만으로 R/P/F1 전반 개선.
- **Diameter layers peak 검증 (H1)**: nl=D_max(6) 에서 sweep peak (F1=0.5824), nl=D_max+1(7) 에서 소폭 하락 (ΔF1=−0.0062 over-smoothing 재등장).
- **합격 기준 (2026-04-24 planner 확정)**: GLM era 실험은 vLLM era 동일 anchor 대비 **ΔF1 ≥ −0.02** (sanity 가 ΔF1=−0.0098 로 통과).
- **Precision 상한(≈0.81)은 Builder가 결정** (Enriched/Triplet) — Wave 1.5 는 Basic Heterograph 기반이라 precision 0.76 대에 멈춤. GLM era new anchor 에서 P=0.8329 로 벽 돌파 — **LLM backbone 이 Builder-driven precision ceiling 에도 영향**.
- **Recall 상한은 Filter의 restore path가 결정** (Reflection이 prune-only 한계 돌파) — QCond+GAT + GLM-4.7 은 restore 없이도 R=0.8438 돌파.
- 두 상한의 **교차 결합이 아직 미검증** (E1/E2 × QCond+GAT × Reflection × GLM-4.7).

---

## 1. 9 제안 전역 배분 (Cross-Module Matrix)

| # | 제안 | Builder | Selector | Extractor | Filter | Pipeline |
|---|------|---------|----------|-----------|--------|----------|
| 1 | RL Schema Linker (GRPO) | | ★ S-I | | | |
| 2 | Relational Foundation Model | B-I | ★ S-II | | | |
| 3 | EHGAT (Edge Hypergraph) | B-II | ★ S-III | | | |
| 4 | Louvain Community PCST | | | ★ E-I | | |
| 5 | Pathfinding + PCST Ensemble | | | ★ E-II | | |
| 6 | Autonomous Agent (AutoLink) | | | | ★ FL-I | |
| 7 | Extractive Decoder-only LLM | | S-IV | | FL-II | |
| 8 | Multi-agent RL Pipeline (MARS-SQL) | | | | | ★ PL-I |
| 9 | Neurosymbolic 3-layer Cooperation | B-III (L1) | S-V (L1) | E-III (L2) | FL-III (L3) | ★ PL-II |

★ = 해당 제안의 **주 구현 지점**.

---

## 2. Cross-Module Dependency Graph (Critical Path)

```
Builder B-III (FK reachability precompute)
    ├─→ Selector S-V (Neurosymbolic L1 score boost)
    ├─→ Extractor E-III (Neurosymbolic L2 cost adjustment)
    └─→ Filter FL-III (Neurosymbolic L3 connectivity verifier)

Builder B-II (Line graph)
    └─→ Selector S-III (EHGAT)

Builder B-I (RFM tokenization)
    └─→ Selector S-II (RFM encoder)

Selector S-IV / Filter FL-II (LLM logprobs)
    └─→ src/llm_client/api_handler.py 확장 (vLLM logprobs 지원)

Pipeline PL-II (Neurosymbolic 3-layer 루프)
    ← B-III + S-V + E-III + FL-III 모두 선결
```

**병목**: **Builder B-III (FK reachability)** — 3 모듈이 의존하는 infra. **가장 먼저 구현**.

---

## 3. 통합 실험 — 2×2×2 확장 (신규 상한 탐색)

기존 2×2×2는 (Seed × PCST × Filter). **신규 축 추가** 로 의미있는 상한을 탐색:

### 3.1 Cross-Module Synergy Grid (최고점 조합)
| 실험 ID | Builder | Selector | Extractor | Filter | 목적 |
|---------|---------|----------|-----------|--------|------|
| `int_01_e1_refl` | Enriched | Ensemble | AdaptivePCST | Reflection | E1 × a05_02 최초 결합 |
| `int_02_e2_refl` | Triplet | Ensemble | EdgePrize | Reflection | E2 × a05_02 |
| `int_03_e2_path_refl` | Triplet | Ensemble | **E-II Pathfinding** | Reflection | Extractor 신규 × Filter 신규 |
| `int_04_ns_full` | Enriched + **B-III** | Ensemble + **S-V** | **E-III** | **FL-III** + Reflection | **Neurosymbolic 3-layer 전체** |
| `int_05_direct_ns` | Enriched + B-III | **SuperNode v2** (directed SN→node + top-k Raw selective) + S-V | E-III | FL-III + XiYan | Direct variant에서 NS 효과. SuperNode v2 전제: directed edge + Raw Score 기준 top-k edge 선별 (2026-04-21 지도교수 의견 3·4 반영, Proposals D/E). |
| `int_06_best_x_autolink` | Triplet | Ensemble | EdgePrize | **FL-I AutoLink** | 최고 × autonomous agent |

**int_04가 논문의 주력 결과 후보** — 모든 기여가 수렴하는 설정.

### 3.2 Backbone 민감도 (논문 섹션 보강)
| 실험 ID | 변경축 | 비교 |
|---------|--------|------|
| `int_07_int04_gpt4o` | FL-III+Reflection backbone을 GPT-4o-mini로 | vs int_04 |
| `int_08_rfm_s_v` | Selector를 RFM (S-II) + S-V 보강 | vs int_04 |

---

## 4. 우선순위 & 타임라인 (작업 순서)

### Phase 0 — Active Waves (2026-04-28 지도교수 보고 전 단기 큐)

> 본 Wave 체계는 2026-04-21 advisor feedback → 제안서 Proposal A~F 를 15~20분 발표 스토리라인 (A > F > C > D > E > B) 에 맞춰 sequencing 한 실행 계층. Phase A/B/C/D/E 장기 로드맵과는 별도로 관리.

- **Wave 1 (closed, 2026-04-21~22)**: 인프라 선결.
  - ✅ Schema diameter cache (`data/processed/dev_diameter.pt`, 11 DB — symlinked NAS)
  - ✅ Selector QCond flags (`supernode_edge_direction`, `supernode_topk`, `num_layers_mode`, `directed_from_sn` in `src/models/gat_network{,_v2}.py`)
  - ✅ Base heterograph T2T toggle spec (B-II.b, `add_t2t_edges`) — 제안서 `abl_bld_t2t_edge.md` 에 반영
- **Wave 1.5 (closed, 2026-04-22 17:05 — no-filter backfill 포함)**: Proposal A stagewise 매트릭스 Extractor 통일 + +Extractor stage 보강.
  - ✅ `s04_stagewise_ensemble_raw_a0` — R=0.6676 / P=0.7236 / F1=0.6944
  - ✅ `s04_stagewise_qcond_raw_basic` — R=0.6622 / P=0.7539 / F1=0.7051
  - ✅ `s04_stagewise_qcond_gat_basic` ★ — R=0.8169 / P=0.7605 / **F1=0.7877** (새 전체 최고, `abl_ens_basic_xiyan` 대비 +0.0014)
  - ✅ no-filter 3 config (`*_no_filter.yaml`) — +Extractor cell 확정 (W1 F1=0.2272 / W2 F1=0.2862 / W3 F1=0.2271, Raw pair Δ F1=+0.0590 encoder 축 효과)
  - 세부: [EXPERIMENT_HISTORY.md §8](EXPERIMENT_HISTORY.md#L1229) (L1250 stagewise cumulative 표)
  - 후속: analyzer 2차 리포트 — `notebooks/analysis_results/stagewise_qcond_ablation.md` §1.1 / §5 슬라이드 3 에 +Extractor Δ 1 문단 보강 (Wave 2 실행과 병행 가능)
- **Wave 2 (closed, 2026-04-24 — GLM era kickoff 완료)**: Proposal C `abl_sel_diameter_layers` 5-cell sweep + GLM era new anchor 재실행. **LLM backbone** vLLM `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` → **GLM-4.7 (`zai-org/glm-4.7`) via Elice ML API (OpenAI-compatible)** 전환.
  - ✅ Phase 1 — 5× GAT 학습 (num_layers ∈ {1,2,3,6,7}) 완료 (2026-04-22 ~ 23, `best_gat_qcond_nl{1,2,3,6,7}.pt`)
  - ✅ Phase 2 — GLM era 7 cells 완료 (2026-04-24: sanity + 5 sweep + new anchor)
    - `s04_04_qcond_a0_xiyan_glm` (sanity): ΔF1=−0.0098 vs vLLM anchor — planner 재정의 기준 ΔF1≥−0.02 통과.
    - `abl_sel_diameter_layers_nl{1,2,3,6,7}_glm` — **peak at nl=6 (F1=0.5824, D_max)**, nl=7(D_max+1) ΔF1=−0.0062 over-smoothing 재등장. **H1 (global fixed peak at D_max) 검증 완료**.
    - `s04_stagewise_qcond_gat_basic_glm` — **F1=0.8383 (새 전체 최고)**, Wave 1.5 vLLM best (F1=0.7877) 대비 **ΔF1=+0.0506** (ΔR=+0.0269, ΔP=+0.0724).
  - 세부: [EXPERIMENT_HISTORY.md Wave 2 Proposal C GLM era kickoff (2026-04-24)](EXPERIMENT_HISTORY.md)
  - 후속: analyzer — `notebooks/analysis_results/diameter_layers_sweep.md` GLM era sweep 분석 + vLLM era 비교 부록
  - ⏸ **Proposal C H2 (per-DB dynamic num_layers)** — selector 세션 인프라 (`db_name` threading in `EnsembleSelector`) 미완료로 deferred. [`DECISIONS.md`](planning/DECISIONS.md) 2026-04-22 17:05 엔트리 §에스컬레이션 #1 유효.
  - 🔜 **Proposal D** `abl_sel_supernode_directed` / **Proposal E** `abl_sel_supernode_topk` — Wave 3 이후 재검토 (§8-1 SuperNode bug fix 후 재학습 필요).
- **Wave 3 (planned, 2026-04-26 ~ 28)**: 발표 패키징.
  - 🔜 **Proposal F** `abl_ext_steiner_backbone_report` — 기존 a03_15/18 데이터 재조직 리포트 (신규 실행 없음, analyzer 단독 큐).
  - 🔜 **Proposal A 확장 셀** (시간 여유 시) — Ensemble Raw 축의 stagewise cumulative 가 cosine 대비 reportable gap 을 보이면 추가 셀 확보.
  - ⛔ **Proposal B** `abl_bld_t2t_edge` — 2026-04-28 이후로 순연 (graph cache regen + GAT 재학습 비용 ≈ 11h, 스토리라인 우선순위 최하).
- **Wave 4 (post-2026-04-28)**: Filter agentic 트랙 재개.
  - 🔜 `a05_filter_agentic` 전체 12 실험 ([~/.claude/plans/vivid-sprouting-sunbeam.md](~/.claude/plans/vivid-sprouting-sunbeam.md)).
  - ⚠ 해당 plan anchor (`abl_ens_basic_xiyan`, F1=0.7863) **outdated** — Wave 1.5 new top (`qcond_gat_basic`, F1=0.7877) 로 anchor refresh 필요. Wave 4 kickoff 전 filter 세션 에스컬레이션으로 plan 갱신.
- **Wave 5 (CLOSED for R 갱신 + Phase 3+4 mechanism breakdown 활성, 2026-05-16)**: R 갱신 시도 final closure 단 학술 agent plan Phase 3 (Shapley) + Phase 4 (통합 점수 + 조건부 Filter) **활성** (paper main contribution Filter Dominance 의 mechanism contribution 정량 + production deployment narrative 다면적 강화). axis #11 builder-axis invariance candidate retain + strengthen.
  - 🗂 **Phase 2 Grid Sweep CLOSED (5/16, 25 cells 완료)** — anchor-band θ ∈ {0.1, 0.125, 0.15} × K (15 cells) F1 spread **0.0057 sub-noise**, P2_02 vs c01_01 ΔF1=**+0.0005** (deterministic 정합 PASS), Welch's t-test anchor-band vs outside-band t=6.14 / p<0.0001 (outside-band systematic decay). best lever p2_03 (θ=0.1, K=30) F1=0.8680 (anchor +0.0016, GLM noise floor 0.0005 의 3.20×) — single-cell statistically robust 아님 (spec ΔF1 ≥ 0.005 미달). **R 갱신 시도 final 중단** + axis #11 plateau evidence retain + strengthen. 상세: [notebooks/analysis_results/phase2_grid_heatmap_2026-05-16.md §3](notebooks/analysis_results/phase2_grid_heatmap_2026-05-16.md).
  - **★★ Phase 3.1 Module Contribution Shapley breakdown (2026-05-16 활성, ~2일)** — paper main contribution Filter Dominance 의 module-level 정량 evidence. Baseline c01_01 (θ=0.1, K=20, F1=0.8664) vs new config (anchor-band 안 조합, e.g., p2_03 θ=0.1 K=30 또는 closure 정합 위해 outside-band θ=0.3 K=40 비교): Selector 단독 변경 (F1_A) + Extractor 단독 변경 (F1_B) + 둘 다 변경 (F1_new) → Shapley contribution = ΔF1_sel / ΔF1_ext / ΔF1_fil. 추가 학습 불요 (Phase 1.1+1.2+2 데이터 만으로 reconstructable, analyzer 단독 위임). 학술 weight: Filter contribution > Selector + Extractor 의 정량 evidence → axis #5/#6/#7/#10/#11 plateau 의 mechanism contribution 정량 base. 학술 agent plan §Phase 3.1 정합. 상세: [planning/improving_exp_plan_by_scholar_agent_2026-05-15.md §Phase 3](planning/improving_exp_plan_by_scholar_agent_2026-05-15.md).
  - **★★ Phase 4.1 Selector + Extractor 통합 점수 α sweep (2026-05-16 활성, ~1.5일)** — `s_integrated(v) = α · 𝟙[v ∈ Top-K] + (1-α) · 𝟙[s_v ≥ θ]` 신규 mode 구현 + α ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} 6 cells × 최적 (θ, K) 결합. 측정: TCR_new + Filter Prune Ratio + 최종 F1. 학술 weight: Selector + Extractor co-design 통합 evidence — paper §3.5 axis #5/#6/#7 의 selector + extractor co-design narrative 보강 (R 갱신 lever 가 아닌 mechanism evidence 축). Code 변경: Extractor seed 선택 logic 의 OR 조건 모드 추가 (module:extractors 위임). 상세: [학술 agent plan §Phase 4.1](planning/improving_exp_plan_by_scholar_agent_2026-05-15.md).
  - **★★ Phase 4.2 조건부 Filter 호출 (2026-05-16 활성, ~1.5일)** — `Confidence(q) = TCR(q) < 0.5 → Filter 호출 skip` mode + Filter 생략 query 비율 + Filter skip 시 F1 손실 측정. 학술 weight: paper §V.5.x.M.3 production deployment narrative + §V.5.x.M.11 Filter Short-Circuit Artifact mechanism 의 cost-effective 정합 (단 5/14 anchor 의 6.32% involuntary skip 과 별개 — voluntary skip mechanism). Code 변경: Filter 호출 wrapper 에 conditional bypass logic (module:filters 위임). 상세: [학술 agent plan §Phase 4.2](planning/improving_exp_plan_by_scholar_agent_2026-05-15.md).
  - **Decision Gate (Phase 3 후)** — Filter Prune% > 60% → Phase 4 정식 진행 / < 50% → Phase 4 선택적. 현 anchor-band Prune% 92~94% 이므로 Phase 4 진행 정합 (학술 agent plan §Decision Gate for R 갱신 + Phase 3+4 mechanism breakdown 활성).
  - **★★ Phase 3.1 Module Contribution Shapley breakdown (2026-05-16 활성, ~2일)** — paper main contribution Filter Dominance 의 module-level 정량 evidence. Baseline c01_01 (θ=0.1, K=20, F1=0.8664) vs new config (anchor-band 안 조합, e.g., p2_03 θ=0.1 K=30 또는 closure 정합 위해 outside-band θ=0.3 K=40 비교): Selector 단독 변경 (F1_A) + Extractor 단독 변경 (F1_B) + 둘 다 변경 (F1_new) → Shapley contribution = ΔF1_sel / ΔF1_ext / ΔF1_fil. 추가 학습 불요 (Phase 1.1+1.2+2 데이터 만으로 reconstructable, analyzer 단독 위임). 학술 weight: Filter contribution > Selector + Extractor 의 정량 evidence → axis #5/#6/#7/#10/#11 plateau 의 mechanism contribution 정량 base. 학술 agent plan §Phase 3.1 정합. 상세: [planning/improving_exp_plan_by_scholar_agent_2026-05-15.md §Phase 3](planning/improving_exp_plan_by_scholar_agent_2026-05-15.md).
  - **★★ Phase 4.1 Selector + Extractor 통합 점수 α sweep (2026-05-16 활성, ~1.5일)** — `s_integrated(v) = α · 𝟙[v ∈ Top-K] + (1-α) · 𝟙[s_v ≥ θ]` 신규 mode 구현 + α ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} 6 cells × 최적 (θ, K) 결합. 측정: TCR_new + Filter Prune Ratio + 최종 F1. 학술 weight: Selector + Extractor co-design 통합 evidence — paper §3.5 axis #5/#6/#7 의 selector + extractor co-design narrative 보강 (R 갱신 lever 가 아닌 mechanism evidence 축). Code 변경: Extractor seed 선택 logic 의 OR 조건 모드 추가 (module:extractors 위임). 상세: [학술 agent plan §Phase 4.1](planning/improving_exp_plan_by_scholar_agent_2026-05-15.md).
  - **★★ Phase 4.2 조건부 Filter 호출 (2026-05-16 활성, ~1.5일)** — `Confidence(q) = TCR(q) < 0.5 → Filter 호출 skip` mode + Filter 생략 query 비율 + Filter skip 시 F1 손실 측정. 학술 weight: paper §V.5.x.M.3 production deployment narrative + §V.5.x.M.11 Filter Short-Circuit Artifact mechanism 의 cost-effective 정합 (단 5/14 anchor 의 6.32% involuntary skip 과 별개 — voluntary skip mechanism). Code 변경: Filter 호출 wrapper 에 conditional bypass logic (module:filters 위임). 상세: [학술 agent plan §Phase 4.2](planning/improving_exp_plan_by_scholar_agent_2026-05-15.md).
  - **Decision Gate (Phase 3 후)** — Filter Prune% > 60% → Phase 4 정식 진행 / < 50% → Phase 4 선택적. 현 anchor-band Prune% 92~94% 이므로 Phase 4 진행 정합 (학술 agent plan §Decision Gate).
  - 🗂 **post-paper backlog #14** (Phase 2 결과 후속) — seed sweep (5+ seeds × p2_03/p2_07) 으로 GLM stochastic noise floor 의 confidence interval 측정 + p2_03/p2_07 systematic lever 검증. 결과에 따라 axis #11 retract + R 갱신 lever 별도 narrative 또는 retain 결정.
  - 🗂 **post-paper backlog #15~#18** (직전 entry retain): Builder Step 1 ③-A / Step 2 ③-B V5-D-2 / Neurosymbolic 3-layer / Triplet × QCondGAT × GLM.
  - ✅ **paper closure narrative final retain**: §3.5 axis #5 topology-invariant + axis #6 architecture-invariant + axis #7 anchor-cluster 4-axis (filter + restore + selector encoder + selector top-K) + axis #10 (Extractor θ R-ceiling) + axis #11 (builder-axis invariance candidate, Phase 2 strengthen) — **R 상한 ≈ 0.870 의 system-level invariance 학술 declaration 정식 채택**.
  - ⏬ **★☆☆ Builder Step 1 ③-A (post-paper backlog #15)** — Builder col_text enrichment 가 selector identity fungibility 와 동일 homogenization 안 흡수 가능성 높음.
  - ⏬ **★☆☆ Builder Step 2 ③-B V5-D-2 (post-paper backlog #16)** — cost ~10~14d + plateau 흡수 가능성 → ROI 불확실.
  - ⏬ **★☆☆ V5 추가 학습 (post-paper backlog)** — 5/15 격하 유지.
  - ⏬ **★☆☆ B-III + S-V + E-III + FL-III 통합 chain (post-paper backlog #17)** — Neurosymbolic 3-layer (제안 #9).
  - ⏬ **★☆☆ Triplet × QCondGAT × GLM 격상 (post-paper backlog #18)** — ΔP ceiling 잠재력 부족.
  - ✅ **Anchor baseline 정정 (Option A)** — c01_01 F1=0.8664 채택, paper §V.5.x.M.5 narrative 일괄 갱신은 analyzer 위임 retain. Phase 2 P2_02 와의 deterministic 일치 검증으로 cross-validate.
  - **paper closure narrative 권고** (analyzer 갱신 예정, Phase 2 결과 도착 전 axis #9/#10 + anchor 정정 부터):
    - §V.5.x.M.X (신규, axis #9): **Extractor θ R-ceiling mechanism** (Phase 1.1 evidence, θ ≥ 0.3 monotonic decay)
    - §V.5.x.M.Y (신규, axis #10): **Selector top-K Filter-Invariant** (Phase 1.2 evidence, K=15~100 spread 0.0019 sub-noise band)
    - §V.5.x.M.5 (갱신): Anchor baseline 정정 + filter 6.32% skip artifact mechanism (post-paper backlog 의 query-level inspection 으로 보강)
    - §3.5 axis #11 (Phase 2 결과 도착 후 재작성): plateau 흡수 시 builder-axis invariance candidate strengthen, R 갱신 시 axis #11 retract + R 갱신 lever 별도 narrative.
  - 상세 정합: [planning/DECISIONS.md 2026-05-16 (Wave 5 Partial Reopen)](planning/DECISIONS.md)
- **Wave 6 (active, 2026-05-16 ~)**: Filter Recall Chain — 학술 agent filter improve plan ([planning/filter/0516_scholar_filter_improve_plan.md](planning/filter/0516_scholar_filter_improve_plan.md)) 의 5 methodology Phase-driven launch. Wave 5 closure axis (anchor stack hyperparameter θ/K/α/TCR) 와 별개 lever 축 — **Filter prompt 변형 axis**. 현재 anchor c01_01 F1=0.8664 / R_fil=0.8748 / P_fil=0.8582 의 FNR 11.76% (R_ext 0.9914 → R_fil 0.8748) 가 Filter prompt 의 "absolutely necessary" / "If irrelevant, exclude" Precision-bias 표현 hypothesis. 목표: R_fil ≥ 0.90 (stretch 0.95) + P_fil ≥ 0.75.
  - ✅ **Phase 1 — M1 Recall-Biased Prompt 3 variants 완료 (5/16)** — mild R=0.9259 (+0.0511 ★) / strong F1=0.8655 sub-noise (R=0.9022) / exclusion_rule R=0.8907. Inclusion bias strength → R-P trade-off monotonic confirm. anchor R 갱신 prompt-level lever first evidence. anchor F1 갱신 sub-noise 미달 (strong -0.0017 GLM noise floor 1.8×). 상세: [wave6_phase1_recall_biased_2026-05-16.md](notebooks/analysis_results/wave6_phase1_recall_biased_2026-05-16.md).
  - ✅ **Phase 2 (a+aggressive) — M2 + M3 + M4 + M5 4 cells 완료 (2026-05-17, commit 3be7b35)** — **Outcome (b) confirmed** (F1 갱신 미달, anchor F1=0.8664 plateau retain). 핵심 결과:
    - **M4 ⭐** w6_p2_m4_bidirectional (Forward M1-A + Backward SQL Schema Analyst union): R=0.9325 / P=0.7593 / **F1=0.8370 ★ F1-best** / **EX=0.5300 ★ EX-max** (ΔEX **+0.0124** ✅ 첫 EX gain evidence)
    - **M3** w6_p2_m3_voting (Multi-Prompt OR default): R=0.9408 / P=0.6859 (P fails 0.75) / F1=0.7934 / EX=0.5202 (ΔEX +0.0026)
    - **M2** w6_p2a_m2cot_strong (CoT + Confidence-Gated thr=0.5 on strong): R=**0.9745** ★ extreme / P=0.2286 collapse / F1=**0.3703** worst / EX=0.5169 sub-noise
    - **M5** w6_p2_m5_two_stage (Stage1 Coarse → Stage2 Fine): R=**0.7739** (-0.1009 ❌ Stage2 over-prune) / P=0.7964 / F1=0.7850 / EX=0.5222
    - 학술 agent §10 success criterion (F1 ≥ 0.8672): 4 cells 모두 미달
    - Outcome (b) 정합 정합 — axis #15 evidence retain + axis #11 Option A retain (prompt-axis + builder-axis 별도) + M4 EX gain 첫 evidence (paper §3 Inter-Module Co-Design 새 mechanism candidate)
  - ✅ **Phase 3 — Aggregate + Pareto + axis #15 정식 채택 결정 완료 (2026-05-17)** — analyzer 산출 [wave6_phase2_results_all_methods_2026-05-17.md](notebooks/analysis_results/wave6_phase2_results_all_methods_2026-05-17.md) §1~§9 + planner 5 사항 정식 채택 (DECISIONS 2026-05-17 Phase 3 통합 채택). **paper §V.5.x.M.15 정식 채택** (M1 R-lift + M4 EX gain dual evidence) + **axis #15 정식 row 격상** + **§3.1 Inter-Module Co-Design Filter ↔ Selector Backward Mechanism bullet 추가** + **post-paper backlog #20 + #21 등록** + **axis #11 Option A retain 확정**. Pareto frontier 4 cells (M1-A + M1-B + M4 + M3 MAJORITY post-hoc).
  - ✅ **Phase 4 — Top 2 조합 C1 (M4 + M1-B strong) 완료 (2026-05-17, commit 778ef06)** — **Partial Degrade 확정**: R=0.9177 / P=0.8109 / **F1=0.8610** sub-noise vs M1-B strong (-0.0045) + EX=0.5150 (M4 EX gain +0.0124 거의 완전 소멸, **ΔEX=-0.0150 vs M4** ❌). **🚨 Backward mechanism Forward-prompt-dependent 첫 정량 evidence**: forward_count 6.30→5.21 (-17%), backward_added 0.18→**0.48 (2.67×)**, overlap rate 96.43%→90.57%. Per-difficulty: challenging Backward gain +0.0207 retain (complex schema partial robust). **paper §3.1 orthogonality hypothesis partial 부정** (entanglement evidence). axis #15 dual → **triple evidence** 확보 (M1 R-lift + M4 EX gain + C1 Partial Degrade). Pareto frontier 5 cells (M1-A + M1-B + M4 + M3 MAJORITY + 🆕 C1).
  - ✅ **Phase 5 — C2 (M4 + M3 MAJORITY Forward) 완료 (2026-05-17, commit 00bbe97)** — **H3 Partial Entanglement 확정** ✅ (H1/H2 부정). C2 R=0.9273 / P=0.7745 / F1=0.8440 / EX=**0.5196 intermediate** ∈ [C1 0.5150, M4 0.5300]. **Voting mechanism 69.3% + Inclusiveness mechanism 30.7%** 정확 정량 (M4:C1 distance ratio 2.26:1). Forward Dominance 3-cell complete coverage (mild→voting→strong). Backward telemetry monotonic 정합 (forward_count 6.30→5.89→5.21, backward_added 0.18→0.30→0.48). Per-difficulty: moderate dominant entanglement region + challenging complex schema partial robust.
  - 🎯 **Wave 6 chain final closure 확정 (2026-05-17)** — **paper §V.5.x.M.15 Quadruple Evidence 격상** (M1 R-lift + M4 EX gain + C1 Partial Degrade + C2 H3 Partial Entanglement) + **Pareto frontier 6 cells** (M1-A + M1-B + M4 + M3 MAJORITY + C1 + C2) + paper §3.1 entanglement 정확 정량 (70/30 split). **Outcome (b) confirmed** (모든 cell F1 ≥ 0.8672 미달). **paper main contribution narrative 완성** — multi-axis dual evidence (Wave 5 axis #5~#14 + Wave 6 axis #15 Quadruple).
- **Wave 9 (active, 2026-05-18 ~)**: Baseline Relog Chain — Baseline 3 cells (G-Retriever / LinkAlign / XiYan-SQL) 의 2026-03-28 measurement 가 outdated SQL Gen prompt 정합. paper §V.5.x.M.2 5/15 갱신 narrative + anchor EX +18.06%p jump (0.3396 → 0.5176) 정합 위해 baseline 3 cells × SQL Gen 만 재실행 (final_nodes 보존 + 신규 prompt). 학술 가치: paper §10 의 6 baseline 비교 표 정합 정확화 + paper main contribution 의 baseline ΔEX 정합 정확성 회복.
  - **★★★ Baseline Relog 3 cells (2026-05-18 활성)**: G-Retriever / LinkAlign / XiYan-SQL × SQL Gen 만. 3 LLM call/q × 1534q = 4,602 calls (~$5~10 + ~1.5h parallel 3 streams). Wave 7 Option A pattern 정합 (anchor relog + SQL Gen 만).
  - Success criterion: 모든 baseline 의 신규 prompt EX 측정 + difficulty-별 (simple/moderate/challenging) 분해 + paper §10 표 정합 갱신.
- **Wave 8 (active, 2026-05-18 ~)**: M4 Bidirectional Filter 발전 실험 — 학술 agent improving plan ([planning/improving_m4_plan_scholar_agent_2026-05-18.md](planning/improving_m4_plan_scholar_agent_2026-05-18.md)) 의 4 direction + 조합 실험. **M4 anchor 고정** (Forward M1-A mild + Backward SQL Schema Analyst + Union + sanitize) 위에 4 독립 컴포넌트 추가. **EX-first framing** (목표: R ≥ 0.9325 retain + Prune ≥ 89% + **EX > 0.5300 M4 초과**).
  - **★★★ Direction 1 — Question Decomposition → Multi-Backward** (LLM N×, RoSL 2025 / Nahid 2025 reference): D1-A decomposer prompt + D1-B per-sub-question Backward + Union. D1-v1 (Backward sub-q union) + D1-v2 (Forward + Backward 모두 sub-q별). 학술 가치: §V.5.x.M.15 axis #15 evidence #5 candidate.
  - **★★★ Direction 2 — FK/PK Connectivity Steiner Closure** (LLM 0×): DB DDL → FK graph → M4 output 의 Steiner closure (직접 FK + 1-hop bridge variant). D2-v1 (direct FK) + D2-v2 (1-hop bridge). 학술 가치: §V.5.x.M.16 신규 (DB-aware Schema Connectivity).
  - **★★★ Direction 3 — Self-Verification Loop (SQL Probe)** (LLM 1~2×): Sketch SQL → DB 실행 → 오류 파싱 → 컬럼 복구 → retry loop. D3-v1 (1 round) + D3-v2 (2 round). 학술 가치: §V.5.x.M.17 신규 (Execution Feedback Loop, AutoLink-style).
  - **★★★ Direction 4 — Value Hint Forward 강화** (LLM 1~2×): Question → Value extraction → Extractor schema value matching → Forward hint 강화. D4-v1 (Value-Hint Forward) + D4-v3 (Forced-Include). 학술 가치: §V.5.x.M.18 신규 (Value Evidence Enhancement).
  - **★★★ Combination — Top 2 directions 결합** (단독 실험 후 결정): Comb-A (D2 + D4-v3) / Comb-B (D1 + D2) / Comb-C (D2 + D4 + D3) / Comb-D (D1 + D2 + D4 + D3). 학술 가치: §V.5.x.M.19 신규 (Top 2 Synergy Evidence).
  - **Success criterion** (학술 agent §8): R_fil ≥ 0.9325 (M4 retain) + Prune% ≥ 89% + EX > 0.5300 (M4 초과). Stretch: R ≥ 0.95 + EX > 0.5400.
  - **핵심 제약** (학술 agent §0.1): LLM 입력에 Full Schema 포함 금지 + Extractor 출력 후보만 사용 + DB metadata 직접 조회 허용 (LLM 입력 아님) + DB 실행 (SQL probe) 허용.
  - **🎯 Launch 결정 (사용자 5/18 옵션 ②)**: 4 directions **동시 launch** (D1 + D2 + D3 + D4) + Top 2 조합 후속. ~1.5~3h wall + ~$16~33 + ~$5~15 조합. Wave 6 chain 정합 (cost-aware parallel coverage).
  - 상세: [planning/improving_m4_plan_scholar_agent_2026-05-18.md](planning/improving_m4_plan_scholar_agent_2026-05-18.md), [planning/DECISIONS.md 2026-05-18 (Wave 8 M4 발전)](planning/DECISIONS.md).
- **Wave 7 (active, 2026-05-18 ~)**: Stage-wise EX Measurement Chain — M4 anchor framework 분석 보고서 ([planning/m4_anchor_framework_analysis_2026-05-17.md §5.5 + §5.6](planning/m4_anchor_framework_analysis_2026-05-17.md)) 의 n/a EX cell 4 종 launch. 학술 가치: Stage-wise Shapley breakdown EX dimension + Filter Dominance Necessity 직접 evidence (Extractor only EX vs Filter EX 대조) + M3 voting full coverage (OR/MAJORITY/AND).
  - **★★★ Wave 7 Stage-wise EX Chain (4 cells, 2026-05-18 활성)** — 사용자 옵션 ① 결정.
    - **(1) Selector only EX**: anchor Selector top-K=20 output → SQL Gen direct (Extractor + Filter bypass). ~20 nodes/q. ~$2~4 + ~1.5h.
    - **(2) Extractor only EX (no Filter)**: anchor Extractor output ~83 nodes/q → SQL Gen direct (Filter bypass). ~$3~6 (large schema token up).
    - **(3) M3 MAJORITY EX**: M3 voted_nodes (MAJORITY ≥2) → SQL Gen direct. ~6 nodes/q. ~$2~4.
    - **(4) M3 AND EX**: M3 voted_nodes (AND =3) → SQL Gen direct. ~3 nodes/q. ~$2~4.
    - Total: 4 cells × 1534q = 6,136 calls + ~$9~18 + ~1.5h parallel (4 streams).
  - Success criterion: 모든 빈 EX 정량 확보. paper §V.5.x.M.12 Shapley breakdown EX dimension 의 stage-wise contribution 정량 완성 + axis #15 Filter ablation table EX axis full coverage.
  - **🔜 다음 단계 (post-Wave-7)**: paper draft 본 작성 진입 또는 학술 agent cover note + post-paper backlog priority 재평가.
  - **Success criterion 2 분기 (paper narrative 정합)** — Phase 2 결과 후 정식 결정:
    - **(a) Phase 2 F1 > 0.8672 통계 robust**: axis #15 full evidence → paper main contribution 추가, axis #11 Option B reinterpret
    - **(b) Phase 2 F1 sub-noise plateau**: axis #15 = axis #5~#14 plateau prompt-level strengthening, axis #11 Option A retain
  - 상세: [planning/filter/0516_scholar_filter_improve_plan.md](planning/filter/0516_scholar_filter_improve_plan.md), [planning/DECISIONS.md 2026-05-16 (Wave 6 Phase 1 결과 + Phase 2 (a) 활성)](planning/DECISIONS.md).

### Phase A — Infrastructure (선결)
- [ ] **Builder B-III** — FK reachability metadata precompute (가장 중요)
- [x] **Schema Graph Diameter precompute** — 각 DB heterograph 의 schema 노드 간 최대 shortest-path `D_max` 계산, `data/processed/*_diameter.pt` 캐시. **2026-04-22 완료** — `scripts/build_diameter_cache.py` + `dev_diameter.pt` (NAS symlink). B-III FK reachability 루틴과 1 패스 공유 예정 (BFS/Dijkstra 경로에서 동시 집계 확장). 소비자: Proposal C (`abl_sel_diameter_layers`, num_layers ∈ {1,2,3,D_max,D_max+1} sweep). 근거: 2026-04-21 advisor Q1.
- [ ] Builder B-II — LineGraph 변환기
- [ ] Builder B-I — RFM serialize API
- [ ] `src/llm_client/api_handler.py` — vLLM logprobs 지원

### Phase B — Low-risk Quick Wins
- 🗂 **Phase 2 Grid Sweep CLOSED (5/16)** — 25 cells 완료, Success criterion (a) plateau 흡수 ✅ confirm + (b) R 갱신 lever ⚠ 잠정 sub-noise (p2_03 F1=0.8680, ΔF1=+0.0016 < spec 0.005). Wave 5 closure narrative final 채택, axis #11 plateau evidence retain + strengthen.
- [ ] **★★ Phase 3.1 Module Contribution Shapley breakdown (2026-05-16 활성, ~2일, analyzer 단독)** — Baseline c01_01 vs new config 의 Selector / Extractor / Filter 단계별 contribution 정량. 추가 학습 불요 (Phase 1+2 데이터 reconstructable). 상세: [improving_exp_plan_by_scholar_agent_2026-05-15.md §Phase 3](planning/improving_exp_plan_by_scholar_agent_2026-05-15.md).
- [ ] **★★ Phase 4.1 Selector + Extractor 통합 점수 α sweep (2026-05-16 활성, ~1.5일)** — `s_integrated = α·𝟙[Top-K] + (1-α)·𝟙[s_v ≥ θ]` 6 α cells (module:extractors 위임). 상세: [학술 agent plan §Phase 4.1](planning/improving_exp_plan_by_scholar_agent_2026-05-15.md).
- [ ] **★★ Phase 4.2 조건부 Filter 호출 (2026-05-16 활성, ~1.5일)** — TCR(q) < 0.5 시 Filter skip mode (module:filters 위임). 상세: [학술 agent plan §Phase 4.2](planning/improving_exp_plan_by_scholar_agent_2026-05-15.md).
- 🗂 **Wave 5 R 갱신 시도 closure final (2026-05-16)** — Phase 2 + Builder Step 1/2 + V5 추가 + B-III chain + Triplet 모두 post-paper backlog 격하. paper closure narrative 정식 채택 + Phase 3+4 mechanism breakdown 활성. 상세: [planning/DECISIONS.md 2026-05-16 (Phase 3+4 활성)](planning/DECISIONS.md).
- 🗂 post-paper backlog #15 — Builder 축 ablation Step 1 (③-A enriched_v3 col_text enrichment).
- 🗂 post-paper backlog #16 — Builder 축 ablation Step 2 (③-B V5-D-2 schema-aware contrastive PLM).
- 🗂 post-paper backlog #17 — B-III FK reachability + S-V + E-III + FL-III 통합 chain.
- 🗂 post-paper backlog #18 — Triplet × QCondGAT × GLM 격상.
- 🗂 post-paper backlog #20 — **M2 Confidence-Gated False Override Mechanism** (Wave 6 Phase 2 M2 F1=-0.4961 catastrophic failure root cause). 분석 spec: Confidence distribution histogram + thr=0.5 false override rate + design alternatives (thr=0.7 / hybrid score-based gating). 산출: `notebooks/analysis_results/m2_confidence_gated_false_override_2026-XX-XX.md`. (5/18 audit 정합: parse_errors 86.25% + default-retain mechanism evidence 확보 — 본 backlog 의 detail 분석 candidate retain)
- 🗂 post-paper backlog #21 — **M5 Stage2 Over-Prune Mechanism** (Wave 6 Phase 2 M5 R=-0.1009 sequential pipeline failure root cause, Stage2 51.12% over-prune). 분석 spec: Stage2 prompt `stage1_schema_str` format 의 LLM understanding 정합 + query-level R-loss + Stage2 prompt redesign candidate. 산출: `notebooks/analysis_results/m5_stage2_over_prune_2026-XX-XX.md`.
- 🗂 post-paper backlog #23 — **Wave 8 Comb-B/C/D Extension** (Wave 8 closure 5/19 후 deferred).
- 🗂 ~~post-paper backlog #24 — Wave 9 Baseline Relog R/P/F1 Post-hoc 측정~~ ✅ **완료 reclassify (5/20)** — Wave 13 Evaluator Alias Resolution Patch + Retrospective R 재측정 chain 의 Phase B 안에 통합 진행. 별도 backlog 등재 불요.
- 🗂 post-paper backlog #26 — **c_v3a Multi-Rerun LLM Stochastic Variance Validation** (사용자 5/20 42nd turn 정합 정정 위 신규 등재, priority 3). Wave 11 c_v3a 의 EX +0.0235 의 정합 정합 = LLM stochastic confound 의 정합 정합 가능성 (1st run → rerun ΔEX +0.0287 stochastic drift evidence). 본 정합 정합 정정 위 **동일 cell 의 multiple rerun** (~3~5 reruns) 위 stochastic variance 의 정확 정합 정합 정정 (c_v3a EX 의 stochastic variance range + c_v3b 의 stochastic variance range 의 정량 정합 정합 정정). paper main contribution Filter mechanism = M4 fixed 정합 정합 위 의 정합 정합 정정 — c_v3a 의 학술 novelty 정합 정합 정정 위 의 post-paper validation candidate. 0 LLM 추가 cost (rerun 만), ~3~5 reruns × 1534q × ~2 LLM/q = ~9000~15000 LLM calls (post-paper extension cost). 산출: `notebooks/analysis_results/c_v3a_stochastic_variance_2026-XX-XX.md` (stochastic variance 의 정합 정합 정정 + c_v3a vs M4 의 EX gap 의 stochastic confound 의 정확 정합 정합). paper drafting 직전 active trigger 안 함 — post-paper validation 의 정합 정합 정정.
- 🗂 post-paper backlog #27 — **Plain GAT 학습 — no_builder Training-Inference Mismatch 해소 검증** (사용자 5/21 trigger 위 신규 등재, priority 4 optional). Wave 15 m15_no_builder cell 이 Plain HeteroGraphBuilder feature 위 enriched-trained `best_gat_qcond_nl3.pt` 재사용 — 정확 Plain GAT model 부재 위 training-inference distribution mismatch 정합. 본 backlog 의 spec: Plain HeteroGraphBuilder feature 위 별도 QCond GAT 학습 (Enriched 와 동일 hyperparameter, ~2~3일 GPU 0,1) → m15_no_builder 의 정합 cell 재측정 위 quantitative finding 검증 (current ΔF1=-0.0030 / ΔEX=-0.0170 marginal 의 정합 정합 정정). **post-paper validation candidate** — Wave 15 의 핵심 finding 인 Filter >> Extractor > Builder ≈ Selector ranking 의 핵심 quantitative 결정에는 영향 없음 (Builder ranking 만 의 정합 정합 정정). 산출: `outputs/checkpoints/best_gat_qcond_plain_nl3.pt` (신규 Plain feature GAT) + `outputs/experiments/abl/wave15_module_ablation/m15_no_builder_plain_qcond_plain_gat_mst_pcst_m4/metrics.txt` (재측정) + `notebooks/analysis_results/wave15_plain_gat_validation_2026-XX-XX.md` (정합 정합 정정 분석). Cost: ~2~3일 학습 (GPU 0,1) + 1 cell × 1534q × ~3 LLM/q = ~5000 LLM calls. paper drafting 직전 active trigger 안 함 — post-paper validation 정합. 상세: [planning/DECISIONS.md 2026-05-21 (Wave 15 결과 채택) §7](planning/DECISIONS.md) + [EXPERIMENT_HISTORY.md Wave 15 entry §5853~5855](EXPERIMENT_HISTORY.md) + [notebooks/analysis_results/paper_draft_V5xM_dual_variant_three_caveat.md §V.5.x.M.21.7](notebooks/analysis_results/paper_draft_V5xM_dual_variant_three_caveat.md).
- 🗂 post-paper backlog #25 — **Wave 6/Wave 8 Cells Per-Difficulty R/P/F1 Retrospective** (Wave 14 closure 5/20 후 deferred, priority 3, paper drafting 직전 active trigger candidate). Wave 14 의 `src/analysis/wave9_per_difficulty_rpf1.py` pattern 위 확장 — 대상 18 cells (Wave 6 9 cells M1-A/B/C+M2~M5+C1+C2 + Wave 8 9 cells D1+D2+D3+D4×2+Comb-A) × 3 difficulty × R/P/F1 = 54 entries. 데이터 source: `outputs/experiments/abl/wave6_recall_biased/*/predictions.jsonl` + `outputs/experiments/abl/wave8_m4_extensions/*/predictions.jsonl` + dev.json (difficulty). 산출: `notebooks/analysis_results/wave6_wave8_per_difficulty_rpf1_2026-05-XX.md`. paper 영향: paper §10 의 본 framework cells per-difficulty 추가 정합 (Wave 9 baseline per-difficulty 와 직접 비교 base) + paper §V.5.x.M.5 의 본 framework difficulty-stratified evidence 보강. **paper main contribution narrative 본질 영향 없음** (sub-noise refinement, paper drafting 직전 priority 3 trigger). 0 LLM, ~30 분 analyzer. Wave 8 Comb-A (D4 v1 + D3 v2 직렬 stacking) closure 후 추가 조합 candidate: **Comb-B (D1 v1 + D3 v2)** — R-axis (sub-q multi-backward) + EX-axis (verify) 의 R-bias trade exploration (D1 v1 P −0.0679 cost retain 우려) / **Comb-C (D2 + D4 v1 + D3 v2)** — 3-axis (구조 + 값 + 검증, D2 mechanism 거의 무효 우려 — 권고 안 됨) / **Comb-D (D4 v1 + 추가 post-filter variant)** — D3 verify 대신 다른 post-filter mechanism candidate. Wave 8 closure narrative (paper §V.5.x.M.19 Stacking Synergy) 정량 base 위에 추가 stacking mechanism 의 정량 검증. paper main contribution 영향 작음 (sub-Pareto 추가 candidate).
- [ ] **★★★ Wave 10 Measurement Framework Audit + 통일 결정 (2026-05-18 활성, post-paper backlog #22 → active 격상, Phase A ✅ + Phase B ✅ Option A + Phase C 활성)** — 사용자 5/18 22nd turn 정합 지적 ("Wave 6 를 Wave 7 에 맞추든 반대든 하나로 통일 시켜야 비교 가 될 거 아니야 별 일 아니라고 넘어가면 안 되는 거잖아"). paper main contribution 의 거의 모든 정량 비교 표가 **mixed framework** (main.py col-only spec vs phase1 capacity index spec) — paper drafting 전 선결 필요. **3-phase 진행 현황**:
  - **[x] Phase A — Analyzer audit 완료 (5/18)**: 두 spec 의 정확 차이 검증 (c01_01 anchor 에서 ΔR=+0.0217 / ΔF1=+0.0165). 분해: (i) gold 에 table 노드 제거 +0.0115 (~53%) + (ii) MST bridge 노드 포함 +0.0102 (~47%). Phase 1 capacity index evaluator source = ad-hoc 미커밋 스크립트 확인. 학술 정합 권고 = Spec A (col-level R/P/F1 dominant convention, BIRD/RoSL/RethinkSL). 산출: [notebooks/analysis_results/measurement_framework_audit_2026-05-18.md](notebooks/analysis_results/measurement_framework_audit_2026-05-18.md).
  - **[x] Phase B — 사용자 결정 (5/18, 23rd turn) = Option A (Spec A 단일 채택, main.py col-only)**: paper main R/P/F1 = Spec A 통일. Spec B 의 5 capacity indices (TCR/TOR/BNR/AUC/Prune%) 는 paper §V.5.x.M.4 mechanism axis 보조 metric 으로 retain. 상세: [planning/DECISIONS.md 2026-05-18 (Wave 10 Phase B 사용자 결정)](planning/DECISIONS.md).
  - **[x] Phase C — Paper 표 갱신 본체 ✅ 완료 (planner 직접 갱신, 5/18 24th turn 사용자 정정 후)**: (1) paper §V.5.x.M.12 ΔR Filter cost row 통일 (**−0.1230 (Spec A, main.py col-only)** 단일) + (2) paper §V.5.x.M.12 R_ext footnote 정정 (Spec A 0.9927 base + Spec B 0.9710 보조 retain) + (3) Wave 7 §1.1 표 갱신 (R_ext=**0.9927 ◆** Spec A base) + footnote 정정 + (4) paper §V.5.x.M.4 capacity indices sub-section 의 Spec B mechanism axis footnote 보강. **paper main contribution narrative (M4 EX gain + F1-EX 분리 + axis #15 Quadruple Evidence) 영향 없음 retain** — sub-noise refinement only. paper §10 baseline 표 / Wave 6 9 cells R/P/F1 (main.py spec) **변경 없음 retain**.
  - **[ ] Phase C — 향후 root 위임 한정 (Wave 9 종료 후)**: Wave 9 baseline relog 3 cells (g_retriever_relog + linkalign_relog + xiyansql_relog) output_*.jsonl 생성 완료 시점, Spec A Filter R/P/F1 통합 + paper §10 6-baseline 비교 표 통합 갱신 + summary_all.csv 통합 → **Wave 9 종료 trigger 위임**.
- [x] **Wave 6 Phase 1 (M1 Recall-Biased Prompt 3 variants) — 완료 (5/16)** — mild R=0.9259 trigger 충족 + strong F1=0.8655 sub-noise sweet spot + exclusion_rule R=0.8907. 상세: [wave6_phase1_recall_biased_2026-05-16.md](notebooks/analysis_results/wave6_phase1_recall_biased_2026-05-16.md).
- [x] **Wave 6 Phase 2 (a+aggressive) — M2 + M3 + M4 + M5 4 cells 완료 (2026-05-17, commit 3be7b35)** — Outcome (b) confirmed (F1 미달 plateau), M4 ⭐ F1-best 0.8370 + EX-max 0.5300 (ΔEX +0.0124 첫 evidence), M2 R-extreme 0.9745 + F1 collapse. 상세: [EXPERIMENT_HISTORY.md Wave 6 Phase 2 entry](EXPERIMENT_HISTORY.md).
- [x] **Wave 6 Phase 3 (Aggregate + Pareto + paper narrative 정식 채택) 완료 (2026-05-17)** — analyzer 산출 + planner 5 사항 정식 채택 (paper §V.5.x.M.15 정식 + axis #15 정식 row + §3.1 Filter ↔ Selector Backward bullet + post-paper backlog #20+#21 + axis #11 Option A retain). 상세: [planning/DECISIONS.md 2026-05-17 (Wave 6 Phase 3 통합 채택)](planning/DECISIONS.md).
- [x] **Wave 6 Phase 4 (Top 2 C1, M4 + M1-B strong) 완료 (2026-05-17, commit 778ef06)** — Partial Degrade 확정 (F1=0.8610 sub-noise vs M1-B + EX -0.0150 vs M4 EX gain 소멸). Backward Forward-prompt-dependent 첫 정량 evidence (backward_added 2.67×, overlap 96.43%→90.57%). axis #15 triple evidence 확보. 상세: [planning/DECISIONS.md 2026-05-17 (Wave 6 Phase 4 C1 결과 + C2 launch)](planning/DECISIONS.md).
- [x] **Wave 6 Phase 5 (C2, M4 + M3 MAJORITY Forward) 완료 (2026-05-17, commit 00bbe97)** — H3 Partial Entanglement 확정 (Voting 69.3% + Inclusiveness 30.7% 정확 정량). axis #15 Triple → Quadruple Evidence 격상. Pareto frontier 6 cells 완성. 상세: [planning/DECISIONS.md 2026-05-17 (Wave 6 chain final closure)](planning/DECISIONS.md).
- 🎯 **Wave 6 chain final closure 확정 (2026-05-17)** — paper §V.5.x.M.15 Quadruple Evidence + Pareto 6 cells + entanglement 70/30 split. paper main contribution narrative 완성 (multi-axis dual evidence).
- [ ] **★★★ Wave 7 Stage-wise EX Chain (4 cells, 2026-05-18 활성)** — (1) Selector only EX + (2) Extractor only EX (no Filter) + (3) M3 MAJORITY EX + (4) M3 AND EX. m4_anchor_framework_analysis 의 n/a cell 완전 채움 + Stage-wise Shapley breakdown EX dimension + Filter Dominance Necessity 직접 evidence + M3 voting full coverage. Total 6,136 calls ~$9~18 + ~1.5h parallel. 상세: [planning/DECISIONS.md 2026-05-18 (Wave 7 Stage-wise EX Chain)](planning/DECISIONS.md).
- [x] **★★★ Wave 8 M4 Bidirectional 발전 4 Direction (8 cells, 2026-05-19 분석 완료) — paper §V.5.x.M.15~18 결정 ✅ (D1+D2 격하 / D3+D4 retain) + Comb-A 사용자 결정 pending** — D1 Multi-Backward + D2 FK Steiner Closure + D3 Self-Verification Loop + D4 Value Hint Forward 4 direction × 2 variants 완료. **5 Cases 판정**: Case 1 (EX > M4) **미달 ❌** (8 cells 모두 EX < M4 0.5300) / Case 2 D3 ✅ (D3 v2 EX=0.5215 sub-noise) / Case 3 D1 ✅ (R-bias trade) / Case 4 D4 v1 ✅ (F1 +0.0023 ⭐) / Case 5 D2+D4 v3 ✅ 격하. **Pareto Frontier 갱신**: R-best D1 v2 (0.9601, +0.0276) + F1-best Wave 8 D4 v1 (0.8393) + P-best Wave 8 D4 v1 (0.7623) + EX-2nd D3 v2 (0.5215). **4 결정** (3 planner 직접 갱신 ✅ + 1 사용자 trigger pending):
  - **[x] §V.5.x.M.15 axis #15 evidence #5 (D1) 격하** — R lift magnitude 약함 (+0.0276 vs RoSL +25.1%) + P-cost dominant (P −0.2093 collapse). axis #15 본문 Quadruple Evidence (M1+M4+C1+C2) retain.
  - **[x] §V.5.x.M.16 (D2 FK Steiner Closure) 격하** — added_count 0.2~0.3/q mechanism 거의 무효 + F1 sub-noise drift. sub-section 신설 안 함.
  - **[x] §V.5.x.M.17 (D3 Self-Verification Loop) retain** — D3 v2 EX retain mechanism (M4 sub-noise) + Diminishing returns. paper sub-section 신설 + narrative draft 채택.
  - **[x] §V.5.x.M.18 (D4 Value Hint Forward) retain** — D4 v1 F1 +0.0023 marginal positive + hint-only vs forced 의 P-cost trigger mechanism. paper sub-section 신설 + narrative draft 채택.
  - **[x] Top 2 조합 Comb-A (D4 v1 + D3 v2) launch ✅ 완료 (5/19, commit 96b7314)** — **Verdict: Fail (Case 2 미달, EX 0.5117 < D3 v2 alone 0.5215, ΔEX −0.0098)** 단 **F1 0.8684 = post-Wave 5 globally best** (anchor c01_01 Wave 5 의 +0.0020 marginal positive). **P-axis dual-lift mechanism**: D4 individual +0.0030 + Stacking synergy **+0.0624 ⭐** = ΔP +0.0654 vs M4 (individual 의 ~20× magnitude). **EX paradox root cause**: D3 v2 의 specific EX mechanism (recovered_count 1, +0.0046 lift) 이 Comb-A 에서 0 collapse (D4 clean schema 가 1-round verify success rate saturate) + EX-down 40 queries 의 schema sparsity penalty. **F1-EX Decoupling 강력 정량 evidence**: F1 +0.0314 + EX -0.0183 simultaneous decoupling (paper §V.5.x.M.12 의 single-cell strongest evidence). Analyzer 산출: [notebooks/analysis_results/wave8_comb_a_2026-05-19.md](notebooks/analysis_results/wave8_comb_a_2026-05-19.md).
  - **[x] Wave 8 closure ✅ (paper main contribution evidence 충분 정합, 5/19)** — paper drafting trigger 가능 base. **Pareto Frontier 완성** (4 axis multi-coverage): R-best D1 v2 (0.9601, Wave 8 신규) + **F1-best Comb-A (0.8684, post-Wave 5 globally best)** ⭐⭐ + **P-best Comb-A (0.8247, post-Wave 5 P-best dual-lift)** ⭐ + EX-best M4 (0.5300, Wave 6 retain). **Paper §V.5.x.M sub-section 완성**: §V.5.x.M.12 (F1-EX Decoupling 강력 정량 evidence 보강) + §V.5.x.M.15 (axis #15 evidence #5 D1 격하) + §V.5.x.M.16 (D2 격하) + §V.5.x.M.17 (D3 context-aware dual mechanism 보강) + §V.5.x.M.18 (D4 stacking platform mechanism 보강) + **§V.5.x.M.19 신규 sub-section** (Pre-Filter + Post-Filter Stacking Synergy F1-best Mechanism). 상세: [planning/DECISIONS.md 2026-05-19 (Wave 8 Comb-A 분석 결과 채택 + Wave 8 closure)](planning/DECISIONS.md). **Comb-B/C/D launch 는 post-paper extension 으로 위임** (post-paper backlog #23 candidate). 후속: HISTORY/CATALOG/ID_MIGRATION 갱신 (root) + paper drafting trigger.
  Analyzer 산출: [notebooks/analysis_results/wave8_m4_extensions_2026-05-19.md](notebooks/analysis_results/wave8_m4_extensions_2026-05-19.md). Planner 결정 + 갱신: [planning/DECISIONS.md 2026-05-19 (Wave 8 M4 발전 4 Direction 분석 결과 채택)](planning/DECISIONS.md). 후속: HISTORY/CATALOG/ID_MIGRATION 갱신 (root). 상세 계획: [planning/improving_m4_plan_scholar_agent_2026-05-18.md](planning/improving_m4_plan_scholar_agent_2026-05-18.md).
- [ ] **★★ MA (Selector Monitor ↔ Inference 정합) 신규 방향 (2026-06-07 활성)** — R@15(training proxy) ↔ inference recall **disconnect 의 constructive 출구**. V6 over-smoothing closure 와 **별개 축** (training-signal ↔ deployment 운영점 정합). **3단계**: **MA-0** 정량 분석 ✅ 완료 (2026-06-07) — **rank ⊥ calibration 확정**: Spearman(proxy, inference recall) = **gold p50 +0.9545 / gold recall@θ=0.1 +0.9273** (★ calibration 압도) vs ROC-AUC +0.59 / top20-recall +0.50 (rank 중) vs **Val R@15 −0.1909 (✗ 무용)**. extractor 가 score≥θ 기반이라 절대 calibration 이 inference recall 결정. → **MA-1 ✅ 확정** 모니터 교체 — trainer best-epoch/early-stop 을 **Val R@15 → gold recall@θ=0.1 (또는 gold p50)**, Val R@15 는 보조로만. → **MA-2 ✅ 게이트 정당화** calibration loss — phase1 collapse 가 calibration 문제 확정 (gold p50=0.0000 절대압축, ROC 0.6381 rank 부분보존, gap +0.2529) → gold score margin/BCE 강화 또는 per-table norm 으로 gold 를 θ 위로. **게이트 = inference recall (EX 아님)**. **★ raise-θ feasibility (2026-06-07, `ma_raise_theta_feasibility`)**: 현 체크포인트 raise-θ **단독은 제한적** — clean 운영점(input≥20%↓∧recall≥95%) 부재, 최선 w3_c θ=0.3 (8.8%↓). 근본원인: input 축소 ∝ [θ,gold] nongold 양 — V6 cells nongold≈0(여지 없음)/M4만 elevated(gold 동반 절단). ⇒ **MA-2 가 raise-θ 의 선결 enabler** (conditional→prerequisite): gold recall@high-θ≈1.0 + gold↑/nongold↓ margin 동시 → clean gap. **실험 순서**: MA-2 calibrated 학습(module:selectors) → θ sweep 실측(root, V7-W5 인프라) → input 축소@iso-recall. 현 cells raise-θ 단독 우선순위↓. 게이트 = inference recall + **Filter input 노드수(효율)**, EX gain 금지. 사용자 목표 = filter LLM input 축소 @ iso-perf. **★ in-flight (2026-06-07)**: v6w6_a(V6-W6 cell4 MA-1 restart) + ma2_a(gold margin) + ma2_b(per-table norm) 학습 중 (MA-1 patch 적용, gold_recall@θ+gold_p50+R@15 epoch별 로깅 ✅). **★ Monitor saturation issue**: gold_recall@θ=0.1 이 일찍 ceiling 1.0 → best-epoch undertrained lock (ma2_a Epoch 3 R@15=0.4324). ⇒ **future default monitor = gold_p50** (ρ=+0.95, ceiling 없음), gold_recall@θ 보조 강등. 현 cells = option D (학습 끝까지 + post-hoc saturation 정량). ⚠ θ sweep 은 saturation-lock best ckpt 아닌 **well-trained ckpt(last-epoch/calibration-sweep re-run)** 사용. **calibration_margin_weight sweep wave** = candidate (gold_p50 monitor re-run → clean ckpt, analyzer 결과 후). log-based post-hoc MA-1 트라젝토리 분석 가능 (ckpt 부재로 inference 불가). 상세: [DECISIONS 2026-06-07 #4 §MA-1 logging+saturation](planning/DECISIONS.md). **★ 정직 caveat**: inference recall 은 이미 extractor over-extract 로 ~0.99 천장, e2e 병목은 precision/Filter (Filter Dominance) → MA 의 확실한 가치 = **방법론 honesty (올바른 체크포인트·보고 지표) + over-extract 의존 감소 (효율, V7-W5 cost 연결)**, **e2e EX gain 보장 안 됨** (MA-2 를 'EX 향상' 으로 framing 금지). disconnect 원인: top-K 랭킹 vs 절대 θ cutoff 운영점 불일치 + 랭킹≠calibration (phase1 best R@15 but gold p50=0.0002) + over-extract washout + gold-recall 출처=cosine. 상세: [DECISIONS 2026-06-07 #4](planning/DECISIONS.md).
- [ ] **★★★ V7 Extractor 개편 Chain (2026-06-04 활성 — 학술 Agent RFP 3종 정합 STE/FKP/FKH)** — 사용자 의문 (Wave 15 no_filter cell P=0.1268, ~95% Full Schema 복구 = Extractor over-extract) 의 직접 대응 chain. 학술 Agent 산출 [`planning/extractor/scholar_agent_extractor_rfp_2026-06-04.md`](planning/extractor/scholar_agent_extractor_rfp_2026-06-04.md) 위 Phase A 후보 선별 3종 + Phase B RFP 정합. **Wave 시퀀스**: **V7-W0** 사전 + 베이스라인 측정 (~1일, root + analyzer, STE-00/FKP-00/FKH-00 공통, M4 anchor `best_gat_qcond_nl3.pt` × seeds {42,123,7}) → **V7-W1** FKH 직렬화 (RFP #3, ~2~3일, root + module:utils, CPU only 병행 가능, FKH-00~04 × 5 seeds {42,123,7,456,789}, EX 게이트 +0.0030 + R/P/F1 변화 없음) → **V7-W2** FKP Pathfinding (RFP #2, ~3~4일, module:extractors + root, FKP-00~06 × 3 seeds, R≥0.9000 + P≥0.2500 + EX≥baseline-0.0050, SchemaGraphSQL Safdarian 2025 정합) → **V7-W3** STE Steiner Tree (RFP #1, ~6~7일, module:extractors + root, STE-00~08 × 3 seeds, R≥0.9000 + P≥0.3000 + EX≥baseline-0.0050, GRAST-SQL Hoang 2025 정합) → (선택) **V7-W4** STE+FKP 조합 (~3일, V7-W3 게이트 통과 후). **planner 정정 ★**: 학술 Agent RFP 위 (i) GPU 영역 `CUDA_VISIBLE_DEVICES=2,3` → **`0,1`** 정정 ([[feedback_gpu_allocation]] 정합, GPU 2,3 다른 연구자 reserve), (ii) Import path `from src.modules.base import BaseExtractor` → **`from modules.base import BaseExtractor`** 정정 (src prefix 안 함, register decorator 추가). **paper 영향**: paper main contribution 무영향 + paper §V.5.x.M 위 신규 sub-section candidates (M.22 STE / M.23 FKP / M.24 FKH). **학위 본 심사 영향**: §III.7 (Filter Dominance) 보강 + §III.10 (신규) "Connectivity-Preserving Extractor Family" candidate. **V6 chain 과 분리 retain** (V6 = selector over-smoothing, V7 = extractor over-extract). 상세: [planning/extractor/extractor_redesign_v7_plan_2026-06-04.md](planning/extractor/extractor_redesign_v7_plan_2026-06-04.md) + [planning/DECISIONS.md 2026-06-04 (V7 Extractor 개편 Chain launch)](planning/DECISIONS.md).
- [~] **★★ V6 Over-Smoothing Chain (Phase 0~4, 2026-06-01 활성 → V6-W0 ✅ + V6-W1 ✅ (2026-06-04 negative result) + V6-W2 ✅ CLOSED (2026-06-05 Multi-level Disconnect) + V6-W3 ✅ CLOSED (2026-06-06 mixed result) + V6-W5 ✅ CLOSED (2026-06-07 capstone: mechanism fixable mad 20~25× 회복 + e2e disconnect ironclad) + **V6 chain closure (V6-W6 지도교수 trigger 1 wave 일시 reopening)** — paper main contribution 무영향)** — **V6-W2 ✅ CLOSED (2026-06-05)**: edge_type split 4 cells × 300 epoch × s11 → selector-only (DirectGATv2Selector 신규) → e2e (M4 anchor swap) → mechanism 검증 wave (extractor thr sweep 12 cells, 진행 중). 결론 = **Multi-level Disconnect** (L1 training-proxy phase1 0.5697 / L2 selector sum F1 0.3027 / L3 e2e no_selfloop EX 0.3331 — stage 마다 best cell 다름) + architecture-agnostic (V6-W0 Projector 도 동일) + **constructive threshold-pass mechanism** (Spearman(EX,pass@0.1)=+1.0, top-20 F1 반전 −0.2). **§III.7.4 신규 sub-section spec** (기존 §III.7 capstone, threshold-pass lead + single-seed/n=4 caveat + 검증 wave pending provisional). **남은 wave 재정의 (2026-06-05, V6 chain 종료 단계)**: P0 검증 wave(진행) → P1 V6-W4′ score calibration (analysis-only, phase1 PairNorm gold-score collapse p50=0.0002 원인) → **V6-W3 hub 차수 축소 ✅ CLOSED (2026-06-06, mixed result)** — 3 cells(VirtualSummary/ColumnPooling/HubLocalVN)×s11 측정 (L1+L2(a)+L2(c) e2e). **mixed**: (i) mechanism informative negative — **L1 FAIL** (hi-deg intra-MAD L3 0.0024~0.0042 무회복, EF2 0.0000 유지) + **L2(a) GAT-only 미상승** (best C F1=0.2646<baseline); (ii) e2e cell C uplift — EX=0.3879 (V6-W2 best +0.0548, **단 threshold-pass 매개=GAT rescue 아님, M4 0.5300 −0.1421 미달**). EF2 paradox (intra-MAD=0.0000 인데 e2e EX=0.4651)=mechanism ⊥ e2e 재확인. 결론 한정: GAT fundamental 한계 증명 아님 (rescuability 미검증), hub reduction 이 GAT 구출 경로는 아님 확인. over-smoothing=GAT message-passing layer 자체(builder 무관, mech(ii-b) DOMINANT). **★ V6 chain architectural rescue 종료 (2026-06-06)**: V1~V5 null + V6-W1 drop-in null + V6-W2 disconnect + V6-W3 builder null = mech(ii-b) DOMINANT 수렴, 추가 rescue wave 정지. 원 V6-W4 domain backlog 유지(재개 안 함). 남은 활성=rescue 아닌 분석(V6-W4′ score calibration analysis-only + V6-W3′ multi-seed user 결정). 다음 개입=GAT layer 자체. **★ collapse origin 진단 (2026-06-07, 가설 B 확정)**: L0 PLM hi-deg intra-MAD=0.4201 정상 → L1 첫 conv 0.0136 (−96.8%), collapse=first-conv single-shared-source aggregation (hub 컬럼 99.4% 단일 table 소스 + self-loop/residual 부재), input/PLM 한계 아님. ⇒ **V6-W5 🟢 활성 (Phase 5, mechanism-targeted final wave)**: column self-loop(W5-a)/per-layer residual(W5-b) — single-shared-source 직접 완화, V6-W5 완료 후 V6 chain 영구 closure. caveat: mad 회복 거의 확실 단 e2e 별개(disconnect), 두 결과 모두 paper-valuable (e2e flat 시 Filter Dominance ironclad). **★ (C) GAT 기여 분해 (2026-06-07): 역할 분리 확정** — GAT 순기여=table-level (gold TABLE net +182, rescue:hurt 3.28), GAT 격리 column net −1181 (multi-gold −1188, collapse functional), **column 식별=cosine 담당**. ⇒ W5 의 GAT column 복원은 cosine redundant+off-role → e2e flat 강하게 예상 (단순 disconnect 아닌 upstream 역할 분리). chain deepest=2중 robustness (downstream Filter Dominance + upstream 역할 분리). paper §V.5.x.M.24.9 + §III.7.4.6 (collapse origin + 역할 분리) + §III.7.4 L4 axis + mech(ii-b) 정밀화(single-shared-source 가 column granularity 파괴, table granularity 만). **★★ V6-W5 ✅ CLOSED (2026-06-07 capstone)**: Primary ✅ PASS — L1 hi-deg intra-MAD **20.7~25.5× 회복** (a 0.2813/b 0.3416/c 0.3463, L0 0.4201 의 67~82%, EF2 0.0000→0.30+) ⇒ 가설 B 인과 확정 (mechanism fixable). Secondary ❌ — **mad 회복이 selector 과제 분별력으로 전환 안 됨**: gold p50 0.017~0.126 (≪ V6-W2 sum 0.8195), top-20 F1 b 0.2977 ≈ baseline, **column net 음수 유지 −1359~−1805 (M4 −1181 보다 더 음수)**, table net 증가 +277~+617 (task-irrelevant table-level differentiation). e2e EX a 0.3168/b 0.3201/c 0.2934 (M4 −0.21), **Spearman(mad,EX)=−0.5000**. **★ 2단계 인과 분리** (mad↑ → gold-분별 ✗ task-irrelevant → e2e ✗ threshold-pass) = **disconnect mechanistic 완결** (selector 표현 품질 ⊥ pipeline 성능, score 분해 수준 확증). 사용자 직관 2건 (extractor θ 재튜닝 / ensemble 제거 GAT-only) 둘 다 negative 확정 (GAT column net 더 음수 → 자립 실패, cosine column 우위 retain). **V6 chain 영구 closure 확정** — 추가 V6 architectural rescue wave 절대 금지. mech(ii-b) DOMINANT 최종: first-conv single-shared-source collapse 는 fixable 하나 회복 differentiation 이 gold-aligned 아닌 task-irrelevant. **★ V6-W6 🟢 활성 (2026-06-07, 지도교수 trigger — closure 일시 reopening)**: directed query SuperNode (`directed_from_sn`) + V6-W5 self-loop 조합. planner 비권장(§closed-candidate)을 advisor 명시 trigger override. prior 재평가: 이전 SuperNode 실패는 collapse 아키텍처(self-loop 없음) 위 → self-loop + directed query-SN 미측정 조합 (gold-alignment 새 경로 가능성 낮으나 non-trivial) + diligence value. 게이트 = V6-W5 동일 2단계 (핵심 column net −1359~−1805 → 양수 전환 + e2e vs V6-W5 marginal), s11. **W6 완료 후 V6 chain 재-closure**. **★ in-flight (2026-06-07)**: cell4 (pre-MA1, killed @e130 R0.5358, archived) → **MA-1 monitor 적용 restart 진행 중** (Epoch ~196, ETA~03:00 KST). MA-2 ma2_b 는 DSN+SL+W6-a 위 per-table norm 결합. 상세: [DECISIONS 2026-06-07 #3](planning/DECISIONS.md) + [V6 plan §1 Phase 6](planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md). **post-V6 우선순위**: (a) GAT/selector 영역 종료 (fixable 단 e2e 무관 결정적 입증), (b) **V7 chain (Extractor compact) + M4 threshold sweep cross-evidence 우선**, (c) filter stage (BiFilter 변형) post-V6 분기 가능, (d) V6-W4′ score calibration / V6-W3′ multi-seed 는 paper main (Filter Dominance ironclad) 정합 위 재평가 (V6-W3′ = single-seed standing 충돌 retain). 상세: [DECISIONS 2026-06-07 §V6-W5 결과](planning/DECISIONS.md) + [v6_w5_l1_mad_disconnect_2026-06-07.md](notebooks/analysis_results/v6_w5_l1_mad_disconnect_2026-06-07.md). 상세: [DECISIONS 2026-06-06 §측정 결과 분기](planning/DECISIONS.md) + [v6_phase3_hub_reduction_2026-06-06.md](notebooks/analysis_results/v6_phase3_hub_reduction_2026-06-06.md). 상세: [DECISIONS 2026-06-05 §V6-W2 closure](planning/DECISIONS.md) + [V6 plan §7.2 §III.7.4 + §7.3](planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md). 후속: root §III.7.4 paper edit + HISTORY 등재. — **(이전 계획) V6-W2~W4 활성 (2026-06-04)**: single seed (s11) retain. **V6-W2** edge-type 분리 (HeteroConv/RGATConv) → **V6-W3** hub 차수 축소 (RFP H1) → **V6-W4** 도메인 아키텍처. — **V6-W1 closure (2026-06-04)**: drop-in 3종 (PairNorm + GCNII IR + JK) 단독 + 조합 + loss sweep 17 cells 모두 R@15 0.5631~0.5736 (spread 0.0105, V1~V5 spread 0.0526 의 1/5 이내) — pseudo-B0 (loss-sweep 0.5667) ±0.007 이내 = negative result 확정. **PairNorm column 보존 demonstrable (P1a/P1d L3 MAD intra 0.083~0.095, B5 family ~30×) 그럼에도 R@15 최저권** = intervention 이 over-smoothing metric 움직였으나 task benefit ZERO. **V6 metric ↔ R@15 causal disconnect 격상** (combined 32 cells ρ=-0.4038, p=0.022, V6-W0 correlational ρ≈0.19 위 음의 유의 상관 격상). **single-seed (s11) caveat**: B0 부재 위 pseudo-B0 (병행 loss-sweep) 활용, vs M4 anchor 0.6097 Δ -0.036 은 seed/trainer confound. **V6-W2 진입 보류** (본 wave 외 별도 trigger 영역 retain — drop-in negative + causal disconnect 위 architectural intervention 추가도 동일 disconnect 예상). analyzer 산출 [`v6_phase1_dropin_ablation_2026-06-04.md`](notebooks/analysis_results/v6_phase1_dropin_ablation_2026-06-04.md) + [DECISIONS 2026-06-01 §V6-W1 closure](planning/DECISIONS.md). 후속: root EXPERIMENT_HISTORY/CATALOG/ID_MIGRATION 3종 V6-W1 17 cells 등재. — RFP `planning/oversmoothing/oversmoothing_rfp_2026-06-01.md` (사용자 입력 작업 지시서) 의 Phase 0~4 정합 5-wave 시퀀스. **선행 V1~V5 chain 의 14-trial mitigation null + mech(ii-b) DOMINANT 5/5 absolute confirm retain** — paper §V.5.4 narrative 무영향. **V6 chain 추가 axis**: (i) 신규 가설 H1 (hub-accelerated over-smoothing + 짧은 평균 경로 — V1~V5 의 mech(ii-b) 외 topology 측 원인 분리 검증), (ii) 신규 진단 protocol (Dirichlet energy + MAD + attention entropy — Oono & Suzuki 2020 + Rusch et al. 2023 survey 정합 정형화, V1~V5 의 L_GAT cos sim 위 정형화), (iii) drop-in 3종 단독+조합 ablation (PairNorm + GCNII IR + JK — V5-B `GCNIIGATv2Conv` reuse 가능 단 격리 단독 측정 신규), (iv) edge-type 분리 + hub 차수 축소 + 도메인 아키텍처 (V1~V5 chain 미탐색). **Wave 시퀀스**: **V6-W0** 사전 준비 + Phase 0 진단 (RFP §3 + Phase 0, ~0.5~1일, root 계측 훅 + analyzer 진단 리포트, 산출: `src/analysis/v6_oversmoothing_diagnostics.py` + 베이스라인 시드 3 + 데이터 차수 통계 + `notebooks/analysis_results/v6_phase0_diagnostics_2026-06-XX.md`) → 게이트 H1/H0/over-squashing 분기 → **V6-W1** Phase 1 drop-in 3종 (PairNorm + GCNII IR + JK, P1a/P1b/P1c/P1d × 3 seeds = 15 runs, ~1~2일, module:selectors + root) → 병행 sweep (InfoNCE temperature + hard negative + BCE:InfoNCE 비율) → 게이트 Dirichlet energy 평탄화 + 성능 향상 → **V6-W2** Phase 2 edge-type 분리 (HeteroConv 또는 RGATConv, Phase 1 조합 retain, ~2~4일, Builder edge type metadata 의존) → **V6-W3** Phase 3 hub 차수 축소 (테이블 virtual node / column→table pooling / Local VN_G, ~2~3일, Builder virtual node 지원 의존) → **V6-W4** Phase 4 도메인 아키텍처 차용 (LGESQL line-graph + RAT-SQL relation-aware bias + Graphix-T5 장기, 별도 연구 단위). **Scope 제외**: 곡률 rewiring (SDRF/FoSR/BORF) — star 구조 commute time 낮음, Phase 0 over-squashing 별도 진단 시만 검토. **Baseline GAT checkpoint**: M4 anchor `best_gat_qcond_nl3.pt` (QCond GAT NL3, 학습 config `configs/training/diameter_layers/train_qcond_nl3.yaml`, DECISIONS 2026-05-26 §후속 정정 #1 정합). **상세**: [planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md](planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md) (V6 plan §0~§9, 2026-06-01 갱신: §0.4 + §1 Phase 0 ✅ + §1 Phase 1 ⏬ 격하 + §2 + §6 + §7.1~§7.4) + [planning/DECISIONS.md 2026-06-01 (V6 chain launch + §V6-W1 격하 분기)](planning/DECISIONS.md). **paper 영향**: paper main contribution 무영향 (본 wave 외 별도 trigger 영역) + supplementary artifact candidate ([V6 plan §7.1](planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md#71-supplementary-artifact-candidate-v1v5-retrospective-v6-metric-retain-evidence-위-강화) 의 §A.1~§A.6 spec). **학위 본 심사 영향**: §III chapter 후속 보강 base evidence ✅ 완성 — V6-W0 retrospective 만으로 §III.3 (Mitigation null) + §III.4 (mech(ii-b) DOMINANT deep dive) + §III.6 (attention paradox) + §III.7 (Filter Dominance 6번째 축) + §III.9 (V5 deep extension) 5 sub-section spec 완성 ([V6 plan §7.2](planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md#72-학위-본-심사-iii-chapter-후속-보강-analyzer-63-정합)). **V6-W0 ✅ 완료 (2026-06-01 analyzer retrospective)**: [`notebooks/analysis_results/v1_v5_retrospective_v6_metrics_2026-06-01.md`](notebooks/analysis_results/v1_v5_retrospective_v6_metrics_2026-06-01.md) (§1~§9 + 산출물: 632-line script + CSV 60 rows + JSON full + JSONL 825 records, 15 cells × 4-7 layers × 55 q stratified subsample). **핵심 finding**: (i) paper §V.5.4 main claim 5개 모두 V6 metric 위 retain (14-trial null + mech(ii-b) DOMINANT 5/5 → 9/10 absolute upgrade + paradox + Skip 부정 + V5C exception), (ii) **V6 metric ↔ R@15 disconnect** Spearman ρ_R@15-intra_sim=-0.16, ρ_R@15-MAD intra=+0.19, ρ_R@15-Dirichlet=-0.27 모두 p > 0.30 → Filter Dominance 6번째 축 (Training-Pathology-Invariant) 정량 강화, (iii) V5-C exception status retain (V5C_hop_only intra_sim 0.6989 best column differentiation + R@15=0.6076 V5 best). **V6-W1 🟢 활성 launch (2026-06-01 사용자 trigger 위 격하 retract)**: motivation = "단독으로도 별로 의미가 없다는 근거" 위 direct measurement 위 negative result 확정 evidence base 강화. Ablation Matrix retain: B0 + P1a PairNorm (scale {0.5,1.0,2.0}) + P1b GCNII IR (α {0.05,0.1,0.2}) + P1c JK (concat/max) + P1d 조합 × 3 seeds + 병행 sweep (InfoNCE temperature / hard negative / BCE:InfoNCE). 학술 위치: paper §V.5.4 narrative 무영향 retain + 학위 본 심사 §III chapter §III.3/§III.4/§III.9 single-cell evidence 강화 + supplementary artifact §A.x 신규 raw data table. ~1~2일 wall + 0 LLM + GPU 0,1. 핸드오프: module:selectors (PairNorm/GCNIIIR/JK/Combo classes) → root (config + `scripts/run_v6_phase1.sh` launch) → analyzer (V6 metric 측정 + 27+ cells matrix 확장 + Spearman correlation 갱신). **V6-W2~W4 본 wave 외 별도 trigger 영역 retain**: V6-W2 edge-type 분리 (HeteroConv/RGATConv) + V6-W3 hub 차수 축소 (테이블 virtual node/column→table pooling/Local VN_G) + V6-W4 도메인 아키텍처 (LGESQL/RAT-SQL/Graphix-T5) — V1~V5 미탐색 영역, 사용자 명시 trigger.
- [x] **★★★ Wave 15 Module Ablation Study (4 cells, 2026-05-20 활성, 2026-05-21 closed — Wave 15 결과 채택 완료, paper §V.5.x.M.21 + paper §10/§2.4/§3.6 narrative 신설 ✅)** — 4-module pipeline 확정 (Enriched Builder + QCond Selector + MST+PCST Extractor + M4 Filter) 후 모듈별 제거 ablation. 사용자 5/20 43rd turn trigger + 5/21 결과 채택. **결과 정량 (post-Wave 13 patch f67fa65, M4 anchor R=0.9357/P=0.7593/F1_harm=0.8383/EX=0.5300)**: (1) **no_builder** Plain+QCond+MST+PCST+M4 → R=0.9373/P=0.7533/F1=0.8353/EX=0.5130 (ΔF1=-0.0030/ΔEX=-0.0170 marginal), (2) **no_selector** Enriched+Cosine+MST+PCST+M4 → R=0.9279/P=0.7581/F1=0.8344/EX=0.4987 (ΔF1=-0.0039/ΔEX=-0.0313 marginal), (3) **no_extractor** Enriched+QCond+TopK20+M4 → R=0.7583/P=0.7447/F1=0.7514/**EX=0.3814** (ΔF1=-0.0869/**ΔEX=-0.1486 ★ dramatic**), (4) **no_filter** Enriched+QCond+MST+PCST+(no filter) → R=**0.9959**/P=**0.1268**/F1=**0.2250**/EX=0.5137 (**ΔF1=-0.6133 ★ dominant** + R upper bound 도달). **Module Importance Ranking: Filter >> Extractor > Builder ≈ Selector** ⭐ — paper main contribution 의 4번째 axis 정합 정량 evidence. **paper 갱신 ✅**: paper_draft_V5xM_dual_variant_three_caveat.md §V.5.x.M.21 신설 (Wave 15 4 Cells Matrix + Module Importance Ranking + §III.B.4 Filter Contribution cross-reference + §V Conclusion 직접 인용 narrative) + paper_research_direction.md §10 Wave 15 Module Ablation Matrix sub-section + §2.4 Wave 15 evidence sub-section + §3.6 Module Importance Ranking sub-section 신설 (planner). **HISTORY/CATALOG/ID_MIGRATION 갱신 ✅** (root). **No Filter cell R-ceiling 도달 (0.9959 ≈ B1/B2 oracle R=1.0000)**: Wave 12 Oracle direct evidence + Filter design intent (R 보존 + P 정제) 의 module-level 결정적 evidence. **Wall ~5h 27m + Cost ~27612 LLM calls ≈ ~$18~25** (GPU 0 only parallel 4 streams, GLM 4.7 via Elice ML). 산출: `configs/experiments/abl/wave15_module_ablation/` + `outputs/experiments/abl/wave15_module_ablation/m15_*/` + `scripts/run_wave15_module_ablation.sh` + `EXPERIMENT_HISTORY.md Wave 15 entry (line 5773~)` + `notebooks/analysis_results/paper_draft_V5xM_dual_variant_three_caveat.md §V.5.x.M.21`. 상세: [planning/DECISIONS.md 2026-05-21 (Wave 15 결과 채택) §1~§7](planning/DECISIONS.md) + [planning/DECISIONS.md 2026-05-20 (Wave 15 신규 활성)](planning/DECISIONS.md). **Plain GAT training-inference mismatch**: m15_no_builder 가 Plain HeteroGraphBuilder feature 위 enriched-trained `best_gat_qcond_nl3.pt` 재사용 — post-paper backlog #27 (Plain GAT 학습 검증) 신규 등재.
- [x] **★ Wave 14 Wave 9 Baseline Relog Per-Difficulty R/P/F1 Post-hoc 측정 (3 cells × 3 difficulty, 2026-05-20 ✅ 완료 + planner 갱신 완료)** — analyzer 직접 (0 LLM, ~10 분, `src/analysis/wave9_per_difficulty_rpf1.py`). **결과 정량 (post-Wave 13 patch, Sanity Check 4종 PASS ✅)**: G-Retriever (simple R=0.9207 EX=0.5114 / moderate 0.9114 0.3125 / challenging 0.9173 0.2690) + LinkAlign (simple 0.8163 0.4314 / moderate 0.6998 0.2112 / challenging 0.6873 0.1586) + XiYan-SQL (simple 0.6267 0.3092 / moderate 0.5561 0.1358 / challenging 0.5567 0.1379). **핵심 finding**: LinkAlign R dramatic drop simple→challenging −0.1290 ★ + Challenging EX Convergence (LinkAlign 0.16 ≈ XiYan-SQL 0.14, R gap +0.13 단 EX gap +0.02) + 3 baseline universal challenging EX decay pattern (mechanism universality). **paper 갱신 ✅ (planner 직접)**: §10 의 🆕 Wave 14 Per-Difficulty Matrix sub-section 신설 + §V.5.x.M.5 의 🆕 Difficulty-Stratified Mechanism Boundary sub-section 신설 (thrombosis_prediction outlier + Wave 14 의 dual dimension). Analyzer 산출: [notebooks/analysis_results/wave9_per_difficulty_rpf1_2026-05-20.md](notebooks/analysis_results/wave9_per_difficulty_rpf1_2026-05-20.md). 상세: [planning/DECISIONS.md 2026-05-20 (Wave 14 결과 채택 + paper §10/§V.5.x.M.5 갱신)](planning/DECISIONS.md). 후속: HISTORY Wave 9 entry per-difficulty 컬럼 추가 (root, paper drafting 직전 timing). — 사용자 trigger (5/20 38th turn): "Wave 9 Baseline relog 의 per-difficulty R/P/F 를 다시 계산하자 어차피 jsonl 파일만 있으면 할 수 있으니까". 직전 turn 의 baseline per-difficulty 정합 답변 의 ⚠️ Wave 9 미측정 영역 정합 정정. **사전 검증 ✅**: `outputs/baselines/wave9_relog/*/predictions.jsonl` 의 `final_nodes` + `difficulty` 가용 (gold_sql 만 dev.json 의 `SQL` field 의 question_id join 필요). **Spec**: post-Wave 13 evaluator (alias-aware patch f67fa65) + per-difficulty 분해 (simple n=925 / moderate n=464 / challenging n=145). **Sanity check**: overall R 정합 retain (G-Retriever 0.9176 / LinkAlign 0.7689 / XiYan-SQL 0.5987 exact match) + per-difficulty EX 정합 retain (Wave 9 §1.1 base) + mass conservation (weighted-mean = overall). **0 LLM, 0 launch, ~10 분 analyzer 직접**. 산출: `notebooks/analysis_results/wave9_per_difficulty_rpf1_2026-05-20.md` + `outputs/analysis/wave9_per_difficulty_rpf1_2026-05-20.csv` + 가능 시 per-query jsonl. **Phase B 후속**: planner 의 paper §10 per-difficulty 추가 갱신 + root 의 EXPERIMENT_HISTORY Wave 9 entry per-difficulty 추가 (paper drafting 직전). 상세: [planning/DECISIONS.md 2026-05-20 (Wave 14 Wave 9 Baseline Relog Per-Difficulty R/P/F1 Post-hoc 측정 신규 활성)](planning/DECISIONS.md).
- [x] **★★★ Wave 13 Evaluator Alias Resolution Patch + Retrospective R 재측정 (2026-05-20 ✅ Phase A+B 완료 + Phase C planner 갱신 ✅, root HISTORY 위임 진행 중)** — Option B 사용자 채택 (5/20 34th turn). **Phase A ✅**: Patch f67fa65 (`src/utils/evaluator.py:9-30` alias-aware `parse_sql_elements`) + smoke test (B3 invariant + B1/B2 R 0.9968 → 1.0000 + false positive 검증). **Phase B ✅**: 64 cells × Sanity Check 4종 PASS (ΔR ≥ 0 all + ΔP = 0 exact all + B3 invariant + B1/B2 exact 1.0). **Aggregate Δ**: ΔR mean +0.00286 + ΔP exact +0.0000 + ΔF1 mean +0.00138 (sub-noise refinement only). **Pareto Frontier post-Wave 13**: R-best D1 v2 0.9601 → **0.9633** retain + F1-best Comb-A (harmonic) 0.8684 → **0.8697** retain + P-best Comb-A 0.8247 invariant + EX-best M4 0.5300 invariant + R upper bound oracle B1/B2 0.9968 → **1.0000** ⭐ exact. **Phase C planner 갱신 ✅** (5/20): paper §V.5.x.M.4 narrative 정합 정정 (R 상한 1.0 + Wave 12 oracle R 정정 + Wave 9 baseline R 정정) + paper §V.5.x.M.4 Caveat 1 M2 R 정정 (0.9745 → 0.9778, Confidence-Gated default-retain mechanism 정량 강화) + paper §V.5.x.M.15 axis #15 cross-reference M2 R 정정 + paper §10 6-baseline 표 R 컬럼 정정 + planning/metric_spec_2026-05-20.md §1.1/§1.3 spec 정합 정정. **paper main contribution narrative 본질 retain** + Pareto position 모두 retain. **Phase C root 위임 진행 중**: EXPERIMENT_HISTORY.md 의 64 cells × R/F1_harmonic 값 정정 + HISTORY Wave 13 closure marker 등재. Analyzer 산출: [notebooks/analysis_results/evaluator_alias_fix_retrospective_2026-05-20.md](notebooks/analysis_results/evaluator_alias_fix_retrospective_2026-05-20.md). 상세: [planning/DECISIONS.md 2026-05-20 (Wave 13 Phase B 결과 채택 + Phase C 갱신 ✅)](planning/DECISIONS.md). — Wave 12 의 R=0.9968 root cause (SQL alias artifact, 18 unique alias names) 의 정합 정정 trigger (5/20 34th turn): "alias 여도 원래 컬럼 이름으로 다 맞춰서 Recall 을 재야할 것 같은데". **Patch spec**: `src/utils/evaluator.py:9-23` 의 `parse_sql_elements` 의 sqlglot `Alias` AST 식별 + alias name 제외 (alias 의 inner expression 의 real columns 는 outer query reference 정합 위 retain). **3-phase chain**: Phase A (code patch + smoke test — root + module:utils 위임, ~5 분 patch + B3 R=P=F1=1.0 retain + B1/B2 R 변화 + 18 alias names 의 DB col false positive 검증) → Phase B (retrospective analyzer chain — Wave 5~12 의 ~50~70 cells × R 재측정 + Wave 9 3 baseline relog 통합 + `notebooks/analysis_results/evaluator_alias_fix_retrospective_2026-05-XX.md` 산출) → Phase C (planner 의 paper §V.5.x.M.4 narrative R-ceiling 정합 정정 0.9968 → ~1.0 + §10 6-baseline 표 R 값 갱신 + §V.5.x.M.12 Triple Evidence retain + Pareto frontier 갱신 + `planning/metric_spec_2026-05-20.md` 정합 정정 + root 의 EXPERIMENT_HISTORY R 값 정정 + Wave 13 신규 entry 등재). **post-paper backlog #24 (Wave 9 R/P/F1 후속) ✅ 완료 reclassify** — 본 chain Phase B 통합 진행. **paper drafting timing 정합 정정**: Wave 13 결과 도착 후 paper §V.5.x.M.4 + §10 정합 정정 후 paper drafting trigger (Wave 8 + 9 + 10 + 11 + 12 + 13 의 6 chain 통합 정량 base). 0 LLM cost, ~1~2h wall (Phase A ~30 분 + Phase B ~30 분 + Phase C ~10 분 + 검증). 상세: [planning/DECISIONS.md 2026-05-20 (Wave 13 Evaluator Alias Resolution Patch + Retrospective R 재측정 신규 활성)](planning/DECISIONS.md).
- [x] **★★ Wave 12 Oracle Baseline R/P/F1 Post-hoc 측정 (3 cells, 2026-05-20 ✅ 완료, analyzer + planner 직접 갱신 완료)** — 사용자 정정 trigger (5/20 31st turn) → Option B 채택 (post-hoc analyzer, 0 LLM, 0 launch). **결과 정량**: B1 Full Schema R=**0.9968** / P=0.1173 / F1=0.1927 / EX=0.5587 + B2 Gold Table R=**0.9968** / P=0.2729 / F1=0.3839 / EX=0.5932 + **B3 Gold Column R=P=F1=1.0000 ⭐ + EX=0.6239** (perfect SL upper bound). **B3 Sanity Check PASS ✅** (Spec A `calculate_schema_metrics` implementation 정확성 confirmed). **B1/B2 R=0.9968 (NOT 1.0) root cause** = sqlglot 의 SQL alias 추출 artifact (19/1534 query, 18 unique alias names: `rnk`, `oxygen_count`, `total_amount`, ...). **본 framework 의 모든 cell 의 R 상한 = 0.9968** (M4 / Comb-A / anchor 모두 동일 artifact, schema linking 능력 의 한계 아님). **per-DB P × schema size negative correlation**: toxicology (11 cols) 0.4217 ↔ european_football_2 (199 cols) 0.0198 의 ~21× spread. **Pareto Frontier 4 axis 의 Oracle Reference 정합**: F1 Comb-A 0.8684 → B3 1.0 (+0.1316 gap) / R D1 v2 0.9601 → B1/B2 0.9968 (+0.0367) / P Comb-A 0.8247 → B3 1.0 (+0.1753) / EX M4 0.5300 → B3 0.6239 (+0.0939 = perfect SL 위 GLM 4.7 의 SQL gen ceiling). **paper 갱신 ✅ 완료** (planner 직접): (1) paper §10 의 🆕 Wave 12 Oracle Baseline R/P/F1 sub-section 신설 + (2) paper §V.5.x.M.4 의 🆕 절대 upper bound reference + R-ceiling 정확화 sub-section 신설 + (3) paper §V.5.x.M.12 의 🆕 F1=1.0 vs EX=0.6239 의 fundamental decoupling sub-section 신설 (Triple Evidence: Wave 7 + Wave 8 Comb-A + Wave 12). Analyzer 산출: [notebooks/analysis_results/oracle_baseline_rpf1_2026-05-20.md](notebooks/analysis_results/oracle_baseline_rpf1_2026-05-20.md). 상세: [planning/DECISIONS.md 2026-05-20 (Wave 12 Oracle Baseline R/P/F1 분석 결과 채택)](planning/DECISIONS.md). 후속: HISTORY/CATALOG 갱신 (root).
- [x] **★ Wave 11 Schema Serialization Direction C (5 cells + 1 Combined, 2026-05-20 ✅ rerun 완료, 사용자 5/20 42nd turn 정합 정정 — c_v3a 의 LLM stochastic confound 정합 정합 위 paper main contribution 학술 weight 약화 + Filter mechanism = M4 fixed 정식 채택 + Pareto EX-best M4 retain + post-paper extension candidate)** — 직전 진행 중 chain 의 closure (5/20). **결과 정량 (post-Wave 13 patch, rerun commit 21ee9ad)**: c_v3a_flat_merged_fk (Flat Merged + FK hint) R=0.9332 / P=0.7579 / F1=0.8365 / **EX=0.5535** ⭐⭐ (M4 +0.0235 dramatic, **post-Wave 5 globally best EX** ★★) + c_v3b_flat_merged_no_fk R=0.9327 / P=0.7583 / F1=0.8365 / EX=0.5013 (c_v3a − c_v3b ΔEX **+0.0522** FK hint EX 필수 mechanism evidence) + c_v0 (M4 base, rerun) EX=0.5209 + 1st run vs rerun LLM stochastic drift +0.0287. **시나리오 1 (긍정적) confirmed** ⭐ — EX ceiling 의 원인 = Filter-Generator Interface (직렬화 방식) 의 인과 evidence. **Schema Content Invariance retain**: c_v3a R/P/F1 모두 M4 의 sub-noise (±0.001 inner). **paper main contribution 3-axis 확장 mechanism evidence**: M4 (Filter-side Bidirectional) + Comb-A (Stacking Synergy) + **c_v3a (Generator-side Serialization)** ⭐⭐ 직교 axes. **F1-EX Decoupling Quadruple Evidence**: Wave 7 stage-wise + Wave 8 Comb-A + Wave 12 Oracle + **Wave 11 c_v3a**. **paper 갱신 ✅ (planner 직접)**: §V.5.x.M.20 신규 sub-section 신설 + §10 Pareto frontier EX-best 갱신 (M4 → c_v3a). 상세: [planning/DECISIONS.md 2026-05-20 (Wave 11 c_v3a Schema Serialization EX-Best 결과 채택)](planning/DECISIONS.md). 후속: HISTORY 갱신 (root) + c_v3a per-difficulty EX 분포 분석 (analyzer post-hoc candidate) + Wave 8 cover note 의 Wave 11 c_v3a 추가 통합 (사용자 → 학술 agent).
- [~] ~~Wave 11 Schema Serialization Direction C (5 cells + 1 Combined, 2026-05-19 active)~~ ✅ closure (위 entry retain) — 사용자 직접 작성 작업 지시서 [`planning/filter_improvement_wave10_2026-05-19.md`](planning/filter_improvement_wave10_2026-05-19.md) (naming "Wave 10" 단 ID = Wave 11, Wave 10 Measurement Framework Audit 와 충돌 회피). **Motivation**: Wave 8 Comb-A 의 F1 +0.0314 + EX -0.0183 simultaneous decoupling paradox 의 인과 검증 — EX ceiling 0.5300 의 원인이 (i) Filter-Generator Interface 인지 (ii) Generator 자체 (LLM 역량 한계) 인지. **핵심 제약**: Filter 가 선택하는 컬럼 집합 = **M4 와 완전히 동일** retain (R/P/F1 invariance, Schema Content Invariance). **5-Cell + 1 Combined**:
  - **C-v0**: Baseline (M4, 현재 직렬화) — base
  - **C-v1 Source-Tagged**: [F]/[B]/[F+B] 태그 추가 (M4 Forward/Backward 출처 신호) — 0 LLM, H1 검증
  - **C-v2 Question Enrichment**: Enriched question 대체 (E-SQL 정합, M4-Constrained No Full Schema) — +1 LLM, H2 검증
  - **C-v3a Flat Merged (FK포함)** / **C-v3b Flat Merged (FK없음)**: table.col flat + FK hint optional (UNJOIN-style) — 0 LLM, H3 검증
  - **Comb-C (C-v2 + C-v1)**: Tagged + Enriched — +1 LLM, H4 검증
  Total LLM cost: ~3000 calls + ~2~3h parallel. **3-phase chain**: Phase A (Filter 모듈 구현 — serializer 3 modules 신규 + filter_pipeline + sql_generator 수정 + M4 F/B 별도 저장 정합 + 사용자 few-shot 12개 수동 작성) → Phase B (root 위임 — config + script + launch + HISTORY 갱신) → Phase C (analyzer 분석 + planner 결정 + paper §V.5.x.M.20 신규 candidate). **시나리오 별 후속**: 긍정적 (C-v2/Comb-C ΔEX > 0) → paper §V.5.x.M.20 신규 sub-section ("Filter 출력 표현 최적화") + Comb-A × C-v2 조합 실험 / 부분적 (C-v1/C-v3 만 소폭) → C-v2 강화 post-paper / 귀무가설 (모든 variant ΔEX ≈ 0) → paper narrative reframe (EX → Prune Rate + F1 final 기여). **Wave 8 closure 와의 정합 변경**: paper drafting timing 을 Wave 11 결과 도착 후 paper §V.5.x.M.20 검토 후의 final integration 으로 연기. 상세: [planning/DECISIONS.md 2026-05-19 (Wave 11 Schema Serialization Direction C 신규 활성)](planning/DECISIONS.md) + [planning/filter_improvement_wave10_2026-05-19.md](planning/filter_improvement_wave10_2026-05-19.md).
- [x] **★★★ Wave 9 Baseline Relog Chain (3 cells, 2026-05-18 ✅ 완료, EX 갱신 ✅, paper §10 + §V.5.x.M.2 갱신 ✅)** — Baseline 3 cells (G-Retriever / LinkAlign / XiYan-SQL) 의 2026-03-28 measurement 가 outdated SQL Gen prompt 정합. **Wave 9 정합 결과**: G-Retriever EX=**0.4283** (+0.1793) / LinkAlign EX=**0.3390** (+0.1389) / XiYan-SQL EX=**0.2405** (+0.0436, schema-sparse penalty). 3 baseline 모두 prompt-axis +Jump 정합 확인. **Prompt-axis ΔΔ 분리 = +0.0515~+0.0574** (anchor 의 prompt-axis 효과 +0.1721~+0.1780 vs baseline 평균 +0.1206) — **본 framework 의 schema linking effect 의 정량 evidence**. **paper main contribution baseline 우위 ΔEX squeeze: +0.2627~+0.3148 → +0.0834~+0.2712** (range 정확화). paper §V.5.x.M.2 EX-Friendly Property narrative 본질 retain (baseline 도 prompt-axis +Jump 정합 evidence). Analyzer 산출: [notebooks/analysis_results/wave9_baseline_relog_2026-05-18.md](notebooks/analysis_results/wave9_baseline_relog_2026-05-18.md). Planner 갱신: paper §10 6 baseline 표 + Prompt-Axis Confounder ΔΔ 분리 sub-section + §V.5.x.M.2 sub-section 신설. 상세: [planning/DECISIONS.md 2026-05-18 (Wave 9 Baseline Relog 분석 결과 채택)](planning/DECISIONS.md). 후속: Wave 9 cells 의 Spec A col-only R/P 측정 (analyzer post-hoc candidate, priority 낮음) + EXPERIMENT_HISTORY 갱신 (root).
- [ ] **Selector S-V** (Neurosymbolic L1, λ 튜닝)
- [ ] **Extractor E-III** (FK prior cost adjustment)
- [ ] **Filter FL-III** (Symbolic Verifier, detect-only 먼저)
- [ ] **Extractor E-II** (Pathfinding ensemble, MSTExtractor 재활용)
- [ ] **Base heterograph T2T edge toggle** — Builder B-II 스펙 확장, table↔table 직접 edge on/off 플래그. s04 재학습 시 결합. 소비자: Proposal B (`abl_bld_t2t_edge`). Wave 3/4 로 순연. 근거: 2026-04-21 advisor 의견 2.
- ⏬ **V5 mitigation 추가 학습** (★☆☆ post-paper backlog, 2026-05-15 격하) — V5 7 cells 의 final F1 ROI 가 sweep 노이즈 (0.0030) 수준. PCST closure invariant 로 selector 축 추가 개선 효과 미달. Filter Dominance / Three-Caveat narrative 확정 후 재검토.

### Phase C — Medium Effort
- [ ] **int_01~03** (Cross-module synergy — 기존 모듈 재조합)
- [ ] **Extractor E-I** (Louvain community)
- [ ] **Filter FL-I** (AutoLink full-exploration, a05_13~15)
- [ ] **Filter FL-II** (Extractive LLM, a05_16~18)
- [ ] **Selector S-III** (EHGAT)

### Phase D — High Effort (후순위)
- [ ] **Selector S-II** (RFM encoder, GPU/latency 높음)
- [ ] **Selector S-I** (RL/GRPO, 학습 불안정성)
- [ ] **Pipeline PL-I** (MARS-SQL — 제안 #8, multi-agent RL)

### Phase E — Integration
- [ ] **int_04** (Neurosymbolic 3-layer 전체) — 논문 주력
- [ ] **int_05~08** (Direct variant, backbone 민감도)

---

## 5. 논문 매핑 (한국지능정보시스템학회 2026 춘계)

**현재 초안**: [notebooks/analysis_results/paper_draft_abstract_intro.md](notebooks/analysis_results/paper_draft_abstract_intro.md)

| 논문 섹션 | 주력 기여 | 근거 실험 |
|-----------|----------|-----------|
| II. Related Works | 9 제안의 선행 연구 조망 | 각 서브세션 plan의 "학술 기여" |
| III. Methodology | Neurosymbolic 3-layer framework 중심 | B-III + S-V + E-III + FL-III |
| IV. Experiments | int_04 주력 + ablation 단계별 | int_01~04, phase별 축별 ablation |
| V. Conclusion | 그래프 prior와 LLM agent의 결합 축 | 전체 |

**핵심 argument 후보**:
1. "Graph-prior-conditioned agentic refinement" (Filter FL-I + S-IV 결합)
2. "Deterministic symbolic verification as guardrail on neural pipeline" (FL-III + E-III)
3. "Tiered schema linking: from GAT score to PCST membership to LLM verdict, each tier as explicit evidence" (pipeline 전반)

---

## 6. 닫힌 주제 (재탐색 금지)
- **방안 A (Score-driven PCST cost)**: `ScoreDrivenPCSTExtractor` BO2 tuned. 완료.
- **방안 B (Bayesian Optimization)**: 완료, F1=0.6751.
- **Idea 2 (Product Cost) / Idea 4 (Component-Aware)**: I24b F1=0.7063 완료.
- 위 변형은 기본값으로만 인용, **신규 실험에서 재변형 금지** (memory rule).

---

## 7. 실험 문서화 규칙 (재확인)
모든 신규 실험 실행 후 3개 파일 동기 업데이트 (memory rule):
- [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md)
- [EXPERIMENT_CATALOG.md](EXPERIMENT_CATALOG.md)
- [EXPERIMENT_ID_MIGRATION.md](EXPERIMENT_ID_MIGRATION.md)

**메트릭 표기**: R, P, F1 순서 / 소수점 4자리.

**ID 명명**:
- 단일 모듈 신규: `abl_{module}_{feature}_{nn}` (예: `abl_ext_path_01`)
- 통합 실험: `int_{nn}_{short_desc}` (예: `int_04_ns_full`)
- a05 라인 연장: `abl_a05_13 ~ a05_22` (filter 신규 3축)

---

## 8. 디렉토리 구조 (신규)

```
configs/experiments/abl/
├── a05_filter_agentic/       # 기존 (a05_01~12) + 신규 (a05_13~22)
├── build/                    # 신규 — B-I/B-II/B-III pilot
│   ├── fk_reach/
│   ├── linegraph/
│   └── rfm_tokens/
├── sel/                      # 신규 — S-I ~ S-V
│   ├── ns_l1/
│   ├── rfm/
│   ├── ehgat/
│   ├── xllm/
│   └── rl/
├── ext/                      # 신규 — E-I ~ E-III
│   ├── louvain/
│   ├── path_ensemble/
│   └── fk_prior/
└── integration/              # 신규 — int_01 ~ int_08
```

3개 디렉토리(configs/outputs/logs) **동일 하위 구조** 유지 (CLAUDE.md rule).

---

## 9. 검증 & 리스크 관리

### 검증 공통
- 모든 실험은 **Recall/Precision/F1 (4소수)** + **쿼리당 처리시간** 리포트.
- 진행중 실험은 ETA 함께 보고 (memory rule).
- End-to-end SQL execution accuracy는 이 단계에서 미포함 (schema linking 단위 평가 유지).

### 리스크 맵

| 리스크 | 영향 | 대응 |
|--------|------|------|
| Builder B-III metadata 크기 폭발 | 캐시 용량 | matrix sparsification, 큰 DB만 옵션 적용 |
| vLLM logprobs 미지원 | S-IV/FL-II 블록 | GPT-4o-mini logprobs proxy or fallback to text parsing |
| RL(GRPO) 학습 불안정 | S-I 블록 | KL constraint + supervised ckpt warm-start |
| Neurosymbolic 3-layer combined regression | int_04 실패 | 각 Layer 독립 ablation 먼저 검증 (int_01~03) |
| LLM 비용 증가 (Filter FL-I 5-step ReAct) | Latency/budget | F4 uncertainty gating으로 hard query만 |
| FK reachability가 DB에 없는 경우 | Layer 전체 비활성 | Gracefully fallback to baseline |
| **SuperNode v2 (directed + top-k Raw) over-smoothing 재등장** | int_05 / Proposal D·E 회귀 | 단방향 edge 로 distant node 가 SN 신호를 못 받는 경로 발생 가능. Top-k 가 너무 작으면 gold 누락. → Proposal D (directed) · E (top-k) ablation 을 int_05 합성 전에 독립 검증, num_layers 와 함께 sweep. 근거: 2026-04-21 advisor 의견 3·4. |

---

## 10. 세션 워크플로우 (루트 ↔ 서브세션)

### 서브세션 진입 시 체크리스트
각 모듈 세션(builders/selectors/extractors/filters)은 시작 시 **순서대로** 읽는다:
1. `/home/hyeonjin/thesis_refactored/CLAUDE.md` — 전역 규칙 (읽기 전용)
2. `/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md` — **본 문서** (루트 로드맵, 읽기 전용)
3. `src/modules/{module}/CLAUDE.md` — 모듈 맥락 (수정 가능)
4. `src/modules/{module}/EXPERIMENT_PLAN_{module}.md` — 모듈 내부 구현 상세 (주 작업 스펙)

### 수정 책임 분할
| 문서 | 수정 책임 |
|------|----------|
| `CLAUDE.md` (루트) | **루트 세션만** |
| `EXPERIMENT_PLAN.md` (루트, 본 문서) | **루트 세션만** |
| `src/modules/{module}/CLAUDE.md` | 해당 모듈 세션 |
| `src/modules/{module}/EXPERIMENT_PLAN_{module}.md` | 해당 모듈 세션 |
| `EXPERIMENT_HISTORY.md` / `CATALOG.md` / `ID_MIGRATION.md` | **실험 실행 주체가 모두 갱신** (memory rule) |

### 루트 세션의 책임
1. 본 문서를 기준으로 각 서브세션에 작업 분배 (subsession 호출 시 루트 PLAN + 해당 모듈 PLAN 참조 명시).
2. Phase A (infrastructure) 완료 시까지 다른 phase 블록.
3. **int_04가 "논문 주력 결과"** 임을 각 서브세션에 공유 — 모든 모듈이 "이 실험에 들어갈 품질"로 구현.
4. 루트 PLAN 변경 시 각 서브세션 PLAN과의 정합성 확인 (Cross-Module Matrix §1).
5. 통합 실험(int_01~08)은 **루트 세션이 직접 실행** — 여러 모듈이 동시 개입하므로.

### 각 서브세션의 책임
- 담당 plan 파일(`EXPERIMENT_PLAN_{subsession}.md`) 을 그대로 작업 스펙으로 사용.
- 인터페이스 계약(`build`, `select`, `extract`, `refine` signature)은 **절대 깨지 않도록** 재확인 — 하류 모듈 파손 방지.
- 신규 구현마다 HISTORY/CATALOG/ID_MIGRATION 3파일 동기 갱신.
- 루트 PLAN의 변경이 필요한 판단(예: 제안 재분류, 우선순위 변경, 새로운 의존성 발견)은 **루트 세션에 에스컬레이션**.
