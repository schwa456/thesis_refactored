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

| 범주 | Anchor | R | P | F1 |
|------|--------|---|---|-----|
| **전체 최고 (Wave 1.5, 2026-04-22)** | `s04_stagewise_qcond_gat_basic` (QCond encoder + GAT α=0.85 + Basic PCST + XiYan) | **0.8169** | 0.7605 | **0.7877** |
| Ensemble (legacy, 2×2×2 anchor) | `abl_ens_basic_xiyan` (Cosine+GAT α=0.85 + Basic PCST + XiYan) | 0.8149 | 0.7597 | 0.7863 |
| Ensemble + Enriched Builder | `abl_ens_enriched_xiyan` (E1, Enriched Builder) | 0.6658 | 0.8147 | 0.7327 |
| Ensemble + Triplet | `abl_ens_triplet_xiyan` (E2) | 0.6823 | 0.8139 | 0.7424 |
| Direct variant 최고 | `abl_a03_17` (SuperNode-Direct + Fixed PCST + XiYan) | 0.6761 | 0.7128 | 0.6940 |
| Filter 최고 | `abl_a05_02` (Reflection, anchor=a03_17) | 0.7320 | 0.6833 | 0.7068 |

**핵심 관찰**:
- **새 전체 최고** (2026-04-22 Wave 1.5): QCond encoder + GAT α=0.85 + Basic PCST + XiYan 이 `abl_ens_basic_xiyan` 대비 Recall **+0.0020**, F1 **+0.0014** 로 소폭 상회. 주력 결과 anchor 재지정.
- **Precision 상한(≈0.81)은 Builder가 결정** (Enriched/Triplet) — Wave 1.5 는 Basic Heterograph 기반이라 precision 0.76 대에 멈춤. Enriched × QCond 결합은 미검증.
- **Recall 상한은 Filter의 restore path가 결정** (Reflection이 prune-only 한계 돌파) — QCond+GAT 는 restore 없이도 R=0.82 돌파.
- 두 상한의 **교차 결합이 아직 미검증** (E1/E2 × QCond+GAT × Reflection).

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
- **Wave 2 (active — 자산 준비 완료, Phase 1/2 실행 승인 대기, 2026-04-22 ~ 25)**: Selector ablation 축 심화 (Proposals C → D → E).
  - 🔧 **Proposal C** `abl_sel_diameter_layers` — num_layers ∈ {L1, L2, L3, L6, L7} 5-cell **global fixed sweep** (L6 = max(D_max over BIRD dev 11 DB), L7 = D_max+1). Anchor `s04_04_qcond_a0_xiyan` (QCond Raw). Configs `configs/experiments/s04_ablation/diameter_layers/layers_L{1,2,3,6,7}.yaml` + `scripts/run_wave2_proposal_c.sh` 준비 완료. 실행 구조: **Phase 1 (5× GAT 학습 ~25h, `VLLM_AUTOKILL=1`) → Phase 2 (vLLM 재기동 + 5× inference)**. vLLM kill / sequential script kill 모두 승인 완료, **user 최종 착수 승인 대기**.
  - ⏸ **Proposal C H2 (per-DB dynamic num_layers)** — §4.2 H2 (DB 별 D_max 맞춤 층수) 는 `src/modules/selectors/ensemble_selector.py` v1 에 `db_name` threading 부재로 **측정 불가**. Selector 세션에 래퍼 구현 에스컬레이션 (`DECISIONS.md` 2026-04-22 17:05 엔트리 (b) + 에스컬레이션 #1). Wave 2.5 또는 별도 mini-wave 로 분리, H1 global fixed 와 병행 개발 가능.
  - 🔜 **Proposal D** `abl_sel_supernode_directed` — SN↔node bidirectional → SN→node directed (1 재학습). Anchor `s04_05_sn_qcond_xiyan` 계 (주의: §8-1 SuperNode split-order bug 수정된 `train_gat.py` 기준 재학습 필요).
  - 🔜 **Proposal E** `abl_sel_supernode_topk` — Raw Score k ∈ {3,5,10,20} Phase 1 (4 재학습). Phase 2 (CE/Cosine 확장) 는 Phase 1 성공 시 승격.
  - ⏳ 순차 실행 이유: GPU 자원 (CUDA_VISIBLE_DEVICES=0,1) 경합 방지 + Anchor 의 재학습 의존 (§8-1 bug fix 적용 후 SuperNode 라인 재학습 필수).
- **Wave 3 (planned, 2026-04-26 ~ 28)**: 발표 패키징.
  - 🔜 **Proposal F** `abl_ext_steiner_backbone_report` — 기존 a03_15/18 데이터 재조직 리포트 (신규 실행 없음, analyzer 단독 큐).
  - 🔜 **Proposal A 확장 셀** (시간 여유 시) — Ensemble Raw 축의 stagewise cumulative 가 cosine 대비 reportable gap 을 보이면 추가 셀 확보.
  - ⛔ **Proposal B** `abl_bld_t2t_edge` — 2026-04-28 이후로 순연 (graph cache regen + GAT 재학습 비용 ≈ 11h, 스토리라인 우선순위 최하).
- **Wave 4 (post-2026-04-28)**: Filter agentic 트랙 재개.
  - 🔜 `a05_filter_agentic` 전체 12 실험 ([~/.claude/plans/vivid-sprouting-sunbeam.md](~/.claude/plans/vivid-sprouting-sunbeam.md)).
  - ⚠ 해당 plan anchor (`abl_ens_basic_xiyan`, F1=0.7863) **outdated** — Wave 1.5 new top (`qcond_gat_basic`, F1=0.7877) 로 anchor refresh 필요. Wave 4 kickoff 전 filter 세션 에스컬레이션으로 plan 갱신.

### Phase A — Infrastructure (선결)
- [ ] **Builder B-III** — FK reachability metadata precompute (가장 중요)
- [x] **Schema Graph Diameter precompute** — 각 DB heterograph 의 schema 노드 간 최대 shortest-path `D_max` 계산, `data/processed/*_diameter.pt` 캐시. **2026-04-22 완료** — `scripts/build_diameter_cache.py` + `dev_diameter.pt` (NAS symlink). B-III FK reachability 루틴과 1 패스 공유 예정 (BFS/Dijkstra 경로에서 동시 집계 확장). 소비자: Proposal C (`abl_sel_diameter_layers`, num_layers ∈ {1,2,3,D_max,D_max+1} sweep). 근거: 2026-04-21 advisor Q1.
- [ ] Builder B-II — LineGraph 변환기
- [ ] Builder B-I — RFM serialize API
- [ ] `src/llm_client/api_handler.py` — vLLM logprobs 지원

### Phase B — Low-risk Quick Wins
- [ ] **Selector S-V** (Neurosymbolic L1, λ 튜닝)
- [ ] **Extractor E-III** (FK prior cost adjustment)
- [ ] **Filter FL-III** (Symbolic Verifier, detect-only 먼저)
- [ ] **Extractor E-II** (Pathfinding ensemble, MSTExtractor 재활용)
- [ ] **Base heterograph T2T edge toggle** — Builder B-II 스펙 확장, table↔table 직접 edge on/off 플래그. s04 재학습 시 결합. 소비자: Proposal B (`abl_bld_t2t_edge`). Wave 3/4 로 순연. 근거: 2026-04-21 advisor 의견 2.

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
