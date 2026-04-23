# 지도교수 의견 수렴 — 2026-04-21 QCondGAT 상세 Ablation 지시

> Planner 세션이 **지도교수님의 피드백**을 받아 PLAN 개정으로 전환한 완료본.
> Draft: `planning/advisor_inputs/_draft.md` (2026-04-21 승격 후 리셋됨).
>
> ## 핵심 전제 (이 미팅 당시)
> 지도교수님 인지 범위: **Query-Conditioned GAT 수준까지** (2026-04-10 5 아이디어 + 방안 A/B 구현 + recall/precision/F1 값).
> 루트 `EXPERIMENT_PLAN.md` 전체는 **미공유**.
>
> ## 레퍼런스
> - 이전 미팅 분석: [advisor_meeting_ideas_analysis.md](../../notebooks/analysis_results/advisor_meeting_ideas_analysis.md)
> - QC-GAT 구현 기록: [query_conditioned_training.md](../../notebooks/analysis_results/query_conditioned_training.md)
> - 병목 분석: [diagnostic_state_2026_04_21.md](../../notebooks/analysis_results/diagnostic_state_2026_04_21.md)
> - s06 over-smoothing ablation: `outputs/analysis/s06_bottleneck*/`

---

## 1. 메타 & 브리핑 ledger

### 1.1 미팅 메타
- **날짜**: 2026-04-21
- **형식**: 정기 미팅 (90분)
- **주제 한 줄 요약**: Node Selection 단계에서 현재까지 제안된 방식의 요소별 성능을 분석해보자
- **관련 이벤트**: 예심 후속
- **현재 PLAN phase (internal)**: 사용자 입력 — "현재 PLAN과 상관 없음" (planner 주: §5에서 PLAN 파급 별도 분석)

### 1.2 지도교수 인지 범위 (브리핑 ledger)

**이번 미팅 직전까지 공유된 범위**:
- 2026-04-10 예심 후 5 아이디어 제시 (prior 분석: `advisor_meeting_ideas_analysis.md`)
- 아이디어 5 (Query-Conditioned GAT) 구현: 방안 A concat / 방안 B super node
- 관련 실험 결과 (구두/수치 공유 수준): concat / super node 방안 모두의 R/P/F1 vs baseline 및 기존 제시 모델 R/P/F1
- 논문 초안 공유 여부: no

**아직 공유되지 않은 PLAN 영역**:
- 9 제안 통합 로드맵 (루트 PLAN §1 Cross-Module Matrix) 전체
- Neurosymbolic 3-layer 프레임 (B-III / S-V / E-III / FL-III)
- int_04 논문 주력 결과 후보 지정
- 최신 2×2×2 재측정 결론 (#6 E+Basic+X, R=0.8149 / P=0.7597 / F1=0.7863 우세)
- Phase A~E 우선순위 체계

**이번 미팅에서 새로 공유한 내용** (→ 다음 미팅부터 "공유된 범위"로 승격):
- Query-Conditioned GAT 기반 BCE Loss 학습 결과
- 성능 저조 상세 분석 → GAT 모델의 Over-smoothing 문제 발견 보고

### 1.3 피드백 대상
- **1차 대상**: Query-Conditioned GAT 방안 A 및 방안 B 구현 내용 및 실험 결과
- **관련 실험 ID**: `s04_01_qcond_a085_xiyan`, `s04_02_supernode_a070_xiyan`, `s04_03_supernode_a085_xiyan`, `s04_04_qcond_a0_xiyan`, `s04_05_supernode_a0_xiyan` (5개 s04 실험군)
- **관련 코드**: `src/models/gat_network_v2.py` (QCond concat + num_layers), `src/models/gat_network.py` (SuperNode 구현 위치), `src/models/direct_classifier.py` (BCE head)
- **관련 문서**: `outputs/analysis/gat_bottleneck_qcond/`, `outputs/analysis/gat_bottleneck/`

---

## 2. 교수님 의견 원문 (raw capture)

### 의견 1 — BCE 영향력 분리를 위한 다축 Ablation
> "BCE의 영향력을 분석하려면 더 자세히 ablation을 수행해야 한다. Ensemble 모델의 Raw Score에 대해서도 Ablation을 진행하고, Query-Conditioned 모델에도 Raw Score를 적용해 봐서 어떤 요소가 기여하는지 세세하게 검증해보자. 그리고 Selection, Extraction, Filter 단계별로 성능을 비교해보자. Baseline도 함께."

- **맥락/근거**: 기존 Ensemble Selection 모델에서 제시한 Raw Score의 기여도 및 Query-Conditioned 모델에서의 정합성 분석
- **질문**: X

### 의견 2 — Over-smoothing 대응 (Diameter 기반 Layer + T2T Edge Ablation)
> "GAT 모델의 Over-smoothing에 관해서, Graph 표현의 Diameter 같은 걸로 Layer 수를 정해볼 수 있지 않을까? 그리고 Table to Table Edge의 효과도 Ablation을 통해 분석해보자."

- **맥락/근거**: QCond 모델 성능 저조 → GAT Over-smoothing 진단 보고에 대한 처방 제시
- **질문**: X (단, "정해볼 수 있지 않을까" 자체가 open question)

### 의견 3 — SuperNode Directed Edge
> "Query-Conditioning 방식은 Concatenation 보다 Super Node가 더 낫지 않나 하는 생각이 든다. 다만 Super Node와의 연결 Edge를 Super Node에서 나가는 단방향 Edge로 만들어서 Super Node가 희석되지 않게 해 보자."

- **맥락/근거**: Over-Smoothing 대응 중 SuperNode dilution 문제에 대한 구조적 해법
- **질문**: X

### 의견 4 — SuperNode Top-k Selective Connection
> "그리고 Super Node를 연결할 때 모든 Node에 Edge를 연결하지 말고, Cross Entropy든 Cosine Similarity든 Raw Score든 사용해서 Top-k Node를 선택한 뒤 거기에만 Super Node를 연결하는 건 어떻겠느냐."

- **맥락/근거**: 의견 3의 연장. SuperNode dilution 추가 방어
- **질문**: "어떻겠느냐" — open suggestion

### 그 외 논의
- **G1**: 다음 보고 시 MST + PCST 조합의 SteinerBackbone 모델 성능도 비교 분석 포함
- **G2**: 항상 각 단계별(Selection / Extraction / Filter)로 성능을 비교할 것 — **보고 규범**

---

## 3. 의견 분류

| 의견 # | 라벨 | 한 줄 해석 |
|--------|------|-----------|
| 1 | **directive** | s04/s05 실험군에 Raw Score × 모델 × 단계 다축 ablation 확장 |
| 2 | **directive + question** | Diameter 기반 num_layers 결정 + T2T edge ablation — 방법론 자체는 open (Q1: diameter 정의) |
| 3 | **directive** | SuperNode ↔ Node edge를 SN→Node 단방향으로 변경 |
| 4 | **suggestion** | SN 연결을 Top-k (CE/Cos/Raw 기준) 로 희소화 |
| G1 | **directive** | 다음 보고에 SteinerBackbone 비교 포함 |
| G2 | **directive / 보고 규범** | Selector/Extractor/Filter 단계별 메트릭 제시 습관화 |

---

## 4. 브리핑 범위 내 직접 영향 — **advisor 가 볼 수 있는 레벨**

| 의견 # | 대상 아티팩트 | 요구 변경 | 강도 |
|--------|--------------|----------|------|
| 1 | `src/modules/selectors/ensemble_selector.py` (α=0/0.85/0.70 스윕), 기존 `s04_04/05` (α=0 실험) 재활용 + Ensemble α=0 (Raw Score only) 신규 매트릭스 | QCond × Raw / Ensemble × Raw / Baseline 을 Selector 단독 + Extractor 통과 + Filter 통과 3단계 모두 측정 | **대** |
| 2 | `src/modules/builders/line_graph_builder.py` (`skip_macro_edges` 플래그 **이미 존재**) + `src/models/gat_network_v2.py` (`num_layers` 현재 default=3, configurable) + `src/modules/builders/graph_builder.py` (diameter 계산 추가 필요) | (a) 기본 heterograph에 **T2T edge on/off** switch 노출 (line_graph 가 아니어도 base graph 단계 — 현재 line_graph 에서만 switch) (b) DB별 schema diameter 계산 → config 자동 num_layers 산출 | **대** |
| 3 | `src/models/gat_network.py` SuperNode 구현 | SN ↔ Node 양방향 edge → SN → Node 단방향으로 변경 (schema node는 SN을 못 보게) | **중** |
| 4 | `src/models/gat_network.py` SuperNode 연결 로직 + 전처리 (Top-k 선택기) | 모든 node 연결 → {CE / Cos / Raw} 기준 Top-k 선택 후 그 k개에만 edge. k 스윕 필요 | **중** |
| G1 | `src/modules/extractors/` (MST + PCST 조합 = SteinerBackbone), 이미 `abl_a03_15`/`abl_a03_18` 존재 | 기존 결과를 **보고 패키지에 명시적으로 포함** (신규 실행 불필요 가능) | **소** (문서/리포트 작업) |
| G2 | `notebooks/log_analyzer.ipynb`, `src/analysis/` 리포트 템플릿 | 모든 보고에 Selector / Extractor / Filter 3단계 R/P/F1 (4자리) 표 포함 | **소** (루틴 변경) |

**→ 대부분 Selector / Builder 모듈 세션 구현 범주**. §9 에스컬레이션으로 이관.

---

## 5. Scope gap — Unbriefed PLAN 파급 영향 — **핵심**

사용자는 §1.1에서 "PLAN과 상관 없음" 으로 명시. 그러나 planner 관점에서 아래 파급이 잠재적으로 존재. **각 항목은 pending-clarification** — 사용자 승인 전까지 PLAN 문서 수정 금지.

| 의견 # | 파급되는 PLAN 요소 (unbriefed) | 파급 이유 | 반영 방향 (제안) |
|--------|-------------------------------|----------|----------------|
| 1 | 루트 PLAN §3.1 Synergy Grid int_01~06 의 reporting 포맷 / 논문 §IV Experiments Ablation 축 | Raw vs GAT 기여 분리 요구는 int_04 (Neurosymbolic 3-layer) 에도 적용 — S-V L1 score boost 의 기여가 GAT 기여인지 α-shift 효과인지 분리 필요 | 논문 Ablation subsection 에 "α sweep" 축 명시화 (실험 추가 불요 — reporting 축) |
| 2 (diameter) | 루트 PLAN §4 Phase A — **"Schema Graph Diameter Metadata" 서브태스크 신설 제안** (Builder 공통 infra) | Diameter 는 **B-III FK Reachability** 와 **동일 레벨의 graph-level metadata**. 동시 precompute 가능 | Phase A 에 1 bullet 추가: `Builder.diameter_metadata` (per-DB diameter, 캐시에 함께) |
| 2 (T2T edge) | Builder B-II (LineGraph) + Selector S-III (EHGAT) | `line_graph_builder.skip_macro_edges` 는 line_graph 단계 스위치. **Base heterograph 단계의 T2T on/off** 는 아직 builder 인프라에 없음 → B-II 스펙 확장 or 신규 B-IV (T2T toggle) | B-II 스펙에 "T2T base-graph toggle" 항목 추가. 모듈 PLAN 업데이트 에스컬레이션 |
| 3 + 4 | 루트 PLAN §3.1 **int_05_direct_ns** (DirectGAT-SuperNode + S-V) + Selector S-V (Neurosymbolic L1 soft routing) | int_05 의 "SuperNode" 전제가 현재 양방향 + 전체 연결. 의견 3/4 반영 시 SN v2 로 재정의됨. S-V 의 soft routing 철학(확률적 score boost)과 Top-k selective edge 는 개념적으로 유사 → S-V 구현 힌트 제공 | int_05 전제 "SuperNode v2 (directed + top-k)" 로 명시. S-V L1 설계 노트에 Top-k edge 정책 참고로 기록 |
| G1 | Extractor E-II (Pathfinding + PCST Ensemble) 의 참고 baseline | SteinerBackbone 은 E-II 의 한 구현 — 이미 존재. 새 실험 필요 없고, 기존 결과를 E-II 설계 근거로 끌어오면 됨 | E-II 설계 문서에 a03_15/a03_18 결과 링크. PLAN 변경 최소 |
| G2 | 루트 CLAUDE.md / 메모리 rule | 단계별 보고는 **memory rule "R/P/F1 4자리"** 와 함께 격상. 모든 세션의 보고 규범 | 루트 세션에 CLAUDE.md "진행중 실험 현황 보고" 섹션에 "Selector/Extractor/Filter 3단계 분해 포함" 추가 요청 |

**판단 원칙 적용**
- 의견 1 / G2 → Reporting 규범 강화 (실험 없이 논문·보고 포맷 조정)
- 의견 2 (diameter) → Phase A 인프라 **quick win** — B-III 병행 작업으로 즉시 추가 가치
- 의견 2 (T2T) → B-II 스펙 확장으로 s04 재학습 시 바로 반영 가능
- 의견 3 + 4 → int_05 재정의 — 실행은 Phase E 전 Super Node 재학습 필요
- G1 → 재실험 불요, 보고 포맷만

**주의**: §5 파급은 planner 해석. 교수님 의사가 아님. 의견 3/4 가 S-V 에 이어지는지는 §10 에 재확인 항목으로 대기.

---

## 6. 기존 PLAN과의 충돌 / 정합성 체크

- [x] **PLAN §6 닫힌 주제**: 충돌 없음. 방안 A/B (PCST cost), Idea 2/4 (Product/Component) 재탐색 아님.
- [x] **PLAN §2 Dependency Graph**: 의견 2 의 Diameter infra 는 **B-III 와 병렬** (상충 아님). T2T toggle 은 B-II 스펙 확장이지 새 상류 의존 발생 아님.
- [x] **논문 주력 (int_04)**: 직접 변경 없음. 의견 1 reporting 축은 int_04 논문 Ablation 에 자연스럽게 흡수 가능.
- [x] **Phase 순서 어김**: 없음. Phase A 에 신규 서브태스크 추가 제안이 유일.
- [x] **메트릭 rule**: 정합. 오히려 단계별 세분화로 강화.
- [x] **최신 2×2×2 결론 (#6 E+Basic+X, F1=0.7863)** 충돌: **교수님은 이 결론을 모름**. 의견 1 의 "Selection/Extraction/Filter 단계 비교" 를 수행할 때 어떤 Extractor(Basic vs Adaptive)를 기준으로 할지 프레이밍 불일치 가능. **§11 브리핑 후보 #1 로 승격**.
- [x] **실행 중/큐잉 실험**: a05_05~10 Qwen filter 큐에 영향 없음. FK-Steiner percentile hold 는 그대로. 단, G1 (SteinerBackbone 보고) 은 a03_15/a03_18 결과 **재문서화** 를 요구 → 루트 세션 리포트 작업.
- [x] **모듈 PLAN 중복**: `EXPERIMENT_PLAN_selectors.md` 에 s04/QCond 계열 언급 없음 (정합성 공백). 의견 1~4 를 계기로 해당 모듈 PLAN 에 **"QCondGAT 계열 ablation track"** 섹션 신설 필요 → 에스컬레이션.

---

## 7. PLAN 개정 초안 (proposed diff)

> **2026-04-21 사용자 승인 완료**. 모든 diff 항목 **approved** → 루트 세션이 `EXPERIMENT_PLAN.md` 실제 수정 (§9 Root 에스컬레이션 프롬프트 사용).

### 7.1 §3.1 Synergy Grid (int_05 전제 재정의) — **approved**
- **Before**: `int_05_direct_ns` — Enriched + B-III | DirectGAT-SuperNode + S-V | E-III | FL-III + XiYan
- **After**: int_05 전제 주석에 "SuperNode v2 (directed SN→node + top-k selective edge, Raw Score 기준 선별)" 명시

### 7.2 §4 Phase A (Infrastructure) — **Diameter Metadata 서브태스크 신설** — **approved**
- **Before**: Phase A 3항목 (B-III, B-II, B-I, vLLM logprobs)
- **After**:
  - 추가 bullet: **"Schema Graph Diameter precompute"** — 각 DB 의 heterograph 에서 schema 노드 간 **최대 shortest-path (D_max)** 계산 후 메타데이터 캐시 (`data/processed/*_diameter.pt`). 비용: O(V·(V+E)) per DB, BIRD dev 11 DB 에서 부담 없음.
  - 동일 Phase 에서 B-III 의 FK reachability 계산 루틴과 **1 패스 공유**.

### 7.3 §4 Phase B (Low-risk Quick Wins) — **T2T edge ablation 추가** — **approved**
- **Before**: S-V / E-III / FL-III / E-II
- **After**: 추가 bullet — **"Base heterograph T2T edge toggle"** (B-II 스펙 확장, Builder 세션 작업). s04 재학습 시 바로 결합 가능.

### 7.4 §9 리스크 맵 (신규 리스크) — **approved**
- **Before**: 6 리스크 항목
- **After**: 추가 행 — "SuperNode v2 (directed + top-k Raw) 에서 over-smoothing 재등장 가능성 — 단방향 edge 로 정보 흐름이 단절되면 distant node 가 SN 신호를 못 받음. Top-k 가 너무 작으면 gold 누락. → 별도 ablation 필수."

### 7.5 §1 Cross-Module Matrix / §5 논문 매핑 / §6 닫힌 주제
- 변경 없음.

---

## 8. 신규 실험 제안 (요약)

각 제안은 `experiment_plan_template.md` 로 별도 작성 권장 (`planning/proposals/<id>.md`).

| # | 실험 ID 후보 | 의견 # | 요지 | 예상 Phase |
|---|-------------|--------|------|-----------|
| A | `abl_sel_rawscore_stagewise` | 1, G2 | Baseline / Ens Raw / Ens GAT / QCond Raw / QCond GAT × 3단계(S/E/F) — 5×3 매트릭스 | Phase B |
| B | `abl_bld_t2t_edge` | 2 | Base heterograph T2T on/off × QCond Direct (s05 anchor) × XiYan | Phase B |
| C | `abl_sel_diameter_layers` | 2 | DB별 **최대 diameter D_max** 계산 후 num_layers ∈ {1, 2, 3, D_max, D_max+1} 스윕 (Q1 답변 반영) | Phase A quick-win |
| D | `abl_sel_supernode_directed` | 3 | SN→node 단방향 edge 버전 (기존 양방향 대비) | Phase B |
| E | `abl_sel_supernode_topk` | 4 | **Phase 1 (권장, Q2 답변 반영)**: Raw Score 기준 Top-k ∈ {3, 5, 10, 20} 스윕. **Phase 2 (성능 양호 시 확장)**: CE / Cosine 기준 추가. | Phase B |
| F | `abl_ext_steiner_backbone_report` | G1 | 신규 실행 없음 — 기존 a03_15/a03_18 결과 재조직해 보고 패키지 포함 | 문서화 |

**선행성**: C (Diameter) → B,D,E 재학습 시 num_layers 튜닝 근거 제공. A 는 기존 s04/s05 결과만으로 일부 셀 채움 가능.

**스토리라인 우선순위** (Q4 답변 반영 — 2026-04-28 15~20분 발표 중요 지점만): **A > F > C > D > E > B**
- A: "Raw×Model×Stage" 매트릭스 → BCE 기여 분리 서사 (의견 1 직격)
- F: SteinerBackbone 비교 (G1 직답 — 기존 결과 재조직)
- C: Diameter → num_layers 튜닝 결과 (의견 2 처방 효과 검증)
- D: SuperNode directed edge (의견 3)
- E: SuperNode top-k Raw (의견 4)
- B: T2T edge ablation — 시간 여유 있을 때만

---

## 9. 에스컬레이션 필요 항목

| 대상 세션 | 요청 내용 | 권장 프롬프트 (copy-paste용) |
|----------|----------|---------------------------|
| **Selector 세션** | `EXPERIMENT_PLAN_selectors.md` 에 "s04/s05 QCondGAT 계열 ablation track" 섹션 신설. 의견 1 (Raw Score ablation), 의견 3 (directed SN edge), 의견 4 (top-k SN connection) 구현 스펙 작성 | "먼저 /home/hyeonjin/thesis_refactored/src/modules/selectors/CLAUDE.md 및 EXPERIMENT_PLAN_selectors.md 를 읽고, planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md §4 의견 1/3/4 에 따라 (1) Raw Score ablation 축, (2) SuperNode directed edge, (3) SuperNode top-k selective connection 구현 스펙을 모듈 PLAN 에 추가하라." |
| **Builder 세션** | `EXPERIMENT_PLAN_builders.md` B-II 스펙 확장 (base heterograph T2T toggle) + 신규 "Schema Graph Diameter precompute" 서브태스크 추가 | "먼저 /home/hyeonjin/thesis_refactored/src/modules/builders/CLAUDE.md 및 EXPERIMENT_PLAN_builders.md 를 읽고, planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md §4 의견 2 에 따라 (1) base heterograph 수준의 table_to_table edge on/off 토글, (2) per-DB schema graph diameter precompute (B-III reachability 와 동시 실행) 두 작업을 모듈 PLAN 에 추가하라." |
| **Analyzer 세션** | Selector / Extractor / Filter 단계별 R/P/F1 (4자리) 분해 리포트. 기존 s04/s05 + baseline 실험 대상. Raw Score × Model 축 포함 | "먼저 /home/hyeonjin/thesis_refactored/src/analysis/CLAUDE.md 를 읽고, outputs/experiments/s04_*/ 및 s05_*/ 및 baseline 실험의 로그에서 Selector(top-k hit rate) / Extractor(post-PCST R/P/F1) / Filter(post-XiYan R/P/F1) 단계별로 분해된 비교표를 만들어 notebooks/analysis_results/stagewise_qcond_ablation.md 로 저장하라. Raw Score(α=0) vs GAT-blend(α>0) 행 분리. 의도: 지도교수 2026-04-21 의견 1 대응." |
| **Root 세션 (PLAN 수정)** | **2026-04-21 사용자 승인** 후 `EXPERIMENT_PLAN.md` 4건 diff 반영 (§7): int_05 전제 재정의 / Phase A Diameter precompute 서브태스크 / Phase B T2T toggle / §9 리스크 SN v2 over-smoothing 재등장 | "먼저 /home/hyeonjin/thesis_refactored/CLAUDE.md 및 EXPERIMENT_PLAN.md 를 읽고, planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md §7 의 4건 diff (7.1~7.4, 모두 사용자 승인 완료) 를 `EXPERIMENT_PLAN.md` 에 반영하라. 동일 변경을 `planning/DECISIONS.md` 에서 승인 완료 상태로 재확인한다." |
| **Root 세션 (보고·규범)** | (1) 2026-04-28 보고 패키지에 a03_15/a03_18 SteinerBackbone 결과를 단계별 cumulative R/P/F1 로 재조직, (2) 루트 CLAUDE.md "진행중 실험 현황 보고" 에 "Selector/Extractor/Filter 3단계 분해 포함" 규범 추가 검토 | "먼저 /home/hyeonjin/thesis_refactored/CLAUDE.md 를 읽고, planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md §4 G1/G2 및 §10 Q3 답변(cumulative) 에 따라 (1) abl_a03_15/a03_18 Steiner 결과를 cumulative 단계별 R/P/F1(4자리) 로 재조직한 슬라이드 원시자료 작성, (2) CLAUDE.md '진행중 실험 현황 보고' 섹션에 단계별 분해 규범 추가 여부 판단하라." |

---

## 10. 교수님께 재확인 필요한 사항 — **2026-04-21 사용자 답변 수렴 완료**

- **Q1 (의견 2)**: "Diameter 기반 num_layers" 에서 diameter 정의
  - **A1**: **Per-DB heterograph 최대 diameter (D_max)**. Layer 수 결정 문제이므로 최대치가 의미 있음.
  - **반영**: §8 실험 C 스펙 `num_layers ∈ {1, 2, 3, D_max, D_max+1}` 확정. §7.2 Phase A precompute 루틴 "max shortest-path" 로 확정. **주의**: BIRD 11 DB 중 D_max 가 큰 DB 에서는 over-smoothing 재등장 가능 → §7.4 리스크 테이블에 명시.
- **Q2 (의견 4)**: Top-k 선택 기준 (CE / Cosine / Raw Score) 중 어느 것?
  - **A2**: **하나만 권장 실행, 성능 양호 시 확장**.
  - **Planner 판단 (권장 1개)**: **Raw Score**. 근거: (a) 의견 1 이 이미 Raw Score ablation 을 요구 → 동일 축 재활용, (b) 기존 Ensemble Selector 의 raw_score 인프라 그대로 사용, (c) BCE/CE 는 현재 bottleneck 분석 대상이라 기준축으로 부적절.
  - **반영**: §8 실험 E 2 phase 로 나눔 — Phase 1 = Raw Score × k ∈ {3,5,10,20}, Phase 2 = 성능 양호 시 CE/Cosine 확장.
- **Q3 (의견 1)**: "단계별 성능 비교" 정의
  - **A3**: **Cumulative** (Selector output → Extractor input → Filter input 순으로 파이프라인 누적 R/P/F1).
  - **반영**: Analyzer 요청 (§9) 에 cumulative 명시. §8 실험 A 의 3단계 = Selector top-k, Extractor post-PCST, Filter post-XiYan 의 cumulative R/P/F1. Root 세션 G1 보고 패키지도 cumulative 기준.
- **Q4 (형식)**: 다음 보고 일정 / 형식 / 분량
  - **A4**: **2026-04-28 (1주 뒤)**, **15~20분 발표 분량**, 전체 실험 중 **중요 지점만** 선별 (준비는 더 많이). 우선순위 필요.
  - **반영**: §8 스토리라인 우선순위 확정 (A > F > C > D > E > B). §11 다음 브리핑 일자 2026-04-28 로 고정.

**처리 완료**: §7 PLAN diff 4건 모두 **approved**. §8 C/E 스펙 확정. 이후 단계는 §9 에스컬레이션.

---

## 11. 다음 브리핑 후보

> 이번 피드백으로 드러난 **다음 미팅에서 공유하면 좋을 PLAN 영역**.

- [ ] **후보 1 (우선)**: 최신 2×2×2 재측정 결론 (#6 E+Basic+X 우세, R=0.8149 / P=0.7597 / F1=0.7863) — **의견 1 의 "단계별 비교" 시 어떤 Extractor(Basic vs Adaptive)를 기준으로 할지** 프레이밍에 필수. 그렇지 않으면 ablation 결론이 outdated baseline 기반이 됨.
- [ ] **후보 2**: SteinerBackbone 정의 + 기존 `abl_a03_15`, `abl_a03_18` 결과 — G1 에 직접 응답. 다음 보고 요구 사항.
- [ ] **후보 3**: `s06_a01_01`~`s06_b5` Over-smoothing 처방 ablation 결과 (PairNorm 등) — 의견 2 의 일부가 **이미 진행 중** 임을 공유. 중복 지시 방지.
- [ ] **후보 4**: Selector S-V (Neurosymbolic L1 soft routing) 개요 — 의견 3/4 의 SuperNode v2 가 S-V 설계와 동형 구조. 큰 그림 연결.

**공유 시점**: **2026-04-28 (1주 뒤, Q4 답변 반영)** — 15~20분 발표, 중요 지점만  
**공유 형태**: (1) cumulative 단계별 비교표 (Baseline + s04/05 + a03_15/18, Q3 cumulative 정의 준수), (2) SuperNode v2 설계 스케치 (directed + Raw Top-k), (3) s06 over-smoothing chart, (4) Diameter → num_layers 효과 (실행되어 있으면)  
**공유 시 주의점**: 후보 4 (S-V) 는 아직 구현 전 → "계획 단계" 로만 제시. 15~20분 안에 **A/F/C 가 핵심, D/E 는 시간 여유 시**.

---

## 12. 결정 요약 (DECISIONS.md 엔트리 초안)

```markdown
## 2026-04-21 — QCondGAT 상세 ablation 지시 (지도교수 의견 반영)

- **결정**: s04/s05 계열 6개 신규 ablation 트랙 (Proposal A~F, §8) 제안. Selector / Builder 모듈 PLAN 확장 에스컬레이션. int_05 전제를 SuperNode v2 (directed + top-k) 로 재정의 제안 (pending). Phase A 에 "Schema Graph Diameter precompute" 서브태스크 신설 제안 (pending). 교수님께 4개 재확인 질문 (§10) 대기 상태.
- **근거**: 지도교수 2026-04-21 정기 미팅 — `planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md` §2.1~§2.4 + §2.G1/G2
  + 브리핑 범위: QCond 방안 A/B + Over-smoothing 진단 (§1.2)
  + 지지 데이터: `outputs/analysis/gat_bottleneck{,_qcond}/`, s06_b0~b5 ablation (이미 존재)
- **영향 범위 (브리핑 내 직접, §4)**: `src/models/gat_network_v2.py`, `src/models/gat_network.py`, `src/modules/builders/line_graph_builder.py`, `src/modules/selectors/ensemble_selector.py`, s04_xx / s05_xx 재설계
- **영향 범위 (Scope gap — PLAN 파급, §5)**: 루트 PLAN §3.1 int_05 / §4 Phase A Diameter / §4 Phase B T2T / §9 리스크 — **사용자 "PLAN 상관 없음" 입장 존중, 모두 pending-clarification**
- **에스컬레이션**: Selector / Builder / Analyzer / Root 세션 (§9 — 4개 copy-paste 프롬프트 준비됨)
- **추가 필요 분석**: Analyzer 에 Stagewise Raw×Model ablation 표 요청 (§9)
- **다음 브리핑 후보**: 2×2×2 재측정(#6) / SteinerBackbone / s06 over-smoothing 결과 / S-V 개요 (§11)
- **교수님께 후속 질문**: 4건 (§10 — diameter 정의 / top-k 기준 / 단계별 정의 / 보고 형식)
```

---

## 13. 수용 상태

- [x] **adopted** (2026-04-21): Q1~Q4 답변 수렴 + §7 PLAN diff 4건 모두 승인. 브리핑 범위 내 직접 변경(§4) + PLAN 파급(§5/§7) 모두 채택. 의견 2 (diameter=D_max), 의견 4 (top-k Raw Score 우선) 구현 스펙 확정.

**결정 시각**: 2026-04-21 (초안) / 2026-04-21 (승인 및 답변 수렴)

---

## 14. 문서화 체크리스트

- [x] 본 원본 파일 저장: `planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md`
- [x] `EXPERIMENT_PLAN.md` 갱신 — **approved (2026-04-21)**, 루트 세션 에스컬레이션 프롬프트 §9 준비 완료
- [x] `planning/DECISIONS.md` 엔트리 추가 (초기본 + 답변 수렴 후속본)
- [ ] 모듈 PLAN 에스컬레이션 — Selector / Builder 세션 (§9 프롬프트 준비 완료, 사용자 실행)
- [x] 신규 실험 제안서 — Proposal A~F, [`planning/proposals/`](../proposals/) 작성 완료 (2026-04-21)
  - [abl_sel_rawscore_stagewise.md](../proposals/abl_sel_rawscore_stagewise.md) — 우선순위 1
  - [abl_ext_steiner_backbone_report.md](../proposals/abl_ext_steiner_backbone_report.md) — 우선순위 2
  - [abl_sel_diameter_layers.md](../proposals/abl_sel_diameter_layers.md) — 우선순위 3
  - [abl_sel_supernode_directed.md](../proposals/abl_sel_supernode_directed.md) — 우선순위 4
  - [abl_sel_supernode_topk.md](../proposals/abl_sel_supernode_topk.md) — 우선순위 5
  - [abl_bld_t2t_edge.md](../proposals/abl_bld_t2t_edge.md) — 우선순위 6 (시간 여유 시)
- [x] HISTORY/CATALOG/ID_MIGRATION — 실험 실행 시점에 루트 세션이 갱신 (memory rule)
- [ ] 논문 초안 수정 — 의견 1 의 α-ablation 축을 §IV Experiments 에 추가 (approved, 루트 세션 위임)
- [x] Analyzer 요청 — §9 에 등록 (cumulative 정의 Q3 반영)
- [x] **template §1.2 default 갱신**: QC-GAT BCE Loss 학습, Over-smoothing 진단 → "공유된 범위" 승격
- [x] **2026-04-28 발표 준비 계획 확정**: 스토리라인 우선순위 A > F > C > D > E > B (§8)
