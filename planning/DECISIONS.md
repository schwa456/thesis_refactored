# Planner Decisions Log

> Planner 세션이 PLAN을 바꿀 때마다 **반드시** 이 파일에 엔트리를 남긴다.
> 세션이 교체되어도 직전 맥락을 복원할 수 있게 하는 연속성 장치.
>
> 엔트리 포맷은 [CLAUDE.md](CLAUDE.md) 하단 템플릿 참조.
> 최신이 위, 과거가 아래 (역시간순).

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
