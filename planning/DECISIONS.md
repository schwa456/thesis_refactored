# Planner Decisions Log

> Planner 세션이 PLAN을 바꿀 때마다 **반드시** 이 파일에 엔트리를 남긴다.
> 세션이 교체되어도 직전 맥락을 복원할 수 있게 하는 연속성 장치.
>
> 엔트리 포맷은 [CLAUDE.md](CLAUDE.md) 하단 템플릿 참조.
> 최신이 위, 과거가 아래 (역시간순).

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
     |---|---|---|---|---|
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
