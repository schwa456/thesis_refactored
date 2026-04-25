# Planner Decisions Log

> Planner 세션이 PLAN을 바꿀 때마다 **반드시** 이 파일에 엔트리를 남긴다.
> 세션이 교체되어도 직전 맥락을 복원할 수 있게 하는 연속성 장치.
>
> 엔트리 포맷은 [CLAUDE.md](CLAUDE.md) 하단 템플릿 참조.
> 최신이 위, 과거가 아래 (역시간순).

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
