# SteinerBackbone Stagewise 보고 (a03_15 / a03_18) — G1 / G2 대응

> **용도**: 2026-04-28 지도교수 보고 패키지. `planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md` §4 G1 (SteinerBackbone 포함) + Q3 (cumulative 단계별 정의) 대응.
> **근거 제안서**: [planning/proposals/abl_ext_steiner_backbone_report.md](../../../planning/proposals/abl_ext_steiner_backbone_report.md)
> **확장 이력**:
> - 2026-04-14: Root 세션 초기 작성 (a03_15/18 final + a03_07/11 partial)
> - **2026-04-21: Analyzer 세션 확장** — Selector top-k (score proxy + binary) 채움 / Fixed PCST 축 대조표 / Δ 분해.
> - **2026-04-24: Analyzer 세션 확장 (Wave 3 Proposal F)** — GLM era new top (`s04_stagewise_qcond_gat_basic_glm` F1=0.8383) 대비 비교 (§3.3), Wave 3 F 트랙 발표 슬라이드 초안 (§6) 신규.
> **메트릭 포맷**: R, P, F1, 4자리.

---

## 1. 대상 실험

| ID | Selector | Extractor | Filter | Config 링크 |
|----|----------|-----------|--------|-------------|
| `abl_a03_15_qcond_binary_steiner_xiyan` | DirectGAT (QCond, binary θ=0.5) | SteinerBackbone (backbone_bonus=0.5, percentile=80, min=3, max=25) | XiYan (Qwen3-Coder-30B-A3B-Instruct-FP8) | `configs/experiments/abl/a03_direct_per_step/abl_a03_15_qcond_binary_steiner_xiyan.yaml` |
| `abl_a03_18_supernode_binary_steiner_xiyan` | DirectGAT (SuperNode, binary θ=0.5) | SteinerBackbone (동일) | XiYan (동일) | `configs/experiments/abl/a03_direct_per_step/abl_a03_18_supernode_binary_steiner_xiyan.yaml` |

**Reference anchors** (stage 분해 좌표):
- `a03_05` QCond binary selector-only — **a03_15의 Selector cumulative 원점**
- `a03_09` SuperNode binary selector-only — **a03_18의 Selector cumulative 원점**
- `a03_07` QCond + SteinerBackbone (no filter) — a03_15의 +Extractor cumulative
- `a03_11` SuperNode + SteinerBackbone (no filter) — a03_18의 +Extractor cumulative
- `a03_13` QCond + Fixed PCST (no filter) — Extractor 축 대조 (QCond)
- `a03_16` SuperNode + Fixed PCST (no filter) — Extractor 축 대조 (SuperNode)
- `a03_14` QCond + Fixed PCST + XiYan — Extractor 축만 다른 짝 (QCond final)
- `a03_17` SuperNode + Fixed PCST + XiYan — **Direct 최고 F1=0.6940 (vLLM era)**

**Era-Level anchors** (Steiner 외부 상위 anchor, 비교 기준):
- `abl_ens_basic_xiyan` — vLLM era 2×2×2 best: R=0.8149 / P=0.7597 / F1=0.7863 (#6 E+Basic+X)
- `s04_stagewise_qcond_gat_basic` — vLLM era Wave 1.5 best: R=0.8169 / P=0.7605 / F1=0.7877
- `s04_stagewise_qcond_gat_basic_glm` — **GLM era Wave 2 best**: R=0.8438 / P=0.8329 / **F1=0.8383** (2026-04-24 갱신, LLM backbone 교체 효과). 본 리포트에서 Steiner 경로와 비교하는 실질적 최신 기준.

---

## 2. Cumulative Stagewise R/P/F1 (Q3 정의 반영)

각 행은 **파이프라인을 stage까지 실행했을 때의 최종 노드셋 대 gold**로 측정한 cumulative 값.

### 2.1 a03_15 계열 (QCond + SteinerBackbone + XiYan)

| Stage | R | P | F1 | 출처 |
|-------|---|---|-----|------|
| Selector — binary θ=0.5 (pipeline 실제 cut) | 0.4871 | 0.2517 | **0.3319** | a03_05 metrics.txt |
| Selector — top-20 (score proxy) † | 0.5363 | 0.1713 | 0.2534 | a03_15 score_analysis (재집계 2026-04-21) |
| + Extractor (SteinerBackbone, no filter) | 0.6072 | 0.2154 | **0.3180** | a03_07 metrics.txt (동일 Selector+Extractor) |
| + Filter (XiYan) — **final** | 0.5247 | 0.6824 | **0.5932** | a03_15 metrics.txt |

### 2.2 a03_18 계열 (SuperNode + SteinerBackbone + XiYan)

| Stage | R | P | F1 | 출처 |
|-------|---|---|-----|------|
| Selector — binary θ=0.5 (pipeline 실제 cut) | 0.6261 | 0.1885 | **0.2898** | a03_09 metrics.txt |
| Selector — top-20 (score proxy) † | 0.5567 | 0.1806 | 0.2662 | a03_18 score_analysis (재집계 2026-04-21) |
| + Extractor (SteinerBackbone, no filter) | 0.7120 | 0.1798 | **0.2871** | a03_11 metrics.txt |
| + Filter (XiYan) — **final** | 0.5855 | 0.6871 | **0.6322** | a03_18 metrics.txt |

† **top-20 score proxy 주의**: DirectGAT binary selector는 실제로는 sigmoid>0.5 컷을 사용 (분모가 가변). Top-20은 scoring quality 지표일 뿐 pipeline 실제 동작 아님. Cumulative 해석은 **binary 행**을 기준으로.

---

## 3. Extractor 축 대조 — Fixed PCST vs SteinerBackbone (동일 Selector/Filter 하)

### 3.1 No-Filter 대조 (Extractor 순효과)

| Selector | Extractor | R | P | F1 | Selector R → +Ext Δ R |
|----------|-----------|---|---|-----|---|
| QCond binary | (selector only, a03_05) | 0.4871 | 0.2517 | 0.3319 | — |
| QCond binary | Fixed PCST (a03_13) | 0.6748 | 0.1979 | 0.3060 | **+0.1877** |
| QCond binary | SteinerBackbone (a03_07) | 0.6072 | 0.2154 | 0.3180 | **+0.1201** |
| SuperNode binary | (selector only, a03_09) | 0.6261 | 0.1885 | 0.2898 | — |
| SuperNode binary | Fixed PCST (a03_16) | 0.7982 | 0.1587 | 0.2648 | **+0.1721** |
| SuperNode binary | SteinerBackbone (a03_11) | 0.7120 | 0.1798 | 0.2871 | **+0.0859** |

**관찰**:
- Fixed PCST 가 SteinerBackbone 보다 Recall 증가폭이 큼 (QCond +0.0676, SuperNode +0.0862).
- SteinerBackbone 은 Precision 소폭 우위 (QCond 0.2154 vs Fixed 0.1979).
- F1은 SteinerBackbone이 Fixed PCST 보다 근소 우위 (QCond +0.0120, SuperNode +0.0223) — **no-filter stage 에서만** Steiner가 이김.

### 3.2 With-Filter 대조 (Extractor + Filter 최종)

| Selector | Extractor | R | P | F1 | Δ F1 vs no-filter |
|----------|-----------|---|---|-----|---|
| QCond binary | Fixed PCST + XiYan (a03_14) | 0.5843 | 0.6929 | **0.6340** | +0.3280 (Fixed) |
| QCond binary | SteinerBackbone + XiYan (a03_15) | 0.5247 | 0.6824 | **0.5932** | +0.2752 (Steiner) |
| SuperNode binary | Fixed PCST + XiYan (a03_17) ★ | 0.6761 | 0.7128 | **0.6940** | +0.4292 (Fixed) |
| SuperNode binary | SteinerBackbone + XiYan (a03_18) | 0.5855 | 0.6871 | **0.6322** | +0.3451 (Steiner) |

★ Direct 계열 최고 F1.

**핵심 관찰 (발표 포인트)**:
- Filter 적용 후에는 Fixed PCST > SteinerBackbone (QCond Δ=+0.0408, SuperNode Δ=+0.0618).
- SteinerBackbone 이 no-filter recall 을 +0.0859~0.1201 올리지만, XiYan 이 해당 이득을 **유지하지 못하고 오히려 −0.0825~−0.1265 반납**.
- 원인 분석: `backbone_bonus=0.5` 로 강제 포함된 저점수 bridge column 이 filter 단계에서 noise 로 작용. XiYan LLM 이 "의미적으로 불확실한 bridge" 를 과도하게 제거하면서 gold 도 함께 탈락.

### 3.3 상위 anchor 와의 비교 (Steiner 경로의 한계 정량화)

Direct 계열 최고 `a03_17` (F1=0.6940) 및 Steiner 경로 `a03_15/18` (F1=0.5932/0.6322) 은 모두 DirectGAT binary selector 위에 구축됐다. 이들이 **Ensemble (α=0.85) + Basic PCST + XiYan** 계열 anchor 와 얼마나 차이가 나는지 확인.

| 경로 | Selector | Extractor | Filter LLM | R | P | F1 | vs vLLM 2×2×2 best | vs GLM Wave 2 best |
|------|----------|-----------|------------|---|---|-----|--------------------|---------------------|
| **Steiner (a03_15)** | QCond binary | SteinerBackbone | XiYan (vLLM) | 0.5247 | 0.6824 | 0.5932 | −0.1931 | −0.2451 |
| **Steiner (a03_18)** | SuperNode binary | SteinerBackbone | XiYan (vLLM) | 0.5855 | 0.6871 | 0.6322 | −0.1541 | −0.2061 |
| **Direct best (a03_17)** | SuperNode binary | Fixed PCST | XiYan (vLLM) | 0.6761 | 0.7128 | 0.6940 | −0.0923 | −0.1443 |
| **vLLM 2×2×2 best (abl_ens_basic_xiyan)** | Ensemble α=0.85 | Basic PCST | XiYan (vLLM) | 0.8149 | 0.7597 | **0.7863** | — (기준) | −0.0520 |
| **GLM Wave 2 best (qcond_gat_basic_glm)** | Ensemble α=0.85 (QCondGAT) | Basic PCST | XiYan (GLM-4.7) | 0.8438 | 0.8329 | **0.8383** | +0.0520 | — (기준) |

**관찰**:
- Steiner 경로의 F1 0.5932~0.6322 는 GLM era new top 대비 **ΔF1=-0.20~-0.25** 로 절대적 격차가 큼. 이 격차는 Steiner 자체 결함이 아니라 (a) Selector 가 DirectGAT binary 로 제한, (b) Filter LLM 이 vLLM era Qwen 로 제한, 두 요인이 합쳐진 결과.
- Direct best (a03_17) 도 GLM era best 대비 ΔF1=-0.1443 — **Selector 축의 상한** (DirectGAT binary 의 F1 천장) 을 보여줌.
- **Steiner 고유 기여 격차**: a03_17 (Fixed PCST) 대비 Steiner (a03_18) 는 ΔF1=-0.0618. 같은 Selector/Filter 하에서 Steiner backbone 이 **no-filter 단계의 +R 이득 (+0.0859) 을 XiYan 후 반납 (-0.1265)** 하는 패턴.
- **발표 F 트랙의 서사적 가치**: "Steiner 는 recall 측에서 이론적 우위가 있지만, 현 Filter 의 구현 특성 (낮은 score bridge 과도 제거) 이 이를 상쇄" — 향후 개선 여지로 연결.

### 3.4 GLM era 재실행 가능성 (참고, 현재 계획 없음)

Steiner 경로를 GLM era 로 재실행하면 어느 정도 갱신될지 가늠:
- Sanity anchor (α=0, Basic PCST, XiYan) ΔF1 ≈ **−0.0099** (노이즈, Steiner 에도 유사 기대)
- New anchor (α=0.85 ensemble, Basic PCST, XiYan) ΔF1 ≈ **+0.0506** (synergy 발현)
- Steiner 는 DirectGAT binary selector → α=0 pure GAT 계열 — sanity anchor 와 유사 ΔF1 ≈ −0.01~0 예상. 즉 **GLM era 재실행해도 a03_15/18 F1 은 0.59~0.63 ± 0.01 범위** 에서 크게 벗어나지 않을 것. **현재 우선순위 낮음**.
- 실행 시 필요 리소스: 2 cells × ~50min = 100min, ~₩1,500.

---

## 4. Δ 분해 표 (anchor 기준)

### 4.1 QCond (a03_13 → a03_07 → a03_15)

| 전환 | Stage Δ R | Stage Δ P | Stage Δ F1 | 해석 |
|---|---|---|---|---|
| Selector-only (a03_05) → + Fixed PCST (a03_13) | +0.1877 | −0.0538 | −0.0259 | PCST가 noise 포함하며 P 희석 |
| Selector-only (a03_05) → + SteinerBackbone (a03_07) | +0.1201 | −0.0363 | −0.0139 | Steiner는 P 덜 희석, F1 유지 |
| + SteinerBackbone (a03_07) → + XiYan (a03_15) | −0.0825 | +0.4670 | +0.2752 | XiYan이 bridge 제거하며 gold도 탈락 |

### 4.2 SuperNode (a03_09 → a03_11 → a03_18)

| 전환 | Δ R | Δ P | Δ F1 | 해석 |
|---|---|---|---|---|
| Selector-only (a03_09) → + Fixed PCST (a03_16) | +0.1721 | −0.0298 | −0.0250 | |
| Selector-only (a03_09) → + SteinerBackbone (a03_11) | +0.0859 | −0.0087 | −0.0027 | Steiner: R +0.086, P 거의 유지 → no-filter에서 최강 |
| + SteinerBackbone (a03_11) → + XiYan (a03_18) | −0.1265 | +0.5073 | +0.3451 | SuperNode Steiner도 XiYan에서 R 크게 손실 |

---

## 5. 해석 메모 (advisor 브리핑 포인트)

1. **SteinerBackbone 은 bridge 회수에서 +R 이득이 확인됨**: Selector→+Extractor 단계에서 QCond +0.1201, SuperNode +0.0859.
2. **XiYan 하류에서 Steiner 의 저점수 bridge 가 noise로 작용** → Fixed PCST 가 최종 F1 우세.
3. **Direct 최고는 `a03_17` (SuperNode + Fixed + XiYan, F1=0.6940)**. Steiner 경로 (a03_18) 는 **−0.0618 F1**.
4. **Stage별 방향성**: Steiner 는 "no-filter F1" 최적, Fixed PCST 는 "with-filter F1" 최적. Filter 와 Extractor 간 **상호작용이 강함** — 단독 평가 위험.
5. **상위 anchor 와의 격차 (Wave 3 Proposal F 신규 관점, §3.3)**: Steiner 경로 F1 0.5932~0.6322 는 Wave 2 GLM era best F1=0.8383 대비 ΔF1=-0.21~-0.25. 격차의 **대부분은 Selector 축 (DirectGAT binary 천장)**, 잔여분은 Filter LLM era. Steiner backbone 자체는 Fixed PCST 대비 ΔF1=-0.06 정도의 기여.
6. **후속 실험 함의**:
   - (a) Steiner `backbone_bonus` sweep (0.5 → 0.3 → 0.1) — bridge 포함 강도 조절.
   - (b) XiYan 프롬프트 개선 — bridge column 이 low-score 라고 해서 제거하지 않도록 hint.
   - (c) Steiner + Filter 전 pre-filter (bridge-aware precision booster).
   - (d) **Selector 축 전환 후 Steiner 재평가** — Ensemble α=0.85 + SteinerBackbone + XiYan 조합을 1 cell 추가 실행 (Wave 2.5 또는 post-deadline). 현재는 DirectGAT binary 에서만 측정되어 진짜 Steiner 의 가치가 은폐됨.
   - 본 분석은 브리핑 범위 내 직접 결론이며, PLAN 파급(E-II Pathfinding + PCST Ensemble)은 planner 세션에서 재검토.

---

## 6. 2026-04-28 발표 슬라이드 초안 — Wave 3 F 트랙 (A > F > C 순)

발표 순서: **A (Wave 1.5 2×2×2 closed) → F (Steiner backbone stagewise) → C (Diameter layers H1 검증)**. F 트랙은 2~3 슬라이드 권장.

### 6.1 Slide F-1: Steiner 경로 Stagewise cumulative

- **Title**: "SteinerBackbone vs Fixed PCST — Stagewise Cumulative (DirectGAT selector)"
- **Body**: §2.1 + §2.2 표 (`a03_15`/`a03_18` cumulative). 4자리 R/P/F1. 각 stage 로 노드수가 어떻게 변하는지 1줄 요약 추가 ("Selector binary → no-filter extractor → XiYan final").
- **Takeaway**: "Steiner 는 no-filter 단계에서 recall +0.086~0.120 올리지만 XiYan 통과 후 그 이득을 반납".

### 6.2 Slide F-2: Extractor 축 대조 (Fixed vs Steiner, with filter)

- **Title**: "Filter 하류 최종 F1 — Fixed PCST 가 Steiner 우세"
- **Body**: §3.2 표 직접 인용 (a03_14 vs a03_15, a03_17 vs a03_18). Direct 최고 = a03_17 F1=0.6940 강조. Steiner 는 -0.04~-0.06.
- **추가 insight** (§3.2 원인 분석): "backbone_bonus=0.5 저점수 bridge 가 XiYan 에서 noise → gold 도 함께 제거".
- **Takeaway**: "Steiner 의 이론적 우위 (recall) 는 현 XiYan 구현과 충돌 — 개선 방향 (§5 #6)".

### 6.3 Slide F-3 (optional, backup): 상위 anchor 와의 비교

- **Title**: "Direct 계열 의 천장 vs Ensemble 계열 best"
- **Body**: §3.3 표 (a03_17 F1=0.6940 ↔ Wave 2 GLM best F1=0.8383, ΔF1=-0.1443). Selector 축이 Steiner 성능을 제한한다는 증거.
- **Takeaway**: "Steiner 재평가는 Selector 축을 Ensemble 로 바꾼 뒤 별도 실험 필요 (post-deadline 큐)".

### 6.4 발표 플로우 연결 포인트 (A → F → C)

| 연결 | 스토리 흐름 |
|------|------------|
| A → F | A 에서 2×2×2 best=F1=0.7863 (Ensemble + Basic PCST + XiYan) 를 선언. F 에서 "Extractor 축 을 Steiner 로 바꾸면 어떻게 될까?" 라는 follow-up 질문에 답. 결론: Steiner 는 이론적 이득 있지만 Filter 와의 상호작용으로 F1 저하. |
| F → C | F 에서 "Extractor 축은 Basic PCST 가 낫다" 결론 후, C 에서 "Selector 축 (GAT num_layers) 조정으로 얻을 수 있는 추가 이득은?" 으로 이어짐. C 에서 nl=6 peak (F1=0.5824) + H2 per-DB dynamic 여지 (§2.3 oracle ΔF1 +0.024) pitch. |

---

## 7. 메타

- **R/P/F1 포맷**: 4자리 소수 (memory rule).
- **Cumulative 정의**: `planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md` §10 Q3 — Selector output → Extractor input → Filter input 순 누적.
- **재집계 데이터**:
  - `score_analysis_abl_a03_15_qcond_binary_steiner_xiyan.jsonl` (query 수 1533 — 1 query missing gold)
  - `score_analysis_abl_a03_18_supernode_binary_steiner_xiyan.jsonl` (query 수 1534)
  - 나머지 cell: 기존 `metrics.txt` 직측.
- **집계 스크립트**: in-session (`/tmp/compute_stein.py` 패턴, 반복 필요 시 `src/analysis/selector_score_analysis.py` 에 통합).

---

## 8. 관련 리포트

- [notebooks/analysis_results/stagewise_qcond_ablation.md](../05_ablation_waves/stagewise_qcond_ablation.md) — Scoring 축 (5×3) cumulative
- [notebooks/analysis_results/diameter_layers_sweep.md](../02_v1_v5_dsn_mitigation/diameter_layers_sweep.md) — Wave 3 Proposal C 분석 (H1 검증 + per-DB D_max)
- [notebooks/analysis_results/selector_gold_score_discrimination.md](../02_v1_v5_dsn_mitigation/selector_gold_score_discrimination.md) — Selector 축 분별력 (DirectGAT binary 한계 진단)
- [planning/proposals/abl_ext_steiner_backbone_report.md](../../../planning/proposals/abl_ext_steiner_backbone_report.md) — 본 리포트의 제안서
- [EXPERIMENT_HISTORY.md](../../../EXPERIMENT_HISTORY.md) §6-15 — a03 direct chain 원 기록
- [EXPERIMENT_CATALOG.md](../../../EXPERIMENT_CATALOG.md) §a03 — direct per-step cluster

---

## 9. Changelog

- **2026-04-14**: Root 세션 초기 작성.
- **2026-04-21**: Analyzer 세션 — Selector cumulative + Δ 분해 보강.
- **2026-04-24**: Analyzer 세션 (Wave 3 Proposal F) — §1 GLM era anchors 추가, §3.3 상위 anchor 대비 격차 정량화 (Steiner vs GLM Wave 2 best: ΔF1=-0.20~-0.25), §3.4 GLM era 재실행 가치 추정 (예상 ΔF1 ±0.01), §5 #5 #6d 항목 추가, §6 Wave 3 F 트랙 발표 슬라이드 F-1/F-2/F-3 초안, A→F→C 연결 포인트.
