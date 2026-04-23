# SteinerBackbone Stagewise 보고 (a03_15 / a03_18) — G1 / G2 대응

> **용도**: 2026-04-28 지도교수 보고 패키지. `planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md` §4 G1 (SteinerBackbone 포함) + Q3 (cumulative 단계별 정의) 대응.
> **근거 제안서**: [planning/proposals/abl_ext_steiner_backbone_report.md](../../planning/proposals/abl_ext_steiner_backbone_report.md)
> **확장 이력**:
> - 2026-04-14: Root 세션 초기 작성 (a03_15/18 final + a03_07/11 partial)
> - **2026-04-21: Analyzer 세션 확장** — Selector top-k (score proxy + binary) 채움 / Fixed PCST 축 대조표 / Δ 분해.
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
- `a03_17` SuperNode + Fixed PCST + XiYan — **Direct 최고 F1=0.6940**

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
5. **후속 실험 함의**:
   - (a) Steiner `backbone_bonus` sweep (0.5 → 0.3 → 0.1) — bridge 포함 강도 조절.
   - (b) XiYan 프롬프트 개선 — bridge column 이 low-score 라고 해서 제거하지 않도록 hint.
   - (c) Steiner + Filter 전 pre-filter (bridge-aware precision booster).
   - 본 분석은 브리핑 범위 내 직접 결론이며, PLAN 파급(E-II Pathfinding + PCST Ensemble)은 planner 세션에서 재검토.

---

## 6. 2026-04-28 발표 슬라이드 참조

- **슬라이드 G1-1**: §2.1 / §2.2 테이블 직접 인용 (cumulative R/P/F1, 4자리).
- **슬라이드 G1-2**: §3.2 Fixed vs Steiner 최종 F1 비교 + §4 Δ 분해로 "Steiner 는 no-filter 단에선 이기지만 XiYan 하류에서 반납" 스토리.
- **백업 슬라이드**: §3.1 no-filter 표 (recall 기여 증명용).

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

- [notebooks/analysis_results/stagewise_qcond_ablation.md](stagewise_qcond_ablation.md) — Scoring 축 (5×3) cumulative
- [planning/proposals/abl_ext_steiner_backbone_report.md](../../planning/proposals/abl_ext_steiner_backbone_report.md) — 본 리포트의 제안서
- [EXPERIMENT_HISTORY.md](../../EXPERIMENT_HISTORY.md) §6-15 — a03 direct chain 원 기록
- [EXPERIMENT_CATALOG.md](../../EXPERIMENT_CATALOG.md) §a03 — direct per-step cluster
