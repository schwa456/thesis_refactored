# SteinerBackbone Stagewise 보고 (a03_15 / a03_18) — G1 대응 원시자료

> **용도**: 2026-04-28 지도교수 보고 패키지. `planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md` §4 G1 (SteinerBackbone 포함) + Q3 (cumulative 단계별 정의) 대응.
> **작성 주체**: Root 세션 (HISTORY 재조직). Selector-only 단계 중 **QCond selector-only 셀은 analyzer 재구성 필요** — §9 Analyzer 에스컬레이션 프롬프트에 포함됨.
> **메트릭 포맷**: R, P, F1 4자리.

---

## 1. 대상 실험

| ID | Selector | Extractor | Filter | Config 링크 |
|----|----------|-----------|--------|-------------|
| `abl_a03_15_qcond_binary_steiner_xiyan` | DirectGAT (QCond, binary θ=0.5) | SteinerBackbone (backbone_bonus=0.5, percentile=80, min=3, max=25) | XiYan (Qwen3-Coder-30B-A3B-Instruct-FP8) | `configs/experiments/abl/a03_direct_per_step/abl_a03_15_qcond_binary_steiner_xiyan.yaml` |
| `abl_a03_18_supernode_binary_steiner_xiyan` | DirectGAT (SuperNode, binary θ=0.5) | SteinerBackbone (동일 파라미터) | XiYan (동일) | `configs/experiments/abl/a03_direct_per_step/abl_a03_18_supernode_binary_steiner_xiyan.yaml` |

Reference anchors (동일 Selector, 다른 Extractor 또는 Filter — stage 분해 좌표 제공):
- `a03_09` SuperNode selector-only (no extractor, no filter)
- `a03_07` QCond + SteinerBackbone (no filter) — **a03_15와 Selector+Extractor 동일**
- `a03_11` SuperNode + SteinerBackbone (no filter) — **a03_18과 Selector+Extractor 동일**
- `a03_13` QCond + Fixed PCST (no filter) — QCond selector 하류 비교용
- `a03_14` QCond + Fixed PCST + XiYan — a03_15와 Extractor 축만 다른 짝
- `a03_17` SuperNode + Fixed PCST + XiYan — a03_18과 Extractor 축만 다른 짝 (**Direct 최고 F1=0.6940**)

---

## 2. Cumulative Stagewise R/P/F1 (Q3 정의 반영)

각 행은 **파이프라인을 stage까지 실행했을 때의 최종 노드셋 대 gold** 로 측정한 cumulative 값.

### 2.1 a03_15 계열 (QCond + SteinerBackbone + XiYan)

| Stage | R | P | F1 | 출처 |
|-------|---|---|-----|------|
| Selector only (QCond binary θ=0.5) | **pending (analyzer)** | **pending** | **pending** | `outputs/.../abl_a03_15_.../output_*.jsonl` 재집계 필요 |
| + Extractor (SteinerBackbone, no filter) | 0.6072 | 0.2154 | 0.3180 | a03_07 (동일 Selector+Extractor, filter 없음) |
| + Filter (XiYan) — **final** | 0.5247 | 0.6824 | 0.5932 | a03_15 metrics.txt |

### 2.2 a03_18 계열 (SuperNode + SteinerBackbone + XiYan)

| Stage | R | P | F1 | 출처 |
|-------|---|---|-----|------|
| Selector only (SuperNode binary θ=0.5) | 0.6261 | 0.1885 | 0.2898 | a03_09 (selector-only 측정치) |
| + Extractor (SteinerBackbone, no filter) | 0.7120 | 0.1798 | 0.2871 | a03_11 (동일 Selector+Extractor, filter 없음) |
| + Filter (XiYan) — **final** | 0.5855 | 0.6871 | 0.6322 | a03_18 metrics.txt |

---

## 3. Fixed PCST vs SteinerBackbone (동일 Selector/Filter 하)

| Selector | Extractor | R | P | F1 |
|----------|-----------|---|---|-----|
| QCond | Fixed PCST + XiYan (a03_14) | 0.5843 | 0.6929 | 0.6340 |
| QCond | SteinerBackbone + XiYan (a03_15) | 0.5247 | 0.6824 | 0.5932 |
| SuperNode | Fixed PCST + XiYan (a03_17, **Direct 최고**) | 0.6761 | 0.7128 | **0.6940** |
| SuperNode | SteinerBackbone + XiYan (a03_18) | 0.5855 | 0.6871 | 0.6322 |

**관찰 (HISTORY §6-15 재인용)**:
- Selector 양쪽 모두에서 Fixed PCST > SteinerBackbone (F1 기준 +4.08%p / +6.18%p)
- SteinerBackbone의 `backbone_bonus=0.5`가 저점수 bridge 강제 포함 → XiYan 단계에서 noise로 잔존
- **Steiner 단계 자체는 R +0.086~0.105** 기여 (a03_07/11 vs 해당 selector-only) — bridge 회수 효과는 실측
- 그러나 이득을 XiYan이 유지 못함 (R 손실이 Fixed 대비 크게 나타남: QCond 0.0825p ↓, SuperNode 0.1265p ↓)

---

## 4. 해석 메모 (advisor 브리핑 포인트)

1. **SteinerBackbone은 bridge 회수에서 +R 이득이 확인됨** (Selector→+Extractor 단계): QCond +0.0≈0.10, SuperNode +0.0860.
2. **XiYan 하류에서 Steiner의 저점수 bridge가 noise로 작용** → Fixed PCST가 최종 F1 우세.
3. **Direct 최고는 `a03_17` (SuperNode + Fixed + XiYan, F1=0.6940)**. Steiner 경로(a03_18)는 −0.0618 F1.
4. **다음 실험 함의**: Steiner에서 backbone_bonus를 낮추거나, XiYan 전에 low-prize bridge를 별도 penalization 하는 접근이 필요. 단, 이 분석은 **브리핑 범위 내 직접 결론**이며 PLAN 파급(E-II Pathfinding + PCST Ensemble)은 planner 세션에서 재검토 중.

---

## 5. Analyzer 요청 큐 (pending)

- [ ] **QCond selector-only (a03_15 앞단) cumulative R/P/F1 재구성**: `outputs/experiments/abl/a03_direct_per_step/abl_a03_15_qcond_binary_steiner_xiyan/output_*.jsonl` 또는 `score_analysis_*.jsonl`에서 selector가 apply_threshold 후 반환한 node set 직접 집계. 저장 위치: 본 파일 §2.1 첫 행에 덮어쓰기, 또는 `notebooks/analysis_results/stagewise_qcond_ablation.md`.
- [ ] (확장) 모든 a03 계열에 대해 stagewise 표 제공 시 base template으로 본 문서 사용.

---

## 6. 메타

- **R/P/F1 포맷**: 4자리 소수 (memory rule).
- **cumulative 정의**: `planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md` §10 Q3.
- **재구성 근거 데이터**: HISTORY §6-15 consolidated table + 각 실험 `metrics.txt`.
- **다음 반영**: 2026-04-28 발표 슬라이드에서 §2 표를 그대로 인용, §3은 "Extractor 축 ablation" 슬라이드로.
