# Stage-wise QCond Ablation — 지도교수 2026-04-21 의견 1 대응

**분석일**: 2026-04-21
**의도**: "GAT 순 기여도는 +214 gold(2.1%)로 미미" 라는 지적에 대한 단계별 분해
**대상 실험군**: `s03/s04/s05/abl_a01/abl_a03` + baselines
**지표 표기**: Recall / Precision / F1 (소수점 4자리)
**메트릭 출처**: `outputs/experiments/*/metrics.txt`, `logs/*/selector/score_analysis_*.jsonl`

---

## 0. TL;DR

지도교수 지적을 **단계별로 재검증**한 결과:

1. **Selector stage (top-20 gold hit)**: Cosine 0.6727 → Ensemble α=0.85 **0.6916 (+0.0189)** → QCond α=0.85 **0.6968 (+0.0052)**. Ensemble의 GAT 기여는 ~2%p에 그치고, QCond/SuperNode 추가 이득은 ~0.5%p.
2. **Raw GAT score (α=0)는 Cosine보다 나쁘다**: top-20 0.5645 (QCond), 0.5968 (SuperNode) — 순수 GAT는 PLM 코사인 대비 −0.08~−0.11. "GAT가 코사인을 거의 **대체**할 수준이 아니라 **보조**한다"는 관찰과 정합.
3. **End-to-end F1**은 **Ensemble α=0.85 + Basic PCST + XiYan (abl_a01_06) = 0.7863**이 최강. QCond 기반 s04_01 (0.7030), SuperNode 기반 s04_03 (0.6959)은 Ensemble baseline에 못 미치나, **Extractor가 다르므로 직접 비교 불가** (s04는 ComponentAwareProductCost).
4. **Filter가 없을 때 Raw GAT는 더 약함**: `abl_a03_13` (QCond binary+PCST, no filter) F1=0.3060 vs `s03_a01_01` (Ens+Basic, no filter) F1=0.2281 — 노드 수(P)가 다르므로 F1 역전은 selector threshold 차이 탓. **Recall 기준**으로는 Ensemble이 월등 (0.9679 vs 0.6748).

→ **지도교수 지적 검증 완료**: GAT 단독 기여는 selector top-k hit 기준 **+0.019** 수준, QCond로 추가된 이득은 **+0.005** 수준. 현재 설계에서 GAT의 역할은 "대체재"가 아닌 "Cosine의 rerank 보조자".

---

## 1. Stage-wise Decomposition

### 1-A. Raw Score baselines (GAT 없음 or α=0)

| Group | Config | Selector | Extractor | Filter | top-20 hit | Post-Selector (R/P/F1) | Post-PCST (R/P/F1) | Post-Filter (R/P/F1) |
|---|---|---|---|---|---|---|---|---|
| Cosine-only | `preliminary_vector_only` | Cos top-k | — (no PCST) | — | **0.6727** | — | — | 0.6825 / 0.7470 / 0.7133 |
| Cosine+PCST | `baseline_g_retriever` | Cos top-k | Basic PCST | — | **0.6727** | — | 0.7577 / 0.7866 / 0.7719 | — |
| Cosine+Basic+XiYan | `abl_a01_05` | Cos top-k | Basic PCST | XiYan | **0.6747** | — | pending† | **0.7987 / 0.7694 / 0.7838** |
| Cosine+Adaptive+XiYan | `abl_a01_07` | Cos top-k | Adaptive PCST | XiYan | 0.6747 | — | pending† | 0.5835 / 0.7829 / 0.6686 |
| Raw QCond α=0 | `s04_04_qcond_a0_xiyan` | Proj score (pure GAT) | CA-Product PCST | XiYan | **0.5645** | — | pending† | 0.5015 / 0.7065 / 0.5866 |
| Raw Super α=0 | `s04_05_supernode_a0_xiyan` | Proj score (pure GAT) | CA-Product PCST | XiYan | **0.5968** | — | pending† | 0.5237 / 0.7155 / 0.6048 |

† Post-PCST without filter 셀은 해당 config가 실행되지 않음 — analyzer queue에 재집계 필요 (`output_*.jsonl`에서 PCST 결과 직접 집계 가능). `s04_06/s04_07` no-filter 변형 예약됨.

**해석**: Raw GAT (α=0) 스코어는 selector top-20 hit에서 Cosine보다 **0.08~0.11p 낮다**. 이는 projector head가 순수 쿼리-노드 유사도를 추정하기에 충분히 학습되지 않았음을 시사. 지도교수 지적의 근본 원인 — GAT가 "대체" 역할을 할 만큼 강하지 않다.

### 1-B. GAT-blend (α>0, Ensemble)

| Group | Config | Selector α | Extractor | Filter | top-20 hit | Post-PCST (R/P/F1) | Post-Filter (R/P/F1) |
|---|---|---|---|---|---|---|---|
| Standard GAT (L=3) | `s03_a01_01_ensemble_basic` | 0.85 | Basic PCST | — | **0.6916** | 0.9679 / 0.1293 / 0.2281 | — |
| Standard GAT (L=3) | `abl_a01_06_ens_basic_xiyan` | 0.85 | Basic PCST | XiYan | **0.6916** | pending† | **0.8149 / 0.7597 / 0.7863** |
| QCond GAT | `s04_01_qcond_a085_xiyan` | 0.85 | CA-Product PCST | XiYan | **0.6968** | pending† | 0.6236 / 0.8056 / 0.7031 |
| SuperNode GAT | `s04_02_supernode_a070_xiyan` | 0.70 | CA-Product PCST | XiYan | **0.6994** | pending† | 0.6089 / 0.7922 / 0.6886 |
| SuperNode GAT | `s04_03_supernode_a085_xiyan` | 0.85 | CA-Product PCST | XiYan | **0.6995** | pending† | 0.6154 / 0.8005 / 0.6959 |

**GAT 기여 정량 (top-20 hit Δ)**:

| 비교 | Δ top-20 hit |
|---|---|
| Cosine → Ensemble α=0.85 (Standard GAT) | **+0.0189** (+2.8%) |
| Ensemble α=0.85 → QCond α=0.85 | **+0.0052** (+0.75%) |
| Ensemble α=0.85 → SuperNode α=0.85 | **+0.0079** (+1.1%) |
| Cosine → QCond α=0.85 | +0.0241 (+3.6%) |
| Cosine → SuperNode α=0.85 | +0.0268 (+4.0%) |

지도교수가 언급한 "+214 gold (2.1%)"는 **Ensemble Standard GAT** 기준으로 정합. QCond/SuperNode는 여기서 추가로 0.5~1.1%p 수준의 미세 gain.

### 1-C. Direct GAT classifier chain (α = N/A, pure GAT score as selector output)

| Config | Selector 모드 | Extractor | Filter | Selector-only (R/P/F1) | Post-PCST (R/P/F1) | Post-Filter (R/P/F1) |
|---|---|---|---|---|---|---|
| `abl_a03_01_qcond_selector_only` | No threshold (keep all) | — | — | 0.9968 / 0.1173 / 0.2096 | — | — |
| `abl_a03_03_supernode_selector_only` | No threshold (keep all) | — | — | 0.9968 / 0.1173 / 0.2096 | — | — |
| `abl_a03_05_qcond_binary_selector_only` | Sigmoid>0.5 binary | — | — | 0.4871 / 0.2517 / 0.3319 | — | — |
| `abl_a03_09_supernode_binary_selector_only` | Sigmoid>0.5 binary | — | — | 0.6261 / 0.1885 / 0.2897 | — | — |
| `abl_a03_13_qcond_binary_fixed` | Sigmoid binary | Fixed PCST | — | — | 0.6748 / 0.1979 / 0.3060 | — |
| `abl_a03_16_supernode_binary_fixed` | Sigmoid binary | Fixed PCST | — | — | 0.7982 / 0.1587 / 0.2648 | — |
| `abl_a03_14_qcond_binary_fixed_xiyan` | Sigmoid binary | Fixed PCST | XiYan | — | — | 0.5843 / 0.6929 / 0.6340 |
| `abl_a03_17_supernode_binary_fixed_xiyan` | Sigmoid binary | Fixed PCST | XiYan | — | — | 0.6761 / 0.7128 / 0.6940 |
| `s05_a01_01_qcond_direct_xiyan` | DirectGAT top-k | CA-Product PCST | XiYan | — | — | 0.4384 / 0.6578 / 0.5261 |
| `s05_a01_02_supernode_direct_xiyan` | DirectGAT top-k | CA-Product PCST | XiYan | — | — | 0.4369 / 0.6553 / 0.5243 |

**해석**:
- Direct GAT (threshold 없이)는 recall=0.9968 (거의 전부 유지)인데 이것은 classifier가 positive 경계를 거의 긋지 못한다는 신호. `apply_threshold`를 켜면 (binary) QCond 0.4871 / SuperNode 0.6261로 크게 떨어짐 → **QCond classifier의 positive-class 학습이 특히 약하다**.
- Post-Filter 단에서 SuperNode binary (`abl_a03_17` F1=0.6940)는 QCond binary (`abl_a03_14` F1=0.6340)를 **+0.060** 능가. 구조적으로 super-node가 classifier로서는 더 calibrate 되었다.
- s05의 DirectGAT top-k (0.5241~0.5261) < abl_a03 binary (0.6340~0.6940): top-k로 자르는 것보다 **binary threshold가 더 나은 설정**이다 (recall-precision 교환점이 threshold 쪽에서 더 좋음).

---

## 2. 지도교수 의견 1 대응 — "GAT 기여도 2.1%" 재해석

### 2-1. Selector stage에서의 GAT 기여 (top-20 hit 기준)

| 시나리오 | top-20 hit | Δ vs Cosine | Δ vs Ens α=0.85 |
|---|---|---|---|
| Cosine | 0.6727 | — | −0.0189 |
| **Ens α=0.85 (Standard GAT)** | 0.6916 | **+0.0189** | 0 |
| Ens α=0.85 (QCond) | 0.6968 | +0.0241 | +0.0052 |
| Ens α=0.85 (SuperNode) | 0.6995 | +0.0268 | +0.0079 |
| Raw QCond α=0 | 0.5645 | **−0.1082** | — |
| Raw Super α=0 | 0.5968 | **−0.0759** | — |

**결론**:
- 지도교수 관찰대로 Ensemble standard-GAT가 Cosine 대비 **+0.019** 기여 (약 2.1%는 full-dataset gold-node-count 기준).
- QCond/SuperNode로 구조를 업그레이드해도 selector hit 증가는 **+0.005~0.008**에 그침.
- **Raw GAT 단독은 Cosine보다 나쁨** → GAT는 대체가 아닌 보조 역할. α=0.85(GAT weight 15%)가 경험적으로 맞음.

### 2-2. Bridge-Table / 비대칭 원인 진단

| 징후 | 근거 |
|---|---|
| Raw GAT 스코어가 PLM cosine보다 낮음 | s04_04, s04_05 top-20 hit < s04_01~03 |
| QCond binary classifier는 positive 예측을 적게 함 | abl_a03_05 R=0.4871 (QCond) vs abl_a03_09 R=0.6261 (Super) |
| Super-node가 QCond보다 classifier로서 더 강함 | `abl_a03_17` F1=0.6940 > `abl_a03_14` F1=0.6340 |
| Super-node가 selector top-20에서도 소폭 우위 | s04_03 (0.6995) > s04_01 (0.6968) |

→ **의심 지점**: QCond 변형이 쿼리-노드 간 sharp alignment는 올려도, **bridge table 같은 쿼리-어휘에 없는 노드**를 잡는 능력은 SuperNode가 더 나음. 이는 의견 1의 "Bridge Table Awareness" 설계 근거와 정합.

### 2-3. Downstream (F1) 복기

| Pipeline | F1 |
|---|---|
| Cos + Basic PCST + XiYan (`abl_a01_05`) | 0.7838 |
| **Ens(Standard) + Basic PCST + XiYan (`abl_a01_06`)** | **0.7863** |
| QCond α=0.85 + CA-Product + XiYan (`s04_01`) | 0.7031 |
| SuperNode α=0.85 + CA-Product + XiYan (`s04_03`) | 0.6959 |

**직접 비교 불가 주의**: `abl_a01_06`의 extractor는 **Basic PCST (node_threshold=0.1)**인 반면 s04는 **ComponentAwareProductCost**. Extractor 교체 영향 > Selector 교체 영향일 가능성 → QCond/SuperNode의 end-to-end 이득을 논하려면 **s04 계열에 Basic PCST 버전**이 필요.

→ **후속 실험 요청 (planner)**: `s04_0{1,2,3}`을 `PCSTExtractor (base_cost=0.05, node_threshold=0.0)` + XiYan으로 다시 돌려 abl_a01_06와 공정 비교. 현재 CA-Product 설정은 s04에서 recall을 상대적으로 깎고 있음 (0.6236 vs 0.8149).

---

## 3. Caveats

1. **Extractor 불일치**: s04 계열 = ComponentAwareProductCostPCSTExtractor, abl_a01_06 = Basic PCSTExtractor. 직접 F1 비교 시 selector 기여를 과소/과대 평가할 수 있음.
2. **Post-PCST no-filter 셀 누락**: `s04_06/s04_07` 미실행. selector/filter 기여를 깨끗이 분리하려면 이 두 config가 필요.
3. **top-20 hit의 의미**: `score_analysis_*.jsonl` 기반의 **pre-PCST gold 포함율**. 실제 selector가 top-k로 자르는지 여부와 무관하게 scorer 품질 지표로만 해석.
4. **Direct GAT no-threshold**: recall 0.9968은 "전체 노드를 유지"의 근사 — 실질적 selector 기능 없음. 비교는 binary 변형(`abl_a03_05/09`)으로만 의미 있음.
5. **"2.1% of gold nodes"와 "top-20 hit +0.019"의 차이**: 지도교수 관찰 "+214 gold nodes"는 10,144개 gold 중 절대값. top-20 hit은 각 쿼리 gold 집합 내 회수율 평균. 두 수치는 서로 다른 단위지만 같은 방향.

---

## 4. Data Sources

### Metrics
- `outputs/experiments/s04_gat_qcond_projector/s04_0{1..5}_*/metrics.txt`
- `outputs/experiments/s05_gat_direct/a01_full_pipeline/s05_a01_0{1,2}_*/metrics.txt`
- `outputs/experiments/abl/a01_2x2x2_selector_extractor_filter/abl_a01_0{5,6,7}_*/metrics.txt`
- `outputs/experiments/abl/a03_direct_per_step/abl_a03_{01,03,05,09,13,14,16,17}_*/metrics.txt`
- `outputs/experiments/s03_gat_ensemble/a01_basic_pcst/s03_a01_01_ensemble_basic/metrics.txt`
- `outputs/baselines/{preliminary_vector_only,baseline_g_retriever}/metrics.txt`

### Selector top-20 hit rates
`logs/<exp>/selector/score_analysis_*.jsonl` 에서 top-K gold inclusion ratio 계산.
(Computed in-session 2026-04-21; raw helper at `src/analysis/selector_score_analysis.py`.)

### Pending post-PCST no-filter cells
`output_*.jsonl` (stage별 노드셋 포함) 에서 PCST 단계 직후 노드셋을 gold와 비교하여 재집계. s04_01/02/03, abl_a01_05/06/07 기준 analyzer queue 처리 필요.

---

## 5. Next Analyzer Tasks

1. [HIGH] s04 계열 post-PCST (no-filter) R/P/F1을 `output_*.jsonl`에서 재집계 → selector vs filter 분리.
2. [MID] `abl_a03_0{5,9}`의 selector-only 결과를 per-difficulty (simple/moderate/challenging)로 분해 → QCond가 어떤 난이도에서 특히 약한지 확인.
3. [MID] s04 extractor 변경 안 하고 selector만 재배치한 A/B 실험 설계를 planner에 요청 (Basic PCST + QCond/SuperNode).
4. [LOW] QCond α 스윕 (0.7, 0.5) — 현재 α=0.85 고정인데 QCond classifier가 raw에서 약하므로 α를 더 높이면 어떤지 확인.
