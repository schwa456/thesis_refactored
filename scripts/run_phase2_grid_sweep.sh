#!/bin/bash
# Phase 2 Grid Sweep (Wave 5 Partial Reopen, DECISIONS 2026-05-16) — 25 cells (5x5)
#
# Spec:
#   theta ∈ {0.1, 0.125, 0.15, 0.175, 0.2}  (anchor-band fine-grained)
#   K ∈ {15, 20, 30, 40, 70}                 (F1 sub-noise, TCR/TOR mechanism spread)
#   Stack: QCondGAT + MSTPCSTUnion + XiYanFilter (GLM-4.7) + LLMSQLGenerator
#
# Anchor 정합: P2_02 (theta=0.1, K=20) = c01_01 deterministic 일치 검증 cell (F1=0.8664).
#
# Success criterion:
#   (a) plateau breadth — anchor-band 안 F1 spread + EX spread 5x5 heatmap (axis #11 evidence)
#   (b) R 갱신 lever — 어떤 (theta, K) cell 이 anchor F1=0.8664 초과 시 closure narrative 재고
#
# GPU 0+1 split (8-conc total = GPU 0 x 4 + GPU 1 x 4, GPU 2/3 절대 금지)
# Failure-tolerant (|| true 로 다른 cell 진행)
# ETA ~4-5h wall (V5 inference 7-conc 정합 기반 추정), cost ~$15-30 GLM API
#
# Launch:
#   nohup bash scripts/run_phase2_grid_sweep.sh > logs/phase2_grid_main.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

mkdir -p logs/phase2_grid

echo "=== Phase 2 Grid Sweep START $(date -Iseconds) ==="
echo "  5x5 = 25 cells (theta x K)"
echo "  GPU 0 (cells 1-13) + GPU 1 (cells 14-25), 4-conc per GPU = 8-conc total"
echo "  ETA ~4-5h wall"

run_one() {
  local cell=$1
  local gpu=$2
  local cfg="experiments/abl/c03_phase2_grid/${cell}"
  local log="logs/phase2_grid/${cell}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] start ${cell}"
  CUDA_VISIBLE_DEVICES=${gpu} conda run -n base python -u src/main.py \
    --config "${cfg}" > "${log}" 2>&1 || echo "[WARN] ${cell} non-zero exit"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] end ${cell}"
}

# GPU 0 cells: p2_01 ~ p2_13 (13 cells)
declare -a GPU0_CELLS=(
  p2_01_theta_0.1_topk_15
  p2_02_theta_0.1_topk_20
  p2_03_theta_0.1_topk_30
  p2_04_theta_0.1_topk_40
  p2_05_theta_0.1_topk_70
  p2_06_theta_0.125_topk_15
  p2_07_theta_0.125_topk_20
  p2_08_theta_0.125_topk_30
  p2_09_theta_0.125_topk_40
  p2_10_theta_0.125_topk_70
  p2_11_theta_0.15_topk_15
  p2_12_theta_0.15_topk_20
  p2_13_theta_0.15_topk_30
)

# GPU 1 cells: p2_14 ~ p2_25 (12 cells)
declare -a GPU1_CELLS=(
  p2_14_theta_0.15_topk_40
  p2_15_theta_0.15_topk_70
  p2_16_theta_0.175_topk_15
  p2_17_theta_0.175_topk_20
  p2_18_theta_0.175_topk_30
  p2_19_theta_0.175_topk_40
  p2_20_theta_0.175_topk_70
  p2_21_theta_0.2_topk_15
  p2_22_theta_0.2_topk_20
  p2_23_theta_0.2_topk_30
  p2_24_theta_0.2_topk_40
  p2_25_theta_0.2_topk_70
)

# 4-conc per GPU rounds (사용자 5/16 명시 8-conc total)
run_gpu_concurrent() {
  local gpu=$1
  shift
  local cells=("$@")
  local total=${#cells[@]}
  local conc=4
  local round=0
  for ((i=0; i<total; i+=conc)); do
    round=$((round+1))
    local end_idx=$((i+conc))
    [ "$end_idx" -gt "$total" ] && end_idx=$total
    echo "[GPU ${gpu}] Round ${round}: cells $((i+1))-${end_idx} of ${total}"
    PIDS=()
    for ((j=0; j<conc && (i+j)<total; j++)); do
      run_one "${cells[$((i+j))]}" "${gpu}" &
      PIDS+=($!)
    done
    wait "${PIDS[@]}"
    echo "[GPU ${gpu}] Round ${round} 종료"
  done
}

# GPU 0 + GPU 1 병렬 launch
run_gpu_concurrent 0 "${GPU0_CELLS[@]}" &
PID_GPU0=$!
run_gpu_concurrent 1 "${GPU1_CELLS[@]}" &
PID_GPU1=$!

wait $PID_GPU0 $PID_GPU1

echo ""
echo "========================================"
echo "  Phase 2 Grid Sweep 25 cells DONE"
echo "========================================"
echo "[$(date -Iseconds)] Metrics 요약 (5x5 heatmap base):"
echo ""
printf "  %-30s | %-7s | %-7s | %-7s | %-7s\n" "cell" "R" "P" "F1" "EX"
echo "  $(printf '%.0s-' {1..78})"
for cell_dir in outputs/experiments/abl/c03_phase2_grid/*/; do
  cell=$(basename "${cell_dir}")
  m="${cell_dir}metrics.txt"
  if [ -f "${m}" ]; then
    R=$(grep "^recall:" "${m}" | awk '{print $2}')
    P=$(grep "^precision:" "${m}" | awk '{print $2}')
    EX=$(grep "^ex:" "${m}" | awk '{print $2}')
    F1=$(awk "BEGIN {printf \"%.4f\", 2*${P}*${R}/(${P}+${R})}" 2>/dev/null)
    printf "  %-30s | %-7s | %-7s | %-7s | %-7s\n" "${cell}" "${R}" "${P}" "${F1}" "${EX}"
  else
    printf "  %-30s | %-7s | %-7s | %-7s | %-7s\n" "${cell}" "(미생성)" "" "" ""
  fi
done

echo ""
echo "=== Phase 2 Grid Sweep DONE $(date -Iseconds) ==="
echo "Next:"
echo "  1. HISTORY/CATALOG/ID_MIGRATION 3종 갱신 — Phase 2 Grid 25 cells entry 추가"
echo "  2. Analyzer 핸드오프: notebooks/analysis_results/phase2_grid_heatmap_2026-05-XX.md"
echo "     - 5x5 heatmap (F1 + EX + TCR + TOR + Filter Prune Ratio)"
echo "     - Success criterion (a/b) 분기 판단"
echo "     - P2_02 vs c01_01 deterministic 일치 검증 (F1 차이 ≤ 0.0010 noise)"
echo "  3. Planner 핸드오프: closure narrative axis #11 갱신 or 재작성 결정"
