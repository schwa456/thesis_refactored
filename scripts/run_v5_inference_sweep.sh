#!/bin/bash
# V5 Inference Sweep — 4 concurrent (GPU 0 × 2 + GPU 1 × 2), 2 round
# 근거: planning/DECISIONS.md 2026-05-15 §0 Option γ + 사용자 5/15 명시 V5 inference 병렬
#
# Stack: anchor (MSTPCSTUnion+XiYan+SQL F1=0.8434 EX=0.4889) 의 selector 만 V5 ckpt 로 교체
# 7 cell × ~1.5h (anchor SQL sweep wall 정합) → 2 round × ~1.5h = ~3h wall
#
# V5 학습 종료 자동 감지 (pgrep train_gat_s06) 후 launch.
#
# Launch:
#   nohup bash scripts/run_v5_inference_sweep.sh > logs/v5_inference_main.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1

mkdir -p logs/v5_inference

echo "=== V5 Inference Sweep $(date -Iseconds) ==="

# ──────────────────────────────────────────────────────────────
# V5 학습 종료 wait
# ──────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] V5 학습 종료 wait (pgrep train_gat_s06)..."
while pgrep -f "train_gat_s06" > /dev/null; do
  sleep 60
done
echo "[$(date +%H:%M:%S)] V5 학습 모두 종료. V5 inference launch."

# ──────────────────────────────────────────────────────────────
# Single cell inference helper
# ──────────────────────────────────────────────────────────────
run_one() {
  local variant=$1
  local gpu=$2
  local cfg="experiments/abl/v5_inference/v5_inf_${variant}"
  local log="logs/v5_inference/${variant}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] start ${variant}"
  CUDA_VISIBLE_DEVICES=${gpu} conda run -n base python -u src/main.py \
    --config "${cfg}" > "${log}" 2>&1 || echo "[WARN] ${variant} non-zero exit"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] end ${variant}"
}

# ──────────────────────────────────────────────────────────────
# Round 1: 4 concurrent (GPU 0 × 2 + GPU 1 × 2)
#   GPU 0: v5a_gate + v5b_gcnii_L2
#   GPU 1: v5b_gcnii_L4 + v5b_gcnii_L6
# ──────────────────────────────────────────────────────────────
echo ""
echo "[$(date -Iseconds)] === Round 1 (4 concurrent) ==="
run_one v5a_gate 0 &
PID_R1A=$!
run_one v5b_gcnii_L2 0 &
PID_R1B=$!
run_one v5b_gcnii_L4 1 &
PID_R1C=$!
run_one v5b_gcnii_L6 1 &
PID_R1D=$!

wait $PID_R1A $PID_R1B $PID_R1C $PID_R1D
echo "[$(date +%H:%M:%S)] Round 1 종료"

# ──────────────────────────────────────────────────────────────
# Round 2: 3 concurrent (GPU 0 × 2 + GPU 1 × 1)
#   GPU 0: v5c_full + v5c_hop_only
#   GPU 1: v5c_cum_only
# ──────────────────────────────────────────────────────────────
echo ""
echo "[$(date -Iseconds)] === Round 2 (3 concurrent) ==="
run_one v5c_full 0 &
PID_R2A=$!
run_one v5c_hop_only 0 &
PID_R2B=$!
run_one v5c_cum_only 1 &
PID_R2C=$!

wait $PID_R2A $PID_R2B $PID_R2C
echo "[$(date +%H:%M:%S)] Round 2 종료"

# ──────────────────────────────────────────────────────────────
# 7 cell 최종 metrics 요약
# ──────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  V5 Inference 7 cell metrics 요약"
echo "========================================"
for variant in v5a_gate v5b_gcnii_L2 v5b_gcnii_L4 v5b_gcnii_L6 v5c_full v5c_hop_only v5c_cum_only; do
  metrics="outputs/experiments/abl/v5_inference/v5_inf_${variant}/metrics.txt"
  if [ -f "${metrics}" ]; then
    R=$(grep "^recall:" "${metrics}" | awk '{print $2}')
    P=$(grep "^precision:" "${metrics}" | awk '{print $2}')
    EX=$(grep "^ex:" "${metrics}" | awk '{print $2}')
    printf "  %-20s R=%s P=%s EX=%s\n" "${variant}" "${R}" "${P}" "${EX}"
  else
    printf "  %-20s (metrics 미생성)\n" "${variant}"
  fi
done

echo ""
echo "=== V5 Inference Sweep DONE $(date -Iseconds) ==="
echo "Next: HISTORY/CATALOG/ID_MIGRATION 7 신규 ID 등재 + Analyzer 핸드오프"
