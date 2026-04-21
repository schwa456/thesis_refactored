#!/usr/bin/env bash
# s06 GAT Bottleneck Fix — sequential additive ablation (B0..B5)
# GPU 0 전용, 하나씩 순차 실행. 중간에 실패해도 다음 실험은 계속.
set -u
set -o pipefail

cd /home/hyeonjin/thesis_refactored

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

CFG_DIR="configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation"
LOG_DIR="outputs/logs/s06"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "$CFG_DIR/s06_a01_01_b0_baseline.yaml"
  "$CFG_DIR/s06_a01_02_b1_pairnorm.yaml"
  "$CFG_DIR/s06_a01_03_b2_initial_residual.yaml"
  "$CFG_DIR/s06_a01_04_b3_listnet.yaml"
  "$CFG_DIR/s06_a01_05_b4_anti_collapse.yaml"
  "$CFG_DIR/s06_a01_06_b5_dual_stream.yaml"
)

SUMMARY="$LOG_DIR/_summary.log"
echo "=== s06 ablation started at $(date '+%F %T') on GPU 0 ===" | tee -a "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  log="$LOG_DIR/$name.log"

  echo ""              | tee -a "$SUMMARY"
  echo "[$(date '+%F %T')] >>> START $name"  | tee -a "$SUMMARY"
  start_ts=$(date +%s)

  python -u src/train_gat_s06.py --config "$cfg" >"$log" 2>&1
  rc=$?

  end_ts=$(date +%s)
  dur=$(( end_ts - start_ts ))
  hh=$(( dur / 3600 )); mm=$(( (dur % 3600) / 60 )); ss=$(( dur % 60 ))
  printf "[$(date '+%%F %%T')] <<< END   %s  rc=%d  elapsed=%02d:%02d:%02d  log=%s\n" \
    "$name" "$rc" "$hh" "$mm" "$ss" "$log" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== s06 ablation finished at $(date '+%F %T') ===" | tee -a "$SUMMARY"
