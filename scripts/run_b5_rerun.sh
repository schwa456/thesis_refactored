#!/usr/bin/env bash
# B5 Dual-Stream 단독 재실행 (fk_node isolated-graph fix 적용 후)
# GPU 0 전용, 로그는 symlink outputs/logs/s06 → archive/wrapper_logs/s06 로 기록.
set -u
set -o pipefail

cd /home/hyeonjin/thesis_refactored
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

CFG="configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation/s06_a01_06_b5_dual_stream.yaml"
LOG_DIR="outputs/logs/s06"
NAME="s06_a01_06_b5_dual_stream"
LOG="$LOG_DIR/${NAME}.log"
SUMMARY="$LOG_DIR/_summary.log"

# 이전 crash 로그 보존
if [[ -f "$LOG" ]]; then
  mv "$LOG" "${LOG%.log}.crash_$(date '+%Y%m%d_%H%M%S').log"
fi

echo "" | tee -a "$SUMMARY"
echo "[$(date '+%F %T')] >>> START $NAME (rerun after fk_node fix)" | tee -a "$SUMMARY"
start_ts=$(date +%s)

python -u src/train_gat_s06.py --config "$CFG" >"$LOG" 2>&1
rc=$?

end_ts=$(date +%s)
dur=$(( end_ts - start_ts ))
hh=$(( dur / 3600 )); mm=$(( (dur % 3600) / 60 )); ss=$(( dur % 60 ))
printf "[%s] <<< END %s rc=%d elapsed=%02d:%02d:%02d log=%s\n" \
  "$(date '+%F %T')" "$NAME" "$rc" "$hh" "$mm" "$ss" "$LOG" | tee -a "$SUMMARY"
