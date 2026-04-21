#!/usr/bin/env bash
# B0 학습 종료를 감시하다가 끝나면 local checkpoint 를 NAS 로 옮긴다.
# Driver 가 _summary.log 에 "END   s06_a01_01_b0" 라인을 찍을 때까지 60초 간격으로 폴링.

set -u

SUMMARY="/home/hyeonjin/thesis_refactored/outputs/logs/s06/_summary.log"
SRC="/home/hyeonjin/thesis_refactored/outputs/checkpoints/best_gat_s06_a01_01_b0.pt"
DST_DIR="/SSL_NAS/peoples/khj/thesis/checkpoints/s06_gat_bottleneck_fix"

echo "[$(date '+%F %T')] watcher: waiting for B0 END in $SUMMARY"

until grep -q "END   s06_a01_01_b0" "$SUMMARY" 2>/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] watcher: B0 END detected"

if [[ -f "$SRC" ]]; then
  mkdir -p "$DST_DIR"
  mv "$SRC" "$DST_DIR/"
  rc=$?
  msg="[$(date '+%F %T')] B0 ckpt moved to NAS (rc=$rc): $DST_DIR/$(basename "$SRC")"
else
  msg="[$(date '+%F %T')] ⚠️ B0 local ckpt not found at $SRC"
fi
echo "$msg" | tee -a "$SUMMARY"
