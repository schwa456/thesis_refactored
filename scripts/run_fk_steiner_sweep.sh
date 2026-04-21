#!/usr/bin/env bash
# s03_a10 FKBackboneSteiner θ_r sweep (0.1 → 1.0, step 0.1)
#   - column_recovery_threshold 세밀 sweep, a10_01~03 (0.0/0.3/0.5) 이미 완료
#   - 신규 8개 run: 0.1 / 0.2 / 0.4 / 0.6 / 0.7 / 0.8 / 0.9 / 1.0
#   - NoFilter, GPU 0 only, 중간 실패 시 다음으로 진행
#   - 전체 약 120분 (8 × 15:45) 예상
#
# 실행: bash scripts/run_fk_steiner_sweep.sh
#      nohup bash scripts/run_fk_steiner_sweep.sh > outputs/logs/s03_a10/_nohup_sweep.log 2>&1 &

set -u
set -o pipefail

cd /home/hyeonjin/thesis_refactored

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

LOG_DIR="outputs/logs/s03_a10"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_04_fk_steiner_r01"
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_05_fk_steiner_r02"
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_06_fk_steiner_r04"
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_07_fk_steiner_r06"
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_08_fk_steiner_r07"
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_09_fk_steiner_r08"
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_10_fk_steiner_r09"
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_11_fk_steiner_r10"
)

SUMMARY="$LOG_DIR/_summary_sweep.log"
echo "=== s03_a10 θ_r sweep started at $(date '+%F %T') on GPU 0 ===" | tee -a "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg")
  log="$LOG_DIR/$name.log"

  echo ""                                              | tee -a "$SUMMARY"
  echo "[$(date '+%F %T')] >>> START $name"            | tee -a "$SUMMARY"
  start_ts=$(date +%s)

  conda run -n base python -u src/main.py --config "$cfg" >"$log" 2>&1 || true
  rc=$?

  end_ts=$(date +%s)
  dur=$(( end_ts - start_ts ))
  hh=$(( dur / 3600 )); mm=$(( (dur % 3600) / 60 )); ss=$(( dur % 60 ))
  printf "[%s] <<< END   %s  rc=%d  elapsed=%02d:%02d:%02d  log=%s\n" \
    "$(date '+%F %T')" "$name" "$rc" "$hh" "$mm" "$ss" "$log" | tee -a "$SUMMARY"
done

echo ""                                                | tee -a "$SUMMARY"
echo "=== s03_a10 θ_r sweep finished at $(date '+%F %T') ==="  | tee -a "$SUMMARY"
echo ""                                                | tee -a "$SUMMARY"
echo "DONE. 다음 단계:"                                 | tee -a "$SUMMARY"
echo "  1. outputs/summary_all.csv 에 8개 행 추가됐는지 확인"      | tee -a "$SUMMARY"
echo "  2. HISTORY/CATALOG/ID_MIGRATION 3개 문서 업데이트"         | tee -a "$SUMMARY"
echo "  3. θ_r 별 Recall/Precision/F1 추이 비교 (11 points total)" | tee -a "$SUMMARY"
