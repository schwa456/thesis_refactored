#!/usr/bin/env bash
# s03_a10 FKBackboneSteinerExtractor ablation (NoFilter)
#   - Schema graph의 Table-FK 백본에서 Steiner closure로 bridge table 구조적 회수
#   - column_recovery_threshold 3개 값 sweep: 0.0 / 0.3 / 0.5
#   - Anchor는 a09_05 (AdaptivePCST) 재사용 — 같은 Selector/Filter 구성이므로 직접 비교
#   - 목표: Recall 0.72 → 0.85+ 도약 관찰
#   - 중간에 실패해도 다음 실험으로 진행
#
# 실행: bash scripts/run_fk_steiner_ablation.sh
#      nohup bash scripts/run_fk_steiner_ablation.sh > outputs/logs/s03_a10/_nohup.log 2>&1 &

set -u
set -o pipefail

cd /home/hyeonjin/thesis_refactored

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

CFG_DIR="configs/experiments/s03_gat_ensemble/a10_fk_steiner"
LOG_DIR="outputs/logs/s03_a10"
mkdir -p "$LOG_DIR"

CONFIGS=(
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_01_fk_steiner_full_col"
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_02_fk_steiner_mid_col"
  "experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_03_fk_steiner_high_col"
)

SUMMARY="$LOG_DIR/_summary.log"
echo "=== s03_a10 FKBackboneSteiner ablation started at $(date '+%F %T') on GPU 0 ===" | tee -a "$SUMMARY"

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
echo "=== s03_a10 ablation finished at $(date '+%F %T') ==="  | tee -a "$SUMMARY"
echo ""                                                | tee -a "$SUMMARY"
echo "DONE. 다음 단계:"                                 | tee -a "$SUMMARY"
echo "  1. outputs/summary_all.csv 에 3개 행 추가됐는지 확인"      | tee -a "$SUMMARY"
echo "  2. HISTORY/CATALOG/ID_MIGRATION 3개 문서 업데이트"         | tee -a "$SUMMARY"
echo "  3. Recall/Precision/F1 비교 (소수점 4자리)"                 | tee -a "$SUMMARY"
echo "  4. a09_05 (AdaptivePCST) anchor와 비교 — Recall +0.1 달성 여부" | tee -a "$SUMMARY"
