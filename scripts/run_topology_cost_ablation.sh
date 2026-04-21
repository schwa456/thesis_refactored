#!/usr/bin/env bash
# s03_a09 TopologyCostPCSTExtractor ablation (1차 pass: NoFilter only)
#   - Filter를 껐기에 extractor 자체의 순효과 측정이 주 목적
#   - XiYan 비교는 vLLM 서버 여유 생길 때 별도 pass로
#   - 새 prototype 2개 (방향 2: TopologyCost / CA TopologyCost) + 비교 anchor 3개
#   - Ensemble selector + 동일 GAT checkpoint로 fair 비교
#   - 중간에 실패해도 다음 실험으로 진행
#
# 실행: bash scripts/run_topology_cost_ablation.sh
#      nohup bash scripts/run_topology_cost_ablation.sh > outputs/logs/s03_a09/_nohup.log 2>&1 &

set -u
set -o pipefail

cd /home/hyeonjin/thesis_refactored

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

CFG_DIR="configs/experiments/s03_gat_ensemble/a09_topology_cost"
LOG_DIR="outputs/logs/s03_a09"
mkdir -p "$LOG_DIR"

# 실행 순서: 새 prototype 먼저 → anchor 뒤에
# (새 prototype이 실패하면 anchor까지 가기 전에 문제를 바로 볼 수 있음)
CONFIGS=(
  "experiments/s03_gat_ensemble/a09_topology_cost/s03_a09_01_topology_no_filter"
  "experiments/s03_gat_ensemble/a09_topology_cost/s03_a09_02_ca_topology_no_filter"
  "experiments/s03_gat_ensemble/a09_topology_cost/s03_a09_03_basic_no_filter_anchor"
  "experiments/s03_gat_ensemble/a09_topology_cost/s03_a09_04_ca_product_no_filter_anchor"
  "experiments/s03_gat_ensemble/a09_topology_cost/s03_a09_05_adaptive_no_filter_anchor"
)

SUMMARY="$LOG_DIR/_summary.log"
echo "=== s03_a09 TopologyCost ablation started at $(date '+%F %T') on GPU 0 ===" | tee -a "$SUMMARY"

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
echo "=== s03_a09 ablation finished at $(date '+%F %T') ==="  | tee -a "$SUMMARY"
echo ""                                                | tee -a "$SUMMARY"
echo "DONE. 다음 단계:"                                 | tee -a "$SUMMARY"
echo "  1. outputs/summary_all.csv 에 6개 행 추가됐는지 확인"      | tee -a "$SUMMARY"
echo "  2. HISTORY/CATALOG/ID_MIGRATION 3개 문서 업데이트"         | tee -a "$SUMMARY"
echo "  3. Recall/Precision/F1 비교 (소수점 4자리)"                 | tee -a "$SUMMARY"
