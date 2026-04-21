#!/usr/bin/env bash
# s06_a01_07_b5_enriched_dual_stream — B5 구조 + Enriched Graph + Batched dual_stream
#
# Notes:
#   - 최초 실행 시 /SSL_NAS/peoples/khj/thesis/train/train_enriched_plm_graphs.pt 를 빌드 (~1h).
#     기존 train_enriched_graphs.pt (token-level) 와는 형식이 달라 별도 캐시가 필요.
#   - batch_size=8, ~4-6h 예상 (기존 B5 batch=1 는 ~29h).
#   - GPU 0 전용.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_REL="experiments/s06_gat_bottleneck_fix/a01_additive_ablation/s06_a01_07_b5_enriched_dual_stream"
TAG="s06_a01_07_b5_enriched"
LOG_DIR="outputs/archive/wrapper_logs/s06"
LOG_FILE="${LOG_DIR}/${TAG}.log"
DRIVER_LOG="${LOG_DIR}/_b5_enriched_driver.log"
SUMMARY_LOG="${LOG_DIR}/_summary.log"
mkdir -p "${LOG_DIR}"

# Cache symlinks: 로컬 data/processed/ → offload NAS (원본 /SSL_NAS/peoples/khj/thesis/train/
# 경로에 저장하면 NFS stall 이 발생하는 현상 관측, offload 로 우회)
OFFLOAD_DIR="/SSL_NAS/peoples/khj/thesis_refactored_offload/processed"
LOCAL_PROC_DIR="data/processed"
mkdir -p "${OFFLOAD_DIR}"
for TAG_NAME in train dev; do
  SRC="${OFFLOAD_DIR}/${TAG_NAME}_enriched_plm_graphs.pt"
  LINK="${LOCAL_PROC_DIR}/${TAG_NAME}_enriched_plm_graphs.pt"
  if [ ! -L "${LINK}" ] && [ ! -e "${LINK}" ]; then
    ln -s "${SRC}" "${LINK}"
  fi
done

{
  echo ""
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> START ${TAG}"
  echo "  config=configs/${CONFIG_REL}.yaml"
  echo "  gpu=0"
  echo "  batch_size=8 (batched dual_stream)"
} | tee -a "${SUMMARY_LOG}" | tee "${DRIVER_LOG}"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  conda run -n base --no-capture-output \
    python src/train_gat_s06.py --config "configs/${CONFIG_REL}.yaml" \
  2>&1 | tee "${LOG_FILE}"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] <<< DONE ${TAG}"
} | tee -a "${SUMMARY_LOG}" | tee -a "${DRIVER_LOG}"
