#!/bin/bash
# B5 head retrain: frozen L_out 위에서 head (linear vs mlp) 비교
# Step 1: train/dev L_out 캐시 추출 (GPU 0)
# Step 2: Exp A (linear, GPU 0) + Exp B (mlp, GPU 1) 병렬 학습
set -e
cd "$(dirname "$0")/.."

CONFIG="configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation/s06_a01_06_b5_dual_stream.yaml"
CKPT="/SSL_NAS/peoples/khj/thesis/checkpoints/s06_gat_bottleneck_fix/best_gat_s06_a01_06_b5.pt"
OUT="outputs/analysis/s06_bottleneck/B5/retrain"
CACHE_TR="${OUT}/lout_cache_train.pt"
CACHE_DEV="${OUT}/lout_cache_dev.pt"

mkdir -p "${OUT}"

# --- Step 1 ---
echo "=========================================="
echo " Step 1: Extract frozen L_out — TRAIN "
echo "=========================================="
if [ ! -f "${CACHE_TR}" ]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    conda run -n base python src/analysis/b5_extract_frozen_lout.py \
      --config "${CONFIG}" --checkpoint "${CKPT}" --split train \
      --output "${CACHE_TR}" 2>&1 | tee "${OUT}/extract_train.log"
else
  echo "  (exists, skipping) ${CACHE_TR}"
fi

echo "=========================================="
echo " Step 1: Extract frozen L_out — DEV  "
echo "=========================================="
if [ ! -f "${CACHE_DEV}" ]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    conda run -n base python src/analysis/b5_extract_frozen_lout.py \
      --config "${CONFIG}" --checkpoint "${CKPT}" --split dev \
      --output "${CACHE_DEV}" 2>&1 | tee "${OUT}/extract_dev.log"
else
  echo "  (exists, skipping) ${CACHE_DEV}"
fi

# --- Step 2 ---
echo "=========================================="
echo " Step 2: Parallel head retrain "
echo "   Exp A (linear) on GPU 0 "
echo "   Exp B (mlp)    on GPU 1 "
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  conda run -n base python src/analysis/b5_head_retrain.py \
    --train_cache "${CACHE_TR}" --dev_cache "${CACHE_DEV}" \
    --head linear --output_dir "${OUT}/exp_a_linear" \
    > "${OUT}/exp_a_linear.log" 2>&1 &
PID_A=$!
echo "  Exp A started (PID=${PID_A})"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  conda run -n base python src/analysis/b5_head_retrain.py \
    --train_cache "${CACHE_TR}" --dev_cache "${CACHE_DEV}" \
    --head mlp --output_dir "${OUT}/exp_b_mlp" \
    > "${OUT}/exp_b_mlp.log" 2>&1 &
PID_B=$!
echo "  Exp B started (PID=${PID_B})"

wait ${PID_A}
echo "  Exp A finished"
wait ${PID_B}
echo "  Exp B finished"

echo "=========================================="
echo " RESULTS "
echo "=========================================="
for exp in exp_a_linear exp_b_mlp; do
  echo "--- ${exp} ---"
  conda run -n base python -c "
import json
d = json.load(open('${OUT}/${exp}/summary.json'))
print(f\"head          : {d['head']}\")
print(f\"n_params      : {d['n_params']}\")
print(f\"best_epoch    : {d['best_epoch']}\")
print(f\"best_val_r15  : {d['best_val_r15']:.4f}\")
print(f\"final VAL AUC : {d['final_val']['auc']:.4f}\")
print(f\"final VAL R15 : {d['final_val']['recall@15']:.4f}\")
print(f\"final DEV AUC : {d['final_dev']['auc']:.4f}\")
print(f\"final DEV R15 : {d['final_dev']['recall@15']:.4f}\")
"
  echo ""
done

echo "All done. Summaries: ${OUT}/{exp_a_linear,exp_b_mlp}/summary.json"
