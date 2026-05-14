#!/bin/bash
# B5 head retrain Phase 2: Exp C / D / E
#   C = mlp + listnet  + none
#   D = mlp + bce      + zscore
#   E = mlp + listnet  + zscore
set -e
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

OUT="outputs/analysis/s06_bottleneck/B5/retrain"
CACHE_TR="${OUT}/lout_cache_train.pt"
CACHE_DEV="${OUT}/lout_cache_dev.pt"

run_exp () {
  local gpu=$1
  local tag=$2
  local loss=$3
  local norm=$4
  local dir="${OUT}/exp_${tag}"
  local log="${OUT}/exp_${tag}.log"
  echo "  [gpu${gpu}] ${tag}: head=mlp loss=${loss} norm=${norm}"
  CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=src \
    conda run -n base python src/analysis/b5_head_retrain.py \
      --train_cache "${CACHE_TR}" --dev_cache "${CACHE_DEV}" \
      --head mlp --loss "${loss}" --normalize "${norm}" \
      --output_dir "${dir}" \
      > "${log}" 2>&1
}

echo "=========================================="
echo " Phase 1: Exp C (GPU 0) || Exp D (GPU 1)"
echo "=========================================="
run_exp 0 c_listnet    listnet none   &
PID_C=$!
run_exp 1 d_bce_zscore bce     zscore &
PID_D=$!
wait ${PID_C} && echo "  Exp C done"
wait ${PID_D} && echo "  Exp D done"

echo "=========================================="
echo " Phase 2: Exp E (GPU 0)"
echo "=========================================="
run_exp 0 e_listnet_zscore listnet zscore

echo "=========================================="
echo " RESULTS (summary) "
echo "=========================================="
for exp in b_mlp c_listnet d_bce_zscore e_listnet_zscore; do
  if [ -f "${OUT}/exp_${exp}/summary.json" ]; then
    echo "--- exp_${exp} ---"
    conda run -n base python -c "
import json
d = json.load(open('${OUT}/exp_${exp}/summary.json'))
print(f\"  loss={d.get('loss','?')}  norm={d.get('normalize','?')}\")
print(f\"  best(val-ES): ep={d['best_epoch']}  val_r15={d['best_val_r15']:.4f}  final_dev_auc={d['final_dev']['auc']:.4f}  final_dev_r15={d['final_dev']['recall@15']:.4f}\")
print(f\"  best(dev-ES): ep={d.get('best_dev_r15_epoch','?')}  dev_auc={d.get('best_dev_auc','?'):.4f}  dev_r15={d.get('best_dev_r15','?'):.4f}\")
"
  fi
done

echo "All done."
