#!/bin/bash
# B5 Head-Only LDBO diagnostic
# 목적: Val split을 DB 단위로 바꿔 "realistic unseen-DB early-stop"을 simulate.
# GAT은 기존 B5 checkpoint (train DB 전체 본 상태) 그대로. Head만 LDBO로 재학습.
# GPU 0만 사용, 순차 실행 (B / C / D / E 4종).
set -e
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

OUT_BASE="outputs/analysis/s06_bottleneck/B5/retrain"
OUT="${OUT_BASE}/ldbo"
CACHE_TR="${OUT_BASE}/lout_cache_train.pt"
CACHE_DEV="${OUT_BASE}/lout_cache_dev.pt"
TRAIN_JSON="/SSL_NAS/peoples/khj/thesis/train/train.json"
# 69 train DBs 중 11개 홀드아웃 → 약 16% (dev scale에 맞춤)
LDBO_FRAC=0.16
SEED=42

mkdir -p "${OUT}"

run_exp () {
  local tag=$1
  local loss=$2
  local norm=$3
  local dir="${OUT}/exp_${tag}"
  local log="${OUT}/exp_${tag}.log"
  echo "--------------------------------------------------"
  echo "[$(date +%H:%M:%S)] ${tag}: loss=${loss} norm=${norm}  (LDBO frac=${LDBO_FRAC})"
  echo "--------------------------------------------------"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    conda run -n base python src/analysis/b5_head_retrain.py \
      --train_cache "${CACHE_TR}" --dev_cache "${CACHE_DEV}" \
      --head mlp --loss "${loss}" --normalize "${norm}" \
      --ldbo_frac ${LDBO_FRAC} --train_json "${TRAIN_JSON}" \
      --seed ${SEED} \
      --output_dir "${dir}" \
      2>&1 | tee "${log}"
  echo "  [done] ${tag}"
}

echo "=========================================="
echo " B5 Head-Only LDBO Diagnostic — GPU 0"
echo "=========================================="
run_exp b_bce_none       bce     none
run_exp c_listnet        listnet none
run_exp d_bce_zscore     bce     zscore
run_exp e_listnet_zscore listnet zscore

echo "=========================================="
echo " LDBO vs Query-Random Comparison"
echo "=========================================="
conda run -n base python - <<'EOF'
import json
from pathlib import Path

base = Path("outputs/analysis/s06_bottleneck/B5/retrain")
rows = []
for tag in ["b_bce_none", "c_listnet", "d_bce_zscore", "e_listnet_zscore"]:
    ldbo_f = base / "ldbo" / f"exp_{tag}" / "summary.json"
    # 기존 query-random 실험과 매핑 (이름 약간 다름)
    qrand_map = {"b_bce_none": "exp_b_mlp", "c_listnet": "exp_c_listnet",
                 "d_bce_zscore": "exp_d_bce_zscore", "e_listnet_zscore": "exp_e_listnet_zscore"}
    qrand_f = base / qrand_map[tag] / "summary.json"
    if not ldbo_f.exists():
        continue
    L = json.load(open(ldbo_f))
    Q = json.load(open(qrand_f)) if qrand_f.exists() else None
    rows.append((tag, L, Q))

print(f"{'Exp':<20} | {'LDBO val R@15':>12} | {'QR val R@15':>12} | "
      f"{'LDBO dev AUC(ES)':>16} | {'QR dev AUC(ES)':>14}")
print("-" * 90)
for tag, L, Q in rows:
    lv = L["best_val_r15"]
    ld = L["final_dev"]["auc"]
    qv = Q["best_val_r15"] if Q else float("nan")
    qd = Q["final_dev"]["auc"] if Q else float("nan")
    print(f"{tag:<20} | {lv:>12.4f} | {qv:>12.4f} | {ld:>16.4f} | {qd:>14.4f}")

print("\nLDBO best dev-AUC/R15 (oracle dev-ES):")
for tag, L, _ in rows:
    print(f"  {tag:<20} dev AUC {L['best_dev_auc']:.4f}@ep{L['best_dev_auc_epoch']} | "
          f"dev R@15 {L['best_dev_r15']:.4f}@ep{L['best_dev_r15_epoch']}")
EOF

echo ""
echo "All done. Outputs: ${OUT}/exp_*/summary.json"
