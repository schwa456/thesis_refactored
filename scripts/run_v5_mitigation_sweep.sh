#!/bin/bash
# Mitigation V5 sweep — (γ) Full Ablation 7 cells (DECISIONS 2026-05-13)
# V5-A 'gate'         : Conservation Law 수정 (Mustafa & Burkholz 2024)            — 1 cell
# V5-B 'gcnii'        : GCNII Identity Mapping (Chen 2020 + Peng 2024) L=2/4/6      — 3 cells
# V5-C 'aero_full'    : Full AERO Theorem 3+4 — hop + cumulative ablation           — 3 cells
#                        v5c_full / v5c_hop_only / v5c_cum_only
#
# 학습 entry: src/train_gat_s06.py (V5 kwargs forwarding patched —
#   gcnii_beta_lambda, aero_hop_attention, aero_cumulative_attention, aero_cumulative_decay)
# 비교 baseline: Phase 1 P80 (R=0.6097) + V4-A LN+GIN (R=0.5929) + V4-B AERO (R=0.5951)
#
# GPU 분배 — Stage 묶음 풀기 (GPU 별 자유 sequential, 사용자 지시 2026-05-13):
#   memory rule: CUDA_VISIBLE_DEVICES=0,1 만, GPU 2,3 reserved
#
#   GPU 0 chain (~35h):
#     v5a_gate (~10h) → v5b_gcnii_L2 (~10h) → v5b_gcnii_L4 (~15h)
#   GPU 1 chain (~50h, V5-B L=6 GPU 1 으로 swap):
#     v5c_full (~10h) → v5c_hop_only (~10h) → v5c_cum_only (~10h) → v5b_gcnii_L6 (~20h)
#
# Wall = max(35h, 50h) = ~50h (기존 4-stage 동기화 ~55h 대비 5h 단축, GPU 1 후반부 idle 제거).
#
# Launch (root 가 nohup + & 로 실행, SSH 끊김 복원성):
#   nohup bash scripts/run_v5_mitigation_sweep.sh > /tmp/v5_sweep.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp

mkdir -p /tmp/v5_train_logs
mkdir -p logs/train

NAS_DIR="/SSL_NAS/peoples/khj/thesis/checkpoints"
LOCAL_CKPT="outputs/checkpoints"

# ──────────────────────────────────────────────────────────────
# 단일 variant 학습 + NAS mv + symlink
# ──────────────────────────────────────────────────────────────
train_one() {
  local variant=$1  # v5a_gate / v5b_gcnii_L{2,4,6} / v5c_{full,hop_only,cum_only}
  local gpu=$2
  local cfg="configs/training/dsn/train_dsn_p80_${variant}.yaml"
  local log="/tmp/v5_train_logs/train_${variant}.log"
  local ckpt_name="best_gat_directed_supernode_p80_${variant}.pt"

  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] start V5 ${variant} (300 epochs)"
  CUDA_VISIBLE_DEVICES=${gpu} conda run -n base python -u src/train_gat_s06.py --config "${cfg}" \
      > "${log}" 2>&1 || echo "[WARN] V5 ${variant} non-zero exit"
  echo "[$(date +%H:%M:%S)] [GPU ${gpu}] end V5 ${variant}"

  # post-train NAS mv + symlink (NAS 우선 저장 정책)
  if [ -f "${LOCAL_CKPT}/${ckpt_name}" ] && [ ! -L "${LOCAL_CKPT}/${ckpt_name}" ]; then
    echo "[$(date +%H:%M:%S)] mv ${ckpt_name} → NAS"
    cp -a "${LOCAL_CKPT}/${ckpt_name}" "${NAS_DIR}/" \
      && rm -f "${LOCAL_CKPT}/${ckpt_name}" \
      && ln -s "${NAS_DIR}/${ckpt_name}" "${LOCAL_CKPT}/${ckpt_name}" \
      && echo "[$(date +%H:%M:%S)] ${ckpt_name} → symlink (NAS)" \
      || echo "[FAIL] NAS mv ${ckpt_name}"
  elif [ -L "${LOCAL_CKPT}/${ckpt_name}" ]; then
    echo "[$(date +%H:%M:%S)] ${ckpt_name} 이미 symlink"
  else
    echo "[WARN] ${ckpt_name} 로컬 ckpt 미존재 — 학습 실패 추정"
  fi
}

echo "========================================"
echo "  Mitigation V5 (γ) Full Ablation — 7 cells (stage-unbound GPU sequential)"
echo "  Started: $(date -Iseconds)"
echo "  GPU 0 chain (~35h): v5a_gate → v5b_gcnii_L2 → v5b_gcnii_L4"
echo "  GPU 1 chain (~50h): v5c_full → v5c_hop_only → v5c_cum_only → v5b_gcnii_L6"
echo "  Expected wall: ~50h"
echo "========================================"

# ──────────────────────────────────────────────────────────────
# GPU 0 chain (sequential, ~35h)
# ──────────────────────────────────────────────────────────────
{
  train_one v5a_gate 0
  train_one v5b_gcnii_L2 0
  train_one v5b_gcnii_L4 0
  echo "[$(date +%H:%M:%S)] [GPU 0] ALL DONE"
} > logs/train/sweep_gpu0.log 2>&1 &
GPU0_PID=$!
echo "[$(date +%H:%M:%S)] GPU 0 chain launched, PID=${GPU0_PID}"

# ──────────────────────────────────────────────────────────────
# GPU 1 chain (sequential, ~50h — V5-B L=6 GPU 1 끝에 swap)
# ──────────────────────────────────────────────────────────────
{
  train_one v5c_full 1
  train_one v5c_hop_only 1
  train_one v5c_cum_only 1
  train_one v5b_gcnii_L6 1
  echo "[$(date +%H:%M:%S)] [GPU 1] ALL DONE"
} > logs/train/sweep_gpu1.log 2>&1 &
GPU1_PID=$!
echo "[$(date +%H:%M:%S)] GPU 1 chain launched, PID=${GPU1_PID}"

wait $GPU0_PID
echo "[$(date +%H:%M:%S)] GPU 0 chain 종료"
wait $GPU1_PID
echo "[$(date +%H:%M:%S)] GPU 1 chain 종료"

# ──────────────────────────────────────────────────────────────
# 학습 요약 (best val recall@15 추출, 7 ckpt)
# ──────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  7 ckpt 최종 val recall@15"
echo "========================================"
for variant in v5a_gate v5b_gcnii_L2 v5b_gcnii_L4 v5b_gcnii_L6 v5c_full v5c_hop_only v5c_cum_only; do
  logger_log=$(ls -t logs/train/dsn_p80_${variant}_*.log 2>/dev/null | head -1)
  if [ -z "${logger_log}" ]; then
    logger_log=$(ls -t logs/train/*dsn_p80_${variant}*.log 2>/dev/null | head -1)
  fi
  if [ -n "${logger_log}" ] && [ -f "${logger_log}" ]; then
    n=$(grep -c "Val Recall@15:" "${logger_log}")
    best=$(grep -oE "Val Recall@15: [0-9.]+" "${logger_log}" | awk '{print $NF}' | sort -rn | head -1)
    printf "  %-24s %d epochs logged, best=%s\n" "${variant}" "${n}" "${best:-N/A}"
  else
    printf "  %-24s logger log 미존재 — /tmp/v5_train_logs/train_%s.log 확인\n" "${variant}" "${variant}"
  fi
done

echo ""
echo "Mitigation V5 sweep finished: $(date -Iseconds)"
echo "Next:"
echo "  - root: HISTORY + CATALOG + ID_MIGRATION 갱신 (7 신규 ID)"
echo "  - analyzer: notebooks/analysis_results/dsn_mitigation_v5_4dir.md (14-trial + V5-C 3 ablation)"
echo "  - planner: paper §V.5.4 narrative pivot + V5-D-2 trigger 결정"
