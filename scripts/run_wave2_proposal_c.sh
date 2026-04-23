#!/bin/bash
# Wave 2 Proposal C — Diameter-based GAT num_layers sweep
# 근거: planning/proposals/abl_sel_diameter_layers.md §4.2 (global fixed mode)
# 제안서 §4.2 H2 per-DB dynamic num_layers 는 selector 세션 인프라 미비로 deferred
#   (EnsembleSelector v1-only, db_name threading 부재 — task #3 escalation)
#
# 스윕 축: model.num_layers ∈ {1, 2, 3, 6, 7}
#   - 6 = global D_max over BIRD dev 11 DBs (data/processed/dev_diameter.pt)
#   - 7 = D_max+1, over-smoothing 재등장 관측
# Anchor: s04_04_qcond_a0_xiyan (QCond concat, α=0, ComponentAwareProductCostPCST + XiYan)
#
# 순서:
#   Phase 1: 5 GAT 학습 (train_gat.py, v1 GAT + Projector)  — vLLM kill 필요 (GPU 경합)
#   Phase 2: 5 inference (src/main.py)  — vLLM 재기동 필요
#
# GPU: CUDA_VISIBLE_DEVICES=0,1 (memory rule; GPU 2,3 은 다른 연구자 예약)
# 체크포인트: outputs/checkpoints/best_gat_qcond_nl{1,2,3,6,7}.pt → NAS symlink 필요 (CLAUDE.md NAS 규칙)
# 예상 시간: 학습 5×~5h + inference 5×~45min ≈ 28~30h
# 승인 필요: vLLM 종료 (kill permission memory rule)

set -u
cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# 사전 조건: vLLM server 가 GPU 0/1 메모리를 점유 중 (약 40GB). 학습 전에 반드시 kill.
# 실행 전 user 가 vLLM 중단 승인 + 직접 kill or VLLM_AUTOKILL=1 환경변수 설정
# -----------------------------------------------------------------------------
VLLM_PID=$(pgrep -f "vllm.entrypoints.openai.api_server" | head -1)
if [ -n "${VLLM_PID}" ]; then
    echo "[!] vLLM 서버가 실행 중입니다 (PID ${VLLM_PID})."
    if [ "${VLLM_AUTOKILL:-0}" = "1" ]; then
        echo "[!] VLLM_AUTOKILL=1 → vLLM 종료"
        kill -TERM "${VLLM_PID}" && sleep 20
    else
        echo "    GPU 메모리 경합으로 학습이 OOM 될 가능성이 큽니다."
        echo "    VLLM_AUTOKILL=1 설정 후 재실행하거나 수동으로 vLLM 종료 후 재실행 하세요."
        echo "    Phase 2 inference 시작 전 vLLM 재기동 필요."
        exit 1
    fi
fi

TRAIN_CFGS=(
    "training/diameter_layers/train_qcond_nl1"
    "training/diameter_layers/train_qcond_nl2"
    "training/diameter_layers/train_qcond_nl3"
    "training/diameter_layers/train_qcond_nl6"
    "training/diameter_layers/train_qcond_nl7"
)

INFER_CFGS=(
    "experiments/s04_ablation/diameter_layers/layers_L1"
    "experiments/s04_ablation/diameter_layers/layers_L2"
    "experiments/s04_ablation/diameter_layers/layers_L3"
    "experiments/s04_ablation/diameter_layers/layers_L6"
    "experiments/s04_ablation/diameter_layers/layers_L7"
)

CKPT_BASENAMES=(
    "best_gat_qcond_nl1.pt"
    "best_gat_qcond_nl2.pt"
    "best_gat_qcond_nl3.pt"
    "best_gat_qcond_nl6.pt"
    "best_gat_qcond_nl7.pt"
)

NAS_CKPT_DIR="/SSL_NAS/peoples/khj/thesis/checkpoints"

echo "========================================"
echo "  Wave 2 Proposal C — diameter_layers sweep"
echo "  Started: $(date -Iseconds)"
echo "  Phase 1: 5 GAT training runs"
echo "========================================"

for cfg in "${TRAIN_CFGS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "  Training ${cfg}  ($(date -Iseconds))"
    echo "----------------------------------------"
    CUDA_VISIBLE_DEVICES=0,1 conda run -n base python src/train_gat.py --config "configs/${cfg}.yaml" \
        || echo "[!] ${cfg} FAILED — continuing"
done

echo ""
echo "========================================"
echo "  Phase 1 완료 — 체크포인트 NAS 이동"
echo "========================================"
for ckpt in "${CKPT_BASENAMES[@]}"; do
    LOCAL="outputs/checkpoints/${ckpt}"
    NAS="${NAS_CKPT_DIR}/${ckpt}"
    if [ -f "${LOCAL}" ] && [ ! -L "${LOCAL}" ]; then
        mv "${LOCAL}" "${NAS}"
        ln -s "${NAS}" "${LOCAL}"
        echo "  moved ${ckpt} → NAS, symlinked"
    else
        echo "  skip ${ckpt} (absent or already symlink)"
    fi
done

echo ""
echo "========================================"
echo "  Phase 2 준비 — vLLM 재기동 필요!"
echo "========================================"
echo "  아래 명령으로 vLLM 재기동 후 수동으로 Phase 2 시작:"
echo "  nohup python -m vllm.entrypoints.openai.api_server \\"
echo "    --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \\"
echo "    --port 8000 --max-model-len 8192 \\"
echo "    --tensor-parallel-size 2 --gpu-memory-utilization 0.8 \\"
echo "    --enforce-eager > /tmp/vllm.log 2>&1 &"
echo ""
echo "  vLLM ready 확인 후 (/v1/models endpoint) 아래 수동 실행:"
echo "    bash scripts/run_wave2_proposal_c_phase2.sh"
echo ""
echo "========================================"
echo "  Phase 1 finished: $(date -Iseconds)"
echo "========================================"
