export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME="/SSL_NAS/peoples/khj/huggingface_cache"

# setsid: 새 session leader 생성 → SSH 끊김/부모 shell 종료에 완전 독립
# nohup:  SIGHUP 차단 (이중 보험)
# </dev/null: stdin 을 닫아 tty 의존성 제거
# &:      background
# disown: job table 에서 분리
setsid nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --port 8000 --max-model-len 16384 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    > logs/vllm_server_log.log 2>&1 < /dev/null &

disown 2>/dev/null || true
