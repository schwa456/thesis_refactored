export CUDA_VISIBLE_DEVICES=2,3
export HF_HOME="/SSL_NAS/peoples/khj/huggingface_cache"

nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
--port 8000 --max-model-len 8192 --tensor-parallel-size 2 \
--gpu-memory-utilization 0.8 \
--enforce-eager \
> logs/vllm_server_log.log &1>& 2