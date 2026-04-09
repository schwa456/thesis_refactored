export CUDA_VISIBLE_DEVICES=0
nohup python src/benchmark.py > logs/benchmark_log.log &1>& 2