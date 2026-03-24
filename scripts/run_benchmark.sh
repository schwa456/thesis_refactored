export CUDA_VISIBLE_DEVICES=2,3
nohup python src/benchmark.py > logs/benchmark_log.log &1>& 2