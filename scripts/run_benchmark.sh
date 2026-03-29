export CUDA_VISIBLE_DEVICES=0,1
nohup python src/benchmark.py > logs/benchmark_log.log &1>& 2