export CUDA_VISIBLE_DEVICES=3
nohup python src/benchmark.py > logs/benchmark_baseline_gat_cls_multi_agent_log.log &1>& 2