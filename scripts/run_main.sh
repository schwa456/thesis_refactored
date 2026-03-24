export CUDA_VISIBLE_DEVICES=2,3
nohup python src/main.py --config config/base_config.yaml > main_log.log &1>& 2