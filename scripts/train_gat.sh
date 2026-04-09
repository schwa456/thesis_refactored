export CUDA_VISIBLE_DEVICES=0,1

nohup python src/train_gat.py --config configs/training/train_gat_config.yaml > logs/train_gat_log.log &1>& 2