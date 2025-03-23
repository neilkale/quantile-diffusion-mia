export CUDA_VISIBLE_DEVICES=1

python SecMI/nk-secmia.py --logdir experiments/cifar10_improving_qmia/t_errors/cifar10/t200k10 \
--dataset data/temp/cifar10_trainval_combo \
--class_path data/processed/cifar10/labels.csv \
--batch_size 4096 \
--k 10 \
--t_sec 200 \
--model_dir experiments/cifar10_improving_qmia/diffusion_logs/cifar10 \
--model_name checkpoint.pt