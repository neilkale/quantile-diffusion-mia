export CUDA_VISIBLE_DEVICES=4,5,6,7
python SecMI/main.py --train --logdir experiments/cifar10_outofdist_qmia/diffusion_logs/cifar10_trainval/ \
--dataset data/temp/cifar10_trainval_combo \
--class_path data/processed/cifar10/labels.csv \
--img_size 32 --batch_size 128 \
--fid_cache experiments/cifar10_outofdist_qmia/stats/cifar10.train.npz \
--total_steps 800001 \
