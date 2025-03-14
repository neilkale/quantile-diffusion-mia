export CUDA_VISIBLE_DEVICES=4,5,6,7
python SecMI/main.py --train --logdir experiments/cifar10_outofdist_qmia/logs/ \
--dataset data/temp/cifar10_trainval_combo \
--img_size 32 --batch_size 128 \
--fid_cache experiments/cifar10_outofdist_qmia/stats/cifar10.train.npz \
--total_steps 800001
