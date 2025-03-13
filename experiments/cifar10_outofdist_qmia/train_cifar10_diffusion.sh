#!/bin/bash

TRAIN_FLAGS="--batch_size 32 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 --image_size 32 --learn_sigma True"
export CUDA_VISIBLE_DEVICES=3,4,6,7
export OPENAI_LOGDIR=/work3/nkale/ml-projects/quantile-diffusion-mia/experiments/cifar10_outofdist_qmia/logs
mpiexec -n 4 python guided_diffusion/scripts/image_train.py --data_dir data/temp/cifar10_trainval_combo/ $TRAIN_FLAGS