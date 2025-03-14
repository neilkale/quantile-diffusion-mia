export CUDA_VISIBLE_DEVICES=4,5,6,7
python quantile_regression/scripts/train.py \
--logdir experiments/cifar10_outofdist_qmia/qr_logs/ \
--data_dir experiments/cifar10_outofdist_qmia/ \
--batch_size 1024 \
--lr 1e-4 \
--image_size 32 \
--num_in_channels 3 \
--channel_reduce 1 \
--num_classes 1 \
--dropout_rate 0