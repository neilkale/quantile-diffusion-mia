export CUDA_VISIBLE_DEVICES=0,1,2,3
python SecMI/nk-secmia.py --logdir experiments/cifar10_outofdist_qmia/t_errors/ \
--member_dataset data/processed/cifar10/val \
--nonmember_dataset data/processed/cifar10/test \
--batch_size 1024 \
--k 10 \
--t_sec 100 \
--model_dir experiments/cifar10_outofdist_qmia/logs/