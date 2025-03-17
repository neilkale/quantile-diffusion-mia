export CUDA_VISIBLE_DEVICES=0,1,2,3
python SecMI/nk-secmia.py --logdir experiments/cifar10_outofdist_qmia/t_errors/cifar10/ \
--dataset data/temp/cifar10_trainval_combo \
--class_path data/processed/cifar10/labels.csv \
--batch_size 1024 \
--k 10 \
--t_sec 100 \
--model_dir experiments/cifar10_outofdist_qmia/diffusion_logs/cifar10 \
--model_name ckpt-step540000.pt