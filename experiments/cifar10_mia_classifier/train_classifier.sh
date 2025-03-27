export CUDA_VISIBLE_DEVICES=0

# Train a quantile regression model for CIFAR-10 (excluding no classes).
#
python experiments/cifar10_mia_classifier/classifier/train.py \
--nonmember_data_path experiments/cifar10_mia_classifier/t_errors/cifar10/t100k10/t_results_eval.pt \
--member_data_path experiments/cifar10_mia_classifier/t_errors/cifar10/t100k10/t_results_train.pt \
--output_dir experiments/cifar10_mia_classifier/logs/cifar10/vanilla \
--n_epochs 400 \
--class_cond \
--batch_size 1024 \
--lr 0.005 \