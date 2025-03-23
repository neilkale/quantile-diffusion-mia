export CUDA_VISIBLE_DEVICES=1,2

# Train a quantile regression model for CIFAR-10 (excluding no classes).
#
python quantile_regression_experimental/train.py \
--nonmember_data_path experiments/cifar10_improving_qmia/t_errors/cifar10/t100k10/t_results_eval.pt \
--member_data_path experiments/cifar10_improving_qmia/t_errors/cifar10/t100k10/t_results_train.pt \
--output_dir experiments/cifar10_improving_qmia/qr_logs/cifar10/t100k10/ \
--n_epochs 10 \
--n_quantiles 50 \
--class_cond \