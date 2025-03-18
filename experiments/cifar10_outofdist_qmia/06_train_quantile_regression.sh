export CUDA_VISIBLE_DEVICES=4,5,6,7

# Train a quantile regression model for CIFAR-10 (excluding no classes).
#
# python quantile_regression/train.py \
# --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval.pt \
# --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
# --output_dir experiments/cifar10_outofdist_qmia/qr_logs/cifar10 \
# --n_epochs 50

# Train quantile regression models for CIFAR-10 excluding 0 to 9 randomly selected classes,
# The training data for each model consists of 2000 samples evenly distributed across the remaining classes.
#
# for i in {0..9}
# do
#     echo "Training quantile regression model excluding $i classes"
#     python quantile_regression/train.py \
#     --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval_exclude${i}_2000.pt \
#     --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
#     --output_dir experiments/cifar10_outofdist_qmia/qr_logs/cifar10_exclude${i} \
#     --n_epochs 50
# done

# Train quantile regression models for CIFAR-10 excluding each class (0 to 9) one at a time,
# The training data for each model consists of all remaining samples from the nonmember set (22500 samples).
#
# for i in {0..9}
# do
#     echo "Training quantile regression model excluding class $i"
#     python quantile_regression/train.py \
#     --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval_excludeclass${i}_22500.pt \
#     --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
#     --output_dir experiments/cifar10_outofdist_qmia/qr_logs/cifar10_excludeclass${i} \
#     --n_epochs 50
# done

# Train a quantile regression model for CIFAR-10 using CelebA as the nonmember data.
# 
python quantile_regression/train.py \
--nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/celeba/t_results.pt \
--member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
--output_dir experiments/cifar10_outofdist_qmia/qr_logs/cifar10_on_celeba \
--n_epochs 50
