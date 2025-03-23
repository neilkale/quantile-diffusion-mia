# Train a quantile regression model for a subset of CIFAR-10 with 22500 samples drawn from all classes (0 to 9).
#
export CUDA_VISIBLE_DEVICES=1
nohup python quantile_regression/train.py \
--nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval_22500.pt \
--member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
--output_dir experiments/cifar10_outofdist_qmia/qr_logs/cifar10_22500 \
--n_epochs 100 &

# Train quantile regression models for CIFAR-10 excluding each class (0 to 9) one at a time,
# The training data for each model consists of all remaining samples from the nonmember set (22500 samples).
#
for i in {0..9}
do
    export CUDA_VISIBLE_DEVICES=$((i % 2))
    echo "Training quantile regression model excluding class $i"
    nohup python quantile_regression/train.py \
    --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval_excludeclass${i}_22500.pt \
    --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
    --output_dir experiments/cifar10_outofdist_qmia/qr_logs/cifar10_excludeclass${i} \
    --n_epochs 100 & 
done
