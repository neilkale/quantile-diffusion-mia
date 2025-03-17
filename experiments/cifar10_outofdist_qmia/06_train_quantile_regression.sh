export CUDA_VISIBLE_DEVICES=4,5,6,7

# python quantile_regression/train.py \
# --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval.pt \
# --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train_exclude9_2000.pt \
# --output_dir experiments/cifar10_outofdist_qmia/qr_logs/cifar10_exclude9 \
# --n_epochs 10

for i in {0..9}
do
    echo "Training quantile regression model excluding $i classes"
    python quantile_regression/train.py \
    --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval_exclude${i}_2000.pt \
    --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
    --output_dir experiments/cifar10_outofdist_qmia/qr_logs/cifar10_exclude${i} \
    --n_epochs 50
done