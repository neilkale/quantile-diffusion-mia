export CUDA_VISIBLE_DEVICES=4,5,6,7

# python quantile_regression/eval.py \
# --model_path experiments/cifar10_outofdist_qmia/qr_logs/cifar10/model_epoch_200.pth \
# --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval.pt \
# --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
# --output_dir experiments/cifar10_outofdist_qmia/qr_results/cifar10_e200

python quantile_regression/eval.py \
--model_path experiments/cifar10_outofdist_qmia/qr_logs/cifar10_22500/model_epoch_100.pth \
--nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval.pt \
--member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
--output_dir experiments/cifar10_outofdist_qmia/qr_results/cifar10_22500_e100

# for i in {0..9}
# do
#     echo "Evaluating quantile regression model excluding $i classes"
#     python quantile_regression/eval.py \
#     --model_path experiments/cifar10_outofdist_qmia/qr_logs/cifar10_exclude${i}/model_epoch_50.pth \
#     --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval.pt \
#     --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
#     --output_dir experiments/cifar10_outofdist_qmia/qr_results/cifar10_exclude${i}
# done

for i in {0..9}
do
    echo "Evaluating quantile regression model excluding class $i"
    python quantile_regression/eval.py \
    --model_path experiments/cifar10_outofdist_qmia/qr_logs/cifar10_excludeclass${i}/model_epoch_100.pth \
    --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval.pt \
    --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
    --output_dir experiments/cifar10_outofdist_qmia/qr_results/cifar10_excludeclass${i}_e100
done

# python quantile_regression/eval.py \
# --model_path experiments/cifar10_outofdist_qmia/qr_logs/cifar10_on_celeba/model_epoch_200.pth \
# --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval.pt \
# --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
# --output_dir experiments/cifar10_outofdist_qmia/qr_results/cifar10_on_celeba_e200 \
# --n_quantiles 100

# python quantile_regression/eval.py \
# --model_path experiments/cifar10_outofdist_qmia/qr_logs/cifar10_on_cifar100/model_epoch_200.pth \
# --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval.pt \
# --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
# --output_dir experiments/cifar10_outofdist_qmia/qr_results/cifar10_on_cifar100_e200
