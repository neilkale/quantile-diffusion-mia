export CUDA_VISIBLE_DEVICES=4,5,6,7

# python quantile_regression/eval.py \
# --model_path experiments/cifar10_outofdist_qmia/qr_logs/cifar10_exclude9/model_epoch_10.pth \
# --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval.pt \
# --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
# --output_dir experiments/cifar10_outofdist_qmia/qr_results/cifar10_exclude9

for i in {0..9}
do
    echo "Evaluating quantile regression model excluding $i classes"
    python quantile_regression/eval.py \
    --model_path experiments/cifar10_outofdist_qmia/qr_logs/cifar10_exclude${i}/model_epoch_50.pth \
    --nonmember_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_eval.pt \
    --member_data_path experiments/cifar10_outofdist_qmia/t_errors/cifar10/t_results_train.pt \
    --output_dir experiments/cifar10_outofdist_qmia/qr_results/cifar10_exclude${i}
done
