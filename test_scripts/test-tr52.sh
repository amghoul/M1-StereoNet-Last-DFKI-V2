# -*- coding: UTF-8 -*-
stages=4
dataset=sceneflow # kitti   sceneflow
model=cf_fact3d # org cf_sepconv  cf_fact3d
mode=test # train   finetune    test
datapath=/ds-av/public_datasets/freiburg_sceneflow_subset/raw # /netscratch/alkoutayni/StereoMatching/Datasets/FlyingThings3D    /netscratch/alkoutayni/StereoMatching/Datasets/kitti2015/training  /netscratch/alkoutayni/StereoMatching/Datasets/kitti2012/training
datatype=2015
flip_vertical=1
epochs=600
lr=0.001
train_bsize=4 #10
test_bsize=1 #4
save_path=results/ablation11
print_freq=30
checkpoint_save_thr=20
abs_thr=3
####load pretrained model
loadmodel=None #/pretrainedModels/org_2Stages_finetune_kitti2015-2020_10_19-21_44_15-epoch-1908-loss1-0.52-lossesSum-1.17-EarlyStopping-stereonet.pth
#### quantization
with_quant=0
quantWL=16 #10
quantFL=8 #8
####for testing
testFile=/sceneflow_model_cf_fact3d_4stages/checkpoints/best_checkpoints/best-tr52-ab11--epoch_286.pth #None #/checkpoint_finetune_kitti2015-2020_12_04-21_28_07-epoch-4000-loss1-0.317-lossesSum-4.391.pth
### for resuming
resume=0 # resume path
resumeFile=None #/sceneflow_model_cf_fact3d_4stages/checkpoints/cf_fact3d_fin_sceneflow-2015-2021_04_07-14_41_20-epoch-100-loss3-0.503-lossesSum-3.351.pth
####for model=cf_fact3d
model_bn=1
BN_1D_last=1
BN_1D=1
BN_2D=0
is_filter1_differ=0
filter1_kernels="331 311 311"
fact_kernels="331 311 311"
####
max_checkpoints_to_save=20
threshold_overfit_epochs=40

python3  args_file.py --stages $stages --dataset $dataset --model $model --mode $mode --datapath $datapath \
                                            --datatype $datatype --flip_vertical $flip_vertical --epochs $epochs --lr $lr --train_bsize $train_bsize \
                                            --test_bsize $test_bsize --save_path $save_path --print_freq $print_freq \
                                            --checkpoint_save_thr $checkpoint_save_thr --abs_thr $abs_thr --loadmodel $loadmodel \
                                            --with_quant $with_quant --quantWL $quantWL --quantFL $quantFL --testFile $testFile \
                                            --resume $resume --resumeFile $resumeFile --model_bn $model_bn --BN_1D_last $BN_1D_last \
                                            --BN_1D $BN_1D --BN_2D $BN_2D --is_filter1_differ $is_filter1_differ --filter1_kernels $filter1_kernels \
                                            --fact_kernels $fact_kernels --max_checkpoints_to_save $max_checkpoints_to_save --threshold_overfit_epochs $threshold_overfit_epochs