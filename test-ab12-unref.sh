# -*- coding: UTF-8 -*-
stages=1
dataset=sceneflow # kitti   sceneflow
model=cf_fact3d # org cf_sepconv  cf_fact3d
mode=test # train   finetune    test
datapath=/ds-av/public_datasets/freiburg_sceneflow_subset/raw #/home/alghoul/myenv/FlyingThings3D # /home/alghoul/myenv/FlyingThings3D    /home/alghoul/myenv/kitti2015/training
datatype=2015
flip_vertical=1
epochs=100
lr=0.001
train_bsize=1 #10
test_bsize=1 #4
save_path=results/ab12_unref # the path of saving checkpoints and log
print_freq=30
checkpoint_save_thr=1
abs_thr=3
####load pretrained model
loadmodel=None #/pretrainedModels/org_2Stages_finetune_kitti2015-2020_10_19-21_44_15-epoch-1908-loss1-0.52-lossesSum-1.17-EarlyStopping-stereonet.pth
#### quantization
with_quant=0
quantWL=16 #10
quantFL=8 #8
####for testing
testFile=/sceneflow_model_cf_fact3d_1stages/checkpoints/cf_fact3d_fin_sceneflow-2015-2021_08_17-15_25_25-epoch-100-loss0-1.516-lossesSum-1.516.pth #/cf_fact3d_fin_sceneflow-2015-2021_01_24-18_06_08-epoch-100-loss3-0.515-lossesSum-3.316.pth
### for resuming
resume=0 # resume path
resumeFile=None #/kitti_model_cf_fact3d_3stages/checkpoints/cf_fact3d_fin_kitti-2015-2020_12_19-16_03_29-epoch-2-loss2-10.031-lossesSum-31.297.pth
####for model=cf_fact3d
model_bn=1
BN_1D_last=1
BN_1D=1
BN_2D=0
is_filter1_differ=0
filter1_kernels="331 113 113"
fact_kernels="331 113 113"
CUDA_VISIBLE_DEVICES=0 python  args_file.py --stages $stages --dataset $dataset --model $model --mode $mode --datapath $datapath \
                                            --datatype $datatype --flip_vertical $flip_vertical --epochs $epochs --lr $lr --train_bsize $train_bsize \
                                            --test_bsize $test_bsize --save_path $save_path --print_freq $print_freq \
                                            --checkpoint_save_thr $checkpoint_save_thr --abs_thr $abs_thr --loadmodel $loadmodel \
                                            --with_quant $with_quant --quantWL $quantWL --quantFL $quantFL --testFile $testFile \
                                            --resume $resume --resumeFile $resumeFile --model_bn $model_bn --BN_1D_last $BN_1D_last \
                                            --BN_1D $BN_1D --BN_2D $BN_2D --is_filter1_differ $is_filter1_differ --filter1_kernels $filter1_kernels \
                                            --fact_kernels $fact_kernels
##########
abs_thr=2
CUDA_VISIBLE_DEVICES=0 python  args_file.py --stages $stages --dataset $dataset --model $model --mode $mode --datapath $datapath \
                                            --datatype $datatype --flip_vertical $flip_vertical --epochs $epochs --lr $lr --train_bsize $train_bsize \
                                            --test_bsize $test_bsize --save_path $save_path --print_freq $print_freq \
                                            --checkpoint_save_thr $checkpoint_save_thr --abs_thr $abs_thr --loadmodel $loadmodel \
                                            --with_quant $with_quant --quantWL $quantWL --quantFL $quantFL --testFile $testFile \
                                            --resume $resume --resumeFile $resumeFile --model_bn $model_bn --BN_1D_last $BN_1D_last \
                                            --BN_1D $BN_1D --BN_2D $BN_2D --is_filter1_differ $is_filter1_differ --filter1_kernels $filter1_kernels \
                                            --fact_kernels $fact_kernels
#######
abs_thr=1
CUDA_VISIBLE_DEVICES=0 python  args_file.py --stages $stages --dataset $dataset --model $model --mode $mode --datapath $datapath \
                                            --datatype $datatype --flip_vertical $flip_vertical --epochs $epochs --lr $lr --train_bsize $train_bsize \
                                            --test_bsize $test_bsize --save_path $save_path --print_freq $print_freq \
                                            --checkpoint_save_thr $checkpoint_save_thr --abs_thr $abs_thr --loadmodel $loadmodel \
                                            --with_quant $with_quant --quantWL $quantWL --quantFL $quantFL --testFile $testFile \
                                            --resume $resume --resumeFile $resumeFile --model_bn $model_bn --BN_1D_last $BN_1D_last \
                                            --BN_1D $BN_1D --BN_2D $BN_2D --is_filter1_differ $is_filter1_differ --filter1_kernels $filter1_kernels \
                                            --fact_kernels $fact_kernels