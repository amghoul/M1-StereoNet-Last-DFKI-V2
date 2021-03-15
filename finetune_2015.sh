# -*- coding: UTF-8 -*-
stages=4
dataset=kitti # kitti   sceneflow
datatype=2015 # 2015    2012
model=cf_fact3d # org cf_sepconv  cf_fact3d
mode=finetune # train   finetune    test
datapath=/home/alghoul/myenv/kitti2015/training # /home/alghoul/myenv/FlyingThings3D    /home/alghoul/myenv/kitti2015/training  /home/alghoul/myenv/kitti2012/training
flip_vertical=1
epochs=3500
lr=0.001
train_bsize=4 #10
test_bsize=4 #4
save_path=results # the path of saving checkpoints and log
print_freq=1
checkpoint_save_thr=100
abs_thr=3
####load pretrained model
loadmodel=/pretrainedModels/cf_fact3d_fin_sceneflow-2015-2021_01_24-18_06_08-epoch-100-loss3-0.515-lossesSum-3.316.pth #/pretrainedModels/org_2Stages_finetune_kitti2015-2020_10_19-21_44_15-epoch-1908-loss1-0.52-lossesSum-1.17-EarlyStopping-stereonet.pth
#### quantization
with_quant=0
quantWL=16 #10
quantFL=8 #8
####for testing
testFile=None #/checkpoint_finetune_kitti2015-2020_12_04-21_28_07-epoch-4000-loss1-0.317-lossesSum-4.391.pth
### for resuming
resume=0 # resume path
resumeFile=None #/kitti_model_cf_fact3d_3stages/checkpoints/cf_fact3d_fin_kitti-2015-2020_12_19-16_03_29-epoch-2-loss2-10.031-lossesSum-31.297.pth
####for model=cf_fact3d
model_bn=1
BN_1D_last=1
BN_1D=1
BN_2D=0
is_filter1_differ=0
filter1_kernels="311 311"
fact_kernels="331 311 311"
CUDA_VISIBLE_DEVICES=0 python  args_file.py --stages $stages --dataset $dataset --model $model --mode $mode --datapath $datapath \
                                            --datatype $datatype --flip_vertical $flip_vertical --epochs $epochs --lr $lr --train_bsize $train_bsize \
                                            --test_bsize $test_bsize --save_path $save_path --print_freq $print_freq \
                                            --checkpoint_save_thr $checkpoint_save_thr --abs_thr $abs_thr --loadmodel $loadmodel \
                                            --with_quant $with_quant --quantWL $quantWL --quantFL $quantFL --testFile $testFile \
                                            --resume $resume --resumeFile $resumeFile --model_bn $model_bn --BN_1D_last $BN_1D_last \
                                            --BN_1D $BN_1D --BN_2D $BN_2D --is_filter1_differ $is_filter1_differ --filter1_kernels $filter1_kernels \
                                            --fact_kernels $fact_kernels