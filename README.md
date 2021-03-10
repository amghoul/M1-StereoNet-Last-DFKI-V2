<!-- preview: ctrl+k v multi-cursor: shift+alt+i Head -->
# GazaNet Model

## **Contents**
- [Model Structure](#model)
    - [Original Model](#org)
    - [GazaNet model](#gazanet)
- [Folder Structure](#FolderStructure)
- [Dataset Path & Structure](#dataset)
    - [FlyingThings3D Dataset](#FT3D)
    - [KITTI 2015 Dataset](#kitti2015)
    - [KITTI 2012 Dataset](#kitti2015)
- [Requirements to run the code](#reqs)
- [Main parameters to run the code](#args)
- [How to run the code](#run)
    - [Training the model](#train)
    - [Finetuning the model](#finetune)
    - [Resuming the model](#resume)
    - [Testing the model](#test)

---

## <a name= model>Model Structure</a>
### <a name= org>Original Model</a>
![Original model](orginalmodel.PNG)
### <a name= gazanet>GazaNet Model</a>
![GazaNet model (Our model)](gazanet.PNG)

---

## <a name= FolderStructure>Folder Structure</a>
```
📦StereoNet-Last-DFKI
 ┣ 📂dataloader
 ┃ ┣ 📜KITTILoader.py --> myImageFolder for KITTI 2015
 ┃ ┣ 📜KITTILoader1.py --> myImageFolder for KITTI 2012
 ┃ ┣ 📜KITTIloader2012.py --> Dataloader for KITTI 2012 path
 ┃ ┣ 📜KITTIloader2015.py --> Dataloader for KITTI 2015 path
 ┃ ┣ 📜listflowfile.py --> Dataloader for FT3D path
 ┃ ┣ 📜preprocess.py
 ┃ ┣ 📜readpfm.py
 ┃ ┣ 📜SecenFlowLoader1.py
 ┃ ┣ 📜SecenFlowLoaderMy.py --> myImageFolder for FT3D
 ┣ 📂models
 ┃ ┣ 📜factorizer.py --> prepare factorizing of conv3d parameters
 ┃ ┣ 📜spatioTemporalConv.py
 ┃ ┣ 📜spatioTemporalConv_General.py --> Actual implementation of factorizing cost filtering layer
 ┃ ┣ 📜StereoNet_Multi.py --> for Original StereoNet model
 ┃ ┣ 📜StereoNet_Multi_FactorizedConv3D.py --> for Our model (GazaNet)
 ┃ ┣ 📜StereoNet_Multi_FactorizedConv3D_temp.py
 ┃ ┣ 📜StereoNet_Multi_SepConv.py
 ┃ ┗ 📜StereoNet_single.py
 ┣ 📂pretrainedModels --> contains pretrained models on FT3d
 ┃ ┣ 📜 cf_fact3d_fin_sceneflow-2015-2021_01_24-18_06_08-epoch-100-loss3-0.515-lossesSum-3.316.pth --> pretrained model for GazaNet model on FT3D
 ┃ ┗ 📜 org_fin_sceneflow-2015-2021_01_24-14_48_42-epoch-68-loss3-0.333-lossesSum-2.302.pth --> pretrained model for original model on FT3D
 ┣ 📂results_V18 --> contains results of running the code
 ┃ ┣ 📜cf_fact3d_fin_kitti-2012-2021_01_30-17_15_18-epoch-3200-loss3-0.148-lossesSum-0.74.pth --> checkpoint after finetuning on KITTI 2012
 ┃ ┣ 📜cf_fact3d_fin_kitti-2015-2021_01_31-02_59_50-epoch-2800-loss3-0.115-lossesSum-0.616.pth --> checkpoint after finetuning on KITTI 2015
 ┃ ┣ 📜cf_fact3d_fin_sceneflow-2015-2021_01_24-18_06_08-epoch-100-loss3-0.515-lossesSum-3.316.pth --> checkpoint after training on FT3d using GazaNet model
 ┃ ┗ 📜org_fin_sceneflow-2015-2021_01_24-14_48_42-epoch-68-loss3-0.333-lossesSum-2.302.pth --> checkpoint after training on FT3d using original model
 ┣ 📂utils
 ┃ ┣ 📜disp_to_color.py --> convert disparity image to colored image
 ┃ ┣ 📜FinalQuant.py --> for quantizing the weights
 ┃ ┣ 📜logger.py
 ┃ ┣ 📜preprocess.py
 ┃ ┣ 📜readpfm.py
 ┃ ┣ 📜utils.py --> contains the loss functions
 ┃ ┗ 📜__init__.py
 ┣ 📜args_file.py --> This file contains the arguments or parameters that required to run the code
 ┣ 📜finetune_2012_V18.sh --> script to finetune the pretrained model on KITTI 2012
 ┣ 📜finetune_2015_V18.sh --> script to finetune the pretrained model on KITTI 2015
 ┣ 📜main_file.py --> main file code which called from args_file.py
 ┣ 📜README.md
 ┣ 📜resume_V18.sh --> script to resume training
 ┣ 📜run_V18.sh --> script to train the model on FT3d from scratch
 ┣ 📜test_kitti_2012_V18.sh --> script to test the model on KITTI 2012
 ┣ 📜test_kitti_2015_V18.sh --> script to test the model on KITTI 2015
 ┣ 📜test_kitti_V18.sh --> script to test the model on KITTI
 ┗ 📜test_V18.sh --> script to test the model on FT3d
 ```

---

## <a name= reqs>Requirements to run the code</a>
- The main packages required to the this code are:
    - Python ==3.6
    - torchsummary==1.5.1
    - torchtext==0.7.0
    - torchvision==0.7.0+cu101
    - apex==0.1
    - matplotlib==3.2.2
    - ninja==1.10.0.post2
    - numpy==1.19.0
    - opencv-python==4.2.0.34
    - pandas==1.1.0
    - Pillow==4.1.1
    - pytorch-memlab==0.2.1
    - pytorch-nemo==0.0.7
    - qtorch==0.2.0
    - termcolor==1.1.0
    - texttable==1.6.3
    - torch==1.6.0+cu101
    - torchaudio==0.6.0
    - torchprof==1.1.1    

---

## <a name= dataset>Dataset Path & Structure</a>
- ### <a name= FT3D>FlyingThings3D Dataset</a>
    **Dataset root path**: filepath= /home/alghoul/myenv/FlyingThings3D
    - **Training dataset path** :    
        - **image_left** = filepath+ /train/image_clean/left
        - **image_right** = filepath+ /train/image_clean/right
        - **disp_L** = filepath+ /train/disparity/left/
        - **disp_R** = filepath+ /train/disparity/right/
        - **disp_L_OCC** = filepath+ /train/disparity_occlusions/left/
    - **Testing dataet path**:
        - **image_left** = filepath+ /val/image_clean/left/
        - **image_right** = filepath+ /val/image_clean/right/
        - **disp_L** = filepath+ /val/disparity/left/
        - **disp_R** = filepath+ /val/disparity/right/
        - **disp_L_OCC** = filepath+ /val/disparity_occlusions/left/

- ### <a name= kitti2015>KITTI 2015 Dataset</a>
    **Dataset root path**: filepath= /home/alghoul/myenv/kitti2015/training
    - **Training and testing dataset**:
        - left_fold  = filepath + /image_2/
        - right_fold = filepath + /image_3/
        - disp_L = filepath + /disp_occ_0/
        - disp_R = filepath + /disp_occ_1/
        - disp_L_noc = filepath + /disp_noc_0/
        - mask_obj_map = filepath + /obj_map/

- ### <a name= kitti2012>KITTI 2012 Dataset</a>
    **Training root path**: datapath=/home/alghoul/myenv/kitti2012/training
    - **Training and testing dataset**:
        - left_fold  = filepath + /colored_0/
        - right_fold = filepath + /colored_1/
        - disp_L   = filepath + /disp_occ/
        - disp_L_noc = filepath + /disp_noc/

---

## <a name= args>Main parameters to run the code</a>
- These paramters are the main parameters to run the code:
    - ###Selecting the dataset
        - dataset: --> {sceneflow, kitti} To select the dataset
        - datapath: {/home/alghoul/myenv/FlyingThings3D,/home/alghoul/myenv/kitti2015/training,/home/alghoul/myenv/kitti2012/training} To select the root path of the datset. Note: This paremeters should match the the value of the "dataset" parameter
    - stages: --> {1,2,3,4} Number of stages
    - model: --> {org, cf_fact3d, cf_sepconv}  to select the model to run it. org: original model, cf_fact3d: GazaNet model (ourmodel), cf_sepconv: eperable conolution model (not used)
    - mode: {train,finetune,test} To select mode of running
    - epochs: Number of epochs
    - lr: Learning rate vfalue {0.001}
    - train_bsize: Train batch size {4}
    - test_bsize: Test batch size {4}
    - save_path: {results}  the path of saving checkpoints, images, and log
    - print_freq: {30} How often epochs print the result on screen while training or finetuning
    - checkpoint_save_thr= {1} How often to save checkpoints
    - abs_thr={1,2,3} Absolute threshold to caluclate the loss
    - ####load pretrained model
        - loadmodel: {None,/pretrainedModels/org_2Stages_finetune_kitti2015-2020_10_19-21_44_15-epoch-1908-loss1-0.- 52-lossesSum-1.17-EarlyStopping-stereonet.pth} --> Load pretrained model for either finetuning or resuming the model. We save the pretrained model in the "pretrainedModels" folder on the root
    - ####quantization
        - with_quant: {0,1) --> 0: no quantization, 1: quatizing the model 
        - quantWL: {16} --> the whole word length in quantization
        - quantFL: {8} --> the Float point length in quantization
        - ####for testing
    testFile: {None, /checkpoint_finetune_kitti2015-2020_12_04-21_28_07-epoch-4000-loss1-0.317-lossesSum-4.- 391.pth} --> the path of the checkpoint used if we put the value of mode = test
    - ###for resuming
        - resume: {0,1} --> 0: no resuming, 1: enable resuming
        - resumeFile: {None, /kitti_model_cf_fact3d_3stages/checkpoints/- cf_fact3d_fin_kitti-2015-2020_12_19-16_03_29-epoch-2-loss2-10.031-lossesSum-31.297.pth} --> The path of the resume file if enable resume
    - ####for GazaNet model=cf_fact3d (our model)
        - nosubspace:{1,2,...} --> number of conv2d layers used in factorizing
        - BN_1D: {0,1} --> 0: don't use batch normalized with Conv1D layer in factorizing, 1: use batch normalized with Conv1D layer in factorizing
        - BN_2D: {0,1} --> 0: don't use batch normalized with Conv2D layer in factorizing, 1: use batch normalized with Conv2D layer in factorizing

---

## <a name= run>How to run the code</a>

### <a name= train>Training the model</a>
- To train the model from scratch using FT3D run "run_V17.sh" script
- Note: refer to the [Main parameters to run the code](#args) for more help
---

### <a name= finetune>Finetuning the model</a>
- To finetune the model using A pretrained model you can run the script:
    - finetune_2012_V18.sh --> to finetune on KITTI 2012
    - finetune_2015_V18.sh --> to finetune on KITTI 2015
- Note: refer to the [Main parameters to run the code](#args) for more help
---

### <a name= resume>Resuming the model</a>
- To resume the model you can run "resume_V18.sh" script

---

### <a name= test>Testing the model</a>
- To test the model you can run the following script:
    - test_V18.sh --> test the model on FT3D dataset
    - test_kitti_2015_V18.sh --> test the model on KITTI 2015 dataset
    - test_kitti_2012_V18.sh --> test the model on KITTI 2012 dataset