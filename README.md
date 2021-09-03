<!-- preview: ctrl+k v multi-cursor: shift+alt+i Head -->
# Single-Scale Disparity Estimation (SSDE) Model

## **Contents**
- [Single-Scale Disparity Estimation (SSDE) Model](#single-scale-disparity-estimation-ssde-model)
  - [**Contents**](#contents)
  - [Model Structure](#model-structure)
    - [Original Model](#original-model)
    - [SSDE Model](#ssde-model)
  - [Folder Structure](#folder-structure)
  - [Requirements to run the code](#requirements-to-run-the-code)
  - [Dataset Path \& Structure](#dataset-path--structure)
  - [Main parameters to run the code](#main-parameters-to-run-the-code)
  - [Project Arguments Reference (Project 2)](#project-arguments-reference-project-2)
    - [1. Selecting the Dataset](#1-selecting-the-dataset)
    - [2. Load Pretrained Model](#2-load-pretrained-model)
    - [3. Quantization](#3-quantization)
    - [4. Resuming Training](#4-resuming-training)
    - [5. SSDE Model Configuration (cf\_fact3d)](#5-ssde-model-configuration-cf_fact3d)
    - [6. Stereo Architecture Hyperparameters](#6-stereo-architecture-hyperparameters)
    - [7. Execution \& Mode Settings](#7-execution--mode-settings)
    - [8. Training \& Optimization Hyperparameters](#8-training--optimization-hyperparameters)
    - [9. Logging, Evaluation \& Checkpoints](#9-logging-evaluation--checkpoints)
  - [How to run the code](#how-to-run-the-code)
    - [Training the model](#training-the-model)
    - [Finetuning the model](#finetuning-the-model)
    - [Resuming the model](#resuming-the-model)
    - [Testing the model](#testing-the-model)

---

## <a name= model>Model Structure</a>
### <a name= org>Original Model</a>
![Original model](Readme_images/orginalmodel.PNG)
### <a name= SSDE>SSDE Model</a>
![SSDE model (Our first model)](Readme_images/SSDE-details.png)

---

## <a name= FolderStructure>Folder Structure</a>
```
📦M1-SSDE
 ┣ 📂dataloader
 ┃ ┣ 📜KITTILoader.py --> myImageFolder for KITTI 2015
 ┃ ┣ 📜KITTILoader1.py --> myImageFolder for KITTI 2012
 ┃ ┣ 📜KITTIloader2012.py --> Dataloader for KITTI 2012 path
 ┃ ┣ 📜KITTIloader2015.py --> Dataloader for KITTI 2015 path
 ┃ ┣ 📜listflowfile.py --> Dataloader for FT3D path
 ┃ ┣ 📜preprocess.py
 ┃ ┣ 📜readpfm.py
 ┃ ┣ 📜SecenFlowLoaderMy.py --> myImageFolder for FT3D
 ┃ ┗ 📜__init__.py
 ┣ 📂models
 ┃ ┣ 📜factorizer.py --> prepare factorizing of conv3d parameters
 ┃ ┣ 📜memory.py
 ┃ ┣ 📜plot.py
 ┃ ┣ 📜spatioTemporalConv_General.py --> Actual implementation of factorizing cost filtering layer
 ┃ ┣ 📜StereoNet_Multi.py --> for Original StereoNet model
 ┃ ┣ 📜StereoNet_Multi_FactorizedConv3D.py --> for Our model (SSDE)
 ┃ ┗ 📜StereoNet_Multi_SepConv.py
 ┣ 📂pretrainedModels --> contains pretrained models on FT3d
 ┣ 📂Readme_images
 ┃ ┣ 📜SSDE.PNG
 ┃ ┗ 📜orginalmodel.PNG
 ┣ 📂results --> contains results of running the code
 ┣ 📂utils
 ┃ ┣ 📜disp_to_color.py --> convert disparity image to colored image, save images
 ┃ ┣ 📜FinalQuant.py --> for quantizing the weights
 ┃ ┣ 📜folder_logs_init.py --> for creating the required result folders, initialzing logs 
 ┃ ┣ 📜logger.py
 ┃ ┣ 📜merge_freeze_model.py --> for merging and freezing a model
 ┃ ┣ 📜readpfm.py
 ┃ ┣ 📜save_load.py --> load dataset, load and save checkpoint, load qauntized model, save losses functions
 ┃ ┣ 📜stop_early.py --> for stop_early 
 ┃ ┣ 📜test.py --> main test function
 ┃ ┣ 📜train.py --> main train, adjust_learning_rate, and test_from_training functions
 ┃ ┣ 📜utils.py --> contains the loss, outliers functions
 ┃ ┗ 📜__init__.py
 ┣ 📜args_file.py --> This file contains the arguments or parameters that required to run the code
 ┣ 📜finetune_2012.sh --> script to finetune the pretrained model on KITTI 2012
 ┣ 📜finetune_2015.sh --> script to finetune the pretrained model on KITTI 2015
 ┣ 📜dataclasses_models.py --> contains the dataclasses models
 ┣ 📜main_file.py --> main file code which called from args_file.py
 ┣ 📜README.md
 ┣ 📜resume.sh --> script to resume training
 ┣ 📜run.sh --> script to train the model on FT3d from scratch
 ┣ 📜test.sh --> script to test the model on FT3d
 ┣ 📜test_kitti_2012.sh --> script to test the model on KITTI 2012
 ┗ 📜test_kitti_2015.sh --> script to test the model on KITTI 2015
```

---

## <a name= reqs>Requirements to run the code</a>
- The main packages required to the this code are:
    - Python==3.6
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
  These paramters are the main parameters to run the code:

  ## Project Arguments Reference (Project 2)

### 1. Selecting the Dataset
* `--dataset`: `{sceneflow, kitti}` (default: `"kitti"`) -> Select dataset to work on.
* `--datapath`: (default: `'/home/alghoul/myenv/kitti2015/training/'`) -> Select the root path of the dataset. Note: This parameter should match the value of the `dataset` parameter.
* `--datatype`: `{2012, 2015}` (default: `'2015'`) -> Select KITTI version 2012 or 2015.
* `--flip_vertical`: `{0, 1}` (default: `0`) -> This is for the KITTI dataset to enable flipping the image up-down (1) or no flipping (0).

### 2. Load Pretrained Model
* `--loadmodel`: (default: `None`) -> Path to load a pretrained model checkpoint.

### 3. Quantization
* `--with_quant`: `{0, 1}` (default: `0`) -> Finetuning or testing the model with quantization (1) or not (0).
* `--quantWL`: (default: `10`) -> Number of whole bits in quantization.
* `--quantFL`: (default: `7`) -> Number of float bits in quantization.

### 4. Resuming Training
* `--resume`: `{0, 1}` (default: `0`) -> Resume training from a specific epoch and checkpoint.
* `--resumeFile`: (default: `None`) -> The full path of the checkpoint file to resume from.

### 5. SSDE Model Configuration (cf_fact3d)
* `--model`: `{org, cf_sepconv, cf_fact3d}` (default: `'org'`) -> Choose the core model architecture:
    * `org`: Original StereoNet
    * `cf_sepconv`: StereoNet with separable convolutions in the cost filtering layer
    * `cf_fact3d`: Factorizing Conv3D in the cost filtering layer (Our Model)
* `--model_bn`: `{0, 1}` (default: `1`) -> Enable Batch Normalization (BN) layers across the entire model. 1: Enable BN according to `BN_1D_last` or (`BN_1D` and `BN_2D`) values. 0: Disable BN regardless of those parameters.
* `--BN_1D`: `{0, 1}` (default: `1`) -> 1: Use Batch Normalization with Conv1D after factorizing Conv3D.
* `--BN_2D`: `{0, 1}` (default: `0`) -> 1: Use Batch Normalization with Conv2D after factorizing Conv3D.
* `--BN_1D_last`: `{0, 1}` (default: `1`) -> 1: Enforce BN for the last conv1D in factorized conv3d regardless of `BN_1D` and `BN_2D` values. 0: Use standard `BN_1D` and `BN_2D` settings.
* `--fact_kernels`: (default: `[[1,1,3],[3,1,1]]`) -> Kernel values for each conv1d or conv2d layer after factorizing conv3d.
* `--is_filter1_differ`: `{0, 1}` (default: `0`) -> 1: Use different conv layers in the first component of the cost filtering layer.
* `--filter1_kernels`: (default: `[[3,1,1],[3,1,1]]`) -> Enabled if `is_filter1_differ=1`. Kernel values for each conv1d or conv2d layer after factorizing conv3d in the first component of the cost filtering layer.

### 6. Stereo Architecture Hyperparameters
* `--maxdisp`: (default: `192`) -> Maximum disparity range.
* `--stages`: `{1, 2, 3, 4}` (default: `2`) -> The stage number of disparity refinement.

### 7. Execution & Mode Settings
* `--mode`: `{train, finetune, test}` (default: `'finetune'`) -> Execution mode selection.
* `--testFile`: (default: `None`) -> Path to the specific evaluation/test file.
* `--gpu`: (default: `'1'`) -> Target GPU device ID.
* `--no-cuda`: (default: `False`) -> Set flag to enable CPU training/testing instead of CUDA.
* `--seed`: (default: `1`) -> Random seed for reproducibility.

### 8. Training & Optimization Hyperparameters
* `--epochs`: (default: `100`) -> Total number of epochs to train.
* `--train_bsize`: (default: `1`) -> Batch size used for training.
* `--test_bsize`: (default: `1`) -> Batch size used for testing/evaluation.
* `--lr`: (default: `1e-3`) -> Base learning rate.
* `--ch_lr_after`: (default: `5`) -> Number of epochs to maintain the initial learning rate before decay rules trigger.
* `--momentum`: (default: `0.9`) -> Momentum factor for optimizer.
* `--weight_decay` or `--wd`: (default: `1e-4`) -> Weight decay (L2 penalty).
* `--stepsize`: (default: `1`) -> Learning rate step size.
* `--gamma` or `--gm`: (default: `0.6`) -> Learning rate decay factor (Gamma).
* `--itersize`: (default: `1`) -> Iteration size gradient accumulation factor.

### 9. Logging, Evaluation & Checkpoints
* `--loss_weights`: (default: `[1.0, 1.0, 1.0, 1.0, 1.0]`) -> Sequence of weights assigned to individual multi-stage losses.
* `--abs_thr`: (default: `3.0`) -> Absolute error rate threshold for metrics calculation.
* `--rel_thr`: (default: `0.05`) -> Relative error rate threshold for metrics calculation.
* `--number_of_thr`: (default: `1`) -> Number of distinct threshold evaluation metrics to compute.
* `--save_path`: (default: `'results'`) -> Destination path directory for saving checkpoints, logs, and outputs.
* `--print_freq`: (default: `1`) -> Console log printing frequency during iterations.
* `--checkpoint_save_thr`: (default: `100`) -> Frequency interval (in epochs) to trigger checkpoint saving.
* `--max_checkpoints_to_save`: (default: `10`) -> Limit on the number of top-performing checkpoints to retain.
* `--threshold_overfit_epochs`: (default: `50`) -> Number of sequential epochs where training loss starts increasing to consider the model has reached an overfit state.  
---

## <a name= run>How to run the code</a>

### <a name= train>Training the model</a>
- To train the model from scratch using FT3D run "run.sh" script
- Note: refer to the [Main parameters to run the code](#args) for more help
---

### <a name= finetune>Finetuning the model</a>
- To finetune the model using A pretrained model you can run the script:
    - finetune_2012_V18.sh --> to finetune on KITTI 2012
    - finetune_2015_V18.sh --> to finetune on KITTI 2015
- Note: refer to the [Main parameters to run the code](#args) for more help
---

### <a name= resume>Resuming the model</a>
- To resume the model you can run "resume.sh" script

---

### <a name= test>Testing the model</a>
- To test the model you can run the following script:
    - test_V18.sh --> test the model on FT3D dataset
    - test_kitti_2015.sh --> test the model on KITTI 2015 dataset
    - test_kitti_2012.sh --> test the model on KITTI 2012 dataset