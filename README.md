# NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation [ICLR-2021]

## Release Notes
The offical PyTorch implementation of [NeMo](https://openreview.net/pdf?id=pmj131uIL9H), published on ICLR 2021. NeMo achieves robust 3D pose estimation method by performing render-and-compare on the level of neural network features.
![Example figure](https://github.com/Angtian/NeMo/blob/main/example.gif)
The figure shows a dynamic example of the pose optimization process of NeMo. Top-left: the input image; Top-right: A mesh superimposed on the input image in the predicted 3D pose. Bottom-left: The occluder location as predicted by NeMo, where yellow is background, green is the non-occluded area and red is the occluded area of the object. Bottom-right: The loss landscape as a function of each camera parameter respectively. The colored vertical lines demonstrate the current prediction and the ground-truth parameter is at center of x-axis.

## Installation
The code is tested with python 3.7, PyTorch 1.5 and PyTorch3D 0.2.0.

### Clone the project and install requirements
```
git clone https://github.com/Angtian/NeMo.git
cd NeMo
pip install -r requirements.txt
```

## Running NeMo
We provide the scripts to train NeMo and to perform inference with NeMo on Pascal3D+ and the Occluded Pascal3D+ datasets. For more details about the OccludedPascal3D+ please refer to this Github repo: [OccludedPASCAL3D](https://github.com/Angtian/OccludedPASCAL3D).

**Step 1: Prepare Datasets**  
Set ENABLE_OCCLUDED to "true" if you need evaluate NeMo under partial occlusions. You can change the path to the datasets in the file PrepareData.sh, after downloading the data. Otherwise this script will automatically download datasets.  
Note: the default behavior of overwrite when generate 3D annotation is set to false, which means it will skips those existing annotation files. To regenerate those annotation files (such as fix corrupted data), the overwrite option need to be set as true.  
Then run the following commands:
```
chmod +x PrepareData.sh
./PrepareData.sh
```

**Step 2: Training NeMo**  
Modify the settings in TrainNeMo.sh.  
GPUS: set avaliable GPUs for training depending on your machine. The standard setting uses 7 gpus (6 for the backbone, 1 for the feature bank). If you have only 4 GPUs available, we suggest to turn off the "--sperate_bank" in training stage.   
MESH_DIMENSIONS: "single" or "multi".  
TOTAL_EPOCHS: The default setting is 800 epochs, which takes 3 to 4 days to train on an 8 GPUs machine. However, 400 training epochs could already yield good accuracy. The final performance for the raw Pascal3D+ over train epochs (SingleCuboid):  
| Training Epochs| 200  | 400  | 600  | 800  |
|----------------|------|------|------|------|
| Acc Pi / 6     | 82.4 | 84.4 | 84.8 | 85.5 |
| Acc Pi / 18    | 57.1 | 59.2 | 59.6 | 60.2 |  

Then, run these commands:  
```
chmod +x TrainNeMo.sh
./TrainNeMo.sh
```

**Step 2 (Alternative): Download Pretrained Model**  
Here we provide the pretrained NeMo Model and backbone for the "SingleCuboid" and "MultiCuboid" setting. Run the following commands to download the pretrained model (SingleCuboid):  
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1X1NCx22TFGJs108TqDgaPqrrKlExZGP-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1X1NCx22TFGJs108TqDgaPqrrKlExZGP-" -O NeMo_Single_799.zip
unzip NeMo_Single_799.zip
```
Download the pretrained model (MultiCuboid):  
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1e6bA6hpFEqZC59qsdl9otkkSHR1SYszz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1e6bA6hpFEqZC59qsdl9otkkSHR1SYszz" -O NeMo_Multi_799.zip
unzip NeMo_Multi_799.zip
```
The expected performance for the pretrained model (Unoccluded):
| SingleCuboid  |plane | bike  | boat  | bottle| bus   |   car | chair | table | mbike | sofa  | train | tv    | Mean  |
|---------------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Pi/6          | 81.1 | 79.8  |  69.0 |  87.6 |  88.2 |  98.5 |  86.2 |  73.9 |  83.7 |  95.4 |  81.3 |  82.0 |  85.5 |
| Pi/18         | 47.8 | 27.9  |  35.9 |  49.0 |  84.4 |  94.9 |  47.3 |  51.4 |  31.0 |  59.6 |  68.1 |  42.4 |  60.2 | 
| MedErr        | 10.6 | 16.8  |  15.4 |  10.3 |  3.2  |   3.2 |   10.5|  9.7  |  14.6 |  8.5  |  5.4  |  12.0 |  8.9  |   

| MultiCuboid |plane | bike |  boat |  bottle | bus |   car |   chair | table | mbike | sofa |  train | tv | Mean |
|---------------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Pi/6   | 78.1 | 81.2 |  64.9 |  88.2 |  94.4 |  98.6 |  87.4 |  81.5 |  84.2 |  96.0 |  90.6 |  85.9 |  87.0 |
| Pi/18  | 38.8 | 35.0 |  33.1 |  50.2 |  91.4 |  95.8 |  48.5 |  58.2 |  36.2 |  65.4 |  77.9 |  54.5 |  62.8 |
| MedErr | 13.0 | 14.1 |  16.4 |  10.0 |  2.6 |   2.9 |   10.4 |  8.3 |   13.9 |  7.3  |   4.7 |   9.0 |   8.5 |

**Step 3: Inference with NeMo**  
The inference stage includes feature extraction and pose optimization. The pose optimization conducts render-and-compare on the neural features w.r.t. the camera pose iteratively. This takes some time to run on the full dataset (3-4 hours for each occlusion level on a 8 GPU machine).  
To run the inference, you need to first change the settings in InferenceNeMo.sh:  
MESH_DIMENSIONS: Set to be same as the training stage.  
GPUS: Our implemention could either utilize 4 or 8 GPUs for the pose optimization. We will automatically distribute workloads over available GPUs and run the optimization in parallel.  
LOAD_FILE_NAME: Change this setting if you do not train 800 epochs, e.g. train NeMo for 400 -> "saved_model_%s_399.pth".  

Then, run these commands to conduct NeMo inference on unoccluded Pascal3D+:
```
chmod +x InferenceNeMo.sh
./InferenceNeMo.sh
```
To conduct inference on the occluded-Pascal3D+ (Note you need enable to create OccludedPascal3D+ dataset during data preparation):
```
./InferenceNeMo.sh FGL1_BGL1
./InferenceNeMo.sh FGL2_BGL2
./InferenceNeMo.sh FGL3_BGL3
```

## Citation
Please cite the following paper if you find this the code useful for your research/projects.
```
@inproceedings{wang2020NeMo,
title = {NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation},
author = {Angtian, Wang and Kortylewski, Adam and Yuille, Alan},
booktitle = {Proceedings International Conference on Learning Representations (ICLR)},
year = {2021},
}
```
