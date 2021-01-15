# NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation [ICLR-2021]

## Release Notes
The offical PyTorch implementation of [NeMo](https://openreview.net/pdf?id=pmj131uIL9H), published on ICLR 2021. NeMo is a robust 3D pose estimation method via feature level render-and-compare.
![Example figure](https://github.com/Angtian/NeMo/blob/main/example.gif)
An dynamic example shows the pose optimization process of NeMo. Top-left: the input image; Top-right: A mesh superimposed on the input image in the predicted 3D pose. Bottom-left: The occluder localization result, where yellow is background, green is the non-occluded area of the object and red is the occluded area as predicted by NeMo. Bottomright: The loss landscape for each individual camera parameter respectively. The colored vertical lines demonstrate the final prediction and the ground-truth parameter is at center of x-axis.

## Installation
The code is tested with python 3.7, PyTorch 1.5 and PyTorch3D 0.2.0.

### Clone the project and install requirements
```
git clone https://github.com/Angtian/NeMo.git
cd NeMo
pip install -r requirements.txt
```

## Running NeMo
We provide the scripts to train NeMo and conducts inference with NeMo on Pascal3D+ and Occluded Pascal3D+ datasets. For more details about Occluded Pascal3D+ please refer to this Github repo: [OccludedPASCAL3D](https://github.com/Angtian/OccludedPASCAL3D).

**Step 1: Prepare Datasets**  
Change the path to datasets in file PrepareData.sh, if you have already download these datasets. Otherwise this script will automatically download datasets. Then run the following commands.
```
chmod +x PrepareData.sh
./PrepareData.sh
```

**Step 2: Training NeMo**  
Modify the settings in TrainNeMo.sh.  
GPUS: set avaliable GPUs for training depend on your machine. The standard setting use 7 gpus (6 for backbone, 1 for feature bank). If you have only 4 GPUs available, we suggest to turn off the "--sperate_bank" in training stage.   
MESH_DIMENSIONS: "single" or "multi".  
TOTAL_EPOCHS: The default setting is 800 epochs, which takes 3 to 4 days to train on an 8 GPUs machine. However, 400 training epochs could already yeild good accuracy. The final performance for the raw Pascal3D+ over train epochs:  
| Training Epochs| 200  | 400  | 600  | 800  |
|----------------|------|------|------|------|
| Acc Pi / 6     | 82.4 | 84.4 | 84.8 | 85.5 |
| Acc Pi / 18    | 57.1 | 59.2 | 59.6 | 60.2 |  

Then, run these commands:  
```
chmod +x TrainNeMo.sh
./TrainNeMo.sh
```

**Step 3: Inference with NeMo**  
The inference stage include feature extraction and pose optimization. The pose optimization conducts in a render-and-compare manner with gradient apply on camera pose iteratively, which will take some time to run (3-4 hours for each occlusion level on a 8 GPUS machine).  
To run the inference, you need first change the settings in InferenceNeMo.sh:
MESH_DIMENSIONS: Set to be same as the training stage.  
GPUS: Our implemention could either utilize 4 or 8 GPUs for the pose optimization. We will automatically distribute workloads over available GPUs and run the optimization in parallel.  
LOAD_FILE_NAME: Change this setting if you do not train 800 epochs, e.g. train NeMo for 400 -> "saved_model_%s_399.pth".  

Then, run these commands to conduct NeMo inference on unoccluded Pascal3D+:
```
chmod +x InferenceNeMo.sh
./InferenceNeMo.sh
```
To conduct inference on occluded-Pascal3D+:
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
