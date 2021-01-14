# NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation [ICLR-2021]
## Release Notes
The offical PyTorch implementation of NeMo, a robust 3D pose estimation method in feature level render-and-compare manner.

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
Modify the variables in TrainNeMo.sh.  
GPUS: set avaliable GPUs for training depend on your machine. The standard setting use 7 gpus (6 for backbone, 1 for feature bank). If you have only 4 GPUs available, we suggest to turn off the "--sperate_bank" in training stage.   
MESH_DIMENSIONS: "single" or "multi".  
TOTAL_EPOCHS: default to 800 epochs, which takes 3 to 4 days on an 8 GPUs machine. The final performance for the raw Pascal3D+ over train epochs:  
| Training Epochs| 200  | 400  | 600  | 800  |
|----------------|------|------|------|------|
| Acc Pi / 6     | 82.4 | 84.4 | 84.8 | 85.5 |
| Acc Pi / 18    | 57.1 | 59.2 | 59.6 | 60.2 |  

Then, run these commands:  
```
chmod +x TrainNeMo.sh
./TrainNeMo.sh
```
