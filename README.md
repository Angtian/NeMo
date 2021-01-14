# NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation [ICLR-2021]
## Release Notes
The offical PyTorch implementation of NeMo, as robust 3D pose estimation method in feature level render-and-compare manner.

## Installation
The code is tested with python 3.7, PyTorch 1.5 and PyTorch3D 0.2.0.

### Clone the project and install requirements
```
git clone https://github.com/Angtian/NeMo.git
cd NeMo
pip install -r requirements.txt
```

## Running NeMo
We provide the scripts to train NeMo and conducts inference with NeMo on Pascal3D+ and Occluded Pascal3D+ datasets. For the details about Occluded Pascal3D+ please refer to this Github repo: [OccludedPASCAL3D](https://github.com/Angtian/OccludedPASCAL3D).

**Step 1: Prepare Datasets**  
Change the path to datasets in file PrepareData.sh, if you have already download these datasets. If not the script will automatically download datasets.
```
chmod +x PrepareData.sh
./PrepareData.sh
```


