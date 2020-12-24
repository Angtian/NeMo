#!/bin/bash
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

ROOT=$(get_abs_filename "./")
DATAROOT="${ROOT}/data"
EXPROOT="${ROOT}/exp"

MESH_DIMENSIONS="single"
GPUS="0, 1, 2, 3, 4, 5, 6"

PATH_PASCAL3DP="${DATAROOT}/PASCAL3D+_release1.1/"
PATH_CACHE_TRAINING_SET="${DATAROOT}/PASCAL3D_train_NeMo/"
SAVED_NETWORK_PATH="${EXPROOT}/NeMo_${MESH_DIMENSIONS}/"

ALL_CATEGORIES=("aeroplane"  "bicycle"  "boat"  "bottle"  "bus"  "car"  "chair"  "diningtable"  "motorbike"  "sofa"  "train"  "tvmonitor")

BATCH_SIZE=108
TOTAL_EPOCHS=800
LEARNING_RATE=0.0001
WEIGHT_CLUTTER=5e-3
NUM_CLUTTER_IMAGE=5
NUM_CLUTTER_GROUP=512

for CATEGORY in "${ALL_CATEGORIES[@]}"; do
    mesh_path="${PATH_PASCAL3DP}/CAD_%s/%s/"
    CUDA_VISIBLE_DEVICES="${GPUS}" python "${ROOT}/code/TrainNeMo.py" \
            --mesh_path "${mesh_path}" --save_dir "${SAVED_NETWORK_PATH}" \
            --type_ "${CATEGORY}" --root_path "${PATH_CACHE_TRAINING_SET}" --mesh_d "${MESH_DIMENSIONS}" \
            --sperate_bank "True" --batch_size $BATCH_SIZE --total_epochs $TOTAL_EPOCHS \
            --lr $LEARNING_RATE --weight_noise $WEIGHT_CLUTTER --num_noise $NUM_CLUTTER_IMAGE \
            --max_group $NUM_CLUTTER_GROUP
done


