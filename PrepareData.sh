#!/bin/bash
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

ROOT=$(get_abs_filename "./")

ENABLE_OCCLUDED=false
DATAROOT="${ROOT}/data"

PATH_PASCAL3DP="${DATAROOT}/PASCAL3D+_release1.1/"
PATH_OCCLUDED_PASCAL3DP="${DATAROOT}/OccludedPascal3D/"

PATH_CACHE_TRAINING_SET="${DATAROOT}/PASCAL3D_train_NeMo/"
PATH_CACHE_TESTING_SET="${DATAROOT}/PASCAL3D_NeMo/"
PATH_CACHE_TESTING_SET_OCC="${DATAROOT}/PASCAL3D_OCC_NeMo/"

OCC_LEVELS=("FGL1_BGL1"  "FGL2_BGL2"  "FGL3_BGL3")
MESH_DIMENSIONS=("single"  "multi")

####################################################################################################
# Download datasets
if [ ! -d "${DATAROOT}" ]; then
    mkdir "${DATAROOT}"
fi

if [ -d "${PATH_PASCAL3DP}" ]; then
    echo "Find Pascal3D+ dataset in ${PATH_PASCAL3DP}"
else
    echo "Download Pascal3D+ dataset in ${PATH_PASCAL3DP}"
    cd "${DATAROOT}"
    wget "ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip"
    unzip "PASCAL3D+_release1.1.zip"
    rm "PASCAL3D+_release1.1.zip"
    cd "${ROOT}"
fi

if [ ! -d "${PATH_PASCAL3DP}/Image_subsets" ]; then
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NsoVXW8ngQCqTHHFSW8YYsCim9EjiXS7' -O Image_subsets.zip
    unzip Image_subsets.zip
    rm Image_subsets.zip
    mv "Image_subsets" "${PATH_PASCAL3DP}"
fi

if ${ENABLE_OCCLUDED}; then
    if [ -d "${PATH_OCCLUDED_PASCAL3DP}" ]; then
        echo "Find OccludedPascal3D+ dataset in ${PATH_OCCLUDED_PASCAL3DP}"
    else
        echo "Download OccludedPascal3D+ dataset in ${PATH_OCCLUDED_PASCAL3DP}"
        cd "${DATAROOT}"
        git clone "https://github.com/Angtian/OccludedPASCAL3D.git"
        cd "OccludedPASCAL3D"
        chmod +x "download_FG.sh"
        ./download_FG.sh
        cd "${ROOT}"
    fi
fi


####################################################################################################
# Run dataset creator
echo "Create raw Pascal3D+ dataset!"
python ./code/dataset/CreatePascal3DNeMo.py --overwrite False --source_path "${PATH_PASCAL3DP}" \
        --save_path_train "${PATH_CACHE_TRAINING_SET}" --save_path_val "${PATH_CACHE_TESTING_SET}"

if ${ENABLE_OCCLUDED}; then
    for OCC_LEVEL in "${OCC_LEVELS[@]}"; do
        echo "Create OccludedPascal3D+ dataset!"
        python ./code/dataset/CreatePascal3DNeMo.py --overwrite False --data_pendix "${OCC_LEVEL}" \
                --save_path_train "" --save_path_val "${PATH_CACHE_TESTING_SET_OCC}" \
                --occ_data_path "${PATH_OCCLUDED_PASCAL3DP}"
    done
fi 


####################################################################################################
# Process meshes
for MESH_D in "${MESH_DIMENSIONS[@]}"; do
    python ./tools/CreateCuboidMesh.py --CAD_path "${PATH_PASCAL3DP}" --mesh_d "${MESH_D}"
done


####################################################################################################
# Create 3D annotations
for MESH_D in "${MESH_DIMENSIONS[@]}"; do
    python ./code/dataset/generate_3Dpascal3D.py --overwrite False \
            --root_path "${PATH_CACHE_TRAINING_SET}" --mesh_path "${PATH_PASCAL3DP}" --mesh_d "${MESH_D}"
    python ./code/dataset/generate_3Dpascal3D.py --overwrite False \
            --root_path "${PATH_CACHE_TESTING_SET}" --mesh_path "${PATH_PASCAL3DP}" --mesh_d "${MESH_D}"
done

####################################################################################################
# Link 3D annotations to occluded datasets
if ${ENABLE_OCCLUDED}; then
    for MESH_D in "${MESH_DIMENSIONS[@]}"; do
        for OCC_LEVEL in "${OCC_LEVELS[@]}"; do
            python ./code/dataset/link_annotations.py --source_path "${PATH_CACHE_TESTING_SET}" --target_path "${PATH_CACHE_TESTING_SET_OCC}" \
                    --occ_level "${OCC_LEVEL}" --mesh_d "${MESH_D}"
            python ./code/dataset/refine_list.py --root_path "${PATH_CACHE_TESTING_SET_OCC}" \
                    --occ_level "${OCC_LEVEL}" --mesh_d "${MESH_D}"
        done
    done
fi 






