import os
import argparse

global args
parser = argparse.ArgumentParser(description='Create Cuboids')

parser.add_argument('--source_path', default='../data/PASCAL3D_NeMo/', type=str)
parser.add_argument('--target_path', default='../data/PASCAL3D_OCC_NeMo/', type=str)
parser.add_argument('--occ_level', default='FGL1_BGL1', type=str)
parser.add_argument('--mesh_d', default='single', type=str)

args = parser.parse_args()

mesh_d = args.mesh_d
source_path = args.source_path + 'annotations3D_%s/' % mesh_d
target_path = args.target_path + 'annotations3D_%s/' % mesh_d
occ_level = args.occ_level

all_cates = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

for cate in all_cates:
    source_list = os.listdir(os.path.join(source_path, cate))
    save_dir = os.path.join(target_path, cate + occ_level)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for n in source_list:
        os.system('ln ' + os.path.join(source_path, cate, n) + ' ' + os.path.join(save_dir, n))
