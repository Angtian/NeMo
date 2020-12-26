import sys
sys.path.append('./code/lib')


import numpy as np
import os
from MeshMemoryMap import MeshConverter
from CalculatePointDirection import cal_point_weight, direction_calculator
from get_n_list import get_n_list
import argparse

global args

parser = argparse.ArgumentParser(description='Generate 3D version of PASCAL3D+ dataset')
parser.add_argument('--overwrite', default='False', type=str, help='')
parser.add_argument('--root_path', default='../data/PASCAL3D_NeMo/', type=str, help='')
parser.add_argument('--mesh_path', default='../data/PASCAL3D+_release1.1/', type=str, help='')
parser.add_argument('--mesh_d', default='single', type=str, help='')

args = parser.parse_args()

args.overwrite = args.overwrite == 'True'

cates = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
d_mesh = args.mesh_d
dx_dict = {'car':'d5', 'bus':'d4', 'motorbike':'d4', 'boat':'d5', 'bicycle':'d5', 'aeroplane':'d56', 'sofa':'d4', 'tvmonitor':'d4', 'chair':'d45', 'diningtable':'d5', 'bottle':'d5', 'train':'d5'}

if 'single' in args.mesh_d:
    single_mesh = True
else:
    single_mesh = False

error_case = list()
print('Generating 3D annotations on %s.\nFinished: ' % args.root_path, end='')
for cate in cates:
    print('%s' % cate, end=' ')

    root_path = args.root_path
    mesh_path = args.mesh_path + 'CAD_%s/' % d_mesh + cate
    destination_path = root_path + 'annotations3D_%s/' % d_mesh + cate
    save_list_path = root_path + 'lists3D_%s/' % d_mesh + cate
    source_path = root_path + 'annotations/' + cate

    source_list_path = root_path + 'lists/' + cate
    image_dir = root_path + 'images/' + cate

    useful_vis_thr = 0.25
    n_list = get_n_list(mesh_path)

    os.makedirs(destination_path, exist_ok=True)

    manager = MeshConverter(path=mesh_path)

    fl_list = os.listdir(source_path)

    direction_dicts = []
    for t in manager.loader:
        direction_dicts.append(direction_calculator(*t))

    for fname in fl_list:
        if (not args.overwrite) and os.path.exists(os.path.join(destination_path, fname)):
            continue
        try:
        # if True:
            annos = np.load(os.path.join(source_path, fname), allow_pickle=True)
            annos = dict(annos)
            if single_mesh:
                annos['cad_index'] = 1
            kps, vis = manager.get_one(annos)
            idx = annos['cad_index'] - 1

            weights = cal_point_weight(direction_dicts[idx], manager.loader[idx][0], annos)

            annos['kp_weights'] = np.abs(weights)
            annos['cropped_kp_list'] = kps
            annos['visible'] = vis
            np.savez(os.path.join(destination_path, fname), **annos)
        except:
            error_case.append(cate + ' ' + fname)

    file_name_pendix = '.JPEG'
    os.makedirs(save_list_path, exist_ok=True)

    annos_list = [t.split('.')[0] + file_name_pendix for t in os.listdir(destination_path)]
    imgs_list = [t.split('.')[0] + file_name_pendix for t in os.listdir(image_dir)]

    inter_list_set = set(annos_list).intersection(set(imgs_list))

    list_list = os.listdir(source_list_path)

    out_names = []
    for list_name in list_list:
        fnames = open(os.path.join(source_list_path, list_name)).readlines()
        fnames = [t.strip() for t in fnames]

        fnames_useful = list(set(fnames).intersection(inter_list_set))
        fnames_useful = [t + '\n' for t in fnames_useful]
        if 'single' not in args.mesh_d:
            out_string = ''.join(fnames_useful)
            with open(os.path.join(save_list_path, list_name), 'w') as fl:
                fl.write(out_string)
        else:
            out_names += fnames_useful
    if 'single' in args.mesh_d:
        out_names = list(set(out_names))
        out_string = ''.join(out_names)
        with open(os.path.join(save_list_path, 'mesh01.txt'), 'w') as fl:
            fl.write(out_string)
print('\nErrors At: ', error_case)
