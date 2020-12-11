import sys
sys.path.append('../lib')


import numpy as np
import os
from MeshMemoryMap import MeshConverter
from CalculatePointDirection import cal_point_weight, direction_calculator
from get_n_list import get_n_list


cates = ['car', 'bus', 'motorbike', 'boat', 'bicycle', 'aeroplane', 'sofa', 'tvmonitor', 'chair', 'diningtable', 'bottle', 'train']
# cates = ['sofa']
d_mesh = 'buildsp'
dx_dict = {'car':'d5', 'bus':'d4', 'motorbike':'d4', 'boat':'d5', 'bicycle':'d5', 'aeroplane':'d56', 'sofa':'d4', 'tvmonitor':'d4', 'chair':'d45', 'diningtable':'d5', 'bottle':'d5', 'train':'d5'}

for cate in cates:
    root_path = '../PASCAL3D/PASCAL3D_train_distcrop_same/'
    # root_path = '../PASCAL3D/PASCAL3D_distcrop/'
    source_path = '../PASCAL3D/PASCAL3D_train/annotations/' + cate
    if d_mesh == 'dx':
        mesh_path = '../PASCAL3D/PASCAL3D+_release1.1/CAD_%s/' % dx_dict[cate] + cate
        destination_path = root_path + 'annotations3D_%s/' % dx_dict[cate] + cate
        save_list_path = root_path + 'lists3D_%s/' % dx_dict[cate] + cate
        
    else:
        mesh_path = '../PASCAL3D/PASCAL3D+_release1.1/CAD_%s/' % d_mesh + cate
        destination_path = root_path + 'annotations3D_%s/' % d_mesh + cate
        save_list_path = root_path + 'lists3D_%s/' % d_mesh + cate
    # mesh_path = '../PASCAL3D/CAD_d4/car'

    source_list_path = root_path + 'lists/' + cate
    image_dir = root_path + 'images/' + cate

    useful_vis_thr = 0.25
    # n_list = [250, 258, 346, 402, 243, 328, 296, 221, 398, 322]
    n_list = get_n_list(mesh_path)
    # n_list = [1123, 1163, 1846, 1732, 1324, 1722, 1503, 1024, 1590, 1675]
    print(n_list)
    nvis = [np.zeros(k) for k in n_list]
    ncount = [0] * len(n_list)

    os.makedirs(destination_path, exist_ok=True)

    fl_list = os.listdir(source_path)
    manager = MeshConverter(path=mesh_path)


    for fname in fl_list:
        try:
        # if True:
            annos = np.load(os.path.join(source_path, fname))
            annos = dict(annos)
            kps, vis = manager.get_one(annos)
            idx = annos['cad_index'] - 1
            ncount[idx] += 1
            nvis[idx] += vis
        except:
            print(fname)


    source_path = root_path + 'annotations/' + cate
    fl_list = os.listdir(source_path)


    direction_dicts = []
    for t in manager.loader:
        print(0)
        direction_dicts.append(direction_calculator(*t))


    get = [t / k for t, k in zip(nvis, ncount)]
    masks = [t > useful_vis_thr for t in get]
    print({i + 1: np.sum(masks[i]) for i in range(len(n_list))})

    np.savez(destination_path + '.npz', masks)

    for fname in fl_list:
        try:
        # if True:
            annos = np.load(os.path.join(source_path, fname))
            annos = dict(annos)
            kps, vis = manager.get_one(annos)
            idx = annos['cad_index'] - 1

            weights = cal_point_weight(direction_dicts[idx], manager.loader[idx][0], annos)

            # annos['kp_weights'] = np.abs(weights)[masks[idx]]
            # annos['cropped_kp_list'] = kps[masks[idx]]
            # annos['visible'] = vis[masks[idx]]
            annos['kp_weights'] = np.abs(weights)
            annos['cropped_kp_list'] = kps
            annos['visible'] = vis
            np.savez(os.path.join(destination_path, fname), **annos)
        except:
            print('Error: ', fname)


    file_name_pendix = '.JPEG'
    os.makedirs(save_list_path, exist_ok=True)

    annos_list = [t.split('.')[0] + file_name_pendix for t in os.listdir(destination_path)]
    imgs_list = [t.split('.')[0] + file_name_pendix for t in os.listdir(image_dir)]

    inter_list_set = set(annos_list).intersection(set(imgs_list))

    list_list = os.listdir(source_list_path)

    for list_name in list_list:
        fnames = open(os.path.join(source_list_path, list_name)).readlines()
        fnames = [t.strip() for t in fnames]

        fnames_useful = list(set(fnames).intersection(inter_list_set))
        fnames_useful = [t + '\n' for t in fnames_useful]
        out_string = ''.join(fnames_useful)
        with open(os.path.join(save_list_path, list_name), 'w') as fl:
            fl.write(out_string)

