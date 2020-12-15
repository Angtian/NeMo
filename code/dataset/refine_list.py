import os
import argparse

global args
parser = argparse.ArgumentParser(description='Create Cuboids')

parser.add_argument('--root_path', default='../data/PASCAL3D_NeMo/', type=str)
parser.add_argument('--occ_level', default='FGL1_BGL1', type=str)
parser.add_argument('--mesh_d', default='single', type=str)

args = parser.parse_args()

all_cates = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

mesh_d = args.mesh_d
catr = args.occ_level

for catf in all_cates:
    cate = catf + catr

    list_dir = args.root_path + 'lists/' + cate
    save_dir = args.root_path + 'lists3D_%s/' % mesh_d + cate
    image_dir = args.root_path + 'images/' + cate
    annos_dir = args.root_path + 'annotations3D_%s/' % mesh_d + cate
    file_name_pendix = '.JPEG'

    os.makedirs(save_dir, exist_ok=True)

    annos_list = [t.split('.')[0] + file_name_pendix for t in os.listdir(annos_dir)]
    imgs_list = [t.split('.')[0] + file_name_pendix for t in os.listdir(image_dir)]

    inter_list_set = set(annos_list).intersection(set(imgs_list))

    list_list = os.listdir(list_dir)

    out_names = []
    for list_name in list_list:
        fnames = open(os.path.join(list_dir, list_name)).readlines()
        fnames = [t.strip() for t in fnames]

        fnames_useful = list(set(fnames).intersection(inter_list_set))
        fnames_useful = [t + '\n' for t in fnames_useful]
        out_names += fnames_useful
        out_string = ''.join(fnames_useful)

        if not mesh_d == 'single':
            with open(os.path.join(save_dir, list_name), 'w') as fl:
                fl.write(out_string)

    if mesh_d == 'single':
        out_names = list(set(out_names))
        out_string = ''.join(out_names)
        with open(os.path.join(save_dir, 'mesh01.txt'), 'w') as fl:
            fl.write(out_string)
