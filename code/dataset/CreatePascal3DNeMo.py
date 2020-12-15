import numpy as np
import BboxTools as bbt
import scipy.io as sio
import os
from PIL import Image
import pickle
import cv2
import math
from kp_list import kp_list_dict, mesh_len
from kp_list import top_50_size_dict
import argparse

global args

parser = argparse.ArgumentParser(description='Create PASCAL3D+ dataset')
parser.add_argument('--overwrite', default='False', type=str, help='')
parser.add_argument('--source_path', default='../data/PASCAL3D+_release1.1', type=str, help='')
parser.add_argument('--save_path_train', default='../data/PASCAL3D_train_NeMo/', type=str, help='')
parser.add_argument('--save_path_val', default='../data/PASCAL3D_NeMo/', type=str, help='')

# for occ: save_path_train -> ''; save_path_val -> '../../PASCAL3D/PASCAL3D_OCC_NeMo/'
parser.add_argument('--data_pendix', default='', type=str, help='')
parser.add_argument('--occ_data_path', default='../data/OccludedPASCAL3D/', type=str, help='')

args = parser.parse_args()

args.overwrite = args.overwrite == 'True'

# Parameters
categories = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

dataset = 'imagenet'
set_types = (['train'] if len(args.save_path_train) > 0 else []) + (['val'] if len(args.save_path_val) > 0 else [])
to_size = 224

dataset_root = args.source_path
save_root = {'train': args.save_path_train, 'val': args.save_path_val}

mesh_para_names = sum([t.split(', ') for t in ['azimuth, elevation, distance, focal, theta, principal, viewport', 'height, width']], [])
mesh_para_names = list(set(mesh_para_names)) + ['cad_index', 'bbox']


def get_anno(record, *args, idx=0):
    out = []
    for key_ in args:
        if key_ == 'height':
            out.append(record['imgsize'][0, 0][0][1])
        elif key_ == 'width':
            out.append(record['imgsize'][0, 0][0][0])
        elif key_ == 'bbox':
            out.append(record['objects'][0, 0]['bbox'][0, idx][0])
        elif key_ == 'cad_index':
            out.append(record['objects'][0, 0]['cad_index'][0, idx][0, 0])
        elif key_ == 'principal':
            px = record['objects'][0, 0]['viewpoint'][0, idx]['px'][0, 0][0, 0]
            py = record['objects'][0, 0]['viewpoint'][0, idx]['py'][0, 0][0, 0]
            out.append(np.array([px, py]))
        elif key_ in ['theta', 'azimuth', 'elevation']:
            out.append(record['objects'][0, 0]['viewpoint'][0, idx][key_][0, 0][0, 0] * math.pi / 180)
        else:
            out.append(record['objects'][0, 0]['viewpoint'][0, idx][key_][0, 0][0, 0])

    if len(out) == 1:
        return out[0]

    return tuple(out)


print('Creating dataset, finished: ', end='')
for category in categories:
    print('%s' % category, end=' ')
    # print(category)
    kp_list = kp_list_dict[category]

    for set_type in set_types:
        if set_type == 'train':
            this_size = top_50_size_dict[category]
            out_shape = [((this_size[0] - 1) // 32 + 1) * 32, ((this_size[1] - 1) // 32 + 1) * 32]
        else:
            # this_size = max_size_dict[category]
            this_size = top_50_size_dict[category]
            out_shape = [((this_size[0] - 1) // 32 + 1) * 32, ((this_size[1] - 1) // 32 + 1) * 32]
        out_shape = [int(out_shape[0]), int(out_shape[1])]
        # Kp_list
        if set_type == 'train':
            save_image_path = save_root['train'] + 'images/%s/' % (category + args.data_pendix)
            save_annotation_path = save_root['train'] + 'annotations/%s/' % (category + args.data_pendix)
            save_list_path = save_root['train'] + 'lists/%s/' % (category + args.data_pendix)
        else:
            save_image_path = save_root['val'] + 'images/%s/' % (category + args.data_pendix)
            save_annotation_path = save_root['val'] + 'annotations/%s/' % (category + args.data_pendix)
            save_list_path = save_root['val'] + 'lists/%s/' % (category + args.data_pendix)

        os.makedirs(save_image_path, exist_ok=True)
        os.makedirs(save_annotation_path, exist_ok=True)
        os.makedirs(save_list_path, exist_ok=True)

        # Path
        list_dir = os.path.join(dataset_root, 'Image_sets')
        pkl_dir = os.path.join(dataset_root, 'Image_subsets')
        anno_dir = os.path.join(dataset_root, 'Annotations', '{}_{}'.format(category, dataset))
        if len(args.data_pendix) > 0:
            load_image_path = os.path.join(args.occ_data_path, 'images', category + args.data_pendix)
            occ_mask_dir = os.path.join(args.occ_data_path, 'annotations', category + args.data_pendix)
        else:
            load_image_path = os.path.join(dataset_root, 'Images', '{}_{}'.format(category, dataset))
            occ_mask_dir = ''

        file_dir = os.path.join(list_dir, '{}_{}_{}.txt'.format(category, dataset, set_type))
        with open(file_dir, 'r') as fh:
            image_names = fh.readlines()
        image_names = [e.strip() for e in image_names if e != '\n']

        subtype_file_dir = os.path.join(list_dir, '{}_{}_subtype.txt'.format(category, dataset))
        with open(subtype_file_dir, 'r') as fh:
            subtype_list = fh.readlines()
        subtype_list = [e.strip() for e in subtype_list if e != '\n']
        pkl_path = os.path.join(pkl_dir, '{}_{}_{}.pkl'.format(category, dataset, set_type))
        subtype_images = pickle.load(open(pkl_path, 'rb'))
        annotations = [[] for _ in range(len(subtype_list))]

        mesh_name_list = ['' for _ in range(mesh_len[category])]
        for i in range(len(subtype_list)):
            name_list = ''
            for img_name in subtype_images[i]:
                name_list += img_name + '.JPEG\n'

                if (not args.overwrite) and os.path.exists(os.path.join(save_annotation_path, img_name + '.npz')) \
                        and os.path.exists(os.path.join(save_image_path, img_name + '.JPEG')):
                    continue

                anno_path = os.path.join(anno_dir, '{}.mat'.format(img_name))
                mat_contents = sio.loadmat(anno_path)
                record = mat_contents['record']
                objects = record['objects']
                azimuth_coarse = objects[0, 0]['viewpoint'][0, 0]['azimuth_coarse'][0, 0][0, 0]
                elevation_coarse = objects[0, 0]['viewpoint'][0, 0]['elevation_coarse'][0, 0][0, 0]
                distance = objects[0, 0]['viewpoint'][0, 0]['distance'][0, 0][0, 0]
                bbox = objects[0, 0]['bbox'][0, 0][0]

                if len(args.data_pendix) > 0:
                    occ_mask = np.load(os.path.join(occ_mask_dir, '{}.npz'.format(img_name)), allow_pickle=True)['occluder_mask']
                else:
                    occ_mask = None

                box = bbt.from_numpy(bbox, sorts=('x0', 'y0', 'x1', 'y1'))

                # resize_rate = 224 / min(box.shape)
                resize_rate = float(200 * get_anno(record, 'distance') / 1000)
                if resize_rate <= 0:
                    resize_rate = 224 / min(box.shape)

                box_ori = box.copy()

                box *= resize_rate

                img = np.array(Image.open(os.path.join(load_image_path, img_name + '.JPEG')))
                box_ori = box_ori.set_boundary(img.shape[0:2])

                img = cv2.resize(img, dsize=(int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate)))

                center = (get_anno(record, 'principal')[::-1] * resize_rate).astype(int)

                box1 = bbt.box_by_shape(out_shape, center)

                if out_shape[0] // 2 - center[0] > 0 or out_shape[1] // 2 - center[1] > 0 or out_shape[0] // 2 + center[0] - img.shape[0] > 0 or out_shape[1] // 2 + center[1] - img.shape[1] > 0:
                    if len(img.shape) == 2:
                        padding = ((max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                                   (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)))
                    else:
                        padding = ((max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                                   (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)),
                                   (0, 0))

                    img = np.pad(img, padding, mode='constant')
                    box = box.shift([padding[0][0], padding[1][0]])
                    box1 = box1.shift([padding[0][0], padding[1][0]])

                box_in_cropped = box.copy()
                box = box1.set_boundary(img.shape[0:2])
                box_in_cropped = box.box_in_box(box_in_cropped)

                img_cropped = box.apply(img)

                proj_foo = bbt.projection_function_by_boxes(box_ori, box_in_cropped, compose=False)

                cropped_kp_list = []
                states_list = []
                for kp in kp_list:
                    states = objects[0, 0]['anchors'][0, 0][kp][0, 0]['status'][0, 0][0, 0]
                    if states == 1:
                        kp_x, kp_y = objects[0, 0]['anchors'][0, 0][kp][0, 0]['location'][0, 0][0]
                        if len(args.data_pendix) > 0 and kp_x < occ_mask.shape[1] and kp_y < occ_mask.shape[0] and occ_mask[int(kp_y), int(kp_x)]:
                            states = 0
                        cropped_kp_x = proj_foo[1](kp_x)
                        cropped_kp_y = proj_foo[0](kp_y)
                    else:
                        cropped_kp_x = cropped_kp_y = 0
                    states_list.append(states)
                    cropped_kp_list.append([cropped_kp_y, cropped_kp_x])

                save_parameters = dict(name=img_name, box=box.numpy(), box_ori=box_ori.numpy(), box_obj=box_in_cropped.numpy(), cropped_kp_list=cropped_kp_list, visible=states_list, occ_mask=occ_mask)

                save_parameters = {**save_parameters, **{k: v for k, v in zip(mesh_para_names, get_anno(record, *mesh_para_names))}}

                mesh_idx = get_anno(record, 'cad_index')
                mesh_name_list[mesh_idx - 1] += img_name + '.JPEG\n'

                np.savez(os.path.join(save_annotation_path, img_name), **save_parameters)
                Image.fromarray(img_cropped).save(os.path.join(save_image_path, img_name + '.JPEG'))

            with open(os.path.join(save_list_path, subtype_list[i] + '.txt'), 'w') as fl:
                fl.write(name_list)

        for i, t_ in enumerate(mesh_name_list):
            with open(os.path.join(save_list_path, 'mesh%02d' % (i + 1) + '.txt'), 'w') as fl:
                fl.write(t_)
print()
