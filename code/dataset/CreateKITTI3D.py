import numpy as np
from PIL import Image
import os
import BboxTools as bbt
import cv2


split = 'training'

annotation_path = './%s/label_2/' % split
image_path = './%s/image_2/' % split
calib_path = './%s/calib/' % split

occ_level_mapping = {0: 'fully_visible', 1: 'partly_occluded', 2: 'largely_occluded', 3: 'unknown'}

save_dir = {'training': 'KITTI_train_distcrop', 'testing': 'KITTI_distcrop'}[split]
cate = 'car'

cate_map = {'car': 'Car'}
resize_to = 224

min_box_size = 100

camera_matrix_num = 2

out_shape = (256, 672)

elevation_default = 0
theta_default = 0


def get_one_anno(anno_line):
    anno_list = anno_line.strip().split(' ')
    cate_ = anno_list[0]
    box_numpy = np.array(
        [int(float(anno_list[4])), int(float(anno_list[5])), int(float(anno_list[6])), int(float(anno_list[7]))])
    return cate_, int(anno_list[2]), box_numpy, float(anno_list[1])


def if_legal(input_, min_size=40, truncated_max=0.2):
    cate_, occ_level, box_np, truncated = input_
    return cate_ == cate_map[cate] and min(
        bbt.from_numpy(box_np, sorts=['y0', 'x0', 'y1', 'x1']).shape) > min_size and truncated < truncated_max and occ_level == this_occ_lv


def project_to_2d(x3d, P):
    if len(x3d.shape) == 1:
        x3d = np.expand_dims(x3d, axis=0)
        single_ = True
    else:
        single_ = False
    # [n, 3 + 1] @ [4, 3] -> [n, 3]
    x2d = np.concatenate((x3d, np.ones((1, 1))), axis=1) @ P
    x2d[:, 0] /= x2d[:, 2]
    x2d[:, 1] /= x2d[:, 2]

    if single_:
        return x2d[0, 0:2]
    else:
        return x2d[:, 0:2]


if __name__ == '__main__':
    for this_occ_lv in range(3):
        print('______________________________________')
        print('Start_level', this_occ_lv)
        save_anno_dir = os.path.join(save_dir, 'annotations', cate)
        save_img_dir = os.path.join(save_dir, 'images', cate)
        save_list_dir = os.path.join(save_dir, 'list', cate + '_' + occ_level_mapping[this_occ_lv])

        os.makedirs(save_anno_dir, exist_ok=True)
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_list_dir, exist_ok=True)

        name_list = [t.split('.')[0] for t in os.listdir(annotation_path)]

        out_name = []
        for this_name in name_list:
            if not (os.path.exists(os.path.join(image_path, this_name + '.png'))
                and os.path.exists(os.path.join(annotation_path, this_name + '.txt'))
                and os.path.exists(os.path.join(calib_path, this_name + '.txt'))):
                print('Miss Image or annotation on', this_name)
                continue
            image = np.array(Image.open(os.path.join(image_path, this_name + '.png')))
            annotations = open(os.path.join(annotation_path, this_name + '.txt')).readlines()
            camera_matrix_txt = open(os.path.join(calib_path, this_name + '.txt')).readlines()[camera_matrix_num]

            camera_matrix = np.array(camera_matrix_txt.strip().split(' ')[1::], dtype=np.float32).reshape((3, 4)).T

            legal_mask = [if_legal(get_one_anno(t), min_size=min_box_size) for t in annotations]

            if sum(legal_mask) == 0:
                continue

            for i, annotation in enumerate(annotations):
                if not legal_mask[i]:
                    continue
                cate_, occ_level, box_np, _ = get_one_anno(annotation)
                box = bbt.from_numpy(box_np, image_boundary=image.shape[0:2], sorts=('x0', 'y0', 'x1', 'y1'))
                # Image.fromarray(box.apply(image)).show()
                annotation = annotation.split(' ')

                distance_ = np.sum(np.array(annotation[11:14], dtype=np.float32) ** 2) ** .5
                c3d = np.array(annotation[11:14], dtype=np.float32) - .5 * np.array([0, float(annotation[8]), 0])
                c2d = project_to_2d(c3d, camera_matrix)

                if np.arctan(c3d[2] / c3d[0]) > 0:
                    azimuth = float(annotation[14]) + np.pi + np.arctan(c3d[2] / c3d[0])
                else:
                    azimuth = (float(annotation[14]) + np.arctan(c3d[2] / c3d[0])) % (2 * np.pi)

                elevation = -np.arcsin(c3d[1] / distance_)

                resize_rate = float(200 * distance_ / 1280)

                box_ori = box.copy()

                box *= resize_rate
                img = cv2.resize(image, dsize=(int(image.shape[1] * resize_rate), int(image.shape[0] * resize_rate)))

                center = (c2d[::-1] * resize_rate).astype(int)

                if out_shape[0] // 2 - center[0] > 0 or out_shape[1] // 2 - center[1] > 0 or out_shape[0] // 2 + center[
                    0] - img.shape[0] > 0 or out_shape[1] // 2 + center[1] - img.shape[1] > 0:
                    if len(img.shape) == 2:
                        padding = (
                        (max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                        (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)))
                    else:
                        padding = (
                        (max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                        (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)),
                        (0, 0))

                    img = np.pad(img, padding, mode='constant')
                    box = box.shift([padding[0][0], padding[1][0]])
                    # box1 = box1.shift([padding[0][0], padding[1][0]])
                    center = (center[0] + padding[0][0], center[1] + padding[1][0])
                    # box = box.shift([((padding[0][0] + 1) // 2) * 2, ((padding[1][0] + 1) // 2) * 2])
                else:
                    if len(img.shape) == 2:
                        padding = ((0, 0),
                                   (0, 0))
                    else:
                        padding = ((0, 0),
                                   (0, 0),
                                   (0, 0))

                box_in_cropped = box.copy()
                box = bbt.box_by_shape(out_shape, center).set_boundary(img.shape[0:2])
                box_in_cropped = box.box_in_box(box_in_cropped)

                img_cropped = box.apply(img)

                proj_foo = bbt.projection_function_by_boxes(box_ori, box_in_cropped, compose=False)

                cropped_kp_list = np.array([])
                states_list = np.array([])

                save_parameters = dict(name=this_name, box=box.numpy(), box_ori=box_ori.numpy(),
                                       box_obj=box_in_cropped.numpy(),
                                       cropped_kp_list=cropped_kp_list, visible=states_list, padding=padding,
                                       resize_rate=resize_rate)

                save_parameters['azimuth'] = azimuth
                save_parameters['distance'] = distance_
                save_parameters['elevation'] = elevation
                save_parameters['theta'] = theta_default
                save_parameters['focal'] = -1
                save_parameters['viewport'] = -1
                save_parameters['principal'] = np.array(c2d)

                save_parameters['bbox'] = box_np
                save_parameters['cad_index'] = 1

                np.savez(os.path.join(save_anno_dir, this_name + '_%d.npz' % i), **save_parameters)
                Image.fromarray(img_cropped).save(os.path.join(save_img_dir, this_name + '_%d.JPEG' % i))

                print('Finished: ' + this_name + '_%d.npz' % i)
                out_name.append(this_name + '_%d\n' % i)

        with open(os.path.join(save_list_dir, cate + '_%s.txt' % occ_level_mapping[this_occ_lv]), 'w') as file_handel:
            file_handel.write(''.join(out_name))

        for kk in range(4):
            with open(os.path.join(save_list_dir, cate + '_%s_folder%d.txt' % (occ_level_mapping[this_occ_lv], kk)), 'w') as  file_handel:
                file_handel.write(''.join([t for i, t in enumerate(out_name) if i % 4 == kk]))