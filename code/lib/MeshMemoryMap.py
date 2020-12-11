from MeshUtils import load_off
import os
import BboxTools as bbt
from ProcessCameraParameters import get_anno, Projector3Dto2D, CameraTransformer
import numpy as np
from CalculateOcc import cal_occ_one_image


mesh_path = '../PASCAL3D/PASCAL3D+_release1.1/CAD_d4/car/'


def normalization(value):
    return (value - value.min()) / (value.max() - value.min())


class MeshLoader(object):
    def __init__(self, path=mesh_path):
        file_list = os.listdir(path)

        l = len(file_list)
        file_list = ['%02d.off' % (i + 1) for i in range(l)]

        self.mesh_points_3d = []
        self.mesh_triangles = []

        for fname in file_list:
            points_3d, triangles = load_off(os.path.join(path, fname))
            self.mesh_points_3d.append(points_3d)
            self.mesh_triangles.append(triangles)

    def __getitem__(self, item):
        return self.mesh_points_3d[item], self.mesh_triangles[item]

    def __len__(self):
        return len(self.mesh_points_3d)


class MeshConverter(object):
    def __init__(self, path=mesh_path):
        self.loader = MeshLoader(path=path)

    def get_one(self, annos, return_distance=False):
        off_idx = get_anno(annos, 'cad_index')
        
        points_3d, triangles = self.loader[off_idx - 1]
        points_2d = Projector3Dto2D(annos)(points_3d).astype(np.int32)
        points_2d = np.flip(points_2d, axis=1)
        cam_3d = CameraTransformer(annos).get_camera_position() #  @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])

        distance = np.sum((-points_3d - cam_3d.reshape(1, -1)) ** 2, axis=1) ** .5
        distance_ = normalization(distance)
        h, w = get_anno(annos, 'height', 'width')
        map_size = (h, w)

        if_visible = cal_occ_one_image(points_2d=points_2d, distance=distance_, triangles=triangles, image_size=map_size)
        box_ori = bbt.from_numpy(get_anno(annos, 'box_ori'))
        box_cropped = bbt.from_numpy(get_anno(annos, 'box_obj').astype(np.int))
        box_cropped.set_boundary(get_anno(annos, 'box_obj').astype(np.int)[4::].tolist())

        if_visible = np.logical_and(if_visible, box_ori.include(points_2d))
        
        projection_foo = bbt.projection_function_by_boxes(box_ori, box_cropped)

        pixels_2d = projection_foo(points_2d)

        # handle the case that points are out of boundary of the image
        pixels_2d = np.max([np.zeros_like(pixels_2d), pixels_2d], axis=0)
        pixels_2d = np.min([np.ones_like(pixels_2d) * (np.array([box_cropped.boundary]) - 1), pixels_2d], axis=0)

        if return_distance:
            return pixels_2d, if_visible, distance_

        return pixels_2d, if_visible


if __name__ == '__main__':
    from PIL import Image, ImageDraw
    name_ = 'n02814533_11997.JPEG'
    anno_path = '../PASCAL3D/annotations/car/'
    converter = MeshConverter()
    pixels, visibile, distance = converter.get_one(np.load(os.path.join(anno_path, name_.split('.')[0] + '.npz'), allow_pickle=True), return_distance=True)
    image = Image.open('../PASCAL3D/images/car/' + name_)

    imd = ImageDraw.ImageDraw(image)

    for p, v in zip(pixels, visibile):
        if v:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        box = bbt.box_by_shape((5, 5), p)
        imd.ellipse(box.pillow_bbox(), fill=color)
    image.show()

    image = Image.open('../PASCAL3D/images/car/' + name_)

    imd = ImageDraw.ImageDraw(image)

    for p, d in zip(pixels, distance):
        color = (0, 255 - int(255 * d), 0)

        box = bbt.box_by_shape((5, 5), p)
        imd.ellipse(box.pillow_bbox(), fill=color)
    image.show()


