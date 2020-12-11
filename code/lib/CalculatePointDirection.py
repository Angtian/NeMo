import numpy as np
from ProcessCameraParameters import CameraTransformer
from itertools import combinations


def circle_circonscrit(T):
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4) = T
    A = np.array([[x4 - x1, y4 - y1, z4 - z1], [x4 - x2, y4 - y2, z4 - z2], [x4 - x3, y4 - y3, z4 - z3]])
    Y = np.array([(x4 ** 2 + y4 ** 2 + z4 ** 2 - x1 ** 2 - y1 ** 2 - z1 ** 2),
                  (x4 ** 2 + y4 ** 2 + z4 ** 2 - x2 ** 2 - y2 ** 2 - z2 ** 2),
                  (x4 ** 2 + y4 ** 2 + z4 ** 2 - x3 ** 2 - y3 ** 2 - z3 ** 2)])
    if np.linalg.det(A) == 0:
        return None, 0
    Ainv = np.linalg.inv(A)
    X = 0.5 * np.dot(Ainv, Y)
    x, y, z = X[0], X[1], X[2]
    r = ((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2) ** .5
    return (x, y, z), r


def l2norm(x, axis=1): return x / np.sum(x ** 2, axis=axis, keepdims=True) ** 0.5


def ransac_one(target, points, non_linear_foo=lambda x: x > 0.01):
    # non_linear_foo = lambda x: x
    non_linear_foo = lambda x: np.exp(x)
    all_combinations = np.array(list(combinations(range(points.shape[0]), 3)))

    distances = np.ones(all_combinations.shape[0]) * 100
    centers = np.zeros((all_combinations.shape[0], 3))
    radius = np.zeros(all_combinations.shape[0])
    for i, selection in enumerate(all_combinations):
        selected_points = points[selection]
        center, r = circle_circonscrit(np.concatenate((selected_points, np.expand_dims(target, axis=0)), axis=0))
        if center is None:
            continue
        dis_caled = np.sum(non_linear_foo(np.abs(np.sum((points - np.array([center])) ** 2, axis=1) ** .5 - r)))

        centers[i] = np.array(center)
        radius[i] = r
        distances[i] = dis_caled
    min_idx = np.argmin(distances)
    center_ = centers[min_idx]
    return l2norm(center_ - target, axis=0)


def direction_calculator(verts, faces):
    out_dict = {i: set([]) for i in range(verts.shape[0])}

    for t in faces:
        for k in t:
            out_dict[k] = out_dict[k].union(set(t) - {k})

    direct_dict = {}


    for k in out_dict.keys():

        # print('array: ', np.array(list(out_dict[k])))
        if len(list(out_dict[k])) <= 2:
            direct_dict[k] = np.array([1, 0, 0])
            continue
        # direct_dict[k] = l2norm(np.mean(l2norm(verts[np.array(list(out_dict[k]))] - np.expand_dims(verts[k], axis=0)), axis=0), axis=0)
        direct_dict[k] = ransac_one(verts[k], verts[np.array(list(out_dict[k]))])

    return direct_dict


def cal_point_weight(direct_dict, vert, anno):
    cam_3d = CameraTransformer(anno).get_camera_position()
    vec_ = cam_3d.reshape((1, -1)) - vert
    vec_ = vec_ / (np.sum(vec_ ** 2, axis=1, keepdims=True) ** .5)
    matrix_dict = np.array([direct_dict[k] for k in direct_dict.keys()])
    return np.sum(vec_ * matrix_dict, axis=1)


if __name__ == '__main__':
    from MeshMemoryMap import load_off, MeshConverter
    from PIL import Image, ImageDraw
    import BboxTools as bbt
    name = 'n02958343_11980'
    annos = np.load('../PASCAL3D/annotations/car/%s.npz' % name)
    # Image.fromarray(im).show()

    verts, faces_idx = load_off('../PASCAL3D/CAD_d4/car/%02d.off' % annos['cad_index'])
    dict_dir = direction_calculator(verts, faces_idx)

    converter = MeshConverter(path='../PASCAL3D/CAD_d4/car/')
    pixels, visibile = converter.get_one(annos)

    weight = cal_point_weight(dict_dir, verts, annos)
    # img = np.zeros(tuple(annos['box_obj'][4::].astype(int)), dtype=np.float32)
    img = Image.open('../PASCAL3D/images/car/%s.JPEG' % name)
    imd = ImageDraw.ImageDraw(img)

    box_ = bbt.Bbox2D([(np.min(pixels[:, 0]), np.max(pixels[:, 0])), (np.min(pixels[:, 1]), np.max(pixels[:, 1]))])
    box_obj = bbt.from_numpy(annos['box_obj'])

    foo_proj = bbt.projection_function_by_boxes(box_, box_obj)

    pixels = foo_proj(pixels)

    print(box_)
    print(box_obj)

    for p, w, v in zip(pixels, weight, visibile):
        if not v:
            continue
        # value = (w + 1) / 2
        value = np.abs(w)
        imd.ellipse(bbt.box_by_shape((5, 5), p).pillow_bbox(), fill=(255, int(255 - value * 255), int(255 - value * 255)))

    img.show()



