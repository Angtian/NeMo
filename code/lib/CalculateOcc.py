import numpy as np
import BboxTools as bbt


def linear_space_solve(posi_, depth_):
    posi_ = np.concatenate([np.transpose(posi_), np.ones((1, posi_.shape[0]))])
    get = np.matmul(depth_.reshape((1, 3)), np.linalg.inv(posi_))
    return lambda x, get=get: np.matmul(x, get[:, 0:2].T) + get[0, 2]


def linear_functional_solve(posi_):
    get = np.matmul(np.ones((1, posi_.shape[0])), np.linalg.inv(posi_.transpose()))
    return lambda x, get=get: np.matmul(x, get.T) - 1


def generate_mask(p0, p1, p2, mask_size, eps=1e-6):
    x_range = (np.ones(mask_size, dtype=np.float32) * np.arange(mask_size[1]).reshape(1, -1)).ravel()
    y_range = (np.ones(mask_size, dtype=np.float32) * np.arange(mask_size[0]).reshape(-1, 1)).ravel()
    positions = np.concatenate([y_range.reshape(-1, 1), x_range.reshape(-1, 1)], axis=1)
    return generate_mask_kernel(p0, p1, p2, mask_size, positions, eps=eps)


def generate_mask_kernel_(p0, p1, p2, mask_size, positions, eps=1e-6):
    points_ = [[p0, p1], [p0, p2], [p1, p2]]
    points_neg = [p2, p1, p0]

    foos = [linear_functional_solve(np.array(p_)) for p_ in points_]
    inds = [foo(np.array([p_])) for foo, p_ in zip(foos, points_neg)]
    mask = np.logical_and.reduce([(foo(positions) * ind > -eps).reshape(mask_size) for foo, ind in zip(foos, inds)])

    return mask


def area_triangle(p0, p1, p2):
    return np.abs(p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1])) / 2


def generate_mask_kernel(p0, p1, p2, mask_size, positions, eps=3):
    A = area_triangle(p0, p1, p2)

    A1 = area_triangle(p0, p1, positions.T)
    A2 = area_triangle(p1, p2, positions.T)
    A3 = area_triangle(p0, p2, positions.T)

    return (np.abs(A - (A1 + A2 + A3)) < eps).reshape(mask_size)


def generate_depth_map_one_triangle(points, depth):
    box = bbt.contain_points(points)
    if box.size < 3 or area_triangle(*points) < 1e-2:
        return np.ones(box.shape, dtype=np.bool), None
    # box = box.pad(1)
    points -= np.array([box.lu])

    mask_size = box.shape
    x_range = (np.ones(mask_size, dtype=np.float32) * np.arange(mask_size[1]).reshape(1, -1)).ravel()
    y_range = (np.ones(mask_size, dtype=np.float32) * np.arange(mask_size[0]).reshape(-1, 1)).ravel()
    positions = np.concatenate([y_range.reshape(-1, 1), x_range.reshape(-1, 1)], axis=1)

    mask_ = generate_mask_kernel(*points.tolist(), mask_size, positions)

    depth_map = linear_space_solve(points, depth)(positions).reshape(mask_size)
    depth_map = depth_map * mask_ + 1e10 * np.logical_not(mask_)
    # assert tuple(depth_map.shape) == tuple(box.shape), 'map size: ' + str(tuple(depth_map.shape)) + ' box size: ' + str(tuple(box.shape))
    return depth_map, box


def cal_occ_one_image(points_2d, distance, triangles, image_size, inf_=1e10, eps=1e-3):
    out_depth = np.ones(image_size, dtype=np.float32) * inf_

    # handle the case that points are out of boundary of the image
    points_2d = np.max([np.zeros_like(points_2d), points_2d], axis=0)
    points_2d = np.min([np.ones_like(points_2d) * (np.array([image_size]) - 1), points_2d], axis=0)

    for tri_ in triangles:
        points = points_2d[tri_]
        depths = distance[tri_]

        get_map, get_box = generate_depth_map_one_triangle(points, depths)
        if not get_box:
            continue

        get_box.set_boundary(out_depth.shape)

        # assert tem_box.size == get_box.size, str(get_box) + '   ' + str(tem_box) + '  ' + str(points.tolist())
        get_box.assign(out_depth, np.min([get_map, get_box.apply(out_depth)], axis=0), auto_fit=False)

    invalid_parts = out_depth > inf_ * 0.9

    out_depth[invalid_parts] = 0

    visible_distance = out_depth[tuple(points_2d.T.tolist())]
    if_visible = np.abs(distance - visible_distance) < eps
    return if_visible


if __name__ == '__main__':
    from PIL import Image
    points = np.array([[ 10,  91],
       [  1,   1],
       [  5, 237]])
    depth = np.array([1.        , 0.24409986, 0.78510327])

    Image.fromarray(generate_mask(*points, (11, 238)).astype(np.uint8) * 255).show()
    # print(generate_depth_map_one_triangle(points, depth))
    # pos = np.array([[100, 100], [200, 300], [300, 200]])
    # depth = np.array([0, 0.5, 1])
    # Image.fromarray((generate_depth_map_one_triangle(pos, depth)[0] * 255).astype(np.uint8)).show()
    # get = linear_functional_solve(pos)

    # Image.fromarray(generate_mask([1, 1], [200, 300], [500, 200], (400, 600)).astype(np.uint8) * 255).show()
    # Image.fromarray(generate_mask([1, 1], [2, 3], [3, 2], (5, 5)).astype(np.uint8) * 255).show()




