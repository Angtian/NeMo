import sys
sys.path.append('./code/lib')


import numpy as np
from MeshUtils import load_off, save_off
import os
import argparse

global args
parser = argparse.ArgumentParser(description='Create Cuboids')

parser.add_argument('--CAD_path', default='../data/PASCAL3D+_release1.1', type=str)
parser.add_argument('--mesh_d', default='single', type=str)

args = parser.parse_args()

cates = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

mesh_path = args.CAD_path + '/CAD/%s'
target_path = args.CAD_path + '/CAD_%s/%s' % (args.mesh_d, '%s')

if args.mesh_d == 'multi':
    multi_ = True
else:
    multi_ = False

linear_coverage = 0.99
number_vertices_ = 1000


def meshelize(x_range, y_range, z_range, number_vertices):
    w, h, d = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
    total_area = (w * h + h * d + w * d) * 2

    # On average, every vertice attarch 6 edges. Each triangle has 3 edges
    mesh_size = total_area / (number_vertices * 2)

    edge_length = (mesh_size * 2) ** .5

    x_samples = x_range[0] + np.linspace(0, w, int(w / edge_length + 1))
    y_samples = y_range[0] + np.linspace(0, h, int(h / edge_length + 1))
    z_samples = z_range[0] + np.linspace(0, d, int(d / edge_length + 1))

    xn = x_samples.size
    yn = y_samples.size
    zn = z_samples.size

    out_vertices = []
    out_faces = []
    base_idx = 0

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[0]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += yn * xn

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[-1]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += yn * xn

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[0], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += zn * xn

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[-1], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += zn * xn

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[0], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
            out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
    base_idx += zn * yn

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[-1], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
            out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
    base_idx += zn * yn

    return np.array(out_vertices), np.array(out_faces)


if __name__ == '__main__':
    print('Create Cuboid Mesh: %s' % args.mesh_d)
    if multi_:
        for cate in cates:
            # print('cate:', cate)
            os.makedirs(target_path % cate, exist_ok=True)
            f_names = os.listdir(mesh_path % cate)
            f_names = [t for t in f_names if len(t) < 7 and '.off' in t]
            for f_name in f_names:
                vertices, faces = load_off(os.path.join(mesh_path % cate, f_name))
                selected_shape = int(vertices.shape[0] * linear_coverage)

                out_pos = []

                for i in range(vertices.shape[1]):
                    v_sorted = np.sort(vertices[:, i])
                    v_group = v_sorted[selected_shape::] - v_sorted[0:-selected_shape]
                    min_idx = np.argmin(v_group)
                    # print(min_idx, min_idx + selected_shape)
                    out_pos.append((v_sorted[min_idx], v_sorted[min_idx + selected_shape]))
                # print(out_pos)
                xvert, xface = meshelize(*out_pos, number_vertices=number_vertices_)

                save_off(os.path.join(target_path % cate, f_name), xvert, xface)
    else:
        for cate in cates:
            # print('cate:', cate)
            os.makedirs(target_path % cate, exist_ok=True)
            f_names = os.listdir(mesh_path % cate)
            f_names = [t for t in f_names if len(t) < 7 and '.off' in t]

            vertices = []

            for f_name in f_names:
                vertices_, faces = load_off(os.path.join(mesh_path % cate, f_name))
                vertices.append(vertices_)

            vertices = np.concatenate(vertices, axis=0)
            selected_shape = int(vertices.shape[0] * linear_coverage)
            out_pos = []

            for i in range(vertices.shape[1]):
                v_sorted = np.sort(vertices[:, i])
                v_group = v_sorted[selected_shape::] - v_sorted[0:-selected_shape]
                min_idx = np.argmin(v_group)
                # print(min_idx, min_idx + selected_shape)
                out_pos.append((v_sorted[min_idx], v_sorted[min_idx + selected_shape]))
            # print(out_pos)
            xvert, xface = meshelize(*out_pos, number_vertices=number_vertices_)

            save_off(os.path.join(target_path % cate, '01.off'), xvert, xface)
