import numpy as np
import os


def load_off(off_file_name):
    file_handle = open(off_file_name)
    # n_points = int(file_handle.readlines(6)[1].split(' ')[0])
    # all_strings = ''.join(list(islice(file_handle, n_points)))

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(' ')[0])
    all_strings = ''.join(file_list[2:2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep='\n')

    all_strings = ''.join(file_list[2 + n_points:])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep='\n')

    array_ = array_.reshape((-1, 3))

    return array_, array_int.reshape((-1, 4))[:, 1::]


def get_n_list(mesh_path):
    name_list = os.listdir(mesh_path)
    name_list = ['%02d.off' % (i + 1) for i in range(len(name_list))]

    out = []
    for n in name_list:
        out.append(load_off(os.path.join(mesh_path,n))[0].shape[0])
    return out


if __name__ == '__main__':
    print(get_n_list('../PASCAL3D/CAD_d4/car/'))