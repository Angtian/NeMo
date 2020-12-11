import numpy as np
import os


def filter_func(bins_valid):
    if type(bins_valid) == str:
        bins_valid = [True if c == 'T' else False for c in bins_valid]
    return lambda x, bins_valid_=bins_valid, bin_size = 2 * np.pi / len(bins_valid): \
        np.any([np.sum((bin_size * i < x) * (x < bin_size * (i + 1))) for i in range(len(bins_valid_)) if bins_valid_[i]])

# cates = ['car']
settings = 'TFFTTFFT'
# settings = 'FTTFFTTF'
para_name = 'azimuth'
mesh_d = 'buildn'

cates = ['aeroplane', 'bus', 'motorbike', 'bottle', 'boat', 'bicycle', 'sofa', 'tvmonitor', 'chair', 'diningtable', 'train', 'car']

for cate in cates:
    annos_path = './annotations/%s/' % cate

    source_path = './lists3D_' + mesh_d + '/%s/' % cate
    save_path = './lists3D_' + mesh_d + '_azum_' + settings + '/%s/' % cate

    os.makedirs(save_path, exist_ok=True)

    foo = filter_func([True if c == 'T' else False for c in settings])

    name_list = os.listdir(annos_path)
    out_ = []

    for n in name_list:
        annos = np.load(os.path.join(annos_path, n))
        if foo(annos[para_name]):
            out_.append(n.split('.')[0])

    list_names = os.listdir(source_path)

    for list_name in list_names:
        in_list = open(os.path.join(source_path, list_name))

        out_list = [t for t in in_list if t.split('.')[0] in out_]
        open(os.path.join(save_path, list_name), 'w').write(''.join(out_list))







