import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


save_dir = '../PASCAL3D_azumith_distribution_red_blue/'

settings = 'TFFTTFFT'

cates = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor', 'all']
# cates = ['car']
# cates = ['bottle']
list_path = '../data/PASCAL3D_NeMo/annotations/%s/'

for cate in cates:
    para_name = 'azimuth'
    if cate == 'all':
        name_list = sum([[os.path.join(list_path % c, t) for t in os.listdir(list_path % c)] for c in cates if c != 'all'], [])
    else:
        name_list = [os.path.join(list_path % cate, t) for t in os.listdir(list_path % cate)]

    plt.clf()

    out_ = []

    for n in name_list:
        annos = np.load(n)
        out_.append(annos[para_name] * 180 / np.pi)

    out_ = np.array(out_)
    out_ += (np.arange(out_.size) % 2 - 0.5) * 10
    out_ = out_ % 360
    N = 32

    bin_size = 360 / N

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False) + (np.pi / N)

    stat = np.array([np.sum((bin_size * i <= out_) * (out_ < bin_size * (i + 1))) for i in range(N)])
    radii = stat ** .5
    width = np.pi * 2 / N

    # colors = plt.cm.viridis(np.reshape(np.linspace(0, 1, N), (2, N // 2)).transpose().ravel())
    bool_label = np.concatenate([np.zeros(N // len(settings)) if c == 'T' else np.ones(N // len(settings)) for c in settings])

    colors = plt.cm.coolwarm(bool_label)

    ax = plt.subplot(111, projection='polar')
    print(theta)
    print(radii)
    ax.bar(theta, radii, width=width, bottom=0, color=colors, alpha=0.5)
    ax.set_rlim(0)
    ax.set_yticklabels([])

    blue_patch = mpatches.Patch(color='royalblue', label='Seen: %d' % np.sum([t for t, l in zip(stat, bool_label) if l == 0]))
    red_patch = mpatches.Patch(color='indianred', label='Unseen: %d' % np.sum([t for t, l in zip(stat, bool_label) if l == 1]))
    plt.legend(handles=[red_patch, blue_patch], loc='upper right', bbox_to_anchor=(1.1, 1.08))

    # plt.show()
    # break
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir + cate + '.png')
