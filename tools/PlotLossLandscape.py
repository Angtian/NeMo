import sys
sys.path.append('../code/lib')


from MeshUtils import *
from PIL import Image
import io
import os
import matplotlib.pyplot as plt

cate = 'chair'
mesh_d = 'buildsp'
occ_level = 'FGL2_BGL2'
# occ_level = ''

if len(occ_level) == 0:
    # mesh_path = '../PASCAL3D/CAD_d4/car/%02d.off'
    mesh_path = '../PASCAL3D/CAD_' + mesh_d + '/' + cate + '/%02d.off'
    img_path = '../PASCAL3D/PASCAL3D_NeMo/images/' + cate + '/%s.JPEG'
    annos_path = '../PASCAL3D/PASCAL3D_NeMo/annotations/' + cate + '/%s.npz'
    # record_file_path = './saved_features/car/resunetpre_3D1024_points1saved_model_car_799.npz'
    record_file_path = './saved_features/' + cate + '/resunetpre_3D512_points1saved_model_' + cate + '_799_' + mesh_d + '.npz'

    save_dir = '../junks/aligns_final/' + cate + '_' + mesh_d + '/'

    names = os.listdir('../PASCAL3D/PASCAL3D_NeMo/images/' + cate)
else:
    mesh_path = '../PASCAL3D/CAD_' + mesh_d + '/' + cate + '/%02d.off'
    img_path = '../PASCAL3D/PASCAL3D_OCC_NeMo/images/' + cate + occ_level + '/%s.JPEG'
    annos_path = '../PASCAL3D/PASCAL3D_NeMo/annotations/' + cate + '/%s.npz'
    record_file_path = './saved_features/' + cate + '_occ/' + occ_level + '_resunetpre_3D512_points1saved_model_' + cate + '_799_' + mesh_d + '.npz'

    save_dir = '../junks/aligns_final/' + cate + occ_level + '_' + mesh_d + '/'
    names = os.listdir('../PASCAL3D/PASCAL3D_NeMo/images/' + cate)

names = [t.split('.')[0] for t in names]

# image_name = 'n03498781_613'
device = 'cuda:0'

down_smaple_rate = 8
image_sizes = {'car': (256, 672), 'bus': (384, 896), 'motorbike': (512, 512), 'boat': (512, 1216),
               'bicycle': (608, 608), 'aeroplane': (320, 1024), 'sofa': (352, 736), 'tvmonitor': (480, 480),
               'chair': (544, 384), 'diningtable': (320, 800), 'bottle': (512, 736), 'train': (256, 608)}

distance_render = {'car': 5, 'bus': 6, 'motorbike': 4.5, 'bottle': 5, 'boat': 8, 'bicycle': 5.2, 'aeroplane': 7,
                   'sofa': 5, 'tvmonitor': 5.5, 'chair': 4, 'diningtable': 7, 'train': 4.5}


def plot_fun(values, para_scans, colors, figsize=(10.5, 4)):
    plt.figure(num=None, figsize=figsize)
    ax = plt.axes()

    for v, p, c in zip(values, para_scans, colors):
        ax.plot(v, p, c)
    plt.axvline(x=0, c='black')
    return ax


def get_one_image_from_plt(plot_functions, plot_args=tuple(), plot_kwargs=dict()):
    plt.cla()
    plt.clf()
    ax = plot_functions(*plot_args, **plot_kwargs)
    positions = ax.get_position()
    pos = [positions.y0, positions.y1, positions.x0, positions.x1]
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    img = np.array(im)
    h, w = img.shape[0:2]
    box = bbt.from_numpy([np.array([int(t[0] * h), int(t[1] * h), int(t[2] * w), int(t[3] * w)]) for t in [pos]][0])
    box = box.pad(1)
    box = box.shift((2, 1))
    img = box.apply(img)
    bbt.draw_bbox(img, bbt.full(img.shape).pad(-2), boundary=(0, 0, 0), boundary_width=11)
    # img = np.transpose(img, (1, 0, 2))
    return img


if __name__ == '__main__':
    os.makedirs(save_dir, exist_ok=True)
    render_image_size = max(image_sizes[cate]) // down_smaple_rate
    cameras = OpenGLPerspectiveCameras(device=device, fov=12.0)
    raster_settings = RasterizationSettings(
        image_size=render_image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )

    all_distance = []
    for image_name in names:
        print(image_name, end=' ')
        annos_file = np.load(annos_path % image_name)

        # xvert, xface = load_off('../PASCAL3D/CAD_d4/car/%02d.off' % annos_file['cad_index'], to_torch=True)
        if mesh_d == 'build':
            xvert, xface = load_off('../PASCAL3D/CAD_' + mesh_d + '/' + cate + '/%02d.off' % 1, to_torch=True)
            subtype = 'mesh%02d' % 1
        else:
            xvert, xface = load_off('../PASCAL3D/CAD_' + mesh_d + '/' + cate + '/%02d.off' % annos_file['cad_index'], to_torch=True)
            subtype = 'mesh%02d' % annos_file['cad_index']
        record_file = np.load(record_file_path)

        feature_bank = torch.from_numpy(record_file['memory_%s' % subtype])
        if image_name not in record_file.keys():
            continue
        predicted_map = record_file[image_name]
        predicted_map = torch.from_numpy(predicted_map).to(device)

        inter_module = MeshInterpolateModule(xvert, xface, feature_bank, rasterizer, post_process=center_crop_fun(predicted_map.shape[1::], (render_image_size, ) * 2))
        inter_module.cuda()

        azimuth_shifts = np.linspace(-3.14, 3.14, 121)
        elevation_shifts = np.linspace(-3.14 / 2, 3.14 / 2, 61)
        theta_shifts = np.linspace(-3.14 / 2, 3.14 / 2, 61)
        distance_shifts = np.linspace(-2, 2, 41)

        get = []
        # for elevation_shift in elevation_shifts:
        for azimuth_shift in azimuth_shifts:
            this_azum = (annos_file['azimuth'] + azimuth_shift + 2 * np.pi) % (2 * np.pi)
            this_elev = annos_file['elevation']
            C = camera_position_from_spherical_angles(distance_render[cate], this_elev, this_azum, degrees=False, device=device)
            theta = torch.from_numpy(annos_file['theta']).type(torch.float32).view(1)
            projected_map = inter_module(C, theta).squeeze()
            sim_ = torch.sum(projected_map * predicted_map, dim=0)

            get.append(1 - (torch.mean(sim_)).item())
            # if np.abs(azimuth_shift) < 1e-5:
            #     print(C, (torch.mean(sim_)).item())
        azum_scan = np.array(get)

        get = []
        for elevation_shift in elevation_shifts:
            this_azum = annos_file['azimuth']
            this_elev = annos_file['elevation'] + elevation_shift
            C = camera_position_from_spherical_angles(distance_render[cate], this_elev, this_azum, degrees=False, device=device)
            theta = torch.from_numpy(annos_file['theta']).type(torch.float32).view(1)
            projected_map = inter_module(C, theta).squeeze()
            sim_ = torch.sum(projected_map * predicted_map, dim=0)
            get.append(1 - (torch.mean(sim_)).item())
            # if np.abs(elevation_shift) < 1e-5:
            #     print(C, (torch.mean(sim_)).item())
        elev_scan = np.array(get)

        get = []
        for theta_shift in theta_shifts:
            this_azum = annos_file['azimuth']
            this_elev = annos_file['elevation']
            C = camera_position_from_spherical_angles(distance_render[cate], this_elev, this_azum, degrees=False, device=device)
            theta = torch.from_numpy(np.array(annos_file['theta'] + theta_shift)).type(torch.float32).view(1)
            projected_map = inter_module(C, theta).squeeze()
            sim_ = torch.sum(projected_map * predicted_map, dim=0)
            get.append(1 - (torch.mean(sim_)).item())
            # if np.abs(theta_shift) < 1e-5:
            #     print(C, (torch.mean(sim_)).item())
        theta_scan = np.array(get)

        get = []
        for distance_shift in distance_shifts:
            this_azum = annos_file['azimuth']
            this_elev = annos_file['elevation']
            C = camera_position_from_spherical_angles(distance_render[cate] + distance_shift, this_elev, this_azum, degrees=False,
                                                      device=device)
            theta = torch.from_numpy(np.array(annos_file['theta'])).type(torch.float32).view(1)
            projected_map = inter_module(C, theta).squeeze()
            sim_ = torch.sum(projected_map * predicted_map, dim=0)
            get.append(1 - (torch.mean(sim_)).item())
            # if np.abs(theta_shift) < 1e-5:
            #     print(C, (torch.mean(sim_)).item())
        distance_scan = np.array(get)
        this_dist = distance_shifts[np.argmax(distance_scan)]
        all_distance.append(this_dist)
        print(this_dist)

        # print(np.max(azum_scan), np.max(elev_scan), np.max(theta_scan))
        # print(azum_scan[75], elev_scan[30], theta_scan[30])
        # print(np.argmax(azum_scan), np.argmax(elev_scan), np.argmax(theta_scan))

        values_ = [azimuth_shifts, elevation_shifts, theta_shifts, distance_shifts]
        scans_ = [azum_scan, elev_scan, theta_scan, distance_scan]
        # colors_ = ['b', 'r', 'g', 'y']
        colors_ = ['b', 'r', 'g']
        img_ = get_one_image_from_plt(plot_functions=plot_fun, plot_args=(values_, scans_, colors_))
        # Image.fromarray(img_).show()

        Image.fromarray(img_).save(save_dir + image_name + '.png')

        # plt.plot(azimuth_shifts, azum_scan, 'b')
        # plt.plot(elevation_shifts, elev_scan, 'r')
        # plt.plot(theta_shifts, theta_scan, 'g')

        # plt.savefig('../junks/scan_align_new/' + image_name + '.png')
        # plt.show()
    print(np.mean(all_distance))
    np.save(save_dir + 'all_distance.npy', all_distance)
