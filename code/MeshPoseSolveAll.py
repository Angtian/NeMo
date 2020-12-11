from lib.MeshUtils import *
from lib.ProcessCameraParameters import get_anno, get_transformation_matrix
import os
import argparse
from scipy.linalg import logm


parser = argparse.ArgumentParser(description='Pose estimation')
parser.add_argument('--type_', default='car', type=str, help='')
parser.add_argument('--mesh_d', default='build', type=str, help='')
parser.add_argument('--turn_off_clutter', default=False, type=bool, help='')
parser.add_argument('--objectnet', default=False, type=bool, help='')
parser.add_argument('--record_pendix', default='', type=str, help='')
parser.add_argument('--pre_render', default=True, type=bool, help='')
args = parser.parse_args()

level = 1


def rotation_theta(theta):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


def cal_err(gt, pred):
    # return radius
    return ((logm(np.dot(np.transpose(pred), gt)) ** 2).sum()) ** 0.5 / (2. ** 0.5)


def cal_rotation_matrix(theta, elev, azum, dis):
    if dis <= 1e-10:
        dis = 0.5

    return rotation_theta(theta) @ get_transformation_matrix(azum, elev, dis)[0:3, 0:3]


def loss_fun0(sim):
    return torch.ones(1, device=device) - torch.mean(sim)


if args.turn_off_clutter:
    def loss_fun(obj_s, clu_s):
        return torch.ones(1, device=device) - torch.mean(obj_s)
else:
    def loss_fun(obj_s, clu_s):
        return torch.ones(1, device=device) - (torch.mean(torch.max(obj_s, clu_s)) - torch.mean(clu_s))


def get_pre_render_samples(azum_samples, elev_samples, theta_samples, device='cpu'):
    with torch.no_grad():
        get_c = []
        get_theta = []
        get_samples = [[azum_, elev_, theta_] for azum_ in azum_samples for elev_ in elev_samples for theta_ in theta_samples]
        out_maps = []
        for sample_ in get_samples:
            theta_ = torch.ones(1, device=device) * sample_[2]
            C = camera_position_from_spherical_angles(set_distance, sample_[1], sample_[0], degrees=False, device=device)

            projected_map = inter_module(C, theta_)
            out_maps.append(projected_map)
            get_c.append(C.detach())
            get_theta.append(theta_)

        get_c_ = torch.Tensor(len(get_samples), 3).type(get_c[0].dtype).to(device)
        torch.cat(get_c, out=get_c_)

        get_theta = torch.cat(get_theta)
    return out_maps, get_c, get_theta


def get_init_pos_rendered(samples_map, samples_pos, samples_theta, predicted_map, clutter_score=None, device='cpu'):
    with torch.no_grad():
        get_loss = []
        for projected_map in samples_map:
            object_score = torch.sum(projected_map.contiguous() * predicted_map.unsqueeze(0), dim=1)

            if clutter_score is None:
                get_loss.append(loss_fun0(object_score).unsqueeze(0))
            else:
                get_loss.append(loss_fun(object_score, clutter_score).unsqueeze(0))

        get_loss = torch.cat(get_loss, dim=0)

        use_indexes = torch.min(get_loss, dim=0)[1]

    # [n_mesh, 3], [n_mesh]
    return samples_pos[use_indexes], samples_theta[use_indexes]


def get_init_pos(azum_samples, elev_samples, theta_samples, predicted_map, clutter_score=None, device='cpu'):
    get_samples = [[azum_, elev_, theta_] for azum_ in azum_samples for elev_ in elev_samples for theta_ in theta_samples]
    get_c = []
    get_loss = []
    get_theta = []
    for sample_ in get_samples:
        theta_ = torch.ones(1, device=device) * sample_[2]
        C = camera_position_from_spherical_angles(set_distance, sample_[1], sample_[0], degrees=False, device=device)
        projected_map = inter_module(C, theta_).squeeze()
        object_score = torch.sum(projected_map * predicted_map, dim=0)
        get_c.append(C.detach())
        get_theta.append(theta_)
        if clutter_score is None:
            get_loss.append(loss_fun0(object_score))
        else:
            get_loss.append(loss_fun(object_score, clutter_score))

    return get_c[int(np.argmin(get_loss))], get_theta[int(np.argmin(get_loss))]


def normalize(x, dim=0):
    return x / torch.sum(x ** 2, dim=dim, keepdim=True)[0] ** .5


if __name__ == '__main__':
    occ_level_s = ['', 'FGL1_BGL1_', 'FGL2_BGL2_', 'FGL3_BGL3_'][0:level]
    cate = args.type_
    mesh_d = args.mesh_d
    train_at = False
    mesh_path = '../data/PASCAL3D+_release1.1/CAD_%s/' % mesh_d + cate
    mesh_path_reference_sub = '../data/PASCAL3D+_release1.1/CAD/' + cate

    # record_names = 'resunetpre_3D512_points1saved_model_%s_799_%s_azum_TFFTTFFT_using_TFFTTFFT.npz'
    record_names = 'resunetpre_3D512_points1saved_model_%s_799_%s' + args.record_pendix + '.npz'

    anno_path = '../data/PASCAL3D_NeMo/annotations/%s/' % cate
    for_ps = False
    record_file_path_ = None

    down_smaple_rate = 8
    lr = 5e-2
    epochs = 300

    thrs = [np.pi / 6, np.pi / 18]

    device = 'cuda:0'

    image_sizes = {'car': (256, 672), 'bus': (320, 800), 'motorbike': (512, 512), 'boat': (480, 1120),
                   'bicycle': (608, 608), 'aeroplane': (320, 1024), 'sofa': (352, 736), 'tvmonitor': (480, 480),
                   'chair': (544, 384), 'diningtable': (320, 800), 'bottle': (512, 736), 'train': (256, 608)}

    distance_render = {'car': 5, 'bus': 5.2, 'motorbike': 4.5, 'bottle': 5.75, 'boat': 8, 'bicycle': 5.2, 'aeroplane': 7,
                       'sofa': 5.4, 'tvmonitor': 5.5, 'chair': 4, 'diningtable': 7, 'train': 3.75}

    print('Record: ', record_names)
    print('Cate: ', cate, ' mesh_d:', mesh_d)
    for occ_level_ in occ_level_s:
        print('occ_level:', occ_level_)
        if record_file_path_ is None:
            if for_ps:
                record_file_path = './saved_features/' + cate + '/' + record_names % (cate, mesh_d)

            else:
                if len(occ_level_) > 0:
                    record_file_path = './saved_features/' + cate + '_occ/' + occ_level_ + record_names % (cate, mesh_d)
                else:
                    record_file_path = './saved_features/' + cate + '/' + record_names % (cate, mesh_d)
        else:
            record_file_path = record_file_path_

        set_distance = distance_render[cate]

        render_image_size = max(image_sizes[cate]) // down_smaple_rate
        subtypes = ['mesh%02d' % i for i in range(1, 1 + len(os.listdir(mesh_path)))]
        record_file = np.load(record_file_path)

        total_error = []
        subtype_error = [[] for _ in range(len(os.listdir(mesh_path_reference_sub)))]

        mesh_path_ = mesh_path + '/%02d.off'

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
        map_shape = (image_sizes[cate][0] // down_smaple_rate, image_sizes[cate][1] // down_smaple_rate)

        azum_sample = np.linspace(0, np.pi * 2, 13)
        elev_sample = np.linspace(- np.pi / 6, np.pi / 3, 4)
        theta_sample = np.linspace(- np.pi / 6, np.pi / 6, 3)

        for k, subtype in enumerate(subtypes):
            xvert, xface = load_off(mesh_path_ % (k + 1), to_torch=True)
            name_list = record_file['names_%s' % subtype]
            feature_bank = torch.from_numpy(record_file['memory_%s' % subtype])
            clutter_bank = torch.from_numpy(record_file['clutter_%s' % subtype])
            inter_module = MeshInterpolateModule(xvert, xface, feature_bank, rasterizer, post_process=center_crop_fun(map_shape, (render_image_size, ) * 2))
            inter_module = inter_module.cuda()
            clutter_bank = clutter_bank.cuda()
            clutter_bank = normalize(torch.mean(clutter_bank, dim=0)).unsqueeze(0)

            if args.pre_render:
                maps_sample, c_sample, t_sample = get_pre_render_samples(azum_sample, elev_sample, theta_sample, device=device)
            else:
                maps_sample, c_sample, t_sample = None, None, None

            print('Start subtype: %s totally %d images.' % (subtype, len(name_list.tolist())))

            for image_name in name_list:
                # Should Not Happens
                if image_name not in record_file.keys():
                    print('Miss: ', image_name)
                    continue

                predicted_map = record_file[image_name]
                predicted_map = torch.from_numpy(predicted_map).to(device)
                clutter_score = torch.nn.functional.conv2d(predicted_map.unsqueeze(0), clutter_bank.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)

                if maps_sample is None:
                    C, theta = get_init_pos(azum_sample, elev_sample, theta_sample, predicted_map, clutter_score=clutter_score, device=device)
                else:
                    C, theta = get_init_pos_rendered(maps_sample, c_sample, t_sample, predicted_map, clutter_score=clutter_score, device=device)

                C = torch.nn.Parameter(C, requires_grad=True)
                theta = torch.nn.Parameter(theta, requires_grad=True)

                if train_at:
                    at_ = torch.nn.Parameter(torch.zeros([1, 3]).to(device), requires_grad=True)
                    optim = torch.optim.Adam(params=[C, theta, at_], lr=lr, betas=(0.4, 0.6))
                else:
                    at_ = None
                    optim = torch.optim.Adam(params=[C, theta], lr=lr, betas=(0.4, 0.6))
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.2)

                records = []

                for epoch in range(epochs):
                    if train_at:
                        projected_map = inter_module(C, theta, at=at_).squeeze()
                    else:
                        projected_map = inter_module(C, theta).squeeze()
                    object_score = torch.sum(projected_map * predicted_map, dim=0)
                    loss = loss_fun(object_score, clutter_score)

                    loss.backward()
                    # with torch.no_grad():
                    #     angel_gradient_modifier(C, alpha=(0.0, 1.0))

                    optim.step()
                    optim.zero_grad()
                    # print(loss.item())
                    distance_pred, elevation_pred, azimuth_pred = camera_position_to_spherical_angle(C)
                    records.append([theta.item(), elevation_pred.item(), azimuth_pred.item(), distance_pred.item()])
                    if (epoch + 1) % 100 == 0:
                        scheduler.step(None)

                distance_pred, elevation_pred, azimuth_pred = camera_position_to_spherical_angle(C)

                theta_pred, distance_pred, elevation_pred, azimuth_pred = theta.item(), distance_pred.item(), elevation_pred.item(), azimuth_pred.item()

                fl_anno = np.load(os.path.join(anno_path, image_name + '.npz'), allow_pickle=True)
                theta_anno, elevation_anno, azimuth_anno, distance_anno = get_anno(fl_anno, 'theta', 'elevation',
                                                                                   'azimuth', 'distance')
                anno_matrix = cal_rotation_matrix(theta_anno, elevation_anno, azimuth_anno, distance_anno)
                pred_matrix = cal_rotation_matrix(theta_pred, elevation_pred, azimuth_pred, distance_pred)

                if np.any(np.isnan(anno_matrix)) or np.any(np.isnan(pred_matrix)) or np.any(np.isinf(anno_matrix)) or np.any(np.isinf(pred_matrix)):
                    error_ = np.pi / 2
                error_ = cal_err(anno_matrix, pred_matrix)

                cad_idx = fl_anno['cad_index']
                subtype_error[cad_idx - 1].append(error_)
                total_error.append(error_)

        for thr in thrs:
            print('Thr: ', thr)
            # print('Subtype\tnimg\tacc')
            # for i, error_list in enumerate(subtype_error):
            #     print('Mesh%02d\t%d\t%.3f' % (i + 1, len(error_list), float(np.mean(np.array(error_list) < thr))))

            print('Average\t%d\t%.3f' % (len(total_error), float(np.mean(np.array(total_error) < thr))))

        print('Med')
        # print('Subtype\tnimg\tacc')
        # for i, error_list in enumerate(subtype_error):
        #     print('Mesh%02d\t%d\t%.3f' % (i + 1, len(error_list), float(np.median(np.array(error_list)))))

        print('Average\t%d\t%.3f' % (len(total_error), float(180 / np.pi * np.median(np.array(total_error)))))
