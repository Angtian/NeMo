import numpy as np
import BboxTools as bbt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.FeatureBanks import NearestMemoryManager
from models.KeypointRepresentationNet import NetE2E, net_stride
import argparse
from lib.get_n_list import get_n_list

from lib.MeshUtils import *
import os
import argparse
from scipy.linalg import logm
from tqdm import tqdm

import pdb

parser = argparse.ArgumentParser(description='CoKe Feature Extraction for NeMo')

parser.add_argument('--local_size', default=1, type=int, help='')
parser.add_argument('--d_feature', default=128, type=int, help='')
parser.add_argument('--type_', default='car', type=str, help='')
parser.add_argument('--img_path', type=str,
                    help='Path to folder contains images')
parser.add_argument('--ckpt', default='../saved_model_car_799.pth', type=str)
parser.add_argument('--backbone', default='resnetext', type=str)
parser.add_argument('--mesh_d', default='single', type=str)
parser.add_argument('--no_reload', action='store_true')
parser.add_argument('--num_noise', default=0, type=int, help='')

parser.add_argument('--turn_off_clutter', default=False, type=bool, help='')
parser.add_argument('--objectnet', default=False, type=bool, help='')
parser.add_argument('--record_pendix', default='', type=str, help='')
parser.add_argument('--pre_render', default=True, type=bool, help='')

parser.add_argument('--save_final_pred', default='./final_pred.npz', type=str, help='')
parser.add_argument('--mesh_path', default='../car/', type=str, help='')
parser.add_argument('--total_epochs', default=300, type=int, help='')
parser.add_argument('--lr', default=5e-2, type=float, help='')
parser.add_argument('--adam_beta_0', default=0.4, type=float, help='')
parser.add_argument('--adam_beta_1', default=0.6, type=float, help='')

parser.add_argument('--vertical_shift', default=30, type=int, help='')
parser.add_argument('--tar_horizontal_size', default=435, type=int, help='')

args = parser.parse_args()


def apply_pad(box_: bbt.Bbox2D, tar_img: np.ndarray, **kwargs):
    out_shape = box_.shape

    if out_shape[0] // 2 - box_.center[0] > 0 or out_shape[1] // 2 - box_.center[1] > 0 or out_shape[0] // 2 + \
            box_.center[
                0] - tar_img.shape[0] > 0 or out_shape[1] // 2 + box_.center[1] - tar_img.shape[1] > 0:
        if len(tar_img.shape) == 2:
            padding = (
                (max(out_shape[0] // 2 - box_.center[0], 0),
                 max(out_shape[0] // 2 + box_.center[0] - tar_img.shape[0], 0)),
                (max(out_shape[1] // 2 - box_.center[1], 0),
                 max(out_shape[1] // 2 + box_.center[1] - tar_img.shape[1], 0)))
        else:
            padding = (
                (max(out_shape[0] // 2 - box_.center[0], 0),
                 max(out_shape[0] // 2 + box_.center[0] - tar_img.shape[0], 0)),
                (max(out_shape[1] // 2 - box_.center[1], 0),
                 max(out_shape[1] // 2 + box_.center[1] - tar_img.shape[1], 0)),
                (0, 0))

        img_out = np.pad(tar_img, padding, **kwargs)
        box_ = box_.shift([padding[0][0], padding[1][0]])
        return box_.apply(img_out)

    else:
        return box_.apply(tar_img)


class UnlabeledCars(Dataset):
    def __init__(self, img_path='', image_list=None, transform=None):
        if image_list is None:
            all_imgs = os.listdir(img_path)
        else:
            all_imgs = [t.strip() + '.jpg' for t in open(image_list).readlines()]
        self.all_imgs = [t.split('.')[0] for t in all_imgs]
        self.img_path = img_path
        print(transform)
        self.transform = transform

    def __getitem__(self, item):
        img_name = self.all_imgs[item]
        img_ = Image.open(os.path.join(self.img_path, img_name + '.jpg')).convert('RGB')

        return self.transform(img_), img_name

    def __len__(self):
        return len(self.all_imgs)


class CustomedCrop(object):
    def __init__(self, crop_size, tar_horizontal):
        self.crop_size = crop_size
        self.tar_horizontal = tar_horizontal

    def __call__(self, im):
        size_ = im.size
        out_size = (self.tar_horizontal, int(size_[1] / size_[0] * self.tar_horizontal),)
        img = np.array(im.resize(out_size))
        crop_box = bbt.box_by_shape(self.crop_size, bbt.full(img).center, ).shift((args.vertical_shift, 0))
        cropped_img = apply_pad(crop_box, img)
        return Image.fromarray(cropped_img)


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


def get_pre_render_samples(distance_samples, azum_samples, elev_samples, theta_samples, device='cpu'):
    with torch.no_grad():
        get_c = []
        get_theta = []
        get_samples = [[azum_, elev_, theta_, dist_, ] for azum_ in azum_samples for elev_ in elev_samples for theta_ in
                       theta_samples for dist_ in distance_samples]
        out_maps = []
        for sample_ in get_samples:
            theta_ = torch.ones(1, device=device) * sample_[2]
            C = camera_position_from_spherical_angles(sample_[3], sample_[1], sample_[0], degrees=False, device=device)

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


def normalize(x, dim=0):
    return x / torch.sum(x ** 2, dim=dim, keepdim=True)[0] ** .5


##########################################################################
mesh_d = args.mesh_d

thr = 0.1

device = 'cuda:0'

cate = args.type_
image_size = (256, 672)

if '%s' in args.ckpt:
    args.ckpt = args.ckpt % args.type_

args.local_size = [args.local_size, args.local_size]

# SingleCuboid: 1, MultiCuboid: number of subtypes
n_list = get_n_list(args.mesh_path)
subtypes = ['mesh%02d' % (i + 1) for i in range(len(n_list))]

# net = NetE2E(net_type='resnet50', local_size=args.local_size,
#              output_dimension=args.d_feature, reduce_function=None, n_noise_points=args.num_noise, pretrain = True)
net = NetE2E(net_type=args.backbone, local_size=args.local_size,
             output_dimension=args.d_feature, reduce_function=None, n_noise_points=args.num_noise, pretrain=True)

net = torch.nn.DataParallel(net).cuda()
net.eval()

transforms_ = transforms.Compose([
    CustomedCrop(image_size, args.tar_horizontal_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

checkpoint = torch.load(args.ckpt, map_location='cuda:0')
net.load_state_dict(checkpoint['state'])

get = {}


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


print('Extract Features, category: ', args.type_)
subtype = subtypes[0]
bank_mems = checkpoint['memory'][0][0:n_list[0]]
clut_mems = checkpoint['memory'][0][n_list[0]::]

dataset_ = UnlabeledCars(img_path=args.img_path, transform=transforms_)
dataloader_ = DataLoader(dataset=dataset_, num_workers=0, batch_size=1, shuffle=False)

print('number images:', len(dataset_))

########################   MeshPoseSolve   ###################
down_smaple_rate = 8
render_image_size = max(image_size) // down_smaple_rate

lr = args.lr
epochs = args.total_epochs

mesh_path_ = args.mesh_path + '/%02d.off'
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
map_shape = (image_size[0] // down_smaple_rate, image_size[1] // down_smaple_rate)

feature_bank = bank_mems.detach().cpu()
clutter_bank = clut_mems.detach().cpu()

xvert, xface = load_off(mesh_path_ % 1, to_torch=True)
inter_module = MeshInterpolateModule(xvert, xface, feature_bank, rasterizer,
                                     post_process=center_crop_fun(map_shape, (render_image_size,) * 2))
inter_module = inter_module.cuda()
clutter_bank = clutter_bank.cuda()
clutter_bank = normalize(torch.mean(clutter_bank, dim=0)).unsqueeze(0)

azum_sample = np.linspace(0, np.pi * 2, 12, endpoint=False)
elev_sample = np.linspace(- np.pi / 6, np.pi / 3, 4)
theta_sample = np.array([0.])
dist_sample = np.array([4.4, 5, 5.7])

maps_sample, c_sample, t_sample = get_pre_render_samples(dist_sample, azum_sample, elev_sample, theta_sample,
                                                         device=device)

# [2, 3]
translation_multi = torch.Tensor([[1, 0, 0], [0, 1, 0]]).to(device)

########################### start loop ###########################
all_preds = dict()

if not args.no_reload and os.path.exists(args.save_final_pred):
    all_preds = dict(np.load(args.save_final_pred, allow_pickle=True))

final_res = []
for j, (img, name_) in enumerate(tqdm(dataloader_)):
    if name_[0] in all_preds.keys():
        continue

    with torch.no_grad():
        img = img.cuda()

        predicted_map = net.module.forward_test(img)

        clutter_score = torch.nn.functional.conv2d(predicted_map,
                                                   clutter_bank.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)

    clutter_score = clutter_score.detach()
    predicted_map = predicted_map.detach()
    C, theta = get_init_pos_rendered(maps_sample, c_sample, t_sample, predicted_map, clutter_score=clutter_score,
                                     device=device)

    trans_2d = torch.zeros((1, 2)).to(device)

    C = torch.nn.Parameter(C, requires_grad=True)
    theta = torch.nn.Parameter(theta, requires_grad=True)
    trans_2d = torch.nn.Parameter(trans_2d, requires_grad=True)

    optim = torch.optim.Adam(params=[C, theta, trans_2d], lr=lr, betas=(args.adam_beta_0, args.adam_beta_1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.2)

    for epoch in range(epochs):
        projected_map = inter_module(C, theta, extra_trans=trans_2d @ translation_multi).squeeze()
        object_score = torch.sum(projected_map * predicted_map, dim=0)
        loss = loss_fun(object_score, clutter_score)

        loss.backward()
        # with torch.no_grad():
        #     angel_gradient_modifier(C, alpha=(0.0, 1.0))

        optim.step()
        optim.zero_grad()
        if (epoch + 1) % 100 == 0:
            scheduler.step(None)

    distance_pred, elevation_pred, azimuth_pred = camera_position_to_spherical_angle(C)
    theta_pred, distance_pred, elevation_pred, azimuth_pred = theta.item(), distance_pred.item(), elevation_pred.item(), azimuth_pred.item()

    preds = np.array([distance_pred, theta_pred, elevation_pred, azimuth_pred])
    preds = np.concatenate((preds, trans_2d.detach().cpu().numpy().ravel()))

    all_preds[name_[0]] = preds

    if (j + 1) % 10000 == 0:
        np.savez(args.save_final_pred, **all_preds)

np.savez(args.save_final_pred, **all_preds)





