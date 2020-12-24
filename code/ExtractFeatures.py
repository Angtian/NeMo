import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
from dataset.Pascal3DPlus import ToTensor, Normalize, Pascal3DPlus
from models.FeatureBanks import NearestMemoryManager
from models.KeypointRepresentationNet import NetE2E, net_stride
import os
import argparse
import numpy as np
from lib.get_n_list import get_n_list


##########################################################################
global args
parser = argparse.ArgumentParser(description='CoKe Feature Extraction for NeMo')

parser.add_argument('--local_size', default=1, type=int, help='')
parser.add_argument('--d_feature', default=128, type=int, help='')
parser.add_argument('--n_points', default=-1, type=int, help='')
parser.add_argument('--batch_size', default=16, type=int, help='')
parser.add_argument('--workers', default=4, type=int, help='')
parser.add_argument('--type_', default='bottle', type=str, help='')
parser.add_argument('--num_noise', default=0, type=int, help='')
parser.add_argument('--max_group', default=16, type=int, help='')
parser.add_argument('--adj_momentum', default=0.9, type=float, help='')
parser.add_argument('--mesh_path', default='../PASCAL3D/PASCAL3D+_release1.1/CAD_%s/%s/', type=str, help='')
parser.add_argument('--save_dir', default='../3DrepresentationData/trained_resnetext_%s/', type=str, help='')
parser.add_argument('--root_path', default='../PASCAL3D/PASCAL3D_NeMo/', type=str, help='')
parser.add_argument('--data_pendix', default='', type=str, help='')
parser.add_argument('--ckpt', default='3D512_points1saved_model_%s_799.pth', type=str)

parser.add_argument('--backbone', default='resnetext', type=str)
parser.add_argument('--mesh_d', default='single', type=str)
parser.add_argument('--objectnet', default=False, type=bool)
parser.add_argument('--eval_kp_score', default=False, type=bool)
parser.add_argument('--save_features_path', default='saved_features', type=str)
parser.add_argument('--save_features_name', default='resnetext_%s_%s', type=str)
parser.add_argument('--azum_sel', default='', type=str)

args = parser.parse_args()

mesh_d = args.mesh_d

thr = 0.1

if args.save_features_name.count('%s') == 2:
    args.save_features_name = args.save_features_name % ('%s', mesh_d)
if '.npz' in args.save_features_name:
    args.save_features_name = args.save_features_name.strip('.npz')

# Generate to unseen pose
unseen_setting = len(args.azum_sel) != 0
if unseen_setting:
    azum_sel = args.azum_sel

    # Unseen
    use_azum_data = ''.join(['F' if t == 'T' else 'T' for t in args.azum_sel])

    # Seen
    # use_azum_data = args.azum_sel

    args.save_dir = args.save_dir.strip('/') + '_azum_' + azum_sel + '/'
else:
    azum_sel = ''
    use_azum_data = ''

args.mesh_path = args.mesh_path % (mesh_d, args.type_)
if '%s' in args.save_dir:
    args.save_dir = args.save_dir % mesh_d

if '%s' in args.ckpt:
    args.ckpt = args.ckpt % args.type_

if not args.objectnet:
    if len(args.data_pendix) == 0:
        if len(azum_sel) > 0:
            save_features = args.save_features_path + '/' + args.type_ + '/' + args.save_features_name % args.type_ + '_azum_%s_using_%s.npz' % (azum_sel, use_azum_data)
        else:
            save_features = args.save_features_path + '/' + args.type_ + '/' + args.save_features_name % args.type_ + '.npz'
    else:
        save_features = args.save_features_path + '/' + args.type_ + '_occ/' + args.save_features_name % args.type_ + '.npz'
else:
    save_features = args.save_features_path + '_objectnet/' + args.type_ + '/' + args.save_features_name % args.type_ + '.npz'

os.makedirs(args.save_features_path + '/' + args.type_, exist_ok=True)

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

transforms = transforms.Compose([
    ToTensor(),
    Normalize(),
])

criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

args.ckpt = os.path.join(args.save_dir, args.ckpt)

checkpoint = torch.load(args.ckpt)
net.load_state_dict(checkpoint['state'])

get = {}


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


print('Extract Features, category: ', args.type_)
for i, (n, subtype) in enumerate(zip(n_list, subtypes)):
    memory_bank = NearestMemoryManager(inputSize=args.d_feature, outputSize=n + args.num_noise * args.max_group,
                                       K=1, num_noise=args.num_noise, num_pos=n, momentum=args.adj_momentum)
    memory_bank = memory_bank.cuda()

    print('subtype: ', subtype, end='\t')
    with torch.no_grad():
        print('number points:', n, end='\t')
        memory_bank.memory.copy_(checkpoint['memory'][i][0:memory_bank.memory.shape[0]])

    if save_features is not None:
        get['memory_%s' % subtype] = checkpoint['memory'][i][0:memory_bank.memory.shape[0]].detach().cpu().numpy()
        get['clutter_%s' % subtype] = checkpoint['memory'][i][memory_bank.memory.shape[0]::].detach().cpu().numpy()
        get['names_%s' % subtype] = []

    if len(azum_sel) > 0:
        list_path = 'lists3D_%s_azum_%s' % (mesh_d, use_azum_data)
    else:
        list_path = 'lists3D_%s' % mesh_d
    anno_path = 'annotations3D_%s' % mesh_d

    Pascal3D_dataset = Pascal3DPlus(transforms=transforms, rootpath=args.root_path, imgclass=args.type_,
                                      subtypes=[subtype], mesh_path=args.mesh_path, anno_path=anno_path, 
                                      list_path=list_path, weighted=True, data_pendix=args.data_pendix)
    Pascal3D_dataloader = torch.utils.data.DataLoader(Pascal3D_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    all_visible = torch.zeros((n, ), dtype=torch.long)
    all_correct = torch.zeros((n, ), dtype=torch.long)

    print('number images:', len(Pascal3D_dataset))

    with torch.no_grad():
        final_res = []
        for j, sample in enumerate(Pascal3D_dataloader):
            img, keypoint, iskpvisible, this_name, box_obj = sample['img'], sample['kp'], sample['iskpvisible'], sample['this_name'], sample['box_obj']
            img = img.cuda()

            feature_map = net.module.forward_test(img)

            if save_features is not None:
                for i, n in enumerate(this_name):
                    get[n] = feature_map[i].detach().cpu().numpy()
                    get['names_%s' % subtype].append(n)

            # Evaluate PCK-kp
            # Only for evaluate dense keypoint detection ability of the backbone. Not necessary for NeMo
            if args.eval_kp_score:
                keypoint = keypoint.cuda()

                iskpvisible = iskpvisible > 0
                iskpvisible = iskpvisible.cuda()
                obj_mask = obj_mask.cuda()
                obj_mask = sample['obj_mask']
                hmap = F.conv2d(feature_map, memory_bank.memory.unsqueeze(2).unsqueeze(3))

                stride_ = net_stride[args.backbone]
                obj_mask = F.max_pool2d(obj_mask.unsqueeze(dim=1), kernel_size=stride_, stride=stride_, padding=(stride_ - 1) // 2)

                hmap = hmap * obj_mask     

                # [n, k, h,w]
                x_ = hmap.size(3)
                hmap = hmap.view(*hmap.shape[0:2], -1)
                
                _, max_ = torch.max(hmap, dim=2)
                max_idx = torch.zeros((*hmap.shape[0:2], 2), dtype=torch.long).to(hmap.device)
                max_idx[:, :, 0] = max_ // x_
                max_idx[:, :, 1] = max_ % x_
                
                max_idx = max_idx * stride_ + stride_ // 2
                
                # [n, k]
                distance = torch.sum((max_idx - keypoint) ** 2, dim=2) ** 0.5
                
                # [n, k]
                correct_keypoints = (distance <= thr * torch.max(box_obj[0], box_obj[1]).view(-1, 1).cuda()).type(torch.long).to(iskpvisible.device)
                
                # [k]
                correct_keypoints = torch.sum(iskpvisible * correct_keypoints, dim=0).cpu()
                
                # [k]
                visible_keypoints = torch.sum(iskpvisible, dim=0).cpu()
                
                all_visible += visible_keypoints.type(torch.long)
                all_correct += correct_keypoints.type(torch.long)

        if args.eval_kp_score:
            print('acc:', all_correct.type(torch.float32) / all_visible.type(torch.float32))
            print('avg:', nanmean(all_correct.type(torch.float32) / all_visible.type(torch.float32)))


def find_2nd(string, substring):
   return string.find(substring, string.find(substring) + 1)


if save_features is not None:
    os.makedirs(save_features[0:find_2nd(save_features, '/')], exist_ok=True)
    np.savez(save_features, **get)
