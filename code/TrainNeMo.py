import torch
import torch.utils.data
import torchvision.transforms as transforms

from dataset.Pascal3DPlus import ToTensor, Normalize, Pascal3DPlus
from models.FeatureBanks import NearestMemoryManager, mask_remove_near
from models.KeypointRepresentationNet import NetE2E
from datetime import datetime
import os
import argparse
from lib.get_n_list import get_n_list

global args
parser = argparse.ArgumentParser(description='CoKe Training for NeMo')

parser.add_argument('--local_size', default=1, type=int, help='')
parser.add_argument('--d_feature', default=128, type=int, help='')
parser.add_argument('--batch_size', default=108, type=int, help='')
parser.add_argument('--workers', default=8, type=int, help='')
parser.add_argument('--total_epochs', default=800, type=int, help='')
parser.add_argument('--distance_thr', default=48, type=int, help='')
parser.add_argument('--T', default=0.07, type=float, help='')
parser.add_argument('--weight_noise', default=5e-3, type=float, help='')
parser.add_argument('--update_lr_epoch_n', default=10, type=int, help='')
parser.add_argument('--update_lr_', default=0.2, type=float, help='')
parser.add_argument('--lr', default=0.0001, type=float, help='')
parser.add_argument('--momentum', default=0.9, type=float, help='')
parser.add_argument('--train_accumulate', default=10, type=int, help='')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='')
parser.add_argument('--type_', default='car', type=str, help='')
parser.add_argument('--num_noise', default=5, type=int, help='')
parser.add_argument('--max_group', default=512, type=int, help='')
parser.add_argument('--adj_momentum', default=0.9, type=float, help='')
parser.add_argument('--backbone', default='resnetext', type=str)
parser.add_argument('--mesh_path', default='../PASCAL3D/PASCAL3D+_release1.1/CAD_%s/%s/', type=str, help='')
parser.add_argument('--save_dir', default='../3DrepresentationData/trained_resnetext_%s/', type=str, help='')
parser.add_argument('--root_path', default='../PASCAL3D/PASCAL3D_train_NeMo/', type=str, help='')
parser.add_argument('--mesh_d', default='single', type=str)
parser.add_argument('--sperate_bank', default='True', type=str)
parser.add_argument('--azum_sel', default='', type=str)

args = parser.parse_args()

mesh_d = args.mesh_d

n_gpus = torch.cuda.device_count()

args.local_size = [args.local_size, args.local_size]
if '%s' in args.mesh_path:
    args.mesh_path = args.mesh_path % (mesh_d, args.type_)
if '%s' in args.save_dir:
    args.save_dir = args.save_dir % mesh_d

sperate_bank = (args.sperate_bank == 'True' or args.sperate_bank == 'true')

n_list = get_n_list(args.mesh_path)
subtypes = ['mesh%02d' % (i + 1) for i in range(len(n_list))]

os.makedirs(args.save_dir, exist_ok=True)

# net = NetE2E(net_type='resnet50', local_size=args.local_size,
#              output_dimension=args.d_feature, reduce_function=None, n_noise_points=args.num_noise, pretrain=True, noise_on_mask=False)
net = NetE2E(net_type=args.backbone, local_size=args.local_size,
             output_dimension=args.d_feature, reduce_function=None, n_noise_points=args.num_noise, pretrain=True, noise_on_mask=False)
net.train()
if sperate_bank:
    net = torch.nn.DataParallel(net, device_ids=[i for i in range(n_gpus - 1)]).cuda()
else:
    net = torch.nn.DataParallel(net).cuda()

bank_set = []
dataloader_set = []

transforms = transforms.Compose([
    ToTensor(),
    Normalize(),
])


unseen_setting = len(args.azum_sel) != 0
if unseen_setting:
    azum_sel = args.azum_sel

    args.save_dir = args.save_dir.strip('/') + '_azum_' + azum_sel + '/'
else:
    azum_sel = ''

n_img_all = []
for n, subtype in zip(n_list, subtypes):
    memory_bank = NearestMemoryManager(inputSize=args.d_feature, outputSize=n + args.num_noise * args.max_group,
                                       K=1, num_noise=args.num_noise, num_pos=n, momentum=args.adj_momentum)
    if sperate_bank:
        ext_gpu = 'cuda:%d' % (n_gpus - 1)
        memory_bank = memory_bank.cuda(ext_gpu)
    else:
        ext_gpu = ''
        memory_bank = memory_bank.cuda()

    if len(azum_sel) > 0:
        list_path = 'lists3D_%s_azum_%s' % (mesh_d, azum_sel)
    else:
        list_path = 'lists3D_%s' % mesh_d
    anno_path = 'annotations3D_%s' % mesh_d

    Pascal3D_dataset = Pascal3DPlus(transforms=transforms, rootpath=args.root_path, imgclass=args.type_,
                                      subtypes=[subtype], mesh_path=args.mesh_path, anno_path=anno_path,
                                      list_path=list_path, weighted=True)

    n_img_all.append(len(Pascal3D_dataset))

    # In case there is no image in such subtype.
    if len(Pascal3D_dataset) == 0:
        bank_set.append(memory_bank)
        dataloader_set.append(None)
        continue

    Pascal3D_dataloader = torch.utils.data.DataLoader(Pascal3D_dataset, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=args.workers)

    bank_set.append(memory_bank)
    dataloader_set.append(Pascal3D_dataloader)

criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()

iter_num = 0
# optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def save_checkpoint(state, filename):
    file = os.path.join(args.save_dir, filename)
    torch.save(state, file)


print('Categroy:', args.type_, ' Number of Training Image:', sum(n_img_all))
print('Start Training!')
for epoch in range(args.total_epochs):
    if (epoch - 1) % args.update_lr_epoch_n == 0:
        lr = args.lr * args.update_lr_
        # optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        for param_group in optim.param_groups:
            param_group['lr'] = lr

    for idx_bank, memory_bank, this_dataloader in zip(range(len(bank_set)), bank_set, dataloader_set):
        if this_dataloader is None:
            continue
        for i, sample in enumerate(this_dataloader):
            # measure data loading time
            img, keypoint, iskpvisible, this_name, box_obj = sample['img'], sample['kp'], sample['iskpvisible'], sample['this_name'], sample['box_obj']
            obj_mask = sample['obj_mask']

            img = img.cuda()
            keypoint = keypoint.cuda()
            iskpvisible = iskpvisible.cuda()
            obj_mask = obj_mask.cuda()

            y_num = n_list[idx_bank]
            index = torch.Tensor([[k for k in range(y_num)]] * img.shape[0])
            index = index.cuda()

            features = net.forward(img, keypoint_positions=keypoint, obj_mask=1 - obj_mask)

            # get: [n, k, l]
            if sperate_bank:
                get, y_idx, noise_sim = memory_bank(features.to(ext_gpu), index.to(ext_gpu), iskpvisible.to(ext_gpu))
            else:
                get, y_idx, noise_sim = memory_bank(features, index, iskpvisible)

            get /= args.T

            mask_distance_legal = mask_remove_near(keypoint, thr=args.distance_thr, num_neg=args.num_noise * args.max_group,
                                                   dtype_template=get, neg_weight=args.weight_noise)

            iskpvisible_float = iskpvisible
            iskpvisible = iskpvisible.type(torch.bool).to(iskpvisible.device)

            loss = criterion(
                ((get.view(-1, get.shape[2]) - mask_distance_legal.view(-1, get.shape[2])))[iskpvisible.view(-1), :],
                y_idx.view(-1)[iskpvisible.view(-1)])

            loss = torch.mean(loss)
            
            loss_main = loss.item()
            if args.num_noise > 0 and True:
                loss_reg = torch.mean(noise_sim) * 0.1
                loss += loss_reg
            else:
                loss_reg = torch.zeros(1)

            loss.backward()
            if iter_num % args.train_accumulate == 0:
                optim.step()
                optim.zero_grad()
                print('n_iter', iter_num, 'epoch', epoch, 'loss', '%.5f' % loss_main, 'loss_reg', '%.5f' % loss_reg.item())
            iter_num += 1
        # torch.cuda.empty_cache()

    if epoch % 200 == 199:
        save_checkpoint(
        {'state': net.state_dict(), 'memory': [mem.memory for mem in bank_set], 'timestamp': int(datetime.timestamp(datetime.now())),
         'args': args}, 'saved_model_%s_%02d.pth' % (args.type_, epoch))




