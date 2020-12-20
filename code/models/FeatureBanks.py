import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn
import math


def one_hot(y, max_size=None):
    if not max_size:
        max_size = int(torch.max(y).item() + 1)
    # print('y.shape',y.shape)
    # print('max_size',max_size)
    y = y.view(-1, 1)
    y_onehot = torch.zeros((y.shape[0], max_size), dtype=torch.float32, device=y.device)
    y_onehot.scatter_(1, y.type(torch.long), 1)
    return y_onehot


def to_mask(y, max_size):
    y_onehot = torch.zeros((len(y), max_size), dtype=torch.float32, device=y[0].device)
    for i in range(len(y)):
        y_onehot[i].scatter_(0, y[i].type(torch.long), 1)
    return y_onehot


def mask_remove_near(keypoints, thr, dtype_template=None, num_neg=0, neg_weight=1, eps=1e5):
    if dtype_template is None:
        dtype_template = torch.ones(1, dtype=torch.float32)
    # keypoints -> [n, k, 2]
    with torch.no_grad():
        # distance -> [n, k, k]
        distance = torch.sum((torch.unsqueeze(keypoints, dim=1) - torch.unsqueeze(keypoints, dim=2)).pow(2), dim=3).pow(0.5)
        if num_neg == 0:
            return ((distance <= thr).type_as(dtype_template) - torch.eye(keypoints.shape[1]).type_as(dtype_template).unsqueeze(dim=0)) * eps
        else:
            tem = (distance <= thr).type_as(dtype_template) - torch.eye(keypoints.shape[1]).type_as(dtype_template).unsqueeze(dim=0)
            return torch.cat([tem * eps, - torch.ones(keypoints.shape[0: 2] + (num_neg, )).type_as(dtype_template) * math.log(neg_weight)], dim=2)


class NearestMemorySelective(nn.Module):
    def forward(self, x, y, visible, n_pos, n_neg, lru, memory, params, eps=1e-8):
        group_size = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()

        similarity = torch.sum(torch.unsqueeze(x[0:n_pos], 1) * torch.unsqueeze(memory, 0), dim=2)

        n_class = n_pos // group_size
        y_onehot = one_hot(y, n_class)

        if not group_size == 1:
            y_onehot = y_onehot.unsqueeze(2).expand(-1, -1, group_size).contiguous().view(y.shape[0], -1)

        y_idx = torch.argmax(similarity[:, 0:n_pos] + y_onehot * 2, dim=1).type(torch.long)
        visible = to_mask(visible, n_pos)

        with torch.no_grad():
            # update memory keypoints
            # [n, k]
            idx_onehot = one_hot(y_idx, n_pos)

            # [k, d]
            get = torch.mm(torch.t(idx_onehot), x[0:n_pos, :])
            counts = torch.t(torch.sum(idx_onehot, dim=0, keepdim=True))
            valid_mask = (counts > 0.1).type(counts.dtype) * visible.view(-1, 1)
            get /= counts + eps

            memory[0:n_pos, :].copy_(F.normalize(
                memory[0:n_pos, :] * (valid_mask * momentum + 1 - valid_mask) + get * (1 - momentum) * valid_mask,
            dim=1, p=2))

            # Update trash bin
            memory[n_pos + lru * n_neg: n_pos + (lru + 1) * n_neg, :].copy_(x[n_pos::, :])

        return similarity, y_idx


class NearestMemoryManager(nn.Module):
    def __init__(self, inputSize, outputSize, K, num_pos, T=0.07, momentum=0.5, Z=None, max_groups=-1, num_noise=-1):
        super(NearestMemoryManager, self).__init__()
        self.nLem = outputSize
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)

        self.memory = torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        self.memory.requires_grad = False

        self.lru = 0
        if max_groups > 0:
            self.max_lru = max_groups
        else:
            self.max_lru = -1

        if num_noise < 0:
            self.num_noise = self.K
        else:
            self.num_noise = num_noise

        self.num_pos = num_pos

        self.accumulate_num = torch.zeros(self.num_pos, dtype=torch.long, device=self.memory.device)
        self.accumulate_num.requires_grad = False

    # x: feature: [128, 128], y: indexes [128] -- a batch of data's index directly from the dataloader.
    def forward(self, x, y, visible):
        n_pos = self.num_pos
        n_neg = self.num_noise

        if self.max_lru == -1 and n_neg > 0 and x.shape[0] <= (self.nLem - n_pos) / n_neg:
            self.max_lru = (self.memory.shape[0] - n_pos) // (n_neg * x.shape[0])

        group_size = int(self.params[0].item())
        momentum = self.params[3].item()

        assert group_size == 1, 'Currently only support group size = 1'

        # x [n, k, d] * memory [l, d] = similarity : [n, k, l]
        if n_neg == 0:
            similarity = torch.matmul(x, torch.transpose(self.memory, 0, 1))

            noise_similarity = torch.zeros(1)
        else:
            t_ = x[:, 0:n_pos, :]
            similarity = torch.matmul(t_, torch.transpose(self.memory, 0, 1))
            noise_similarity = torch.matmul(x[:, n_pos:, :], torch.transpose(self.memory[0:n_pos, :], 0, 1))

        n_class = n_pos // group_size

        with torch.set_grad_enabled(False):
            # [n, k, k]
            y_onehot = one_hot(y, n_class).view(x.shape[0], -1, n_class)

            # useless branch
            if not group_size == 1:
                # [n, k, k * g]
                y_onehot = y_onehot.unsqueeze(2).expand(-1, -1, group_size).contiguous().view(x.shape[0], -1, n_pos)
                visible = visible.unsqueeze(2).expand(-1, -1, group_size).contiguous().view(x.shape[0], -1, n_pos)

                y_idx = torch.argmax(similarity[:, :, 0:n_pos] + y_onehot * 2, dim=2).type(torch.long)
            else:
                y_idx = y.type(torch.long)

            # update memory keypoints
            # [n, k, d]
            get = torch.bmm(torch.transpose(y_onehot, 1, 2), x[:, 0:n_pos, :] * visible.type(x.dtype).view(*visible.shape, 1))

            # [k, d]
            get = torch.mean(get, dim=0) # / torch.sum(visible, dim=0).view(-1, 1)

            if n_neg > 0:
                if x.shape[0] > (self.nLem - n_pos) / n_neg:
                    self.memory = F.normalize(torch.cat([self.memory[0:n_pos, :] * momentum + get * (1 - momentum), x[:, n_pos::, :].contiguous().view(-1, x.shape[2])[0:self.memory.shape[0] - n_pos]], dim=0), dim=1, p=2)
                else:
                    neg_parts = torch.cat([self.memory[n_pos:n_pos + self.lru * n_neg * x.shape[0], :],
                                           x[:, n_pos::, :].contiguous().view(-1, x.shape[2]),
                                           self.memory[n_pos + (self.lru + 1) * n_neg * x.shape[0]::, :]], dim=0)

                    self.memory = F.normalize(torch.cat([self.memory[0:n_pos, :] * momentum + get * (1 - momentum), neg_parts], dim=0), dim=1, p=2)
            else:
                self.memory = F.normalize(self.memory[0:n_pos, :] * momentum + get * (1 - momentum), dim=1, p=2)

            self.accumulate_num += torch.sum((visible > 0).type(self.accumulate_num.dtype).to(self.accumulate_num.dtype), dim=0)
            self.lru += 1
            self.lru = self.lru % self.max_lru
        # out.shape: [d, n_neg + n_pos]
        return similarity, y_idx, noise_similarity

    def forward_local(self, x, y, visible):
        # print('x',x.shape)
        # print('noise',self.num_noise)
        n_pos = self.num_pos
        n_neg = self.num_noise

        if self.max_lru == -1 and n_neg > 0:
            self.max_lru = (self.memory.shape[0] - n_pos) // (n_neg * x.shape[0])

        group_size = int(self.params[0].item())
        momentum = self.params[3].item()

        assert group_size == 1, 'Currently only support group size = 1'

        # x [n, k, d] * memory [l, d] = similarity : [n, k, l]
        if n_neg == 0:
            t_ = x[:, 0:n_pos, :]

            t_neg = x[:, n_pos::, :]
            similarity = torch.matmul(t_, torch.transpose(self.memory, 0, 1))

            # [n, k, 1, d] * [n, 1, c, d] -> [n, k, c]
            similarity_neg = torch.sum(t_.unsqueeze(2) * t_neg.unsqueeze(1), dim=3)

            similarity = torch.cat([similarity, similarity_neg], dim=2)

            noise_similarity = torch.zeros(1)
        else:
            t_ = x[:, 0:n_pos, :]
            similarity = torch.matmul(t_, torch.transpose(self.memory, 0, 1))
            noise_similarity = torch.matmul(x[:, n_pos:, :], torch.transpose(self.memory[0:n_pos, :], 0, 1))

        n_class = n_pos // group_size

        with torch.set_grad_enabled(False):
            # [n, k, k]
            y_onehot = one_hot(y, n_class).view(x.shape[0], -1, n_class)

            if not group_size == 1:
                # [n, k, k * g]
                y_onehot = y_onehot.unsqueeze(2).expand(-1, -1, group_size).contiguous().view(x.shape[0], -1, n_pos)
                visible = visible.unsqueeze(2).expand(-1, -1, group_size).contiguous().view(x.shape[0], -1, n_pos)

                y_idx = torch.argmax(similarity[:, :, 0:n_pos] + y_onehot * 2, dim=2).type(torch.long)
            else:
                y_idx = y.type(torch.long)

            # update memory keypoints
            # [n, k, d]
            get = torch.bmm(torch.transpose(y_onehot, 1, 2), x[:, 0:n_pos, :] * visible.type(x.dtype).view(*visible.shape, 1))

            # [k, d]
            get = torch.mean(get, dim=0) # / torch.sum(visible, dim=0).view(-1, 1)

            self.accumulate_num += torch.sum(visible.type(self.accumulate_num.dtype), dim=0)
            self.lru += 1
            self.lru = self.lru % self.max_lru

        # out.shape -- torch.Size([128, 4097])
        return similarity, y_idx, noise_similarity

    def set_zero(self, n_pos):
        self.accumulate_num = torch.zeros(n_pos, dtype=torch.long, device=self.memory.device)
        self.memory.fill_(0)

    def accumulate_memory(self, x, y, visible, eps=1e-8):
        # print(self.params)
        group_size = int(self.params[0].item())

        # Currently only support group size = 1
        assert group_size == 1, 'Currently only support group size = 1'

        n_pos = self.num_pos
        n_neg = self.num_noise

        with torch.no_grad():
            # print(visible.shape)

            # update memory keypoints
            # [n, k, k]
            idx_onehot = one_hot(y, n_pos).view(x.shape[0], -1, n_pos)

            # [k, d]
            get = torch.bmm(torch.transpose(idx_onehot, 1, 2), x[:, 0:n_pos, :] * visible.type(x.dtype).view(*visible.shape, 1))
            get = torch.sum(get, dim=0)

            self.memory[0:n_pos, :].copy_(self.memory[0:n_pos, :] + get)
            self.accumulate_num += torch.sum(visible.type(self.accumulate_num.dtype), dim=0)

    def normalize_memory(self):
        self.memory.copy_(F.normalize(self.memory, p=2, dim=1))

    def cuda(self, device=None):
        super().cuda(device)
        self.accumulate_num = self.accumulate_num.cuda(device)
        self.memory = self.memory.cuda(device)
        return self