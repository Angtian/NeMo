import numpy as np
import argparse


parser = argparse.ArgumentParser(description='NeMo Calculate Accuracy')
parser.add_argument('--load_accuracy', default='', type=str, help='')
parser.add_argument('--data_pendix', default='', type=str, help='')

args = parser.parse_args()

cates = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
cates1 = ['plane', 'bike', 'boat', 'bottle', 'bus', 'car', 'chair', 'table', 'mbike', 'sofa', 'train', 'tv']

out_ = np.zeros((4, len(cates)), dtype=np.float32)

for i, cate in enumerate(cates):
    this_record = np.load(args.load_accuracy + '/%s%s.npz' % (cate, args.data_pendix), allow_pickle=True)
    total_error = this_record['total_error']
    out_[0, i] = float(np.mean(np.array(total_error) < np.pi / 6)) * 100
    out_[1, i] = float(np.mean(np.array(total_error) < np.pi / 18)) * 100
    out_[2, i] = float(180 / np.pi * np.median(np.array(total_error)))
    out_[3, i] = np.array(total_error).size

print('Metric | ', end='')
for cate in cates1:
    print(cate, end='\t')
print('Mean')

print('Pi/6   | ', end='')
for i, _ in enumerate(cates):
    print('%.1f' % out_[0, i], end='\t')
print('%.1f' % (np.sum(out_[3, :] * out_[0, :]) / np.sum(out_[3, :])))

print('Pi/18  | ', end='')
for i, _ in enumerate(cates):
    print('%.1f' % out_[1, i], end='\t')
print('%.1f' % (np.sum(out_[3, :] * out_[1, :]) / np.sum(out_[3, :])))

print('MedErr | ', end='')
for i, _ in enumerate(cates):
    print('%.1f' % out_[2, i], end='\t')
print('%.1f' % (np.sum(out_[3, :] * out_[2, :]) / np.sum(out_[3, :])))
