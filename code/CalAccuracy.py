import numpy as np
import argparse


parser = argparse.ArgumentParser(description='NeMo Calculate Accuracy')
parser.add_argument('--load_accuracy', default='', type=str, help='')
parser.add_argument('--data_pendix', default='', type=str, help='')

args = parser.parse_args()

cates = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
cates1 = ['plane', 'bike', 'boat', 'bottle', 'bus', 'car', 'chair', 'table', 'mbike', 'sofa', 'train', 'tv']

out_ = np.zeros((3, len(cates)), dtype=np.float32)

for i, cate in enumerate(cates):
    this_record = np.load(args.load_accuracy + '/%s%s.npz' % (cate, args.data_pendix), allow_pickle=True)
    out_[0, i] = float(np.mean(np.array(total_error) < np.pi / 6)) * 100
    out_[1, i] = float(np.mean(np.array(total_error) < np.pi / 18)) * 100
    out_[2, i] = float(180 / np.pi * np.median(np.array(total_error))

print('Metric | ', end='')
for cate1 in cates:
    print(cate, end='\t')
print()

print('Pi/6   | ', end='')
for i, _ in enumerate(cates):
    print('%.1f' % out_[0, i], end='\t')
print()

print('Pi/18  | ', end='')
for i, _ in enumerate(cates):
    print('%.1f' % out_[1, i], end='\t')
print()

print('MedErr | ', end='')
for i, _ in enumerate(cates):
    print('%.1f' % out_[2, i], end='\t')
print()
