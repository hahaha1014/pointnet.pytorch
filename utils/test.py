from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import sys
sys.path.append("../")
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls
import time


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--class_choice', type=str, default='', help='class choice')
parser.add_argument('--num_points', type=int, default=2048, help='input num points')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

opt = parser.parse_args()
print(opt)

# d = ShapeNetDataset(
#     root=opt.dataset,
#     class_choice=[opt.class_choice],
#     split='test',
#     npoints=opt.num_points,
#     data_augmentation=False)

# idx = opt.idx

# print("model %d/%d" % (idx, len(d)))
# point, seg = d[idx]
# print(point.size(), seg.size())
# point_np = point.numpy()

# state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(k= 8)
# classifier.load_state_dict(state_dict)
classifier.eval().to('cuda:0')

# point = point.transpose(1, 0).contiguous()

# point = Variable(point.view(1, point.size()[0], point.size()[1]))
# print(point.shape)
point = torch.randn(24, 3, 2048).to('cuda:0')
total_time = 0

with torch.no_grad():
    for i in range(100):
        start_time = time.time()
        pred, _, _ = classifier(point)
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
    print("Average inference time: ", total_time/100*1000, end = ' ')
    print("ms")

pred_choice = pred.data.max(2)[1]
print(pred_choice)

