from __future__ import print_function
from show3d_balls import showpoints
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
import matplotlib.pyplot as plt
import time

test_file_path =  "result.txt"
test_data = np.loadtxt(test_file_path).astype(np.float32)

seg = np.loadtxt("gt.txt").astype(np.int64)
seg = seg - 1

point = test_data[:,:3]

pred_choice = test_data[:,3].astype(int)
pred_choice = pred_choice - 1

print(point.shape)

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg, :]

pred_choice = pred_choice.astype(int)
pred_choice = np.array([pred_choice])
#pred_choice = pred_choice.T
print(pred_choice.shape)
pred_color = cmap[pred_choice[0], :]

num_same_elements = np.sum(seg == pred_choice)
print("mIoU: ", num_same_elements/2500.0)

print(pred_color.shape)
showpoints(point, gt, pred_color)
