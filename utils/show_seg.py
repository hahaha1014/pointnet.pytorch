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


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, defaFult=0, help='model index')
parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--class_choice', type=str, default='', help='class choice')
parser.add_argument('--num_points', type=int, default=4096, help='input num points')

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    npoints=opt.num_points,
    data_augmentation=False)

idx = opt.idx

print("model %d/%d" % (idx, len(d)))
point, seg = d[idx]
print(point.size(), seg.size())
point_np = point.numpy()


color_map = {
    0: [128, 255, 0],    # 绿色
    1: [0, 255, 0],    # 绿色
    2: [255, 0, 0],    # 红色
    3: [0, 0, 255],    # 蓝色
    4: [255, 255, 0],  # 黄色
    5: [255, 165, 0],  # 橙色
    6: [255, 20, 147], # 粉红色
    7: [0, 255, 255],  # 青色
    8: [128, 0, 128]   # 紫色
}


cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]
seg_colors = np.array([color_map[val] for val in seg.numpy()-1])#####


state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval()

point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
total_time = 0
for i in range(50):
    start_time = time.time()
    pred, _, _ = classifier(point)
    end_time = time.time()
    inference_time = end_time - start_time
    total_time += inference_time

print("Average inference time: ", total_time/50*1000, end = ' ')
print("ms")

pred_choice = pred.data.max(2)[1]
print(pred_choice)

print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]
pred_colors = np.array([color_map[val] for val in pred_choice.numpy()[0]])#####

print(point_np.shape)
print(gt.shape)
print(pred_color.shape)

num_same_elements = np.sum(seg.numpy() - 1 == pred_choice.numpy())
print('mIoU: %f' %(num_same_elements/len(seg)))

# 保存到文本文件
# 将point_np和gt按列拼合
# seg = seg - 1
# seg = np.expand_dims(seg, axis=1)
pred_choice_np = pred_choice.cpu().numpy().T 
# combined_data = np.concatenate((point_np, seg), axis=1)
# np.savetxt('gt.txt', combined_data, delimiter=' ')
combined_data1 = np.concatenate((point_np, pred_choice_np), axis=1)
np.savetxt('pred.txt', combined_data1, delimiter=' ')

# showpoints(point_np, gt, pred_color)
showpoints(point_np, c_gt=seg_colors,c_pred=pred_colors)
