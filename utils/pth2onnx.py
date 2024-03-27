from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import sys
sys.path.append("../")
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.onnx as onnx

#导出onnx model forward输出不能为tuple
def build_pointnet(num_classes, feature_transform, phase, model_path = '/home/data6T/pxy/pointnet.pytorch/utils/seg/seg_model_Scene_15.pth'):

    class PointNetForTRT(nn.Module):
        def __init__(self):
            super().__init__()
            self.pointnet = PointNetDenseCls(num_classes, feature_transform)
            self.pointnet.load_state_dict(torch.load(model_path))
            print("load pre-trained model successful!!!")
        def forward(self, x):
            y = x.squeeze(3)
            return self.pointnet(y)
        
    Net = PointNetForTRT()
    if phase == "train":
        return Net.train()
    else:
        return Net.eval()

            

    Net = PointNetDenseCls(num_classes, feature_transform)

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()

def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = build_pointnet(num_classes = 8, feature_transform = False, phase = 'test')    # 导入模型
    # model.load_state_dict(torch.load(checkpoint))  # 初始化权重
    model.eval()
    # model.to(device)

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")

if __name__ == '__main__':

    checkpoint = '/home/data6T/pxy/pointnet.pytorch/utils/seg/seg_model_Scene_15.pth'
    onnx_path = './pointnet.onnx'
    input = torch.randn(1, 3, 2500, 1)
    # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(input, checkpoint, onnx_path)
