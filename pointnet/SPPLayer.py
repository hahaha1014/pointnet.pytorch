import torch
import torch.nn as nn
import torch.nn.functional as F

class SPPLayer(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    B, C, N = x.shape  

    # 最大池化,步长2
    x1 = F.max_pool1d(x, kernel_size=2048, stride=1)
    # print(x1.shape)
    # print(x1)
    
    x2 = F.max_pool1d(x, kernel_size=1024, stride=1024) 
    # print(x2.shape)
    # print(x2)

    x3 = F.max_pool1d(x, kernel_size=512, stride=512)
    # print(x3.shape)

    x4 = F.max_pool1d(x, kernel_size=256, stride=256)
    # print(x4.shape)

    # 拼接
    x = torch.cat((x1, x2, x3, x4), dim=2)
    # print(x.shape) 

    # # # 全连接
    # linear = nn.Linear(15,N)
    # x = linear(x)

    return x
    
    
if __name__ == "__main__":

    x = torch.rand(1, 1024, 2048)
    model = SPPLayer()
    print(x)
    print(model(x).shape)
    # print(model(x))