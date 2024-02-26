import torch
import torch.nn as nn
from thop import profile

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b,c,h,w = x.size()
        # print("**********")
        # print(x.size())
        y = self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x)

if __name__ == "__main__":

    x = torch.ones(16, 64, 4096, 1)
    model = SELayer(64)

    print(model(x).shape)
    print("Total parameters: ", count_parameters(model))
    # 使用thop的profile函数计算FLOPs
    flops, params = profile(model, inputs=(x,))
    # 打印FLOPs
    print("Total FLOPs: ", flops)