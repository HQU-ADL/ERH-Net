# @Version  :1.0
# @Author   :High_cold
# @File     :MEEM.py
# @Time     :2025/7/2 上午11:08
import torch
from torch import nn
# from .common import LayerNorm2d
from torch.nn import functional as F


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class MEEM(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, width=4, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, out_dim, 1, bias=False),
            norm(out_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out

# 测试入口
if __name__ == "__main__":
    # 第二个即插即用模块是多尺度边缘增强模块: MEEM
    MEEM = MEEM(in_dim=64, out_dim=64, hidden_dim=32)
    input = torch.randn(1, 64, 128, 128)
    # 执行前向传播
    output = MEEM(input)
    print('MEEM Input size:', input.size())
    print('MEEM Output size:', output.size())