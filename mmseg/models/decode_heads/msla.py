import torch
import torch.nn as nn
from .linear_attention import LinearAttention


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.relu(x)
        return x



class MSLA(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dw_conv_3x3 = DepthwiseConv(dim // 4, kernel_size=3)
        self.dw_conv_5x5 = DepthwiseConv(dim // 4, kernel_size=5)
        self.dw_conv_7x7 = DepthwiseConv(dim // 4, kernel_size=7)
        self.dw_conv_9x9 = DepthwiseConv(dim // 4, kernel_size=9)

        self.linear_attention = LinearAttention(dim = dim // 4, num_heads = num_heads)

        self.final_conv = nn.Conv2d(dim, dim, 1)

        self.scale_weights = nn.Parameter(torch.ones(4), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        split_size = c // 4
        x_3x3 = x[:, :split_size, :, :]
        x_5x5 = x[:, split_size:2 * split_size, :, :]
        x_7x7 = x[:, 2 * split_size:3 * split_size:, :, :]
        x_9x9 = x[:, 3 * split_size:, :, :]

        x_3x3 = self.dw_conv_3x3(x_3x3)
        x_5x5 = self.dw_conv_5x5(x_5x5)
        x_7x7 = self.dw_conv_7x7(x_7x7)
        x_9x9 = self.dw_conv_9x9(x_9x9)


        att_3x3 = self.linear_attention(x_3x3)
        att_5x5 = self.linear_attention(x_5x5)
        att_7x7 = self.linear_attention(x_7x7)
        att_9x9 = self.linear_attention(x_9x9)


        processed_input = torch.cat([
            att_3x3 * self.scale_weights[0],
            att_5x5 * self.scale_weights[1],
            att_7x7 * self.scale_weights[2],
            att_9x9 * self.scale_weights[3]
        ], dim=1)

        final_output = self.final_conv(processed_input)

        return final_output

# 测试入口
if __name__ == "__main__":
    batch = 1
    channels = 32
    height, width = 128, 128
    x = torch.randn(batch, channels, height, width)

    MSLA = MSLA(dim=channels, num_heads=4)
    out = MSLA(x)

    print("EGA++ 输入维度：", x.shape)
    print("\n哔哩哔哩：CV缝合救星!\n")
    print("EGA++ 输出维度：", out.shape)