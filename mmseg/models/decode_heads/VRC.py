import torch
import torch.nn as nn
from typing import Tuple, Union


# https://arxiv.org/pdf/2309.11523
# https://github.com/qhfan/RMT/blob/main/classfication_release/RMT.py


def rotate_every_two(x):
    # 对输入张量x在最后一个维度上进行旋转（每两个元素交换位置）
    x1 = x[:, :, :, :, ::2]  # 取奇数列
    x2 = x[:, :, :, :, 1::2]  # 取偶数列
    x = torch.stack([-x2, x1], dim=-1)  # 将奇数列与偶数列交换，并变为负数的偶数列
    return x.flatten(-2)  # 将倒数第二维度展平


def theta_shift(x, sin, cos):
    # 对输入x进行旋转变换，其中sin和cos为旋转角度的正弦和余弦值
    return (x * cos) + (rotate_every_two(x) * sin)


class DWConv2d(nn.Module):
    # Depthwise Convolution Layer（深度卷积层）
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        # depthwise卷积，输入和输出的通道数相同，groups=dim表示每个输入通道独立卷积
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: 输入张量，形状为 (b h w c)，
        b: batch size，h: 高度，w: 宽度，c: 通道数
        '''
        x = x.permute(0, 3, 1, 2)  # 将x从 (b h w c) 转换为 (b c h w)，以适应卷积的格式
        x = self.conv(x)  # 对x进行深度卷积
        x = x.permute(0, 2, 3, 1)  # 将输出从 (b c h w) 转回 (b h w c)
        return x


class RetNetRelPos2d(nn.Module):
    # 用于生成相对位置编码的模块
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))  # 计算角度
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()  # 将角度展开
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))  # 计算衰减
        self.register_buffer('angle', angle)  # 注册角度作为模型参数
        self.register_buffer('decay', decay)  # 注册衰减作为模型参数

    def generate_2d_decay(self, H: int, W: int):
        '''
        生成2D衰减掩码，结果形状为 (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)  # 高度方向的索引
        index_w = torch.arange(W).to(self.decay)  # 宽度方向的索引
        grid = torch.meshgrid([index_h, index_w])  # 生成网格坐标
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # 将坐标堆叠并展平
        mask = grid[:, None, :] - grid[None, :, :]  # 计算网格点之间的差异
        mask = (mask.abs()).sum(dim=-1)  # 计算距离
        mask = mask * self.decay[:, None, None]  # 应用衰减
        return mask

    def generate_1d_decay(self, l: int):
        '''
        生成1D衰减掩码，结果形状为 l*l
        '''
        index = torch.arange(l).to(self.decay)  # 生成索引
        mask = index[:, None] - index[None, :]  # 计算差异
        mask = mask.abs()  # 获取绝对值
        mask = mask * self.decay[:, None, None]  # 应用衰减
        return mask

    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen[0] * slen[1] - 1))  # 计算正弦值
            cos = torch.cos(self.angle * (slen[0] * slen[1] - 1))  # 计算余弦值
            retention_rel_pos = ((sin, cos), self.decay.exp())  # 返回旋转的相对位置编码

        elif chunkwise_recurrent:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # 计算正弦值
            sin = sin.reshape(slen[0], slen[1], -1)  # 将其调整为 (h w d1) 形状
            cos = torch.cos(index[:, None] * self.angle[None, :])  # 计算余弦值
            cos = cos.reshape(slen[0], slen[1], -1)  # 将其调整为 (h w d1) 形状

            mask_h = self.generate_1d_decay(slen[0])  # 生成高度方向的衰减掩码
            mask_w = self.generate_1d_decay(slen[1])  # 生成宽度方向的衰减掩码

            retention_rel_pos = ((sin, cos), (mask_h, mask_w))  # 返回包含 sin, cos 和掩码的元组

        else:
            index = torch.arange(slen[0] * slen[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])  # 计算正弦值
            sin = sin.reshape(slen[0], slen[1], -1)  # 将其调整为 (h w d1) 形状
            cos = torch.cos(index[:, None] * self.angle[None, :])  # 计算余弦值
            cos = cos.reshape(slen[0], slen[1], -1)  # 将其调整为 (h w d1) 形状
            mask = self.generate_2d_decay(slen[0], slen[1])  # 生成2D衰减掩码
            retention_rel_pos = ((sin, cos), mask)  # 返回旋转后的相对位置编码及掩码

        return retention_rel_pos
class VisionRetentionChunk(nn.Module):
    # Vision Retention Chunk，利用卷积与注意力机制提取图像特征
    def __init__(self, embed_dim, embed_dim2, num_heads=4, value_factor=1):
        super().__init__()
        self.factor = value_factor  # 用于控制v_proj的输出维度
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = self.embed_dim * self.factor // num_heads  # 每个头的维度
        self.key_dim = self.embed_dim // num_heads  # 键的维度
        self.scaling = self.key_dim ** -0.5  # 缩放因子，用于归一化注意力分数
        # 查询、键、值投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        # 使用深度卷积处理值（v）
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)

        # 输出投影层
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim2, bias=True)
        self.reset_parameters()

        self.Relpos = RetNetRelPos2d(embed_dim, num_heads, 2, 4)

    def reset_parameters(self):
        # 对投影层参数进行初始化
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor):
        '''
        x: 输入张量，形状为 (b h w c)
        rel_pos: 相对位置编码，包含sin和cos的值，以及掩码（mask_h, mask_w）
        '''
        x = x.permute(0, 2, 3, 1)

        bsz, h, w, _ = x.size()  # 获取批大小、图像高度、宽度

        rel_pos = self.Relpos((h, w), chunkwise_recurrent=True)

        (sin, cos), (mask_h, mask_w) = rel_pos  # 从输入的相对位置中获取sin、cos和掩码

        # 通过查询、键、值投影层获取q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)  # 通过深度卷积处理值（v）

        k *= self.scaling  # 对键进行缩放
        # 重塑q和k，准备进行注意力计算
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # (b n h w d1)

        # 对q和k进行旋转变换
        qr = theta_shift(q, sin, cos)  # (b n h w d1)
        kr = theta_shift(k, sin, cos)  # (b n h w d1)

        # 计算水平维度的注意力
        qr_w = qr.transpose(1, 2)  # (b h n w d1)
        kr_w = kr.transpose(1, 2)  # (b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # (b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # (b h n w w)
        qk_mat_w = qk_mat_w + mask_w  # 加上掩码
        qk_mat_w = torch.softmax(qk_mat_w, -1)  # 归一化注意力分数
        v = torch.matmul(qk_mat_w, v)  # (b h n w d2)

        # 计算垂直维度的注意力
        qr_h = qr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4)  # (b w n h d1)
        v = v.permute(0, 3, 2, 1, 4)  # (b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # (b w n h h)
        qk_mat_h = qk_mat_h + mask_h  # 加上掩码
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # 归一化注意力分数
        output = torch.matmul(qk_mat_h, v)  # (b w n h d2)

        # 将输出展平并通过最终的投影层
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # (b h w n*d2)
        output = output + lepe  # 加上深度卷积的输出
        output = self.out_proj(output)  # 最后的投影层

        output = output.permute(0, 3, 1, 2)

        return output

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck_VRC(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = VisionRetentionChunk(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class C2f_VRC(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_VRC(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



def main():

    # 创建一个假输入张量 (batch_size, height, width, channels)
    x = torch.randn(8, 128, 64, 64)  # 随机初始化输入

    # 实例化 VisionRetentionChunk 模型
    model = VisionRetentionChunk(embed_dim=128,embed_dim2=128, num_heads=4, value_factor=1)

    # 前向传播
    output = model(x)

    # 打印输出张量的形状，确认模型的输出维度
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()





# # Ultralytics YOLO 🚀, AGPL-3.0 license
# # YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect
#
# # Parameters
# nc: 80 # number of classes
# scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
#   # [depth, width, max_channels]
#   n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
#   s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
#   m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
#   l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
#   x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
#
# # YOLOv8.0n backbone
# backbone:
#   # [from, repeats, module, args]
#   - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
#   - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
#   - [-1, 3, C2f_VRC, [128, True]]
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
#   - [-1, 6, C2f_VRC, [256, True]]
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
#   - [-1, 6, C2f_VRC, [512, True]]
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#   - [-1, 3, C2f_VRC, [1024, True]]
#   - [-1, 1, SPPF, [1024, 5]] # 9
#
# # YOLOv8.0n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4
#   - [-1, 3, C2f, [512]] # 12
#
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 4], 1, Concat, [1]] # cat backbone P3
#   - [-1, 3, C2f, [256]] # 15 (P3/8-small)
#
#   - [-1, 1, Conv, [256, 3, 2]]
#   - [[-1, 12], 1, Concat, [1]] # cat head P4
#   - [-1, 3, C2f, [512]] # 18 (P4/16-medium)
#
#   - [-1, 1, Conv, [512, 3, 2]]
#   - [[-1, 9], 1, Concat, [1]] # cat head P5
#   - [-1, 3, C2f, [1024]] # 21 (P5/32-large)
#
#   - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)


