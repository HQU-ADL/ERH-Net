# @Version  :1.0
# @Author   :High_cold
# @File     :SBSA.py.py
# @Time     :2025/6/4 上午10:56
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange

##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, out_channels, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.factor = 2
        self.idx_dict = {}
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, out_channels, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[..., t_pad[0]:hw - t_pad[1]]

    def comp2real(self, x):
        b, _, h, w = x.shape
        return torch.cat([x.real, x.imag], 1)

    #        return torch.stack([x.real, x.imag], 2).view(b,-1,h,w)
    def real2comp(self, x):
        xr, xi = x.chunk(2, dim=1)
        return torch.complex(xr, xi)

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def get_idx_map(self, h, w):
        l1_u = torch.arange(h // 2).view(1, 1, -1, 1)
        l2_u = torch.arange(w).view(1, 1, 1, -1)
        half_map_u = l1_u @ l2_u
        l1_d = torch.arange(h - h // 2).flip(0).view(1, 1, -1, 1)
        l2_d = torch.arange(w).view(1, 1, 1, -1)
        half_map_d = l1_d @ l2_d
        return torch.cat([half_map_u, half_map_d], 2).view(1, 1, -1).argsort(-1)

    def get_idx(self, x):
        h, w = x.shape[-2:]
        if (h, w) in self.idx_dict:
            return self.idx_dict[(h, w)]
        idx_map = self.get_idx_map(h, w).to(x.device).detach()
        self.idx_dict[(h, w)] = idx_map
        return idx_map

    def attn(self, qkv):
        h = qkv.shape[2]
        q, k, v = qkv.chunk(3, dim=1)

        q, pad_w, idx = self.fft(q)
        q, pad = self.pad(q, self.factor)
        k, pad_w, _ = self.fft(k)
        k, pad = self.pad(k, self.factor)
        v, pad_w, _ = self.fft(v)
        v, pad = self.pad(v, self.factor)

        q = rearrange(q, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        k = rearrange(k, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)
        v = rearrange(v, 'b (head c) (factor hw) -> b head (c factor) hw', head=self.num_heads, factor=self.factor)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head (c factor) hw -> b (head c) (factor hw)', head=self.num_heads, factor=self.factor)
        out = self.unpad(out, pad)
        out = self.ifft(out, pad_w, idx, h)
        return out

    def fft(self, x):
        x, pad = self.pad(x, 2)
        x = torch.fft.rfft2(x.float(), norm="ortho")
        x = self.comp2real(x)
        idx = self.get_idx(x)
        # 添加设备转换 ======= 修复代码 =======
        idx = idx.to(x.device)  # 确保idx与x在同一设备
        # ===================================
        b, c = x.shape[:2]
        x = x.contiguous().view(b, c, -1)
        x = torch.gather(x, 2, index=idx.repeat(b, c, 1))  # b, 6c, h*(w//2+1)
        return x, pad, idx

    def ifft(self, x, pad, idx, h):
        b, c = x.shape[:2]
        x = torch.scatter(x, 2, idx.repeat(b, c, 1), x)
        x = x.view(b, c, h, -1)
        x = self.real2comp(x)
        x = torch.fft.irfft2(x, norm='ortho')  # .abs()
        x = self.unpad(x, pad)
        return x

    def forward(self, x):
        b, c, h, w = x.shape

        attn_map = x

        qkv = self.qkv_dwconv(self.qkv(x))

        #        qkv, pad_w, idx = self.fft(qkv)
        #        qkv, pad = self.pad(qkv, self.factor)

        attn_map = qkv
        out = self.attn(qkv)
        attn_map = out

        #        out = self.unpad(out, pad)
        #        out = self.ifft(out, pad_w, idx, h)

        out = self.project_out(out)
        attn_map = out
        return out

    '''
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    '''

##########################################################################

if __name__ == '__main__':
    import torch

    # 参数设置
    batch_size = 1  # 批量大小
    channels = 64  # 通道数 (dim)
    height = 512  # 图像高度
    width = 512  # 图像宽度
    num_heads = 4  # 注意力头数

    # 创建随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # 初始化Attention模块
    model = Attention(dim=channels, out_channels=channels, num_heads=num_heads, bias=True)

    # 打印模型结构
    print(model)

    # 执行前向传播
    output = model(x)

    # 打印输入输出形状
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("微信公众号: AI缝合术!")