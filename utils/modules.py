import torch
import torch.nn as nn
from copy import deepcopy
import math

from einops import rearrange, repeat



def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        # 使用residual连结
        return self.norm(self.fn(x, **kwargs) + x)
        # return self.fn(self.norm(x), **kwargs)


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


# Attention
class Attention(nn.Module):
    def __init__(self, 
                 dim,            # 输入X的特征长度
                 heads=8,        # multi-head的个数
                 dim_head = 64,  # 每个head的dim
                 dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        # 这个判断条件就是multi-head还是single-head
        # 如果是multi-head，那么project_out = True，即把多个head输出的q拼在一起再用FC处理一次
        # 如果是single-head，则 = False，不需要拼接q，也就不需要FC处理了
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5     # sqrt(d_k)

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # W_q, W_k, W_v

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 输入：x -> [B, N, C1]
        # self.to_qkv一次得到Q,K,V三个变量，然后用chunk拆分成三分，每个都是[B, N, h*d]
        # 这里，M = N
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # 使用爱因斯坦求和方式来改变shape：[B, N, h*d] -> [B, h, N, d]
        # 其中h=heads，即multi-head的个数
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # Q*K^T / sqrt(d_k) : [B, h, N, d] X [B, h, d, N] = [B, h, N, N]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # 使用softmax求概率
        attn = self.attend(dots)
        # 自注意力机制：softmax(Q*K^T / sqrt(d_k)) * V ：[B, h, N, N] X [B, h, N, d] = [B, h, N, d]
        out = torch.matmul(attn, v)
        # 改变shape：[B, h, N, d] -> [B, N, h*d]=[B, N, C2], C2 = h*d
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


# Transformer
class Transformer(nn.Module):
    def __init__(self, 
                 dim,            # 输入X的特征长度
                 depth,          # Encoder的层数
                 heads,          # multi-head的个数
                 dim_head,       # 每个head的dim
                 mlp_dim,        # FFN中的dim
                 dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            # 注意，PreNorm里内部已经做了residual。
            x = attn(x) 
            x = ff(x)
        return x


class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        # create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
