import torch
import torch.nn as nn
from copy import deepcopy
import math

from torch.nn.modules.pooling import AvgPool2d


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


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act='relu'):
        super(Conv, self).__init__()
        if act is not None:
            if act == 'relu':
                self.convs = nn.Sequential(
                    nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.ReLU(inplace=True) if act else nn.Identity()
                )
            elif act == 'leaky':
                self.convs = nn.Sequential(
                    nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
                )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.convs(x)


class BasicBlockv1(nn.Module):
    def __init__(self, c):
        super().__init__()
        c_ = c // 2
        self.cv1 = Conv(c_, c_, k=1)
        self.cv2 = Conv(c_, c_, k=3, p=1, g=c_, act=None)
        self.cv3 = Conv(c_, c_, k=1)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x2 = self.cv3(self.cv2(self.cv1(x2)))
        y = torch.cat([x1, x2], dim=1)

        return channel_shuffle(y, 2)


class ShuffleBlockv1(nn.Module):
    def __init__(self, c, n=1):
        super().__init__()
        branch = [BasicBlockv1(c) for _ in range(n)]
        self.branch = nn.Sequential(*branch)

    def forward(self, x):
        return self.branch(x)


class ShuffleBlockv2(nn.Module):
    # ShuffleBlock with Downsample
    def __init__(self, c1, c2, s=2):
        super().__init__()
        c_ = c2 // 2
        # branch-1
        self.cv1_1 = Conv(c1, c1, k=3, p=1, s=s, g=c1, act=None)
        self.cv1_2 = Conv(c1, c_, k=1)
        # branch-2
        self.cv2_1 = Conv(c1, c_, k=1)
        self.cv2_2 = Conv(c_, c_, k=3, p=1, s=s, g=c_, act=None)
        self.cv2_3 = Conv(c_, c_, k=1)

    def forward(self, x):
        # branch-1
        x1 = self.cv1_2(self.cv1_1(x))
        x2 = self.cv2_3(self.cv2_2(self.cv2_1(x)))
        y = torch.cat([x1, x2], dim=1)
        
        return channel_shuffle(y, 2)


class BottleneckDW(nn.Module):
    # Depth-wise bottleneck
    def __init__(self, c1, c2, s=1, shortcut=False, act='relu', e=1.0):
        super(BottleneckDW, self).__init__()
        c_ = int(c1 * e)
        self.h1 = nn.Sequential(
            Conv(c1, c_, k=1, act=act),
            Conv(c_, c_, k=3, p=1, s=s, g=c_, act=None),
            Conv(c_, c2, k=1, act=act)
        )
        self.shortcut = shortcut and s == 1
        if self.shortcut:
            self.h2 = nn.Identity()
        else:
            self.h2 = nn.Sequential(
                Conv(c1, c2, k=1, act='relu'),
                nn.AvgPool2d(2, 2)
            )

    def forward(self, x):
        h1 = self.h1(x)
        h2 = self.h2(x)
        return h1 + h2


class CSPBlockDW(nn.Module):
    # Depth-wise CSPBlock
    def __init__(self, c, n=1, shortcut=False, act='relu'):
        super(CSPBlockDW, self).__init__()
        c_ = c // 2
        branch = [BottleneckDW(c_, c_, shortcut=shortcut, act=act) for _ in range(n)]
        self.branch = nn.Sequential(*branch)
        self.conv = Conv(c_ * 2, c, k=1)

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x2 = self.branch(x2)
        y = torch.cat([x1, x2], dim=1)

        return self.conv(y)


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
