import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from .deform_conv_func import deform_conv


class DeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
    ):
        assert not bias
        super(DeformConv, self).__init__()
        self.with_bias = bias

        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(in_channels, groups)
        assert out_channels % groups == 0, "out_channels {} cannot be divisible by groups {}".format(
            out_channels, groups
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return deform_conv(
            input, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups
        )

    def __repr__(self):
        return "".join(
            [
                "{}(".format(self.__class__.__name__),
                "in_channels={}, ".format(self.in_channels),
                "out_channels={}, ".format(self.out_channels),
                "kernel_size={}, ".format(self.kernel_size),
                "stride={}, ".format(self.stride),
                "dilation={}, ".format(self.dilation),
                "padding={}, ".format(self.padding),
                "groups={}, ".format(self.groups),
                "deformable_groups={}, ".format(self.deformable_groups),
                "bias={})".format(self.with_bias),
            ]
        )
