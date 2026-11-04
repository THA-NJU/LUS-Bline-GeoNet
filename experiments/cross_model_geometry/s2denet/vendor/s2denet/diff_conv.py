"""Pixel-difference convolution and Multi-order Differential Convolution (MDiffConv).

Re-parameterizes a standard conv kernel into differential forms (Su et al., 2021,
"Pixel difference networks for efficient edge detection"). MDiffConv chains one
second-order term ("np2d") with three first-order terms ("cd", "ad", "rd") to
implement the module described in Sec. 3.1 / Fig. 4 of the paper.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_pdc_func(pdc_type):
    assert pdc_type in ['cv', 'cd', 'ad', 'rd', 'np2d'], 'unknown PDC type: %s' % str(pdc_type)

    if pdc_type == 'cv':
        return F.conv2d

    if pdc_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc

        return func

    if pdc_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    if pdc_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            device = weights.device
            buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=device)

            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0

            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func

    # pdc_type == 'np2d': second-order differential form (Eq. 1-4 in the paper).
    def func(x, weights, bias=None, stride=1, padding=1, dilation=1, groups=1):
        assert dilation in [1, 2], 'dilation for np2d_conv should be in 1 or 2'
        assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for np2d_conv should be 3x3'

        weights_t = torch.flip(weights, [2, 3])
        combined_weights = weights + weights_t

        weights_c = 2 * weights.sum(dim=[2, 3], keepdim=True)
        yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
        y = F.conv2d(x, combined_weights, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y - yc

    return func


class PDConv2d(nn.Module):
    """Standard nn.Conv2d-compatible wrapper around a pixel-difference conv function."""

    def __init__(self, pdc_func, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        super(PDConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc_func = create_pdc_func(pdc_func)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.pdc_func(input, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups)


class DiffConvUnit(nn.Module):
    """PDConv2d + SiLU + pointwise conv + residual shortcut."""

    def __init__(self, pdc_func, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DiffConvUnit, self).__init__()
        self.depthwise = PDConv2d(pdc_func=pdc_func, kernel_size=kernel_size,
                                   in_channels=in_channels, out_channels=in_channels,
                                   groups=min(in_channels, out_channels), padding=(kernel_size - 1) // 2)
        self.silu = nn.SiLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                       groups=1, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                    groups=1, bias=False)

    def forward(self, x):
        short = x.clone()
        x = self.depthwise(x)
        x = self.silu(x)
        x = self.pointwise(x)
        if self.in_channels != self.out_channels:
            short = self.shortcut(short)
        return x + short


class MDiffConv(nn.Module):
    """Multi-order Differential Convolution (Sec. 3.1, Fig. 4).

    A second-order differential conv ("np2d") followed by a chain of first-order
    differential convs ("cd" -> "ad" -> "rd"), then a 1x1 fusion conv.
    """

    def __init__(self, in_channels, stage='l'):
        super(MDiffConv, self).__init__()
        assert stage in ['l', 'd'], 'stage must be "l" (shallow) or "d" (deep)'
        self.stage = stage

        self.np2d = DiffConvUnit(pdc_func='np2d', in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                  padding=1)
        self.cd = DiffConvUnit(pdc_func='cd', in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                padding=1)
        self.ad = DiffConvUnit(pdc_func='ad', in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                padding=1)
        self.rd = DiffConvUnit(pdc_func='rd', in_channels=in_channels, out_channels=in_channels, kernel_size=3,
                                dilation=1, padding=1)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        np2d = self.np2d(x)
        cd = self.cd(np2d)
        ad = self.ad(cd)
        rd = self.rd(ad)
        return self.conv1(rd)
