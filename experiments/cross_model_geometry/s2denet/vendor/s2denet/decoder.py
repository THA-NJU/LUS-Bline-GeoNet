"""Dual-path decoder (Sec. 3.3, Fig. 7): an independent boundary decoder and a
mask decoder that share a layer-wise fusion mechanism. Boundary features are
injected additively into the mask decoding pathway at every scale so that
boundary constraints propagate throughout decoding.
"""
import torch
import torch.nn as nn

from s2denet.upsample import DySample


class AdaptiveConv(nn.Module):
    """Depthwise-ish conv block gated by a learnable scalar weight (sigmoid),
    used to align/re-weight features from different branches before fusion."""

    def __init__(self, in_channels, out_channels, kz=3, pd=1):
        super(AdaptiveConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kz, padding=pd,
                      groups=out_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
        )
        self.weight = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        return self.conv(x) * self.weight.sigmoid()


class FusionBlock(nn.Module):
    """Fuses two feature maps of potentially different resolutions: aligns
    channels with AdaptiveConv, then upsamples the smaller map with DySample
    before summation."""

    def __init__(self, in_channel, out_channel, factor=None):
        super(FusionBlock, self).__init__()
        self.pre_conv1 = AdaptiveConv(in_channel[0], out_channel, kz=5, pd=2)
        self.pre_conv2 = AdaptiveConv(in_channel[1], out_channel, kz=5, pd=2)
        self.up = DySample(out_channel)

    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])

        _, _, H1, W1 = x1.size()
        _, _, H2, W2 = x2.size()
        if H1 != H2:
            x2 = self.up(x2)
        return x1 + x2


class BoundaryDecoder(nn.Module):
    """Boundary decoder branch: progressively fuses the high-frequency features
    from deep to shallow scales and predicts a boundary map at each scale."""

    def __init__(self, in_channels, num_class=1):
        super(BoundaryDecoder, self).__init__()
        self.num_class = num_class
        self.fuse32 = FusionBlock(in_channel=(in_channels * 4, in_channels * 8), out_channel=in_channels * 4,
                                   factor=2)
        self.fuse21 = FusionBlock(in_channel=(in_channels * 2, in_channels * 4), out_channel=in_channels * 2,
                                   factor=2)
        self.fuse10 = FusionBlock(in_channel=(in_channels * 1, in_channels * 2), out_channel=in_channels * 1,
                                   factor=2)
        self.head = nn.Conv2d(in_channels * 1, num_class, kernel_size=3, padding=1)

    def forward(self, x):
        s2 = self.fuse32(x[2], x[3])
        s1 = self.fuse21(x[1], s2)
        s0 = self.fuse10(x[0], s1)
        out = self.head(s0)
        return s2, s1, s0, out.sigmoid()


class MaskDecoder(nn.Module):
    """Mask decoder branch: fuses low-frequency features from deep to shallow
    scales, additively injecting boundary features from BoundaryDecoder at
    every scale (Sec. 3.3)."""

    def __init__(self, in_channels, num_class=1):
        super(MaskDecoder, self).__init__()
        self.num_class = num_class

        self.fuse32 = FusionBlock(in_channel=(in_channels * 4, in_channels * 8), out_channel=in_channels * 4,
                                   factor=2)
        self.fuse21 = FusionBlock(in_channel=(in_channels * 2, in_channels * 4), out_channel=in_channels * 2,
                                   factor=2)
        self.fuse10 = FusionBlock(in_channel=(in_channels * 1, in_channels * 2), out_channel=in_channels * 1,
                                   factor=2)
        self.head = nn.Conv2d(in_channels * 1, num_class, kernel_size=3, padding=1)

        self.boundary_decoder = BoundaryDecoder(in_channels=in_channels)

    def forward(self, x):
        # x = [lf1, lf2, lf3, lf4, hf1, hf2, hf3, hf4]
        edge = self.boundary_decoder([x[4], x[5], x[6], x[7]])

        s2 = self.fuse32(x[2], x[3]) + edge[0]
        s1 = self.fuse21(x[1], s2) + edge[1]
        s0 = self.fuse10(x[0], s1) + edge[2]
        out = self.head(s0) + edge[3]

        if self.num_class == 1:
            out = out.sigmoid()

        return out, edge[3]
