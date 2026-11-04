"""S2DENet encoder: four-stage hierarchical processing with shallow suppression
(L1, L2) and deep enhancement (L3, L4), as described in Sec. 3 / Fig. 3.

Each stage splits its input into a low-frequency branch (structural, handled by
DiffSA in shallow stages / standard self-attention in deep stages) and a
high-frequency branch (boundary + speckle, handled by MDiffConv). Shallow
stages subtract the modulated high-frequency response from the input
("suppression"); deep stages add it back ("enhancement").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from s2denet.diff_conv import MDiffConv, DiffConvUnit
from s2denet.diffsa import DiffSABlock
from s2denet.filters import low_pass_conv, high_pass_conv
from s2denet.transformer_blocks import TransformerBlock


class MultiScaleConvPool(nn.Module):
    """Multi-scale ConvPool (Sec. 3.1, Fig. 5).

    Integrates boundary-preserving convolutions at multiple kernel sizes with
    large-receptive-field average pooling to fuse boundary cues with smooth
    structural context, ahead of interacting with the low-frequency pathway.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = DiffConvUnit('cv', in_channels, in_channels * 2, kernel_size=1)
        self.conv3 = DiffConvUnit('cv', in_channels * 2, in_channels * 2, kernel_size=3, padding=1)
        self.conv5 = DiffConvUnit('cv', in_channels * 2, in_channels * 2, kernel_size=5, padding=3)

        self.avgpool_7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=(7 - 1) // 2)
        self.avgpool_11 = nn.AvgPool2d(kernel_size=11, stride=1, padding=(11 - 1) // 2)
        self.conv_fusion = DiffConvUnit('cv', in_channels * 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out_11 = self.conv1(x)
        out_33 = self.conv3(out_11)
        out_77 = self.conv5(out_11)
        avgpool7 = self.avgpool_7(out_11)
        avgpool11 = self.avgpool_11(out_11)
        out_conv = out_33 + out_77 + avgpool7 + avgpool11
        return self.conv_fusion(out_conv)


class S2DEStage(nn.Module):
    """One encoder stage: frequency decoupling + depth-differentiated processing.

    stage='l' (shallow, L1/L2): high-frequency suppression (x - hf) followed by
    MultiScaleConvPool, low-frequency handled by DiffSA.
    stage='d' (deep, L3/L4): high-frequency enhancement (x + hf), low-frequency
    handled by standard self-attention (TransformerBlock).
    """

    def __init__(self, in_channels, size, stage):
        super(S2DEStage, self).__init__()
        self.stage = stage
        self.in_channels = in_channels

        if stage == 'l':
            self.low_freq = DiffSABlock(dim=in_channels, size=size, num_heads=4, ffn_expansion_factor=2.66,
                                         bias=False, LayerNorm_type='WithBias')
            self.multiscale_pool = MultiScaleConvPool(in_channels)
        else:
            self.low_freq = TransformerBlock(in_channels, 4, 2.66, False, 'WithBias')

        self.high_freq = MDiffConv(in_channels, stage)

    def forward(self, x):
        yl = low_pass_conv(x)
        yh = high_pass_conv(x)

        hf = self.high_freq(yh)

        if self.stage == 'l':
            f = x - hf
            out_conv = self.multiscale_pool(f)
        else:
            f = x + hf
            out_conv = f

        lf = self.low_freq(yl)
        out = out_conv * lf
        return out, hf


class S2DEEncoder(nn.Module):
    """Four-stage encoder producing low-/high-frequency features at each scale."""

    def __init__(self, in_channels=8):
        super(S2DEEncoder, self).__init__()
        self.init = nn.Conv2d(in_channels=3, out_channels=in_channels, kernel_size=7, padding=3)

        self.stage1 = S2DEStage(in_channels, stage='l', size=128)
        self.stage2 = S2DEStage(in_channels * 2, stage='l', size=64)
        self.stage3 = S2DEStage(in_channels * 4, stage='d', size=32)
        self.stage4 = S2DEStage(in_channels * 8, stage='d', size=16)

        self.down1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1, groups=in_channels)
        self.down2 = nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2, padding=1,
                                groups=in_channels)
        self.down3 = nn.Conv2d(in_channels * 4, in_channels * 8, kernel_size=3, stride=2, padding=1,
                                groups=in_channels)

    def forward(self, rgb_x):
        if rgb_x.size()[1] == 1:
            rgb_x = rgb_x.repeat(1, 3, 1, 1)

        x = self.init(rgb_x)

        lf1, hf1 = self.stage1(x)
        ds1 = self.down1(lf1 + hf1)

        lf2, hf2 = self.stage2(ds1)
        ds2 = self.down2(lf2 + hf2)

        lf3, hf3 = self.stage3(ds2)
        ds3 = self.down3(lf3 + hf3)

        lf4, hf4 = self.stage4(ds3)

        return lf1, lf2, lf3, lf4, hf1, hf2, hf3, hf4
