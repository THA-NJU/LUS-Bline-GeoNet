"""S2DENet: Shallow Suppression and Deep Enhancement Network for General
Ultrasound Image Segmentation.

Combines the S2DEEncoder (four-stage, depth-differentiated hierarchical
processing) with the dual-path decoder (MaskDecoder + BoundaryDecoder) into
the full segmentation model described in the paper.
"""
import torch.nn as nn

from s2denet.encoder import S2DEEncoder
from s2denet.decoder import MaskDecoder


class S2DENet(nn.Module):
    """Full S2DENet model.

    Args:
        in_channels: base channel width `c` (Fig. 3). Use 8 for S2DENet,
            4 for S2DENet-Tiny.
        num_class: number of output segmentation classes (1 for binary masks).
        return_edge: if True, forward() also returns the predicted boundary
            map (used for auxiliary edge supervision during training).
    """

    def __init__(self, in_channels=8, num_class=1, return_edge=True):
        super(S2DENet, self).__init__()
        self.encoder = S2DEEncoder(in_channels)
        self.decoder = MaskDecoder(in_channels, num_class)
        self.return_edge = return_edge

    def forward(self, x):
        lf1, lf2, lf3, lf4, hf1, hf2, hf3, hf4 = self.encoder(x)
        out, edge = self.decoder([lf1, lf2, lf3, lf4, hf1, hf2, hf3, hf4])
        if self.return_edge:
            return out, edge
        return out


def S2DENet_Tiny(num_class=1, return_edge=True):
    """S2DENet-Tiny variant (c=4), as reported in Table 4 (0.05M params)."""
    return S2DENet(in_channels=4, num_class=num_class, return_edge=return_edge)
