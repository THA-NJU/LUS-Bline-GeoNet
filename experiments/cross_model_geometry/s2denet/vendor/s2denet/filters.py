"""Fixed 3x3 low-pass / high-pass filters used for frequency decoupling
(Sec. 3, Fig. 3: "fixed 3x3 Laplacian high-pass filter and 3x3 average low-pass
filter for computational efficiency").
"""
import torch
import torch.nn.functional as F


def low_pass_conv(x):
    """3x3 averaging (box blur) kernel, shared across channels."""
    B, C, H, W = x.shape
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0
    kernel = kernel.to(x.device)
    kernel = kernel.repeat(C, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=C)


def high_pass_conv(x):
    """3x3 Laplacian kernel, shared across channels."""
    B, C, H, W = x.shape
    kernel = torch.tensor([[0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    kernel = kernel.to(x.device)
    kernel = kernel.repeat(C, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=C)
