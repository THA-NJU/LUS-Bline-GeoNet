"""Differential Self-Attention (DiffSA), Sec. 3.2 / Fig. 6 / Eq. 5:

    DiffSA(Q, K, V) = softmax((Q1 - lambda*Q2)(K1 - beta*K2)^T / sqrt(d_k)) V

Applied to the low-frequency pathway in shallow encoder stages (L1, L2) to
suppress common-mode speckle noise shared by (Q1, Q2) and (K1, K2) before the
attention scores are computed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from s2denet.transformer_blocks import LayerNorm, FeedForward


class DiffSA(nn.Module):
    """Differential Self-Attention module (Eq. 5)."""

    def __init__(self, dim, size, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # lambda / beta gates (Eq. 5): learnable scalars mixing the secondary
        # (Q2, K2) representations into the primary (Q1, K1) ones.
        self.lambda_gate = nn.Parameter(torch.tensor(1.0))
        self.beta_gate = nn.Parameter(torch.tensor(1.0))

        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=qkv_bias)  # Q1, K1, Q2, K2, V
        self.dwconv = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, padding=1, groups=dim * 5, bias=qkv_bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q1 = F.normalize(q1, dim=-1)
        k1 = F.normalize(k1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        k2 = F.normalize(k2, dim=-1)

        q = q1 - self.lambda_gate * q2
        k = k1 - self.beta_gate * k2

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=H, w=W)

        out = self.project_out(out)
        return self.proj_drop(out)


class DiffSABlock(nn.Module):
    """DiffSA + GDFN feed-forward, with residual connections and LayerNorm
    (Sec. 3.2: "residual connections and layer normalization ... combined with
    feedforward networks to form complete processing units")."""

    def __init__(self, dim, size, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(DiffSABlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = DiffSA(dim=dim, num_heads=num_heads, size=size, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
