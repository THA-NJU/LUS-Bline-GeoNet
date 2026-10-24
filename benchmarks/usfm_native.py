"""USFM HVIT encoder and UPerHead used by the benchmark."""
import torch
from torch import nn
from torch.nn import functional as F
import usdsgen
import usdsgen.models
from mmseg.models import build_segmentor


class USFMModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        config = dict(type='EncoderDecoder', pretrained=None,
            backbone=dict(type='HVITBackbone4Seg', img_size=224, patch_size=16,
                          embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                          qkv_bias=True, use_abs_pos_emb=False, use_rel_pos_bias=True,
                          init_values=.1, drop_path_rate=.1, out_indices=[3,5,7,11]),
            decode_head=dict(type='UPerHead', in_channels=[768]*4, in_index=[0,1,2,3],
                             pool_scales=[1,2,3,6], channels=768, dropout_ratio=.1,
                             num_classes=num_classes, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                             align_corners=False, loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.)),
            train_cfg=dict(), test_cfg=dict(mode='whole'))
        self.mmseg_model = build_segmentor(config)

    def forward(self, image):
        features = self.mmseg_model.extract_feat(image)
        logits = self.mmseg_model.decode_head.forward(features)
        return F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)
