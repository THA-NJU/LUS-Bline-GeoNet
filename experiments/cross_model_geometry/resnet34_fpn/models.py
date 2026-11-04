"""V6 geometry adapters for three randomly initialized native backbones.

The geometric query head, renderer and residual refinement are inherited from
the unchanged geometry.GeometryBase. Native two-class logits are preserved;
only their exactly equivalent foreground-minus-background logit is exposed.
Constructors initialize model parameters randomly.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import torch
from torch import nn
from torch.nn import functional as F

from geometry import RibbonNet as GeometryBase, ribbon


MODEL_NAMES = ("vm_unet_v2", "segformer_b2", "fpn_resnet34")
PREPROCESSING = {
    "vm_unet_v2": {"mean": None, "std": None},
    "segformer_b2": {
        "mean": 0.10222768713985789,
        "std": 0.1449912555111227,
    },
    "fpn_resnet34": {"mean": 0.15732, "std": 0.16688},
}


def _build_native(name: str) -> tuple[nn.Module, int]:
    if name == "vm_unet_v2":
        from vendor.vm_unet import VMUNet

        native = VMUNet(
            in_channels=3,
            num_classes=2,
            img_size=512,
            embed_dim=64,
            depths=[2, 2, 6, 2],
            depths_decoder=[2, 6, 2, 2],
            mid_channel=32,
            drop_path_rate=0.2,
            deep_supervision=True,
        )
        return native, 32
    if name == "segformer_b2":
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        # MiT-B2 configuration with random parameter initialization.
        config = SegformerConfig(
            num_channels=3,
            num_encoder_blocks=4,
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=[64, 128, 320, 512],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1,
            initializer_range=0.02,
            drop_path_rate=0.1,
            layer_norm_eps=1e-6,
            decoder_hidden_size=768,
            num_labels=2,
            reshape_last_stage=True,
            semantic_loss_ignore_index=255,
        )
        return SegformerForSemanticSegmentation(config), 768
    if name == "fpn_resnet34":
        import segmentation_models_pytorch as smp

        native = smp.FPN(
            encoder_name="resnet34",
            encoder_depth=5,
            encoder_weights=None,
            in_channels=3,
            classes=2,
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            decoder_merge_policy="add",
            decoder_dropout=0.2,
            activation=None,
            upsampling=4,
            aux_params=None,
        )
        return native, 128
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")


def _two_class_logits(output) -> torch.Tensor:
    logits = output if isinstance(output, torch.Tensor) else output.logits
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise RuntimeError(f"Expected Bx2xHxW native logits, got {logits.shape}")
    return logits


class GeometryAdapter(GeometryBase):
    """Shared V6 graph; only native input/feature adapters differ by model."""

    def __init__(self, name: str, queries: int = 16):
        if name not in MODEL_NAMES:
            raise ValueError(name)
        if queries != 16:
            raise ValueError("This fixed V6 protocol requires 16 geometry queries")
        super().__init__(queries)
        del self.enc, self.dec, self.pixel
        self.model_name = name
        self.initialization = "random_only_no_pretrained_weights"
        self.backbone, decoder_channels = _build_native(name)
        self.deep_adapter = nn.Conv2d(512, 256, 1)
        self.feature_adapter = nn.Conv2d(decoder_channels, 32, 1)
        self.alpha = 1.0
        self.last_feature_shapes: Dict[str, tuple] = {}
        mean = PREPROCESSING[name]["mean"]
        std = PREPROCESSING[name]["std"]
        self.register_buffer(
            "input_mean", torch.tensor(0.0 if mean is None else mean).reshape(1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std", torch.tensor(1.0 if std is None else std).reshape(1, 1, 1, 1),
            persistent=False,
        )
        # GeometryBase already zeroes this layer. Assert instead of applying a
        # broad initializer that could silently undo this identity guarantee.
        assert torch.count_nonzero(self.refine[-1].weight).item() == 0
        assert torch.count_nonzero(self.refine[-1].bias).item() == 0

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Input must be Bx1xHxW grayscale scaled by 1/255")
        if image.shape[-2:] != (512, 512):
            raise ValueError("This experiment requires native 512x512 images")
        rgb = image.repeat(1, 3, 1, 1)
        return (rgb - self.input_mean) / self.input_std

    def native_two_class_logits(self, image: torch.Tensor) -> torch.Tensor:
        """Native pixel logits for equivalence checks, without geometry hooks."""
        logits = _two_class_logits(self.backbone(self.preprocess(image)))
        return F.interpolate(
            logits.float(), size=image.shape[-2:], mode="bilinear", align_corners=False,
        )

    @contextmanager
    def _feature_taps(self) -> Iterator[dict]:
        # Call-local hooks retain no activation graph on the model between
        # forwards. They observe native code; they do not replace any output.
        taps = {}

        def encoder_tap(_module, _inputs, output):
            if self.model_name == "segformer_b2":
                taps["deep"] = output.hidden_states[-1]
            else:
                taps["deep"] = output[-1]

        def decoder_tap(_module, inputs):
            taps["decoder"] = inputs[0]

        if self.model_name == "vm_unet_v2":
            encoder = self.backbone.encoder
            final_classifier = self.backbone.seg_outs[3]
        elif self.model_name == "segformer_b2":
            encoder = self.backbone.segformer
            final_classifier = self.backbone.decode_head.classifier
        else:
            encoder = self.backbone.encoder
            final_classifier = self.backbone.segmentation_head
        handles = [encoder.register_forward_hook(encoder_tap),
                   final_classifier.register_forward_pre_hook(decoder_tap)]
        try:
            yield taps
        finally:
            for handle in handles:
                handle.remove()

    def _native_features(self, image):
        with self._feature_taps() as taps:
            # HF's native segmentation forward already requests encoder hidden
            # states for its decode head, so no alternate encoder path is used.
            raw = _two_class_logits(self.backbone(self.preprocess(image)))
        if set(taps) != {"deep", "decoder"}:
            raise RuntimeError(f"Native feature taps missing: {sorted(taps)}")
        deep, feature = taps["deep"], taps["decoder"]
        expected_channels = {"vm_unet_v2": 32, "segformer_b2": 768, "fpn_resnet34": 128}
        if deep.shape[1:] != (512, 16, 16):
            raise RuntimeError(f"Unexpected deepest native feature: {deep.shape}")
        if feature.shape[1:] != (expected_channels[self.model_name], 128, 128):
            raise RuntimeError(f"Unexpected native decoder feature: {feature.shape}")
        self.last_feature_shapes = {"deep": tuple(deep.shape), "decoder": tuple(feature.shape)}
        native = F.interpolate(raw.float(), size=image.shape[-2:], mode="bilinear", align_corners=False)
        pixel = native[:, 1:2] - native[:, 0:1]
        return deep, feature, pixel

    def forward(self, image):
        deep, feature, pixel = self._native_features(image)
        x = self.deep_adapter(deep)
        b = x.shape[0]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 16, device=x.device),
            torch.linspace(-1, 1, 16, device=x.device), indexing="ij",
        )
        pos = torch.stack([xx, yy])[None]
        memory = (F.adaptive_avg_pool2d(x, (16, 16)) + self.position(pos)).flatten(2).transpose(1, 2)
        q = self.query[None].expand(b, -1, -1)
        q = q + self.cross(q, memory, memory, need_weights=False)[0]
        q = q + self.self_attn(q, q, q, need_weights=False)[0]
        q = q + self.ff(self.norm(q))
        z = self.shape(q).float()
        y0 = 0.95 * z[..., 1].sigmoid()
        y1 = y0 + (1 - y0) * (0.01 + 0.99 * z[..., 3].sigmoid())
        params = torch.cat([
            torch.stack([z[..., 0].sigmoid(), y0, z[..., 2].sigmoid(), y1], -1),
            0.002 + 0.498 * z[..., 4:].sigmoid(),
        ], -1)
        obj = self.object(q).squeeze(-1).float()
        masks = ribbon(params)
        shape = 1 - (1 - masks * obj.sigmoid()[..., None, None]).prod(1, keepdim=True)
        shape = F.interpolate(shape, size=image.shape[-2:], mode="bilinear", align_corners=False)
        features = F.interpolate(
            self.feature_adapter(feature), size=image.shape[-2:], mode="bilinear", align_corners=False,
        )
        residual = self.refine(torch.cat([
            features, pixel.sigmoid().to(features.dtype), shape.to(features.dtype),
        ], 1))
        final = pixel + self.alpha * residual
        return dict(logits=final, pixel=pixel, params=params, objects=obj, geometry=shape)


def build_model(name: str) -> GeometryAdapter:
    """Construct one of the three models entirely from random initialization."""
    return GeometryAdapter(name)
