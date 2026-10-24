from __future__ import annotations

import numpy as np

LUS_NORMALIZATION_NAME = "lus_ultrasound_1m_v1"
LUS_GRAY_MEAN = 0.15732
LUS_GRAY_STD = 0.16688

CNN_DOMAIN_NORMALIZED_MODELS = frozenset({
    "monai_unet",
    "monai_vnet",
    "monai_attention_unet",
    "monai_unetplusplus",
    "mednext",
    "nnunet_2d",
    "aau_net",
    "fpn_resnet34",
    "deeplabv3plus_resnet34",
})


def uses_lus_normalization(model_name: str) -> bool:
    return model_name in CNN_DOMAIN_NORMALIZED_MODELS


def normalize_lus_gray(image: np.ndarray) -> np.ndarray:
    """Input must already be float32 in [0, 1]."""
    return (image - LUS_GRAY_MEAN) / LUS_GRAY_STD


def preprocessing_metadata(model_name: str) -> dict:
    enabled = uses_lus_normalization(model_name)
    return {
        "normalization": LUS_NORMALIZATION_NAME if enabled else "model_native",
        "normalization_enabled": enabled,
        "image_scale": "uint8_to_float32_div_255",
        "mean": LUS_GRAY_MEAN if enabled else None,
        "std": LUS_GRAY_STD if enabled else None,
        "statistics_source": (
            "approximately_one_million_unlabeled_ultrasound_images"
            if enabled else None
        ),
    }
