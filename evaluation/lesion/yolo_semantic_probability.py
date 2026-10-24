"""Recover dense foreground probabilities from Ultralytics semantic logits.

Ultralytics ``Results.semantic_mask`` contains the final class-ID map.  AP and
FROC require a continuous score, so the evaluator installs a small postprocess
tap that preserves the raw semantic logits and transforms them with the same
resize/unpadding operation used by Ultralytics before the logits are discarded.
"""
from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


RESULT_ATTRIBUTE = "_lusbench_foreground_probability"
TAP_ATTRIBUTE = "_lusbench_semantic_probability_tap_installed"


def _unwrap_predictions(predictions: Any) -> torch.Tensor:
    value = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    if not torch.is_tensor(value) or value.ndim != 4:
        raise RuntimeError(
            "Expected raw YOLO semantic logits with shape [B,C,H,W]; "
            f"received {type(value).__name__} with shape={getattr(value, 'shape', None)}"
        )
    return value


def foreground_probability_from_logits(
    logits: torch.Tensor,
    input_shape: tuple[int, int],
    original_shape: tuple[int, int],
    foreground_class: int = 1,
    scale_masks_fn=None,
) -> np.ndarray:
    """Transform one ``[C,H,W]`` logit tensor into native foreground probability."""
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError(f"Semantic logits must have shape [C,H,W], got {getattr(logits, 'shape', None)}")
    channels = int(logits.shape[0])
    if channels == 1:
        if foreground_class not in (0, 1):
            raise ValueError(f"Single-channel semantic output has no class {foreground_class}")
    elif foreground_class < 0 or foreground_class >= channels:
        raise ValueError(f"foreground_class={foreground_class} is invalid for {channels} output classes")

    value = logits[None].float()
    if tuple(value.shape[-2:]) != tuple(input_shape):
        value = functional.interpolate(
            value, size=input_shape, mode="bilinear", align_corners=False
        )
    if scale_masks_fn is None:
        from ultralytics.utils import ops
        scale_masks_fn = ops.scale_masks
    value = scale_masks_fn(value, original_shape)[0]
    probability = (
        torch.sigmoid(value[0])
        if channels == 1
        else torch.softmax(value, dim=0)[foreground_class]
    )
    array = probability.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.ndim != 2 or tuple(array.shape) != tuple(original_shape):
        raise RuntimeError(
            f"Recovered probability has shape {array.shape}; expected {original_shape}"
        )
    if not np.isfinite(array).all() or float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise RuntimeError("Recovered YOLO semantic probability is invalid")
    return array


def install_probability_tap(model: Any, foreground_class: int = 1) -> None:
    """Install once after the first ``model.predict`` initialized its predictor."""
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        raise RuntimeError("YOLO predictor is not initialized; run one warm-up prediction first")
    if getattr(predictor, TAP_ATTRIBUTE, False):
        return
    original_postprocess = predictor.postprocess

    def postprocess_with_probability(self, predictions, image, original_images, *args, **kwargs):
        raw = _unwrap_predictions(predictions)
        results = original_postprocess(predictions, image, original_images, *args, **kwargs)
        if len(results) != int(raw.shape[0]):
            raise RuntimeError(
                f"YOLO result/logit batch mismatch: {len(results)} != {int(raw.shape[0])}"
            )
        input_shape = tuple(int(value) for value in image.shape[-2:])
        for result, one_logits in zip(results, raw):
            original_shape = tuple(int(value) for value in result.orig_img.shape[:2])
            probability = foreground_probability_from_logits(
                one_logits, input_shape, original_shape, foreground_class
            )
            setattr(result, RESULT_ATTRIBUTE, probability)
        return results

    predictor.postprocess = MethodType(postprocess_with_probability, predictor)
    setattr(predictor, TAP_ATTRIBUTE, True)


def result_foreground_probability(result: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    value = getattr(result, RESULT_ATTRIBUTE, None)
    if value is None:
        raise RuntimeError(
            "YOLO semantic foreground probability was not captured; "
            "the probability tap must be installed after predictor initialization"
        )
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or tuple(array.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"YOLO semantic probability shape {array.shape} != {expected_shape}"
        )
    if not np.isfinite(array).all() or float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise RuntimeError("YOLO semantic probability contains invalid values")
    return array
