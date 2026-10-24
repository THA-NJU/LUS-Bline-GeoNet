#!/usr/bin/env python3
"""Run the installed SAM3 evaluator while repairing empty-candidate scores."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from evaluation.foundation import evaluate_sam3_official as evaluator


def union_probability(output: Mapping[str, Any], size: int) -> np.ndarray | None:
    logits = output.get("masks_logits")
    if logits is None:
        masks = output.get("masks")
        if torch.is_tensor(masks) and (masks.numel() == 0 or masks.shape[0] == 0):
            return np.zeros((size, size), dtype=np.float32)
        return None
    if not torch.is_tensor(logits):
        return None
    if logits.numel() == 0 or logits.shape[0] == 0:
        return np.zeros((size, size), dtype=np.float32)
    while logits.ndim > 3 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 3 or tuple(logits.shape[-2:]) != (size, size):
        return None
    return (
        torch.sigmoid(logits.float()).amax(dim=0)
        .detach().cpu().numpy().astype(np.float32)
    )


evaluator.union_probability = union_probability

if __name__ == "__main__":
    evaluator.main()
