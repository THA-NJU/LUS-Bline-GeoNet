#!/usr/bin/env python3
"""Recompute lesion artifacts for a trained YOLO semantic checkpoint.

This is deliberately independent of the historical YOLO training/evaluation
scripts.  It preserves raw semantic logits before Ultralytics reduces them to
``Results.semantic_mask`` and writes only the auditable ``lesion`` artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from evaluation.lesion.metrics import LesionRun
from evaluation.lesion.yolo_semantic_probability import (
    install_probability_tap,
    result_foreground_probability,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lesion-output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", type=int, choices=(224, 512), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--foreground-class", type=int, default=1)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def normalized_key(path: Path) -> str:
    stem = path.stem
    if stem.casefold().endswith("_mask"):
        stem = stem[:-5]
    return stem.casefold()


def pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = data_root / split / "images"
    mask_dir = data_root / split / "masks"
    images = {
        normalized_key(path): path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    }
    masks = {
        normalized_key(path): path
        for path in sorted(mask_dir.iterdir())
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    }
    if not images or images.keys() != masks.keys():
        raise RuntimeError(
            f"Image/mask mismatch in {data_root / split}: "
            f"images={len(images)}, masks={len(masks)}, "
            f"missing_masks={sorted(images.keys() - masks.keys())[:10]}, "
            f"missing_images={sorted(masks.keys() - images.keys())[:10]}"
        )
    return [(images[key], masks[key]) for key in sorted(images)]


def load_gt(path: Path, size: int) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if value.ndim == 3:
        value = value[..., 0]
    if value.shape != (size, size):
        raise RuntimeError(f"Native GT shape {value.shape} != {(size, size)}: {path}")
    return (value > 0).astype(np.uint8)


def class_prediction(result, size: int, foreground_class: int) -> np.ndarray:
    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is None:
        raise RuntimeError("Ultralytics result lacks semantic_mask")
    value = semantic_mask.data
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    class_map = np.squeeze(np.asarray(value))
    if class_map.ndim != 2:
        raise RuntimeError(f"Unexpected semantic class-map shape: {class_map.shape}")
    if class_map.shape != (size, size):
        class_map = cv2.resize(
            class_map.astype(np.int32), (size, size), interpolation=cv2.INTER_NEAREST
        )
    return (class_map == foreground_class).astype(np.uint8)


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.lesion_output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    selected_pairs = pairs(data_root, args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(checkpoint))

    # First call constructs the Ultralytics predictor.  All subsequent calls
    # pass through the raw-logit tap.
    warmup = str(selected_pairs[0][0])
    model.predict(
        source=warmup, imgsz=args.size, batch=1,
        device=args.device, verbose=False, stream=False,
    )
    install_probability_tap(model, foreground_class=args.foreground_class)
    warm_results = model.predict(
        source=warmup, imgsz=args.size, batch=1,
        device=args.device, verbose=False, stream=False,
    )
    if args.smoke_only:
        probability = result_foreground_probability(
            warm_results[0], (args.size, args.size)
        )
        print(
            f"YOLO_PROBABILITY_SMOKE_OK model={args.model} size={args.size} "
            f"shape={probability.shape} "
            f"range=[{float(probability.min()):.6g},{float(probability.max()):.6g}]"
        )
        return 0

    lesion_run = LesionRun(
        args.model, args.size, output_dir, split=args.split,
        thresholds=(0.2, 0.3, 0.5), pixel_threshold=0.5,
        fpi_budgets=(0.25, 0.5, 1.0, 2.0),
    )
    for start in tqdm(
        range(0, len(selected_pairs), args.batch_size),
        desc=f"{args.model} {args.size} {args.split}", unit="batch",
    ):
        chunk = selected_pairs[start:start + args.batch_size]
        results = model.predict(
            source=[str(image) for image, _ in chunk],
            imgsz=args.size,
            batch=min(args.batch_size, len(chunk)),
            device=args.device,
            verbose=False,
            stream=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"Prediction count mismatch: {len(results)} != {len(chunk)}")
        for (image_path, mask_path), result in zip(chunk, results):
            pred = class_prediction(result, args.size, args.foreground_class)
            probability = result_foreground_probability(result, (args.size, args.size))
            inference_ms = float(
                getattr(result, "speed", {}).get("inference", float("nan"))
            )
            lesion_run.add(
                image_path.name,
                load_gt(mask_path, args.size),
                pred,
                probability,
                inference_time_ms=inference_ms,
                score_source="mean_softmax_foreground_probability",
            )
    summary = lesion_run.finalize()
    if (
        not summary["ranked_metrics_available"]
        or int(summary["probability_images"]) != int(summary["images"])
    ):
        raise RuntimeError(
            f"Probability coverage incomplete: "
            f"{summary['probability_images']}/{summary['images']}"
        )
    print(
        f"YOLO_RANKED_OK model={args.model} size={args.size} split={args.split} "
        f"probability={summary['probability_images']}/{summary['images']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
