#!/usr/bin/env python3
"""Evaluate Size_512/224 SAM2, MedSAM, or SAMUS benchmark checkpoints.

This evaluator deliberately imports the corresponding training script and
reuses its dataset, preprocessing, model builder, prompt protocol, checkpoint
loader, AMP policy, and fixed threshold.  It only replaces the evaluation
accumulator so that the prediction path is unchanged while additional metrics
are reported.

The selected training script is run in its existing ``--eval-only`` mode.
Arguments after ``--`` are forwarded to that script.  Do not include
``--eval-only`` yourself; this wrapper adds it.
python tools/evaluate.py --model samus --size 512 -- \
  --model samus \
  --benchmark-script experiments/foundation/train_samus_512.py \
  --output-dir ./output/eval_samus \
  --gpu 0 \
  -- \
  --data-root ../Myidea/Consolidation_Multiscale_Patches/Size_512 \
  --checkpoint pretrained/samus/sam_vit_b_01ec64.pth \
  --samus-source-dir ~/third_party/SAMUS \
  --resume ./benchmark_samus_consolidation_size512_20260722_230836/best_model.pth \
  --min-mask-pixels 0 \
  --threshold 0.5 \
  --batch-size 1 \
  --num-workers 4
Example:

    python tools/evaluate.py --model sam2 --size 512 -- \
      --model sam2 \
      --benchmark-script experiments/foundation/train_sam2_512.py \
      --output-dir output/eval_sam2 \
      -- \
      --data-root ../Myidea/Consolidation_Multiscale_Patches/Size_512 \
      --checkpoint pretrained/sam2/sam2.1_hiera_base_plus.pt \
      --model-cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
      --resume output/sam2_run/best_model.pth \
      --min-mask-pixels 0 \
      --threshold 0.5 \
      --batch-size 1 \
      --num-workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


MODEL_SCRIPT_DEFAULTS = {
    "sam2": "train_sam2_512.py",
    "medsam": "train_medsam_512.py",
    "samus": "train_samus_512.py",
}

CASE_FIELDS = (
    "split", "filename", "dice", "iou", "precision", "recall",
    "specificity", "tp", "fp", "fn", "tn", "gt_pixels", "pred_pixels",
    "hd", "hd95", "gt_cc", "pred_cc", "cc_delta", "abs_cc_delta",
    "pred_empty", "threshold",
)
REQUIRED_DETAILED_FIELDS = frozenset(CASE_FIELDS)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Evaluate a SAM-family checkpoint with pixel, boundary, topology, "
            "lesion, and confusion-matrix metrics."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(MODEL_SCRIPT_DEFAULTS),
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=None,
        help=(
            "Original training script used to create the checkpoint. When "
            "omitted, the model-specific filename is resolved beside this "
            "evaluator and then in the current directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Evaluation output directory; forwarded to the training script.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help=(
            "Physical GPU index exposed as cuda:0 before importing the model "
            "script. Use 0 on a single-GPU machine."
        ),
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"),
        default=["test"], help="Deterministic checkpoint-evaluation splits.",
    )
    parser.add_argument(
        "forwarded",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to the original benchmark script.",
    )
    args = parser.parse_args()
    forwarded = list(args.forwarded)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return args, forwarded


def resolve_benchmark_script(
    model: str,
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark script not found: {path}")
        return path

    filename = MODEL_SCRIPT_DEFAULTS[model]
    candidates = (
        Path(__file__).resolve().parents[2] / "experiments" / "foundation" / filename,
        Path.cwd() / filename,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Cannot find {filename}. Supply --benchmark-script explicitly."
    )


def import_benchmark_module(path: Path):
    module_name = f"_size512_{path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import benchmark script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def population_std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if values else 0.0


def require_spatial_metric_backend() -> None:
    """Fail before model loading when exact boundary metrics are unavailable."""
    try:
        import numpy as np
        from scipy.ndimage import binary_erosion, distance_transform_edt, label
    except Exception as exc:
        raise RuntimeError(
            "Exact HD/HD95 and 8-connected component metrics require numpy "
            "and scipy in the selected SAM environment. Install scipy before "
            "starting inference; no model task has been run yet."
        ) from exc

    probe = np.zeros((3, 3), dtype=bool)
    probe[1, 1] = True
    if int(label(probe, structure=np.ones((3, 3), dtype=np.uint8))[1]) != 1:
        raise RuntimeError("SciPy connected-component preflight failed")
    surface = np.logical_xor(
        probe,
        binary_erosion(
            probe, structure=np.ones((3, 3), dtype=bool), border_value=0
        ),
    )
    distances = distance_transform_edt(~surface)
    if distances.shape != probe.shape:
        raise RuntimeError("SciPy distance-transform preflight failed")


def spatial_case_metrics(prediction, truth, penalty: float) -> dict[str, Any]:
    """Compute exact symmetric surface HD/HD95 and 8-connected CC counts."""
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt, label

    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(truth, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    gt_cc = int(label(gt, structure=structure)[1])
    pred_cc = int(label(pred, structure=structure)[1])
    cc_delta = gt_cc - pred_cc

    if not gt.any():
        hd = None
        hd95 = None
    elif not pred.any():
        hd = float(penalty)
        hd95 = float(penalty)
    else:
        pred_surface = np.logical_xor(
            pred,
            binary_erosion(pred, structure=structure, border_value=0),
        )
        gt_surface = np.logical_xor(
            gt,
            binary_erosion(gt, structure=structure, border_value=0),
        )
        distances = np.concatenate(
            (
                distance_transform_edt(~gt_surface)[pred_surface],
                distance_transform_edt(~pred_surface)[gt_surface],
            )
        ).astype(np.float64, copy=False)
        hd = float(distances.max())
        hd95 = float(np.percentile(distances, 95))

    return {
        "hd": hd,
        "hd95": hd95,
        "gt_cc": gt_cc,
        "pred_cc": pred_cc,
        "cc_delta": cc_delta,
        "abs_cc_delta": abs(cc_delta),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_cases(
    model_name: str,
    split: str,
    threshold: float,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"No rows collected for split {split!r}")

    positive_rows = [row for row in rows if int(row["gt_pixels"]) > 0]
    if not positive_rows:
        raise RuntimeError(f"No GT-positive rows collected for split {split!r}")

    # The locked benchmark protocol excludes GT-empty cases from per-case
    # segmentation and topology statistics.
    tp = sum(int(row["tp"]) for row in positive_rows)
    fp = sum(int(row["fp"]) for row in positive_rows)
    fn = sum(int(row["fn"]) for row in positive_rows)
    tn = sum(int(row["tn"]) for row in positive_rows)
    dice_values = [float(row["dice"]) for row in positive_rows]
    iou_values = [float(row["iou"]) for row in positive_rows]
    precision_values = [float(row["precision"]) for row in positive_rows]
    recall_values = [float(row["recall"]) for row in positive_rows]
    specificity_values = [float(row["specificity"]) for row in positive_rows]
    hd_values = [float(row["hd"]) for row in positive_rows]
    hd95_values = [float(row["hd95"]) for row in positive_rows]
    cc_delta_values = [float(row["cc_delta"]) for row in positive_rows]
    abs_cc_delta_values = [
        float(row["abs_cc_delta"]) for row in positive_rows
    ]
    empty_prediction_count = sum(
        int(row["pred_empty"]) for row in positive_rows
    )
    empty_rows = [row for row in rows if int(row["gt_pixels"]) == 0]
    empty_false_positives = sum(
        int(row["pred_pixels"]) > 0 for row in empty_rows
    )
    image_manifest = "\n".join(
        sorted(str(row["filename"]) for row in rows)
    )

    precision_micro = safe_ratio(tp, tp + fp)
    recall_micro = safe_ratio(tp, tp + fn)
    specificity_micro = safe_ratio(tn, tn + fp)
    accuracy_micro = safe_ratio(tp + tn, tp + fp + fn + tn)
    return {
        "model": model_name,
        "split": split,
        "threshold": threshold,
        "image_count": len(rows),
        "positive_image_count": len(positive_rows),
        "evaluated_gt_nonempty_count": len(positive_rows),
        "empty_image_count": len(empty_rows),
        "whole_image_dice_mean": mean(dice_values),
        "whole_image_dice_std": population_std(dice_values),
        "whole_image_dice_median": percentile(dice_values, 0.5),
        "whole_image_dice_p25": percentile(dice_values, 0.25),
        "whole_image_dice_p75": percentile(dice_values, 0.75),
        "whole_image_iou_mean": mean(iou_values),
        "whole_image_iou_std": population_std(iou_values),
        "pixel_dice_mean": mean(dice_values),
        "pixel_dice_std": population_std(dice_values),
        "pixel_iou_mean": mean(iou_values),
        "pixel_iou_std": population_std(iou_values),
        "pixel_precision_mean": mean(precision_values),
        "pixel_precision_std": population_std(precision_values),
        "pixel_recall_mean": mean(recall_values),
        "pixel_recall_std": population_std(recall_values),
        "hd_mean": mean(hd_values),
        "hd_std": population_std(hd_values),
        "hd95_mean": mean(hd95_values),
        "hd95_std": population_std(hd95_values),
        "cc_delta_mean": mean(cc_delta_values),
        "cc_delta_std": population_std(cc_delta_values),
        "abs_cc_delta_mean": mean(abs_cc_delta_values),
        "abs_cc_delta_std": population_std(abs_cc_delta_values),
        "empty_prediction_count": empty_prediction_count,
        "empty_prediction_rate": safe_ratio(
            empty_prediction_count, len(positive_rows)
        ),
        "global_dice": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "global_iou": safe_ratio(tp, tp + fp + fn),
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "specificity_micro": specificity_micro,
        "accuracy_micro": accuracy_micro,
        "balanced_accuracy_micro": 0.5
        * (recall_micro + specificity_micro),
        "precision_macro": mean(precision_values),
        "precision_macro_std": population_std(precision_values),
        "recall_macro": mean(recall_values),
        "recall_macro_std": population_std(recall_values),
        "specificity_macro": mean(specificity_values),
        "specificity_macro_std": population_std(specificity_values),
        "empty_fp_rate": safe_ratio(
            empty_false_positives,
            len(empty_rows),
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "evaluated_filename_sha256": hashlib.sha256(
            image_manifest.encode("utf-8")
        ).hexdigest(),
    }


def dice_bin_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = [0] * 10
    for row in rows:
        dice = min(max(float(row["dice"]), 0.0), 1.0)
        counts[min(int(dice * 10), 9)] += 1
    total = max(len(rows), 1)
    return [
        {
            "dice_bin": f"{index * 10:02d}-{(index + 1) * 10:02d}%",
            "count": count,
            "fraction": count / total,
        }
        for index, count in enumerate(counts)
    ]


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_rich_evaluate(module, model_name: str, reports: dict[str, Any], output_dir: Path):
    torch = module.torch
    functional = module.F
    numpy = module.np
    tqdm = module.tqdm

    def rich_evaluate(
        model,
        loader,
        device,
        amp,
        threshold: float,
        split: str,
        collect_cases: bool,
    ) -> tuple[float, list[dict[str, Any]]]:
        del collect_cases
        model.eval()
        original_case_rows: list[dict[str, Any]] = []
        detailed_rows: list[dict[str, Any]] = []
        dice_values: list[float] = []
        lesion_run = None

        with torch.inference_mode():
            for batch in tqdm(
                loader,
                desc=f"Evaluating {split} + metrics",
                leave=False,
            ):
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                with module.autocast_context(amp):
                    logits, _ = model(images)

                # Reuse the original function so the reported Mean Dice remains
                # bit-for-bit aligned with the training script's evaluation.
                batch_dice = module.per_image_dice(
                    logits,
                    masks,
                    threshold,
                )
                resized_logits = functional.interpolate(
                    logits.float(),
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                probability = resized_logits.sigmoid()
                prediction = probability >= threshold
                truth = masks >= 0.5

                if lesion_run is None:
                    from evaluation.lesion.metrics import LesionRun
                    lesion_run = LesionRun(
                        model_name, int(truth.shape[-1]), output_dir / "lesion",
                        split=split,
                    )

                batch_size = truth.shape[0]
                prediction_flat = prediction.reshape(batch_size, -1)
                truth_flat = truth.reshape(batch_size, -1)
                tp = (prediction_flat & truth_flat).sum(dim=1)
                fp = (prediction_flat & ~truth_flat).sum(dim=1)
                fn = (~prediction_flat & truth_flat).sum(dim=1)
                tn = (~prediction_flat & ~truth_flat).sum(dim=1)

                for index, name in enumerate(batch["name"]):
                    case_tp = int(tp[index].item())
                    case_fp = int(fp[index].item())
                    case_fn = int(fn[index].item())
                    case_tn = int(tn[index].item())
                    dice = float(batch_dice[index].item())
                    union = case_tp + case_fp + case_fn
                    iou = (
                        case_tp / union
                        if union > 0
                        else 1.0
                    )
                    precision = safe_ratio(
                        case_tp,
                        case_tp + case_fp,
                    )
                    recall = safe_ratio(
                        case_tp,
                        case_tp + case_fn,
                    )
                    specificity = safe_ratio(
                        case_tn,
                        case_tn + case_fp,
                    )
                    gt_pixels = int(truth[index].sum().item())
                    pred_pixels = int(prediction[index].sum().item())
                    truth_np = truth[index, 0].detach().cpu().numpy()
                    prediction_np = prediction[index, 0].detach().cpu().numpy()
                    spatial = spatial_case_metrics(
                        prediction_np,
                        truth_np,
                        penalty=float(truth.shape[-1]),
                    )
                    if lesion_run is not None:
                        lesion_run.add(
                            name,
                            truth_np,
                            prediction_np,
                            probability[index, 0].detach().float().cpu().numpy(),
                        )
                    if gt_pixels > 0:
                        dice_values.append(dice)
                    original_case_rows.append(
                        {
                            "split": split,
                            "filename": name,
                            "dice": dice,
                            "gt_pixels": gt_pixels,
                            "pred_pixels": pred_pixels,
                            "threshold": threshold,
                        }
                    )
                    detailed_row = {
                        "split": split,
                        "filename": name,
                        "dice": dice,
                        "iou": iou,
                        "precision": precision,
                        "recall": recall,
                        "specificity": specificity,
                        "tp": case_tp,
                        "fp": case_fp,
                        "fn": case_fn,
                        "tn": case_tn,
                        "gt_pixels": gt_pixels,
                        "pred_pixels": pred_pixels,
                        "hd": spatial["hd"],
                        "hd95": spatial["hd95"],
                        "gt_cc": spatial["gt_cc"],
                        "pred_cc": spatial["pred_cc"],
                        "cc_delta": spatial["cc_delta"],
                        "abs_cc_delta": spatial["abs_cc_delta"],
                        "pred_empty": int(pred_pixels == 0),
                        "threshold": threshold,
                    }
                    missing = REQUIRED_DETAILED_FIELDS.difference(detailed_row)
                    if missing:
                        raise RuntimeError(
                            "Internal detailed-row schema error before completing "
                            f"the first batch: {sorted(missing)}"
                        )
                    detailed_rows.append(detailed_row)

        if not dice_values:
            raise RuntimeError(
                f"No samples were evaluated for split {split!r}"
            )
        # Persist the expensive per-case spatial measurements before any
        # finalization or aggregation. If a later reporting step fails, this
        # recovery CSV can be aggregated without running the model again.
        write_csv(
            output_dir / f"{split}_cases_metrics_recovery.csv",
            detailed_rows,
            CASE_FIELDS,
        )
        if lesion_run is not None:
            lesion_run.finalize()
        reports[split] = {
            "summary": summarize_cases(
                model_name,
                split,
                threshold,
                detailed_rows,
            ),
            "rows": detailed_rows,
        }
        return float(numpy.mean(dice_values)), original_case_rows

    return rich_evaluate


def ensure_forwarded_args(
    forwarded: list[str],
    output_dir: Path,
    splits: Sequence[str],
) -> list[str]:
    forbidden = {"--eval-only", "--output-dir", "--eval-splits"}
    collisions = [arg for arg in forwarded if arg in forbidden]
    if collisions:
        raise ValueError(
            "Do not place --eval-only or --output-dir after '--'; the "
            "evaluator controls them."
        )
    if "--resume" not in forwarded:
        raise ValueError(
            "Arguments after '--' must include --resume BEST_MODEL.pth"
        )
    return [
        *forwarded,
        "--no-augment",
        "--eval-splits",
        *splits,
        "--output-dir",
        str(output_dir),
        "--eval-only",
    ]


def main() -> None:
    args, forwarded = parse_args()
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")
    # Run this before importing a training script or loading a checkpoint so a
    # missing metric dependency cannot waste another inference pass.
    require_spatial_metric_backend()

    # The imported benchmark scripts always use cuda:0. Restrict visibility
    # before importing torch so cuda:0 maps to the requested physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    script_path = resolve_benchmark_script(
        args.model,
        args.benchmark_script,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    forwarded = ensure_forwarded_args(forwarded, output_dir, args.splits)

    module = import_benchmark_module(script_path)
    reports: dict[str, Any] = {}
    module.evaluate = make_rich_evaluate(module, args.model, reports, output_dir)

    old_argv = sys.argv
    sys.argv = [str(script_path), *forwarded]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    missing_splits = sorted(set(args.splits) - set(reports))
    if missing_splits:
        raise RuntimeError(
            f"The benchmark script returned without requested evaluations: {missing_splits}"
        )

    summaries = {
        split: payload["summary"]
        for split, payload in reports.items()
    }
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "benchmark_script": str(script_path),
        "gpu_requested": args.gpu,
        "metric_conventions": {
            "main_dice": (
                "whole_image_dice_mean: Dice per image, then arithmetic mean"
            ),
            "main_iou": (
                "whole_image_iou_mean: IoU per image, then arithmetic mean"
            ),
            "micro": "pool TP/FP/FN/TN across all pixels in the requested split first",
            "macro": "compute each metric per image, then arithmetic mean",
            "macro_std": "population standard deviation across GT-positive images",
            "gt_empty": "excluded from Dice/IoU/Precision/Recall/HD/HD95/CC aggregation",
            "empty_prediction_hd_penalty": "native track width in pixels (512 or 224)",
            "hd": "maximum symmetric Euclidean surface distance",
            "hd95": "95th percentile of pooled bidirectional Euclidean surface distances",
            "connected_components": "8-connectivity; cc_delta = gt_cc - pred_cc",
            "undefined_precision_recall": 0.0,
            "empty_empty_dice_iou": 1.0,
        },
        "splits": summaries,
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_fields = list(next(iter(summaries.values())).keys())
    write_csv(
        output_dir / "metrics_summary.csv",
        summaries.values(),
        summary_fields,
    )
    for split, payload in reports.items():
        rows = payload["rows"]
        # This standardized summary is the source consumed by the final
        # collector.  It intentionally replaces any legacy sparse summary.
        (output_dir / f"{split}_summary.json").write_text(
            json.dumps(payload["summary"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_csv(
            output_dir / f"{split}_cases_metrics.csv",
            rows,
            CASE_FIELDS,
        )
        # Keep the historical canonical filename, but enrich it with all
        # per-case metrics so no later recovery depends on discarded masks.
        write_csv(
            output_dir / f"{split}_cases.csv",
            rows,
            CASE_FIELDS,
        )
        write_csv(
            output_dir / f"{split}_dice_bins.csv",
            dice_bin_rows(rows),
            ("dice_bin", "count", "fraction"),
        )

    print("\n" + "=" * 76)
    print(f"MODEL: {args.model}")
    for split in args.splits:
        summary = summaries[split]
        print(f"{split}: images={summary['image_count']}, "
              f"Dice={summary['whole_image_dice_mean']:.6f}, "
              f"IoU={summary['whole_image_iou_mean']:.6f}")
    report_split = "test" if "test" in summaries else args.splits[-1]
    report = summaries[report_split]
    print(
        f"{report_split} Global Dice / Global IoU: "
        f"{report['global_dice']:.6f} / {report['global_iou']:.6f}"
    )
    print(
        f"{report_split} Precision / Recall / Specificity (micro): "
        f"{report['precision_micro']:.6f} / "
        f"{report['recall_micro']:.6f} / "
        f"{report['specificity_micro']:.6f}"
    )
    print(f"Metrics written to: {output_dir}")


if __name__ == "__main__":
    main()
