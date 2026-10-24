"""Streaming pixel- and lesion-level metrics for one test-only inference run.

The caller supplies native-resolution GT, binary prediction, and (when
available) a calibrated foreground probability map for every image.  No model
or preprocessing assumptions live in this module.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)


def _ratio(a: int, b: int) -> float | None:
    return float(a / b) if b else None


def _binary(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 2-D array")
    return array.astype(bool, copy=False)


def _probability(value: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape != shape or array.dtype.kind != "f":
        raise ValueError(f"Probability must be floating HxW with shape {shape}")
    if not np.isfinite(array).all() or float(array.min()) < 0 or float(array.max()) > 1:
        raise ValueError("Probability contains non-finite values or values outside [0,1]")
    return array.astype(np.float64, copy=False)


def _components(mask: np.ndarray):
    labels, count = ndimage.label(mask, STRUCTURE_8)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    return labels, int(count), areas


def _component_data(gt: np.ndarray, pred: np.ndarray):
    gt_labels, ng, gt_areas = _components(gt)
    pred_labels, npred, pred_areas = _components(pred)
    overlap = (gt_labels > 0) & (pred_labels > 0)
    encoded, intersections = np.unique(
        pred_labels[overlap].astype(np.int64) * (ng + 1)
        + gt_labels[overlap].astype(np.int64),
        return_counts=True,
    )
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(npred)]
    overlaps: list[dict[str, Any]] = []
    for code, intersection in zip(encoded, intersections):
        pred_id, gt_id = divmod(int(code), ng + 1)
        union = int(pred_areas[pred_id] + gt_areas[gt_id] - intersection)
        iou = float(intersection / union)
        adjacency[pred_id - 1].append((gt_id - 1, iou))
        overlaps.append({
            "gt_component_id": gt_id,
            "pred_component_id": pred_id,
            "intersection_pixels": int(intersection),
            "union_pixels": union,
            "component_iou": iou,
        })
    return gt_labels, pred_labels, ng, npred, gt_areas, pred_areas, adjacency, overlaps


def _maximum_matches(adjacency: Sequence[Sequence[tuple[int, float]]], ng: int,
                     threshold: float) -> list[tuple[int, int, float]]:
    edges = [(p, g, iou) for p, values in enumerate(adjacency)
             for g, iou in values if iou > threshold]
    if not edges:
        return []
    graph = csr_matrix(
        (np.ones(len(edges), dtype=np.uint8),
         ([p for p, _, _ in edges], [g for _, g, _ in edges])),
        shape=(len(adjacency), ng),
    )
    assignment = maximum_bipartite_matching(graph, perm_type="column")
    iou_lookup = {(p, g): iou for p, g, iou in edges}
    return [(p, int(g), float(iou_lookup[(p, int(g))]))
            for p, g in enumerate(assignment) if g >= 0]


def _ranked_decisions(adjacency, scores, threshold):
    used_gt: set[int] = set()
    rows = []
    for pred_id in sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i)):
        candidates = [(g, iou) for g, iou in adjacency[pred_id]
                      if iou > threshold and g not in used_gt]
        if candidates:
            gt_id, iou = min(candidates, key=lambda item: (-item[1], item[0]))
            used_gt.add(gt_id)
            rows.append((pred_id, 1, gt_id, iou))
        else:
            rows.append((pred_id, 0, None, None))
    return rows


def _ranked_curve(detections, gt_lesions: int, images: int, budgets):
    ordered = sorted(detections,
                     key=lambda row: (-row["score"], row["image_id"],
                                      row["pred_component_id"]))
    curve = [{"score_threshold": None, "tp": 0, "fp": 0,
              "precision": 1.0, "recall": 0.0 if gt_lesions else None,
              "fp_per_image": 0.0}]
    tp = fp = index = 0
    while index < len(ordered):
        score = ordered[index]["score"]
        end = index
        while end < len(ordered) and ordered[end]["score"] == score:
            tp += int(ordered[end]["is_tp"])
            fp += 1 - int(ordered[end]["is_tp"])
            end += 1
        curve.append({"score_threshold": score, "tp": tp, "fp": fp,
                      "precision": tp / (tp + fp),
                      "recall": tp / gt_lesions if gt_lesions else None,
                      "fp_per_image": fp / images})
        index = end
    ap = None
    if gt_lesions:
        recall = np.asarray([point["recall"] for point in curve] + [1.0])
        precision = np.asarray([point["precision"] for point in curve] + [0.0])
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        ap = float(np.sum(np.diff(recall) * envelope[1:]))
    froc = {}
    for budget in budgets:
        candidates = [point for point in curve if point["fp_per_image"] <= budget]
        best = max(candidates, key=lambda point: (point["tp"], -point["fp"]))
        froc[str(float(budget))] = {
            "sensitivity": best["recall"],
            "actual_fp_per_image": best["fp_per_image"],
            "score_threshold": best["score_threshold"],
            "budget_exceeds_observed_max": budget > curve[-1]["fp_per_image"],
        }
    return ap, froc, curve


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


class LesionRun:
    """Accumulate one model/size/split run and emit auditable CSV artifacts."""

    def __init__(self, model: str, size: int, output_dir: Path,
                 split: str = "test",
                 thresholds=(0.2, 0.3, 0.5), pixel_threshold=0.5,
                 fpi_budgets=(0.25, 0.5, 1.0, 2.0)):
        self.model = str(model)
        self.size = int(size)
        self.output_dir = Path(output_dir)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.split = split
        self.thresholds = tuple(float(value) for value in thresholds)
        self.pixel_threshold = float(pixel_threshold)
        self.fpi_budgets = tuple(float(value) for value in fpi_budgets)
        self.ids: set[str] = set()
        self.cases: list[dict[str, Any]] = []
        self.gt_components: list[dict[str, Any]] = []
        self.pred_components: list[dict[str, Any]] = []
        self.overlaps: list[dict[str, Any]] = []
        self.detections = {threshold: [] for threshold in self.thresholds}
        self.fixed = {threshold: [] for threshold in self.thresholds}
        self.probability_images = 0
        self.score_sources: set[str] = set()

    def add(self, image_id: str, gt: Any, prediction: Any,
            probability: Any | None = None, inference_time_ms: float | None = None,
            score_source: str = "mean_foreground_probability"):
        image_id = Path(str(image_id)).stem
        if image_id in self.ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        gt = _binary(gt, "GT")
        pred = _binary(prediction, "prediction")
        if gt.shape != pred.shape or gt.shape != (self.size, self.size):
            raise ValueError(f"Native shape mismatch for {image_id}: {gt.shape}, {pred.shape}")
        prob = _probability(probability, gt.shape) if probability is not None else None
        self.ids.add(image_id)
        if prob is not None:
            self.probability_images += 1
            self.score_sources.add(score_source)

        tp_pixels = int((gt & pred).sum())
        fp_pixels = int((~gt & pred).sum())
        fn_pixels = int((gt & ~pred).sum())
        gt_pixels, pred_pixels = int(gt.sum()), int(pred.sum())
        dice_den = 2 * tp_pixels + fp_pixels + fn_pixels
        iou_den = tp_pixels + fp_pixels + fn_pixels
        case = {
            "image_id": image_id,
            "gt_pixels": gt_pixels,
            "pred_pixels": pred_pixels,
            "pixel_tp": tp_pixels,
            "pixel_fp": fp_pixels,
            "pixel_fn": fn_pixels,
            "dice": 2 * tp_pixels / dice_den if dice_den else 1.0,
            "iou": tp_pixels / iou_den if iou_den else 1.0,
            "recall": _ratio(tp_pixels, tp_pixels + fn_pixels),
            "precision": _ratio(tp_pixels, tp_pixels + fp_pixels) or 0.0,
            "empty_prediction": int(pred_pixels == 0),
            "inference_time_ms": inference_time_ms,
        }
        (_, pred_labels, ng, npred, gt_areas, pred_areas,
         adjacency, overlaps) = _component_data(gt, pred)
        case.update(gt_components=ng, pred_components=npred)
        gt_best: list[tuple[int | None, float]] = [(None, 0.0) for _ in range(ng)]
        for pred_index, candidates in enumerate(adjacency):
            for gt_index, component_iou in candidates:
                if component_iou > gt_best[gt_index][1]:
                    gt_best[gt_index] = (pred_index + 1, float(component_iou))
        for gt_id in range(1, ng + 1):
            best_pred, best_iou = gt_best[gt_id - 1]
            self.gt_components.append({
                "image_id": image_id,
                "gt_component_id": gt_id,
                "area_pixels": int(gt_areas[gt_id]),
                "best_pred_component_id": best_pred,
                "best_iou": best_iou,
            })
        scores = None
        if prob is not None:
            sums = np.bincount(pred_labels.ravel(), weights=prob.ravel(),
                               minlength=npred + 1)
            scores = sums[1:] / pred_areas[1:] if npred else np.asarray([])
        for pred_id in range(1, npred + 1):
            best = max(adjacency[pred_id - 1], key=lambda item: item[1], default=None)
            self.pred_components.append({
                "image_id": image_id, "pred_component_id": pred_id,
                "area_pixels": int(pred_areas[pred_id]),
                "score": None if scores is None else float(scores[pred_id - 1]),
                "score_source": None if scores is None else score_source,
                "best_gt_component_id": None if best is None else best[0] + 1,
                "best_iou": 0.0 if best is None else float(best[1]),
            })
        for row in overlaps:
            self.overlaps.append({"image_id": image_id, **row})

        for threshold in self.thresholds:
            matches = _maximum_matches(adjacency, ng, threshold)
            count = len(matches)
            suffix = str(threshold).replace(".", "")
            case[f"lesion_tp_iou{suffix}"] = count
            case[f"lesion_fp_iou{suffix}"] = npred - count
            case[f"lesion_fn_iou{suffix}"] = ng - count
            for pred_id, gt_id, iou in matches:
                self.fixed[threshold].append({
                    "image_id": image_id, "pred_component_id": pred_id + 1,
                    "gt_component_id": gt_id + 1, "matched_iou": iou,
                })
            if scores is not None:
                for pred_id, is_tp, gt_id, iou in _ranked_decisions(
                    adjacency, scores, threshold
                ):
                    self.detections[threshold].append({
                        "image_id": image_id, "pred_component_id": pred_id + 1,
                        "score": float(scores[pred_id]), "is_tp": int(is_tp),
                        "matched_gt_component_id": None if gt_id is None else gt_id + 1,
                        "matched_iou": iou,
                    })
        self.cases.append(case)

    def finalize(self) -> dict[str, Any]:
        if not self.cases:
            raise RuntimeError("No cases were accumulated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ranked_available = self.probability_images == len(self.cases)
        total_gt = len(self.gt_components)
        total_pred = len(self.pred_components)
        metric_rows = []
        summary_metrics = []
        for threshold in self.thresholds:
            tp = sum(int(row[f"lesion_tp_iou{str(threshold).replace('.', '')}"])
                     for row in self.cases)
            fp, fn = total_pred - tp, total_gt - tp
            ap = froc = curve = None
            if ranked_available:
                ap, froc, curve = _ranked_curve(
                    self.detections[threshold], total_gt, len(self.cases), self.fpi_budgets
                )
                tag = int(round(threshold * 100))
                _write_csv(self.output_dir / f"{self.split}_detections_iou{tag}.csv",
                           self.detections[threshold])
                _write_csv(self.output_dir / f"{self.split}_pr_froc_iou{tag}.csv", curve)
            metric = {
                "model": self.model, "size": self.size, "split": self.split,
                "iou_threshold": threshold, "images": len(self.cases),
                "gt_lesions": total_gt, "pred_lesions": total_pred,
                "lesion_tp": tp, "lesion_fp": fp, "lesion_fn": fn,
                "lesion_precision": _ratio(tp, tp + fp) or 0.0,
                "lesion_recall": _ratio(tp, tp + fn),
                "lesion_f1": _ratio(2 * tp, 2 * tp + fp + fn),
                "ap_lesion": ap,
                "froc_recall_fpi_0_25": None if froc is None else froc["0.25"]["sensitivity"],
                "froc_recall_fpi_0_5": None if froc is None else froc["0.5"]["sensitivity"],
                "froc_recall_fpi_1": None if froc is None else froc["1.0"]["sensitivity"],
                "froc_recall_fpi_2": None if froc is None else froc["2.0"]["sensitivity"],
            }
            metric_rows.append(metric)
            summary_metrics.append({**metric, "froc": froc})
        prefix = self.split
        _write_csv(self.output_dir / f"{prefix}_cases.csv", self.cases)
        _write_csv(self.output_dir / f"{prefix}_gt_components.csv", self.gt_components,
                   ("image_id", "gt_component_id", "area_pixels",
                    "best_pred_component_id", "best_iou"))
        _write_csv(self.output_dir / f"{prefix}_pred_components.csv", self.pred_components,
                   ("image_id", "pred_component_id", "area_pixels", "score", "score_source",
                    "best_gt_component_id", "best_iou"))
        _write_csv(self.output_dir / f"{prefix}_component_overlaps.csv", self.overlaps,
                   ("image_id", "gt_component_id", "pred_component_id",
                    "intersection_pixels", "union_pixels", "component_iou"))
        _write_csv(self.output_dir / f"{prefix}_lesion_metrics.csv", metric_rows)
        gt_nonempty = sum(int(row["gt_pixels"] > 0) for row in self.cases)
        eligible_cases = [row for row in self.cases if row["gt_pixels"] > 0]
        pixel_case_metrics: dict[str, Any] = {
            "evaluated_gt_nonempty_count": gt_nonempty,
        }
        for name in ("dice", "iou", "recall", "precision"):
            values = np.asarray(
                [float(row[name]) for row in eligible_cases], dtype=np.float64
            )
            pixel_case_metrics[f"pixel_{name}_mean"] = (
                float(values.mean()) if values.size else None
            )
            pixel_case_metrics[f"pixel_{name}_std"] = (
                float(values.std(ddof=0)) if values.size else None
            )
        empty_prediction_count = sum(
            int(row["gt_pixels"] > 0 and row["empty_prediction"])
            for row in self.cases
        )
        summary = {
            "protocol_version": "lusbench-lesion-v2",
            "model": self.model, "size": self.size, "split": self.split,
            "images": len(self.cases),
            "pixel_threshold": self.pixel_threshold,
            "connectivity": 8, "iou_rule": "strictly_greater_than",
            "gt_component_preprocessing": "remove area < 5 pixels in derived GT",
            "prediction_component_filter": "none",
            "fixed_matching": "maximum_cardinality_one_to_one",
            "ranked_matching": "descending_component_score_greedy_one_to_one",
            "ap_definition": "all_point_interpolated_precision_envelope_area",
            "froc_fixed_fpi_rule": "best_empirical_sensitivity_with_observed_fpi_not_exceeding_budget",
            "probability_images": self.probability_images,
            "score_sources": sorted(self.score_sources),
            "ranked_metrics_available": ranked_available,
            **pixel_case_metrics,
            "empty_prediction_count": empty_prediction_count,
            "empty_prediction_rate": empty_prediction_count / gt_nonempty if gt_nonempty else None,
            "metrics": summary_metrics,
        }
        (self.output_dir / f"{prefix}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return summary
