# Evaluation protocol

The common evaluator accepts paired 512 × 512 image and binary-mask directories. Original and manually processed image tracks are evaluated separately against their corresponding mask directories. Input checksums and the checkpoint checksum are recorded with each result.

## Metrics

The 59-field summary comprises:

- Mean and population standard deviation for Dice, IoU, pixel recall, pixel precision, HD, HD95, connected-component count difference, absolute count difference and inference time.
- Empty-prediction count and rate.
- Thirteen lesion fields at each strict IoU threshold 0.2, 0.3 and 0.5: GT count, prediction count, TP, FP, FN, precision, recall, F1, AP and four FROC operating points.

Components use 8-connectivity. Lesion TP uses maximum-cardinality one-to-one matching over pairs exceeding the specified IoU threshold. F1 is computed from pooled TP/FP/FN. HD and HD95 use symmetric boundary distances; an empty prediction receives a 512-pixel penalty. The B-line evaluation track requires nonempty ground truth.

The optional `centerline100` condition removes short B-line components using the supplied centerline rule. Raw results are always retained. Prediction masks are consumed in memory; the evaluator writes metrics and case records.

## Timing

Per-case latency is measured with CUDA synchronization around a prepared-input forward pass and probability conversion. Image loading, host-to-device transfer, CPU metrics and centerline postprocessing are outside this interval. CUDA Graph's static-input copy is inside it. Warmup precedes measurements. YOLO's native `predict` pipeline includes its own preprocessing/postprocessing and is identified separately in the result metadata.

The measured B-GeoNet deployment reference is **4.557 ± 0.016 ms/input** on an **NVIDIA GeForce RTX 4060 Ti 16 GB**, with PyTorch 2.5.1 / CUDA 12.4 (300 synchronized batch-1 runs, 512 × 512, BF16 network / FP32 raster). This timing standard deviation describes those repeated latency measurements.

Checkpoint selection and training hyperparameter selection use validation data.
