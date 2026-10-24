"""B-line canonical 59 fields; all pixel aggregates are per-image ddof=0."""
import numpy as np
from scipy import ndimage

PIXEL = ['dice', 'iou', 'recall', 'precision', 'hd', 'hd95', 'cc_delta', 'abs_cc_delta', 'inference_time_ms']
LESION = ['gt_lesions', 'pred_lesions', 'lesion_tp', 'lesion_fp', 'lesion_fn', 'lesion_precision',
          'lesion_recall', 'lesion_f1', 'ap_lesion', 'froc_recall_fpi_0_25', 'froc_recall_fpi_0_5',
          'froc_recall_fpi_1', 'froc_recall_fpi_2']
FIELDS = [f'{m}_{s}' for m in PIXEL for s in ('mean', 'std')] + ['empty_prediction_count', 'empty_prediction_rate'] + [f'{m}_iou{t}' for t in (20, 30, 50) for m in LESION]
STRUCT = np.ones((3, 3), dtype=np.uint8)


def case(gt, pred, ms):
    gt, pred = np.asarray(gt, dtype=bool), np.asarray(pred, dtype=bool)
    if gt.shape != (512, 512) or pred.shape != gt.shape or not gt.any():
        raise ValueError('Expected original 512x512 nonempty B-line GT')
    if ms is None or not np.isfinite(ms) or ms < 0:
        raise ValueError('Missing or invalid inference latency')
    ng, npred = int(gt.sum()), int(pred.sum())
    tp = int((gt & pred).sum())
    if npred == 0:
        hd = hd95 = 512.0
    else:
        gs = gt & ~ndimage.binary_erosion(gt, structure=STRUCT, border_value=0)
        ps = pred & ~ndimage.binary_erosion(pred, structure=STRUCT, border_value=0)
        distances = np.concatenate([ndimage.distance_transform_edt(~ps)[gs], ndimage.distance_transform_edt(~gs)[ps]])
        hd, hd95 = float(distances.max()), float(np.percentile(distances, 95, method='linear'))
    delta = int(ndimage.label(gt, STRUCT)[1] - ndimage.label(pred, STRUCT)[1])
    return {'dice': 2*tp/(ng+npred), 'iou': tp/(ng+npred-tp), 'recall': tp/ng,
            'precision': tp/npred if npred else 0., 'hd': hd, 'hd95': hd95,
            'cc_delta': delta, 'abs_cc_delta': abs(delta), 'inference_time_ms': float(ms),
            'empty_prediction': int(npred == 0)}


def aggregate(rows, lesion):
    result = {}
    for metric in PIXEL:
        values = np.array([r[metric] for r in rows], dtype=np.float64)
        result[metric+'_mean'], result[metric+'_std'] = float(values.mean()), float(values.std(ddof=0))
    result['empty_prediction_count'] = sum(r['empty_prediction'] for r in rows)
    result['empty_prediction_rate'] = result['empty_prediction_count']/len(rows)
    for item in lesion['metrics']:
        tag = int(round(item['iou_threshold']*100))
        for key in LESION:
            result[f'{key}_iou{tag}'] = item[key]
    if set(result) != set(FIELDS) or not all(v is not None and np.isfinite(v) for v in result.values()):
        raise ValueError('59-field schema incomplete or nonfinite; ranked probability outputs required')
    return result
