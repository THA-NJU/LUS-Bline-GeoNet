"""Evaluate B-GeoNet or a registered benchmark on paired 512-pixel images/masks."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from metrics.canonical import case, aggregate
from metrics.lesion_metrics import LesionRun
from metrics.centerline100 import filter_short


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def csvwrite(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class BGeoNetAdapter:
    def __init__(self, checkpoint, eager=False):
        from bgeonet.runtime import Runtime
        self.runtime = Runtime(checkpoint, graph=not eager, compile_modules=not eager)
        self.model = self.runtime.model

    def prepare(self, path):
        with Image.open(path) as image:
            array = np.asarray(image.convert('L')).copy()
        if array.shape != (512, 512):
            raise ValueError('Input must already be 512 x 512')
        return torch.from_numpy(array[None, None]).float().cuda() / 255

    def start(self, image):
        pass

    def predict(self, image):
        return self.runtime(image)

    def unpack(self, result):
        probability = result.cpu().numpy().copy()
        return probability > .5, probability


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--masks', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--postprocess', choices=['none', 'centerline100'], default='none')
    parser.add_argument('--eager', action='store_true', help='Use eager B-GeoNet execution')
    args = parser.parse_args()
    paths = sorted(args.images.glob('*.png'))
    if not paths or {p.name for p in paths} != {p.name for p in args.masks.glob('*.png')}:
        raise ValueError('Expected nonempty paired PNG image and mask directories')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = [dict(filename=p.name, image_sha256=sha(p), mask_sha256=sha(args.masks/p.name)) for p in paths]
    (args.output/'inputs.json').write_text(json.dumps(manifest, indent=2))
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    if args.model == 'bgeonet':
        adapter = BGeoNetAdapter(args.checkpoint, args.eager)
    else:
        from .adapters import Adapter
        registry = json.loads(Path(__file__).with_name('models.json').read_text())
        job = dict(registry[args.model], checkpoint=str(args.checkpoint))
        adapter = Adapter(job)
    modes = ['raw'] + (['centerline100'] if args.postprocess == 'centerline100' else [])
    runs, rows = {}, {}
    for mode in modes:
        output = args.output/mode; output.mkdir()
        runs[mode] = LesionRun(args.model, 512, output/'lesion', split='test')
        rows[mode] = []
    image = adapter.prepare(paths[0]); adapter.start(image)
    for _ in range(10):
        adapter.predict(image)
    for index, path in enumerate(paths):
        image = adapter.prepare(path)
        torch.cuda.synchronize(); start = time.perf_counter()
        result = adapter.predict(image)
        torch.cuda.synchronize(); ms = (time.perf_counter()-start)*1000
        pred, probability = adapter.unpack(result)
        pred, probability = np.asarray(pred, bool), np.asarray(probability, np.float32)
        if pred.shape != (512, 512) or probability.shape != pred.shape or not np.isfinite(probability).all():
            raise ValueError('Invalid prediction')
        with Image.open(args.masks/path.name) as image:
            gt = np.asarray(image)
        if gt.ndim != 2 or not set(np.unique(gt)).issubset({0, 1}):
            raise ValueError('Masks must be single-channel, label 0/1 PNGs')
        gt = gt > 0
        predictions = {'raw': (pred, 0.)}
        if 'centerline100' in modes:
            start = time.perf_counter(); post, _ = filter_short(pred)
            predictions['centerline100'] = (post, (time.perf_counter()-start)*1000)
        for mode, (mask, post_ms) in predictions.items():
            runs[mode].add(path.name, gt, mask, probability, ms,
                           score_source='native_instance_confidence' if args.model.startswith('yolo11') else 'mean_foreground_probability')
            rows[mode].append(dict(image_id=path.stem, filename=path.name, **case(gt, mask, ms), postprocessing_time_ms=post_ms))
        if index % 100 == 0:
            print(f'{index+1}/{len(paths)}', flush=True)
    for mode in modes:
        lesion = runs[mode].finalize()
        detail = {Path(row['image_id']).stem: row for row in runs[mode].cases}
        for row in rows[mode]:
            row.update({k: v for k, v in detail[row['image_id']].items() if k not in row})
        metrics = aggregate(rows[mode], lesion)
        report = dict(model=args.model, images=len(paths), postprocessing=mode,
                      checkpoint_sha256=sha(args.checkpoint), std_ddof=0, threshold=.5,
                      timing='CUDA synchronized prepared-input forward and probability conversion; excludes I/O and CPU metrics. YOLO includes its native predict preprocessing/postprocessing.', **metrics)
        output = args.output/mode
        csvwrite(output/'test_cases.csv', rows[mode]); csvwrite(output/'metrics.csv', [metrics])
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
