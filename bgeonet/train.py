"""Train B-GeoNet with validation-selected checkpoints and full-state resume."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import BlineDataset, collate
from .geometry import losses
from .model import BGeoNet


def rng_state():
    state = np.random.get_state()
    return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(), numpy_rng=[state[0], state[1].tolist(), *state[2:]])


def restore_rng(state):
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    n = state['numpy_rng']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))


def save(path, checkpoint):
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/bgeonet.json'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(batch_size=args.batch_size, workers=args.workers, validation_batch=4)
    torch.set_num_threads(4)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text())
    datasets = {s: BlineDataset(args.root, s, metadata, s == 'train') for s in ('train', 'val')}
    if any(not len(d) for d in datasets.values()):
        raise ValueError('Both training and validation sets must be nonempty')
    loaders = {s: DataLoader(d, batch_size=args.batch_size if s == 'train' else 4,
                             shuffle=s == 'train', num_workers=args.workers,
                             collate_fn=collate, pin_memory=True) for s, d in datasets.items()}
    model = BGeoNet(config['queries']).cuda()
    model.alpha = float(config['C'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    warmup, maximum = config['warmup_epochs'], config['max_epochs']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e:
        min(1., (e + 1) / warmup) * (.5 + .5 * math.cos(math.pi * max(0, e - warmup) / (maximum - warmup))))
    first, best, best_epoch = 1, -1., 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['config'] != config:
            raise ValueError('Resume configuration differs from the checkpoint')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint)
        first, best, best_epoch = checkpoint['epoch'] + 1, checkpoint['best'], checkpoint['best_epoch']
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))

    def forward(batch):
        image, target = (t.cuda(non_blocking=True) for t in batch[:2])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(image)
        with torch.autocast('cuda', enabled=False):
            loss, terms = losses(result, target, *batch[2:5], config['A'], config['B'])
        return result, target, loss, terms

    for epoch in range(first, maximum + 1):
        begin = time.perf_counter(); model.train(); sums = {}; count = 0
        for batch in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            result, target, loss, terms = forward(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            for name, value in dict(total=loss, **terms).items():
                sums[name] = sums.get(name, 0.) + float(value.detach()) * len(target)
            count += len(target)
        model.eval(); scores = []
        with torch.no_grad():
            for batch in loaders['val']:
                result, target, loss, _ = forward(batch)
                if not torch.isfinite(result['logits']).all():
                    raise FloatingPointError('Non-finite validation logits')
                pred = (result['logits'].float().sigmoid() > .5).float()
                dice = (2 * (pred * target).sum((1, 2, 3)) + 1e-7) / ((pred + target).sum((1, 2, 3)) + 1e-7)
                scores.extend(dice.cpu().tolist())
        score = float(np.mean(scores)); scheduler.step(); improved = score > best
        if improved:
            best, best_epoch = score, epoch
        row = dict(epoch=epoch, val_dice=score, best=best, best_epoch=best_epoch,
                   train_loss={k: v/count for k, v in sums.items()}, seconds=time.perf_counter()-begin)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          epoch=epoch, best=best, best_epoch=best_epoch, config=config, history_row=row, **rng_state())
        if improved:
            save(args.output / 'best.pth', checkpoint)
        save(args.output / 'last.pth', checkpoint)
        with (args.output / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if epoch - best_epoch >= config['patience']:
            break


if __name__ == '__main__':
    main()
