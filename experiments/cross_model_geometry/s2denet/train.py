"""Scratch geometry adaptation with full-state, audited epoch checkpoints."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False


def rng_state():
    state = np.random.get_state()
    return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),
                numpy_rng=[state[0], state[1].tolist(), state[2], state[3], state[4]])


def restore_rng(checkpoint):
    torch.set_rng_state(checkpoint['torch_rng'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    random.setstate(checkpoint['python_rng'])
    state = checkpoint['numpy_rng']
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))


def check_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            check_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            check_nested_equal(x, y)
    else:
        assert a == b


def dataset_metadata(root, metadata):
    from data_v4 import BlineV4
    meta = json.loads(Path(metadata).read_text())
    assert meta['queries'] == 16
    datasets = {split: BlineV4(root, split, meta, split == 'train') for split in ('train', 'val')}
    assert len(datasets['train']) == 27814 and len(datasets['val']) == 3691
    for split, dataset in datasets.items():
        names = {path.name for path in dataset.files}
        assert names == set(meta[split]), f'{split}: metadata names differ from dataset'
        mask_names = {path.name for path in (Path(root) / split / 'masks').glob('*.png')}
        assert names == mask_names, f'{split}: image/mask names differ'
    return meta, datasets


def data_fingerprint(root, datasets):
    """Content hashes, not only timestamps; test data is never read."""
    digest = hashlib.sha256()
    counts = {}
    for split, dataset in datasets.items():
        counts[split] = len(dataset)
        for image in dataset.files:
            for kind in ('images', 'masks'):
                rel = f'{split}/{kind}/{image.name}'
                value = sha256(Path(root) / rel)
                digest.update((rel + '\0' + value + '\n').encode())
    return dict(sha256=digest.hexdigest(), counts=counts, method='sorted relative path + file SHA256')


def source_hashes():
    return {path.relative_to(HERE).as_posix(): sha256(path)
            for path in sorted(HERE.rglob('*.py')) if '__pycache__' not in path.parts}


def package_versions():
    result = {'torch': str(torch.__version__), 'cuda': str(torch.version.cuda), 'numpy': str(np.__version__)}
    for name in ('scipy', 'einops', 'transformers', 'timm', 'segmentation-models-pytorch', 'mamba-ssm'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def make_model_optimizer(model_name):
    from models import build_model
    model = build_model(model_name).cuda()
    model.alpha = 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: min(1., (epoch + 1) / 10) *
        (.5 + .5 * math.cos(math.pi * max(0, epoch - 10) / 590)))
    return model, optimizer, scheduler


def native_edge_loss(probability, target):
    """Preserve S2DENet benchmark edge BCE+Dice, 3x3 morphological target."""
    import torch.nn.functional as F
    truth=target.float()
    band=(F.max_pool2d(truth,3,1,1)+F.max_pool2d(-truth,3,1,1)>0).float()
    p=probability.float()
    bce=F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),band)
    dice=(1-(2*(p*band).sum((1,2,3))+1)/(p.sum((1,2,3))+band.sum((1,2,3))+1)).mean()
    return bce+dice


def forward_loss(model, batch):
    from model_v41 import losses
    x, y = [item.cuda(non_blocking=True) for item in batch[:2]]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(x)
    assert result['logits'].shape == y.shape
    assert result['params'].shape == (x.shape[0], 16, 6)
    with torch.autocast('cuda', enabled=False):
        loss, parts = losses(result, y, *batch[2:5], .5, .25)
        geometry_weight = getattr(model, 'geometry_aux_weight', .25)
        loss = loss + (geometry_weight - .25) * (.3 * parts['shape'] + .2 * parts['params'] + .1 * parts['object'])
        parts['geometry_aux_weight'] = geometry_weight
    edge_aux = native_edge_loss(result['native_edge'], y)
    loss = loss + .2 * edge_aux
    parts['native_edge_aux'] = edge_aux
    if float(loss.detach()) > 1000:
        raise FloatingPointError('Divergence guard: loss exceeds 1000')
    if not torch.isfinite(loss).item():
        raise FloatingPointError('Nonfinite loss')
    for key in ('logits', 'pixel', 'params', 'objects', 'geometry'):
        if not torch.isfinite(result[key]).all().item():
            raise FloatingPointError('Nonfinite output: ' + key)
    return result, loss, parts, y


def training_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    result, loss, parts, y = forward_loss(model, batch)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if float(grad_norm) > 1e6:
        raise FloatingPointError('Divergence guard: unclipped grad norm exceeds 1e6')
    if not torch.isfinite(grad_norm).item():
        raise FloatingPointError('Nonfinite gradient norm')
    optimizer.step()
    return float(loss.detach()), float(grad_norm), {key: float(value) for key, value in parts.items()}


def worst_batch(dataset, metadata, batch_size):
    """Use real train images with the largest valid instance counts."""
    from data_v4 import collate
    assert 0 < batch_size <= len(dataset), 'Candidate exceeds available unique training images'
    ranked = sorted(range(len(dataset)),
                    key=lambda i: (-len(metadata['train'][dataset.files[i].name]['targets']),
                                   dataset.files[i].name))
    dataset.augment = False
    indices = ranked[:batch_size]
    batch = collate([dataset[index] for index in indices])
    return batch, dict(filenames=batch[5], instances_per_image=[len(item) for item in batch[2]])


def checkpoint(model, optimizer, scheduler, epoch, best, best_epoch, config, history):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                epoch=epoch, best=best, best_epoch=best_epoch, config=config, history=history,
                **rng_state())


def restore(checkpoint_value, model, optimizer, scheduler, config):
    assert checkpoint_value['config'] == config, 'Resume configuration/data/source hashes differ'
    model.load_state_dict(checkpoint_value['model'], strict=True)
    optimizer.load_state_dict(checkpoint_value['optimizer'])
    scheduler.load_state_dict(checkpoint_value['scheduler'])
    restore_rng(checkpoint_value)


def validate_resume_checkpoint(value, config, role, cuda_devices=None):
    required = {'model', 'optimizer', 'scheduler', 'epoch', 'best', 'best_epoch', 'config',
                'history', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng'}
    assert required <= value.keys(), f'{role}: incomplete checkpoint'
    assert value['config'] == config, f'{role}: configuration/data/source hashes differ'
    epoch = value['epoch']
    assert isinstance(epoch, int) and 1 <= epoch <= 600, f'{role}: invalid epoch'
    history = value['history']
    assert isinstance(history, list) and len(history) == epoch, f'{role}: incomplete history'
    best, best_epoch = -1., 0
    for index, row in enumerate(history, 1):
        assert row['epoch'] == index, f'{role}: non-contiguous history'
        score = row['val_dice']
        assert math.isfinite(score) and 0 <= score <= 1.0000001, f'{role}: invalid validation Dice'
        if score > best:
            best, best_epoch = score, index
        assert row['best'] == best and row['best_epoch'] == best_epoch, f'{role}: invalid best trajectory'
        for name in ('train_loss', 'val_loss', 'seconds'):
            if name in row:
                assert math.isfinite(row[name]), f'{role}: nonfinite history {name}'
    assert value['best'] == best and value['best_epoch'] == best_epoch, f'{role}: best/history disagree'
    if role == 'best':
        assert epoch == best_epoch, 'best.pth is not a best-epoch checkpoint'
    assert isinstance(value['model'], dict) and value['model'], f'{role}: missing model'
    for name, tensor in value['model'].items():
        assert isinstance(tensor, torch.Tensor) and torch.isfinite(tensor).all().item(), f'{role}: invalid model tensor {name}'
    optimizer = value['optimizer']
    assert optimizer.get('state') and optimizer.get('param_groups'), f'{role}: missing initialized optimizer'
    ids = [item for group in optimizer['param_groups'] for item in group['params']]
    assert len(set(ids)) == len(ids) and set(optimizer['state']) <= set(ids), f'{role}: invalid optimizer IDs'
    for state in optimizer['state'].values():
        assert {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys(), f'{role}: incomplete AdamW state'
        for tensor in state.values():
            if isinstance(tensor, torch.Tensor):
                assert torch.isfinite(tensor).all().item(), f'{role}: nonfinite optimizer state'
    scheduler = value['scheduler']
    assert scheduler['last_epoch'] == epoch, f'{role}: scheduler epoch differs'
    assert scheduler['_last_lr'] == [group['lr'] for group in optimizer['param_groups']], f'{role}: scheduler/optimizer LR differs'
    state = value['torch_rng']
    assert isinstance(state, torch.Tensor) and state.device.type == 'cpu' and state.dtype == torch.uint8 and state.ndim == 1
    torch.Generator(device='cpu').set_state(state)
    cuda_states = value['cuda_rng']
    assert isinstance(cuda_states, (list, tuple)) and cuda_states, f'{role}: missing CUDA RNG'
    if cuda_devices is not None:
        assert len(cuda_states) == cuda_devices, f'{role}: CUDA RNG device count differs'
    for state in cuda_states:
        assert isinstance(state, torch.Tensor) and state.device.type == 'cpu' and state.dtype == torch.uint8 and state.ndim == 1 and state.numel() > 0
    random.Random(0).setstate(value['python_rng'])
    state = value['numpy_rng']
    assert len(state) == 5 and all(isinstance(item, int) and 0 <= item <= 4294967295 for item in state[1])
    np.random.RandomState(0).set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))


def select_resume_checkpoint(output, config, cuda_devices=None):
    """Recover the best-before-last commit window without discarding either artifact."""
    output = Path(output)
    candidates = {}
    for role in ('best', 'last'):
        path = output / (role + '.pth')
        if path.exists():
            value = torch.load(path, map_location='cpu', weights_only=True)
            validate_resume_checkpoint(value, config, role, cuda_devices)
            candidates[role] = value
    assert candidates, 'Resume requires at least one complete best.pth or last.pth'
    chosen_role = max(candidates, key=lambda name: (candidates[name]['epoch'], name == 'last'))
    selected = candidates[chosen_role]
    for role, value in candidates.items():
        assert value['history'] == selected['history'][:value['epoch']], 'Checkpoint histories are not prefixes of the selected checkpoint'
        assert value['model'].keys() == selected['model'].keys(), 'Checkpoint model keys differ'
        for name, tensor in value['model'].items():
            other = selected['model'][name]
            assert tensor.shape == other.shape and tensor.dtype == other.dtype, 'Checkpoint tensor schema differs: ' + name
        if value['epoch'] == selected['epoch']:
            check_nested_equal(value, selected)
    if 'best' in candidates:
        assert candidates['best']['epoch'] == selected['best_epoch'], 'Stored best is inconsistent with latest complete checkpoint'
    log = output / 'history.jsonl'
    old_history = [json.loads(row) for row in log.read_text().splitlines()] if log.exists() else []
    assert len(old_history) <= selected['epoch'] and old_history == selected['history'][:len(old_history)], 'History log is not a prefix of the selected complete checkpoint'
    evidence = output / ('resume_evidence_' + str(time.time_ns()))
    evidence.mkdir()
    files = {}
    for name in ('best.pth', 'last.pth', 'history.jsonl', 'config.json'):
        source = output / name
        if source.exists():
            destination = evidence / name
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            files[name] = dict(sha256=sha256(destination), bytes=destination.stat().st_size)
    best_missing = 'best' not in candidates
    # A last checkpoint taken on its best epoch is itself the authentic best state.
    if best_missing and selected['epoch'] == selected['best_epoch']:
        source = output / (chosen_role + '.pth')
        try:
            os.link(source, output / 'best.pth')
        except OSError:
            shutil.copy2(source, output / 'best.pth')
        best_missing = False
    audit = dict(selected=chosen_role + '.pth', selected_epoch=selected['epoch'],
                 candidate_epochs={role: value['epoch'] for role, value in candidates.items()},
                 best_epoch=selected['best_epoch'], historical_best_checkpoint_missing=best_missing,
                 history_prefixes_verified=True, full_states_and_all_rng_verified=True,
                 evidence_directory=str(evidence), files=files, time=time.time())
    atomic_json(evidence / 'audit.json', audit)
    atomic_json(output / 'resume_audit.json', audit)
    return selected, audit


def status(output, **fields):
    fields.update(updated_at=time.time(), pid=os.getpid())
    atomic_json(output / 'status.json', fields)
    print(json.dumps(fields, allow_nan=False), flush=True)


def smoke(args, config, model, optimizer, scheduler, datasets, metadata):
    from data_v4 import collate
    output = Path(args.output)
    batch, selected = worst_batch(datasets['train'], metadata, args.batch_size)
    model.train()
    first_loss, _, _ = training_step(model, optimizer, batch)
    scheduler.step()
    model.eval()
    with torch.no_grad():
        result, eval_loss, eval_parts, eval_target = forward_loss(model, batch)
        expected = result['logits'].detach().cpu()
    del result, eval_loss, eval_parts, eval_target
    atomic_save(output / 'smoke_checkpoint.pth',
                checkpoint(model, optimizer, scheduler, 1, 0., 1, config, []))
    # Free the first model before constructing the second: probe selected a true maximum batch.
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    saved = torch.load(output / 'smoke_checkpoint.pth', map_location='cpu', weights_only=True)
    model, optimizer, scheduler = make_model_optimizer(args.model)
    restore(saved, model, optimizer, scheduler, config)
    check_nested_equal(model.state_dict(), saved['model'])
    check_nested_equal(optimizer.state_dict(), saved['optimizer'])
    check_nested_equal(scheduler.state_dict(), saved['scheduler'])
    check_nested_equal(rng_state(), {key: saved[key] for key in rng_state()})
    model.eval()
    with torch.no_grad():
        actual, eval_loss, eval_parts, eval_target = forward_loss(model, batch)
        torch.testing.assert_close(actual['logits'].cpu(), expected, rtol=1e-4, atol=1e-5)
    del actual, eval_loss, eval_parts, eval_target, expected, saved
    model.train()
    resumed_loss, norm, _ = training_step(model, optimizer, batch)
    gradients = dict(model.named_parameters())
    gradient_audit = {}
    for prefix in ('backbone.', 'deep_adapter.', 'feature_adapter.', 'shape.', 'object.', 'refine.'):
        values = [parameter.grad for name, parameter in gradients.items()
                  if name.startswith(prefix) and parameter.grad is not None]
        passed = bool(values) and all(torch.isfinite(value).all().item() for value in values)
        passed = passed and any(torch.count_nonzero(value).item() > 0 for value in values)
        assert passed, 'Missing, zero, or nonfinite gradients: ' + prefix
        gradient_audit[prefix] = True
    del gradients
    model.eval()
    validation_batch = collate([datasets['val'][index] for index in range(min(4,args.batch_size))])
    with torch.no_grad():
        validation_result, validation_loss, validation_parts, validation_target = forward_loss(model, validation_batch)
        validation_loss_value = float(validation_loss)
    del validation_result, validation_loss, validation_parts, validation_target
    report = dict(passed=True, strict_model_reload=True, full_optimizer_scheduler_rng_restore=True,
                  eval_logits_close=True, finite_resumed_update=True, batch=args.batch_size,
                  first_loss=first_loss, resumed_loss=resumed_loss, resumed_grad_norm=norm,
                  validation_images=len(validation_batch[0]), validation_loss=validation_loss_value,
                  gradients=gradient_audit,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), selection=selected)
    atomic_json(output / 'smoke_passed.json', report)
    status(output, status='SMOKE_PASSED', report=report)


def train(args):
    from data_v4 import collate
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'training_ready.json').exists() or (output / 'smoke_passed.json').exists():
        raise RuntimeError('Completed output preserved; refusing another run')
    if (output / 'config.json').exists() and not args.resume:
        raise RuntimeError('Existing output preserved; use explicit --resume for a complete epoch checkpoint')
    if args.resume and not any((output / name).exists() for name in ('best.pth', 'last.pth')):
        raise RuntimeError('--resume requires a complete best.pth or last.pth')
    if args.mode == 'smoke' and args.resume:
        raise RuntimeError('Smoke uses an isolated fresh output')
    seed_everything()
    status(output, status='VERIFYING_TRAIN_VAL_DATA', model=args.model)
    metadata, datasets = dataset_metadata(args.root, args.metadata)
    fingerprint = data_fingerprint(args.root, datasets)
    model, optimizer, scheduler = make_model_optimizer(args.model)
    config = dict(model=args.model, variant='geometry', mode=args.mode, root=str(Path(args.root).resolve()),
                  metadata=str(Path(args.metadata).resolve()), metadata_sha256=sha256(args.metadata),
                  data_fingerprint=fingerprint, source_sha256=source_hashes(), packages=package_versions(),
                  seed=42, initialization='random scratch; no probe/model checkpoint warm-start',
                  experiment='s2denet_geom_warmup10_seed42', native_edge_aux_weight=.2, native_edge_target_kernel=3, image_mean=.100638, image_std=.145868,
                  geometry_aux_schedule='0.25 * min(1, epoch/10); pixel auxiliary stays 0.125',
                  A=.5, B=.25, C=1., lambda_cons=0., batch=args.batch_size, batch_size=args.batch_size,
                  gradient_accumulation=1, validation_batch=min(4,args.batch_size), lr=1e-4, weight_decay=1e-4,
                  warmup=10, max_epochs=600, patience=10, precision='BF16 forward; FP32 geometry/loss',
                  threshold='>0.5', selection='maximum validation image-mean Dice', queries=16,
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  num_workers=args.workers, test_used=False)
    start, best, best_epoch, history = 1, -1., 0, []
    if args.resume:
        saved, resume_audit = select_resume_checkpoint(output, config, torch.cuda.device_count())
        restore(saved, model, optimizer, scheduler, config)
        start, best, best_epoch = saved['epoch'] + 1, saved['best'], saved['best_epoch']
        history = saved['history']
        print(json.dumps(dict(status='RESUMED_COMPLETE_EPOCH', **resume_audit)), flush=True)
        del saved
    atomic_json(output / 'config.json', config)
    if args.mode == 'smoke':
        # Ownership must leave this scope so smoke can free the original CUDA model completely.
        objects = [model, optimizer, scheduler]
        del model, optimizer, scheduler
        smoke(args, config, objects.pop(0), objects.pop(0), objects.pop(0), datasets, metadata)
        return
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size if split == 'train' else min(4,args.batch_size),
                shuffle=split == 'train', num_workers=args.workers, pin_memory=True,
                persistent_workers=False, collate_fn=collate, drop_last=False)
               for split, dataset in datasets.items()}
    for epoch in range(start, 601):
        if epoch - 1 - best_epoch >= 10:
            break
        model.train()
        model.geometry_aux_weight = .25 * min(1., epoch / 10.)
        torch.cuda.reset_peak_memory_stats()
        began = time.time()
        total, count = 0., 0
        status(output, status='TRAINING', model=args.model, epoch=epoch, step=0,
               steps=len(loaders['train']), best=best, best_epoch=best_epoch, batch=args.batch_size)
        for step, batch in enumerate(loaders['train'], 1):
            try:
                loss, grad_norm, parts = training_step(model, optimizer, batch)
            except Exception as exc:
                failure = dict(epoch=epoch, step=step, filenames=batch[5],
                               instances_per_image=[len(item) for item in batch[2]], error=repr(exc),
                               state='PARTIAL_EPOCH_DIAGNOSTIC_ONLY_NOT_A_RESUME_CHECKPOINT')
                atomic_json(output / f'failure_e{epoch}_step{step}.json', failure)
                try:
                    atomic_save(output / f'failure_e{epoch}_step{step}.pth',
                                dict(failure=failure, **checkpoint(model, optimizer, scheduler,
                                     epoch-1, best, best_epoch, config, history)))
                except Exception as save_error:
                    atomic_json(output / f'failure_e{epoch}_step{step}_save_error.json',
                                dict(error=repr(save_error)))
                raise
            total += loss * len(batch[0])
            count += len(batch[0])
            if step == 1 or step % 50 == 0 or step == len(loaders['train']):
                status(output, status='TRAINING', model=args.model, epoch=epoch, step=step,
                       steps=len(loaders['train']), loss=loss, loss_parts=parts, grad_norm=grad_norm,
                       best=best, best_epoch=best_epoch, elapsed_seconds=time.time()-began,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
        model.eval()
        scores, val_loss, val_count = [], 0., 0
        with torch.no_grad():
            for step, batch in enumerate(loaders['val'], 1):
                result, loss, _, target = forward_loss(model, batch)
                prediction = (result['logits'].float().sigmoid() > .5).float()
                values = (2*(prediction*target).sum((1,2,3))+1e-7)/((prediction+target).sum((1,2,3))+1e-7)
                scores.extend(values.cpu().tolist())
                val_loss += float(loss)*len(target)
                val_count += len(target)
                if step == 1 or step % 100 == 0:
                    status(output, status='VALIDATING', model=args.model, epoch=epoch, step=step,
                           steps=len(loaders['val']), best=best, best_epoch=best_epoch)
                del result, prediction, target, loss
        assert count == 27814 and val_count == 3691
        score = float(np.mean(scores))
        assert math.isfinite(score)
        scheduler.step()
        improved = score > best
        if improved:
            best, best_epoch = score, epoch
        row = dict(epoch=epoch, train_loss=total/count, val_loss=val_loss/val_count,
                   val_dice=score, best=best, best_epoch=best_epoch, train_count=count,
                   val_count=val_count, seconds=time.time()-began, lr=optimizer.param_groups[0]['lr'],
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
        history.append(row)
        saved = checkpoint(model, optimizer, scheduler, epoch, best, best_epoch, config, history)
        if improved:
            atomic_save(output / 'best.pth', saved)
        atomic_save(output / 'last.pth', saved)
        del saved
        temp = output / 'history.jsonl.tmp'
        temp.write_text(''.join(json.dumps(item, allow_nan=False)+'\n' for item in history))
        temp.replace(output / 'history.jsonl')
        status(output, status='EPOCH_COMPLETE', model=args.model, **row)
        if epoch - best_epoch >= 10:
            break
    if not (output / 'best.pth').exists():
        raise RuntimeError('Training ended without the historical best checkpoint; last is not a substitute')
    best_saved = torch.load(output / 'best.pth', map_location='cpu', weights_only=True)
    validate_resume_checkpoint(best_saved, config, 'best', torch.cuda.device_count())
    assert best_saved['epoch'] == best_epoch and best_saved['best'] == best
    del best_saved
    ready = dict(status='TRAINED_TEST_PENDING', model=args.model, best_epoch=best_epoch,
                 best_val_dice=best, final_epoch=history[-1]['epoch'], test_executed=False,
                 stopped_by_patience=history[-1]['epoch']-best_epoch >= 10)
    atomic_json(output / 'training_ready.json', ready)
    status(output, **ready)


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'root', 'metadata', 'output'):
        parser.add_argument('--'+name, required=True, **({'choices': ('s2denet',)} if name == 'model' else {}))
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--mode', choices=('train', 'smoke'), default='train')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert args.batch_size > 0 and args.workers >= 0
    try:
        train(args)
    except Exception as exc:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        error = dict(status='FAILED_PRESERVED', error=repr(exc), traceback=traceback.format_exc(),
                     model=args.model, mode=args.mode, time=time.time(), pid=os.getpid())
        atomic_json(output / ('failure_'+str(time.time_ns())+'.json'), error)
        # Preserve an existing completed status on accidental duplicate invocations.
        if not (output / 'training_ready.json').exists() and not (output / 'smoke_passed.json').exists():
            atomic_json(output / 'status.json', error)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
