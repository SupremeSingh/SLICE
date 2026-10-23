"""Train the compressor, freeze it, and fit an independently solved Ridge head."""
from pathlib import Path
import time

import numpy as np
import torch

from .collection import load_split, iter_answers, retained
from .features import load_model, capture, concatenate
from .io import read_json, write_json, require_empty, save_checkpoint, load_checkpoint, sha256
from .metrics import metrics
from .model import Bottleneck, objective, predict_features


def normalization(x, batch_size=4096):
    """Training mean and one isotropic scale, without copying the complete cache."""
    count, dimension = x.shape
    if not count:
        raise ValueError('Empty training features')
    total = np.zeros(dimension, dtype=np.float64)
    squares = 0.0
    for start in range(0, count, batch_size):
        batch = np.asarray(x[start:start + batch_size], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError('Nonfinite training features')
        total += batch.sum(0)
        squares += np.square(batch).sum()
    mean = total / count
    variance = max(0.0, squares / (count * dimension) - np.square(mean).mean())
    return (torch.tensor(mean, dtype=torch.float32),
            torch.full((dimension,), max(variance ** .5, 1e-8)))


@torch.inference_mode()
def encode(model, x, mean, scale, device, batch_size=1024):
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        v = torch.as_tensor(np.array(x[start:start + batch_size], copy=True),
                            dtype=torch.float32, device=device)
        chunks.append(model.features((v - mean.to(device)) / scale.to(device)).cpu())
    if not chunks:
        raise ValueError('No prefixes')
    return torch.cat(chunks)


def add_statistics(a, b, z, y):
    phi = torch.cat((z.cpu().double(), torch.ones(len(z), 1, dtype=torch.float64)), dim=1)
    target = torch.tensor(np.array(y, copy=True), dtype=torch.float64).flatten()
    if target.shape != (len(phi),) or not torch.isin(target, torch.tensor([0., 1.])).all():
        raise ValueError('One binary outcome per prefix required')
    a += phi.T @ phi
    b += phi.T @ target


def solve_ridge(a, b, epsilon):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Positive finite Ridge epsilon required')
    system = a + epsilon * torch.eye(len(b), dtype=torch.float64)
    w = torch.linalg.solve(system, b)
    residual = torch.linalg.vector_norm(system @ w - b)
    denominator = (torch.linalg.matrix_norm(system) * torch.linalg.vector_norm(w)
                   + torch.linalg.vector_norm(b)).clamp_min(1e-30)
    relative = float(residual / denominator)
    if not torch.isfinite(w).all() or relative > 1e-8:
        raise ValueError('Ridge solve failed its numerical check')
    return w, dict(relative_residual=relative, epsilon=epsilon, intercept_penalized=True)


def fit_ridge(z, y, epsilon):
    a = torch.zeros(z.shape[1] + 1, z.shape[1] + 1, dtype=torch.float64)
    b = torch.zeros(z.shape[1] + 1, dtype=torch.float64)
    for start in range(0,len(y),2048):
        add_statistics(a,b,z[start:start+2048],y[start:start+2048])
    return solve_ridge(a,b,epsilon)


def all_token_ridge(record, manifest, answers, device, epsilon):
    """Recompute complete training prefixes; keep only compressed statistics."""
    if sha256(Path(answers) / 'manifest.json') != manifest['collection_manifest_sha256']:
        raise ValueError('Answer collection differs from the feature cache')
    identity = manifest['model']
    backbone, _, _ = load_model(identity['id'], identity['revision'], device,
                                hashes=identity['hashes'])
    model = Bottleneck(record['input_dim'], record['width']).to(device)
    model.load_state_dict(record['state'])
    model.eval().requires_grad_(False)
    size = record['width'] + 1
    a, b = torch.zeros(size, size, dtype=torch.float64), torch.zeros(size, dtype=torch.float64)
    count = 0
    for _, question, batch, path in iter_answers(answers, 'train'):
        if sha256(path) != manifest['answers_sha256'].get(path.name):
            raise ValueError('A training answer changed since extraction')
        for row in batch['responses']:
            if not retained(row):
                continue
            length, prompt_length = len(row['token_ids']), len(batch['prompt_ids'])
            ids = torch.tensor([batch['prompt_ids'] + row['token_ids']], device=device)
            indices = list(range(prompt_length - 1, prompt_length + length - 1))
            blocks = capture(backbone, ids, indices, manifest['layers'])
            x = concatenate(blocks, manifest['layers'], manifest['rms_eps'])
            z = encode(model, x, record['mean'], record['scale'], device)
            add_statistics(a, b, z, np.full(length, row['score']))
            count += length
        print(f'Ridge states: {count}; question: {question["id"]}', flush=True)
    if not count:
        raise ValueError('No retained training-token states')
    weights, checks = solve_ridge(a, b, epsilon)
    from .collection import fitting_population
    return weights, dict(checks, training_prefixes=count, training_population=fitting_population(answers))


def update_validation(record, features, device='cpu'):
    """Bind validation to the exact feature transformation and final linear head."""
    from .model import require_correctness_head
    require_correctness_head(record)
    import hashlib
    from .registry import feature_identity, canonical_hash
    val, _ = load_split(features, 'val')
    if 'validation' in record and 'selection_validation_sampled_head' not in record:
        record['selection_validation_sampled_head'] = record['validation'].copy()
        record['selection_validation_sampled_head_binding'] = record.get('validation_binding')
    digest = hashlib.sha256(feature_identity(record).encode())
    weights = record['weights'].detach().cpu().contiguous()
    digest.update(canonical_hash([str(weights.dtype), list(weights.shape),
                                 record.get('head', {'method': 'ridge'})]).encode())
    digest.update(weights.numpy().tobytes())
    record['validation'] = metrics(val['y'], predict_features(record, val['x'], device))
    record['validation_binding'] = dict(head_id=digest.hexdigest(),
        feature_manifest_sha256=sha256(Path(features) / 'manifest.json'),
        split='val', population='saved sampled validation prefixes', raw_values=True)
    return record


def fit(features, out, width=768, kinds=('plain', 'sparse'), seeds=(42, 43, 44),
        learning_rates=(.0003, .001), epochs=30, batch_size=2048,
        epsilon=.01, device='cpu', answers=None):
    if min(width, epochs, batch_size) < 1 or not seeds or not learning_rates:
        raise ValueError('Positive dimensions, fitting budgets, seeds and learning rates required')
    if any(kind not in ('plain', 'sparse') for kind in kinds) or not kinds:
        raise ValueError('Choose plain and/or sparse')
    if len(set(kinds)) != len(kinds) or len(set(seeds)) != len(seeds):
        raise ValueError('Kinds and seeds must be unique')
    if any(not np.isfinite(lr) or lr <= 0 for lr in learning_rates):
        raise ValueError('Learning rates must be finite and positive')
    train, manifest = load_split(features, 'train')
    val, _ = load_split(features, 'val')
    if set(np.unique(train['question'])) & set(np.unique(val['question'])):
        raise ValueError('Training and validation questions overlap')
    if len(np.unique(train['y'])) != 2 or len(np.unique(val['y'])) != 2:
        raise ValueError('Both outcomes are required in training and validation')
    mean, scale = normalization(train['x'])
    out = require_empty(out)
    selections, trials = {}, []
    for kind in kinds:
        penalty = .01 if kind == 'sparse' else 0.0
        for seed in seeds:
            candidates = []
            for lr in learning_rates:
                started = time.perf_counter()
                torch.manual_seed(seed)
                model = Bottleneck(manifest['dimension'], width).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
                generator = torch.Generator().manual_seed(seed)
                best, state, curve = float('inf'), None, []
                for epoch in range(epochs):
                    model.train()
                    for indices in torch.randperm(len(train['y']), generator=generator).split(batch_size):
                        x = torch.tensor(np.array(train['x'][indices.numpy()], copy=True), device=device)
                        y = torch.tensor(np.array(train['y'][indices.numpy()], copy=True), device=device)
                        loss = objective(model, (x - mean.to(device)) / scale.to(device), y, penalty)
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite compressor loss')
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    zval = encode(model, val['x'], mean, scale, device)
                    with torch.inference_mode():
                        p = model.readout(zval.to(device)).flatten().sigmoid().cpu().numpy()
                    brier = metrics(val['y'], p)['brier']
                    curve.append(brier)
                    if brier < best:
                        best = brier
                        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    print(f'{kind} seed={seed} lr={lr} epoch={epoch+1} val_brier={brier:.5f}', flush=True)
                model.load_state_dict(state)
                ztrain = encode(model, train['x'], mean, scale, device)
                zval = encode(model, val['x'], mean, scale, device)
                weights, checks = fit_ridge(ztrain, train['y'], epsilon)
                record = dict(format='slice-probe-v1', input_dim=manifest['dimension'], width=width,
                    state=state, mean=mean, scale=scale, weights=weights,
                    model=manifest['model'], layers=manifest['layers'], rms_eps=manifest['rms_eps'],
                    layer_selection=manifest.get('layer_selection'),
                    kind=kind, seed=seed, penalty=penalty, learning_rate=lr, validation_curve=curve,
                    train_prior=float(train['y'].mean()), ridge_states='sampled',
                    training_population=dict(manifest.get('training_population', {}),
                        states='eight sampled prefixes per verifiable answer'),
                    feature_manifest_sha256=sha256(Path(features) / 'manifest.json'),
                    fit_checks=checks, seconds=time.perf_counter() - started)
                update_validation(record, features, device)
                name = f'{kind}-seed{seed}-lr{lr}.pt'
                save_checkpoint(out / name, record)
                candidates.append((record['validation']['brier'], name))
                trials.append(dict(path=name, validation_brier=record['validation']['brier'],
                                   kind=kind, seed=seed, seconds=record['seconds']))
            _, chosen = min(candidates)
            record = load_checkpoint(out / chosen)
            # Checkpoint selection is locked before complete-token readout fitting.
            if answers is not None:
                started = time.perf_counter()
                record['weights'], record['fit_checks'] = all_token_ridge(record, manifest, answers, device, epsilon)
                record['ridge_states'] = 'all-training-tokens'
                record['all_token_fit_seconds'] = time.perf_counter() - started
            update_validation(record, features, device)
            if 'training_population' in record['fit_checks']:
                record['training_population'] = record['fit_checks']['training_population']
            final = out / f'{kind}-seed{seed}.pt'
            save_checkpoint(final, record)
            selections[f'{kind}-{seed}'] = dict(path=final.name, sha256=sha256(final),
                selected_trial=chosen, validation_brier_sampled_readout=min(candidates)[0])
    write_json(out / 'selection.json', dict(format='slice-selection-v1', selected=selections, trials=trials,
        feature_manifest_sha256=sha256(Path(features) / 'manifest.json'),
        scope='Validation-only selection; no test loading. Epoch by temporary-classifier Brier; learning rate by sampled-prefix Ridge Brier.'))
    return selections
