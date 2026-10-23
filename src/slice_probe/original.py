"""Import existing original sparse-study heads; no fitting or label regeneration."""
from pathlib import Path

import numpy as np
import torch

from .features import prompt
from .io import load_checkpoint, read_json, require_empty, save_checkpoint, sha256, write_json
from .model import predict_features
from .registry import publish, version_name


def checked_tensor(path, checksum=None):
    path = Path(path)
    checksum = checksum or read_json(path.with_suffix('.json'))['sha256']
    if sha256(path) != checksum:
        raise ValueError(f'Original artifact checksum mismatch: {path}')
    return torch.load(path, weights_only=True, map_location='cpu')


def import_original(study, model, registry, out, *, prefix=None, samples=512):
    """Wrap original compressors/Ridge weights after checking saved predictions.

    All model assets, selected compressors, fit sidecars, test-cache files and
    question snapshots are checked. Validation uses evenly spaced saved test
    rows, not refitting. Every original seed is published independently.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError('samples must be positive')
    study, model, registry, out = map(lambda p: Path(p).resolve(), (study, model, registry, out))
    protocol = read_json(study / 'protocol.json')
    prefix = prefix or f'original-sparse{protocol["width"]}'
    source = Path(protocol['source']).resolve()
    if out.is_relative_to(study) or out.is_relative_to(source):
        raise ValueError('Import output must be separate from the original studies')
    manifest = read_json(source / 'manifest.json')
    if sha256(source / 'manifest.json') != protocol['source_manifest']:
        raise ValueError('Original source manifest differs from the compressor protocol')
    if sha256(source / 'questions.json') != manifest['questions_sha256']:
        raise ValueError('Original question snapshot changed')
    questions = read_json(source / 'questions.json')
    if not questions or any(q['prompt'] != prompt(q['question']) for q in questions):
        raise ValueError('Original prompt differs; refusing to substitute the current prompt')
    if ('config.json' not in manifest['model_hashes']
            or not any(name.endswith('.safetensors') for name in manifest['model_hashes'])):
        raise ValueError('Original manifest must pin model configuration and weights')
    for name, digest in manifest['model_hashes'].items():
        path = (model / name).resolve()
        if not path.is_relative_to(model) or sha256(path) != digest:
            raise ValueError(f'Original model/tokenizer asset mismatch: {name}')
    config = read_json(model / 'config.json')
    # Original jspace indices are zero-based; SLICE's capture contract is one-based.
    layers = [index + 1 for index in manifest['layers']]
    if (len(set(layers)) != len(layers) or not layers
            or not 1 <= min(layers) <= max(layers) <= config['num_hidden_layers']):
        raise ValueError('Invalid original block list')
    packed = source / 'packed/rms_hidden_concat'
    complete_path = packed / 'complete.json'
    if sha256(complete_path) != protocol['packed']['rms_hidden_concat']:
        raise ValueError('Original packed-feature manifest changed')
    complete = read_json(complete_path)
    for name in ('test-x.npy', 'test-labels.pt'):
        if sha256(packed / name) != complete['files'][name]:
            raise ValueError(f'Original test cache changed: {name}')
    labels = checked_tensor(packed / 'test-labels.pt', complete['files']['test-labels.pt'])
    x = np.load(packed / 'test-x.npy', mmap_mode='r')
    if x.shape != (complete['counts']['test'], config['hidden_size'] * len(layers)) or not len(x):
        raise ValueError('Original cache dimensions differ from the encoder')
    indices = np.unique(np.linspace(0, len(x) - 1, min(samples, len(x)), dtype=np.int64))
    screening = read_json(study / 'screen.json')
    versions = [version_name(f'{prefix}-seed{seed}') for seed in protocol['seeds']]
    if any((registry / version).exists() for version in versions):
        raise FileExistsError('An original artifact version is already published; load it or use another prefix')
    out = require_empty(out)
    records, checks = [], {}
    with np.load(study / 'predictions.npz', allow_pickle=False) as saved:
        if (not np.array_equal(saved['y'], labels['y'].numpy())
                or not np.array_equal(saved['question'], labels['question'].numpy())):
            raise ValueError('Saved original predictions and test-cache labels are unaligned')
        for seed in protocol['seeds']:
            key = f'sparse-{seed}'
            item = screening['selected'][key]
            path = (study / item['path']).resolve()
            if not path.is_relative_to(study):
                raise ValueError('Original checkpoint path escapes study')
            compressor = checked_tensor(path, item['sha256'])
            head_path = study / f'{key}-ridge-fit.pt'
            head = checked_tensor(head_path)
            if compressor['seed'] != seed or compressor['width'] != protocol['width']:
                raise ValueError('Original selected compressor identity mismatch')
            record = dict(format='slice-probe-v1', input_dim=x.shape[1], width=protocol['width'],
                state=compressor['state'], mean=compressor['mean'], scale=compressor['scale'],
                weights=head['weights'], model=dict(id=str(model), revision=None,
                    hashes=manifest['model_hashes']), layers=layers, rms_eps=manifest['rms_eps'],
                kind='sparse', seed=seed, ridge_states='all-training-tokens',
                head=dict(method='ridge'), fit_checks=head['checks'])
            actual = predict_features(record, x[indices], 'cpu')
            expected = saved[f'{key}/ridge'][indices]
            matches = bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-6))
            checks[key] = dict(matches=matches, rows=len(indices), total_rows=len(x),
                max_absolute_error=float(np.max(np.abs(actual - expected))),
                compressor_sha256=item['sha256'], ridge_sha256=sha256(head_path))
            records.append(record)
    write_json(out / 'validation.json', dict(heads=checks, rtol=1e-5, atol=1e-6,
        scope='Saved-feature prediction check on evenly spaced original test rows. No fitting or generation.'))
    if not all(row['matches'] for row in checks.values()):
        raise ValueError('Original prediction check failed; see validation.json. Nothing was published')
    paths = []
    for version, record in zip(versions, records):
        checkpoint = out / f'seed{record["seed"]}.pt'
        save_checkpoint(checkpoint, record)
        # Explicit equality checks guard the adapter's most important contract.
        wrapped = load_checkpoint(checkpoint)
        if not torch.equal(wrapped['weights'], record['weights']):
            raise ValueError('Original Ridge weights changed during serialization')
        destination = publish(checkpoint, registry, version, evaluation=out / 'validation.json',
            metadata=dict(origin='existing original SLICE study; no retraining', study=str(study),
                source=str(source), protocol_sha256=sha256(study / 'protocol.json'),
                source_manifest_sha256=protocol['source_manifest'],
                predictions_sha256=sha256(study / 'predictions.npz'),
                question_snapshot_sha256=manifest['questions_sha256'],
                validation=checks[f'sparse-{record["seed"]}'],
                encoder_identity='Original local assets pinned by SHA256; upstream revision not inferred'))
        paths.append(str(destination))
    result = dict(artifacts=paths, validation=str(out / 'validation.json'), retrained=False)
    write_json(out / 'result.json', result)
    return result
