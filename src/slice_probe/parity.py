"""Compare managed artifacts with standalone extraction and training on saved data."""
from pathlib import Path

import numpy as np
import torch

from .collection import extract, load_split
from .comparison import environment
from .io import load_checkpoint, read_json, require_empty, sha256, write_json
from .model import predict_features
from .training import fit
from .worker import completed_files


def compare_arrays(left, right, *, rtol=1e-5, atol=1e-6):
    """Bound memory while reporting numerical, rather than file-byte, parity."""
    if isinstance(left, torch.Tensor):
        left = left.detach().cpu().numpy()
    if isinstance(right, torch.Tensor):
        right = right.detach().cpu().numpy()
    if left.shape != right.shape or left.dtype != right.dtype:
        return dict(matches=False, shape_left=list(left.shape), shape_right=list(right.shape),
                    dtype_left=str(left.dtype), dtype_right=str(right.dtype))
    matches, exact, maximum = True, True, 0.0
    for start in range(0, len(left), 4096):
        a, b = left[start:start + 4096], right[start:start + 4096]
        exact = exact and bool(np.array_equal(a, b))
        if np.issubdtype(a.dtype, np.floating):
            matches = matches and bool(np.allclose(a, b, rtol=rtol, atol=atol))
        else:
            matches = matches and bool(np.array_equal(a, b))
        if a.size:
            maximum = max(maximum, float(np.max(np.abs(a.astype('float64') - b.astype('float64')))))
    return dict(matches=matches, bitwise_equal=exact, max_absolute_error=maximum,
                shape=list(left.shape), dtype=str(left.dtype))


def checkpoint_comparison(left, right, *, weights=True, rtol=1e-5, atol=1e-6):
    tensors = {name: compare_arrays(left[name], right[name], rtol=rtol, atol=atol)
               for name in ('mean', 'scale') + (('weights',) if weights else ())}
    for name in sorted(set(left['state']) | set(right['state'])):
        tensors['state.' + name] = (compare_arrays(left['state'][name], right['state'][name],
            rtol=rtol, atol=atol) if name in left['state'] and name in right['state']
            else dict(matches=False))
    keys = ('model', 'layers', 'rms_eps', 'width', 'kind', 'seed', 'learning_rate', 'penalty')
    if weights:
        keys += ('ridge_states',)
    metadata = {name: left.get(name) == right.get(name) for name in keys}
    return dict(matches=all(metadata.values()) and all(v['matches'] for v in tensors.values()),
                metadata=metadata, tensors=tensors)


def run_parity(job_dir, out, device='cpu', *, rtol=1e-5, atol=1e-6):
    """Read a completed Ridge job; never regenerate answers or modify its artifacts.

    Standalone fit performs its own validation selection and all-token readout.
    Evaluation only begins after that fit finishes. Tolerances are recorded, and
    timing fields/checkpoint hashes are not used as numerical equality checks.
    """
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError('Parity tolerances must be finite and nonnegative')
    root, out = Path(job_dir).resolve(), Path(out).resolve()
    if out.is_relative_to(root):
        raise ValueError('Parity output must be outside the immutable managed job')
    status = read_json(root / 'status.json')
    selected = read_json(root / 'selected/selection.json')
    managed = load_checkpoint(root / 'selected' / selected['checkpoint'])
    if status['status'] != 'completed' or managed.get('head', {}).get('method', 'ridge') != 'ridge':
        raise ValueError('Parity requires a completed managed Ridge job')
    for name in ('answers', 'features', 'fitted', 'selected', 'evaluation'):
        directory = root / name
        if read_json(directory / '.complete.json')['files'] != completed_files(directory):
            raise ValueError(f'Managed {name} stage has changed')
    out = require_empty(out)
    cfg = read_json(root / 'fitted/config.json')
    manifest = read_json(root / 'features/manifest.json')
    original_environment, current = read_json(root / 'fitted/environment.json'), environment()
    core = ('data.py', 'features.py', 'collection.py', 'model.py', 'training.py')
    provenance = dict(
        core_source={name: original_environment['source_sha256'].get(name) ==
                     current['source_sha256'].get(name) for name in core},
        packages=original_environment['packages'] == current['packages'],
        python=original_environment['python'] == current['python'],
        threads=original_environment['threads'] == current['threads'],
        cuda=original_environment['cuda'] == current['cuda'],
        gpu=read_json(root / 'smoke/report.json').get('gpu') == (
            torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None))
    write_json(out / 'inputs.json', dict(job_id=status['job_id'], configuration=cfg,
        answers_manifest_sha256=sha256(root / 'answers/manifest.json'),
        managed_environment=original_environment, standalone_environment=current,
        rtol=rtol, atol=atol, provenance=provenance))
    layers = manifest['layer_selection']
    extract(root / 'answers', out / 'features', device, layers['base'], layers['num_slices'],
            baselines=True, checkpoints=manifest['checkpoint_positions'])
    fresh = read_json(out / 'features/manifest.json')
    arrays = {}
    for name in sorted(set(manifest['files']) | set(fresh['files'])):
        if name not in manifest['files'] or name not in fresh['files']:
            arrays[name] = dict(matches=False)
        else:
            arrays[name] = compare_arrays(np.load(root / 'features' / name, mmap_mode='r'),
                np.load(out / 'features' / name, mmap_mode='r'), rtol=rtol, atol=atol)
    metadata = {name: manifest.get(name) == fresh.get(name) for name in
        ('model', 'layers', 'rms_eps', 'dimension', 'layer_selection', 'statuses', 'counts')}
    # Use the standalone public training operation, including its own checkpoint
    # selection, rather than copying the managed compressor into a new head.
    selections = fit(out / 'features', out / 'probes', width=cfg['width'],
        seeds=cfg['seeds'], learning_rates=cfg['learning_rates'], epochs=cfg['epochs'],
        batch_size=cfg['batch_size'], epsilon=cfg['ridge_epsilon'], device=device,
        answers=root / 'answers' if managed['ridge_states'] == 'all-training-tokens' else None)
    compressors = {}
    for name, item in selections.items():
        standalone = load_checkpoint(out / 'probes' / item['path'])
        reference = load_checkpoint(root / 'fitted/slice' / item['path'])
        compressors[name] = checkpoint_comparison(reference, standalone, weights=False,
                                                  rtol=rtol, atol=atol)
        managed_trial = read_json(root / 'fitted/slice/selection.json')['selected'][name]['selected_trial']
        compressors[name]['selected_trial_matches'] = managed_trial == item['selected_trial']
        compressors[name]['matches'] &= compressors[name]['selected_trial_matches']
    standalone = load_checkpoint(out / 'probes' / f'sparse-seed{managed["seed"]}.pt')
    selected_check = checkpoint_comparison(managed, standalone, rtol=rtol, atol=atol)
    left, _ = load_split(root / 'features', 'test')
    right, _ = load_split(out / 'features', 'test')
    managed_scores = predict_features(managed, left['x'], device)
    standalone_scores = predict_features(standalone, right['x'], device)
    predictions = compare_arrays(managed_scores, standalone_scores, rtol=rtol, atol=atol)
    with np.load(root / 'evaluation/predictions.npz') as saved:
        persisted = compare_arrays(saved['raw_value'], managed_scores, rtol=rtol, atol=atol)
    np.savez_compressed(out / 'predictions.npz', managed=managed_scores,
                        standalone=standalone_scores, y=left['y'])
    numerical = (all(metadata.values()) and all(v['matches'] for v in arrays.values())
                 and all(v['matches'] for v in compressors.values())
                 and selected_check['matches'] and predictions['matches'] and persisted['matches'])
    comparable = all(provenance['core_source'].values()) and all(
        value for name, value in provenance.items() if name != 'core_source')
    report = dict(matches=numerical and comparable, numerical_matches=numerical,
        provenance_matches=comparable, provenance=provenance, feature_metadata=metadata,
        arrays=arrays, compressors=compressors, selected_head=selected_check,
        predictions=predictions, saved_predictions=persisted, rtol=rtol, atol=atol,
        scope='Same saved questions/answers and fitting settings. No generation or new training data. '
              'Numerical parity does not establish predictive quality or historical reproduction.')
    write_json(out / 'report.json', report)
    return report
