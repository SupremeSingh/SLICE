"""Compute-node smoke gate tied to the approved request and frozen encoder."""
from pathlib import Path

import numpy as np
import torch

from .comparison import environment
from .features import capture, concatenate, geometric_layers, load_model, prompt
from .io import read_json, sha256, write_json


def execution_identity(bundle):
    execution = dict(bundle.get('execution', {}))
    snapshot = execution.get('source_snapshot')
    if snapshot and not Path(__file__).resolve().is_relative_to(Path(snapshot).resolve()):
        raise ValueError('Worker is not importing the approved per-job source snapshot')
    image = execution.get('container_image')
    if image:
        execution['container_sha256'] = sha256(image)
    return execution


def smoke(out, request, bundle, device, execution):
    """Exercise the actual encoder, selected blocks and verifier before collection."""
    from .data import grade

    if execution.get('gres') and not str(device).startswith('cuda'):
        raise ValueError('Allocated GPU is unavailable; check the container runtime and --nv')
    model, tokenizer, identity = load_model('Qwen/Qwen2.5-Math-1.5B', device=device)
    layers = geometric_layers(len(model.layers), request.layer_base, request.num_slices)
    ids = tokenizer.encode(prompt('What is 1 + 1?'), add_special_tokens=False)
    if not ids or len(ids) + request.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError('Smoke prompt plus generation limit exceeds model context')
    full = capture(model, torch.tensor([ids + [ids[-1]]], device=device), [len(ids) - 1], layers)
    partial = capture(model, torch.tensor([ids], device=device), [len(ids) - 1], layers)
    for layer in layers:
        torch.testing.assert_close(full[layer], partial[layer], rtol=1e-4, atol=1e-5)
    x = concatenate(partial, layers, model.config.rms_norm_eps).numpy()
    if x.shape != (1, len(layers) * model.config.hidden_size) or not np.isfinite(x).all():
        raise ValueError('Smoke feature shape or finiteness check failed')
    if grade(r'The answer is \boxed{2}.', '2')['status'] != 'correct':
        raise ValueError('Math-Verify smoke case failed')
    write_json(out / 'report.json', dict(
        passed=True, request_hash=bundle['request_hash'], model=identity,
        layers=layers, dimension=x.shape[1], execution=execution, environment=environment(),
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        scope='Runtime/causality smoke check for this exact request; not a scientific result.'))


def check_gate(path, bundle, execution):
    report = read_json(path)
    current = environment()
    if (report.get('passed') is not True or report['request_hash'] != bundle['request_hash']
            or report['execution'] != execution):
        raise ValueError('Smoke gate does not match approved request/execution')
    for key in ('source_sha256', 'packages', 'python'):
        if report['environment'][key] != current[key]:
            raise ValueError(f'Runtime changed since smoke check: {key}')
    return report
