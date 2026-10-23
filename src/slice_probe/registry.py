"""Immutable, checksummed artifacts shared by independent SLICE consumers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import fcntl

import torch

from .features import INSTRUCTION, prompt
from .io import load_checkpoint, read_json, sha256, write_json


VERSION = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z')


def version_name(value: str) -> str:
    if not VERSION.fullmatch(value):
        raise ValueError('Version must contain 1–96 letters, numbers, underscores or hyphens')
    return value


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


FEATURE_IMPLEMENTATION = 'slice-frozen-features-v1'


def feature_identity(record: dict) -> str:
    """Head-independent identity of the exact encoder and compressed coordinates."""
    spec = {key: record[key] for key in
            ('model', 'layers', 'rms_eps', 'input_dim', 'width')}
    # v1 retains existing IDs. Bump this contract when transformation semantics
    # change; subsequent versions become part of the digest, even with identical tensors.
    if FEATURE_IMPLEMENTATION != 'slice-frozen-features-v1':
        spec['implementation'] = FEATURE_IMPLEMENTATION
    spec['prompt'] = prompt('{question}')
    spec['precision'] = 'float32-eager-no-tf32'
    spec['tokenization'] = 'encode(prompt, special_tokens=False) + encode(prefix, special_tokens=False)'
    digest = hashlib.sha256(canonical_hash(spec).encode())
    tensors = {name: record[name] for name in ('mean', 'scale')}
    tensors.update({key: value for key, value in record['state'].items()
                    if key.startswith('projection.')})
    for name, tensor in sorted(tensors.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(canonical_hash([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def inspect_artifact(path: Path | str) -> dict:
    root = Path(path).resolve()
    manifest = read_json(root / 'manifest.json')
    if manifest.get('format') != 'slice-critic-v1':
        raise ValueError('Unknown critic artifact format')
    version_name(manifest['version'])
    for name, expected in manifest['files'].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha256(target) != expected:
            raise ValueError(f'Artifact integrity failure: {name}')
    for name in ('checkpoint.pt', 'checkpoint.sha256.json'):
        if name not in manifest['files']:
            raise ValueError(f'Artifact missing required checksum: {name}')
    record = load_checkpoint(root / 'checkpoint.pt')
    if feature_identity(record) != manifest['feature_id']:
        raise ValueError('Artifact feature identity changed')
    if manifest['checkpoint_sha256'] != sha256(root / 'checkpoint.pt'):
        raise ValueError('Artifact checkpoint identity changed')
    if manifest['features']['layers'] != record['layers']:
        raise ValueError('Published block list differs from the checkpoint')
    return manifest


def publish(checkpoint, registry, version, *, calibration=None, evaluation=None,
            feature_manifest=None, metadata=None, statistics=None, trajectory_manifest=None) -> Path:
    """Publish only complete directories. Existing versions are never overwritten."""
    from .model import restore

    version_name(version)
    checkpoint = Path(checkpoint)
    record = load_checkpoint(checkpoint)
    restore(record)  # Validate dimensions, weights and normalization before publication.
    if any(not torch.isfinite(value).all() for value in record['state'].values()):
        raise ValueError('Cannot publish nonfinite compressor weights')
    if not record.get('model', {}).get('hashes'):
        raise ValueError('A published encoder must have pinned model asset hashes')
    registry = Path(registry).resolve()
    registry.mkdir(parents=True, exist_ok=True)
    destination = registry / version
    if destination.exists():
        raise FileExistsError(f'Critic version already exists: {version}')
    temporary = Path(tempfile.mkdtemp(prefix='.publishing-', dir=registry))
    try:
        shutil.copy2(checkpoint, temporary / 'checkpoint.pt')
        shutil.copy2(checkpoint.with_suffix('.sha256.json'), temporary / 'checkpoint.sha256.json')
        if calibration is not None:
            data = read_json(calibration)
            matches = [name for name, digest in data['checkpoint_sha256'].items()
                       if digest == sha256(checkpoint) and name in data['calibrators']]
            if len(matches) != 1:
                raise ValueError('Calibration must identify exactly this checkpoint')
            shutil.copy2(calibration, temporary / 'calibration.json')
        if evaluation is not None:
            source = Path(evaluation)
            if source.is_dir():
                if any(p.is_symlink() for p in source.rglob('*')):
                    raise ValueError('Evaluation artifacts must not contain symlinks')
                shutil.copytree(source, temporary / 'evaluation')
            else:
                shutil.copy2(source, temporary / 'evaluation.json')
        features = read_json(feature_manifest) if feature_manifest else {}
        if feature_manifest:
            if record.get('feature_manifest_sha256') != sha256(feature_manifest):
                raise ValueError('Feature manifest does not identify checkpoint training data')
            shutil.copy2(feature_manifest, temporary / 'feature-manifest.json')
            if features.get('questions_sha256'):
                questions = Path(feature_manifest).parent / 'questions.json'
                if sha256(questions) != features['questions_sha256']:
                    raise ValueError('Question split snapshot differs from feature manifest')
                shutil.copy2(questions, temporary / 'questions.json')
        for source, name in ((statistics, 'statistics.npz'),
                             (trajectory_manifest, 'trajectory-manifest.json')):
            if source is not None:
                shutil.copy2(source, temporary / name)
        collection = features.get('collection', {})
        manifest = dict(
            format='slice-critic-v1', version=version,
            feature_implementation=FEATURE_IMPLEMENTATION,
            training_population=record.get('training_population'),
            validation=record.get('validation'), validation_binding=record.get('validation_binding'),
            feature_id=feature_identity(record), checkpoint_sha256=sha256(checkpoint),
            encoder=record['model'], tokenizer_id=canonical_hash(record['model']),
            features=dict(layers=record['layers'], layer_selection=record.get('layer_selection'),
                          input_dim=record['input_dim'], width=record['width'],
                          rms_eps=record['rms_eps'], projection='row-normalized linear + ReLU',
                          normalization='training mean and isotropic scale; stored in checkpoint',
                          precision='float32-eager-no-tf32'),
            prompt=dict(instruction=INSTRUCTION, version='completion-boxed-v1',
                        tokenization='prompt and prefix encoded separately, no special tokens'),
            head=record.get('head', dict(method='ridge', intercept=True,
                                        fit_checks=record.get('fit_checks', {}),
                                        states=record.get('ridge_states'))),
            generation=collection or None, metadata=metadata or {},
            files={str(p.relative_to(temporary)): sha256(p)
                   for p in temporary.rglob('*') if p.is_file()},
        )
        write_json(temporary / 'manifest.json', manifest)
        inspect_artifact(temporary)
        # Serialize publication across processes, including a race with an empty
        # destination. Readers see either no version or the complete directory.
        with (registry / '.publish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if destination.exists():
                raise FileExistsError(f'Critic version already exists: {version}')
            os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
