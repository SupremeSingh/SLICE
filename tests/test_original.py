"""Reuse original stored heads, with external saved-prediction validation."""
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from slice_probe.collection import load_split
from slice_probe.features import prompt
from slice_probe.io import load_checkpoint, read_json, sha256, write_json
from slice_probe.model import predict_features
from slice_probe.original import import_original
from slice_probe.registry import inspect_artifact
from test_library import fitted


@pytest.fixture
def original_study(fitted, tmp_path):
    answers, features, checkpoint, _ = fitted
    record = load_checkpoint(checkpoint)
    source, study, model = (tmp_path / name for name in ('source', 'original-study', 'model-assets'))
    write_json(model / 'config.json', dict(hidden_size=3, num_hidden_layers=2))
    (model / 'model.safetensors').write_bytes(b'test-only pinned asset; no model load')
    questions = [dict(q, prompt=prompt(q['question'])) for q in read_json(answers / 'questions.json')]
    write_json(source / 'questions.json', questions)
    write_json(source / 'manifest.json', dict(
        questions_sha256=sha256(source / 'questions.json'),
        model_hashes={p.name: sha256(p) for p in model.iterdir()},
        layers=[n - 1 for n in record['layers']], rms_eps=record['rms_eps']))
    packed = source / 'packed/rms_hidden_concat'
    packed.mkdir(parents=True)
    shutil.copy2(features / 'test-x.npy', packed / 'test-x.npy')
    data, _ = load_split(features, 'test')
    torch.save({name: torch.from_numpy(np.array(data[name], copy=True))
                for name in ('y', 'question')}, packed / 'test-labels.pt')
    write_json(packed / 'complete.json', dict(counts={'test': len(data['y'])},
        files={p.name: sha256(p) for p in packed.iterdir()}))
    write_json(study / 'protocol.json', dict(source=str(source), width=record['width'], seeds=[7],
        source_manifest=sha256(source / 'manifest.json'),
        packed={'rms_hidden_concat': sha256(packed / 'complete.json')}))
    compressor = study / 'sparse-7-0.001.pt'
    torch.save({key: record[key] for key in ('state', 'mean', 'scale', 'width', 'seed')}, compressor)
    write_json(compressor.with_suffix('.json'), {'sha256': sha256(compressor)})
    write_json(study / 'screen.json', dict(selected={'sparse-7': dict(
        path=compressor.name, sha256=sha256(compressor))}))
    head = study / 'sparse-7-ridge-fit.pt'
    torch.save(dict(weights=record['weights'], checks=record['fit_checks']), head)
    write_json(head.with_suffix('.json'), {'sha256': sha256(head)})
    np.savez_compressed(study / 'predictions.npz', **{'sparse-7/ridge': predict_features(record, data['x'])},
                        y=data['y'], question=data['question'])
    return study, model, record


def test_import_original_preserves_weights_and_saved_predictions(original_study, tmp_path):
    study, model, original = original_study
    before = {p: sha256(p) for p in study.iterdir() if p.is_file()}
    result = import_original(study, model, tmp_path / 'registry', tmp_path / 'import')
    assert result['retrained'] is False
    artifact = result['artifacts'][0]
    manifest = inspect_artifact(artifact)
    wrapped = load_checkpoint(Path(artifact) / 'checkpoint.pt')
    for name in ('weights', 'mean', 'scale'):
        torch.testing.assert_close(original[name], wrapped[name], rtol=0, atol=0)
    for name in original['state']:
        torch.testing.assert_close(original['state'][name], wrapped['state'][name], rtol=0, atol=0)
    assert wrapped['layers'] == original['layers']
    assert manifest['metadata']['validation']['matches']
    assert manifest['metadata']['validation']['total_rows'] > 0
    assert {p: sha256(p) for p in before} == before


def test_import_original_refuses_bad_predictions_without_publishing(original_study, tmp_path):
    study, model, _ = original_study
    path = study / 'predictions.npz'
    with np.load(path) as data:
        copied = {key: data[key].copy() for key in data.files}
    copied['sparse-7/ridge'] += 1
    np.savez_compressed(path, **copied)
    with pytest.raises(ValueError, match='prediction check failed'):
        import_original(study, model, tmp_path / 'registry', tmp_path / 'import')
    assert not (tmp_path / 'registry').exists()
    assert not read_json(tmp_path / 'import/validation.json')['heads']['sparse-7']['matches']


def test_import_original_refuses_changed_model_assets(original_study, tmp_path):
    study, model, _ = original_study
    (model / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='asset mismatch'):
        import_original(study, model, tmp_path / 'registry', tmp_path / 'import')
