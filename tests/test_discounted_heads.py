"""Discounted returns must not silently become binary correctness reports."""
import numpy as np
import pytest

from slice_probe import Critic, inference, lstd, training
from slice_probe.evaluation import evaluate
from slice_probe.io import load_checkpoint, read_json, save_checkpoint, sha256, write_json
from slice_probe.registry import feature_identity, publish
from test_library import fitted


def test_discounted_fit_preserves_features_and_selection_but_omits_correctness_validation(fitted, tmp_path):
    answers, features, checkpoint, _ = fitted
    trajectories = lstd.extract_trajectories(checkpoint, answers, tmp_path / 'trajectories',
                                             feature_manifest=features / 'manifest.json')
    result = lstd.fit_lstd(checkpoint, trajectories, features / 'manifest.json', tmp_path / 'discounted',
                           discount_gamma=.5)
    record = load_checkpoint(result)
    original = load_checkpoint(checkpoint)
    assert feature_identity(record) == feature_identity(original)
    assert 'validation' not in record and 'validation_binding' not in record
    assert record['selection_validation_sampled_head'] == original['validation']
    assert record['validation_status']['status'] == 'not_evaluated'
    assert 'discounted' in record['head']['target']
    assert read_json(result.parent / 'fit.json')['validation_status'] == record['validation_status']
    critic = Critic.load(result, device='cpu')
    scored = critic.score_batch([dict(question='A sum?')])[0]
    assert 'raw_value' in scored and 'predicted_correct' not in scored
    assert scored['discount_gamma'] == .5
    assert 'discounted' in scored['value_target']
    with pytest.raises(ValueError, match='gamma=1'):
        critic.score_batch([dict(question='A sum?')], return_calibrated_probability=True)
    with pytest.raises(ValueError, match='gamma=1'):
        training.update_validation(record, features)
    # Raw discounted artifacts remain publishable, and retain frozen-feature access.
    artifact = publish(result, tmp_path / 'registry', 'discounted')
    assert Critic.load(artifact, device='cpu').feature_id == critic.feature_id


def test_standalone_discounted_prediction_has_no_correctness_threshold(fitted, tmp_path):
    _, features, checkpoint, _ = fitted
    record = load_checkpoint(checkpoint)
    record['head'] = dict(method='lstd', discount_gamma=.5, target='discounted terminal correctness reward')
    path = tmp_path / 'discounted.pt'
    save_checkpoint(path, record)
    result = inference.predict(path, 'A sum?')
    assert np.isfinite(result['score']) and result['raw_value'] == result['score']
    assert 'predicted_correct' not in result
    assert result['discount_gamma'] == .5
    with pytest.raises(ValueError, match='gamma=1'):
        inference.predict(path, 'A sum?', calibration=tmp_path / 'unused.json')
    with pytest.raises(ValueError, match='gamma=1'):
        Critic.load(path, calibration=tmp_path / 'unused.json')
    with pytest.raises(ValueError, match='gamma=1'):
        publish(path, tmp_path / 'registry', 'invalid', calibration=tmp_path / 'unused.json')
    with pytest.raises(ValueError, match='gamma=1'):
        publish(path, tmp_path / 'registry', 'stale-correctness-validation')
    probes = tmp_path / 'discounted-probes'
    probes.mkdir()
    save_checkpoint(probes / 'discounted.pt', record)
    write_json(probes / 'selection.json', dict(feature_manifest_sha256=sha256(features / 'manifest.json'),
        selected={'sparse-7':dict(path='discounted.pt', sha256=sha256(probes / 'discounted.pt'))}))
    with pytest.raises(ValueError, match='gamma=1'):
        evaluate(features, probes, tmp_path / 'evaluation')
