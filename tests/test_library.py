"""Contracts for frozen features, immutable publication and trajectory fitting."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from slice_probe import Critic, collection, critic as library, inference, lstd, training
from slice_probe.io import load_checkpoint, read_json, write_json
from slice_probe.registry import feature_identity, inspect_artifact, publish
from test_workflow import ToyBackbone, answer_fixture


class Tokenizer:
    def encode(self, text, **kwargs):
        return [1, 2, 3] if text.endswith('Solution:') else [4, 5]


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    answers = answer_fixture(tmp_path)
    identity = dict(id='toy', revision='a' * 40, hashes={'config.json': 'b' * 64})
    write_json(answers / 'manifest.json', dict(model=identity))
    backbone = ToyBackbone()
    backbone.norm = torch.nn.Identity()
    load = lambda *a, **kw: (backbone, Tokenizer(), identity)
    for module in (collection, library, inference, lstd):
        monkeypatch.setattr(module, 'load_model', load)
    features = tmp_path / 'features'
    collection.extract(answers, features, 'cpu', baselines=True, checkpoints=[0, 1])
    training.fit(features, tmp_path / 'probes', width=2, seeds=[7], kinds=['sparse'],
                 learning_rates=[.001], epochs=1, batch_size=16)
    return answers, features, tmp_path / 'probes/sparse-seed7.pt', backbone


def test_resident_scores_features_exact_tokens_and_input_errors(fitted, tmp_path):
    _, _, checkpoint, backbone = fitted
    critic = Critic.load(checkpoint, device='cpu')
    states = [dict(state_id='q', question='A sum?', prefix='some work')]
    old = inference.predict(checkpoint, 'A sum?', 'some work')['score']
    assert critic.score_batch(states)[0]['raw_value'] == pytest.approx(old, abs=1e-12)
    z = critic.encode_batch(states)
    record = load_checkpoint(checkpoint)
    expected = z.astype('float64') @ record['weights'][:-1].numpy() + float(record['weights'][-1])
    np.testing.assert_allclose([r['raw_value'] for r in critic.score_batch(states)], expected)
    assert not critic.compressor.training and not critic.encoder.training
    assert all(not p.requires_grad for p in critic.compressor.parameters())
    exact = dict(state_id='exact', token_ids=[1, 2, 3, 4, 5], tokenizer_id=critic.tokenizer_id)
    np.testing.assert_array_equal(critic.encode_batch([exact]), z)
    with pytest.raises(ValueError, match='tokenizer'):
        critic.encode_batch([dict(token_ids=[1], tokenizer_id='wrong')])
    result = critic.score_batch([dict(state_id='bad', question=''),
                                 dict(token_ids=[1] * 129, tokenizer_id=critic.tokenizer_id)])
    assert all('error' in row for row in result)
    with pytest.raises(ValueError, match='calibrator'):
        critic.score_batch(states, return_calibrated_probability=True)
    with ThreadPoolExecutor(2) as pool:
        values = list(pool.map(lambda _: critic.encode_batch(states), range(4)))
    assert all(np.array_equal(z, value) for value in values)
    assert all(not layer._forward_hooks for layer in backbone.layers)


def test_registry_publication_is_atomic_and_immutable(fitted, tmp_path):
    _, features, checkpoint, _ = fitted
    root = tmp_path / 'registry'
    path = publish(checkpoint, root, 'v1', feature_manifest=features / 'manifest.json')
    manifest = inspect_artifact(path)
    assert manifest['features']['layers'] == [2, 1]
    assert manifest['feature_id'] == feature_identity(load_checkpoint(checkpoint))
    assert Critic.load(registry=root, version='v1', device='cpu').version == 'v1'
    with pytest.raises(FileExistsError):
        publish(checkpoint, root, 'v1')
    with pytest.raises(ValueError):
        publish(checkpoint, root, '../escape')
    (path / 'checkpoint.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        inspect_artifact(path)


def test_http_scoring_matches_local_library_and_reuses_encoder(fitted, tmp_path):
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from slice_probe.api import create_app

    _, _, checkpoint, _ = fitted
    registry = tmp_path / 'registry'
    path = publish(checkpoint, registry, 'private', metadata={'project': 'owner'})
    config = tmp_path / 'service.json'
    write_json(config, dict(database=str(tmp_path / 'jobs.sqlite'), registry=str(registry),
        cluster_root='/cluster/jobs', cluster_python='/cluster/python', partition='gpu', account='research'))
    calls = []

    def load(*args, **kwargs):
        calls.append('loaded')
        return Critic.load(*args, **kwargs)

    app = create_app(config, tokens={'owner': 'a' * 32, 'other': 'b' * 32}, loader=load)
    state = dict(state_id='s', question='A sum?', prefix='work')
    local = Critic.load(path).score_batch([state])[0]['raw_value']
    with TestClient(app) as client:
        for _ in range(2):
            result = client.post('/v1/scoring-batches', headers={'Authorization': 'Bearer ' + 'a' * 32},
                                 json=dict(critic_version='private', states=[state]))
            assert result.status_code == 200, result.text
            assert result.json()['results'][0]['raw_value'] == pytest.approx(local, abs=1e-12)
        assert calls == ['loaded']
        assert client.get('/v1/critics/private', headers={'Authorization': 'Bearer ' + 'b' * 32}).status_code == 404


@pytest.mark.parametrize('trace_lambda', [0., .4, 1.])
def test_lstd_matches_independent_eligibility_sum_and_terminal_boundary(trace_lambda):
    stats = lstd.LSTD(2, trace_lambda=trace_lambda, discount_gamma=.95)
    episodes = [(np.array([[1., 2.], [3., 1.], [0., 2.]]), np.array([0., 0., 1.])),
                (np.array([[2., 1.]]), np.array([0.]))]
    expected_a, expected_b = np.zeros((3, 3)), np.zeros(3)
    for x, rewards in episodes:
        stats.add_episode(x, rewards)
        phi = np.column_stack([x, np.ones(len(x))])
        for t in range(len(phi)):
            eligibility = sum((.95 * trace_lambda) ** (t - k) * phi[k] for k in range(t + 1))
            following = phi[t + 1] if t + 1 < len(phi) else np.zeros(3)
            expected_a += np.outer(eligibility, phi[t] - .95 * following)
            expected_b += eligibility * rewards[t]
    np.testing.assert_allclose(stats.a, expected_a)
    np.testing.assert_allclose(stats.b, expected_b)
    weights, checks = stats.solve()
    np.testing.assert_allclose(weights, np.linalg.solve(expected_a + .01 * np.eye(3), expected_b))
    assert checks['transitions'] == 4 and checks['episodes'] == 2
    previous = stats.a.copy()
    with pytest.raises(ValueError):
        stats.add_episode([[np.nan, 1]], [1])
    np.testing.assert_array_equal(stats.a, previous)


def test_lstd_excludes_truncation_preserves_features_and_checks_provenance(fitted, tmp_path):
    answers, features, checkpoint, _ = fitted
    trajectories = tmp_path / 'trajectories'
    lstd.extract_trajectories(checkpoint, answers, trajectories, feature_manifest=features / 'manifest.json')
    result = lstd.fit_lstd(checkpoint, trajectories, features / 'manifest.json', tmp_path / 'lstd', trace_lambda=.7)
    original, updated = load_checkpoint(checkpoint), load_checkpoint(result)
    assert feature_identity(original) == feature_identity(updated)
    for key in original['state']:
        torch.testing.assert_close(original['state'][key], updated['state'][key], atol=0, rtol=0)
    assert updated['head']['trace_lambda'] == .7
    assert updated['fit_checks']['transitions'] == 5
    manifest = read_json(trajectories / 'manifest.json')
    manifest['entries'].append(dict(manifest['entries'][0], trajectory_id='forged-new-id'))
    write_json(trajectories / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='Duplicate answer'):
        lstd.fit_lstd(checkpoint, trajectories, features / 'manifest.json', tmp_path / 'bad')
    batch_path = answers / 'answers/00000.json'
    batch = read_json(batch_path)
    batch['responses'][0]['capped'] = True
    write_json(batch_path, batch)
    # Altering saved termination metadata after extraction is rejected.
    with pytest.raises(ValueError, match='changed'):
        lstd.extract_trajectories(checkpoint, answers, tmp_path / 'changed',
                                  feature_manifest=features / 'manifest.json')


def test_complete_undiscounted_lambda_one_matches_monte_carlo_ridge():
    episodes = [([[1., 2.], [3., -1.]], 1.), ([[0., 2.], [2., 1.], [4., 1.]], 0.)]
    statistics = lstd.LSTD(2, trace_lambda=1., discount_gamma=1.)
    x, y = [], []
    for states, outcome in episodes:
        rewards = np.zeros(len(states))
        rewards[-1] = outcome
        statistics.add_episode(states, rewards)
        x.extend(states)
        y.extend([outcome] * len(states))
    weights, _ = statistics.solve()
    expected, _ = training.fit_ridge(torch.tensor(x), np.array(y), .01)
    torch.testing.assert_close(weights, expected, rtol=1e-10, atol=1e-10)
