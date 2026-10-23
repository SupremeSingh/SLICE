"""Regressions for predictor metadata and exact-token feature consumption."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slice_probe import Critic, collection, training
from slice_probe.extraction_parity import audit_sequence
from slice_probe.io import load_checkpoint, save_checkpoint
from slice_probe.metrics import metrics
from slice_probe.model import predict_features, restore
from slice_probe.registry import feature_identity
from test_library import fitted
from test_original import original_study


@pytest.mark.parametrize('tail,budget,eos,reason,capped', [
    ([3, 9], 2, [8, 9], 'eos', False),
    ([3, 4], 2, [9], 'token_budget', True),
    ([9], 2, [9], 'eos', False),
    ([3], 2, [9], 'unknown', None),
])
def test_actual_termination(tail, budget, eos, reason, capped):
    assert collection.termination(tail, budget, eos) == dict(termination_reason=reason, capped=capped)


def test_effective_eos_uses_generation_config_not_tokenizer():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[8, 9]),
                            config=SimpleNamespace(eos_token_id=7))
    assert collection.effective_eos_ids(model, SimpleNamespace(eos_token_id=6)) == [8, 9]
    model.generation_config.eos_token_id = None
    assert collection.effective_eos_ids(model, None) == [7]


def test_final_head_metadata_and_feature_identity(fitted, tmp_path):
    _, features, checkpoint, _ = fitted
    record = load_checkpoint(checkpoint)
    old = record['validation'].copy()
    feature_id = feature_identity(record)
    binding = record['validation_binding']['head_id']
    record['weights'] = torch.tensor([0., 0., 2.], dtype=torch.float64)
    training.update_validation(record, features)
    val, _ = collection.load_split(features, 'val')
    assert record['validation'] == metrics(val['y'], predict_features(record, val['x']))
    assert record['selection_validation_sampled_head'] == old
    assert record['validation_binding']['head_id'] != binding
    assert feature_identity(record) == feature_id
    path = tmp_path / 'outside.pt'
    save_checkpoint(path, record)
    critic = Critic.load(path, device='cpu')
    assert critic.score_batch([dict(token_ids=[1, 2], tokenizer_id=critic.tokenizer_id)])[0]['raw_value'] == 2.
    before = critic.encode_trajectory([1, 2, 3, 4], [2, 3], tokenizer_id=critic.tokenizer_id)
    assert critic.to('cpu') is critic
    np.testing.assert_array_equal(before, critic.encode_trajectory(
        [1, 2, 3, 4], [2, 3], tokenizer_id=critic.tokenizer_id))
    record['state']['projection.bias'] += 1
    assert feature_identity(record) != feature_id


def reference_blocks(backbone, tokens, prompt_length, response_length, positions, layers):
    saved, hooks = {}, []
    for layer in layers:
        def hook(module, inputs, output, layer=layer):
            saved[layer] = output[0, [prompt_length+p-1 for p in positions]].detach().cpu()
        hooks.append(backbone.layers[layer].register_forward_hook(hook))
    try:
        backbone(input_ids=tokens)
    finally:
        for hook in hooks:
            hook.remove()
    return saved


def test_fresh_extraction_audit_detects_transformation_drift(fitted):
    _, _, checkpoint, _ = fitted
    critic = Critic.load(checkpoint, device='cpu')
    original = restore(critic.record)
    rms = lambda x, eps: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    report = audit_sequence(critic, [1, 2, 3, 4, 5], 2, reference_blocks, rms, original)
    assert report['matches'] and report['prefix_lengths'][0] == 2
    bad_rms = lambda x, eps: rms(x, eps) * 2
    report = audit_sequence(critic, [1, 2, 3, 4, 5], 2, reference_blocks, bad_rms, original)
    assert not report['matches']
    assert not report['checks']['rms_concatenation']['matches']


def test_fitting_populations_preserve_ridge_caps(fitted):
    from slice_probe.io import read_json, write_json
    answers, _, _, _ = fitted
    path = answers / 'answers/00000.json'
    batch = read_json(path)
    batch['responses'][0]['capped'] = True
    write_json(path, batch)
    ridge = collection.fitting_population(answers)
    lstd = collection.fitting_population(answers, complete_only=True)
    assert ridge['counts']['included_capped'] == 1
    assert lstd['counts']['excluded_capped_or_unknown'] == 1
    assert ridge['counts']['included'] == lstd['counts']['included'] + 1


def test_extraction_parity_command_wiring(original_study, tmp_path):
    from slice_probe.extraction_parity import run_extraction_parity
    from slice_probe.io import read_json, write_json, sha256
    study, model, _ = original_study
    protocol = read_json(study / 'protocol.json')
    from pathlib import Path
    source = Path(protocol['source'])
    questions = read_json(source / 'questions.json')
    for i, question in enumerate(questions):
        question.update(index=i, prompt_token_ids=[1, 2, 3])
        write_json(source / 'trajectories' / f'{i:05d}.json',
                   dict(responses=[dict(token_ids=[4, 5, 6])]))
    write_json(source / 'questions.json', questions)
    manifest = read_json(source / 'manifest.json')
    manifest['questions_sha256'] = sha256(source / 'questions.json')
    write_json(source / 'manifest.json', manifest)
    protocol['source_manifest'] = sha256(source / 'manifest.json')
    write_json(study / 'protocol.json', protocol)
    root = tmp_path / 'original-project'
    modules = root / 'src/math_rl'
    modules.mkdir(parents=True)
    (modules / 'critic_layers.py').write_text('from test_critic_contract import reference_blocks as sampled_block_states\n')
    (modules / 'direct_jlens.py').write_text(
        'import torch\ndef rms_coordinates(x, eps):\n    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)\n')
    (modules / 'sparse_bottleneck.py').write_text('from slice_probe.model import Bottleneck\n')
    out = tmp_path / 'fresh-audit'
    report = run_extraction_parity(study, model, root, out, device='cpu', samples=2)
    assert report['matches']
    assert report['max_absolute_error'] < 1e-5
    assert len(report['results']['7']['sequences']) == 2
    assert read_json(out / 'report.json')['original_code_sha256']


def test_failed_reload_leaves_coherent_cpu_extractor(fitted, monkeypatch):
    _, _, checkpoint, _ = fitted
    critic = Critic.load(checkpoint, device='cpu')
    original_id = critic.feature_id
    def failed_transfer(device):
        raise RuntimeError('simulated device transfer failure')
    monkeypatch.setattr(critic.encoder, 'to', failed_transfer)
    with pytest.raises(RuntimeError, match='transfer failure'):
        critic.to('cpu')
    assert critic.device == 'cpu' and critic.feature_id == original_id
    assert critic._mean.device.type == critic._scale.device.type == 'cpu'
    assert critic.encode_trajectory([1, 2, 3], [2], tokenizer_id=critic.tokenizer_id).shape == (1, 2)
