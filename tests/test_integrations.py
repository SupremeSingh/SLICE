"""Optional real-library integration tests; no pretrained downloads required."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from slice_probe import collection, features
from slice_probe.io import write_json


def test_dataset_runtime_numpy_arrow_compatibility():
    from importlib.util import find_spec
    from slice_probe.preflight import check_data_runtime

    if find_spec('datasets') is None:
        pytest.skip('Install the data extra to exercise the dataset runtime')
    # An installed but broken dependency must fail, not become an optional skip.
    check_data_runtime()


def test_data_runtime_failure_precedes_model_loading(tmp_path, monkeypatch):
    from slice_probe import preflight

    def broken_runtime():
        raise ImportError('incompatible Arrow runtime')

    def unexpected_model(*args, **kwargs):
        pytest.fail('Model loading must follow the dataset runtime check')

    monkeypatch.setattr(preflight, 'check_data_runtime', broken_runtime)
    monkeypatch.setattr(preflight, 'load_model', unexpected_model)
    with pytest.raises(ImportError, match='incompatible Arrow'):
        preflight.smoke(tmp_path, None, {}, 'cpu', {})


def test_real_three_block_transformer_causality_and_extraction(tmp_path):
    transformers=pytest.importorskip('transformers')
    tokenizers=pytest.importorskip('tokenizers')
    torch.manual_seed(13)
    config=transformers.Qwen2Config(vocab_size=32,hidden_size=24,intermediate_size=48,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
    config._attn_implementation='eager'
    lm=transformers.Qwen2ForCausalLM(config).float().eval().requires_grad_(False)
    backbone=lm.model
    ids=torch.tensor([[1,2,3,4,5,6]])
    full=features.capture(backbone,ids,[2,4],[3,2,1])
    for index,stop in enumerate((3,5)):
        partial=features.capture(backbone,ids[:,:stop],[stop-1],[3,2,1])
        for layer in full:torch.testing.assert_close(full[layer][index],partial[layer][0],rtol=1e-5,atol=1e-6)
    model=tmp_path/'model';lm.save_pretrained(model)
    tokenizer=tokenizers.Tokenizer(tokenizers.models.WordLevel(
        vocab={'[UNK]':0,**{f't{i}':i for i in range(1,32)}},unk_token='[UNK]'))
    transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]').save_pretrained(model)
    _,identity=features.model_location(str(model))
    answers=tmp_path/'answers';write_json(answers/'manifest.json',dict(model=identity))
    questions=[]
    for i,split in enumerate(('train','val','test')):
        q=dict(id=f'q{i}',split=split,question=f'Topic {split}',reference='1');questions.append(q)
        write_json(answers/'answers'/f'{i:05d}.json',dict(id=q['id'],prompt_ids=[1,2,3],responses=[
            dict(status='correct',score=1,token_ids=[4,5,6]),dict(status='incorrect',score=0,token_ids=[7,8,9])]))
    write_json(answers/'questions.json',questions)
    collection.extract(answers,tmp_path/'features','cpu',baselines=True,checkpoints=[0,1,2])
    final,m=collection.load_split(tmp_path/'features','test',representation='final',anchors=True)
    assert final['x'].shape==(6,24) and m['representations']['two_thirds']['block']==2
    with torch.inference_mode():expected=backbone.norm(full[3][0]).numpy()
    np.testing.assert_allclose(final['x'][0],expected,rtol=1e-5,atol=1e-6)
    # Exercise every probe, validation calibration and the confirmation boundary
    # on actual transformer features, not the ToyBackbone used by unit tests.
    from slice_probe.comparison import fit_comparison, evaluate_comparison
    from slice_probe.io import read_json
    write_json(tmp_path/'comparison-config.json', dict(width=4,resnet_width=4,epochs=1,
        batch_size=16,seeds=[7],learning_rates=[.001],mode='confirmation',
        test_scope='Random tiny transformer integration test; not scientific evidence'))
    digest=fit_comparison(tmp_path/'features',tmp_path/'fitted',tmp_path/'comparison-config.json')
    results=evaluate_comparison(tmp_path/'features',tmp_path/'fitted',tmp_path/'evaluation',lock_sha256=digest)
    assert len(results)==7 and 'sparse/ridge' in results
    report=read_json(tmp_path/'evaluation/summary.json')
    assert report['within_question']['sparse/ridge']['by_token_count']['0']['mean_auroc']==.5


@pytest.mark.parametrize('text,target,expected',[
    (r'The answer is \boxed{3}.','3','correct'),
    (r'The answer is \boxed{4}.','3','incorrect'),
    ('No numeric answer.','3','parse_failure'),
    (r'\boxed{3','3','malformed_box'),
    (r'\boxed{3} or \boxed{4}','3','ambiguous_final'),
    ('Answer: 3 or 4.','3','ambiguous_final'),
    ('Answer: 3, 4.','3','ambiguous_final'),
    (r'\boxed{3}. Actually, the final answer is 4.','3','ambiguous_final'),
    ('The answer is 1,234.','1234','correct'),
])
def test_real_verifier_cases(text,target,expected):
    pytest.importorskip('math_verify')
    from slice_probe.data import grade
    result=grade(text,target)
    assert result['status']==expected
    if expected not in ('correct','incorrect'):assert result['score'] is None


def test_verifier_timeout_boundary(monkeypatch):
    verifier=pytest.importorskip('math_verify')
    from math_verify.errors import TimeoutException
    from slice_probe.data import grade
    def timed_out(*a,**kw):raise TimeoutException('controlled test timeout')
    monkeypatch.setattr(verifier,'parse',timed_out)
    assert grade(r'\boxed{3}','3')['status']=='parse_timeout'
    monkeypatch.undo()
    monkeypatch.setattr(verifier,'verify',timed_out)
    assert grade(r'\boxed{3}','3')['status']=='verify_timeout'


def test_published_resident_critic_matches_original_real_transformer_inference(tmp_path, monkeypatch):
    transformers = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    from slice_probe import Critic
    from slice_probe.inference import predict
    from slice_probe.io import save_checkpoint
    from slice_probe.model import Bottleneck
    from slice_probe.registry import publish

    torch.manual_seed(17)
    config = transformers.Qwen2Config(vocab_size=512, hidden_size=24, intermediate_size=48,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512)
    config._attn_implementation = 'eager'
    lm = transformers.Qwen2ForCausalLM(config).float().eval().requires_grad_(False)
    model = tmp_path / 'model'
    lm.save_pretrained(model)
    vocabulary = {char: i for i, char in enumerate(sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
    vocabulary['<|endoftext|>'] = len(vocabulary)
    transformers.Qwen2Tokenizer(vocab=vocabulary, merges=[]).save_pretrained(model)
    _, identity = features.model_location(str(model))
    compressor = Bottleneck(72, 4)
    checkpoint = tmp_path / 'probe.pt'
    save_checkpoint(checkpoint, dict(format='slice-probe-v1', input_dim=72, width=4,
        model=identity, layers=[3, 2, 1], rms_eps=config.rms_norm_eps,
        mean=torch.zeros(72), scale=torch.ones(72), state=compressor.state_dict(),
        weights=torch.randn(5, dtype=torch.float64)))
    artifact = publish(checkpoint, tmp_path / 'registry', 'random-test')
    critic = Critic.load(artifact, device='cpu')
    state = dict(state_id='s', question='t1 t2', prefix='t3 t4')
    expected = predict(checkpoint, state['question'], state['prefix'])['score']
    assert critic.score_batch([state])[0]['raw_value'] == pytest.approx(expected, abs=1e-7)
    ids = critic._tokens(state)
    exact = dict(token_ids=ids, tokenizer_id=critic.tokenizer_id)
    np.testing.assert_array_equal(critic.encode_batch([state]), critic.encode_batch([exact]))
    lengths = [len(ids), 1, len(ids) - 1, 1]
    calls = []
    handle = critic.encoder.register_forward_pre_hook(lambda *args: calls.append(1))
    try:
        trajectory = critic.encode_trajectory(ids, lengths, tokenizer_id=critic.tokenizer_id)
    finally:
        handle.remove()
    assert calls == [1]
    prefixes = [dict(token_ids=ids[:n], tokenizer_id=critic.tokenizer_id) for n in lengths]
    np.testing.assert_allclose(trajectory, critic.encode_batch(prefixes), rtol=1e-5, atol=1e-6)
    assert not lm.training and not critic.encoder.training
    if pytest.importorskip('math_verify'):
        from importlib.util import find_spec
        if find_spec('datasets') is None:
            pytest.skip('Install the data extra for the managed smoke check')
        from slice_probe import preflight
        from slice_probe.requests import TrainingRequest
        from slice_probe.registry import canonical_hash
        monkeypatch.setattr(preflight, 'load_model', lambda *a, **kw: (
            critic.encoder, critic.tokenizer, identity))
        request = TrainingRequest.parse(dict(dataset='owner/test', layer_base=1,
            num_slices=3, max_new_tokens=16))
        bundle = dict(request_hash=canonical_hash(request.to_dict()))
        preflight.smoke(tmp_path / 'smoke', request, bundle, 'cpu', {})
        report = preflight.check_gate(tmp_path / 'smoke/report.json', bundle, {})
        assert report['layers'] == [3, 2, 1] and report['passed']
        with pytest.raises(ValueError, match='does not match'):
            preflight.check_gate(tmp_path / 'smoke/report.json', {'request_hash': 'changed'}, {})
