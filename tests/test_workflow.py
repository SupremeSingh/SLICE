import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from slice_probe import collection, features, training
from slice_probe.data import family, load_questions, number, reference
from slice_probe.demo import run
from slice_probe.io import load_checkpoint, save_checkpoint, sha256, write_json
from slice_probe.metrics import auc, metrics
from slice_probe.model import Bottleneck, objective, predict_features


class CausalBlock(nn.Module):
    def forward(self, x):
        return x.cumsum(dim=1) / 4


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([CausalBlock(), CausalBlock()])
        self.config = SimpleNamespace(rms_norm_eps=1e-6, hidden_size=3, max_position_embeddings=128)

    def forward(self, input_ids, **kwargs):
        x = input_ids.float().unsqueeze(-1).repeat(1,1,3)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(last_hidden_state=x)


def test_features_are_causal_at_question_and_partial_prefixes():
    model = ToyBackbone()
    ids = torch.tensor([[2,3,4,5,6,7]])
    full = features.capture(model, ids, [2,4], [2,1])
    for i,stop in enumerate((3,5)):
        prefix = features.capture(model, ids[:,:stop], [stop-1], [2,1])
        for layer in (1,2):
            torch.testing.assert_close(full[layer][i], prefix[layer][0])
    assert features.concatenate(full, [2,1]).shape == (2,6)
    assert all(not block._forward_hooks for block in model.layers)
    with pytest.raises(ValueError): features.capture(model, ids, [6], [1])


def test_sample_positions_and_geometric_layers():
    assert features.geometric_layers(28) == [28,27,25,21,13]
    assert features.sample_positions(1,0,0) == [0]*8
    p = features.sample_positions(15,2,3)
    assert p[0] == 0 and all(1<=v<15 for v in p[1:])
    assert p == features.sample_positions(15,2,3)


def test_ridge_raw_statistics_and_regularization():
    z = torch.tensor([[1.,2.],[3.,1.],[2.,5.]])
    y = np.array([1.,0.,1.])
    w,checks = training.fit_ridge(z,y,.01)
    phi = torch.cat((z.double(),torch.ones(3,1,dtype=torch.float64)),dim=1)
    torch.testing.assert_close(w,torch.linalg.solve(phi.T@phi+.01*torch.eye(3),phi.T@torch.tensor(y)))
    assert checks['relative_residual']<1e-8
    with pytest.raises(ValueError):training.fit_ridge(z,y,0)
    with pytest.raises(ValueError):training.fit_ridge(z,np.array([1.,2.,1.]),.01)


def test_projection_penalty_and_checkpoint_integrity(tmp_path):
    torch.manual_seed(3)
    model = Bottleneck(6,2)
    x,y = torch.randn(8,6),torch.tensor([0.,1.]*4)
    plain,sparse = objective(model,x,y,0),objective(model,x,y,.01)
    torch.testing.assert_close(sparse-plain,.01*model.features(x).mean())
    record=dict(format='slice-probe-v1',input_dim=6,width=2,state=model.state_dict(),
                mean=torch.zeros(6),scale=torch.ones(6),weights=torch.ones(3,dtype=torch.float64))
    path=tmp_path/'probe.pt';save_checkpoint(path,record)
    scores=predict_features(load_checkpoint(path),x.numpy())
    assert scores.shape==(8,) and np.isfinite(scores).all()
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed'):load_checkpoint(path)


def test_numeric_reference_and_family_guard(tmp_path):
    assert number('1,234.50')==number('1234.5')
    assert reference('Compute the result.',r'Answer: \boxed{3}','3')=='3'
    assert reference('Find all roots.',r'\boxed{3}','3') is None
    assert reference('Compute.',r'\boxed{3} and \boxed{4}','3') is None
    assert family('There are 5 pears.')==family('There are 17 pears.')
    rows=[dict(id=str(i),question=f'Unique topic {i}',reference='3',split=split)
          for i,split in enumerate(('train','val','test'))]
    path=tmp_path/'questions.jsonl';path.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match='family'):load_questions(path)
    example=Path(__file__).resolve().parents[1]/'examples/questions.jsonl'
    assert len(load_questions(example))==7


def test_metrics_ties_and_nonbinary_labels():
    assert auc(np.array([0,1]),np.array([.5,.5]))==.5
    assert metrics(np.array([0,1]),np.array([-1.,2.]))['outside_probability_range']==1
    with pytest.raises(ValueError):metrics(np.array([0,2]),np.array([.1,.9]))


def answer_fixture(tmp_path):
    root=tmp_path/'answers';root.mkdir()
    model=dict(id='toy',revision=None,hashes={})
    write_json(root/'manifest.json',dict(model=model))
    rows=[dict(id=f'q{i}',split=split,question=question,reference='1')
          for i,(split,question) in enumerate([('train','A sum?'),('val','An area?'),('test','A distance?')])]
    write_json(root/'questions.json',rows)
    for i,q in enumerate(rows):
        write_json(root/'answers'/f'{i:05d}.json',dict(id=q['id'],prompt_ids=[1,2,3],responses=[
            dict(status='correct',score=1,token_ids=[4,5],capped=False),
            dict(status='incorrect',score=0,token_ids=[6,7,8],capped=False)]))
    return root


def test_extraction_and_all_token_readout_without_real_model(tmp_path,monkeypatch):
    answers=answer_fixture(tmp_path)
    monkeypatch.setattr(collection,'load_model',lambda *a,**k:(ToyBackbone(),None,{}))
    destination=tmp_path/'features'
    collection.extract(answers,destination,'cpu')
    data,m=collection.load_split(destination,'train')
    assert data['x'].shape==(16,6) and set(data['y'])=={0.,1.}
    assert (data['position'][::8]==0).all()
    monkeypatch.setattr(training,'load_model',lambda *a,**k:(ToyBackbone(),None,{}))
    model=Bottleneck(6,2)
    record=dict(input_dim=6,width=2,state=model.state_dict(),mean=torch.zeros(6),scale=torch.ones(6))
    weights,checks=training.all_token_ridge(record,m,answers,'cpu',.01)
    assert weights.shape==(3,) and checks['training_prefixes']==5
    (answers/'answers/00000.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):training.all_token_ridge(record,m,answers,'cpu',.01)


def test_complete_training_evaluation_and_frozen_selection(tmp_path):
    run(tmp_path/'demo')
    report=json.loads((tmp_path/'demo/evaluation/summary.json').read_text())
    assert set(report['results'])=={'constant','plain','sparse'}
    assert report['test_questions']==12 and report['test_prefixes']==96
    assert report['by_prefix']['question_only']['prefixes']==12
    for kind in ('plain','sparse'):
        checkpoint=load_checkpoint(tmp_path/f'demo/probes/{kind}-seed7.pt')
        assert checkpoint['weights'].shape==(9,)
        assert checkpoint['ridge_states']=='sampled'
    # Test outcomes cannot affect fitting: changing a test file only changes its hash.
    mpath=tmp_path/'demo/features/manifest.json';m=json.loads(mpath.read_text())
    test=tmp_path/'demo/features/test-y.npy';np.save(test,1-np.load(test))
    m['files'][test.name]=sha256(test);write_json(mpath,m)
    training.fit(tmp_path/'demo/features',tmp_path/'refit',width=8,seeds=[7],
                 learning_rates=[.01],epochs=4,batch_size=64)
    a=load_checkpoint(tmp_path/'demo/probes/sparse-seed7.pt')
    b=load_checkpoint(tmp_path/'refit/sparse-seed7.pt')
    torch.testing.assert_close(a['weights'],b['weights'],rtol=0,atol=0)
    for key in a['state']:torch.testing.assert_close(a['state'][key],b['state'][key],rtol=0,atol=0)


def test_standalone_copy_has_no_parent_imports(tmp_path):
    project=Path(__file__).resolve().parents[1]
    copied=tmp_path/'independent';shutil.copytree(project,copied,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.egg-info'))
    env=dict(os.environ,PYTHONPATH=str(copied/'src'))
    result=subprocess.run([sys.executable,'-m','slice_probe','--help'],cwd=tmp_path,env=env,text=True,capture_output=True,check=True)
    assert 'prepare-numina' in result.stdout and 'predict' in result.stdout
    for path in (copied/'src').rglob('*.py'):
        assert 'math_rl' not in path.read_text()


def test_configurable_layer_selection():
    assert features.geometric_layers(28, 3, 3) == [28, 26, 20]
    assert features.geometric_layers(28, 1, 5) == [28, 27, 26, 25, 24]
    assert features.geometric_layers(28, 1.5, 5) == [28, 27, 26, 25, 24]
    assert features.geometric_layers(2, 2, 1) == [2]
    for base in (float('nan'), float('inf'), .5):
        with pytest.raises(ValueError): features.geometric_layers(28, base)
    for count in (0, -1, 29):
        with pytest.raises(ValueError): features.geometric_layers(28, 2, count)
    with pytest.raises(ValueError, match='supplies only'):
        features.geometric_layers(28, 3, 5)


def test_custom_layers_reach_checkpoint(tmp_path, monkeypatch):
    answers = answer_fixture(tmp_path)
    monkeypatch.setattr(collection, 'load_model', lambda *a, **k: (ToyBackbone(), None, {}))
    collection.extract(answers, tmp_path/'features', 'cpu', layer_base=1, num_slices=1)
    data, manifest = collection.load_split(tmp_path/'features', 'train')
    assert data['x'].shape == (16, 3)
    assert manifest['layers'] == [2]
    training.fit(tmp_path/'features', tmp_path/'probes', width=2, seeds=[7],
                 learning_rates=[.01], epochs=1, batch_size=16)
    probe = load_checkpoint(tmp_path/'probes/sparse-seed7.pt')
    assert probe['layers'] == [2] and probe['input_dim'] == 3
    assert probe['layer_selection']['base'] == 1
    assert predict_features(probe, data['x']).shape == (16,)


def test_bundled_real_text_sample_workflow(tmp_path, monkeypatch):
    from slice_probe import sample
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(char) % 32 for char in text]
    monkeypatch.setattr(sample, 'tokenizer_for', lambda *a: (
        Tokenizer(), dict(id='toy', revision=None, hashes={}), 4096))
    sample.prepare_sample(tmp_path/'answers')
    rows = json.loads((tmp_path/'answers/questions.json').read_text())
    assert len(rows) == 7 and len({features.prompt(q['question']) for q in rows}) == 7
    for split in ('train', 'val', 'test'):
        batches = list(collection.iter_answers(tmp_path/'answers', split))
        assert batches
        for _, _, batch, _ in batches:
            assert {r['score'] for r in batch['responses']} == {0, 1}
            assert all(r['label_source'] == 'authored-example' for r in batch['responses'])
    monkeypatch.setattr(collection, 'load_model', lambda *a, **k: (ToyBackbone(), None, {}))
    collection.extract(tmp_path/'answers', tmp_path/'features', 'cpu', num_slices=1)
    training.fit(tmp_path/'features', tmp_path/'probes', width=4, seeds=[7],
                 learning_rates=[.01], epochs=1, batch_size=16)
    from slice_probe.evaluation import evaluate
    evaluate(tmp_path/'features', tmp_path/'probes', tmp_path/'evaluation', 'cpu')
    assert (tmp_path/'evaluation/summary.json').exists()


def test_auto_device_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert features.resolve_device('auto') == 'cpu'
    assert features.resolve_device('cpu') == 'cpu'
    with pytest.raises(ValueError, match='CUDA is unavailable'):
        features.resolve_device('cuda:0')
