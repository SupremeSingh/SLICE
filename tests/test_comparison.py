"""Scientific controls: matched states, no test fitting, and locked provenance."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from slice_probe import collection, comparison, evaluation
from slice_probe.calibration import fit_platt, apply_calibrator, risk_coverage
from slice_probe.diagnostics import question_carry_forward, within_question_ranking
from slice_probe.io import write_json,read_json,sha256,load_checkpoint
from test_workflow import ToyBackbone,answer_fixture


def matched_features(tmp_path,monkeypatch):
    answers=answer_fixture(tmp_path)
    model=ToyBackbone();model.norm=nn.Identity()
    monkeypatch.setattr(collection,'load_model',lambda *a,**k:(model,None,{}))
    features=tmp_path/'features'
    collection.extract(answers,features,'cpu',baselines=True,checkpoints=[0,1,2])
    return answers,features


def small_config(path,mode='exploratory'):
    cfg=dict(width=3,resnet_width=4,epochs=1,batch_size=16,seeds=[7],learning_rates=[.001],
             calibration='platt',mode=mode,test_scope='Toy data, not a scientific test')
    write_json(path,cfg)
    return path


def test_carry_forward_and_matched_question_ranking():
    # Two questions, each with one correct and one wrong continuation at lengths 0,1.
    q=np.repeat([0,1],4);r=np.tile([0,0,1,1],2);pos=np.tile([0,1,0,1],2)
    y=np.tile([1,1,0,0],2)
    p=np.array([.6,.9,.6,.1,.4,.8,.4,.2])
    initial=question_carry_forward(p,q,pos)
    np.testing.assert_allclose(initial[0],[.6]*4+[.4]*4)
    ranking=within_question_ranking(y,p,q,r,pos,repetitions=10)['by_token_count']
    assert ranking['0']['mean_auroc']==.5 and ranking['1']['mean_auroc']==1
    assert ranking['1']['mixed_questions']==2
    # Question difficulty cancels completely when scores never update.
    assert within_question_ranking(y,initial,q,r,pos,10)['by_token_count']['1']['mean_auroc']==.5
    bad=p.copy();bad[2]=.7
    with pytest.raises(ValueError,match='causality'):question_carry_forward(bad,q,pos)


def test_calibration_and_risk_ties():
    y=np.array([0,0,1,1]);p=np.array([-2.,-.4,.8,3.])
    c=fit_platt(p,y);prob=apply_calibrator(p,c)
    assert np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1))
    assert np.all(np.diff(prob)>=0)
    curve=risk_coverage(np.array([0,1,0,1]),np.full(4,.5))
    assert curve==[dict(threshold=.5,coverage=1.,risk=.5)]
    with pytest.raises(ValueError):fit_platt(p,np.ones(4))


def test_identical_states_for_all_representations(tmp_path,monkeypatch):
    _,features=matched_features(tmp_path,monkeypatch)
    common,m=collection.load_split(features,'train')
    for rep in ('final','two_thirds'):
        other,_=collection.load_split(features,'train',representation=rep)
        for key in ('y','question','response','position'):
            np.testing.assert_array_equal(common[key],other[key])
    anchor,_=collection.load_split(features,'test',anchors=True)
    assert set(anchor['position'])=={0,1,2}
    assert len(anchor['y'])==5  # length-2 and length-3 answers; terminal excluded.
    assert m['representations']['final']['convention']=='post-final learned model normalization'


def test_full_comparison_lock_and_release(tmp_path,monkeypatch):
    answers,features=matched_features(tmp_path,monkeypatch)
    cfg=small_config(tmp_path/'config.json',mode='confirmation')
    original=comparison.load_split
    def never_test(*args,**kwargs):
        assert args[1]!='test','Fitting attempted to read test states'
        return original(*args,**kwargs)
    monkeypatch.setattr(comparison,'load_split',never_test)
    original_numpy_load=np.load
    def no_test_file(path,*a,**kw):
        if isinstance(path,(str,Path)) and Path(path).parent==features and Path(path).name.startswith('test-'):
            raise AssertionError('Fitting read a test array directly')
        return original_numpy_load(path,*a,**kw)
    monkeypatch.setattr(np,'load',no_test_file)
    digest=comparison.fit_comparison(features,tmp_path/'fitted',cfg)
    monkeypatch.setattr(comparison,'load_split',original)
    monkeypatch.setattr(np,'load',original_numpy_load)
    with pytest.raises(ValueError,match='requires --lock-sha256'):
        comparison.evaluate_comparison(features,tmp_path/'fitted',tmp_path/'bad')
    result=comparison.evaluate_comparison(features,tmp_path/'fitted',tmp_path/'evaluation',lock_sha256=digest)
    assert set(result)=={'final/ridge','two_thirds/ridge','concat/ridge','concat/resnet10',
                         'plain/ridge','sparse/ridge','constant'}
    report=read_json(tmp_path/'evaluation/summary.json')
    assert report['lock_sha256']==digest and report['within_question']['sparse/ridge']['by_token_count']['0']['mean_auroc']==.5
    assert 'sparse-7' in report['calibrated_by_head']
    with np.load(tmp_path/'evaluation/predictions.npz') as data:
        assert 'response' in data and 'calibrated-sparse-7' in data
    comparison.release_bundle(answers,features,tmp_path/'fitted',tmp_path/'evaluation',tmp_path/'bundle')
    release=read_json(tmp_path/'bundle/release.json')
    assert not release['feature_arrays_included']
    for name,checksum in release['files'].items():assert sha256(tmp_path/'bundle'/name)==checksum
    assert (tmp_path/'bundle/collection/questions.json').exists()
    verified=comparison.verify_bundle(tmp_path/'bundle',tmp_path/'verified')
    assert set(verified)==set(result)
    independent = read_json(tmp_path/'verified/verified-metrics.json')
    assert independent['within_question']['sparse/ridge']['by_token_count']['0']['mean_auroc'] == .5
    assert independent['question_difficulty']['sparse/ridge'] == report['question_difficulty']['sparse/ridge']
    question_path = answers/'questions.json'
    original_questions = question_path.read_bytes()
    changed = read_json(question_path);changed[0]['id'] = 'changed-after-fitting'
    write_json(question_path, changed)
    with pytest.raises(ValueError, match='Question IDs/splits changed'):
        comparison.release_bundle(answers,features,tmp_path/'fitted',tmp_path/'evaluation',tmp_path/'bad-release')
    question_path.write_bytes(original_questions)
    (tmp_path/'fitted/config.json').write_text('{}')
    with pytest.raises(ValueError,match='Locked artifact changed'):
        comparison.evaluate_comparison(features,tmp_path/'fitted',tmp_path/'changed',lock_sha256=digest)


def test_standard_evaluation_rejects_training_manifest_substitution(tmp_path):
    from slice_probe.demo import run
    run(tmp_path/'demo')
    manifest=tmp_path/'demo/features/manifest.json';row=read_json(manifest)
    row['scope']='Changed metadata with all arrays untouched';write_json(manifest,row)
    with pytest.raises(ValueError,match='Training feature manifest'):
        evaluation.evaluate(tmp_path/'demo/features',tmp_path/'demo/probes',tmp_path/'invalid')
