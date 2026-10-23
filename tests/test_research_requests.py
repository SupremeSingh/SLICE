"""Explicit full-study data sizes and operator opt-in preserve pilot bounds."""
import pytest
from pathlib import Path

from slice_probe.datasets import prepare_dataset
from slice_probe.io import read_json
from slice_probe.requests import TrainingRequest
from slice_probe.slurm import Slurm


def research_request():
    return read_json(Path(__file__).resolve().parents[1] /
                     'configs/training.numina-10k.json')


def test_full_study_has_ten_thousand_training_questions_and_original_layers():
    request = TrainingRequest.parse(research_request())
    assert request.split_counts() == dict(train=10000, val=1000, test=2000)
    assert request.preview()['layers'] == [28, 27, 25, 21, 13]
    assert request.preview()['max_generated_answers'] == 104000
    assert request.training_seeds == (42, 43, 44)
    assert request.epochs == 30 and request.ridge_states == 'all-training-tokens'
    custom = TrainingRequest.parse(dict(research_request(), layer_base=3, num_slices=3))
    assert custom.preview()['layers'] == [28, 26, 20]


def test_large_request_requires_consistent_explicit_counts():
    for changes in (dict(train_questions=None), dict(test_questions=1999),
                    dict(validation_questions=True), dict(train_questions=0),
                    dict(max_questions=13001)):
        with pytest.raises(ValueError):
            TrainingRequest.parse(dict(research_request(), **changes))
    with pytest.raises(ValueError, match='explicit'):
        TrainingRequest.parse(dict(dataset='owner/data', max_questions=13000))


def test_numina_passes_exact_research_counts_to_original_selector(tmp_path, monkeypatch):
    from slice_probe import data
    calls = []
    monkeypatch.setattr(data, 'prepare_numina', lambda *args: calls.append(args))
    prepare_dataset(TrainingRequest.parse(research_request()), tmp_path / 'questions')
    assert calls == [(tmp_path / 'questions', dict(train=10000, val=1000, test=2000), None)]


def test_controller_requires_explicit_opt_in_before_planning_large_job():
    cfg = dict(database='/tmp/jobs.sqlite', registry='/cluster/critics',
        cluster_root='/cluster/jobs', cluster_python='/cluster/python', partition='gpu')
    job = dict(id='a' * 32, project='research', request_hash='b' * 64,
               request=TrainingRequest.parse(research_request()).to_dict())
    with pytest.raises(ValueError, match='explicitly raise'):
        Slurm(cfg).plan(job)
    plan = Slurm(dict(cfg, max_questions=13000, max_total_minutes=43200)).plan(job)
    assert '--time=1440' in plan['sbatch']
    assert plan['bundle']['request']['train_questions'] == 10000
