"""Exercise the actual supervised fitting pipeline across a worker restart."""
import shutil
from pathlib import Path

import pytest

from slice_probe import collection, comparison, datasets, preflight, data, training
from slice_probe.io import read_json, write_json
from slice_probe.registry import canonical_hash, inspect_artifact
from slice_probe.requests import TrainingRequest
from slice_probe.worker import run_job
from test_library import fitted


@pytest.mark.parametrize('head', ['ridge', 'lstd'])
def test_training_to_publication_resumes_without_duplicate_collection(fitted, tmp_path, monkeypatch, head):
    answers, _, _, _ = fitted
    request = TrainingRequest.parse(dict(dataset='owner/example', max_questions=10,
                                         num_slices=1, compression_width=2, epochs=1,
                                         seed=7, head=head, max_new_tokens=16))
    root = tmp_path / 'job'
    bundle = dict(job_id='a' * 32, project='example', request=request.to_dict(),
                  request_hash=canonical_hash(request.to_dict()), registry=str(tmp_path / 'registry'))
    write_json(root / 'request.json', bundle)
    collected = []

    def prepare(request, out, *, encoder_identity):
        assert encoder_identity['id'] == 'toy'
        out.mkdir()
        questions = read_json(answers / 'questions.json')
        import json
        (out / 'questions.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in questions))

    def collect(questions, out, *args):
        assert args[-3:] == (request.responses_per_question, request.max_new_tokens,
                            request.generation_seed)
        collected.append('collected')
        shutil.copytree(answers, out)

    monkeypatch.setattr(datasets, 'prepare_dataset', prepare)
    monkeypatch.setattr(collection, 'collect', collect)
    # This test supplies saved data and a toy encoder; real optional data-runtime
    # imports are exercised by test_integrations with the data extra installed.
    monkeypatch.setattr(preflight, 'check_data_runtime', lambda: None)
    monkeypatch.setattr(preflight, 'load_model', collection.load_model)
    monkeypatch.setattr(training, 'load_model', collection.load_model)
    monkeypatch.setattr(data, 'grade', lambda *args: dict(status='correct'))
    original = comparison.evaluate_comparison

    def interrupted(*args, **kwargs):
        raise RuntimeError('simulated worker interruption')

    monkeypatch.setattr(comparison, 'evaluate_comparison', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        run_job(root / 'request.json', device='cpu')
    assert read_json(root / 'status.json')['status'] == 'failed'
    assert not list((tmp_path / 'registry').glob('job-*'))
    monkeypatch.setattr(comparison, 'evaluate_comparison', original)
    run_job(root / 'request.json', device='cpu')
    assert collected == ['collected']
    if head == 'ridge':
        from slice_probe.parity import run_parity
        from slice_probe.io import sha256
        before = sha256(root / 'selected/ridge.pt')
        report = run_parity(root, tmp_path / 'standalone-parity', device='cpu')
        assert report['matches'] and report['selected_head']['matches']
        assert report['predictions']['matches']
        assert sha256(root / 'selected/ridge.pt') == before
    status = read_json(root / 'status.json')
    assert status['status'] == 'completed'
    assert read_json(root / 'comparison-config.json')['seeds'] == [7, 8, 9]
    if head == 'ridge':
        assert status['evaluation']['head_training_distribution'] == 'all-training-tokens'
    manifest = inspect_artifact(status['artifact_path'])
    assert set(status['artifact_paths']) == {'7', '8', '9'}
    assert set(status['evaluation']['raw_by_seed']) == {'7', '8', '9'}
    assert status['evaluation']['question_only_reference'] == 'canonical prefix-only features'
    for seed, path in status['artifact_paths'].items():
        assert inspect_artifact(path)['metadata']['head_seed'] == int(seed)
    assert manifest['head']['method'] == head
    from slice_probe.collection import load_split
    from slice_probe.io import load_checkpoint
    from slice_probe.metrics import metrics
    from slice_probe.model import predict_features
    published = load_checkpoint(Path(status['artifact_path']) / 'checkpoint.pt')
    val, _ = load_split(root / 'features', 'val')
    assert published['validation'] == metrics(val['y'], predict_features(published, val['x']))
    assert published['selection_validation_sampled_head']
    assert manifest['validation_binding'] == published['validation_binding']
    assert manifest['training_population'] == published['training_population']
    assert manifest['features']['layers'] == [2]
    assert manifest['metadata']['project'] == 'example'
    assert set(status['evaluation']['baselines']) >= {'constant', 'final/ridge'}
    run_job(root / 'request.json', device='cpu')
    assert collected == ['collected']
    if head == 'ridge':
        exercise_legacy_finalization(root, tmp_path, monkeypatch)


def exercise_legacy_finalization(root, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from slice_probe import finalization
    from slice_probe.io import sha256
    from slice_probe.worker import completed_files
    legacy = tmp_path / 'legacy-job'
    shutil.copytree(root, legacy)
    # Simulate the already-running source version: one final head, three fitted compressors.
    for seed in (8, 9):
        shutil.rmtree(legacy / f'selected/seed{seed}')
    (legacy / 'selected/heads.json').unlink()
    write_json(legacy / 'selected/.complete.json', dict(files=completed_files(legacy / 'selected')))
    before = {str(p.relative_to(legacy)):sha256(p) for p in legacy.rglob('*') if p.is_file()}
    out = tmp_path / 'finalization'
    finalization.prepare_finalization(legacy, out)
    monkeypatch.setattr(finalization.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='1234'))
    with pytest.raises(ValueError, match='still live'):
        finalization.finalize_job(out / 'plan.json')
    monkeypatch.setattr(finalization.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=''))
    original_fit = training.all_token_ridge
    fitted_seeds = []
    def tracked(record, *args):
        fitted_seeds.append(record['seed'])
        return original_fit(record, *args)
    monkeypatch.setattr(training, 'all_token_ridge', tracked)
    result = finalization.finalize_job(out / 'plan.json')
    assert result['status'] == 'completed'
    assert fitted_seeds == [8, 9]
    assert set(result['artifact_paths']) == {'7', '8', '9'}
    assert result['evaluation']['question_only_reference'] == 'canonical prefix-only features'
    assert read_json(out / 'selected/reuse.json')['checkpoint_sha256'] == sha256(legacy / 'selected/ridge.pt')
    assert {str(p.relative_to(legacy)):sha256(p) for p in legacy.rglob('*') if p.is_file()} == before
    finalization.finalize_job(out / 'plan.json')
    assert fitted_seeds == [8, 9]


def test_expired_budget_stops_before_starting_another_stage(tmp_path):
    from slice_probe.worker import BudgetExhausted

    request = TrainingRequest.parse(dict(dataset='owner/data'))
    write_json(tmp_path / 'request.json', dict(job_id='b' * 32,
        request=request.to_dict(), request_hash=canonical_hash(request.to_dict()),
        registry=str(tmp_path / 'registry')))
    write_json(tmp_path / 'deadline.json', dict(expires=0))
    with pytest.raises(BudgetExhausted):
        run_job(tmp_path / 'request.json', device='cpu')
    assert read_json(tmp_path / 'status.json')['status'] == 'failed'
    assert not (tmp_path / 'questions').exists()


def test_budget_watchdog_survives_verifier_alarm_cleanup(tmp_path, monkeypatch):
    import time
    pytest.importorskip('math_verify')
    from slice_probe.data import grade
    from slice_probe.worker import BudgetExhausted

    request = TrainingRequest.parse(dict(dataset='owner/data'))
    write_json(tmp_path / 'request.json', dict(job_id='c' * 32,
        request=request.to_dict(), request_hash=canonical_hash(request.to_dict()),
        registry=str(tmp_path / 'registry')))
    write_json(tmp_path / 'deadline.json', dict(expires=time.time() + .5))

    def smoke(*args):
        assert grade(r'The answer is \boxed{3}.', '3')['status'] == 'correct'
        time.sleep(2)

    monkeypatch.setattr(preflight, 'smoke', smoke)
    with pytest.raises(BudgetExhausted):
        run_job(tmp_path / 'request.json', device='cpu')
    assert read_json(tmp_path / 'status.json')['stage'] == 'smoke'
