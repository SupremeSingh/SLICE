"""Exercise the actual supervised fitting pipeline across a worker restart."""
import shutil

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

    def prepare(request, out):
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
    assert manifest['head']['method'] == head
    assert manifest['features']['layers'] == [2]
    assert manifest['metadata']['project'] == 'example'
    assert set(status['evaluation']['baselines']) >= {'constant', 'final/ridge'}
    run_job(root / 'request.json', device='cpu')
    assert collected == ['collected']


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
