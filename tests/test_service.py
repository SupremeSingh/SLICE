"""Private API ownership, approval, idempotency and ambiguous submission recovery."""
import json
import subprocess

import pytest

from slice_probe.datasets import select_questions
from slice_probe.io import write_json
from slice_probe.jobs import JobStore
from slice_probe.requests import TrainingRequest
from slice_probe.slurm import Slurm
from slice_probe.slurm import validate_settings


@pytest.fixture
def config(tmp_path):
    value = dict(database=str(tmp_path / 'jobs.sqlite'), registry=str(tmp_path / 'registry'),
                 cluster_root='/cluster/slice-service', cluster_python='/cluster/venv/bin/python',
                 partition='gpu', account='research', cpus=2, memory_gb=8, max_total_minutes=500)
    path = tmp_path / 'service.json'
    write_json(path, value)
    return path, value


def test_request_bounds_and_family_splits():
    request = TrainingRequest.parse(dict(dataset='owner/data', max_questions=10,
                                         layer_base=3, num_slices=3))
    assert request.preview()['layers'] == [28, 26, 20]
    for extra in (dict(num_slices=5, layer_base=3), dict(head='ppo'),
                  dict(max_questions=1001), dict(discount_gamma=.9), dict(layer_base=True)):
        with pytest.raises(ValueError):
            TrainingRequest.parse(dict(dataset='owner/data', **extra))
    rows = [dict(problem=f'{word} has 2 apples.', answer='2') for word in
            ('Alice', 'Bob', 'Charlie', 'Diana', 'Ed', 'Fran', 'George', 'Helen', 'Ian', 'Jane')]
    rows.insert(1, dict(problem='Alice has 7 apples.', answer='7'))
    a, excluded = select_questions(rows, request)
    b, _ = select_questions(rows, request)
    assert a == b and excluded['duplicate_family'] == 1
    assert {row['split'] for row in a} == {'train', 'val', 'test'}
    assert len({row['id'] for row in a}) == 10


def test_idempotency_atomic_reservation_and_budget(config):
    path, cfg = config
    store = JobStore(cfg['database'])
    value = dict(dataset='owner/data')
    first = store.create('project', 'same', value)
    assert store.create('project', 'same', value)['id'] == first['id']
    with pytest.raises(ValueError, match='another request'):
        store.create('project', 'same', dict(value, num_slices=3))
    second = store.create('other', 'same', value)
    store.reserve(first['id'], max_total_minutes=300)
    with pytest.raises(ValueError, match='active'):
        store.reserve(second['id'], max_total_minutes=300)
    store.update(first['id'], 'completed', detail={'reserved': True})
    with pytest.raises(ValueError, match='allowance'):
        store.reserve(second['id'], max_total_minutes=300)


def test_submission_timeout_is_not_retried_and_can_be_reconciled(config, monkeypatch):
    _, cfg = config
    store, scheduler = JobStore(cfg['database']), Slurm(cfg)
    job = store.create('project', 'request-1', dict(dataset='owner/data'))
    calls = []

    def command(args, **kw):
        calls.append(args)
        if args[0] == 'sbatch':
            assert '--gres=gpu:1' in args and f'--job-name=slice-{job["id"]}' in args
            raise subprocess.TimeoutExpired('ssh', 30)
        return ''

    monkeypatch.setattr(scheduler, 'command', command)
    assert scheduler.approve(store, job['id'])['status'] == 'submission_unknown'
    with pytest.raises(ValueError, match='reconcile'):
        scheduler.approve(store, job['id'])
    assert sum(c[0] == 'sbatch' for c in calls) == 1

    def reconciled(args, **kw):
        if args[0] == 'squeue':
            return '1234|RUNNING\n'
        if args[0] == 'sacct':
            return '1234|PENDING|slice-' + job['id'] + '\n'
        if args[0] == 'cat':
            return json.dumps(dict(job_id=job['id'], request_hash=job['request_hash'],
                                   stage='features', status='running'))
        return ''

    monkeypatch.setattr(scheduler, 'command', reconciled)
    result = scheduler.refresh(store, job['id'])
    assert result['scheduler_id'] == '1234' and result['status'] == 'running'
    assert result['detail']['worker']['stage'] == 'features'
    assert scheduler.cancel(store, job['id'])['status'] == 'cancelling'


def test_direct_submit_reuses_attempts_and_preserves_limits(config, monkeypatch):
    _, cfg = config
    store, scheduler = JobStore(cfg['database']), Slurm(cfg)
    submitted = []

    def command(args, **kwargs):
        if args[0] == 'sbatch':
            submitted.append(args)
            raise subprocess.TimeoutExpired('sbatch', 30)
        return ''

    monkeypatch.setattr(scheduler, 'command', command)
    request = dict(dataset='owner/data')
    first = scheduler.submit(store, 'project', 'pilot', request)
    assert first['status'] == 'submission_unknown'
    assert scheduler.submit(store, 'project', 'pilot', request) == first
    assert len(submitted) == 1
    with pytest.raises(ValueError, match='another request'):
        scheduler.submit(store, 'project', 'pilot', dict(request, num_slices=3))
    with pytest.raises(ValueError, match='active'):
        scheduler.submit(store, 'project', 'second', request)
    store.update(first['id'], 'completed', detail={'reserved': True})
    scheduler.config['max_total_minutes'] = first['request']['wall_minutes']
    with pytest.raises(ValueError, match='allowance'):
        scheduler.submit(store, 'project', 'second', request)
    assert len(submitted) == 1


def test_concurrent_direct_submit_calls_sbatch_once(config, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    _, cfg = config
    store, scheduler = JobStore(cfg['database']), Slurm(cfg)
    barrier, submitted = Barrier(2), []
    create = store.create

    def simultaneous_create(*args):
        job = create(*args)
        barrier.wait(timeout=10)
        return job

    def command(args, **kwargs):
        if args[0] == 'sbatch':
            submitted.append(args)
            return '1234\n'
        return ''

    monkeypatch.setattr(store, 'create', simultaneous_create)
    monkeypatch.setattr(scheduler, 'command', command)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: scheduler.submit(
            store, 'project', 'same', dict(dataset='owner/data')), range(2)))
    assert results[0]['id'] == results[1]['id']
    assert len(submitted) == 1
    assert store.get(results[0]['id'])['scheduler_id'] == '1234'


def test_cluster_init_pins_host_and_refuses_missing_ledger(tmp_path, monkeypatch):
    from slice_probe.controller import init_cluster
    from slice_probe.slurm import settings
    import socket

    checkout = tmp_path / 'SLICE'
    (checkout / 'src/slice_probe').mkdir(parents=True)
    path = checkout / 'artifacts/cluster-service.json'
    state = tmp_path / 'node-local'
    cfg = init_cluster(path, state_dir=state, checkout=checkout)
    assert not cfg.get('ssh_host')
    assert cfg['controller_host'] == socket.gethostname()
    assert cfg['cluster_source'] == str(checkout / 'src')
    assert cfg['gres'] == 'gpu:a5000:1'
    assert state.stat().st_mode & 0o777 == 0o700
    store = JobStore(cfg['database'])
    job = store.create('project', 'pilot', dict(dataset='owner/data'))
    assert init_cluster(path, checkout=checkout) == cfg
    assert store.get(job['id'])['status'] == 'awaiting_approval'
    monkeypatch.setattr(socket, 'gethostname', lambda: 'another-login-host')
    with pytest.raises(ValueError, match='Reconnect'):
        settings(path)
    monkeypatch.setattr(socket, 'gethostname', lambda: cfg['controller_host'])
    from pathlib import Path
    Path(cfg['database']).unlink()
    with pytest.raises(ValueError, match='ledger is missing'):
        init_cluster(path, checkout=checkout)


def test_direct_submit_cli_dispatches_without_ssh(config, monkeypatch, capsys):
    from slice_probe.cli import main
    import sys

    path, cfg = config
    request_path = path.parent / 'request.json'
    write_json(request_path, dict(dataset='owner/data'))
    submitted = []

    def command(self, args, **kwargs):
        if args[0] == 'sbatch':
            submitted.append(args)
            return '5678\n'
        return ''

    monkeypatch.setattr(Slurm, 'command', command)
    monkeypatch.setattr(sys, 'argv', ['slice-probe', 'jobs', '--config', str(path),
        'submit', '--request', str(request_path), '--project', 'project',
        '--idempotency-key', 'pilot'])
    main()
    first = json.loads(capsys.readouterr().out)
    main()
    assert json.loads(capsys.readouterr().out) == first
    assert first['scheduler_id'] == '5678' and first['status'] == 'queued'
    assert len(submitted) == 1


def test_api_requests_require_auth_and_cannot_approve_or_cross_projects(config):
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    from fastapi.testclient import TestClient
    from slice_probe.api import create_app

    path, _ = config
    app = create_app(path, tokens={'a': 'a' * 32, 'b': 'b' * 32})
    with TestClient(app) as client:
        assert client.post('/v1/training-jobs', json={}).status_code == 401
        headers = {'Authorization': 'Bearer ' + 'a' * 32, 'Idempotency-Key': 'job-1'}
        first = client.post('/v1/training-jobs', headers=headers, json={'dataset': 'owner/data'})
        assert first.status_code == 202, first.text
        job = first.json()
        assert job['status'] == 'awaiting_approval'
        assert client.post('/v1/training-jobs', headers=headers,
                           json={'dataset': 'owner/data'}).json()['id'] == job['id']
        other = {'Authorization': 'Bearer ' + 'b' * 32}
        assert client.get('/v1/jobs/' + job['id'], headers=other).status_code == 404
        assert client.post('/v1/jobs/' + job['id'] + '/approve', headers=headers).status_code == 404
        assert client.post('/v1/jobs/' + job['id'] + '/cancel', headers=headers).json()['status'] == 'cancelled'
        assert client.post('/v1/jobs/' + job['id'] + '/cancel', headers=headers).json()['status'] == 'cancelled'
        assert client.get('/v1/usage', headers=headers).json()['reserved_gpu_minutes'] == 0


def test_stage_recovery_verifies_completed_outputs_and_retries_partial_work(tmp_path):
    from slice_probe.worker import stage

    calls = []

    def finish(out):
        calls.append('run')
        write_json(out / 'value.json', {'answer': 42})

    path = stage(tmp_path, 'features', finish)
    assert stage(tmp_path, 'features', finish) == path and calls == ['run']
    write_json(path / 'value.json', {'answer': 0})
    with pytest.raises(ValueError, match='changed'):
        stage(tmp_path, 'features', finish)
    write_json(tmp_path / '.fitted-working/partial.json', {})
    stage(tmp_path, 'fitted', finish)
    assert not (tmp_path / 'fitted/partial.json').exists()


def test_cs_cluster_plan_uses_typed_gpu_container_snapshot_and_exact_arguments(config):
    import shlex

    _, cfg = config
    cfg.update(partition='compsci-gpu', account=None, gres='gpu:a5000:1',
               container_runtime='singularity', container_image='/cluster/images/pinned image.sif',
               container_binds=['/usr/xtmp/ms785:/usr/xtmp/ms785'],
               cluster_source='/usr/xtmp/ms785/SLICE/src')
    store = JobStore(cfg['database'])
    job = store.create('project', 'plan', dict(dataset='owner/data'))
    plan = Slurm(cfg).plan(job)
    assert '--gres=gpu:a5000:1' in plan['sbatch']
    assert '--partition=compsci-gpu' in plan['sbatch']
    assert not any(arg.startswith('--account=') for arg in plan['sbatch'])
    assert f'--chdir={plan["directory"]}' in plan['sbatch']
    command = shlex.split(plan['script'].splitlines()[-1])
    assert command[:5] == ['exec', 'srun', '--ntasks=1', 'singularity', 'exec']
    assert '--nv' in command and '/cluster/images/pinned image.sif' in command
    assert 'PYTHONPATH=' + plan['directory'] + '/source' in command
    assert 'SLICE_NUM_THREADS=2' in command
    assert store.get(job['id'])['status'] == 'awaiting_approval'
    with pytest.raises(ValueError, match='one GPU'):
        validate_settings(dict(cfg, gres='gpu:a5000:2'))
    with pytest.raises(ValueError, match='together'):
        validate_settings(dict(cfg, container_image=None))


def test_native_cs_profile_launches_project_python_without_container(config):
    from pathlib import Path
    import shlex
    from slice_probe.io import read_json

    _, local = config
    profile = Path(__file__).resolve().parents[1] / 'configs/service.duke-cs.native.example.json'
    cfg = read_json(profile)
    cfg['database'] = local['database']
    store = JobStore(cfg['database'])
    job = store.create('project', 'native', dict(dataset='owner/data'))
    plan = Slurm(cfg).plan(job)
    command = shlex.split(plan['script'].splitlines()[-1])
    assert command[:4] == ['exec', 'srun', '--ntasks=1', 'env']
    assert '/usr/xtmp/ms785/SLICE/.venv-native/bin/python' in command
    assert not {'apptainer', 'singularity', '--nv', '--bind'} & set(command)
    assert '--gres=gpu:a5000:1' in plan['sbatch']
    assert plan['bundle']['execution']['container_image'] is None


def test_source_snapshot_and_logs_are_prepared_before_submission(config, monkeypatch):
    _, cfg = config
    cfg.update(cluster_source='/cluster/SLICE/src', gres='gpu:a5000:1')
    store, scheduler = JobStore(cfg['database']), Slurm(cfg)
    job = store.create('project', 'snapshot', dict(dataset='owner/data'))
    commands = []

    def command(args, **kw):
        commands.append(args)
        return '9876' if args[0] == 'sbatch' else ''

    monkeypatch.setattr(scheduler, 'command', command)
    result = scheduler.approve(store, job['id'])
    assert result['scheduler_id'] == '9876' and result['status'] == 'queued'
    assert commands[0][0] == 'mkdir' and commands[-1][0] == 'sbatch'
    archive = next(args for args in commands if args[0] == 'tar' and '-cf' in args)
    assert '--exclude=__pycache__' in archive and '--exclude=*.pyc' in archive
    assert result['detail']['execution']['gres'] == 'gpu:a5000:1'


@pytest.mark.parametrize('exit_code,expected', [('0:0', 'completed'), ('1:0', 'failed'), ('', 'completion_unknown')])
def test_completion_requires_scheduler_exit_code_and_worker_record(config, monkeypatch, exit_code, expected):
    _, cfg = config
    store, scheduler = JobStore(cfg['database']), Slurm(cfg)
    job = store.create('project', 'done', dict(dataset='owner/data'))
    store.update(job['id'], 'queued', scheduler_id='1234', detail={'reserved': True})

    def command(args, **kw):
        if args[0] == 'squeue':
            return ''
        if args[0] == 'sacct':
            return f'1234|COMPLETED|slice-{job["id"]}|{exit_code}|00:01:00\n'
        return json.dumps(dict(job_id=job['id'], request_hash=job['request_hash'], status='completed'))

    monkeypatch.setattr(scheduler, 'command', command)
    assert scheduler.refresh(store, job['id'])['status'] == expected
