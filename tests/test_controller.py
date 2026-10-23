"""Consistent ledger snapshots and conservative host-bound recovery."""
import json
from pathlib import Path
import socket
import sqlite3
import sys

import pytest

from slice_probe.controller import backup_ledger, init_cluster, restore_ledger
from slice_probe.io import read_json, write_json
from slice_probe.jobs import JobStore
from slice_probe.slurm import Slurm


@pytest.fixture
def controller(tmp_path):
    checkout = tmp_path / 'SLICE'
    (checkout / 'src/slice_probe').mkdir(parents=True)
    path = checkout / 'artifacts/service.json'
    cfg = init_cluster(path, state_dir=tmp_path / 'node-local', checkout=checkout)
    return path, cfg, JobStore(cfg['database'])


def stage_request(cfg, job):
    write_json(Path(cfg['cluster_root']) / 'jobs' / job['id'] / 'request.json',
               dict(job_id=job['id'], request_hash=job['request_hash'], project=job['project']))


def test_backup_includes_wal_and_restore_requires_reconciliation(controller, tmp_path):
    path, cfg, store = controller
    live = sqlite3.connect(store.path)
    try:
        live.execute('PRAGMA journal_mode=WAL')
        job = store.create('project', 'stable-key', dict(dataset='owner/data'))
        stage_request(cfg, job)
        snapshot = tmp_path / 'backup'
        backup_ledger(path, snapshot)
        # The snapshot is readable without access to the source's WAL files.
        with sqlite3.connect(snapshot / 'ledger.sqlite') as db:
            assert db.execute('SELECT id FROM jobs').fetchone()[0] == job['id']
        with pytest.raises(FileExistsError):
            backup_ledger(path, snapshot)
    finally:
        live.close()
    Path(store.path).unlink()
    result = restore_ledger(path, snapshot)
    assert result['reconcile'] == [job['id']]
    recovered = JobStore(store.path)
    restored = recovered.get(job['id'])
    assert restored['status'] == 'submission_unknown' and restored['detail']['reserved']
    assert recovered.create('project', 'stable-key', dict(dataset='owner/data'))['id'] == job['id']
    with pytest.raises(ValueError, match='active'):
        other = recovered.create('project', 'new-key', dict(dataset='owner/data'))
        recovered.reserve(other['id'], max_total_minutes=1440)
    with pytest.raises(FileExistsError, match='overwrite'):
        restore_ledger(path, snapshot)


def test_restore_refuses_snapshot_missing_a_staged_request(controller, tmp_path):
    path, cfg, store = controller
    snapshot = tmp_path / 'backup'
    backup_ledger(path, snapshot)
    job = store.create('project', 'later', dict(dataset='owner/data'))
    stage_request(cfg, job)
    Path(store.path).unlink()
    with pytest.raises(ValueError, match='predates'):
        restore_ledger(path, snapshot)
    assert not Path(store.path).exists()


def test_restore_rejects_corrupt_snapshot_and_wrong_host(controller, tmp_path, monkeypatch):
    path, cfg, store = controller
    snapshot = tmp_path / 'backup'
    backup_ledger(path, snapshot)
    Path(store.path).unlink()
    actual = socket.gethostname()
    monkeypatch.setattr(socket, 'gethostname', lambda: 'other-host')
    with pytest.raises(ValueError, match='original controller host'):
        restore_ledger(path, snapshot)
    monkeypatch.setattr(socket, 'gethostname', lambda: actual)
    with (snapshot / 'ledger.sqlite').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        restore_ledger(path, snapshot)
    assert not Path(store.path).exists()


def test_init_checks_host_before_creating_state(tmp_path):
    with pytest.raises(ValueError, match='Run initialization on'):
        init_cluster(tmp_path / 'service.json', expected_host='not-this-host')
    assert not (tmp_path / 'service.json').exists()


def test_cli_backs_up_identity_before_submission(controller, monkeypatch, capsys):
    from slice_probe.cli import main

    path, cfg, store = controller
    request = path.parent / 'request.json'
    write_json(request, dict(dataset='owner/data'))

    def command(self, args, **kwargs):
        if args[0] == 'sbatch':
            snapshots = list((path.parent / 'controller-backups').glob('*/ledger.sqlite'))
            assert snapshots
            with sqlite3.connect(snapshots[0]) as db:
                assert db.execute('SELECT idempotency_key FROM jobs').fetchone()[0] == 'pilot'
            return '1234\n'
        return ''

    monkeypatch.setattr(Slurm, 'command', command)
    monkeypatch.setattr(sys, 'argv', ['slice-probe', 'jobs', '--config', str(path),
        'submit', '--request', str(request), '--project', 'project', '--idempotency-key', 'pilot'])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'queued'
    snapshots = list((path.parent / 'controller-backups').glob('*/ledger.sqlite'))
    assert len(snapshots) == 2
    statuses = []
    for snapshot in snapshots:
        with sqlite3.connect(snapshot) as db:
            statuses.append(db.execute('SELECT status FROM jobs').fetchone()[0])
    assert set(statuses) == {'awaiting_approval', 'queued'}
