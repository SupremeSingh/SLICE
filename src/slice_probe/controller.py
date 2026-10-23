"""Initialize a native Duke controller on the current login host."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import sqlite3
import tempfile
import uuid

from .jobs import JobStore, now
from .slurm import settings, validate_settings
from .io import read_json, sha256, write_json
from .registry import canonical_hash


def init_cluster(path, *, state_dir=None, checkout=None, expected_host=None):
    """Create configuration once, with a private node-local SQLite ledger.

    /var/tmp is working storage, not a durable backup. The shared configuration
    pins the login host and requires the ledger to exist on subsequent runs.
    """
    if expected_host is not None and socket.gethostname() != expected_host:
        raise ValueError(f'Run initialization on {expected_host}, not {socket.gethostname()}')
    path = Path(path).absolute()
    if path.exists():
        return settings(path)
    checkout = Path(checkout or Path.cwd()).resolve()
    source = checkout / 'src'
    if not (source / 'slice_probe').is_dir():
        raise ValueError('Run init-cluster from the SLICE checkout containing src/slice_probe')
    state = Path(state_dir or f'/var/tmp/slice-controller-{os.getuid()}').absolute()
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    if state.is_symlink() or state.stat().st_uid != os.getuid():
        raise ValueError('Controller state directory must be owned by you and not a symlink')
    state.chmod(0o700)
    config = validate_settings(dict(
        database=str(state / 'jobs.sqlite'),
        registry=str(checkout.parent / 'slice-critics'),
        cluster_registry=str(checkout.parent / 'slice-critics'),
        cluster_root=str(checkout.parent / 'slice-service'),
        cluster_source=str(source), cluster_python=sys.executable,
        partition='compsci-gpu', account=None, gres='gpu:a5000:1',
        cpus=8, memory_gb=64, max_total_minutes=1440, device='auto',
        controller_host=socket.gethostname(), require_existing_database=True,
    ))
    JobStore(config['database'])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing configuration, even during concurrent setup.
    with path.open('x') as stream:
        stream.write(json.dumps(config, indent=2) + '\n')
    return config


def backup_ledger(config_path, out=None):
    """Publish a private immutable snapshot directory on shared backup storage."""
    config_path = Path(config_path).absolute()
    config = settings(config_path)
    out = Path(out) if out is not None else (
        config_path.parent / 'controller-backups' / uuid.uuid4().hex)
    out = out.absolute()
    if out.exists():
        raise FileExistsError(f'Backup already exists: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ledger-', dir=out.parent) as temporary:
        staging = Path(temporary) / 'snapshot'
        staging.mkdir(mode=0o700)
        JobStore(config['database']).backup(staging / 'ledger.sqlite')
        write_json(staging / 'config.json', config)
        write_json(staging / 'manifest.json', dict(
            created=now(),
            controller_host=config.get('controller_host', socket.gethostname()),
            config_sha256=canonical_hash(config), ledger_sha256=sha256(staging / 'ledger.sqlite')))
        os.rename(staging, out)
    return dict(backup_path=str(out), controller_host=config.get('controller_host'))


def restore_ledger(config_path, backup):
    """Restore only a missing ledger on its original host, failing closed on gaps.

    Requests staged after the backup must not disappear from the budget/idempotency
    history. Their shared bundles cause refusal. Known staged nonterminal requests
    become submission_unknown and must be refreshed against Slurm before new work.
    No sbatch is executed, and an existing ledger is never overwritten.
    """
    config = validate_settings(read_json(config_path))
    host = config.get('controller_host')
    if host is None or socket.gethostname() != host:
        raise ValueError(f'Restore requires the original controller host: {host}')
    database, backup = Path(config['database']), Path(backup)
    if database.exists():
        raise FileExistsError('Refusing to overwrite an existing controller ledger')
    manifest = read_json(backup / 'manifest.json')
    if (manifest['controller_host'] != host or manifest['config_sha256'] != canonical_hash(config)
            or manifest['ledger_sha256'] != sha256(backup / 'ledger.sqlite')
            or read_json(backup / 'config.json') != config):
        raise ValueError('Backup identity or checksum does not match this controller')
    # Inspect the backup read-only; do not create or migrate a database in it.
    source = sqlite3.connect((backup / 'ledger.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    try:
        if source.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Backup integrity check failed')
        source.row_factory = sqlite3.Row
        jobs = {row['id']: JobStore.decode(row) for row in source.execute('SELECT * FROM jobs')}
        staged = []
        for path in (Path(config['cluster_root']) / 'jobs').glob('*/request.json'):
            bundle = read_json(path)
            job = jobs.get(bundle['job_id'])
            if (job is None or job['request_hash'] != bundle['request_hash']
                    or job['project'] != bundle['project']):
                raise ValueError('Backup predates a staged request; recover a newer ledger. '
                                 'Do not submit or initialize another controller')
            staged.append(job)
        database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix='.restore-', dir=database.parent) as temporary:
            candidate = Path(temporary) / 'ledger.sqlite'
            target = sqlite3.connect(candidate)
            try:
                source.backup(target)
            finally:
                target.close()
            store = JobStore(candidate)
            for job in staged:
                if job['status'] not in ('completed', 'failed', 'cancelled'):
                    store.update(job['id'], 'submission_unknown', detail=dict(
                        job['detail'], reserved=True, recovery='Refresh against Slurm before new submission'))
            # Exclusive publication: never replace a ledger created concurrently.
            os.link(candidate, database)
    finally:
        source.close()
    return dict(database=str(database), restored=True,
                reconcile=[job['id'] for job in staged
                           if job['status'] not in ('completed', 'failed', 'cancelled')])
