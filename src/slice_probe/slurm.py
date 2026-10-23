"""Operator-controlled Slurm submission, cancellation and reconciliation."""
from __future__ import annotations

import json
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import socket
from pathlib import Path
from datetime import datetime, timedelta

from .io import read_json
from .jobs import ACTIVE, TERMINAL


class SubmissionUnknown(RuntimeError):
    pass


def settings(path):
    config = validate_settings(read_json(path))
    if config.get('controller_host') and config['controller_host'] != socket.gethostname():
        raise ValueError(f"Reconnect to controller host {config['controller_host']}; "
                         'do not initialize a second ledger to retry a submission')
    if config.get('require_existing_database') and not Path(config['database']).is_file():
        raise ValueError('Controller ledger is missing; reconcile existing Slurm jobs and '
                         'recover the ledger before submitting more work')
    return config


def validate_settings(value):
    config = dict(value)
    required = {'database', 'registry', 'cluster_root', 'cluster_python', 'partition'}
    if not required <= config.keys():
        raise ValueError(f'Missing service settings: {sorted(required - config.keys())}')
    for name in ('cluster_root', 'cluster_python', 'cluster_registry', 'cluster_source', 'container_image'):
        if config.get(name) is not None and (not isinstance(config[name], str)
                or not PurePosixPath(config[name]).is_absolute() or '\n' in config[name]):
            raise ValueError(f'{name} must be an absolute cluster path')
    if config.get('ssh_host') and not re.fullmatch(r'[\w][\w.@-]*', config['ssh_host']):
        raise ValueError('Invalid SSH hostname/alias')
    for name in ('partition', 'account'):
        if name == 'account' and config.get(name) is None:
            continue
        if not isinstance(config[name], str) or not re.fullmatch(r'[\w.-]+', config[name]):
            raise ValueError(f'Invalid Slurm {name}')
    config.setdefault('gres', 'gpu:1')
    if not isinstance(config['gres'], str) or not re.fullmatch(r'gpu(?::[\w-]+)?:1', config['gres']):
        raise ValueError('SLICE requests one GPU: use gpu:1 or gpu:TYPE:1')
    runtime = config.get('container_runtime')
    if runtime not in (None, 'apptainer', 'singularity'):
        raise ValueError('container_runtime must be apptainer or singularity')
    if bool(runtime) != bool(config.get('container_image')):
        raise ValueError('Set container_runtime and container_image together')
    binds = config.get('container_binds', [])
    if not isinstance(binds, list):
        raise ValueError('container_binds must be a list of absolute bind paths')
    for bind in binds:
        if not isinstance(bind, str) or '\n' in bind:
            raise ValueError('Invalid container bind')
        parts = bind.split(':')
        paths = parts[:2]
        if (not 1 <= len(parts) <= 3 or any(not PurePosixPath(p).is_absolute() for p in paths)
                or (len(parts) == 3 and parts[2] not in ('ro', 'rw'))):
            raise ValueError('Binds must be /host[:/container[:ro|rw]]')
    for name, default in (('cpus', 8), ('memory_gb', 32), ('max_total_minutes', 1440)):
        config.setdefault(name, default)
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if (type(config.get('max_questions', 1000)) is not int
            or not 10 <= config.get('max_questions', 1000) <= 13000):
        raise ValueError('max_questions must be an integer in [10, 13000]')
    return config


class Slurm:
    def __init__(self, config):
        self.config = validate_settings(config)

    def submit(self, store, project, key, request):
        """An operator's explicit submission is approval; retries reuse the job.

        Reservation remains transactional. A concurrent caller that loses the
        reservation returns the existing job without attempting another sbatch.
        Failed or uncertain attempts require reconciliation, never auto-retry.
        """
        job = store.create(project, key, request)
        if job['status'] != 'awaiting_approval':
            return job
        try:
            return self.approve(store, job['id'])
        except ValueError:
            current = store.get(job['id'])
            if current['status'] != 'awaiting_approval':
                return current
            raise

    def command(self, arguments, *, stdin=None, timeout=30):
        if self.config.get('ssh_host'):
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                       self.config['ssh_host'], shlex.join([str(a) for a in arguments])]
        else:
            command = [str(a) for a in arguments]
        return subprocess.run(command, input=stdin, text=True, capture_output=True,
                              check=True, timeout=timeout).stdout

    def directory(self, identifier):
        if not re.fullmatch(r'[0-9a-f]{32}', identifier):
            raise ValueError('Invalid job identifier')
        return PurePosixPath(self.config['cluster_root']) / 'jobs' / identifier

    def plan(self, job):
        """Render the exact launch plan without SSH, file writes or submission."""
        if job['request']['max_questions'] > self.config.get('max_questions', 1000):
            raise ValueError('Request exceeds controller max_questions; explicitly raise the research limit first')
        identifier = job['id']
        directory = self.directory(identifier)
        execution = {name: self.config.get(name) for name in
                     ('partition', 'account', 'gres', 'cpus', 'memory_gb', 'cluster_python',
                      'container_runtime', 'container_image', 'container_binds')}
        execution['source_snapshot'] = str(directory / 'source') if self.config.get('cluster_source') else None
        bundle = dict(job_id=identifier, request=job['request'], request_hash=job['request_hash'],
                      project=job['project'], execution=execution,
                      registry=self.config.get('cluster_registry',
                          str(PurePosixPath(self.config['cluster_root']) / 'critics')))
        environment = ['env', 'PYTHONUNBUFFERED=1', f'OMP_NUM_THREADS={self.config["cpus"]}',
                       f'SLICE_NUM_THREADS={self.config["cpus"]}']
        if execution['source_snapshot']:
            environment.append('PYTHONPATH=' + execution['source_snapshot'])
        command = ['srun', '--ntasks=1']
        if self.config.get('container_runtime'):
            command += [self.config['container_runtime'], 'exec', '--nv']
            for bind in self.config.get('container_binds', []):
                command += ['--bind', bind]
            command += ['--pwd', str(directory), self.config['container_image']]
        command += environment + [self.config['cluster_python'], '-m', 'slice_probe',
                                   'job-run', '--job', str(directory / 'request.json')]
        script = '#!/usr/bin/env bash\nset -euo pipefail\n' + 'exec ' + shlex.join(command) + '\n'
        args = ['sbatch', '--parsable', '--requeue', '--nodes=1', '--ntasks=1',
                f'--job-name=slice-{identifier}', f'--comment=slice-{identifier}',
                f'--partition={self.config["partition"]}', f'--gres={self.config["gres"]}',
                f'--cpus-per-task={self.config["cpus"]}', f'--mem={self.config["memory_gb"]}G',
                f'--time={job["request"]["wall_minutes"]}', f'--chdir={directory}',
                f'--output={directory}/slurm-%j.log']
        if self.config.get('account'):
            args.append(f'--account={self.config["account"]}')
        args.append(str(directory / 'job.sh'))
        return dict(directory=str(directory), bundle=bundle, script=script, sbatch=args)

    def approve(self, store, identifier):
        plan = self.plan(store.get(identifier))
        job = store.reserve(identifier, max_total_minutes=self.config['max_total_minutes'])
        directory = self.directory(identifier)
        detail = dict(reserved=True, execution=plan['bundle']['execution'])
        store.update(identifier, 'submitting', detail=detail)
        try:
            self.command(['mkdir', '-p', directory])
            if self.config.get('cluster_source'):
                self.command(['mkdir', '-p', directory / 'source'])
                # Bytecode caches can describe an older checkout. Archive source
                # without them so recorded source hashes describe executed code.
                self.command(['tar', '--exclude=__pycache__', '--exclude=*.pyc', '--exclude=*.pyo',
                              '-C', self.config['cluster_source'], '-cf',
                              directory / 'source.tar', 'slice_probe'])
                self.command(['tar', '--no-same-owner', '-C', directory / 'source',
                              '-xf', directory / 'source.tar'])
            # tee is a fixed program; request data is stdin, never shell code.
            self.command(['tee', directory / 'request.json'], stdin=json.dumps(plan['bundle']))
            self.command(['tee', directory / 'job.sh'], stdin=plan['script'])
        except (OSError, subprocess.SubprocessError) as exc:
            return store.update(identifier, 'submission_failed',
                                detail=dict(detail, error=str(exc),
                                            note='No sbatch call was attempted; submit a new request to retry.'))
        try:
            output = self.command(plan['sbatch']).strip()
            if not re.fullmatch(r'\d+(?:;[\w.-]+)?', output):
                raise SubmissionUnknown(f'Unexpected sbatch response: {output[:300]}')
        except (OSError, subprocess.SubprocessError, SubmissionUnknown) as exc:
            # Even a nonzero SSH result may hide a successfully accepted job.
            return store.update(identifier, 'submission_unknown',
                                detail=dict(detail, error=str(exc),
                                            note='Reconcile by unique job name; do not resubmit.'))
        return store.update(identifier, 'queued', scheduler_id=output.split(';')[0],
                            detail=detail)

    def refresh(self, store, identifier):
        job = store.get(identifier)
        if job['status'] not in ACTIVE:
            return job
        directory = self.directory(identifier)
        # Reconcile a lost response using the request's unique scheduler name.
        live = self.command(['squeue', '--noheader', '--name', f'slice-{identifier}',
                             '--format=%i|%T']).strip().splitlines()
        accounting = self.command(['sacct', '--noheader', '--parsable2', '--name',
                                   f'slice-{identifier}', '--starttime',
                                   (datetime.fromisoformat(job['created']) - timedelta(days=1)).date().isoformat(),
                                   '--format=JobIDRaw,State,JobName%64,ExitCode,Elapsed']).strip().splitlines()
        entries = [line.split('|')[:2] for line in live if line.strip()]
        entries += [line.split('|')[:2] for line in accounting
                    if line.strip() and line.split('|')[0].isdigit()]
        ids = {row[0] for row in entries}
        if len(ids) > 1:
            raise SubmissionUnknown('Multiple scheduler jobs found for one request; inspect manually')
        if not ids:
            # Absence may mean delayed accounting; it does not prove failure.
            return job
        scheduler_id = next(iter(ids))
        if job['scheduler_id'] and job['scheduler_id'] != scheduler_id:
            raise SubmissionUnknown('Scheduler identity changed')
        # Accounting may lag a live job; live state takes precedence.
        states = {row[1].split()[0].rstrip('+') for row in
                  ([line.split('|')[:2] for line in live if line.strip()] or entries)}
        if states & {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'REQUEUED', 'SUSPENDED'}:
            status = 'running' if 'RUNNING' in states else 'queued'
        elif 'COMPLETED' in states:
            completed = [line.split('|') for line in accounting
                         if line.split('|')[0] == scheduler_id]
            codes = [row[3] for row in completed if len(row) >= 4 and row[3]]
            status = 'completion_unknown' if not codes else (
                'completed' if all(code == '0:0' for code in codes) else 'failed')
        elif any(s.startswith('CANCELLED') for s in states):
            status = 'cancelled'
        else:
            status = 'failed'
        detail = dict(job['detail'], scheduler_states=sorted(states), reserved=True)
        detail['accounting'] = [line for line in accounting if line.split('|')[0] == scheduler_id]
        try:
            progress = json.loads(self.command(['cat', directory / 'status.json']))
            if progress.get('job_id') != identifier or progress.get('request_hash') != job['request_hash']:
                raise ValueError('Worker status does not match the approved request')
            detail['worker'] = progress
            if status == 'completed' and progress.get('status') != 'completed':
                status = 'failed'
        except subprocess.CalledProcessError:
            detail['worker_status_missing'] = True
            if status == 'completed':
                status = 'failed'
                detail['error'] = 'Slurm completed without a worker completion record'
        if job['status'] == 'cancelling' and status not in TERMINAL:
            status = 'cancelling'
        return store.update(identifier, status, scheduler_id=scheduler_id, detail=detail)

    def cancel(self, store, identifier):
        job = store.cancel_unsubmitted(identifier)
        if job['status'] in TERMINAL or job['status'] == 'submission_failed':
            return job
        # Submission and cancellation cannot interleave into an untracked job:
        # uncertain/submitting requests require explicit reconciliation first.
        if not job['scheduler_id']:
            raise ValueError('Reconcile submission before cancelling its scheduler job')
        job = store.update(identifier, 'cancelling', detail=job['detail'])
        self.command(['scancel', job['scheduler_id']])
        return job
