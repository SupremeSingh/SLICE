"""Private job ledger and explicit approval boundary for Slurm work.

SQLite lives on the control machine's local disk, never on a cluster shared
filesystem. Workers communicate through atomic per-job status files instead.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from .registry import canonical_hash
from .requests import TrainingRequest


ACTIVE = ('submitting', 'submission_unknown', 'queued', 'running', 'cancelling', 'completion_unknown')
TERMINAL = ('completed', 'failed', 'cancelled')


def now():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL,
                    request TEXT NOT NULL, status TEXT NOT NULL,
                    scheduler_id TEXT, detail TEXT NOT NULL DEFAULT '{}',
                    created TEXT NOT NULL, updated TEXT NOT NULL,
                    UNIQUE(project, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS actions (
                    project TEXT NOT NULL, operation TEXT NOT NULL, key TEXT NOT NULL,
                    target TEXT NOT NULL, UNIQUE(project, operation, key)
                );
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            raise KeyError('Unknown job')
        result = dict(row)
        for key in ('request', 'detail'):
            result[key] = json.loads(result[key])
        return result

    def create(self, project, key, value):
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            raise ValueError('A 1–128 character Idempotency-Key is required')
        request = TrainingRequest.parse(value).to_dict()
        digest = canonical_hash(request)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM jobs WHERE project=? AND idempotency_key=?',
                             (project, key)).fetchone()
            if old:
                if old['request_hash'] != digest:
                    raise ValueError('Idempotency key was already used with another request')
                return self.decode(old)
            identifier = uuid.uuid4().hex
            stamp = now()
            db.execute('INSERT INTO jobs(id,project,idempotency_key,request_hash,request,status,created,updated) '
                       'VALUES(?,?,?,?,?,?,?,?)', (identifier, project, key, digest,
                       json.dumps(request), 'awaiting_approval', stamp, stamp))
            return self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone())

    def get(self, identifier, project=None):
        with self.connection() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone()
            if row is None or (project is not None and row['project'] != project):
                raise KeyError('Unknown job')
            return self.decode(row)

    def list(self, project=None):
        with self.connection() as db:
            rows = db.execute('SELECT * FROM jobs ORDER BY created DESC').fetchall()
            return [self.decode(row) for row in rows if project is None or row['project'] == project]

    def backup(self, path):
        """Produce a consistent standalone snapshot, including committed WAL data."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb'):
            pass
        with self.connection() as source:
            target = sqlite3.connect(path)
            try:
                source.backup(target)
                if target.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError('Controller backup integrity check failed')
            finally:
                target.close()
        return path

    def update(self, identifier, status, *, scheduler_id=None, detail=None):
        with self.connection() as db:
            db.execute('UPDATE jobs SET status=?,scheduler_id=COALESCE(?,scheduler_id),'
                       'detail=?,updated=? WHERE id=?',
                       (status, scheduler_id, json.dumps(detail or {}), now(), identifier))
        return self.get(identifier)

    def record_action(self, project, operation, key, target):
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            raise ValueError('A 1–128 character Idempotency-Key is required')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT target FROM actions WHERE project=? AND operation=? AND key=?',
                             (project, operation, key)).fetchone()
            if old and old['target'] != target:
                raise ValueError('Idempotency key was already used for another target')
            db.execute('INSERT OR IGNORE INTO actions VALUES(?,?,?,?)', (project, operation, key, target))

    def cancel_unsubmitted(self, identifier):
        """Serialize cancellation with the approval reservation transaction."""
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone()
            job = self.decode(row)
            if job['status'] == 'awaiting_approval':
                db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?', ('cancelled', now(), identifier))
        return self.get(identifier)

    def reserve(self, identifier, *, max_total_minutes):
        """Reserve the single active slot and a conservative lifetime allowance."""
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = [self.decode(r) for r in db.execute('SELECT * FROM jobs').fetchall()]
            job = next((r for r in rows if r['id'] == identifier), None)
            if job is None:
                raise KeyError('Unknown job')
            if job['status'] != 'awaiting_approval':
                raise ValueError('Only an awaiting-approval job may be submitted; reconcile uncertain submissions')
            if any(r['status'] in ACTIVE for r in rows):
                raise ValueError('One training job is already active or awaiting reconciliation')
            # Failed/cancelled reservations are not refunded automatically.
            # Only explicit scheduler-verified settlement substitutes elapsed time.
            reserved = sum(r['detail'].get('charged_minutes', r['request']['wall_minutes']) for r in rows
                           if r['status'] != 'awaiting_approval' and r['detail'].get('reserved', False))
            if reserved + job['request']['wall_minutes'] > max_total_minutes:
                raise ValueError('Configured cumulative GPU-minute allowance would be exceeded')
            db.execute('UPDATE jobs SET status=?,detail=?,updated=? WHERE id=?',
                       ('submitting', json.dumps({'reserved': True}), now(), identifier))
        return self.get(identifier)

    def settle_budget(self, identifier, minutes, accounting):
        """Explicit operator action after scheduler-confirmed terminal accounting."""
        if type(minutes) is not int or minutes < 1:
            raise ValueError('Settled minutes must be a positive integer')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            job = self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone())
            if job['status'] not in TERMINAL or not job['detail'].get('reserved'):
                raise ValueError('Only a reserved terminal job may be settled')
            if 'budget_settlement' not in job['detail']:
                detail = dict(job['detail'], charged_minutes=minutes, budget_settlement=dict(
                    requested_minutes=job['request']['wall_minutes'], charged_minutes=minutes,
                    accounting=accounting, settled=now()))
                db.execute('UPDATE jobs SET detail=?,updated=? WHERE id=?',
                           (json.dumps(detail), now(), identifier))
        return self.get(identifier)


def public_job(job):
    """Never expose internal paths, submission commands or credentials in requests."""
    return {key: job[key] for key in
            ('id', 'project', 'status', 'scheduler_id', 'detail', 'created', 'updated')}
