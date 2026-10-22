"""Private control API. Importing this module never loads an encoder."""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import secrets
import subprocess
from threading import RLock

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .critic import Critic
from .jobs import JobStore, public_job
from .registry import inspect_artifact, version_name
from .requests import TrainingRequest
from .slurm import Slurm, settings


class Batch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    critic_version: str
    states: list[dict] = Field(min_length=1, max_length=64)
    return_calibrated_probability: bool = False


def create_app(config_path=None, *, tokens=None, loader=None):
    """Factory for uvicorn; credentials are environment-only, never in settings."""
    config_path = config_path or os.environ.get('SLICE_SERVICE_CONFIG')
    if not config_path:
        raise ValueError('Set SLICE_SERVICE_CONFIG to the service JSON path')
    config = settings(config_path)
    keys = tokens if tokens is not None else json.loads(os.environ.get('SLICE_API_KEYS', '{}'))
    if not keys or any(not isinstance(k, str) or not k or not isinstance(v, str) or len(v) < 24
                       for k, v in keys.items()) or len(set(keys.values())) != len(keys):
        raise ValueError('SLICE_API_KEYS must map project names to unique secrets of at least 24 characters')
    store, scheduler = JobStore(config['database']), Slurm(config)
    registry = Path(config['registry']).resolve()
    resident, lock = OrderedDict(), RLock()
    load = loader or Critic.load
    app = FastAPI(title='SLICE private API', version='1')
    app.state.store = store
    app.state.scheduler = scheduler

    def project(authorization: str = Header(default='')):
        scheme, _, token = authorization.partition(' ')
        for name, key in keys.items():
            if scheme.lower() == 'bearer' and secrets.compare_digest(token, key):
                return name
        raise HTTPException(401, 'A valid project bearer key is required')

    def owned(identifier, owner):
        try:
            return store.get(identifier, owner)
        except KeyError:
            raise HTTPException(404, 'Unknown job') from None

    def artifact(version, owner):
        try:
            path = registry / version_name(version)
            manifest = inspect_artifact(path)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, 'Unknown critic') from None
        # Manually published artifacts can be shared between your projects.
        # Managed jobs always publish explicit ownership metadata.
        if manifest.get('metadata', {}).get('project') not in (None, owner):
            raise HTTPException(404, 'Unknown critic')
        return path, manifest

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.post('/v1/training-jobs', status_code=202)
    def create_training(value: dict, owner=Depends(project),
                        idempotency_key: str = Header(alias='Idempotency-Key')):
        try:
            request = TrainingRequest.parse(value)
            job = store.create(owner, idempotency_key, value)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return dict(public_job(job), preview=request.preview())

    @app.get('/v1/jobs/{identifier}')
    def get_job(identifier: str, owner=Depends(project)):
        job = owned(identifier, owner)
        if job['status'] in ('queued', 'running', 'cancelling', 'completion_unknown'):
            try:
                job = scheduler.refresh(store, identifier)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                return dict(public_job(job), refresh_error=str(exc))
        return public_job(job)

    @app.post('/v1/jobs/{identifier}/cancel')
    def cancel(identifier: str, owner=Depends(project),
               idempotency_key: str = Header(alias='Idempotency-Key')):
        owned(identifier, owner)
        try:
            store.record_action(owner, 'cancel', idempotency_key, identifier)
            return public_job(scheduler.cancel(store, identifier))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/v1/jobs/{identifier}/artifacts')
    def artifacts(identifier: str, owner=Depends(project)):
        job = owned(identifier, owner)
        worker = job['detail'].get('worker', {})
        return dict(status=job['status'], critic_version=worker.get('critic_version'),
                    cluster_path=worker.get('artifact_path'),
                    note='Shared cluster paths; artifacts are not downloaded or exposed publicly.')

    @app.get('/v1/usage')
    def usage(owner=Depends(project)):
        rows = store.list(owner)
        reserved = sum(j['request']['wall_minutes'] for j in rows if j['detail'].get('reserved'))
        return dict(reserved_gpu_minutes=reserved,
                    accounting='Conservative submitted wall-time allowance, not measured GPU billing.',
                    jobs=[public_job(j) for j in rows])

    @app.get('/v1/critics/{version}')
    def get_critic(version: str, owner=Depends(project)):
        return artifact(version, owner)[1]

    @app.post('/v1/scoring-batches')
    def scoring(value: Batch, owner=Depends(project)):
        path, manifest = artifact(value.critic_version, owner)
        for state in value.states:
            if not isinstance(state.get('state_id'), str) or len(state['state_id']) > 256:
                raise HTTPException(422, 'Every state needs a string state_id of at most 256 characters')
            if (len(str(state.get('question', ''))) + len(str(state.get('prefix', ''))) > 64_000
                    or (isinstance(state.get('token_ids'), list) and len(state['token_ids']) > 32_768)):
                raise HTTPException(413, 'State exceeds API input limits; input was not truncated')
        if len({s['state_id'] for s in value.states}) != len(value.states):
            raise HTTPException(422, 'state_id values must be unique within a batch')
        # Serialize cache replacement and hook execution. One resident encoder
        # avoids keeping an unbounded number of model copies on a personal GPU.
        with lock:
            if value.critic_version not in resident:
                resident.clear()
                resident[value.critic_version] = load(path, device=config.get('device', 'auto'))
            critic = resident[value.critic_version]
            try:
                result = critic.score_batch(value.states,
                    return_calibrated_probability=value.return_calibrated_probability)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        return dict(critic_version=manifest['version'], results=result)

    return app
