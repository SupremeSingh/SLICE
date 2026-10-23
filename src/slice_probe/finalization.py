"""Finish all requested heads from immutable cached work, without new generation."""
from pathlib import Path
import fcntl
import os
import shutil
import subprocess

from .comparison import environment, evaluate_comparison
from .io import read_json, require_empty, sha256, write_json
from .registry import canonical_hash
from .requests import TrainingRequest
from .worker import completed_files, stage, fit_optional_head, evaluate_selected, publish_heads


def prepare_finalization(job_dir, out):
    source = Path(job_dir).resolve()
    out = Path(out).resolve()
    if out.is_relative_to(source):
        raise ValueError('Finalization output must be outside the original immutable job')
    package = Path(__file__).resolve().parent
    if out.is_relative_to(package):
        raise ValueError('Finalization output must be outside the source package')
    bundle = read_json(source / 'request.json')
    request = TrainingRequest.parse(bundle['request'])
    if canonical_hash(request.to_dict()) != bundle['request_hash']:
        raise ValueError('Original request identity changed')
    out = require_empty(out)
    shutil.copytree(package, out / 'source/slice_probe',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    runtime = environment()
    plan = dict(format='slice-finalization-v1', source_job=str(source), bundle=bundle,
        source_sha256=runtime['source_sha256'],
        runtime={key:runtime[key] for key in ('packages', 'python')},
        version='finalized-' + bundle['job_id'],
        execution_policy=dict(gpus=1, cpus=8, memory_gb=96, max_wall_minutes=10080),
        scope='Reuse completed answers/features/compressors. Fit missing final heads for all requested seeds.')
    write_json(out / 'plan.json', plan)
    return dict(plan=str(out / 'plan.json'), output=str(out), source_snapshot=str(out / 'source'))


def finalize_job(plan_path, device='cpu'):
    """Never edit source stages, reselect compressors, regenerate or resubmit work."""
    plan_path = Path(plan_path).resolve()
    out = plan_path.parent
    plan = read_json(plan_path)
    runtime = environment()
    if (plan.get('format') != 'slice-finalization-v1'
            or runtime['source_sha256'] != plan['source_sha256']
            or {key:runtime[key] for key in ('packages', 'python')} != plan['runtime']):
        raise ValueError('Run finalization using its prepared source snapshot')
    source, bundle = Path(plan['source_job']), plan['bundle']
    request = TrainingRequest.parse(bundle['request'])
    if (read_json(source / 'request.json') != bundle
            or canonical_hash(request.to_dict()) != bundle['request_hash']):
        raise ValueError('Original request changed after preparation')
    live = subprocess.run(['squeue', '--noheader', '--name', 'slice-' + bundle['job_id'], '--format=%i'],
                          check=True, text=True, capture_output=True, timeout=30).stdout.strip()
    if live:
        raise ValueError('Original scheduler job is still live; do not finalize concurrently')
    with (source / '.worker.lock').open('r+') as original_lock, (out / '.finalize.lock').open('a') as lock:
        fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The source worker may have failed during reporting; its completed data
        # and fitted compressors remain valid and independently checked.
        for name in ('answers', 'features', 'fitted'):
            directory = source / name
            if read_json(directory / '.complete.json')['files'] != completed_files(directory):
                raise ValueError(f'Original completed {name} stage is missing or changed')
        reusable = source / 'selected'
        if reusable.exists():
            if read_json(reusable / '.complete.json')['files'] != completed_files(reusable):
                raise ValueError('Original selected stage changed')
        else:
            reusable = None
        write_json(out / 'provenance.json', dict(request_hash=bundle['request_hash'],
            source_job=bundle['job_id'], environment=environment(),
            execution_policy=plan['execution_policy'],
            scheduler_job_id=os.environ.get('SLURM_JOB_ID'),
            source_stage_markers={name:sha256(source/name/'.complete.json')
                                  for name in ('answers', 'features', 'fitted')}))
        try:
            write_json(out / 'status.json', dict(status='running', stage='selected'))
            selected = stage(out, 'selected', lambda target: fit_optional_head(
                target, source / 'fitted', source / 'features', source / 'answers',
                request, device, reusable=reusable))
            write_json(out / 'status.json', dict(status='running', stage='comparison'))
            comparison = stage(out, 'comparison', lambda target: evaluate_comparison(
                source / 'features', source / 'fitted', target, device, sha256(source / 'fitted/lock.json')))
            write_json(out / 'status.json', dict(status='running', stage='evaluation'))
            evaluation = stage(out, 'evaluation', lambda target: evaluate_selected(
                target, selected, source / 'features', read_json(comparison / 'summary.json'), device))
            paths = publish_heads(selected, evaluation, source / 'features', bundle['registry'], plan['version'],
                dict(request_hash=bundle['request_hash'], request=request.to_dict(),
                     project=bundle.get('project'), source_job=bundle['job_id'],
                     finalization_provenance=read_json(out / 'provenance.json')))
            result = dict(status='completed', stage='published', artifact_paths=paths,
                artifact_path=paths[str(request.seed)], evaluation=read_json(evaluation / 'summary.json'))
            write_json(out / 'status.json', result)
            return result
        except BaseException as exc:
            write_json(out / 'status.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}'))
            raise
