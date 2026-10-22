"""Resumable, single-job supervised pipeline executed inside a Slurm allocation."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import shutil
import signal
import time
from threading import Timer

import numpy as np

from .io import load_checkpoint, read_json, sha256, write_json
from .jobs import now
from .registry import canonical_hash, inspect_artifact, publish
from .requests import TrainingRequest


class Cancelled(BaseException):
    """Control signal: must not be swallowed by verifier error handling."""
    pass


class BudgetExhausted(BaseException):
    pass


def completed_files(directory):
    return {str(p.relative_to(directory)): sha256(p)
            for p in directory.rglob('*') if p.is_file() and p.name != '.complete.json'}


def stage(root, name, action):
    """Commit a completed stage atomically; partial stages are never consumed."""
    destination = root / name
    marker = destination / '.complete.json'
    if destination.exists():
        saved = read_json(marker)
        if saved['files'] != completed_files(destination):
            raise ValueError(f'Completed stage changed: {name}')
        return destination
    temporary = root / ('.' + name + '-working')
    # Only collection supports genuine within-stage resumption. Other partial
    # stages are retried in a fresh directory; no partial matrices are reused.
    if temporary.exists() and name != 'answers':
        shutil.rmtree(temporary)
    action(temporary)
    write_json(temporary / '.complete.json', dict(files=completed_files(temporary)))
    os.rename(temporary, destination)
    return destination


def fit_optional_head(out, fitted, features, answers, request, device):
    from .calibration import apply_calibrator, fit_platt, select_acceptance
    from .collection import load_split
    from .lstd import extract_trajectories, fit_lstd
    from .model import predict_features

    out.mkdir()
    checkpoint = fitted / 'slice' / f'sparse-seed{request.seed}.pt'
    if request.head == 'lstd':
        cache = extract_trajectories(checkpoint, answers, out / 'trajectories', device,
                                     feature_manifest=features / 'manifest.json')
        checkpoint = fit_lstd(checkpoint, cache, features / 'manifest.json', out / 'fit',
                              trace_lambda=request.trace_lambda,
                              discount_gamma=request.discount_gamma,
                              regularization=request.regularization)
    record = load_checkpoint(checkpoint)
    val, _ = load_split(features, 'val')
    raw = predict_features(record, val['x'], device)
    calibrator = fit_platt(raw, val['y'])
    write_json(out / 'calibration.json', dict(
        checkpoint_sha256={'selected': sha256(checkpoint)},
        calibrators={'selected': calibrator},
        operating_points={'selected': select_acceptance(val['y'], apply_calibrator(raw, calibrator))}))
    # Relative path is stable across the stage's temporary-directory rename.
    relative = os.path.relpath(checkpoint, out)
    write_json(out / 'selection.json', dict(checkpoint=relative,
        checkpoint_sha256=sha256(checkpoint), feature_manifest_sha256=sha256(features / 'manifest.json'),
        calibration_sha256=sha256(out / 'calibration.json'), target='verified final-answer correctness'))


def evaluate_selected(out, selected, features, comparison_report, device):
    from .calibration import calibrated_report
    from .collection import load_split
    from .diagnostics import question_carry_forward, within_question_ranking
    from .evaluation import paired_interval
    from .metrics import metrics
    from .model import predict_features

    selection = read_json(selected / 'selection.json')
    checkpoint = selected / selection['checkpoint']
    if (sha256(checkpoint) != selection['checkpoint_sha256']
            or sha256(features / 'manifest.json') != selection['feature_manifest_sha256']
            or sha256(selected / 'calibration.json') != selection['calibration_sha256']):
        raise ValueError('Selected head changed before evaluation')
    record = load_checkpoint(checkpoint)
    test, _ = load_split(features, 'test')
    raw = predict_features(record, test['x'], device)
    data = read_json(selected / 'calibration.json')
    probability, calibrated = calibrated_report(test['y'], raw, data['calibrators']['selected'],
                                                data['operating_points']['selected'])
    initial = question_carry_forward(raw, test['question'], test['position'])
    advancing = test['position'] > 0
    out.mkdir()
    np.savez_compressed(out / 'predictions.npz', raw_value=raw, probability=probability,
                        y=test['y'], question=test['question'], response=test['response'],
                        position=test['position'])
    anchors, _ = load_split(features, 'test', anchors=True)
    anchor_scores = predict_features(record, anchors['x'], device)
    within = within_question_ranking(anchors['y'], anchor_scores, anchors['question'],
                                     anchors['response'], anchors['position'])
    np.savez_compressed(out / 'checkpoint-predictions.npz', raw_value=anchor_scores,
                        y=anchors['y'], question=anchors['question'],
                        response=anchors['response'], position=anchors['position'])
    write_json(out / 'summary.json', dict(
        raw=metrics(test['y'], raw), calibrated=calibrated,
        question_only=metrics(test['y'][~advancing], raw[~advancing]),
        updated_minus_initial=paired_interval(test['y'][advancing], raw[advancing],
            initial[:, advancing], test['question'][advancing]) if advancing.any() else None,
        baselines=comparison_report['results'], checkpoint_sha256=sha256(checkpoint),
        head=record.get('head', {'method': 'ridge'}),
        within_question=within,
        head_training_distribution=record.get('ridge_states'),
        baseline_training_distribution='matched sampled prefixes',
        scope='New pilot, conditional on verifiable answers. LSTD uses complete-token trajectories; '
              'baseline fitting distributions differ. Raw values and calibrated probabilities are distinct.'))


def run_job(path, *, device='auto'):
    from .collection import collect, extract
    from .comparison import evaluate_comparison, fit_comparison
    from .datasets import prepare_dataset
    from .features import resolve_device
    from .preflight import smoke, check_gate, execution_identity

    path = Path(path).resolve()
    root = path.parent
    bundle = read_json(path)
    request = TrainingRequest.parse(bundle['request'])
    if canonical_hash(request.to_dict()) != bundle['request_hash']:
        raise ValueError('Approved request hash changed')
    device = resolve_device(device)

    def status(state, name, **extra):
        write_json(root / 'status.json', dict(job_id=bundle['job_id'],
            request_hash=bundle['request_hash'], status=state, stage=name, updated=now(), **extra))

    def stop(signum, frame):
        raise Cancelled('Worker received cancellation/termination signal')

    def exhausted(signum, frame):
        raise BudgetExhausted('Cumulative job wall-time allowance exhausted')

    # A lock protects restart/requeue overlap. Never place the SQLite database here.
    with (root / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved_request = root / 'approved.json'
        if saved_request.exists() and read_json(saved_request) != bundle:
            raise ValueError('Cannot resume with changed training settings')
        write_json(saved_request, bundle)
        previous = signal.signal(signal.SIGTERM, stop)
        # Math-Verify owns SIGALRM for its parse/verify timeouts. Use a separate
        # signal and watchdog so neither timeout can disable the other.
        budget_handler = signal.signal(signal.SIGUSR1, exhausted)
        watchdog = None
        current = 'starting'
        try:
            deadline_path = root / 'deadline.json'
            if not deadline_path.exists():
                write_json(deadline_path, dict(expires=time.time() + request.wall_minutes * 60))
            remaining = read_json(deadline_path)['expires'] - time.time()
            if remaining <= 0:
                raise BudgetExhausted('Job deadline expired before resumption')
            # Preserve the original deadline across requeues. Waiting between
            # attempts also consumes this conservative elapsed-time allowance.
            watchdog = Timer(remaining, os.kill, args=(os.getpid(), signal.SIGUSR1))
            watchdog.daemon = True
            watchdog.start()
            execution = execution_identity(bundle)
            gate = None
            def execute(name, action):
                nonlocal current
                current = name
                status('running', name)
                if gate is not None:
                    check_gate(gate / 'report.json', bundle, execution)
                return stage(root, name, action)

            gate = execute('smoke', lambda out: smoke(out, request, bundle, device, execution))
            smoke_report = check_gate(gate / 'report.json', bundle, execution)
            questions = execute('questions', lambda out: prepare_dataset(request, out))
            answers = execute('answers', lambda out: collect(
                questions / 'questions.jsonl', out, 'Qwen/Qwen2.5-Math-1.5B', smoke_report['model']['revision'],
                device, request.responses_per_question, request.max_new_tokens, request.seed))
            if read_json(answers / 'manifest.json')['model'] != smoke_report['model']:
                raise ValueError('Collected encoder identity differs from the smoke-tested encoder')
            features = execute('features', lambda out: extract(
                answers, out, device, request.layer_base, request.num_slices,
                baselines=True, checkpoints=[0, 32, 128]))
            write_json(root / 'comparison-config.json', dict(
                width=request.compression_width, epochs=request.epochs, seeds=[request.seed],
                ridge_epsilon=request.regularization, mode='exploratory',
                test_scope='New API pilot; freshness is not established by automatic splitting.'))
            fitted = execute('fitted', lambda out: fit_comparison(
                features, out, root / 'comparison-config.json', device))
            selected = execute('selected', lambda out: fit_optional_head(
                out, fitted, features, answers, request, device))
            comparison = execute('comparison', lambda out: evaluate_comparison(
                features, fitted, out, device, sha256(fitted / 'lock.json')))
            evaluation = execute('evaluation', lambda out: evaluate_selected(
                out, selected, features, read_json(comparison / 'summary.json'), device))
            current = 'publishing'
            status('running', current)
            choice = read_json(selected / 'selection.json')
            version = 'job-' + bundle['job_id']
            destination = Path(bundle['registry']) / version
            if destination.exists():
                manifest = inspect_artifact(destination)
                if (manifest['checkpoint_sha256'] != choice['checkpoint_sha256']
                        or manifest['metadata'].get('request_hash') != bundle['request_hash']):
                    raise ValueError('Published version belongs to a different request or head')
            else:
                publish(selected / choice['checkpoint'], bundle['registry'], version,
                        calibration=selected / 'calibration.json', evaluation=evaluation,
                        feature_manifest=features / 'manifest.json',
                        statistics=selected / 'fit/statistics.npz' if request.head == 'lstd' else None,
                        trajectory_manifest=selected / 'trajectories/manifest.json' if request.head == 'lstd' else None,
                        metadata=dict(request_hash=bundle['request_hash'], request=request.to_dict(),
                                      project=bundle.get('project'), smoke=smoke_report))
            manifest = inspect_artifact(destination)
            status('completed', 'published', critic_version=version,
                   artifact_path=str(destination), feature_id=manifest['feature_id'],
                   evaluation=read_json(evaluation / 'summary.json'))
        except BaseException as exc:
            status('cancelled' if isinstance(exc, (Cancelled, KeyboardInterrupt)) else 'failed',
                   current, error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if watchdog is not None:
                watchdog.cancel()
                watchdog.join()
            signal.signal(signal.SIGUSR1, budget_handler)
            signal.signal(signal.SIGTERM, previous)
