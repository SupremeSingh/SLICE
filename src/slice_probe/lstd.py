"""Optional episodic LSTD(lambda) readouts over unchanged frozen SLICE features.

A += e_t (phi_t - gamma * phi_next)^T; b += e_t r_t
with e_t = gamma * lambda * e_(t-1) + phi_t. Terminal next features,
including the intercept, are zero. Statistics are raw sums, in float64.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .collection import iter_answers, retained
from .features import capture, concatenate, load_model
from .io import load_checkpoint, read_json, require_empty, save_checkpoint, sha256, write_json
from .model import restore
from .registry import feature_identity
from .training import encode


class LSTD:
    def __init__(self, width, *, trace_lambda=.9, discount_gamma=1., regularization=.01):
        if type(width) is not int or width < 1:
            raise ValueError('Positive integer feature width required')
        for name, value in (('lambda', trace_lambda), ('gamma', discount_gamma)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and in [0, 1]')
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError('Positive finite regularization required')
        self.width = width
        self.trace_lambda = trace_lambda
        self.discount_gamma = discount_gamma
        self.regularization = regularization
        self.a = np.zeros((width + 1, width + 1), dtype=np.float64)
        self.b = np.zeros(width + 1, dtype=np.float64)
        self.episodes = self.transitions = 0

    def add_episode(self, features, rewards):
        """Add one complete, ordered episode. No implicit truncated termination."""
        x = np.asarray(features, dtype=np.float64)
        r = np.asarray(rewards, dtype=np.float64)
        if (x.ndim != 2 or x.shape[1] != self.width or not len(x)
                or r.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(r).all()):
            raise ValueError('Finite ordered features and one reward per transition required')
        phi = np.column_stack((x, np.ones(len(x))))
        eligibility = np.zeros(self.width + 1, dtype=np.float64)
        # These updates apply only after validating the complete episode.
        for t, current in enumerate(phi):
            successor = phi[t + 1] if t + 1 < len(phi) else np.zeros_like(current)
            eligibility *= self.discount_gamma * self.trace_lambda
            eligibility += current
            self.a += np.outer(eligibility, current - self.discount_gamma * successor)
            self.b += eligibility * r[t]
        self.episodes += 1
        self.transitions += len(x)

    def solve(self):
        if not self.transitions:
            raise ValueError('No complete transitions to fit')
        system = self.a + self.regularization * np.eye(self.width + 1)
        # LSTD matrices are generally nonsymmetric: never use a Cholesky solver.
        weights = np.linalg.solve(system, self.b)
        denominator = max(np.linalg.norm(system) * np.linalg.norm(weights)
                          + np.linalg.norm(self.b), 1e-30)
        residual = float(np.linalg.norm(system @ weights - self.b) / denominator)
        if not np.isfinite(weights).all() or not np.isfinite(residual) or residual > 1e-8:
            raise ValueError('LSTD solve failed its numerical check')
        return torch.from_numpy(weights), dict(
            relative_residual=residual, condition_number=float(np.linalg.cond(system)),
            episodes=self.episodes, transitions=self.transitions,
            regularization=self.regularization, intercept_penalized=True)


def extract_trajectories(checkpoint, answers, out, device='cpu', *, feature_manifest):
    """Cache compressed pre-action states for complete *training* answers only.

    Unknown termination metadata and generation caps are excluded, never treated
    as terminal failures. Terminal reward is the verified binary outcome. No
    sampled-prefix cache is interpreted as a transition sequence.
    """
    record = load_checkpoint(checkpoint)
    source = Path(answers)
    collection = read_json(source / 'manifest.json')
    if collection['model'] != record['model']:
        raise ValueError('Trajectory encoder differs from fitted representation')
    source_manifest = read_json(feature_manifest)
    if (record.get('feature_manifest_sha256') != sha256(feature_manifest)
            or source_manifest['collection_manifest_sha256'] != sha256(source / 'manifest.json')
            or source_manifest['questions_sha256'] != sha256(source / 'questions.json')):
        raise ValueError('Trajectory collection differs from representation training data')
    out = require_empty(out)
    identity = record['model']
    backbone, _, _ = load_model(identity['id'], identity['revision'], device, hashes=identity['hashes'])
    compressor = restore(record, device)
    entries = []
    excluded = dict(unverified=0, capped_or_unknown_termination=0)
    for i, question, batch, path in iter_answers(source, 'train'):
        if source_manifest['answers_sha256'].get(path.name) != sha256(path):
            raise ValueError('Training answers changed since feature extraction')
        for j, row in enumerate(batch['responses']):
            if not retained(row):
                excluded['unverified'] += 1
                continue
            if row.get('capped') is not False:
                excluded['capped_or_unknown_termination'] += 1
                continue
            length, prompt_length = len(row['token_ids']), len(batch['prompt_ids'])
            ids = torch.tensor([batch['prompt_ids'] + row['token_ids']], device=device)
            indices = list(range(prompt_length - 1, prompt_length + length - 1))
            blocks = capture(backbone, ids, indices, record['layers'])
            x = concatenate(blocks, record['layers'], record['rms_eps'])
            z = encode(compressor, x.numpy(), record['mean'], record['scale'], device).numpy()
            rewards = np.zeros(length, dtype=np.float64)
            rewards[-1] = row['score']
            filename = f'{i:05d}-{j:04d}.npz'
            np.savez(out / filename, features=z, rewards=rewards)
            entries.append(dict(trajectory_id=f'{question["id"]}/{j}', question_index=i,
                                question_id=question['id'], response=j, file=filename,
                                sha256=sha256(out / filename), answer_sha256=sha256(path),
                                answer_file=path.name, transitions=length))
    if not entries:
        raise ValueError('No verified, explicitly completed training answers for LSTD')
    write_json(out / 'manifest.json', dict(
        format='slice-trajectories-v1', feature_id=feature_identity(record),
        source_feature_manifest_sha256=record.get('feature_manifest_sha256'),
        collection_manifest_sha256=sha256(source / 'manifest.json'),
        questions_sha256=sha256(source / 'questions.json'),
        reward='zero except verified correctness on the final transition',
        terminal='zero next features including intercept; caps and unknown termination excluded',
        split='train', entries=entries, excluded=excluded))
    return out


def fit_lstd(checkpoint, trajectories, feature_manifest, out, *, trace_lambda=.9,
             discount_gamma=1., regularization=.01):
    """Fit a new head; preserve every encoder/compressor tensor unchanged."""
    record = load_checkpoint(checkpoint)
    root = Path(trajectories).resolve()
    manifest = read_json(root / 'manifest.json')
    source = read_json(feature_manifest)
    if (manifest.get('format') != 'slice-trajectories-v1' or manifest['split'] != 'train'
            or manifest['feature_id'] != feature_identity(record)
            or record['feature_manifest_sha256'] != sha256(feature_manifest)
            or manifest['source_feature_manifest_sha256'] != sha256(feature_manifest)
            or manifest['collection_manifest_sha256'] != source['collection_manifest_sha256']
            or manifest['questions_sha256'] != source['questions_sha256']):
        raise ValueError('Trajectory/representation/training collection identity mismatch')
    question_path = Path(feature_manifest).parent / 'questions.json'
    if sha256(question_path) != source['questions_sha256']:
        raise ValueError('Question snapshot changed')
    train_ids = {i: q['id'] for i, q in enumerate(read_json(question_path)) if q['split'] == 'train'}
    statistics = LSTD(record['width'], trace_lambda=trace_lambda,
                      discount_gamma=discount_gamma, regularization=regularization)
    seen = set()
    contributions = set()
    for entry in manifest['entries']:
        if entry['trajectory_id'] in seen:
            raise ValueError('Duplicate trajectory contribution')
        seen.add(entry['trajectory_id'])
        contribution = (entry['question_index'], entry['response'])
        if contribution in contributions:
            raise ValueError('Duplicate answer contribution under a different trajectory ID')
        contributions.add(contribution)
        if (train_ids.get(entry['question_index']) != entry['question_id']
                or source['answers_sha256'].get(entry['answer_file']) != entry['answer_sha256']):
            raise ValueError('Trajectory does not belong to the training collection')
        path = (root / entry['file']).resolve()
        if not path.is_relative_to(root) or sha256(path) != entry['sha256']:
            raise ValueError('Trajectory cache changed')
        with np.load(path, allow_pickle=False) as data:
            if len(data['features']) != entry['transitions']:
                raise ValueError('Trajectory length mismatch')
            statistics.add_episode(data['features'], data['rewards'])
    weights, checks = statistics.solve()
    out = require_empty(out)
    result = dict(record, weights=weights, fit_checks=checks, ridge_states='complete-training-trajectories',
                  head=dict(method='lstd', trace_lambda=trace_lambda, discount_gamma=discount_gamma,
                            regularization=regularization, intercept=True,
                            reward=manifest['reward'], terminal=manifest['terminal'],
                            statistics='raw sums', target='discounted verified final-answer outcome'),
                  trajectory_manifest_sha256=sha256(root / 'manifest.json'))
    # Validation used to select the frozen compressor/Ridge recipe is not an
    # evaluation of this newly fitted head.
    if 'validation' in result:
        result['source_ridge_validation'] = result.pop('validation')
    save_checkpoint(out / 'lstd.pt', result)
    np.savez(out / 'statistics.npz', a=statistics.a, b=statistics.b)
    write_json(out / 'fit.json', dict(head=result['head'], checks=checks,
                                     feature_id=manifest['feature_id'], trajectories=sorted(seen)))
    return out / 'lstd.pt'
