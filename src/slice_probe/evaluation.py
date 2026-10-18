"""Evaluate locked probes on held-out prefixes with paired question intervals."""
from pathlib import Path
import csv
import numpy as np

from .collection import load_split
from .io import read_json, load_checkpoint, require_empty, write_json, sha256
from .metrics import metrics
from .model import predict_features


def metric_average(y, predictions):
    rows = [metrics(y, p) for p in np.atleast_2d(predictions)]
    return dict(by_seed=rows, mean={key: None if any(r[key] is None for r in rows)
        else float(np.mean([r[key] for r in rows]))
        for key in ('accuracy', 'brier', 'auroc', 'ece', 'outside_probability_range')})


def paired_interval(y, a, b, questions, repetitions=2000):
    if repetitions < 1:
        raise ValueError('Positive bootstrap budget required')
    group = np.unique(questions, return_inverse=True)[1]
    difference = np.square(np.atleast_2d(a) - y).mean(0) - np.square(np.atleast_2d(b) - y).mean(0)
    per_question = np.bincount(group, weights=difference) / np.bincount(group)
    rng = np.random.default_rng(913)
    samples = [rng.choice(per_question, len(per_question), replace=True).mean() for _ in range(repetitions)]
    return dict(question_mean_brier_difference=float(per_question.mean()),
                descriptive_95_interval=np.quantile(samples, [.025, .975]).tolist())


def evaluate(features, probes, out, device='cpu'):
    test, manifest = load_split(features, 'test')
    train, _ = load_split(features, 'train')
    val, _ = load_split(features, 'val')
    sets = [set(d['question']) for d in (train, val, test)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Question overlap across train/validation/test')
    selections = read_json(Path(probes) / 'selection.json')['selected']
    predictions, checkpoints = {}, {}
    for name, item in selections.items():
        path = Path(probes) / item['path']
        if sha256(path) != item['sha256']:
            raise ValueError('Selected checkpoint changed')
        record = load_checkpoint(path)
        if (record['layers'] != manifest['layers'] or record['rms_eps'] != manifest['rms_eps']
                or record['model']['hashes'] != manifest['model']['hashes']):
            raise ValueError('Probe and feature encoder differ')
        predictions[name] = predict_features(record, test['x'], device)
        checkpoints[name] = dict(kind=record['kind'], seed=record['seed'], ridge_states=record['ridge_states'])
    if not predictions:
        raise ValueError('No selected probes')
    groups = {'constant': np.full((1, len(test['y'])), float(train['y'].mean()))}
    for kind in sorted({v['kind'] for v in checkpoints.values()}):
        groups[kind] = np.stack([predictions[k] for k,v in checkpoints.items() if v['kind'] == kind])
    results = {name: metric_average(test['y'], p) for name,p in groups.items()}
    masks = {'question_only': test['position'] == 0,
             'tokens_1_32': (test['position'] > 0) & (test['position'] <= 32),
             'tokens_33_128': (test['position'] > 32) & (test['position'] <= 128),
             'tokens_129_plus': test['position'] > 128}
    by_prefix = {label: dict(prefixes=int(mask.sum()), results={name: metric_average(test['y'][mask], p[:,mask])
        for name,p in groups.items()}) for label,mask in masks.items() if mask.any()}
    comparisons = {f'{name}_minus_constant': paired_interval(test['y'], p, groups['constant'], test['question'])
                   for name,p in groups.items() if name != 'constant'}
    if 'plain' in groups and 'sparse' in groups:
        comparisons['sparse_minus_plain'] = paired_interval(test['y'], groups['sparse'], groups['plain'], test['question'])
    out = require_empty(out)
    write_json(out / 'summary.json', dict(results=results, by_prefix=by_prefix, comparisons=comparisons,
        test_questions=len(sets[2]), test_prefixes=len(test['y']), checkpoints=checkpoints,
        scope='Retained-answer conditional correctness prediction. Metrics average heads, not ensemble predictions. '
              'Question bootstraps are descriptive and unadjusted; head seeds share data. Raw Ridge scores are not calibrated probabilities.'))
    np.savez_compressed(out / 'predictions.npz', **predictions, y=test['y'], question=test['question'], position=test['position'])
    with (out / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['representation', 'accuracy', 'brier', 'auroc', 'ece', 'outside_probability_range'])
        writer.writeheader()
        writer.writerows(dict(representation=name, **row['mean']) for name,row in results.items())
    lines = ['SLICE: eventual-correctness prediction', 'representation     accuracy    Brier    AUROC']
    for name,r in results.items():
        row = r['mean']
        auc = 'NA' if row['auroc'] is None else f'{row["auroc"]:.4f}'
        lines.append(f'{name:18} {row["accuracy"]:.4f}    {row["brier"]:.5f}    {auc}')
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return results
