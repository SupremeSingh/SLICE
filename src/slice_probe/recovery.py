"""CPU-only request preparation pinned to a previous run's verified assets."""
from pathlib import Path

from .datasets import prepare_dataset
from .io import read_json, require_empty, write_json
from .registry import canonical_hash
from .requests import TrainingRequest


def preflight_request(job_dir, out):
    root, out = Path(job_dir), require_empty(out)
    bundle = read_json(root / 'request.json')
    smoke = read_json(root / 'smoke/report.json')
    selection = read_json(root / 'questions/selection.json')
    if (canonical_hash(bundle['request']) != bundle['request_hash']
            or smoke['request_hash'] != bundle['request_hash'] or smoke.get('passed') is not True):
        raise ValueError('Previous request and verified encoder identities do not match')
    request = TrainingRequest.parse(dict(bundle['request'], revision=selection['revision'],
                                         model_revision=smoke['model']['revision']))
    prepare_dataset(request, out / 'questions', encoder_identity=smoke['model'])
    prepared = read_json(out / 'questions/selection.json')
    write_json(out / 'request.json', request.to_dict())
    report = dict(ready=True, request=str((out / 'request.json').resolve()),
        counts=prepared['counts'], exclusions=prepared['exclusions'],
        prompt_budget=prepared['prompt_budget'], dataset_revision=prepared['revision'],
        source_job=bundle['job_id'], scope='CPU data/tokenizer preflight only; no model inference or training.')
    write_json(out / 'report.json', report)
    return report
