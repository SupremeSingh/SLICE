"""Conservative, bounded ingestion of scalar-reference Hugging Face datasets."""
from collections import Counter
import hashlib
import json
import re

from .data import family, load_questions, number, reference
from .io import require_empty, sha256, write_json


def select_questions(rows, request):
    """Select unique number-template families without examining model outcomes."""
    selected, excluded = {}, Counter()
    for index, row in enumerate(rows):
        if index >= 100_000:
            break
        question = row.get(request.question_column)
        target = row.get(request.reference_column)
        if not isinstance(question, str) or not question.strip() or len(question) > 16_000:
            excluded['question'] += 1
            continue
        value = number(target)
        if value is None:
            excluded['non_scalar_reference'] += 1
            continue
        if (re.search(r'!\[|\\includegraphics|https?://|\b(?:diagram|figure|prove that)\b', question, re.I)
                or reference(question, r'\boxed{' + str(value) + '}', str(value)) is None):
            excluded['unsupported_task'] += 1
            continue
        key = family(question)
        if key in selected:
            excluded['duplicate_family'] += 1
            continue
        selected[key] = dict(id=f'{request.dataset}/{request.dataset_split}/{index}',
                             question=question.strip(), reference=str(value))
        if len(selected) == request.max_questions:
            break
    if len(selected) < request.max_questions:
        raise ValueError(f'Only {len(selected)} eligible unique questions; requested {request.max_questions}')
    ordered = sorted(selected.items(), key=lambda item: hashlib.sha256(
        f'slice-api-v1:{request.seed}:{item[0]}'.encode()).hexdigest())
    train_end = int(len(ordered) * .7)
    val_end = train_end + max(1, int(len(ordered) * .15))
    questions = [dict(row, split='train' if i < train_end else 'val' if i < val_end else 'test')
                 for i, (_, row) in enumerate(ordered)]
    return questions, dict(excluded)


def prepare_dataset(request, out):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(request.dataset, revision=request.revision).sha
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Dataset revision must resolve to an immutable commit SHA')
    # Remote dataset scripts are never executed. Repositories must be supported
    # by the installed datasets library's built-in data-file loaders.
    rows = load_dataset(request.dataset, name=request.dataset_config,
                        split=request.dataset_split, revision=revision,
                        streaming=True, trust_remote_code=False)
    questions, exclusions = select_questions(rows, request)
    out = require_empty(out)
    path = out / 'questions.jsonl'
    path.write_text(''.join(json.dumps(q) + '\n' for q in questions))
    load_questions(path)
    write_json(out / 'selection.json', dict(
        dataset=request.dataset, revision=revision, config=request.dataset_config,
        source_split=request.dataset_split, questions_sha256=sha256(path),
        counts={s: sum(q['split'] == s for q in questions) for s in ('train', 'val', 'test')},
        exclusions=exclusions, seed=request.seed,
        scope='New deterministic family-separated pilot; no claim of semantic decontamination.'))
