"""Conservative, bounded ingestion of scalar-reference Hugging Face datasets."""
from collections import Counter
import hashlib
import json
import re

from .data import family, load_questions, number, reference
from .io import require_empty, sha256, write_json


class PromptBudget:
    """Eligibility only: never truncate or inspect generated answers."""
    def __init__(self, tokenizer, context_limit, max_new_tokens):
        if type(context_limit) is not int or context_limit <= max_new_tokens:
            raise ValueError('Model context leaves no room for a prompt')
        self.tokenizer = tokenizer
        self.context_limit = context_limit
        self.max_new_tokens = max_new_tokens
        self.limit = min(512, context_limit - max_new_tokens)

    def __call__(self, question):
        from .features import prompt
        length = len(self.tokenizer.encode(prompt(question), add_special_tokens=False))
        return 0 < length <= self.limit


def load_prompt_budget(identity, max_new_tokens):
    from transformers import AutoTokenizer
    from .features import model_location
    from .io import read_json

    directory, _ = model_location(identity['id'], identity['revision'], identity['hashes'])
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    return PromptBudget(tokenizer, read_json(directory / 'config.json')['max_position_embeddings'],
                        max_new_tokens)


def select_questions(rows, request, *, question_filter=None):
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
        if question_filter is not None and not question_filter(question):
            excluded['prompt_budget'] += 1
            continue
        selected[key] = dict(id=f'{request.dataset}/{request.dataset_split}/{index}',
                             question=question.strip(), reference=str(value))
        if len(selected) == request.max_questions:
            break
    if len(selected) < request.max_questions:
        raise ValueError(f'Only {len(selected)} eligible unique questions; requested {request.max_questions}')
    ordered = sorted(selected.items(), key=lambda item: hashlib.sha256(
        f'slice-api-v1:{request.seed}:{item[0]}'.encode()).hexdigest())
    counts = request.split_counts()
    train_end = counts['train']
    val_end = train_end + counts['val']
    questions = [dict(row, split='train' if i < train_end else 'val' if i < val_end else 'test')
                 for i, (_, row) in enumerate(ordered)]
    return questions, dict(excluded)


def prepare_dataset(request, out, *, encoder_identity=None):
    budget = load_prompt_budget(encoder_identity, request.max_new_tokens) if encoder_identity else None
    def record_budget():
        if budget is not None:
            from pathlib import Path
            from .io import read_json
            path = Path(out) / 'selection.json'
            selected = read_json(path)
            selected['prompt_budget'] = dict(encoder=encoder_identity,
                max_prompt_tokens=budget.limit, context_limit=budget.context_limit,
                max_new_tokens=request.max_new_tokens, truncation=False)
            write_json(path, selected)
    if request.dataset == 'AI-MO/NuminaMath-1.5':
        if (request.question_column != 'problem' or request.reference_column != 'answer'
                or request.dataset_split != 'train' or request.dataset_config is not None):
            raise ValueError('The original Numina recipe requires problem/answer columns and the train split without a config')
        # Delegate rather than duplicate: metadata filters, boxed-reference
        # validation, full eligible-family scan and slice-v1 ordering stay exact.
        from .data import prepare_numina
        kwargs = dict(question_filter=budget) if budget is not None else {}
        prepare_numina(out, request.split_counts(), request.revision, **kwargs)
        record_budget()
        return
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
    questions, exclusions = select_questions(rows, request, question_filter=budget)
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
    record_budget()
