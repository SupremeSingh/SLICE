"""Prompt eligibility is checked before splitting; no truncation or outcome filter."""
import sys
from types import SimpleNamespace

import pytest

from slice_probe import datasets
from slice_probe.datasets import PromptBudget, select_questions
from slice_probe.io import read_json, write_json
from slice_probe.recovery import preflight_request
from slice_probe.registry import canonical_hash
from slice_probe.requests import TrainingRequest


class Tokenizer:
    def encode(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False}
        return [1] * (513 if '12345' in text else 512)


def rows():
    names = ('Alder', 'Birch', 'Cedar', 'Dogwood', 'Elm', 'Fir', 'Ginkgo',
             'Hazel', 'Ivy', 'Juniper', 'Kapok', 'Larch', 'Maple', 'Oak', 'Pine')
    good = [dict(problem=f'{name} has 2 apples.', answer='2', solution=r'\boxed{2}',
        source='cn_k12', synthetic=False, question_type='math-word-problem',
        problem_is_valid='Yes', solution_is_valid='Yes') for name in names]
    return [dict(good[0], problem='Alder has 12345 apples.')] + good


def test_prompt_budget_preserves_boundary_and_native_context():
    budget = PromptBudget(Tokenizer(), 4096, 3584)
    assert budget('2') and not budget('12345')
    assert not PromptBudget(Tokenizer(), 4095, 3584)('2')
    with pytest.raises(ValueError, match='no room'):
        PromptBudget(Tokenizer(), 3584, 3584)


def test_numina_filters_overlong_candidates_before_exact_split_selection(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=lambda *a, **kw: rows()))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: SimpleNamespace(dataset_info=lambda *a, **kw: SimpleNamespace(sha='a' * 40))))
    monkeypatch.setattr(datasets, 'load_prompt_budget', lambda identity, tokens:
                        PromptBudget(Tokenizer(), 4096, tokens))
    identity = dict(id='model', revision='b' * 40, hashes={'config.json': 'c' * 64})
    request = TrainingRequest.parse(dict(dataset='AI-MO/NuminaMath-1.5', max_questions=10))
    datasets.prepare_dataset(request, tmp_path / 'questions', encoder_identity=identity)
    selection = read_json(tmp_path / 'questions/selection.json')
    assert selection['counts'] == dict(train=7, val=1, test=2)
    assert selection['exclusions']['prompt_budget'] == 1
    assert selection['eligible'] == 15
    assert selection['prompt_budget']['encoder'] == identity
    text = (tmp_path / 'questions/questions.jsonl').read_text()
    assert '12345' not in text and len(text.splitlines()) == 10


def test_generic_selection_replenishes_rejected_prompts():
    request = TrainingRequest.parse(dict(dataset='owner/data', max_questions=10))
    selected, excluded = select_questions(rows(), request,
        question_filter=PromptBudget(Tokenizer(), 4096, 3584))
    assert len(selected) == 10 and excluded['prompt_budget'] == 1
    assert all('12345' not in q['question'] for q in selected)


def test_retry_cpu_preflight_pins_previous_model_and_dataset(tmp_path, monkeypatch):
    from slice_probe import recovery

    root = tmp_path / 'failed'
    request = TrainingRequest.parse(dict(dataset='owner/data', max_questions=10)).to_dict()
    request.pop('model_revision')  # Read requests written before this field existed.
    digest = canonical_hash(request)
    identity = dict(id='model', revision='a' * 40, hashes={'config.json': 'c' * 64})
    write_json(root / 'request.json', dict(job_id='d' * 32, request=request, request_hash=digest))
    write_json(root / 'smoke/report.json', dict(passed=True, model=identity, request_hash=digest))
    write_json(root / 'questions/selection.json', dict(revision='b' * 40))

    def prepare(request, out, *, encoder_identity):
        assert request.model_revision == 'a' * 40 and request.revision == 'b' * 40
        assert encoder_identity == identity
        write_json(out / 'selection.json', dict(counts=request.split_counts(),
            exclusions={'prompt_budget': 1}, prompt_budget={'max_prompt_tokens': 512}, revision=request.revision))

    monkeypatch.setattr(recovery, 'prepare_dataset', prepare)
    report = preflight_request(root, tmp_path / 'ready')
    assert report['ready'] and report['counts'] == dict(train=7, val=1, test=2)
    pinned = read_json(tmp_path / 'ready/request.json')
    assert pinned['model_revision'] == 'a' * 40 and pinned['revision'] == 'b' * 40
