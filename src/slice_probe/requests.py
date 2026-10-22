"""Bounded supervised training requests, independent of the HTTP transport."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import re

from .features import geometric_layers


@dataclass(frozen=True)
class TrainingRequest:
    dataset: str
    question_column: str = 'problem'
    reference_column: str = 'answer'
    dataset_config: str | None = None
    dataset_split: str = 'train'
    revision: str | None = None
    max_questions: int = 1000
    responses_per_question: int = 4
    max_new_tokens: int = 1024
    layer_base: float = 2.
    num_slices: int = 5
    compression_width: int = 768
    epochs: int = 30
    seed: int = 42
    head: str = 'ridge'
    trace_lambda: float = .9
    discount_gamma: float = 1.
    regularization: float = .01
    wall_minutes: int = 240

    @classmethod
    def parse(cls, value):
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown training settings: {sorted(unknown)}')
        try:
            request = cls(**value)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(request.dataset, str) or not re.fullmatch(
                r'[\w.-]+/[\w.-]+', request.dataset, flags=re.ASCII):
            raise ValueError('dataset must be a Hugging Face owner/repository ID')
        for name in ('question_column', 'reference_column', 'dataset_split'):
            if not isinstance(getattr(request, name), str) or not getattr(request, name).strip():
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('dataset_config', 'revision'):
            if getattr(request, name) is not None and not isinstance(getattr(request, name), str):
                raise ValueError(f'{name} must be a string')
        bounds = dict(max_questions=(10, 1000), responses_per_question=(2, 4),
                      max_new_tokens=(1, 3584), num_slices=(1, 28),
                      compression_width=(1, 768), epochs=(1, 30),
                      seed=(0, 2**31 - 1), wall_minutes=(1, 1440))
        for name, (low, high) in bounds.items():
            value = getattr(request, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}]')
        for name in ('layer_base', 'trace_lambda', 'discount_gamma', 'regularization'):
            value = getattr(request, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        geometric_layers(28, request.layer_base, request.num_slices)
        if request.head not in ('ridge', 'lstd'):
            raise ValueError('head must be ridge or lstd')
        if not 0 <= request.trace_lambda <= 1 or not 0 <= request.discount_gamma <= 1:
            raise ValueError('lambda and gamma must be in [0, 1]')
        if request.regularization <= 0:
            raise ValueError('regularization must be positive')
        # The published correctness evaluation compares scores with binary final
        # outcomes, so managed jobs use an undiscounted target. Low-level LSTD
        # supports gamma < 1, with explicitly discounted targets.
        if request.discount_gamma != 1:
            raise ValueError('Managed correctness jobs require gamma=1; use low-level LSTD for discounted targets')
        return request

    def to_dict(self):
        return asdict(self)

    def preview(self):
        layers = geometric_layers(28, self.layer_base, self.num_slices)
        return dict(model='Qwen/Qwen2.5-Math-1.5B', layers=layers,
                    input_width=1536 * len(layers), compression_width=self.compression_width,
                    max_questions=self.max_questions,
                    max_generated_answers=self.max_questions * self.responses_per_question,
                    max_generated_tokens=self.max_questions * self.responses_per_question * self.max_new_tokens,
                    wall_minutes=self.wall_minutes, head=self.head,
                    target='verified final-answer correctness',
                    note='Pilot limits bound work, not prediction accuracy. Approval is required.')
