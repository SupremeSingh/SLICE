"""Separate question difficulty from information in a partial continuation."""
import numpy as np
from .metrics import auc


def question_carry_forward(scores, question, position):
    """Repeat a question's initial score; never use its outcomes to construct it."""
    scores = np.atleast_2d(scores)
    result = np.empty_like(scores)
    for q in np.unique(question):
        initial = (question == q) & (position == 0)
        if not initial.any():
            raise ValueError('Every question needs an initial L=0 state')
        values = scores[:, initial]
        if not np.allclose(values, values[:, :1], rtol=1e-4, atol=1e-5):
            raise ValueError('Question-only scores differ across continuations; check causality/identity')
        result[:, question == q] = values.mean(1, keepdims=True)
    return result


def within_question_ranking(y, scores, question, response, position, repetitions=2000):
    """Rank continuations within each question at identical absolute token counts.

    One score per (question, response, position), so repeated sampled prefixes do
    not create extra pair weight. Only mixed-outcome questions contribute AUROC.
    """
    scores = np.atleast_2d(scores)
    reports = {}
    for step in np.unique(position):
        per_question = []
        answers = 0
        for q in np.unique(question[position == step]):
            mask = (question == q) & (position == step)
            rs = np.unique(response[mask])
            targets, values = [], []
            for r in rs:
                selected = mask & (response == r)
                target = np.unique(y[selected])
                if len(target) != 1:
                    raise ValueError('A continuation has inconsistent outcome labels')
                targets.append(target[0]); values.append(scores[:, selected].mean(1))
            targets = np.asarray(targets)
            if len(np.unique(targets)) < 2:
                continue
            values = np.stack(values, axis=1)
            # L=0 contains the same input for all continuations. Treat rounding noise
            # as a tie, rather than ranking numerical jitter against future outcomes.
            per_question.append(.5 if step==0 else float(np.mean([auc(targets,p) for p in values])))
            answers += len(rs)
        if not per_question:
            reports[str(int(step))] = dict(mixed_questions=0, answers=0, mean_auroc=None,
                descriptive_95_interval=None)
            continue
        rng = np.random.default_rng(117)
        boot = [rng.choice(per_question, len(per_question), replace=True).mean()
                for _ in range(repetitions)]
        reports[str(int(step))] = dict(mixed_questions=len(per_question), answers=answers,
            mean_auroc=float(np.mean(per_question)),
            descriptive_95_interval=np.quantile(boot, [.025, .975]).tolist())
    return dict(by_token_count=reports,
        scope='Equal-question mean AUROC on mixed-outcome questions at the same prefix length. '
              'Heads averaged, not ensembled. Question-only carry-forward ties at AUROC 0.5. '
              'Requires surviving to this length; not a causal effect of reasoning or fresh-data evidence.')
