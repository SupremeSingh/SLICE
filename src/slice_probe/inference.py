"""Score a question and optional partial answer with a fitted probe."""
import torch

from .features import load_model, prompt, capture, concatenate
from .io import load_checkpoint
from .model import predict_features


def predict(checkpoint, question, prefix='', device='cpu', model_override=None):
    if not question.strip():
        raise ValueError('A nonempty question is required')
    record = load_checkpoint(checkpoint)
    identity = record['model']
    model, tokenizer, _ = load_model(model_override or identity['id'], identity['revision'], device,
                                     hashes=identity['hashes'])
    ids = tokenizer.encode(prompt(question), add_special_tokens=False)
    ids += tokenizer.encode(prefix, add_special_tokens=False)
    if not ids or len(ids) > model.config.max_position_embeddings:
        raise ValueError('Input exceeds the model context')
    states = capture(model, torch.tensor([ids], device=device), [len(ids)-1], record['layers'])
    x = concatenate(states, record['layers'], record['rms_eps']).numpy()
    score = float(predict_features(record, x, device)[0])
    return dict(score=score, predicted_correct=score >= .5, prefix_tokens=len(ids),
                scope='Estimated eventual correctness under the collection policy; raw Ridge score, not a calibrated probability or proof.')
