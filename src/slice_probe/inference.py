"""Score a question and optional partial answer with a fitted probe."""
import torch

from .features import load_model, prompt, capture, concatenate
from .io import load_checkpoint, read_json, sha256


def predict(checkpoint, question, prefix='', device='cpu', model_override=None, calibration=None):
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
    if record['format']=='slice-baseline-v1' and record['representation']!='concat':
        block=record['representation_spec']['block']
        states=capture(model,torch.tensor([ids],device=device),[len(ids)-1],[block])
        x=states[block]
        if record['representation']=='final':
            with torch.inference_mode():x=model.norm(x.to(device)).float().cpu()
    else:
        states=capture(model,torch.tensor([ids],device=device),[len(ids)-1],record['layers'])
        x=concatenate(states,record['layers'],record['rms_eps'])
    from .comparison import predict as predict_readout
    score=float(predict_readout(record,x.numpy(),device)[0])
    result=dict(score=score,predicted_correct=score>=.5,prefix_tokens=len(ids),
        scope='Estimated eventual correctness under the collection policy; raw score, not a calibrated probability or proof.')
    if calibration is not None:
        from .calibration import apply_calibrator
        data=read_json(calibration)
        names=[name for name,digest in data['checkpoint_sha256'].items() if digest==sha256(checkpoint)]
        if len(names)!=1 or names[0] not in data['calibrators']:
            raise ValueError('Calibrator does not identify this checkpoint')
        probability=float(apply_calibrator([score],data['calibrators'][names[0]])[0])
        result.update(calibrated_probability=probability,calibrated_predicted_correct=probability>=.5,
            scope='Validation-calibrated estimate under the collection policy, not a guaranteed probability or proof.')
    return result
