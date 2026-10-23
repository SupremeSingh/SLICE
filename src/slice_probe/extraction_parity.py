"""Fresh-token extraction audit against the original project's actual functions."""
from pathlib import Path
import importlib.util
import math

import numpy as np
import torch

from .critic import Critic
from .features import capture, concatenate
from .io import read_json, require_empty, save_checkpoint, sha256, write_json
from .original import checked_tensor
from .parity import compare_arrays


def original_module(root, relative):
    path = root / relative
    spec = importlib.util.spec_from_file_location('_slice_reference_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    # Execute the actual source without writing __pycache__ into the reference checkout.
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


@torch.inference_mode()
def audit_sequence(critic, token_ids, prompt_length, original_blocks, original_rms,
                   original_compressor, *, rtol=1e-4, atol=1e-5):
    """Compare original hooks/math with the public API, including causal prefixes."""
    response_length = len(token_ids) - prompt_length
    if response_length < 2:
        raise ValueError('Need two continuation tokens for intermediate-prefix parity')
    positions = sorted(set([0, 1, response_length // 2, response_length - 1]))
    lengths = [prompt_length + p for p in positions]
    layers = critic.record['layers']
    zero_layers = [n - 1 for n in layers]
    tokens = torch.tensor([token_ids], device=critic.device)
    reference = original_blocks(critic.encoder, tokens, prompt_length, response_length, positions, zero_layers)
    actual = capture(critic.encoder, tokens, [n - 1 for n in lengths], layers)
    checks = {}

    def compare(name, expected, observed):
        a = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
        b = observed.detach().cpu().numpy() if torch.is_tensor(observed) else np.asarray(observed)
        checks[name] = compare_arrays(a, b, rtol=rtol, atol=atol)

    for layer in layers:
        compare(f'block-{layer}', reference[layer - 1].float(), actual[layer])
    xref = torch.cat([original_rms(reference[n].float(), critic.record['rms_eps']) for n in zero_layers], -1)
    x = concatenate(actual, layers, critic.record['rms_eps'])
    compare('rms_concatenation', xref, x)
    centered = ((xref.to(critic.device) - critic.record['mean'].to(critic.device))
                / critic.record['scale'].to(critic.device))
    compare('centering_scaling', centered, (x.to(critic.device) - critic._mean) / critic._scale)
    zref = original_compressor.features(centered).cpu()
    z = critic.encode_trajectory(token_ids, lengths, tokenizer_id=critic.tokenizer_id)
    compare('compressed_features', zref, z)
    w = critic.record['weights'].double().cpu()
    compare('raw_predictions', zref.double() @ w[:-1] + w[-1],
            z.astype(np.float64) @ w[:-1].numpy() + float(w[-1]))
    scored = critic.score_batch([dict(token_ids=token_ids[:length], tokenizer_id=critic.tokenizer_id)
                                 for length in lengths])
    compare('public_raw_predictions', zref.double() @ w[:-1] + w[-1],
            np.asarray([row['raw_value'] for row in scored], dtype=np.float64))
    for index, length in enumerate(lengths):
        prefix = capture(critic.encoder, tokens[:, :length], [length - 1], layers)
        for layer in layers:
            compare(f'causal-prefix-{length}-block-{layer}', actual[layer][index:index+1], prefix[layer])
        single = critic.encode_trajectory(token_ids[:length], [length], tokenizer_id=critic.tokenizer_id)
        compare(f'causal-prefix-{length}-features', z[index:index+1], single)
    return dict(prefix_lengths=lengths, question_only_prefix=prompt_length, checks=checks,
                matches=all(item['matches'] for item in checks.values()))


def run_extraction_parity(study, model, original_root, out, *, device='cuda', samples=3,
                          rtol=1e-4, atol=1e-5):
    if type(samples) is not int or samples < 1 or any(not math.isfinite(v) or v < 0 for v in (rtol, atol)):
        raise ValueError('Positive samples and finite nonnegative tolerances required')
    study, model, root = (Path(p).resolve() for p in (study, model, original_root))
    protocol = read_json(study / 'protocol.json')
    source = Path(protocol['source']).resolve()
    out = Path(out).resolve()
    if any(out.is_relative_to(p) for p in (study, source, model, root)):
        raise ValueError('Audit output must be outside original inputs')
    manifest = read_json(source / 'manifest.json')
    if sha256(source / 'manifest.json') != protocol['source_manifest']:
        raise ValueError('Original source manifest changed')
    if sha256(source / 'questions.json') != manifest['questions_sha256']:
        raise ValueError('Original questions changed')
    modules = ['src/math_rl/critic_layers.py', 'src/math_rl/direct_jlens.py', 'src/math_rl/sparse_bottleneck.py']
    blocks, rms, bottleneck = [original_module(root, name) for name in modules]
    questions = read_json(source / 'questions.json')
    sequences = []
    # Selection uses availability/length only, never rewards or prediction scores.
    for question in questions:
        path = source / 'trajectories' / f'{question["index"]:05d}.json'
        batch = read_json(path)
        for response_index, row in enumerate(batch['responses']):
            if len(row.get('token_ids', [])) >= 2:
                ids = question['prompt_token_ids'] + row['token_ids'][:128]
                sequences.append(dict(question_id=question['id'], response=response_index,
                    token_ids=ids, prompt_length=len(question['prompt_token_ids']),
                    source_sha256=sha256(path)))
                break
        if len(sequences) == samples:
            break
    if len(sequences) != samples:
        raise ValueError('Insufficient saved trajectories for requested parity sample')
    out = require_empty(out)
    screen = read_json(study / 'screen.json')
    results = {}
    for seed in protocol['seeds']:
        selected = screen['selected'][f'sparse-{seed}']
        path = (study / selected['path']).resolve()
        if not path.is_relative_to(study):
            raise ValueError('Original checkpoint path escapes study')
        comp = checked_tensor(path, selected['sha256'])
        head_path = study / f'sparse-{seed}-ridge-fit.pt'
        head = checked_tensor(head_path)
        record = dict(format='slice-probe-v1', input_dim=len(comp['mean']), width=comp['width'],
            state=comp['state'], mean=comp['mean'], scale=comp['scale'], weights=head['weights'],
            model=dict(id=str(model), revision=None, hashes=manifest['model_hashes']),
            layers=[n+1 for n in manifest['layers']], rms_eps=manifest['rms_eps'])
        checkpoint = out / f'seed{seed}.pt'
        save_checkpoint(checkpoint, record)
        critic = Critic.load(checkpoint, device=device)
        original = bottleneck.Bottleneck(record['input_dim'], record['width']).to(critic.device)
        original.load_state_dict(comp['state'])
        original.eval().requires_grad_(False)
        results[str(seed)] = dict(compressor_sha256=selected['sha256'], head_sha256=sha256(head_path),
            sequences=[dict(sequence, audit=audit_sequence(critic, sequence['token_ids'],
                sequence['prompt_length'], blocks.sampled_block_states, rms.rms_coordinates,
                original, rtol=rtol, atol=atol)) for sequence in sequences])
        identity_before = critic.feature_id
        target = critic.device
        critic.to('cpu')
        cpu_tensors = [*critic.encoder.parameters(), *critic.compressor.parameters(),
                       critic._mean, critic._scale]
        offloaded = all(value.device.type == 'cpu' for value in cpu_tensors)
        del cpu_tensors
        critic.to(target)
        first = sequences[0]
        reloaded = audit_sequence(critic, first['token_ids'], first['prompt_length'],
            blocks.sampled_block_states, rms.rms_coordinates, original, rtol=rtol, atol=atol)
        results[str(seed)]['offload_reload'] = dict(matches=offloaded and reloaded['matches']
            and critic.feature_id == identity_before, reloaded_audit=reloaded)
        del original, critic
    report = dict(matches=all(row['audit']['matches'] for result in results.values() for row in result['sequences']),
        model=record['model'], precision='float32 eager, TF32 disabled', rtol=rtol, atol=atol,
        layers=record['layers'], original_code_sha256={name:sha256(root/name) for name in modules},
        protocol_sha256=sha256(study/'protocol.json'), results=results,
        scope='Fresh extraction through original hooks/RMS/compressor and public SLICE API on identical saved tokens. '
              'Shared hash-verified Qwen backbone; does not audit generation, verifier or dataset-selection equivalence.')
    report['max_absolute_error'] = max(check.get('max_absolute_error', 0.)
        for result in results.values() for row in result['sequences']
        for check in row['audit']['checks'].values())
    report['matches'] = report['matches'] and all(result['offload_reload']['matches'] for result in results.values())
    from .comparison import environment
    report['environment'] = environment()
    write_json(out / 'report.json', report)
    return report
