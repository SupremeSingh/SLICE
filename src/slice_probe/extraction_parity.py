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


def numerical_comparison(expected, observed, *, rtol, atol):
    """Keep the acceptance tolerance fixed and expose the scale of any failure."""
    a = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
    b = observed.detach().cpu().numpy() if torch.is_tensor(observed) else np.asarray(observed)
    result = compare_arrays(a, b, rtol=rtol, atol=atol)
    if a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all():
        reference, actual = a.astype(np.float64), b.astype(np.float64)
        delta = np.abs(reference - actual)
        allowance = atol + rtol * np.abs(reference)
        index = np.unravel_index(np.argmax(delta), delta.shape)
        result.update(elements=int(a.size), outside_tolerance=int(np.count_nonzero(delta > allowance)),
            reference_rms=float(np.sqrt(np.mean(reference ** 2))),
            relative_l2_error=float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30)),
            worst_absolute=dict(index=list(map(int, index)), expected=float(reference[index]),
                observed=float(actual[index]), allowance=float(allowance[index])))
    return result


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
    checks, categories = {}, {}

    def compare(name, expected, observed, category='source_implementation'):
        checks[name] = numerical_comparison(expected, observed, rtol=rtol, atol=atol)
        checks[name]['category'] = category
        categories.setdefault(category, []).append(name)

    for layer in layers:
        compare(f'block-{layer}', reference[layer - 1].float(), actual[layer])
    xref = torch.cat([original_rms(reference[n].float(), critic.record['rms_eps']) for n in zero_layers], -1)
    x = concatenate(actual, layers, critic.record['rms_eps'])
    compare('rms_concatenation', xref, x)
    centered = ((xref.to(critic.device) - critic.record['mean'].to(critic.device))
                / critic.record['scale'].to(critic.device))
    compare('centering_scaling', centered, (x.to(critic.device) - critic._mean) / critic._scale)
    zref = original_compressor.features(centered).cpu()
    # Compare compressors on the SAME captured forward, independently of the
    # public API's intentional truncation to the largest requested prefix.
    zsame = critic.compressor.features((x.to(critic.device) - critic._mean) / critic._scale).cpu()
    compare('compressed_features', zref, zsame)
    z = critic.encode_trajectory(token_ids, lengths, tokenizer_id=critic.tokenizer_id)
    compare('public_trajectory_features', zref, z, 'prefix_execution')
    w = critic.record['weights'].double().cpu()
    compare('raw_predictions', zref.double() @ w[:-1] + w[-1],
            zsame.double() @ critic._weights[:-1] + critic._weights[-1])
    scored = critic.score_batch([dict(token_ids=token_ids[:length], tokenizer_id=critic.tokenizer_id)
                                 for length in lengths])
    compare('public_raw_predictions', zref.double() @ w[:-1] + w[-1],
            np.asarray([row['raw_value'] for row in scored], dtype=np.float64), 'prefix_execution')
    for index, length in enumerate(lengths):
        # Change only future IDs while keeping every tensor shape identical.
        # This isolates causal masking from shape-dependent floating-point math.
        altered = tokens.clone()
        vocab = getattr(critic.encoder.config, 'vocab_size', max(token_ids) + 2)
        altered[:, length:] = (altered[:, length:] + 1) % vocab
        unchanged = capture(critic.encoder, altered, [n - 1 for n in lengths], layers)
        prefix = capture(critic.encoder, tokens[:, :length], [length - 1], layers)
        for layer in layers:
            compare(f'fixed-shape-prefix-{length}-block-{layer}', actual[layer][index:index+1],
                    unchanged[layer][index:index+1], 'fixed_shape_causality')
            compare(f'causal-prefix-{length}-block-{layer}', actual[layer][index:index+1],
                    prefix[layer], 'prefix_execution')
        single = critic.encode_trajectory(token_ids[:length], [length], tokenizer_id=critic.tokenizer_id)
        compare(f'causal-prefix-{length}-features', z[index:index+1], single, 'prefix_execution')
    return dict(prefix_lengths=lengths, question_only_prefix=prompt_length, checks=checks,
                category_matches={category:all(checks[name]['matches'] for name in names)
                                  for category, names in categories.items()},
                matches=all(item['matches'] for item in checks.values()))


def select_sequences(source, questions, samples, long_samples=0, long_min_tokens=3584, context_limit=None):
    """Select genuine saved trajectories by length only; never pad long inputs."""
    if (type(samples) is not int or samples < 1 or type(long_samples) is not int or long_samples < 0
            or type(long_min_tokens) is not int or long_min_tokens < 3):
        raise ValueError('Positive short samples, nonnegative long samples and a valid long-token minimum required')
    if long_samples and (type(context_limit) is not int or context_limit < long_min_tokens):
        raise ValueError('Long-token minimum must fit the model context')
    short, long = [], []
    for question in questions:
        path = source / 'trajectories' / f'{question["index"]:05d}.json'
        batch = read_json(path)
        short_found = long_found = False
        for response_index, row in enumerate(batch['responses']):
            tail = row.get('token_ids', [])
            prompt = question['prompt_token_ids']
            if len(tail) < 2:
                continue
            def entry(ids, kind):
                return dict(question_id=question['id'], response=response_index,
                    token_ids=ids, prompt_length=len(prompt), source_sha256=sha256(path),
                    sample_kind=kind, token_count=len(ids), saved_token_count=len(prompt)+len(tail))
            if len(short) < samples and not short_found:
                short.append(entry(prompt + tail[:128], 'short'))
                short_found = True
            if (len(long) < long_samples and not long_found
                    and long_min_tokens <= len(prompt) + len(tail) <= context_limit):
                long.append(entry(prompt + tail, 'long'))
                long_found = True
        if len(short) == samples and len(long) == long_samples:
            return short + long
    raise ValueError(f'Insufficient genuine saved trajectories: short {len(short)}/{samples}, '
                     f'long {len(long)}/{long_samples}; long inputs are never padded or repeated')


def run_extraction_parity(study, model, original_root, out, *, device='cuda', samples=3,
                          rtol=1e-4, atol=1e-5, long_samples=0, long_min_tokens=3584):
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
    context_limit = read_json(model / 'config.json').get('max_position_embeddings')
    sequences = select_sequences(source, questions, samples, long_samples, long_min_tokens, context_limit)
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
        first = sequences[0]
        lengths = results[str(seed)]['sequences'][0]['audit']['prefix_lengths']
        before_reload = critic.encode_trajectory(first['token_ids'], lengths, tokenizer_id=critic.tokenizer_id)
        target = critic.device
        critic.to('cpu')
        cpu_tensors = [*critic.encoder.parameters(), *critic.compressor.parameters(),
                       critic._mean, critic._scale]
        offloaded = all(value.device.type == 'cpu' for value in cpu_tensors)
        del cpu_tensors
        critic.to(target)
        after_reload = critic.encode_trajectory(first['token_ids'], lengths, tokenizer_id=critic.tokenizer_id)
        reloaded = numerical_comparison(before_reload, after_reload, rtol=rtol, atol=atol)
        results[str(seed)]['offload_reload'] = dict(matches=offloaded and reloaded['matches']
            and critic.feature_id == identity_before, all_feature_tensors_offloaded=offloaded,
            feature_id_preserved=critic.feature_id == identity_before, features=reloaded)
        del original, critic
    report = dict(matches=all(row['audit']['matches'] for result in results.values() for row in result['sequences']),
        model=record['model'], precision='float32 eager, TF32 disabled', rtol=rtol, atol=atol,
        coverage=dict(short_samples=samples, long_samples=long_samples,
                      long_min_tokens=long_min_tokens if long_samples else None,
                      context_limit=context_limit, token_counts=[row['token_count'] for row in sequences]),
        layers=record['layers'], original_code_sha256={name:sha256(root/name) for name in modules},
        protocol_sha256=sha256(study/'protocol.json'), results=results,
        scope='Fresh extraction through original hooks/RMS/compressor and public SLICE API on identical saved tokens. '
              'Shared hash-verified Qwen backbone; does not audit generation, verifier or dataset-selection equivalence.')
    report['max_absolute_error'] = max(check.get('max_absolute_error', 0.)
        for result in results.values() for row in result['sequences']
        for check in row['audit']['checks'].values())
    report['max_absolute_error'] = max(report['max_absolute_error'],
        max(result['offload_reload']['features'].get('max_absolute_error', 0.) for result in results.values()))
    category_names = ('source_implementation', 'fixed_shape_causality', 'prefix_execution')
    report['category_matches'] = {name:all(row['audit']['category_matches'][name]
        for result in results.values() for row in result['sequences']) for name in category_names}
    report['category_matches']['offload_reload'] = all(result['offload_reload']['matches'] for result in results.values())
    report['matches'] = all(report['category_matches'].values())
    report['failed_checks'] = [dict(seed=seed, sequence=index, check=name, **check)
        for seed, result in results.items() for index, row in enumerate(result['sequences'])
        for name, check in row['audit']['checks'].items() if not check['matches']]
    report['comparison_policy'] = ('Acceptance tolerances are unchanged across categories. '
        'Source checks use the same sequence extent; fixed-shape causality changes only future tokens; '
        'prefix execution measures changes in forward length and compressor batch shape; '
        'offload compares identical API calls before and after CPU/GPU transfer.')
    from .comparison import environment
    report['environment'] = environment()
    write_json(out / 'report.json', report)
    return report
