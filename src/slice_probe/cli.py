"""Portable CLI; model and dataset dependencies are loaded only when needed."""
import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(prog='slice-probe', description='Predict eventual answer correctness from frozen language-model states.')
    sub = p.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Synthetic CPU smoke test; no downloads')
    demo.add_argument('--out', type=Path, required=True)
    prepare = sub.add_parser('prepare-numina', help='Select new disjoint Numina question families')
    prepare.add_argument('--out', type=Path, required=True)
    for split, count in [('train',10000),('val',1000),('test',2000)]:
        prepare.add_argument('--'+split, type=int, default=count)
    prepare.add_argument('--revision')
    collect = sub.add_parser('collect', help='Generate complete answers and correctness labels')
    collect.add_argument('--questions', type=Path, required=True)
    collect.add_argument('--out', type=Path, required=True)
    collect.add_argument('--model', default='Qwen/Qwen2.5-Math-1.5B')
    collect.add_argument('--revision')
    collect.add_argument('--device', default='auto')
    collect.add_argument('--responses', type=int, default=8)
    collect.add_argument('--max-new-tokens', type=int, default=3584)
    collect.add_argument('--seed', type=int, default=271828)
    extract = sub.add_parser('extract', help='Save causal prefix features from retained answers')
    extract.add_argument('--answers', type=Path, required=True)
    extract.add_argument('--out', type=Path, required=True)
    extract.add_argument('--device', default='auto')
    extract.add_argument('--layer-base', type=float, default=2.0)
    extract.add_argument('--num-slices', type=int)
    extract.add_argument('--baselines', action='store_true', help='Also save final and two-thirds features on identical states')
    extract.add_argument('--checkpoints', nargs='+', type=int, default=[], help='Extra evaluation-only token counts, e.g. 0 32 128')
    train = sub.add_parser('train', help='Fit compressors and frozen Ridge readouts')
    train.add_argument('--features', type=Path, required=True)
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--width', type=int, default=768)
    train.add_argument('--kinds', nargs='+', choices=['plain','sparse'], default=['plain','sparse'])
    train.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    train.add_argument('--learning-rates', nargs='+', type=float, default=[.0003,.001])
    train.add_argument('--epochs', type=int, default=30)
    train.add_argument('--batch-size', type=int, default=2048)
    train.add_argument('--epsilon', type=float, default=.01)
    train.add_argument('--device', default='auto')
    train.add_argument('--answers', type=Path, help='Matching collection: fit Ridge on every training-token state')
    evaluate = sub.add_parser('evaluate', help='Evaluate locked probes on held-out prefixes')
    evaluate.add_argument('--features', type=Path, required=True)
    evaluate.add_argument('--probes', type=Path, required=True)
    evaluate.add_argument('--out', type=Path, required=True)
    evaluate.add_argument('--device', default='auto')
    infer = sub.add_parser('predict', help='Score a question and optional partial answer')
    infer.add_argument('--checkpoint', type=Path, required=True)
    infer.add_argument('--question', required=True)
    infer.add_argument('--prefix', default='')
    infer.add_argument('--device', default='auto')
    infer.add_argument('--calibration', type=Path, help='Matching locked calibration.json; checkpoint hash must match')
    infer.add_argument('--model', help='Override model location; original hashes must match')
    layers = sub.add_parser('layers', help='Preview geometric block selection without a model download')
    layers.add_argument('--num-layers', type=int, required=True)
    layers.add_argument('--layer-base', type=float, default=2.0)
    layers.add_argument('--num-slices', type=int)
    sample = sub.add_parser('sample', help='Tokenize bundled, hand-labeled example answers; no generation or verifier')
    sample.add_argument('--out', type=Path, required=True)
    sample.add_argument('--model', default='Qwen/Qwen2.5-0.5B')
    sample.add_argument('--revision')
    compare = sub.add_parser('compare', help='Fit all matched-state baselines and regenerate a comparison table')
    inputs = compare.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--answers', type=Path, help='Re-extract matched sampled states from saved answers; not an all-token fit')
    inputs.add_argument('--features', type=Path)
    compare.add_argument('--out', type=Path, required=True)
    compare.add_argument('--config', type=Path)
    compare.add_argument('--device', default='auto')
    compare.add_argument('--figures', action='store_true', help='Plot validation-calibrated reliability and risk–coverage')
    compare.add_argument('--layer-base', type=float, default=2.)
    compare.add_argument('--num-slices', type=int)
    locked = sub.add_parser('comparison-fit', help='Fit/choose probes and lock artifacts before opening test arrays')
    locked.add_argument('--features', type=Path, required=True)
    locked.add_argument('--out', type=Path, required=True)
    locked.add_argument('--config', type=Path)
    locked.add_argument('--device', default='auto')
    tested = sub.add_parser('comparison-evaluate', help='Evaluate sealed comparison artifacts')
    tested.add_argument('--features', type=Path, required=True)
    tested.add_argument('--run', type=Path, required=True)
    tested.add_argument('--out', type=Path, required=True)
    tested.add_argument('--lock-sha256')
    tested.add_argument('--device', default='auto')
    tested.add_argument('--figures', action='store_true')
    bundle = sub.add_parser('release-bundle', help='Export IDs, provenance, checkpoints and per-prefix predictions')
    for field in ('answers','features','run','evaluation','out'):
        bundle.add_argument('--'+field, type=Path, required=True)
    bundle.add_argument('--include-answers', action='store_true')
    bundle.add_argument('--include-features', action='store_true', help='Also copy potentially large arrays for re-evaluation without model extraction')
    verify = sub.add_parser('verify-bundle', help='Recompute a released comparison table without model inference')
    verify.add_argument('--bundle', type=Path, required=True)
    verify.add_argument('--out', type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    import torch
    torch.set_num_threads(8)
    if hasattr(args, 'device'):
        from .features import resolve_device
        args.device = resolve_device(args.device)
    if args.command == 'demo':
        from .demo import run
        run(args.out)
    elif args.command == 'prepare-numina':
        from .data import prepare_numina
        prepare_numina(args.out, dict(train=args.train,val=args.val,test=args.test), args.revision)
    elif args.command == 'collect':
        from .collection import collect
        collect(args.questions,args.out,args.model,args.revision,args.device,args.responses,args.max_new_tokens,args.seed)
    elif args.command == 'extract':
        from .collection import extract
        extract(args.answers,args.out,args.device,args.layer_base,args.num_slices,args.baselines,args.checkpoints)
    elif args.command == 'train':
        from .training import fit
        fit(args.features,args.out,args.width,args.kinds,args.seeds,args.learning_rates,
            args.epochs,args.batch_size,args.epsilon,args.device,args.answers)
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        evaluate(args.features,args.probes,args.out,args.device)
    elif args.command == 'predict':
        from .inference import predict
        print(json.dumps(predict(args.checkpoint,args.question,args.prefix,args.device,args.model,args.calibration),indent=2))
    elif args.command == 'layers':
        from .features import geometric_layers
        print(json.dumps(dict(layers=geometric_layers(args.num_layers,args.layer_base,args.num_slices)),indent=2))
    elif args.command == 'sample':
        from .sample import prepare_sample
        prepare_sample(args.out,args.model,args.revision)
    elif args.command == 'compare':
        from .comparison import compare, fit_comparison, evaluate_comparison
        if args.answers is not None:
            from .io import require_empty
            from .collection import extract
            out = require_empty(args.out)
            features = out/'features'
            extract(args.answers, features, args.device, args.layer_base, args.num_slices,
                    baselines=True, checkpoints=[0,32,128])
            digest = fit_comparison(features, out/'fitted', args.config, args.device)
            evaluate_comparison(features, out/'fitted', out/'evaluation', args.device, digest, args.figures)
        else:
            if args.layer_base != 2. or args.num_slices is not None:
                raise ValueError('Layer selection is already fixed in --features; use --answers to re-extract')
            compare(args.features,args.out,args.config,args.device,args.figures)
    elif args.command == 'comparison-fit':
        from .comparison import fit_comparison
        fit_comparison(args.features,args.out,args.config,args.device)
    elif args.command == 'comparison-evaluate':
        from .comparison import evaluate_comparison
        evaluate_comparison(args.features,args.run,args.out,args.device,args.lock_sha256,args.figures)
    elif args.command == 'release-bundle':
        from .comparison import release_bundle
        release_bundle(args.answers,args.features,args.run,args.evaluation,args.out,args.include_answers,args.include_features)
    elif args.command == 'verify-bundle':
        from .comparison import verify_bundle
        verify_bundle(args.bundle,args.out)
