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
    infer.add_argument('--model', help='Override model location; original hashes must match')
    layers = sub.add_parser('layers', help='Preview geometric block selection without a model download')
    layers.add_argument('--num-layers', type=int, required=True)
    layers.add_argument('--layer-base', type=float, default=2.0)
    layers.add_argument('--num-slices', type=int)
    sample = sub.add_parser('sample', help='Tokenize bundled, hand-labeled example answers; no generation or verifier')
    sample.add_argument('--out', type=Path, required=True)
    sample.add_argument('--model', default='Qwen/Qwen2.5-0.5B')
    sample.add_argument('--revision')
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
        extract(args.answers,args.out,args.device,args.layer_base,args.num_slices)
    elif args.command == 'train':
        from .training import fit
        fit(args.features,args.out,args.width,args.kinds,args.seeds,args.learning_rates,
            args.epochs,args.batch_size,args.epsilon,args.device,args.answers)
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        evaluate(args.features,args.probes,args.out,args.device)
    elif args.command == 'predict':
        from .inference import predict
        print(json.dumps(predict(args.checkpoint,args.question,args.prefix,args.device,args.model),indent=2))
    elif args.command == 'layers':
        from .features import geometric_layers
        print(json.dumps(dict(layers=geometric_layers(args.num_layers,args.layer_base,args.num_slices)),indent=2))
    elif args.command == 'sample':
        from .sample import prepare_sample
        prepare_sample(args.out,args.model,args.revision)
