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
    publish = sub.add_parser('publish', help='Publish an immutable supervised critic artifact')
    publish.add_argument('--checkpoint', type=Path, required=True)
    publish.add_argument('--registry', type=Path, required=True)
    publish.add_argument('--version', required=True)
    for field in ('calibration', 'evaluation', 'feature-manifest', 'statistics', 'trajectory-manifest'):
        publish.add_argument('--' + field, type=Path)
    inspect = sub.add_parser('inspect', help='Verify and inspect a published artifact')
    inspect.add_argument('--artifact', type=Path, required=True)
    for name in ('score-batch', 'encode-batch'):
        batch = sub.add_parser(name, help='Use one resident critic on a JSON list of states')
        batch.add_argument('--artifact', type=Path, required=True)
        batch.add_argument('--input', type=Path, required=True)
        batch.add_argument('--out', type=Path, required=True)
        batch.add_argument('--device', default='auto')
        if name == 'score-batch':
            batch.add_argument('--calibrated', action='store_true')
    trajectories = sub.add_parser('extract-trajectories', help='Extract complete training episodes for optional LSTD')
    for field in ('checkpoint', 'answers', 'feature-manifest', 'out'):
        trajectories.add_argument('--' + field, type=Path, required=True)
    trajectories.add_argument('--device', default='auto')
    lstd = sub.add_parser('fit-lstd', help='Fit LSTD(lambda) without changing the frozen representation')
    for field in ('checkpoint', 'trajectories', 'feature-manifest', 'out'):
        lstd.add_argument('--' + field, type=Path, required=True)
    lstd.add_argument('--trace-lambda', type=float, default=.9)
    lstd.add_argument('--discount-gamma', type=float, default=1.)
    lstd.add_argument('--regularization', type=float, default=.01)
    jobs = sub.add_parser('jobs', help='Operator submission and Slurm reconciliation')
    jobs.add_argument('--config', type=Path, required=True)
    actions = jobs.add_subparsers(dest='action', required=True)
    init = actions.add_parser('init-cluster', help='Configure native Duke submission on this login host')
    init.add_argument('--state-dir', type=Path, help='Node-local ledger directory (default: /var/tmp/slice-controller-UID)')
    init.add_argument('--host', default='compsci-login-01', help='Required initialization host (default: compsci-login-01)')
    backup = actions.add_parser('backup', help='Write a consistent immutable controller snapshot')
    backup.add_argument('--out', type=Path, required=True)
    recover = actions.add_parser('restore', help='Recover a missing local ledger from a checked snapshot')
    recover.add_argument('--backup', type=Path, required=True)
    for name in ('create', 'submit'):
        create = actions.add_parser(name, help='Record a request' if name == 'create' else
                                    'Create and submit once; this command authorizes compute')
        create.add_argument('--request', type=Path, required=True)
        create.add_argument('--project', required=True)
        create.add_argument('--idempotency-key', required=True)
    actions.add_parser('list')
    for name in ('show', 'approve', 'refresh', 'cancel', 'settle'):
        action = actions.add_parser(name)
        action.add_argument('--id', required=True)
    worker = sub.add_parser('job-run', help='Run a prepared job inside its compute allocation')
    worker.add_argument('--job', type=Path, required=True)
    worker.add_argument('--device', default='auto')
    parity = sub.add_parser('parity', help='Compare a completed managed Ridge job with standalone training on its saved data')
    parity.add_argument('--job-dir', type=Path, required=True)
    parity.add_argument('--out', type=Path, required=True)
    parity.add_argument('--device', default='auto')
    parity.add_argument('--rtol', type=float, default=1e-5)
    parity.add_argument('--atol', type=float, default=1e-6)
    original = sub.add_parser('import-original', help='Publish existing original sparse-study weights without retraining')
    for name in ('study', 'model', 'registry', 'out'):
        original.add_argument('--' + name, type=Path, required=True)
    original.add_argument('--prefix', help='Artifact prefix (default: original-sparseWIDTH)')
    original.add_argument('--samples', type=int, default=512)
    extraction = sub.add_parser('extraction-parity', help='Audit fresh Qwen extraction against original project functions')
    for name in ('study', 'model', 'original-root', 'out'):
        extraction.add_argument('--' + name, type=Path, required=True)
    extraction.add_argument('--device', default='cuda')
    extraction.add_argument('--samples', type=int, default=3)
    extraction.add_argument('--rtol', type=float, default=1e-4)
    extraction.add_argument('--atol', type=float, default=1e-5)
    prepare = sub.add_parser('preflight-request', help='CPU-check a retry selection using a previous job’s pinned assets')
    prepare.add_argument('--job-dir', type=Path, required=True)
    prepare.add_argument('--out', type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    import os
    import torch
    threads = int(os.environ.get('SLICE_NUM_THREADS', '8'))
    if threads < 1:
        raise ValueError('SLICE_NUM_THREADS must be positive')
    torch.set_num_threads(threads)
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
    elif args.command == 'publish':
        from .registry import publish
        print(publish(args.checkpoint, args.registry, args.version,
                      calibration=args.calibration, evaluation=args.evaluation,
                      feature_manifest=args.feature_manifest, statistics=args.statistics,
                      trajectory_manifest=args.trajectory_manifest))
    elif args.command == 'inspect':
        from .registry import inspect_artifact
        print(json.dumps(inspect_artifact(args.artifact), indent=2))
    elif args.command in ('score-batch', 'encode-batch'):
        from .critic import Critic
        from .io import read_json, write_json
        critic = Critic.load(args.artifact, device=args.device)
        states = read_json(args.input)
        if not isinstance(states, list):
            raise ValueError('--input must contain a JSON list of state objects')
        if args.command == 'score-batch':
            write_json(args.out, critic.score_batch(states, return_calibrated_probability=args.calibrated))
        else:
            import numpy as np
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('wb') as stream:
                np.save(stream, critic.encode_batch(states))
            write_json(args.out.with_suffix('.json'), dict(feature_id=critic.feature_id,
                critic_version=critic.version, state_ids=[s.get('state_id') for s in states]))
    elif args.command == 'extract-trajectories':
        from .lstd import extract_trajectories
        extract_trajectories(args.checkpoint, args.answers, args.out, args.device,
                             feature_manifest=args.feature_manifest)
    elif args.command == 'fit-lstd':
        from .lstd import fit_lstd
        print(fit_lstd(args.checkpoint, args.trajectories, args.feature_manifest, args.out,
                       trace_lambda=args.trace_lambda, discount_gamma=args.discount_gamma,
                       regularization=args.regularization))
    elif args.command == 'jobs':
        if args.action == 'init-cluster':
            from .controller import init_cluster, backup_ledger
            result = init_cluster(args.config, state_dir=args.state_dir, expected_host=args.host)
            backup_ledger(args.config)
            print(json.dumps(result, indent=2))
            return
        if args.action == 'restore':
            from .controller import restore_ledger, backup_ledger
            result = restore_ledger(args.config, args.backup)
            backup_ledger(args.config)
            print(json.dumps(result, indent=2))
            return
        if args.action == 'backup':
            from .controller import backup_ledger
            print(json.dumps(backup_ledger(args.config, args.out), indent=2))
            return
        from .jobs import JobStore
        from .slurm import settings, Slurm
        from .io import read_json
        from .requests import TrainingRequest
        config = settings(args.config)
        store, scheduler = JobStore(config['database']), Slurm(config)
        if args.action == 'create':
            result = store.create(args.project, args.idempotency_key, read_json(args.request))
        elif args.action == 'submit':
            if config.get('controller_host'):
                from .controller import backup_ledger
                # Save the request identity before sbatch can possibly accept it.
                store.create(args.project, args.idempotency_key, read_json(args.request))
                backup_ledger(args.config)
            result = scheduler.submit(store, args.project, args.idempotency_key, read_json(args.request))
        elif args.action == 'list':
            result = store.list()
        elif args.action == 'show':
            result = store.get(args.id)
            result['preview'] = TrainingRequest.parse(result['request']).preview()
            result['launch'] = scheduler.plan(result)
        elif args.action == 'approve':
            if config.get('controller_host'):
                from .controller import backup_ledger
                backup_ledger(args.config)
            result = scheduler.approve(store, args.id)
        elif args.action == 'refresh':
            result = scheduler.refresh(store, args.id)
        elif args.action == 'settle':
            result = scheduler.settle(store, args.id)
        else:
            result = scheduler.cancel(store, args.id)
        if config.get('controller_host') and args.action in ('create', 'submit', 'approve', 'refresh', 'cancel', 'settle'):
            from .controller import backup_ledger
            backup_ledger(args.config)
        print(json.dumps(result, indent=2))
    elif args.command == 'job-run':
        from .worker import run_job
        run_job(args.job, device=args.device)
    elif args.command == 'parity':
        from .parity import run_parity
        report = run_parity(args.job_dir, args.out, args.device, rtol=args.rtol, atol=args.atol)
        print(json.dumps(dict(report=str(args.out / 'report.json'), matches=report['matches'],
            numerical_matches=report['numerical_matches'], provenance_matches=report['provenance_matches']), indent=2))
        if not report['matches']:
            raise SystemExit(1)
    elif args.command == 'import-original':
        from .original import import_original
        print(json.dumps(import_original(args.study, args.model, args.registry, args.out,
            prefix=args.prefix, samples=args.samples), indent=2))
    elif args.command == 'extraction-parity':
        from .extraction_parity import run_extraction_parity
        report = run_extraction_parity(args.study, args.model, args.original_root, args.out,
            device=args.device, samples=args.samples, rtol=args.rtol, atol=args.atol)
        print(json.dumps(dict(matches=report['matches'], report=str(args.out / 'report.json')), indent=2))
        if not report['matches']:
            raise SystemExit(1)
    elif args.command == 'preflight-request':
        from .recovery import preflight_request
        print(json.dumps(preflight_request(args.job_dir, args.out), indent=2))
