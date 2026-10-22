"""Matched-state baselines, locked validation selection, and test comparison."""
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import csv
import platform
import shutil
import numpy as np
import torch
from torch import nn

from .collection import load_split
from .training import normalization, fit
from .model import predict_features
from .evaluation import metric_average, paired_interval
from .diagnostics import question_carry_forward, within_question_ranking
from .calibration import fit_platt, apply_calibrator, select_acceptance, calibrated_report
from .io import read_json,write_json,sha256,require_empty,save_checkpoint,load_checkpoint

DEFAULTS=dict(width=768,resnet_width=256,epochs=30,batch_size=2048,
    seeds=[42,43,44],learning_rates=[.0003,.001],ridge_epsilon=.01,
    calibration='platt',mode='exploratory',test_scope='Freshness not established automatically')


def configuration(path=None):
    cfg=dict(DEFAULTS)
    supplied=read_json(path) if path else {}
    if set(supplied)-set(cfg):raise ValueError(f'Unknown settings: {set(supplied)-set(cfg)}')
    cfg.update(supplied)
    if cfg['mode'] not in ('exploratory','confirmation') or cfg['calibration'] not in ('none','platt'):
        raise ValueError('Invalid mode or calibration setting')
    if min(cfg['width'],cfg['resnet_width'],cfg['epochs'],cfg['batch_size'])<1:
        raise ValueError('Positive dimensions and fitting budgets required')
    if not cfg['seeds'] or len(set(cfg['seeds']))!=len(cfg['seeds']):raise ValueError('Unique seeds required')
    if not cfg['learning_rates'] or any(not np.isfinite(x) or x<=0 for x in cfg['learning_rates']):
        raise ValueError('Positive finite learning rates required')
    if not np.isfinite(cfg['ridge_epsilon']) or cfg['ridge_epsilon']<=0:raise ValueError('Positive epsilon required')
    return cfg


def environment():
    packages={}
    for name in ('numpy','torch','transformers','datasets','pyarrow','math-verify','slice-probe'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    return dict(python=platform.python_version(),platform=platform.platform(),packages=packages,
        threads=torch.get_num_threads(),cuda=torch.version.cuda,
        source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})


class ResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))
    def forward(self,x):return x+self.net(x)/2


def residual_probe(dimension,width):
    # 1 input + 4 residual blocks with 2 linear maps each + 1 output = 10 maps.
    return nn.Sequential(nn.Linear(dimension,width),nn.GELU(),
        *(ResidualBlock(width) for _ in range(4)),nn.Linear(width,1))


@torch.inference_mode()
def predict(record,x,device='cpu',batch_size=1024):
    if record['format']=='slice-probe-v1':return predict_features(record,x,device,batch_size)
    if record['format']!='slice-baseline-v1':raise ValueError('Unknown baseline format')
    mean,scale=record['mean'].to(device),record['scale'].to(device)
    if record['architecture']=='resnet10':
        model=residual_probe(record['input_dim'],record['width']).to(device).eval().requires_grad_(False)
        model.load_state_dict(record['state'])
    predictions=[]
    for start in range(0,len(x),batch_size):
        values=torch.tensor(np.array(x[start:start+batch_size],copy=True),dtype=torch.float32,device=device)
        values=(values-mean)/scale
        if record['architecture']=='ridge':
            w=record['weights'];p=values.cpu().double()@w[:-1]+w[-1]
        else:p=model(values).flatten().sigmoid().cpu()
        predictions.append(p.cpu().numpy())
    result=np.concatenate(predictions)
    if not np.isfinite(result).all():raise ValueError('Nonfinite baseline prediction')
    return result


def matrix_fit(x,y,mean,scale,epsilon):
    """Stream raw sampled-state sums; no full standardized design copy."""
    from .training import add_statistics,solve_ridge
    size=x.shape[1]+1
    a=torch.zeros(size,size,dtype=torch.float64);b=torch.zeros(size,dtype=torch.float64)
    for start in range(0,len(x),2048):
        values=torch.tensor(np.array(x[start:start+2048],copy=True))
        add_statistics(a,b,(values-mean)/scale,y[start:start+2048])
    return solve_ridge(a,b,epsilon)


def fit_baseline(features,representation,out,cfg,device,architecture):
    train,manifest=load_split(features,'train',representation=representation)
    val,_=load_split(features,'val',representation=representation)
    mean,scale=normalization(train['x']);selections={}
    for seed in (cfg['seeds'] if architecture=='resnet10' else [0]):
        base=dict(format='slice-baseline-v1',architecture=architecture,representation=representation,
            input_dim=train['x'].shape[1],mean=mean,scale=scale,seed=seed,
            feature_manifest_sha256=sha256(Path(features)/'manifest.json'),
            model=manifest['model'],layers=manifest['layers'],rms_eps=manifest['rms_eps'],
            representation_spec=manifest['representations'][representation],
            ridge_states='sampled',state_weighting='one row, one weight; identical sampled states')
        if architecture=='ridge':
            base['weights'],base['fit_checks']=matrix_fit(train['x'],train['y'],mean,scale,cfg['ridge_epsilon'])
            chosen=base
        else:
            candidates=[]
            for lr in cfg['learning_rates']:
                torch.manual_seed(seed)
                model=residual_probe(base['input_dim'],cfg['resnet_width']).to(device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
                generator=torch.Generator().manual_seed(seed);best=float('inf');state=None;curve=[]
                for epoch in range(cfg['epochs']):
                    model.train()
                    for rows in torch.randperm(len(train['y']),generator=generator).split(cfg['batch_size']):
                        x=torch.tensor(np.array(train['x'][rows.numpy()],copy=True),device=device)
                        y=torch.tensor(np.array(train['y'][rows.numpy()],copy=True),device=device)
                        logits=model((x-mean.to(device))/scale.to(device)).flatten()
                        loss=nn.functional.binary_cross_entropy_with_logits(logits,y)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite residual loss')
                        optimizer.zero_grad();loss.backward();optimizer.step()
                    state_now={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    candidate=dict(base,state=state_now,width=cfg['resnet_width'])
                    error=float(np.square(predict(candidate,val['x'],device)-val['y']).mean())
                    curve.append(error)
                    if error<best:best=error;state=state_now
                    print(f'resnet10 seed={seed} lr={lr} epoch={epoch+1} val_brier={error:.5f}',flush=True)
                candidates.append((best,lr,dict(base,state=state,width=cfg['resnet_width'],
                    validation_curve=curve,selected_epoch=int(np.argmin(curve))+1,learning_rate=lr)))
            chosen=min(candidates,key=lambda item:(item[0],item[1]))[2]
            chosen['validation_trials']=[dict(brier=score,learning_rate=lr) for score,lr,_ in candidates]
        name=f'{representation}-{architecture}-seed{seed}'
        path=out/f'{name}.pt';save_checkpoint(path,chosen)
        selections[name]=dict(path=path.name,sha256=sha256(path),representation=representation,
            group=f'{representation}/{architecture}',seed=seed)
    return selections


def fit_comparison(features,out,config=None,device='cpu'):
    """Never load test arrays here. Seal selected artifacts before any evaluation."""
    cfg=configuration(config);manifest=read_json(Path(features)/'manifest.json')
    if not {'concat','final','two_thirds'}<=set(manifest.get('representations',{})):
        raise ValueError('Rerun extract with --baselines before comparing all representations')
    train,_=load_split(features,'train');val,_=load_split(features,'val')
    if set(train['question'])&set(val['question']):raise ValueError('Training/validation question overlap')
    if any(len(np.unique(d['y']))!=2 for d in (train,val)):raise ValueError('Both outcomes required')
    out=require_empty(out);write_json(out/'config.json',cfg)
    write_json(out/'environment.json',environment())
    selections={}
    for representation in ('final','two_thirds','concat'):
        selections.update(fit_baseline(features,representation,out,cfg,device,'ridge'))
    selections.update(fit_baseline(features,'concat',out,cfg,device,'resnet10'))
    fitted=fit(features,out/'slice',width=cfg['width'],seeds=cfg['seeds'],
        learning_rates=cfg['learning_rates'],epochs=cfg['epochs'],batch_size=cfg['batch_size'],
        epsilon=cfg['ridge_epsilon'],device=device)
    for name,item in fitted.items():
        selections[name]=dict(item,path='slice/'+item['path'],representation='concat',
            group=name.rsplit('-',1)[0]+'/ridge',seed=int(name.rsplit('-',1)[1]))
    calibrators={};operating_points={}
    for name,item in selections.items():
        if cfg['calibration']=='none':continue
        data,_=load_split(features,'val',representation=item['representation'])
        raw=predict(load_checkpoint(out/item['path']),data['x'],device)
        calibrators[name]=fit_platt(raw,data['y'])
        operating_points[name]=select_acceptance(data['y'],apply_calibrator(raw,calibrators[name]))
    write_json(out/'calibration.json',dict(calibrators=calibrators,operating_points=operating_points,
        checkpoint_sha256={name:item['sha256'] for name,item in selections.items()}))
    write_json(out/'selection.json',dict(selected=selections))
    artifacts={}
    for item in selections.values():
        for path in (out/item['path'],(out/item['path']).with_suffix('.sha256.json')):
            artifacts[str(path.relative_to(out))]=sha256(path)
    for path in ('config.json','environment.json','calibration.json','selection.json','slice/selection.json'):
        artifacts[path]=sha256(out/path)
    lock=dict(format='slice-comparison-lock-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        feature_manifest_sha256=sha256(Path(features)/'manifest.json'),artifacts=artifacts,
        config=cfg,training_state_distribution='matched sampled prefixes for every method',
        scope='Selected and locked using training/validation only. Test freshness is a user assertion, '
              'not established by hashes. Report historical all-token fitting separately.')
    write_json(out/'lock.json',lock)
    digest=sha256(out/'lock.json');write_json(out/'lock.sha256.json',dict(sha256=digest))
    print(f'LOCK SHA256: {digest}',flush=True)
    return digest


def checked_lock(run,features,expected=None):
    run=Path(run);path=run/'lock.json';actual=sha256(path)
    if actual!=read_json(run/'lock.sha256.json')['sha256'] or (expected and actual!=expected):
        raise ValueError('Comparison lock changed')
    lock=read_json(path)
    if lock['format']!='slice-comparison-lock-v1':raise ValueError('Unknown comparison lock')
    if lock['config']['mode']=='confirmation' and expected is None:
        raise ValueError('Confirmation requires --lock-sha256 from the pre-test fitting step')
    if lock['feature_manifest_sha256']!=sha256(Path(features)/'manifest.json'):
        raise ValueError('Locked feature manifest changed')
    for name,digest in lock['artifacts'].items():
        path=run/name
        if not path.resolve().is_relative_to(run.resolve()):raise ValueError('Artifact outside run directory')
        if sha256(path)!=digest:raise ValueError(f'Locked artifact changed: {name}')
    if lock['config']['mode']=='confirmation':
        old=read_json(run/'environment.json');current=environment()
        if old['packages']!=current['packages']:
            raise ValueError('Dependency versions changed since the confirmation lock')
        if old['source_sha256']!=current['source_sha256']:
            raise ValueError('Code changed since the confirmation lock; use the recorded code version')
    return lock,actual


def evaluate_comparison(features,run,out,device='cpu',lock_sha256=None,figures=False):
    """Evaluate the sealed model set once; all test arrays are first opened here."""
    lock,digest=checked_lock(run,features,lock_sha256);run=Path(run)
    datasets={rep:load_split(features,'test',representation=rep)[0] for rep in ('concat','final','two_thirds')}
    test=datasets['concat'];manifest=read_json(Path(features)/'manifest.json')
    train,_=load_split(features,'train');val,_=load_split(features,'val')
    sets=[set(d['question']) for d in (train,val,test)]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Question split leakage')
    for data in datasets.values():
        if any(not np.array_equal(data[k],test[k]) for k in ('y','question','response','position')):
            raise ValueError('Baseline prefix states are not matched')
    selection=read_json(run/'selection.json')['selected'];raw={};groups={};by_head={};calibrated={}
    calibration=read_json(run/'calibration.json');records={}
    for name,item in selection.items():
        record=load_checkpoint(run/item['path']);records[name]=record
        if record['feature_manifest_sha256']!=lock['feature_manifest_sha256']:
            raise ValueError('Selected probe was fitted on different features')
        raw[name]=predict(record,datasets[item['representation']]['x'],device)
        groups.setdefault(item['group'],[]).append(raw[name])
        if name in calibration['calibrators']:
            calibrated[name],by_head[name]=calibrated_report(test['y'],raw[name],
                calibration['calibrators'][name],calibration['operating_points'][name])
    groups={name:np.stack(values) for name,values in groups.items()}
    groups['constant']=np.full((1,len(test['y'])),float(train['y'].mean()))
    results={name:metric_average(test['y'],p) for name,p in groups.items()}
    carry={name:question_carry_forward(p,test['question'],test['position']) for name,p in groups.items()}
    advancing=test['position']>0
    difficulty={name:dict(initial_carried_forward=metric_average(test['y'],carry[name]),
        updated_minus_initial=paired_interval(test['y'][advancing],p[:,advancing],
            carry[name][:,advancing],test['question'][advancing]) if advancing.any() else None)
        for name,p in groups.items()}
    anchors={};anchor_predictions={};within={}
    if manifest.get('anchor_counts',{}).get('test',0):
        anchor_data={rep:load_split(features,'test',representation=rep,anchors=True)[0] for rep in datasets}
        anchor=anchor_data['concat'];anchor_groups={}
        for name,item in selection.items():
            p=predict(records[name],anchor_data[item['representation']]['x'],device)
            anchor_predictions[name]=p;anchor_groups.setdefault(item['group'],[]).append(p)
        for group,values in anchor_groups.items():
            values=np.stack(values);initial=question_carry_forward(values,anchor['question'],anchor['position'])
            within[group]=within_question_ranking(anchor['y'],values,anchor['question'],anchor['response'],anchor['position'])
            anchors[group]={}
            for step in np.unique(anchor['position']):
                mask=anchor['position']==step
                anchors[group][str(int(step))]=dict(prefixes=int(mask.sum()),questions=len(set(anchor['question'][mask])),
                    updated=metric_average(anchor['y'][mask],values[:,mask]),
                    initial=metric_average(anchor['y'][mask],initial[:,mask]),
                    paired=paired_interval(anchor['y'][mask],values[:,mask],initial[:,mask],anchor['question'][mask]))
    else:
        within={'scope':'No fixed token-count diagnostics cached; extract --checkpoints 0 32 128.'}
    comparisons={f'{name}_minus_concat_ridge':paired_interval(test['y'],p,groups['concat/ridge'],test['question'])
                 for name,p in groups.items() if name not in ('concat/ridge','constant')}
    out=require_empty(out)
    write_json(out/'evaluation-environment.json',environment())
    write_json(out/'summary.json',dict(results=results,comparisons=comparisons,question_difficulty=difficulty,
        fixed_token_checkpoints=anchors,within_question=within,calibrated_by_head=by_head,
        lock_sha256=digest,mode=lock['config']['mode'],test_scope=lock['config']['test_scope'],
        test_questions=len(sets[2]),test_prefixes=len(test['y']),
        feature_manifest_sha256=lock['feature_manifest_sha256'],
        scope='Matched sampled states and equal row weights for all methods. Means across heads, not ensembles. '
              'Carry-forward controls question difficulty; fixed-position within-question AUROC compares continuations. '
              'Question intervals are descriptive and unadjusted. Verifiable-answer conditional; '
              'fixed-position comparisons condition on surviving to the checkpoint. No causal or runtime claim.'))
    np.savez_compressed(out/'predictions.npz',**raw,y=test['y'],question=test['question'],
        response=test['response'],position=test['position'],
        **{f'calibrated-{name}':p for name,p in calibrated.items()},
        **{f'initial-{name}':p for name,p in carry.items()})
    if anchor_predictions:
        np.savez_compressed(out/'checkpoint-predictions.npz',**anchor_predictions,
            **{k:anchor[k] for k in ('y','question','response','position')})
    with (out/'metrics.csv').open('w',newline='') as stream:
        fields=['representation','accuracy','brier','auroc','ece','outside_probability_range']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows(dict(representation=name,**values['mean']) for name,values in results.items())
    lines=['Matched sampled-prefix comparison','representation | accuracy | Brier | AUROC']
    for name,result in results.items():
        r=result['mean'];auc='NA' if r['auroc'] is None else f'{r["auroc"]:.4f}'
        lines.append(f'{name} | {r["accuracy"]:.4f} | {r["brier"]:.5f} | {auc}')
    lines+=['','See summary.json for carry-forward and fixed-position within-question tests.',
            'Raw scores remain available; percentages of confidence require validation calibration.',
            f'Pre-test lock: {digest}']
    (out/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
    if figures:plot_decisions(out)
    return results


def plot_decisions(evaluation):
    """Optional plots of validation-calibrated test probabilities; no fitting here."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(evaluation);data=read_json(out/'summary.json')['calibrated_by_head']
    if not data:raise ValueError('No calibrated predictions; use calibration=platt')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for name,row in data.items():
        bins=row['metrics']['calibration'];curve=row['risk_coverage']
        axes[0].plot([b['prediction'] for b in bins],[b['success_rate'] for b in bins],'.-',label=name)
        axes[1].plot([r['coverage'] for r in curve],[r['risk'] for r in curve],label=name)
    axes[0].plot([0,1],[0,1],'k--',linewidth=1)
    axes[0].set(xlabel='Calibrated predicted correctness',ylabel='Observed correctness',xlim=(0,1),ylim=(0,1))
    axes[1].set(xlabel='Accepted fraction',ylabel='Error rate among accepted',xlim=(0,1),ylim=(0,1))
    axes[1].legend(fontsize=6);fig.tight_layout()
    for suffix in ('svg','png','pdf'):fig.savefig(out/f'calibration-and-risk.{suffix}',dpi=300)
    plt.close(fig)


def compare(features,out,config=None,device='cpu',figures=False):
    out=require_empty(out)
    digest=fit_comparison(features,out/'fitted',config,device)
    return evaluate_comparison(features,out/'fitted',out/'evaluation',device,digest,figures)


def release_bundle(answers,features,run,evaluation,out,include_answers=False,include_features=False):
    """Copy standalone artifacts for review; no external project imports or symlinks."""
    run=Path(run);out=Path(out).resolve()
    for source in (answers,features,run,evaluation):
        source=Path(source).resolve()
        if out==source or source in out.parents:
            raise ValueError('Release output must be separate from every input directory')
    lock,digest=checked_lock(run,features,sha256(run/'lock.json'))
    if read_json(Path(evaluation)/'summary.json')['lock_sha256']!=digest:
        raise ValueError('Evaluation does not belong to the locked comparison')
    manifest=read_json(Path(features)/'manifest.json')
    if manifest.get('collection_manifest_sha256')!=sha256(Path(answers)/'manifest.json'):
        raise ValueError('Answer collection and feature cache differ')
    if manifest.get('questions_sha256') != sha256(Path(answers)/'questions.json'):
        raise ValueError('Question IDs/splits changed since feature extraction')
    collection = read_json(Path(answers)/'manifest.json')
    question_source = Path(answers)/'source-questions.jsonl'
    if question_source.exists() and sha256(question_source) != collection['questions_sha256']:
        raise ValueError('Source question selection changed since collection')
    if include_features:
        for name, checksum in manifest['files'].items():
            path = (Path(features)/name).resolve()
            if not path.is_relative_to(Path(features).resolve()) or sha256(path) != checksum:
                raise ValueError(f'Feature changed before release: {name}')
    out=require_empty(out)
    for file in ('manifest.json','questions.json','source-questions.jsonl'):
        source=Path(answers)/file
        if not source.exists():continue
        dest=out/'collection'/file;dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest)
    shutil.copy2(Path(features)/'manifest.json',out/'feature-manifest.json')
    for file in (*lock['artifacts'],'lock.json','lock.sha256.json'):
        dest=out/'fitted'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(run/file,dest)
    shutil.copytree(evaluation,out/'evaluation')
    if include_answers:
        for name,checksum in manifest['answers_sha256'].items():
            if sha256(Path(answers)/'answers'/name)!=checksum:raise ValueError('Answer changed before release')
        shutil.copytree(Path(answers)/'answers',out/'collection/answers')
    if include_features:
        shutil.copytree(features,out/'features')
    checksums={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'release.json',dict(format='slice-review-bundle-v1',files=checksums,
        includes_answers=include_answers,feature_arrays_included=include_features,
        scope='Question IDs/splits, collection settings/model revision, configuration, selected checkpoints, '
              'calibrators, raw/calibrated predictions and hashes. Feature arrays/base-model weights are not bundled. '
              'Check source dataset/model licenses before public distribution. This exports new standalone runs, '
              'not missing artifacts from the historical reported study.'))


def verify_bundle(bundle,out):
    """Recompute the raw comparison table from released predictions, without a model."""
    bundle=Path(bundle).resolve();release=read_json(bundle/'release.json')
    if release['format']!='slice-review-bundle-v1':raise ValueError('Unknown review bundle')
    for name,digest in release['files'].items():
        path=(bundle/name).resolve()
        if not path.is_relative_to(bundle):raise ValueError('Release file outside bundle')
        if sha256(path)!=digest:raise ValueError(f'Released artifact changed: {name}')
    lock = read_json(bundle/'fitted/lock.json')
    lock_hash = sha256(bundle/'fitted/lock.json')
    if lock_hash != read_json(bundle/'fitted/lock.sha256.json')['sha256']:
        raise ValueError('Released pre-test lock changed')
    if lock['feature_manifest_sha256'] != sha256(bundle/'feature-manifest.json'):
        raise ValueError('Released feature manifest differs from pre-test lock')
    for name, checksum in lock['artifacts'].items():
        path = (bundle/'fitted'/name).resolve()
        if not path.is_relative_to((bundle/'fitted').resolve()) or sha256(path) != checksum:
            raise ValueError(f'Released locked artifact changed: {name}')
    feature_manifest = read_json(bundle/'feature-manifest.json')
    if feature_manifest['questions_sha256'] != sha256(bundle/'collection/questions.json'):
        raise ValueError('Released question IDs/splits differ from the feature manifest')
    if feature_manifest['collection_manifest_sha256'] != sha256(bundle/'collection/manifest.json'):
        raise ValueError('Released generation settings differ from the feature manifest')
    reported=read_json(bundle/'evaluation/summary.json');selection=read_json(bundle/'fitted/selection.json')['selected']
    if reported['lock_sha256'] != lock_hash:
        raise ValueError('Released evaluation differs from the locked configuration')
    groups={}
    with np.load(bundle/'evaluation/predictions.npz',allow_pickle=False) as saved:
        y=saved['y']
        for name,item in selection.items():groups.setdefault(item['group'],[]).append(saved[name])
        groups['constant']=saved['initial-constant']
        results={name:metric_average(y,p) for name,p in groups.items()}
        question, position = saved['question'], saved['position']
        questions = read_json(bundle/'collection/questions.json')
        allowed = {i for i, row in enumerate(questions) if row['split'] == 'test'}
        if not set(question).issubset(allowed):
            raise ValueError('Released predictions contain non-test question IDs')
        carry = {name: question_carry_forward(p, question, position) for name, p in groups.items()}
        for name, initial in carry.items():
            if not np.allclose(initial, saved[f'initial-{name}'], rtol=1e-10, atol=1e-12):
                raise ValueError(f'Released carry-forward scores differ: {name}')
        advancing = position > 0
        difficulty = {name: dict(
            initial_carried_forward=metric_average(y, initial),
            updated_minus_initial=paired_interval(y[advancing], np.atleast_2d(groups[name])[:, advancing],
                initial[:, advancing], question[advancing]) if advancing.any() else None)
            for name, initial in carry.items()}
    within = {}
    checkpoint_path = bundle/'evaluation/checkpoint-predictions.npz'
    if checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            anchor_groups = {}
            for name, item in selection.items():
                anchor_groups.setdefault(item['group'], []).append(saved[name])
            within = {name: within_question_ranking(saved['y'], p, saved['question'],
                saved['response'], saved['position']) for name, p in anchor_groups.items()}
    for name,row in results.items():
        for key,value in row['mean'].items():
            old=reported['results'][name]['mean'][key]
            if (value is None)!=(old is None) or (value is not None and not np.isclose(value,old,atol=1e-12,rtol=1e-10)):
                raise ValueError(f'Released predictions differ from report: {name}/{key}')
    out=require_empty(out);write_json(out/'verified-metrics.json',dict(results=results,
        question_difficulty=difficulty,within_question=within,lock_sha256=lock_hash,
        scope='Recomputed from released raw predictions and labels, including question-only controls. '
              'Checks integrity, not verifier correctness or test freshness.'))
    print(f'Verified {len(results)} methods from released per-prefix predictions.')
    return results
