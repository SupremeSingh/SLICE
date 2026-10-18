"""Reference collection and causal feature extraction using Hugging Face models."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .data import load_questions,grade
from .features import load_model,prompt,geometric_layers,sample_positions,capture,concatenate
from .io import read_json,write_json,sha256,require_empty


def collect(questions_path,out,model,revision,device,responses=8,max_new_tokens=3584,seed=271828):
    questions=load_questions(questions_path)
    if responses<1 or max_new_tokens<1:raise ValueError('Positive generation budgets required')
    loaded,tokenizer,identity=load_model(model,revision,device,'bfloat16' if device.startswith('cuda') and torch.cuda.is_bf16_supported() else 'float32',causal=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=dict(format='slice-answers-v1',model=identity,questions_sha256=sha256(questions_path),
        responses=responses,max_new_tokens=max_new_tokens,seed=seed,temperature=1.,
        top_p=1.,top_k=0,do_sample=True,prompt_version='completion-boxed-v1',
        generation_dtype=str(next(loaded.parameters()).dtype))
    from .comparison import environment
    cfg['environment']=environment()
    cfg['verifier_policy']='single-scalar-box-v3'
    selection=Path(questions_path).parent/'selection.json'
    if selection.exists():
        cfg['dataset_selection']=read_json(selection)
        if cfg['dataset_selection'].get('questions_sha256')!=sha256(questions_path):
            raise ValueError('Dataset selection does not identify this questions file; regenerate selection')
    meta=out/'manifest.json'
    if meta.exists() and read_json(meta)!=cfg:raise ValueError('Collection settings changed; use a new output')
    if not meta.exists() and any(out.iterdir()):raise ValueError('Unrecognized nonempty output')
    limit=loaded.config.max_position_embeddings
    encoded=[tokenizer.encode(prompt(q['question']),add_special_tokens=False) for q in questions]
    if any(not ids or len(ids)>512 or len(ids)+max_new_tokens>limit for ids in encoded):
        raise ValueError('A prompt exceeds 512 tokens or the model context; filter inputs or reduce --max-new-tokens')
    write_json(meta,cfg);write_json(out/'questions.json',questions)
    (out/'source-questions.jsonl').write_bytes(Path(questions_path).read_bytes())
    (out/'answers').mkdir(exist_ok=True)
    for i,(q,ids) in enumerate(zip(questions,encoded)):
        path=out/'answers'/f'{i:05d}.json'
        if path.exists():
            old=read_json(path)
            if old['id']!=q['id'] or len(old['responses'])!=responses:raise ValueError('Stale answer batch')
            continue
        answers=[];tokens=torch.tensor([ids],device=device)
        for j in range(responses):
            torch.manual_seed(seed+i*responses+j)
            with torch.inference_mode():
                output=loaded.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),do_sample=True,
                    temperature=1.,top_p=1.,top_k=0,max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,use_cache=True)
            tail=output[0,len(ids):].tolist();text=tokenizer.decode(tail,skip_special_tokens=True)
            score=grade(text,q['reference']) if tail else dict(score=None,status='empty_response',extracted=None)
            answers.append(dict(text=text,token_ids=tail,capped=len(tail)>=max_new_tokens,**score))
        write_json(path,dict(id=q['id'],prompt_ids=ids,responses=answers))
        print(f'Saved {i+1}/{len(questions)}: {q["id"]}',flush=True)


def retained(row):
    return row['status'] in ('correct','incorrect') and bool(row['token_ids'])


def iter_answers(directory,split=None):
    directory=Path(directory)
    for i,q in enumerate(read_json(directory/'questions.json')):
        if split is not None and q['split']!=split:continue
        path=directory/'answers'/f'{i:05d}.json';record=read_json(path)
        if record.get('id')!=q['id']:raise ValueError('Answer identity changed')
        yield i,q,record,path


def extract(answers,out,device,layer_base=2.0,num_slices=None,baselines=False,checkpoints=()):
    answers=Path(answers);cfg=read_json(answers/'manifest.json');identity=cfg['model']
    backbone,_,_=load_model(identity['id'],identity['revision'],device,hashes=identity['hashes'])
    layers=geometric_layers(len(backbone.layers),layer_base,num_slices);epsilon=backbone.config.rms_norm_eps
    dim=backbone.config.hidden_size*len(layers)
    checkpoints=sorted(set(checkpoints))
    if any(not isinstance(p,int) or p<0 for p in checkpoints):
        raise ValueError('Checkpoint token counts must be nonnegative integers')
    depth=len(backbone.layers);two_thirds=(2*depth+2)//3
    capture_layers=list(dict.fromkeys(layers+([two_thirds,depth] if baselines else [])))
    if baselines and not hasattr(backbone,'norm'):
        raise ValueError('Baseline extraction requires the decoder final norm module')
    representations={'concat':dict(dimension=dim,tag='x',convention='per-block RMS, no learned gain')}
    if baselines:
        representations.update(final=dict(dimension=backbone.config.hidden_size,tag='final',
            block=depth,convention='post-final learned model normalization'),
            two_thirds=dict(dimension=backbone.config.hidden_size,tag='two_thirds',
            block=two_thirds,convention='raw intermediate block output'))
    def transform(blocks):
        values={'concat':concatenate(blocks,layers,epsilon)}
        if baselines:
            values['two_thirds']=blocks[two_thirds]
            with torch.inference_mode():
                values['final']=backbone.norm(blocks[depth].to(device)).float().cpu()
        return values
    counts={};anchor_counts={};hashes={};split_examples={};statuses={}
    # Fail before allocating large arrays if labels/collection are incomplete.
    for split in ('train','val','test'):
        count=0;anchors=0;status=Counter();files={}
        for i,q,r,path in iter_answers(answers,split):
            files[path.name]=sha256(path)
            for row in r['responses']:
                status[row['status']]+=1;count+=8*retained(row)
                if retained(row):anchors+=sum(p<len(row['token_ids']) for p in checkpoints)
        if count==0:raise ValueError(f'No retained {split} answers')
        counts[split]=count;anchor_counts[split]=anchors;statuses[split]=dict(status);hashes.update(files)
    out=require_empty(out)
    for split in ('train','val','test'):
        count=counts[split]
        arrays={name:np.lib.format.open_memmap(out/f'{split}-{spec["tag"]}.npy',mode='w+',
            dtype='float32',shape=(count,spec['dimension'])) for name,spec in representations.items()}
        anchors={name:np.lib.format.open_memmap(out/f'{split}-anchor-{spec["tag"]}.npy',mode='w+',
            dtype='float32',shape=(anchor_counts[split],spec['dimension']))
            for name,spec in representations.items()} if anchor_counts[split] else {}
        values={k:[] for k in ('y','question','position','response')}
        anchor_values={k:[] for k in values};cursor=anchor_cursor=0
        for i,q,record,path in iter_answers(answers,split):
            if sha256(path) != hashes[path.name]: raise ValueError('Answers changed during extraction')
            checkpoint_cache={}
            for j,row in enumerate(record['responses']):
                if not retained(row):continue
                positions=sample_positions(len(row['token_ids']),i,j)
                fixed=[p for p in checkpoints if p<len(row['token_ids'])]
                ids=torch.tensor([record['prompt_ids']+row['token_ids']],device=device)
                absolute=[len(record['prompt_ids'])-1+p for p in positions]
                transformed=transform(capture(backbone,ids,absolute,capture_layers))
                fixed_features=[]
                for p in fixed:
                    prefix=tuple(record['prompt_ids']+row['token_ids'][:p])
                    if prefix not in checkpoint_cache:
                        prefix_ids=torch.tensor([prefix],device=device)
                        checkpoint_cache[prefix]=transform(capture(backbone,prefix_ids,[len(prefix)-1],capture_layers))
                    fixed_features.append(checkpoint_cache[prefix])
                for name,x in transformed.items():
                    arrays[name][cursor:cursor+8]=x[:8].numpy()
                    if fixed:anchors[name][anchor_cursor:anchor_cursor+len(fixed)]=torch.cat([value[name] for value in fixed_features]).numpy()
                cursor+=8;anchor_cursor+=len(fixed)
                for storage,pos in ((values,positions),(anchor_values,fixed)):
                    storage['y'].extend([row['score']]*len(pos));storage['question'].extend([i]*len(pos))
                    storage['position'].extend(pos);storage['response'].extend([j]*len(pos))
            print(f'Extracted {split}: {q["id"]}',flush=True)
        if cursor!=count or anchor_cursor!=anchor_counts[split]:raise ValueError('State count changed during extraction')
        for x in [*arrays.values(),*anchors.values()]:x.flush()
        del arrays,anchors
        for prefix,storage in (('',values),('anchor-',anchor_values)):
            if prefix and not anchor_counts[split]:continue
            for key,v in storage.items():
                np.save(out/f'{split}-{prefix}{key}.npy',np.asarray(v,dtype='float32' if key=='y' else 'int64'))
        split_examples[split]=len(values['y'])
    write_json(out/'questions.json',read_json(answers/'questions.json'))
    write_json(out/'manifest.json',dict(format='slice-features-v1',model=identity,layers=layers,rms_eps=epsilon,
        dimension=dim,representations=representations,checkpoint_positions=checkpoints,
        anchor_counts=anchor_counts,questions_sha256=sha256(out/'questions.json'),
        collection=cfg,layer_selection=dict(base=layer_base,num_slices=num_slices,
        rule='unique-floor-geometric-v1'),prefixes_per_answer=8,counts=split_examples,statuses=statuses,answers_sha256=hashes,
        collection_manifest_sha256=sha256(answers/'manifest.json'),
        files={p.name:sha256(p) for p in out.glob('*.npy')}))


def load_split(directory,split,verify=True,representation='concat',anchors=False):
    directory=Path(directory);m=read_json(directory/'manifest.json');data={}
    if m.get('format') != 'slice-features-v1' or split not in ('train','val','test'): raise ValueError('Unknown feature format/split')
    if m.get('questions_sha256') and sha256(directory/'questions.json')!=m['questions_sha256']:
        raise ValueError('Question IDs/splits snapshot changed')
    spec=m.get('representations',{'concat':dict(dimension=m['dimension'],tag='x')}).get(representation)
    if spec is None:raise ValueError(f'Missing representation {representation}; rerun extract --baselines')
    prefix='anchor-' if anchors else ''
    for key in ('x','y','question','position','response'):
        tag=spec['tag'] if key=='x' else key
        path=directory/f'{split}-{prefix}{tag}.npy'
        if verify and sha256(path)!=m['files'][path.name]:raise ValueError(f'Changed cache: {path}')
        data[key]=np.load(path,mmap_mode='r',allow_pickle=False)
    n=m['anchor_counts'][split] if anchors else m['counts'][split]
    if any(len(v)!=n for v in data.values()) or data['x'].shape!=(n,spec['dimension']):raise ValueError('Incomplete feature cache')
    if not np.isin(data['y'],[0,1]).all(): raise ValueError('Nonbinary cache labels')
    if (data['position']<0).any(): raise ValueError('Negative prefix positions')
    return data,m
