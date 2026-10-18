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
        responses=responses,max_new_tokens=max_new_tokens,seed=seed,temperature=1.,prompt_version='completion-boxed-v1')
    meta=out/'manifest.json'
    if meta.exists() and read_json(meta)!=cfg:raise ValueError('Collection settings changed; use a new output')
    if not meta.exists() and any(out.iterdir()):raise ValueError('Unrecognized nonempty output')
    limit=loaded.config.max_position_embeddings
    encoded=[tokenizer.encode(prompt(q['question']),add_special_tokens=False) for q in questions]
    if any(not ids or len(ids)>512 or len(ids)+max_new_tokens>limit for ids in encoded):
        raise ValueError('A prompt exceeds 512 tokens or the model context; filter inputs or reduce --max-new-tokens')
    write_json(meta,cfg);write_json(out/'questions.json',questions)
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


def extract(answers,out,device,layer_base=2.0,num_slices=None):
    answers=Path(answers);cfg=read_json(answers/'manifest.json');identity=cfg['model']
    backbone,_,_=load_model(identity['id'],identity['revision'],device,hashes=identity['hashes'])
    layers=geometric_layers(len(backbone.layers),layer_base,num_slices);epsilon=backbone.config.rms_norm_eps
    dim=backbone.config.hidden_size*len(layers)
    counts={};hashes={};split_examples={};statuses={}
    # Fail before allocating large arrays if labels/collection are incomplete.
    for split in ('train','val','test'):
        count=0;status=Counter();files={}
        for i,q,r,path in iter_answers(answers,split):
            files[path.name]=sha256(path)
            for row in r['responses']:
                status[row['status']]+=1;count+=8*retained(row)
        if count==0:raise ValueError(f'No retained {split} answers')
        counts[split]=count;statuses[split]=dict(status);hashes.update(files)
    out=require_empty(out)
    for split in ('train','val','test'):
        count=counts[split];xs=np.lib.format.open_memmap(out/f'{split}-x.npy',mode='w+',dtype='float32',shape=(count,dim))
        values={k:[] for k in ('y','question','position','response')};cursor=0
        for i,q,record,path in iter_answers(answers,split):
            if sha256(path) != hashes[path.name]: raise ValueError('Answers changed during extraction')
            for j,row in enumerate(record['responses']):
                if not retained(row):continue
                positions=sample_positions(len(row['token_ids']),i,j)
                ids=torch.tensor([record['prompt_ids']+row['token_ids']],device=device)
                absolute=[len(record['prompt_ids'])-1+p for p in positions]
                x=concatenate(capture(backbone,ids,absolute,layers),layers,epsilon)
                xs[cursor:cursor+8]=x.numpy();cursor+=8
                values['y'].extend([row['score']]*8);values['question'].extend([i]*8)
                values['position'].extend(positions);values['response'].extend([j]*8)
            print(f'Extracted {split}: {q["id"]}',flush=True)
        if cursor != count: raise ValueError('Retained state count changed during extraction')
        xs.flush();del xs
        for key,v in values.items():np.save(out/f'{split}-{key}.npy',np.asarray(v,dtype='float32' if key=='y' else 'int64'))
        split_examples[split]=len(values['y'])
    write_json(out/'manifest.json',dict(format='slice-features-v1',model=identity,layers=layers,rms_eps=epsilon,
        dimension=dim,layer_selection=dict(base=layer_base,num_slices=num_slices,
        rule='unique-floor-geometric-v1'),prefixes_per_answer=8,counts=split_examples,statuses=statuses,answers_sha256=hashes,
        collection_manifest_sha256=sha256(answers/'manifest.json'),
        files={p.name:sha256(p) for p in out.glob('*.npy')}))


def load_split(directory,split,verify=True):
    directory=Path(directory);m=read_json(directory/'manifest.json');data={}
    if m.get('format') != 'slice-features-v1' or split not in ('train','val','test'): raise ValueError('Unknown feature format/split')
    for key in ('x','y','question','position','response'):
        path=directory/f'{split}-{key}.npy'
        if verify and sha256(path)!=m['files'][path.name]:raise ValueError(f'Changed cache: {path}')
        data[key]=np.load(path,mmap_mode='r',allow_pickle=False)
    n=m['counts'][split]
    if any(len(v)!=n for v in data.values()) or data['x'].shape!=(n,m['dimension']):raise ValueError('Incomplete feature cache')
    if not np.isin(data['y'],[0,1]).all(): raise ValueError('Nonbinary cache labels')
    if (data['position']<0).any(): raise ValueError('Negative prefix positions')
    return data,m
