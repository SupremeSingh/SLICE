"""Causal states from selected depths of a Qwen-style transformer."""
from pathlib import Path
import math
import torch

INSTRUCTION=r'Solve the math problem step by step. Put your final numeric answer inside a single \boxed{...}.'


def prompt(question):
    return INSTRUCTION+'\n\nQuestion: '+question+'\n\nSolution:'


def geometric_layers(count, base=2.0, num_slices=None):
    """One-based blocks, final first; rounded distances are forced to be unique.

    Distance i is max(previous_distance + 1, floor(base**i)), starting at 1.
    Base 1 selects consecutive final blocks. An explicit count must fit the model.
    """
    if not isinstance(count, int) or count < 1:
        raise ValueError('A positive integer transformer block count is required')
    if not math.isfinite(base) or base < 1:
        raise ValueError('Layer base must be finite and at least 1')
    if num_slices is not None and (not isinstance(num_slices, int) or not 1 <= num_slices <= count):
        raise ValueError('Number of slices must be between 1 and the block count')
    result = []
    distance = 0
    for exponent in range(count):
        try:
            geometric = base ** exponent
        except OverflowError:
            break
        if not math.isfinite(geometric) or geometric > count:
            break
        distance = max(distance + 1, math.floor(geometric))
        if distance > count:
            break
        result.append(count - distance + 1)
        if num_slices is not None and len(result) == num_slices:
            return result
    if num_slices is not None:
        raise ValueError(f'Base {base:g} supplies only {len(result)} slices in {count} blocks; '
                         'reduce --num-slices or --layer-base')
    return result


def resolve_device(device):
    """Prefer CUDA, then Apple Silicon, with a CPU fallback for auto selection."""
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda:0'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    selected = torch.device(device)
    if selected.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu or --device auto')
    if selected.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('Apple MPS is unavailable; use --device cpu or --device auto')
    if selected.type not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Supported devices: cpu, cuda[:index], mps, auto')
    return str(selected)


def sample_positions(length,question,response,seed=1729):
    import numpy as np
    if length<1:raise ValueError('Empty answer')
    rng=np.random.default_rng(np.random.SeedSequence([seed,question,response]))
    return [0]+(rng.integers(1,length,7).tolist() if length>1 else [0]*7)


@torch.inference_mode()
def capture(backbone,ids,indices,layers):
    if ids.ndim!=2 or ids.shape[0]!=1 or not indices or min(indices)<0 or max(indices)>=ids.shape[1]:
        raise ValueError('Invalid prefix indexing')
    if not layers or len(set(layers))!=len(layers) or min(layers)<1 or max(layers)>len(backbone.layers):
        raise ValueError('Invalid one-based layer selection')
    saved={};hooks=[]
    def hook(layer):
        def record(module,inputs,output):
            value=output[0] if isinstance(output,tuple) else output
            saved[layer]=value[0,indices].detach().float().cpu().clone()
        return record
    try:
        for layer in layers:hooks.append(backbone.layers[layer-1].register_forward_hook(hook(layer)))
        backbone(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
    finally:
        for handle in hooks:handle.remove()
    if set(saved)!=set(layers) or any(not torch.isfinite(v).all() for v in saved.values()):
        raise ValueError('Missing/nonfinite features')
    return saved


def concatenate(blocks,layers,epsilon=1e-6):
    return torch.cat([blocks[l]*torch.rsqrt(blocks[l].square().mean(-1,keepdim=True)+epsilon) for l in layers],-1)


def model_location(model,revision=None,expected_hashes=None):
    from .io import sha256
    if Path(model).is_dir():
        directory=Path(model).resolve();resolved=revision
    else:
        from huggingface_hub import HfApi,snapshot_download
        resolved=revision or HfApi().model_info(model).sha
        if len(resolved)!=40:raise ValueError('Model revision must be an immutable commit SHA')
        directory=Path(snapshot_download(model,revision=resolved,allow_patterns=[
            '*.json','*.safetensors','*.txt','*.model','*.tiktoken']))
    for name,checksum in (expected_hashes or {}).items():
        if sha256(directory/name)!=checksum:raise ValueError(f'Model asset changed: {name}')
    hashes=expected_hashes or {p.name:sha256(p) for p in directory.iterdir() if p.is_file() and (p.suffix in ('.json','.safetensors') or p.name in ('merges.txt','vocab.json'))}
    return directory,dict(id=model,revision=resolved,hashes=hashes)


def load_model(model,revision=None,device='cpu',dtype='float32',causal=False,hashes=None):
    from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer
    directory,identity=model_location(model,revision,hashes)
    # Feature extraction deliberately recomputes in float32 without TF32.
    if dtype=='float32':
        import os
        if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1':
            raise ValueError('Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE for feature extraction')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    cls=AutoModelForCausalLM if causal else AutoModel
    loaded=cls.from_pretrained(directory,torch_dtype=getattr(torch,dtype),attn_implementation='eager',local_files_only=True)
    loaded=loaded.to(device).eval().requires_grad_(False)
    backbone=loaded.model if causal else loaded
    if not hasattr(backbone,'layers'):raise ValueError('Only Qwen-style decoder backbones are supported')
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    return loaded,tokenizer,identity
