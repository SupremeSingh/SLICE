"""Checked, portable files for datasets and fitted probes."""
import hashlib
import json
from pathlib import Path
import torch


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def save_checkpoint(path, record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(record,temp);temp.replace(path)
    write_json(path.with_suffix('.sha256.json'),dict(sha256=sha256(path)))


def load_checkpoint(path):
    path=Path(path)
    if sha256(path)!=read_json(path.with_suffix('.sha256.json'))['sha256']:
        raise ValueError(f'Changed checkpoint: {path}')
    return torch.load(path,weights_only=True,map_location='cpu')


def require_empty(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if any(path.iterdir()):raise ValueError(f'Use an empty output directory: {path}')
    return path
