"""A supervised feature compressor followed by a frozen linear readout."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(self,input_dim,width=768):
        super().__init__()
        if min(input_dim,width)<1:raise ValueError('Dimensions must be positive')
        self.projection=nn.Linear(input_dim,width)
        self.readout=nn.Linear(width,1) # Temporary classifier, discarded at deployment.

    def features(self,x):
        return F.relu(F.linear(x,F.normalize(self.projection.weight,dim=1),self.projection.bias))

    def forward(self,x):
        z=self.features(x)
        return self.readout(z).flatten(),z


def objective(model,x,y,penalty):
    logits,z=model(x)
    return F.binary_cross_entropy_with_logits(logits,y)+penalty*z.mean()


def restore(record,device='cpu'):
    if record['format']!='slice-probe-v1':raise ValueError('Unknown probe format')
    model=Bottleneck(record['input_dim'],record['width'])
    model.load_state_dict(record['state'],strict=True)
    model=model.to(device).eval().requires_grad_(False)
    if record['weights'].shape!=(record['width']+1,):raise ValueError('Readout size mismatch')
    for key in ('mean','scale','weights'):
        if not torch.isfinite(record[key]).all():raise ValueError(f'Nonfinite {key}')
    if record['mean'].shape!=(record['input_dim'],) or record['scale'].shape!=(record['input_dim'],):
        raise ValueError('Normalization size mismatch')
    if (record['scale']<=0).any():raise ValueError('Nonpositive scale')
    return model


@torch.inference_mode()
def predict_features(record,x,device='cpu',batch_size=1024):
    model=restore(record,device);outputs=[]
    mean,scale=record['mean'].to(device),record['scale'].to(device)
    readout_device = 'cpu' if str(device).startswith('mps') else device
    weights=record['weights'].to(device=readout_device,dtype=torch.float64)
    for start in range(0,len(x),batch_size):
        v=torch.as_tensor(np.array(x[start:start+batch_size],copy=True),dtype=torch.float32,device=device)
        z=model.features((v-mean)/scale).to(readout_device).double()
        outputs.append((z@weights[:-1]+weights[-1]).cpu())
    if not outputs:raise ValueError('No features')
    return torch.cat(outputs).numpy()
