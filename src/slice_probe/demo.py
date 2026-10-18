"""A tiny synthetic smoke test, explicitly separate from research results."""
import numpy as np

from .io import require_empty, write_json, sha256
from .training import fit
from .evaluation import evaluate


def run(out):
    out = require_empty(out)
    features = out / 'features'
    features.mkdir()
    rng = np.random.default_rng(7)
    counts = {}
    for split, offset, questions in [('train',0,32),('val',32,12),('test',44,12)]:
        question = np.repeat(np.arange(offset, offset+questions), 8)
        latent = rng.normal(size=questions)
        y = np.repeat((latent>0).astype('float32'),8)
        x = rng.normal(scale=.25,size=(len(y),24)).astype('float32')
        x[:,0] += np.repeat(latent,8)
        x[:,1] -= np.repeat(latent,8)
        data = dict(x=x,y=y,question=question,position=np.tile(np.arange(8),questions),
                    response=np.zeros(len(y),dtype='int64'))
        for key, value in data.items():
            np.save(features / f'{split}-{key}.npy',value)
        counts[split] = len(y)
    write_json(features / 'manifest.json',dict(format='slice-features-v1',dimension=24,
        model=dict(id='synthetic',revision=None,hashes={}),layers=[2,1],rms_eps=1e-6,
        counts=counts,files={p.name:sha256(p) for p in features.glob('*.npy')},
        scope='Synthetic smoke test: not LLM activations or a research benchmark.'))
    fit(features,out/'probes',width=8,seeds=[7],learning_rates=[.01],epochs=4,batch_size=64)
    evaluate(features,out/'probes',out/'evaluation')
    print('Synthetic smoke test complete. These scores are not the reported research results.')
