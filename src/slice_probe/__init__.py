"""SLICE: Sparse Latent Inference for Correctness Estimation."""
__version__ = "0.1.0"


def __getattr__(name):
    if name == "Critic":
        from .critic import Critic
        return Critic
    raise AttributeError(name)
