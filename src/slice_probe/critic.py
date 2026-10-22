"""Resident frozen features and explicit, versioned correctness scores."""
from __future__ import annotations

from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np
import torch

from .calibration import apply_calibrator
from .features import capture, concatenate, load_model, prompt, resolve_device
from .io import load_checkpoint, read_json, sha256
from .model import restore
from .registry import canonical_hash, feature_identity, inspect_artifact, version_name


class Critic:
    """One resident encoder/compressor. Calls serialize access to model hooks.

    Batches currently execute unpadded states sequentially to preserve the original
    feature conventions. encode_batch returns compressed features, without an
    intercept column. score_batch reports validation errors per state.
    """

    @classmethod
    def load(cls, path=None, *, registry=None, version=None, device='auto',
             model_override=None, calibration=None):
        if registry is not None:
            if path is not None or version is None:
                raise ValueError('Use a path OR registry plus version')
            path = Path(registry) / version_name(version)
        if path is None:
            raise ValueError('A checkpoint or artifact path is required')
        path = Path(path)
        manifest = inspect_artifact(path) if path.is_dir() else None
        checkpoint = path / 'checkpoint.pt' if manifest else path
        if manifest and calibration is None and (path / 'calibration.json').exists():
            calibration = path / 'calibration.json'
        return cls(checkpoint, device, model_override, calibration, manifest)

    def __init__(self, checkpoint, device, model_override, calibration, manifest):
        self.record = load_checkpoint(checkpoint)
        self.device = resolve_device(device)
        self.compressor = restore(self.record, self.device)
        identity = self.record['model']
        self.encoder, self.tokenizer, _ = load_model(
            model_override or identity['id'], identity['revision'], self.device,
            hashes=identity['hashes'])
        self.encoder.eval().requires_grad_(False)
        self.feature_id = feature_identity(self.record)
        self.tokenizer_id = canonical_hash(identity)
        self.version = manifest['version'] if manifest else 'checkpoint-' + sha256(checkpoint)
        self.checkpoint_sha256 = sha256(checkpoint)
        self._lock = RLock()
        self._mean = self.record['mean'].to(self.device)
        self._scale = self.record['scale'].to(self.device)
        self._weights = self.record['weights'].cpu().double()
        self.calibrator = None
        if calibration is not None:
            data = read_json(calibration)
            matches = [name for name, digest in data['checkpoint_sha256'].items()
                       if digest == self.checkpoint_sha256 and name in data['calibrators']]
            if len(matches) != 1:
                raise ValueError('Calibrator does not identify this checkpoint')
            self.calibrator = data['calibrators'][matches[0]]

    def _tokens(self, state):
        if not isinstance(state, dict):
            raise ValueError('Each state must be an object')
        if 'token_ids' in state:
            if 'question' in state or 'prefix' in state:
                raise ValueError('Supply full token_ids OR question/prefix')
            if state.get('tokenizer_id') != self.tokenizer_id:
                raise ValueError('Exact tokens require the matching tokenizer_id')
            ids = state['token_ids']
            if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError('token_ids must be a list of nonnegative integers')
            vocab_size = getattr(self.encoder.config, 'vocab_size', None)
            if vocab_size is not None and any(i >= vocab_size for i in ids):
                raise ValueError('Token ID exceeds encoder vocabulary')
        else:
            question, prefix = state.get('question'), state.get('prefix', '')
            if not isinstance(question, str) or not question.strip() or not isinstance(prefix, str):
                raise ValueError('A nonempty question and string prefix are required')
            ids = self.tokenizer.encode(prompt(question), add_special_tokens=False)
            ids += self.tokenizer.encode(prefix, add_special_tokens=False)
        if not ids or len(ids) > self.encoder.config.max_position_embeddings:
            raise ValueError('Input exceeds model context or is empty; inputs are never truncated')
        return ids

    @torch.inference_mode()
    def _encode_ids(self, ids):
        blocks = capture(self.encoder, torch.tensor([ids], device=self.device),
                         [len(ids) - 1], self.record['layers'])
        x = concatenate(blocks, self.record['layers'], self.record['rms_eps']).to(self.device)
        z = self.compressor.features((x - self._mean) / self._scale).cpu().numpy()[0]
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite compressed features')
        return z

    def encode_batch(self, states):
        """Return (N, width) float32 features; invalid inputs raise ValueError."""
        with self._lock:
            tokens = [self._tokens(state) for state in states]
            if not tokens:
                return np.empty((0, self.record['width']), dtype=np.float32)
            return np.stack([self._encode_ids(ids) for ids in tokens])

    def score_batch(self, states, *, return_calibrated_probability=False):
        if return_calibrated_probability and self.calibrator is None:
            raise ValueError('No matching validation-fitted calibrator is available')
        results = []
        with self._lock:
            for state in states:
                result = dict(state_id=state.get('state_id') if isinstance(state, dict) else None, critic_version=self.version,
                              feature_id=self.feature_id, checkpoint_sha256=self.checkpoint_sha256)
                try:
                    ids = self._tokens(state)
                    start = perf_counter()
                    z = torch.from_numpy(self._encode_ids(ids)).double()
                    encoded = perf_counter()
                    score = float(z @ self._weights[:-1] + self._weights[-1])
                    if not np.isfinite(score):
                        raise ValueError('Nonfinite score')
                    result.update(raw_value=score,
                                  token_count=len(ids),
                                  encoding_seconds=encoded - start,
                                  head_seconds=perf_counter() - encoded,
                                  head=self.record.get('head', {'method': 'ridge'}),
                                  tokenizer_id=self.tokenizer_id)
                    if self.record.get('head', {}).get('discount_gamma', 1.) == 1.:
                        result['predicted_correct'] = score >= .5
                    if return_calibrated_probability:
                        result['calibrated_probability'] = float(apply_calibrator([score], self.calibrator)[0])
                except ValueError as exc:
                    result['error'] = str(exc)
                results.append(result)
        return results
