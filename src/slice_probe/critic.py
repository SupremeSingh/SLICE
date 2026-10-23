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
from .model import restore, require_correctness_head
from .registry import canonical_hash, feature_identity, inspect_artifact, version_name


class Critic:
    """One resident encoder/compressor. Calls serialize access to model hooks.

    encode_batch executes unrelated unpadded states sequentially; encode_trajectory
    shares one causal pass across prefixes of the same sequence. Features have no
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
        if calibration is not None:
            require_correctness_head(self.record)
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

    def to(self, device):
        """Move all resident feature tensors together; `to('cpu')` offloads CUDA.

        Calls serialize with inference. The feature identity and raw head are
        unchanged. CUDA's allocator may retain freed memory for reuse.
        """
        selected = resolve_device(device)
        with self._lock:
            try:
                self.encoder.to(selected)
                self.compressor.to(selected)
                self._mean = self._mean.to(selected)
                self._scale = self._scale.to(selected)
            except Exception:
                # A failed GPU reload may have moved only part of the model.
                # Restore a coherent CPU instance before propagating the error.
                self.encoder.cpu()
                self.compressor.cpu()
                self._mean = self._mean.cpu()
                self._scale = self._scale.cpu()
                self.device = 'cpu'
                raise
            self.device = selected
        return self

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

    @torch.inference_mode()
    def encode_trajectory(self, token_ids, prefix_lengths, *, tokenizer_id,
                          batch_size=1024):
        """Encode many prefixes in one causal encoder pass, preserving order.

        token_ids are exact, unpadded prompt-plus-response IDs. Each prefix length
        is a total visible-token count in [1, len(token_ids)], not an answer-only
        offset. For prompt length P, pre-action states are P, P+1, ... . Duplicate
        lengths are allowed. Returns (N, width) float32 features without an
        intercept; terminal-state zeroing remains the caller's responsibility.
        batch_size bounds compressor batches, not encoder sequence length.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        with self._lock:
            ids = self._tokens(dict(token_ids=token_ids, tokenizer_id=tokenizer_id))
            if (not isinstance(prefix_lengths, (list, tuple))
                    or any(type(n) is not int or not 1 <= n <= len(ids) for n in prefix_lengths)):
                raise ValueError('prefix_lengths must contain total token counts in [1, len(token_ids)]')
            if not prefix_lengths:
                return np.empty((0, self.record['width']), dtype=np.float32)
            lengths, inverse = np.unique(prefix_lengths, return_inverse=True)
            blocks = capture(self.encoder,
                torch.tensor([ids[:int(lengths[-1])]], device=self.device),
                (lengths - 1).tolist(), self.record['layers'])
            x = concatenate(blocks, self.record['layers'], self.record['rms_eps'])
            chunks = []
            for start in range(0, len(x), batch_size):
                values = x[start:start + batch_size].to(self.device)
                chunks.append(self.compressor.features(
                    (values - self._mean) / self._scale).cpu().numpy())
            features = np.concatenate(chunks)
            if not np.isfinite(features).all():
                raise ValueError('Nonfinite compressed features')
            return features[inverse]

    def score_batch(self, states, *, return_calibrated_probability=False):
        if return_calibrated_probability:
            require_correctness_head(self.record)
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
                                  discount_gamma=self.record.get('head', {}).get('discount_gamma', 1.),
                                  value_target=self.record.get('head', {}).get('target', 'eventual verified final-answer correctness'),
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
