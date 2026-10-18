"""Validation-only monotone calibration and selective answer acceptance."""
import numpy as np
import torch
from .metrics import metrics


def fit_platt(scores, y):
    scores, y = np.asarray(scores), np.asarray(y)
    if scores.ndim!=1 or scores.shape!=y.shape or len(np.unique(y))!=2 or not np.isin(y,[0,1]).all() or not np.isfinite(scores).all():
        raise ValueError('Calibration requires finite validation scores and both outcomes')
    mean, scale = float(scores.mean()), max(float(scores.std()), 1e-8)
    x = torch.tensor((scores - mean) / scale, dtype=torch.float64)
    target = torch.tensor(y.copy(), dtype=torch.float64)
    a = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([a,b], max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        logits = torch.nn.functional.softplus(a) * x + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target) + 1e-4*(a*a+b*b)
        loss.backward()
        return loss
    optimizer.step(closure)
    result = dict(mean=mean,scale=scale,slope=float(torch.nn.functional.softplus(a).detach()),
                  intercept=float(b.detach()),method='validation-only monotone Platt',regularization=1e-4)
    if not all(np.isfinite(result[k]) for k in ('mean','scale','slope','intercept')):
        raise ValueError('Nonfinite calibration fit')
    return result


def apply_calibrator(scores, calibrator):
    logits = calibrator['slope']*(np.asarray(scores)-calibrator['mean'])/calibrator['scale']+calibrator['intercept']
    return 1/(1+np.exp(-np.clip(logits,-700,700)))


def risk_coverage(y, scores):
    """Accept high predicted correctness; ties enter together, never by label order."""
    y,scores=np.asarray(y),np.asarray(scores)
    if not len(y) or y.ndim!=1 or y.shape!=scores.shape or not np.isin(y,[0,1]).all() or not np.isfinite(scores).all():
        raise ValueError('Risk-coverage requires aligned binary labels and finite scores')
    order = np.argsort(-scores,kind='stable');p=scores[order];targets=y[order]
    ends = np.r_[np.flatnonzero(p[:-1]!=p[1:])+1,len(p)]
    errors = np.cumsum(1-targets)
    selected = np.unique(np.searchsorted(ends,np.linspace(1,len(p),min(100,len(p)))))
    return [dict(threshold=float(p[ends[j]-1]),coverage=float(ends[j]/len(p)),
                 risk=float(errors[ends[j]-1]/ends[j])) for j in selected]


def select_acceptance(y, scores, target_risk=.1):
    """An empirical validation operating point, not a guaranteed risk bound."""
    curve = risk_coverage(y,scores)
    candidates = [r for r in curve if r['risk']<=target_risk]
    # None means accept nothing; do not invent a threshold from test labels.
    return dict(target_validation_risk=target_risk,
                threshold=max(candidates,key=lambda r:r['coverage'])['threshold'] if candidates else None)


def calibrated_report(y, scores, calibrator, acceptance):
    p=apply_calibrator(scores,calibrator)
    threshold=acceptance['threshold']
    keep=np.zeros(len(p),dtype=bool) if threshold is None else p>=threshold
    return p,dict(metrics=metrics(y,p),risk_coverage=risk_coverage(y,p),
        validation_selected_acceptance=dict(acceptance,coverage=float(keep.mean()),
            risk=float((1-np.asarray(y)[keep]).mean()) if keep.any() else None),
        scope='Calibration and acceptance threshold fitted on validation only. '
              'Test risk is descriptive; validation risk is not a finite-sample guarantee.')
