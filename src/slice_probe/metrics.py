"""Binary-outcome metrics; raw Ridge Brier, clipped probability log loss."""
import numpy as np


def auc(y, prediction):
    positive = y == 1
    npos, nneg = positive.sum(), (~positive).sum()
    if not npos or not nneg:
        return None
    order = np.argsort(prediction, kind="stable")
    _, start, counts = np.unique(prediction[order], return_index=True, return_counts=True)
    ranks = np.empty(len(y))
    ranks[order] = np.repeat(start + (counts + 1) / 2, counts)
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if (not len(y) or y.ndim != 1 or y.shape != prediction.shape
            or not np.isin(y, [0, 1]).all() or not np.isfinite(prediction).all()):
        raise ValueError("Empty, mismatched or nonfinite predictions")
    p = np.clip(prediction, 1e-7, 1 - 1e-7)
    calibration = []
    bins = np.minimum((np.clip(prediction, 0, 1) * 10).astype(int), 9)
    for i in range(10):
        mask = bins == i
        if mask.any():
            calibration.append(dict(count=int(mask.sum()), prediction=float(prediction[mask].mean()),
                                    success_rate=float(y[mask].mean())))
    return dict(n=len(y), accuracy=float(((prediction >= .5) == y).mean()),
                brier=float(np.mean((prediction - y) ** 2)),
                cross_entropy=float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                auroc=auc(y, prediction), outside_probability_range=float(((prediction < 0) | (prediction > 1)).mean()),
                calibration=calibration,
                ece=sum(b["count"] * abs(b["prediction"] - b["success_rate"]) for b in calibration) / len(y))
