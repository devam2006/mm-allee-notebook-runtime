"""Explicit absolute/global-relative trajectory errors."""

import numpy as np


def errors(prediction, reference):
    pred = np.asarray(prediction, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if pred.shape != ref.shape or pred.size == 0:
        raise ValueError("Metric inputs must have identical nonempty shapes")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(ref)):
        raise ValueError("Metric inputs must be finite")
    delta = pred - ref
    norm = np.linalg.norm(ref.ravel())
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))),
        "relative_l2": None if norm == 0 else float(np.linalg.norm(delta.ravel()) / norm),
    }
