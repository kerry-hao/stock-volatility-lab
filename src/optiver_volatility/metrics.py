"""Competition metric and training weights; true targets must be positive."""
import numpy as np


def positive_targets(values):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Targets must be a nonempty finite positive vector")
    return y


def rmspe(y_true, y_pred):
    y = positive_targets(y_true)
    p = np.asarray(y_pred, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("Predictions must be finite and match the target shape")
    return float(np.sqrt(np.mean(np.square((y - p) / y))))


def rmspe_weights(y_true):
    y = positive_targets(y_true)
    # Scale before inversion to avoid overflow; the common factor cancels.
    w = np.square(y.min() / y)
    return w / w.mean()


def lgb_rmspe(y_true, y_pred):
    return "rmspe", rmspe(y_true, y_pred), False


def optimal_constant(y_true):
    y = positive_targets(y_true)
    w = rmspe_weights(y)
    return float(np.sum(w * y) / w.sum())


def fit_scale(y_true, past_rv):
    y = positive_targets(y_true)
    p = np.asarray(past_rv, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Past volatility must be finite, nonnegative and aligned")
    z = p / y
    den = np.dot(z, z)
    return float(z.sum() / den) if den > 0 else 1.0
