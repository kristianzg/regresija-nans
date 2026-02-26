# =============================================================================
# Metrike evaluacije
# =============================================================================

from __future__ import annotations
import numpy as np

def mse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yt - yp) ** 2))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

def r2(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot

def adj_r2(y_true, y_pred, p: int) -> float:
    n = int(np.asarray(y_true).shape[0])
    r2v = r2(y_true, y_pred)
    if n <= p + 1:
        return float("nan")
    return 1.0 - (1.0 - r2v) * (n - 1) / (n - p - 1)
