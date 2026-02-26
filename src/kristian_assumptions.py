# =======================================
# Provera pretpostavki linearne regresije
# =======================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def durbin_watson(residuals: np.ndarray) -> float:
    """Nezavisnost gresaka, DW oko 2 sugerise nezavisnost."""
    e = np.asarray(residuals, dtype=float).ravel()
    if e.size < 3:
        return float("nan")
    num = np.sum(np.diff(e) ** 2)
    den = np.sum(e**2)
    if den <= 0:
        return float("nan")
    return float(num / den)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF = 1 / (1 - R^2) za svaku kolonu."""
    Xv = X.to_numpy(dtype=float)
    cols = list(X.columns)
    vifs = []

    for j, col in enumerate(cols):
        y = Xv[:, j]
        X_other = np.delete(Xv, j, axis=1)
        model = LinearRegression()
        model.fit(X_other, y)
        r2 = model.score(X_other, y)
        vif = np.inf if (1.0 - r2) <= 1e-12 else 1.0 / (1.0 - r2)
        vifs.append(vif)

    out = pd.DataFrame({"feature": cols, "VIF": vifs})
    return out.sort_values("VIF", ascending=False).reset_index(drop=True)


def qq_points(residuals: np.ndarray):
    """Empirijski kvantili standardizovanih gresaka vs N(0,1) standardni uzorak."""
    e = np.asarray(residuals, dtype=float).ravel()
    e = e[np.isfinite(e)]
    n = e.size
    if n < 10:
        return np.array([]), np.array([])

    e_std = (e - np.mean(e)) / (np.std(e, ddof=1) + 1e-12)
    emp = np.sort(e_std)

    rng = np.random.default_rng(0)
    theo = np.sort(rng.normal(0.0, 1.0, size=n))
    return theo, emp


def homoskedasticity_indicators(y_pred: np.ndarray, residuals: np.ndarray) -> dict:
    """
    Jednaka varijansa (homoskedastičnost):
    - corr_abs_resid_pred: korelacija |e| i y_hat (bliže 0 je bolje)
    - bp_like_lm: Breusch-Pagan-like LM = n * R^2 iz pomoćne regresije e^2 ~ y_hat
    """
    yp = np.asarray(y_pred, dtype=float).ravel()
    e = np.asarray(residuals, dtype=float).ravel()

    abs_e = np.abs(e)
    if abs_e.size < 3 or np.std(abs_e, ddof=1) < 1e-12 or np.std(yp, ddof=1) < 1e-12:
        corr_abs = float("nan")
    else:
        corr_abs = float(np.corrcoef(abs_e, yp)[0, 1])

    y_aux = (e**2).reshape(-1, 1)
    X_aux = yp.reshape(-1, 1)
    aux = LinearRegression().fit(X_aux, y_aux)
    r2_aux = float(aux.score(X_aux, y_aux))
    bp_like_lm = float(len(e) * max(0.0, r2_aux))

    return {
        "corr_abs_resid_pred": corr_abs,
        "bp_like_lm": bp_like_lm,
    }


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Sažetak reziduala za normalnost/nezavisnost/homoskedastičnost."""
    e = np.asarray(y_true - y_pred, dtype=float).ravel()
    s = pd.Series(e)

    hs = homoskedasticity_indicators(y_pred=y_pred, residuals=e)

    return {
        "residual_mean": float(s.mean()),
        "residual_std": float(s.std(ddof=1)),
        "residual_skew": float(s.skew()),
        "residual_kurtosis": float(s.kurtosis()),
        "durbin_watson": durbin_watson(e),
        **hs,
    }
