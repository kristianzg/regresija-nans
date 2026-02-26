# =============================================
# Generatori kontrolisanog krsenja pretpostavki
# =============================================

import numpy as np
import pandas as pd


def add_multicollinearity(
    X: pd.DataFrame, eps: float, copies: int, rng: np.random.Generator
) -> pd.DataFrame:
    Xn = X.copy()
    var = Xn.var(numeric_only=True)
    base_col = var.sort_values(ascending=False).index[0]
    base = Xn[base_col].to_numpy(dtype=float)
    sigma = np.std(base, ddof=1) + 1e-12
    for i in range(copies):
        noise = rng.normal(0.0, eps * sigma, size=base.shape[0])
        Xn[f"{base_col}_dup{i+1}"] = base + noise
    return Xn


def heteroskedastic_noise(
    y: np.ndarray, driver: np.ndarray, strength: float, rng: np.random.Generator
) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    d = np.asarray(driver, dtype=float).ravel()
    d = (d - d.min()) / (d.max() - d.min() + 1e-12)
    base_sigma = np.std(y, ddof=1) * 0.05
    sigma = base_sigma * (1.0 + strength * d)
    noise = rng.normal(0.0, sigma, size=y.shape[0])
    return y + noise


def heavy_tailed_noise(
    y: np.ndarray, df: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    z = rng.normal(0.0, 1.0, size=y.shape[0])
    u = rng.chisquare(df, size=y.shape[0])
    t = z / np.sqrt(u / df)
    base_sigma = np.std(y, ddof=1) * 0.05
    return y + scale * base_sigma * t
