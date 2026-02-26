# ==========================================================
# Vizuelizacije (EDA, pretpostavke, agregacija i boxplotovi)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .kristian_assumptions import qq_points


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, out_path):
    """Vizuelizacija distribucija promenljivih."""
    cols = list(df.columns)
    n = len(cols)
    rows = 2
    cols_n = int(np.ceil(n / rows))

    plt.figure(figsize=(4 * cols_n, 6))
    for i, c in enumerate(cols, start=1):
        plt.subplot(rows, cols_n, i)
        x = df[c].to_numpy(dtype=float)
        plt.hist(x, bins=20)
        plt.xlabel(c)
        plt.ylabel("frekvencija")
    plt.suptitle("Distribucije promenljivih")
    _save(out_path)


def plot_correlation_heatmap(X: pd.DataFrame, out_path):
    corr = X.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr.to_numpy(), aspect="auto")
    plt.colorbar()
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=45, ha="right")
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.title("Matrica korelacije (Pearson)")
    _save(out_path)


def plot_vif(vif_df: pd.DataFrame, out_path):
    df = vif_df.sort_values("VIF", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["VIF"])
    plt.xlabel("VIF")
    plt.title("VIF po atributu")
    _save(out_path)


def plot_linearity_grid(X: pd.DataFrame, y: np.ndarray, target: str, out_path):
    cols = list(X.columns)
    n = len(cols)
    rows = 2
    cols_n = int(np.ceil(n / rows))

    plt.figure(figsize=(4 * cols_n, 6))
    for i, c in enumerate(cols, start=1):
        plt.subplot(rows, cols_n, i)
        plt.scatter(X[c].to_numpy(dtype=float), y, s=10)
        plt.xlabel(c)
        plt.ylabel(target)
    plt.suptitle(f"Linearnost: ulazi vs {target}")
    _save(out_path)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path, title: str):
    e = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, e, s=10)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Predikcije")
    plt.ylabel("Reziduali")
    plt.title(title)
    _save(out_path)


def plot_qq(residuals: np.ndarray, out_path, title: str):
    theo, emp = qq_points(residuals)
    plt.figure(figsize=(6, 6))
    plt.scatter(theo, emp, s=10)
    if len(theo) > 0:
        x = np.array([theo.min(), theo.max()])
        plt.plot(x, x, linewidth=1)
    plt.xlabel("Teorijski kvantili")
    plt.ylabel("Empirijski kvantili reziduala")
    plt.title(title)
    _save(out_path)


def plot_rmse_boxplot(results_df: pd.DataFrame, out_path, title: str):
    if results_df.empty:
        return

    models = sorted(results_df["model"].unique())
    data = [results_df.loc[results_df["model"] == m, "rmse"].to_numpy() for m in models]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=models, showmeans=True)
    plt.ylabel("RMSE")
    plt.title(title)
    plt.xticks(rotation=15)
    _save(out_path)


def plot_rel_rmse_change(rel_df: pd.DataFrame, out_path, title: str):
    if rel_df.empty:
        return

    models = sorted(rel_df["model"].unique())
    data = [
        rel_df.loc[rel_df["model"] == m, "rmse_rel_change"].to_numpy() for m in models
    ]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=models, showmeans=True)
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Relativna promena RMSE")
    plt.title(title)
    plt.xticks(rotation=15)
    _save(out_path)


def plot_coef_boxplots(coef_df: pd.DataFrame, out_dir, suffix: str):
    """Boxplotovi koeficijenata (samo linearni modeli)."""
    if coef_df.empty:
        return

    for (scenario, model), sub in coef_df.groupby(["scenario", "model"]):
        feats = sorted(sub["feature"].unique())
        data = [sub.loc[sub["feature"] == f, "coef"].to_numpy() for f in feats]

        plt.figure(figsize=(max(10, len(feats) * 0.6), 5))
        plt.boxplot(data, labels=feats, showmeans=True)
        plt.title(f"Koeficijenti - {model} - {scenario} ({suffix})")
        plt.ylabel("coef")
        plt.xticks(rotation=45, ha="right")
        _save(out_dir / f"coef_box_{scenario}_{model}_{suffix}.png")
