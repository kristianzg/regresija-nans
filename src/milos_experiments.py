# =============================================================================
# Ponovljeni eksperimenti (N split i k-fold) i osetljivost modela
# =============================================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold

from .zajednicki_metrics import mse, rmse, r2, adj_r2
from .milos_models import extract_linear_coefs


def evaluate_once(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    p = X_train.shape[1]
    out = {
        "mse": mse(y_test, pred),
        "rmse": rmse(y_test, pred),
        "r2": r2(y_test, pred),
        "adj_r2": adj_r2(y_test, pred, p=p),
    }
    return out, pred


def repeated_splits(X: pd.DataFrame, y: np.ndarray, models, cfg: dict, scenario_name: str, scenario_func):
    """N split eksperimenti za jedan scenario."""
    rows = []
    coef_rows = []

    for i in range(cfg["repeats"]):
        seed = cfg["random_seed"] + i * 1009
        rng = np.random.default_rng(seed)

        Xs, ys = scenario_func(X, y, rng)

        X_train, X_test, y_train, y_test = train_test_split(
            Xs, ys, test_size=cfg["test_size"], random_state=seed, shuffle=True
        )

        feature_names = list(X_train.columns)

        for model_name, make_model, is_linear in models:
            model = make_model()
            metrics, _ = evaluate_once(model, X_train, X_test, y_train, y_test)

            rows.append({
                "experiment": "split",
                "repeat": i,
                "seed": seed,
                "scenario": scenario_name,
                "model": model_name,
                **metrics,
            })

            if is_linear:
                coefs = extract_linear_coefs(model, feature_names)
                if coefs is not None:
                    for fname, c in coefs.items():
                        coef_rows.append({
                            "repeat": i,
                            "seed": seed,
                            "scenario": scenario_name,
                            "model": model_name,
                            "feature": fname,
                            "coef": float(c),
                        })

    return pd.DataFrame(rows), pd.DataFrame(coef_rows)


def repeated_kfold(X: pd.DataFrame, y: np.ndarray, models, cfg: dict, scenario_name: str, scenario_func):
    """Repeated K-Fold eksperimenti za jedan scenario."""
    rng = np.random.default_rng(cfg["random_seed"] + 777)
    Xs, ys = scenario_func(X, y, rng)

    rkf = RepeatedKFold(
        n_splits=cfg["kfold_splits"],
        n_repeats=cfg["kfold_repeats"],
        random_state=cfg["random_seed"],
    )

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(Xs)):
        X_train = Xs.iloc[train_idx]
        X_test = Xs.iloc[test_idx]
        y_train = ys[train_idx]
        y_test = ys[test_idx]

        for model_name, make_model, _ in models:
            model = make_model()
            metrics, _ = evaluate_once(model, X_train, X_test, y_train, y_test)

            rows.append({
                "experiment": "kfold",
                "fold": fold_idx,
                "scenario": scenario_name,
                "model": model_name,
                **metrics,
            })

    return pd.DataFrame(rows)


def sensitivity_relative_rmse(base_results: pd.DataFrame, scenario_results: pd.DataFrame, key_col: str):
    """
    Relativna promena RMSE po istom indeksu eksperimenta.
    key_col je "repeat" za split ili "fold" za kfold rezultate.
    """
    b = base_results[[key_col, "model", "rmse"]].rename(columns={"rmse": "rmse_base"})
    s = scenario_results[[key_col, "model", "rmse"]].rename(columns={"rmse": "rmse_scenario"})
    m = b.merge(s, on=[key_col, "model"], how="inner")
    m["rmse_rel_change"] = (m["rmse_scenario"] - m["rmse_base"]) / (m["rmse_base"] + 1e-12)
    return m


def coef_summary(coef_df: pd.DataFrame):
    if coef_df.empty:
        return pd.DataFrame()

    g = coef_df.groupby(["scenario", "model", "feature"])["coef"]
    out = g.agg(
        coef_mean="mean",
        coef_std="std",
        coef_median="median",
        coef_q25=lambda s: s.quantile(0.25),
        coef_q75=lambda s: s.quantile(0.75),
    ).reset_index()
    out["coef_iqr"] = out["coef_q75"] - out["coef_q25"]

    out = out.fillna(0.0)
    return out.sort_values(["scenario", "model", "coef_std"], ascending=[True, True, False])
