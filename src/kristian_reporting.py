# =======================================================
# Agregacija metrika i priprema tabela za boxplot analizu
# =======================================================

import pandas as pd


def aggregate_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Agregacija metrika po scenario/model."""
    if results_df.empty:
        return pd.DataFrame()

    g = results_df.groupby(["scenario", "model"])
    out = g.agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        rmse_median=("rmse", "median"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        adj_r2_mean=("adj_r2", "mean"),
        adj_r2_std=("adj_r2", "std"),
        n=("rmse", "size"),
    ).reset_index()

    out["rmse_iqr"] = g["rmse"].quantile(0.75).values - g["rmse"].quantile(0.25).values

    out = out.fillna(0.0)
    return out.sort_values(
        ["scenario", "rmse_mean", "rmse_std"], ascending=[True, True, True]
    )


def relative_rmse_change(
    base_df: pd.DataFrame, scenario_df: pd.DataFrame, key_col: str
):
    """
    Relativna promena RMSE po istom indeksu eksperimenta.
    key_col: "repeat" (split) ili "fold" (kfold)
    """
    if base_df.empty or scenario_df.empty:
        return pd.DataFrame()

    b = base_df[[key_col, "model", "rmse"]].rename(columns={"rmse": "rmse_base"})
    s = scenario_df[[key_col, "model", "rmse"]].rename(
        columns={"rmse": "rmse_scenario"}
    )

    m = b.merge(s, on=[key_col, "model"], how="inner")
    m["rmse_rel_change"] = (m["rmse_scenario"] - m["rmse_base"]) / (
        m["rmse_base"] + 1e-12
    )
    return m


def aggregate_relative_change(rel_df: pd.DataFrame) -> pd.DataFrame:
    if rel_df.empty:
        return pd.DataFrame()

    g = rel_df.groupby("model")["rmse_rel_change"]
    out = g.agg(
        rel_mean="mean",
        rel_std="std",
        rel_median="median",
        rel_q25=lambda s: s.quantile(0.25),
        rel_q75=lambda s: s.quantile(0.75),
    ).reset_index()
    out["rel_iqr"] = out["rel_q75"] - out["rel_q25"]

    out = out.fillna(0.0)
    return out.sort_values(["rel_mean", "rel_std"], ascending=[True, True])
