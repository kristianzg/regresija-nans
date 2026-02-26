# ==================================================================
# Workflow: EDA, pretpostavke, perturbacije, agregacija i boxplotovi
# ==================================================================

from pathlib import Path
import pandas as pd
import numpy as np

from .kristian_data_eda import (
    load_energy_efficiency,
    split_xy_heating,
    basic_stats,
    FEATURE_COLS,
    TARGET_COL,
)
from .kristian_assumptions import compute_vif, residual_summary
from .kristian_ols import build_ols_model
from .kristian_perturbations import (
    add_multicollinearity,
    heteroskedastic_noise,
    heavy_tailed_noise,
)
from .kristian_visualizations import (
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_vif,
    plot_linearity_grid,
    plot_residuals,
    plot_qq,
    plot_rmse_boxplot,
)
from .kristian_reporting import (
    aggregate_metrics,
    relative_rmse_change,
    aggregate_relative_change,
)


def load_and_prepare_heating_data(data_dir: Path):
    """Ucitavanje podataka i izdvajanje X/y za heating_load (Y1)."""
    df = load_energy_efficiency(data_dir)
    X, y = split_xy_heating(df)
    return df, X, y


def run_eda_and_assumption_checks(
    df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, out_fig: Path, out_tab: Path
):
    """
    1) EDA: statistike, distribucije, korelacija, VIF
    2) Provera pretpostavki LR: linearnost, reziduali, QQ, DW i indikatori homoskedasticnosti
    """
    # EDA
    basic_stats(df).to_csv(out_tab / "basic_stats.csv")
    plot_feature_distributions(
        df[FEATURE_COLS + [TARGET_COL]], out_fig / "feature_distributions.png"
    )

    plot_correlation_heatmap(X, out_fig / "corr_heatmap.png")
    vif_df = compute_vif(X)
    vif_df.to_csv(out_tab / "vif.csv", index=False)
    plot_vif(vif_df, out_fig / "vif.png")

    # Pretpostavke: linearnost
    plot_linearity_grid(X, y, TARGET_COL, out_fig / "linearity_heating.png")

    # OLS fit nad celim skupom radi analize reziduala
    ols = build_ols_model()
    ols.fit(X, y)
    pred = ols.predict(X)
    resid = y - pred

    # Homoskedasticnost + normalnost (vizuelno)
    plot_residuals(
        y,
        pred,
        out_fig / "residuals_vs_pred_heating.png",
        "Reziduali vs predikcije (heating)",
    )
    plot_qq(resid, out_fig / "qq_plot_heating.png", "QQ-plot reziduala (heating)")

    # Nezavisnost gresaka (DW) + summary
    rs = residual_summary(y, pred)
    pd.DataFrame([{"target": TARGET_COL, **rs}]).to_csv(
        out_tab / "assumption_summary_heating.csv", index=False
    )


def kristian_ols_registry():
    """OLS deo modela"""
    return [("OLS", build_ols_model, True)]


def build_kristian_scenarios(cfg: dict):
    """Generatori kontrolisanog kršenja pretpostavki linearne regresije."""

    def sc_base(X0, y0, rng):
        return X0.copy(), y0.copy()

    def sc_multicol(X0, y0, rng):
        X2 = add_multicollinearity(
            X0, eps=cfg["collin_eps"], copies=cfg["collin_copies"], rng=rng
        )
        return X2, y0.copy()

    def sc_hetero(X0, y0, rng):
        driver = X0.iloc[:, 0].to_numpy(dtype=float)
        y2 = heteroskedastic_noise(
            y0, driver=driver, strength=cfg["hetero_strength"], rng=rng
        )
        return X0.copy(), y2

    def sc_non_normal(X0, y0, rng):
        y2 = heavy_tailed_noise(
            y0, df=cfg["heavy_tail_df"], scale=cfg["heavy_tail_scale"], rng=rng
        )
        return X0.copy(), y2

    return {
        "base": sc_base,
        "multicollinearity": sc_multicol,
        "heteroskedastic": sc_hetero,
        "non_normal": sc_non_normal,
    }


def postprocess_split_results(
    results_splits: pd.DataFrame, out_fig: Path, out_tab: Path
):
    """agregacija metrika, boxplot i rel. promena RMSE (split)."""
    results_splits.to_csv(out_tab / "results_splits_heating.csv", index=False)

    split_agg = aggregate_metrics(results_splits)
    split_agg.to_csv(out_tab / "agg_splits_heating.csv", index=False)

    # Boxplot performansi (split, bazni scenario)
    base_split_plot = results_splits[results_splits["scenario"] == "base"].copy()
    plot_rmse_boxplot(
        base_split_plot,
        out_fig / "rmse_box_split_base_heating.png",
        title="RMSE boxplot (split) - base - heating",
    )

    # Relativna promena RMSE vs base (split)
    rel_tables_split = {}
    base_split = results_splits[results_splits["scenario"] == "base"].copy()
    for sc_name in ["multicollinearity", "heteroskedastic", "non_normal"]:
        sc_df = results_splits[results_splits["scenario"] == sc_name].copy()
        rel = relative_rmse_change(base_split, sc_df, key_col="repeat")
        rel.to_csv(
            out_tab / f"rmse_rel_change_split_{sc_name}_heating.csv", index=False
        )

        rel_agg = aggregate_relative_change(rel)
        rel_agg.to_csv(
            out_tab / f"rmse_rel_change_split_{sc_name}_agg_heating.csv", index=False
        )
        rel_tables_split[sc_name] = rel_agg

    return split_agg, rel_tables_split


def postprocess_kfold_results(
    results_kfold: pd.DataFrame, out_fig: Path, out_tab: Path
):
    """agregacija metrika, boxplot i rel. promena RMSE (k-fold)."""
    results_kfold.to_csv(out_tab / "results_kfold_heating.csv", index=False)

    kfold_agg = aggregate_metrics(results_kfold)
    kfold_agg.to_csv(out_tab / "agg_kfold_heating.csv", index=False)

    # Boxplot performansi (k-fold, bazni scenario)
    base_kfold_plot = results_kfold[results_kfold["scenario"] == "base"].copy()
    plot_rmse_boxplot(
        base_kfold_plot,
        out_fig / "rmse_box_kfold_base_heating.png",
        title="RMSE boxplot (k-fold) - base - heating",
    )

    # Relativna promena RMSE vs base (k-fold)
    base_kfold = results_kfold[results_kfold["scenario"] == "base"].copy()
    for sc_name in ["multicollinearity", "heteroskedastic", "non_normal"]:
        sc_df = results_kfold[results_kfold["scenario"] == sc_name].copy()
        rel = relative_rmse_change(base_kfold, sc_df, key_col="fold")
        rel.to_csv(
            out_tab / f"rmse_rel_change_kfold_{sc_name}_heating.csv", index=False
        )

        rel_agg = aggregate_relative_change(rel)
        rel_agg.to_csv(
            out_tab / f"rmse_rel_change_kfold_{sc_name}_agg_heating.csv", index=False
        )

    return kfold_agg
