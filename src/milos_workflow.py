# =============================================================================
# Workflow: modeli, ponovljeni eksperimenti, stabilnost i interpretacija
# =============================================================================

from __future__ import annotations
from pathlib import Path
import pandas as pd

from .milos_models import build_non_ols_models
from .milos_experiments import repeated_splits, repeated_kfold, coef_summary
from .milos_interpretation import build_interpretation_markdown, build_joint_conclusion


def milos_model_registry(cfg: dict):
    """Modeli: Ridge, Lasso, Huber, RANSAC, XGBoost."""
    return build_non_ols_models(cfg)


def run_split_experiments(X, y, models, cfg: dict, scenarios: dict, out_tab: Path):
    """Ponovljeni split eksperimenti po svim scenarijima."""
    split_frames = []
    coef_frames = []

    for sc_name, sc_fn in scenarios.items():
        print(f"[split] scenario: {sc_name}")
        res_df, coef_df = repeated_splits(X, y, models, cfg, sc_name, sc_fn)

        res_df.to_csv(out_tab / f"results_splits_{sc_name}_heating.csv", index=False)
        split_frames.append(res_df)

        if not coef_df.empty:
            coef_df.to_csv(out_tab / f"coefs_{sc_name}_heating.csv", index=False)
            coef_frames.append(coef_df)

    results_splits = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()
    coef_all = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    return results_splits, coef_all


def run_kfold_experiments(X, y, models, cfg: dict, scenarios: dict, out_tab: Path):
    """Ponovljeni k-fold eksperimenti po svim scenarijima."""
    kfold_frames = []

    for sc_name, sc_fn in scenarios.items():
        print(f"[kfold] scenario: {sc_name}")
        kdf = repeated_kfold(X, y, models, cfg, sc_name, sc_fn)
        kdf.to_csv(out_tab / f"results_kfold_{sc_name}_heating.csv", index=False)
        kfold_frames.append(kdf)

    results_kfold = pd.concat(kfold_frames, ignore_index=True) if kfold_frames else pd.DataFrame()
    return results_kfold


def save_coef_analysis(coef_all: pd.DataFrame, out_tab: Path):
    """Analiza stabilnosti koeficijenata linearnih modela."""
    if coef_all.empty:
        return

    coef_all.to_csv(out_tab / "coefs_all_heating.csv", index=False)
    csum = coef_summary(coef_all)
    csum.to_csv(out_tab / "coef_summary_heating.csv", index=False)


def write_interpretation_outputs(split_agg: pd.DataFrame, kfold_agg: pd.DataFrame, rel_tables_split: dict, out_tab: Path):
    """Kvantitativna i kvalitativna interpretacija i zaključak."""
    interp_md = build_interpretation_markdown(
        split_agg=split_agg,
        kfold_agg=kfold_agg,
        rel_tables_split=rel_tables_split,
    )
    (out_tab / "interpretacija_milos.md").write_text(interp_md, encoding="utf-8")

    joint_md = build_joint_conclusion(split_agg=split_agg, rel_tables=rel_tables_split)
    (out_tab / "zakljucak_zajednicki.md").write_text(joint_md, encoding="utf-8")
