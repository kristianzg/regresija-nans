# =============================================================================
# Centralni pipeline
# =============================================================================

from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path
import gc

from .config import DEFAULT_CONFIG

# Kristian workflow
from .kristian_workflow import (
    load_and_prepare_heating_data,
    run_eda_and_assumption_checks,
    kristian_ols_registry,
    build_kristian_scenarios,
    postprocess_split_results,
    postprocess_kfold_results,
)

# Miloš workflow
from .milos_workflow import (
    milos_model_registry,
    run_split_experiments,
    run_kfold_experiments,
    save_coef_analysis,
    write_interpretation_outputs,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=DEFAULT_CONFIG["repeats"])
    p.add_argument("--test_size", type=float, default=DEFAULT_CONFIG["test_size"])
    p.add_argument("--kfold_splits", type=int, default=DEFAULT_CONFIG["kfold_splits"])
    p.add_argument("--kfold_repeats", type=int, default=DEFAULT_CONFIG["kfold_repeats"])
    return p.parse_args()


def make_cfg(args):
    cfg = dict(DEFAULT_CONFIG)
    cfg["repeats"] = args.repeats
    cfg["test_size"] = args.test_size
    cfg["kfold_splits"] = args.kfold_splits
    cfg["kfold_repeats"] = args.kfold_repeats
    return cfg


def run_pipeline(cfg: dict, root: Path):
    data_dir = root / "data"
    out_fig = root / "outputs" / "figures"
    out_tab = root / "outputs" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    # data, EDA i pretpostavke
    df, X, y = load_and_prepare_heating_data(data_dir)
    run_eda_and_assumption_checks(df=df, X=X, y=y, out_fig=out_fig, out_tab=out_tab)

    # Modeli
    models = kristian_ols_registry() + milos_model_registry(cfg)

    # generatori scenario-a
    scenarios = build_kristian_scenarios(cfg)

    # split/kfold eksperimenti
    results_splits, coef_all = run_split_experiments(X, y, models, cfg, scenarios, out_tab)
    results_kfold = run_kfold_experiments(X, y, models, cfg, scenarios, out_tab)

    # agregacije i boxplotovi
    split_agg, rel_tables_split = postprocess_split_results(results_splits, out_fig, out_tab)
    kfold_agg = postprocess_kfold_results(results_kfold, out_fig, out_tab)

    # stabilnost koeficijenata + interpretacija
    save_coef_analysis(coef_all, out_tab)
    write_interpretation_outputs(split_agg, kfold_agg, rel_tables_split, out_tab)

    gc.collect()
    print("Rezultati su u outputs/figures i outputs/tables.")


def main():
    args = parse_args()
    cfg = make_cfg(args)
    root = Path(__file__).resolve().parent.parent
    run_pipeline(cfg=cfg, root=root)


if __name__ == "__main__":
    main()
