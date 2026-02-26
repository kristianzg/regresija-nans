# =============================================================================
# Implementacija regularizovanih, robusnih i XGBoost modela
# =============================================================================

from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, RANSACRegressor, LinearRegression
from xgboost import XGBRegressor


def build_non_ols_models(cfg: dict):
    """Vraća listu tuple-ova: (ime_modela, konstruktor, is_linear)."""

    def ridge():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=cfg["ridge_alpha"], random_state=cfg["random_seed"])),
        ])

    def lasso():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=cfg["lasso_alpha"], max_iter=cfg["lasso_max_iter"], tol=cfg["lasso_tol"], random_state=cfg["random_seed"])),
        ])

    def huber():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", HuberRegressor(epsilon=cfg["huber_epsilon"], max_iter=cfg["huber_max_iter"])),
        ])

    def ransac():
        base = LinearRegression()
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RANSACRegressor(
                estimator=base,
                min_samples=cfg["ransac_min_samples"],
                residual_threshold=cfg["ransac_residual_threshold"],
                random_state=cfg["random_seed"],
                max_trials=cfg["ransac_max_trials"],
            )),
        ])

    def xgboost():
        # Nelinearan ensemble model
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=cfg["xgb_n_estimators"],
            max_depth=cfg["xgb_max_depth"],
            learning_rate=cfg["xgb_learning_rate"],
            subsample=cfg["xgb_subsample"],
            colsample_bytree=cfg["xgb_colsample_bytree"],
            reg_alpha=cfg["xgb_reg_alpha"],
            reg_lambda=cfg["xgb_reg_lambda"],
            random_state=cfg["random_seed"],
            n_jobs=cfg["xgb_n_jobs"],
            tree_method="hist",
        )

    return [
        ("Ridge", ridge, True),
        ("Lasso", lasso, True),
        ("Huber", huber, True),
        ("RANSAC", ransac, True),
        ("XGBoost", xgboost, False),
    ]


def extract_linear_coefs(model, feature_names):
    """
    Vraća koeficijente za linearne modele.
    Za XGBoost vraća None (nema standardne linearne koeficijente).
    """
    core = model.named_steps["model"] if hasattr(model, "named_steps") else model

    if hasattr(core, "estimator_") and hasattr(core.estimator_, "coef_"):
        coefs = np.asarray(core.estimator_.coef_, dtype=float).ravel()
    else:
        coefs = getattr(core, "coef_", None)
        if coefs is None:
            return None
        coefs = np.asarray(coefs, dtype=float).ravel()

    if len(feature_names) != len(coefs):
        m = min(len(feature_names), len(coefs))
        feature_names = feature_names[:m]
        coefs = coefs[:m]

    return dict(zip(feature_names, coefs))
