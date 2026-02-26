# =============================================================================
# Konfiguracija projekta
# =============================================================================

DEFAULT_CONFIG = {
    # Ponovljeni split eksperimenti
    "repeats": 10,
    "test_size": 0.2,
    "random_seed": 42,

    # Repeated K-Fold
    "kfold_splits": 5,
    "kfold_repeats": 2,

    # Perturbacije (kontrolisano kršenje pretpostavki)
    "hetero_strength": 1.5,
    "heavy_tail_df": 3,
    "heavy_tail_scale": 1.0,
    "collin_eps": 0.02,
    "collin_copies": 3,

    # Miloš - modeli (bez OLS)
    "ridge_alpha": 1.0,
    "lasso_alpha": 0.02,
    "lasso_max_iter": 5000,
    "lasso_tol": 1e-3,
    "huber_epsilon": 1.35,
    "huber_max_iter": 200,
    "ransac_min_samples": 0.6,
    "ransac_residual_threshold": None,
    "ransac_max_trials": 20,

    # XGBoost (umereni parametri radi stabilnog vremena izvršavanja)
    "xgb_n_estimators": 30,
    "xgb_max_depth": 3,
    "xgb_learning_rate": 0.1,
    "xgb_subsample": 0.9,
    "xgb_colsample_bytree": 0.9,
    "xgb_reg_alpha": 0.0,
    "xgb_reg_lambda": 1.0,
    "xgb_n_jobs": 1,
}
