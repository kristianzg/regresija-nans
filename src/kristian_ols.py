# ===============================================
# Implementacija i analiza standardnog OLS modela
# ===============================================

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def build_ols_model():
    """Standardna visestruka linearna regresija (OLS) u pipeline-u sa skaliranjem"""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
