# ==============================================================
# Učitavanje podataka i EDA (Exploratory Data Analysis) priprema
# ==============================================================

from __future__ import annotations
from pathlib import Path
import pandas as pd

COLMAP = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}

FEATURE_COLS = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
]

TARGET_COL = (
    "heating_load"  # Fokusiracemo se na jednu izlaznu promenljivu, potreba za grejanje,
)


def load_energy_efficiency(data_dir: Path) -> pd.DataFrame:
    """Ucitavanje ENB2012_data.xlsx i preuzimanje naziva kolona."""
    xlsx_path = data_dir / "ENB2012_data.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Nedostaje fajl: {xlsx_path}")

    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: COLMAP.get(c, c) for c in df.columns})

    expected = list(COLMAP.values())
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Nedostaju očekivane kolone: {missing}")

    return df[expected].copy()


def split_xy_heating(df: pd.DataFrame):
    """Vraća X i y za target heating_load (Y1)."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].to_numpy(dtype=float)
    return X, y


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Osnovne statistike za sve ulazne i izlazne promenljive."""
    return df.describe(include="all").T
