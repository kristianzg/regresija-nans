# regresija-nans

# Projekat: Uticaj kršenja pretpostavki linearne regresije na tačnost i stabilnost regresionih modela

## Koraci izrade

### 1) Eksplorativna analiza podataka (Kristian)

- Osnovne statističke karakteristike:
  - `src/kristian_data_eda.py` (`basic_stats`)
- Vizuelizacija distribucija promenljivih:
  - `src/kristian_visualizations.py` (`plot_feature_distributions`)
- Kolinearnost između atributa:
  - korelacija: `plot_correlation_heatmap`
  - VIF: `src/kristian_assumptions.py` (`compute_vif`)

### 2) Provera pretpostavki linearne regresije (Kristian)

- Linearnost:
  - `plot_linearity_grid`
- Nezavisnost grešaka:
  - `durbin_watson` + `residual_summary`
- Normalnost grešaka:
  - `plot_qq` + `residual_skew`, `residual_kurtosis`
- Jednaka varijansa:
  - `plot_residuals` + indikatori `corr_abs_resid_pred` i `bp_like_lm`
- Multikolinearnost (VIF):
  - `compute_vif`

### 3) Izgradnja regresionih modela

- OLS (Kristian):
  - `src/kristian_ols.py`
- Ridge, Lasso, Huber, RANSAC, XGBoost (Miloš):
  - `src/milos_models.py`

### 4) Kontrolisano narušavanje pretpostavki (Kristian)

- Skoro-linearno zavisni atributi:
  - `add_multicollinearity`
- Heteroskedastični šum:
  - `heteroskedastic_noise`
- Nenormalne greške:
  - `heavy_tailed_noise`

### 5) Analiza stabilnosti modela

- Poređenje koeficijenata (Kristian + Miloš):
  - koeficijenti i statistike stabilnosti koeficijenata: `milos_experiments.py` + `coef_summary`
  - dodatni helper za boxplot koeficijenata (po potrebi): `kristian_visualizations.py` (`plot_coef_boxplots`)
- Osetljivost na narušene pretpostavke (Miloš):
  - ponovljeni eksperimenti: `repeated_splits`, `repeated_kfold`
  - relativna promena RMSE: `kristian_reporting.py` (`relative_rmse_change`)
- Poređenje ponašanja svih modela (uključujući XGBoost):
  - agregacije: `kristian_reporting.py` (`aggregate_metrics`)
  - interpretacija: `milos_interpretation.py`

### 6) Poređenje rezultata i interpretacija

- Kvantitativno poređenje performansi:
  - tabele `agg_splits_heating.csv`, `agg_kfold_heating.csv`
- Kvalitativna analiza:
  - `outputs/tables/interpretacija_milos.md`
- Krajnji zaključak:
  - `outputs/tables/zakljucak_zajednicki.md`

---

## Podela rada (po fajlovima)

### Kristian Zhou-Gubić, SV25/2024

- `src/kristian_data_eda.py`
- `src/kristian_assumptions.py`
- `src/kristian_ols.py`
- `src/kristian_perturbations.py`
- `src/kristian_visualizations.py`
- `src/kristian_reporting.py`

### Miloš Vukić, SV22/2024

- `src/milos_models.py`
- `src/milos_experiments.py`
- `src/milos_interpretation.py`

### Zajedničko

- `run_project.py` (orkestracija)
- `src/config.py`
- `src/zajednicki_metrics.py`

---

## Pokretanje

```bash
pip install -r requirements.txt
python run_project.py
```

Opcionalno (brzi test):

```bash
python run_project.py --repeats 5 --kfold_splits 3 --kfold_repeats 1
```

Rezultati se nalaze u:

- `outputs/figures`
- `outputs/tables`
