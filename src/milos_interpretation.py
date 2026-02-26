# =============================================================================
# Poređenje performansi i interpretacija rezultata
# =============================================================================

from __future__ import annotations
import pandas as pd


PERTURB_SCENARIOS = ["multicollinearity", "heteroskedastic", "non_normal"]


def _base_ranking_text(split_agg: pd.DataFrame) -> str:
    base = split_agg[split_agg["scenario"] == "base"].sort_values("rmse_mean")
    if base.empty:
        return "Nema dostupnih rezultata za bazni scenario."

    lines = ["### Kvantitativno poređenje (bazni scenario)"]
    for i, row in enumerate(base.itertuples(index=False), start=1):
        lines.append(
            f"{i}. {row.model}: RMSE={row.rmse_mean:.4f} (std={row.rmse_std:.4f}), "
            f"R²={row.r2_mean:.4f}, Adj.R²={row.adj_r2_mean:.4f}"
        )
    return "\n".join(lines)


def _stability_text(rel_tables: dict[str, pd.DataFrame]) -> str:
    lines = ["### Analiza stabilnosti i osetljivosti"]

    for scen in PERTURB_SCENARIOS:
        df = rel_tables.get(scen)
        if df is None or df.empty:
            lines.append(f"- {scen}: nema rezultata.")
            continue

        best = df.sort_values(["rel_mean", "rel_std"]).iloc[0]
        worst = df.sort_values(["rel_mean", "rel_std"], ascending=[False, False]).iloc[0]

        lines.append(
            f"- Scenario **{scen}**: najstabilniji je **{best['model']}** "
            f"(prosečna rel. promena RMSE={best['rel_mean']:.4f}), "
            f"dok je najosetljiviji **{worst['model']}** "
            f"(prosečna rel. promena RMSE={worst['rel_mean']:.4f})."
        )

    lines.append(
        "- Vrednosti relativne promene RMSE bliže nuli znače veću stabilnost modela "
        "u odnosu na narušavanje pretpostavki."
    )

    return "\n".join(lines)


def _qualitative_text(split_agg: pd.DataFrame) -> str:
    lines = ["### Kvalitativna analiza dobijenih rezultata"]

    base = split_agg[split_agg["scenario"] == "base"].sort_values("rmse_mean")
    if not base.empty:
        best_model = base.iloc[0]["model"]
        lines.append(
            f"- U baznom scenariju, model **{best_model}** ostvaruje najbolji kompromis "
            "između tačnosti i stabilnosti (najniži prosečni RMSE)."
        )

    for scen in PERTURB_SCENARIOS:
        sub = split_agg[split_agg["scenario"] == scen]
        if sub.empty:
            continue
        ordered = sub.sort_values("rmse_mean")
        lines.append(
            f"- U scenariju **{scen}**, najbolji po RMSE je **{ordered.iloc[0]['model']}**, "
            f"dok je najslabiji **{ordered.iloc[-1]['model']}**."
        )

    lines.append(
        "- Linearni modeli omogućavaju interpretaciju preko koeficijenata, dok XGBoost "
        "može bolje opisati nelinearne odnose ali bez direktnih linearnih koeficijenata."
    )

    return "\n".join(lines)


def build_joint_conclusion(split_agg: pd.DataFrame, rel_tables: dict[str, pd.DataFrame]) -> str:
    lines = ["## Zajednički zaključak"]

    base = split_agg[split_agg["scenario"] == "base"].sort_values("rmse_mean")
    if not base.empty:
        top2 = base.head(2)["model"].tolist()
        lines.append(
            f"- U baznim uslovima, najtačniji modeli su: **{top2[0]}**"
            + (f" i **{top2[1]}**." if len(top2) > 1 else ".")
        )

    for scen, rel_df in rel_tables.items():
        if rel_df is None or rel_df.empty:
            continue
        stable = rel_df.sort_values(["rel_mean", "rel_std"]).iloc[0]["model"]
        lines.append(f"- Pri narušavanju pretpostavki ({scen}), najstabilniji model je **{stable}**.")

    lines.append(
        "- Regularizovani i robusni modeli često bolje podnose kršenje pretpostavki od čistog OLS-a, "
        "dok XGBoost može dati konkurentnu tačnost uz drugačiji bias/variance profil."
    )
    lines.append(
        "- Ukupan izbor modela treba donositi kao kompromis između tačnosti (RMSE/R²), "
        "stabilnosti (promena RMSE) i interpretabilnosti (koeficijenti)."
    )

    return "\n".join(lines)


def build_interpretation_markdown(
    split_agg: pd.DataFrame,
    kfold_agg: pd.DataFrame,
    rel_tables_split: dict[str, pd.DataFrame],
) -> str:
    lines = [
        "# Interpretacija rezultata",
        "",
        _base_ranking_text(split_agg),
        "",
        _stability_text(rel_tables_split),
        "",
        _qualitative_text(split_agg),
        "",
        "### Napomena o k-fold evaluaciji",
    ]

    kb = kfold_agg[kfold_agg["scenario"] == "base"].sort_values("rmse_mean")
    if kb.empty:
        lines.append("- K-fold rezultati nisu dostupni.")
    else:
        top = kb.iloc[0]
        lines.append(
            f"- U k-fold baznom scenariju, najbolji prosečni RMSE ima **{top['model']}** "
            f"(RMSE={top['rmse_mean']:.4f}, std={top['rmse_std']:.4f})."
        )

    return "\n".join(lines)
