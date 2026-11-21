"""Stage 01: EDA, outlier limits, sensor VIF and target distribution MLE."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

from pems_common import (
    ANALYSIS,
    FIGURES_DIR,
    RESULTS_DIR,
    SENSORS,
    TARGETS,
    ensure_directories,
    random_split_indices,
)

load_data = import_module("00_load_data").load_data

TABLE01_COLUMNS = [
    "column",
    "n",
    "mean",
    "std",
    "min",
    "q01",
    "q25",
    "median",
    "q75",
    "q99",
    "max",
    "missing",
    "iqr_low",
    "iqr_high",
    "iqr_outliers",
    "sigma3_low",
    "sigma3_high",
    "sigma3_outliers",
    "vif",
]
TABLE02_COLUMNS = [
    "target",
    "distribution",
    "parameters",
    "n",
    "loglik",
    "aic",
    "bic",
    "best_by_loglik",
]


def training_partition(df: pd.DataFrame):
    """Return the canonical random 80% descriptive-training partition."""
    train_idx, holdout_idx = random_split_indices(len(df))
    return df.iloc[train_idx].copy(), df.iloc[holdout_idx].copy()


def descriptive_table(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ANALYSIS:
        x = train[column].to_numpy(dtype=float)
        q01, q25, median, q75, q99 = np.quantile(
            x, [0.01, 0.25, 0.50, 0.75, 0.99]
        )
        iqr = q75 - q25
        iqr_low = q25 - 1.5 * iqr
        iqr_high = q75 + 1.5 * iqr
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1))
        sigma3_low = mean - 3.0 * std
        sigma3_high = mean + 3.0 * std
        vif = np.nan
        if column in SENSORS:
            others = [sensor for sensor in SENSORS if sensor != column]
            regression = LinearRegression().fit(train[others], train[column])
            r_squared = float(regression.score(train[others], train[column]))
            vif = float(1.0 / (1.0 - r_squared))
        rows.append(
            {
                "column": column,
                "n": len(x),
                "mean": mean,
                "std": std,
                "min": float(np.min(x)),
                "q01": float(q01),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "q99": float(q99),
                "max": float(np.max(x)),
                "missing": int(np.isnan(x).sum()),
                "iqr_low": float(iqr_low),
                "iqr_high": float(iqr_high),
                "iqr_outliers": int(np.sum((x < iqr_low) | (x > iqr_high))),
                "sigma3_low": float(sigma3_low),
                "sigma3_high": float(sigma3_high),
                "sigma3_outliers": int(
                    np.sum((x < sigma3_low) | (x > sigma3_high))
                ),
                "vif": vif,
            }
        )
    return pd.DataFrame(rows, columns=TABLE01_COLUMNS)


def plot_univariate(train: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    n_vars = len(ANALYSIS)
    n_rows = int(np.ceil(n_vars / 2))
    fig, axes = plt.subplots(
        n_rows, 4, figsize=(17, 3.2 * n_rows), constrained_layout=True
    )
    palette = sns.color_palette("husl", n_vars)
    for index, column in enumerate(ANALYSIS):
        row = index // 2
        offset = (index % 2) * 2
        kde_ax = axes[row, offset]
        box_ax = axes[row, offset + 1]
        x = train[column].to_numpy(dtype=float)
        sns.kdeplot(x=x, ax=kde_ax, fill=True, color=palette[index])
        kde_ax.set_title(f"{column} — KDE")
        kde_ax.set_xlabel(column)
        sns.boxplot(x=x, ax=box_ax, color=palette[index])
        box_ax.set_title(f"{column} — boxplot")
        box_ax.set_xlabel(column)
    for index in range(n_vars, n_rows * 2):
        row = index // 2
        offset = (index % 2) * 2
        axes[row, offset].axis("off")
        axes[row, offset + 1].axis("off")
    fig.suptitle(
        "CHE F315 gas turbine: univariate KDE and boxplots "
        "(random 80% training partition)",
        fontsize=14,
    )
    fig.savefig(FIGURES_DIR / "fig01_univariate_kde_boxplots.png", dpi=140)
    plt.close(fig)


def plot_correlation(train: pd.DataFrame) -> None:
    correlation = train[ANALYSIS].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Pearson correlation matrix — 11 analysis variables", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_correlation_heatmap_11x11.png", dpi=150)
    plt.close(fig)


def fit_mle_table(train: pd.DataFrame):
    rows = []
    fits = {}
    for target in TARGETS:
        x = train[target].to_numpy(dtype=float)
        n = len(x)
        candidates = {}
        candidates["norm"] = (stats.norm, stats.norm.fit(x), {"loc", "scale"})
        gamma_params = stats.gamma.fit(x, floc=0)
        lognorm_params = stats.lognorm.fit(x, floc=0)
        candidates["gamma"] = (
            stats.gamma,
            gamma_params,
            {"a", "scale"},
        )
        candidates["lognorm"] = (
            stats.lognorm,
            lognorm_params,
            {"s", "scale"},
        )
        logliks = {}
        for name, (distribution, params, free_names) in candidates.items():
            log_density = distribution.logpdf(x, *params)
            loglik = float(np.sum(log_density))
            k = len(free_names)
            aic = float(2 * k - 2 * loglik)
            bic = float(k * np.log(n) - 2 * loglik)
            logliks[name] = loglik
            parameter_dict = {
                key: float(value)
                for key, value in zip(
                    ("loc", "scale") if name == "norm" else
                    (("a", "loc", "scale") if name == "gamma" else
                     ("s", "loc", "scale")),
                    params,
                    strict=True,
                )
            }
            rows.append(
                {
                    "target": target,
                    "distribution": name,
                    "parameters": json.dumps(parameter_dict, sort_keys=True),
                    "n": n,
                    "loglik": loglik,
                    "aic": aic,
                    "bic": bic,
                }
            )
            fits[(target, name)] = (distribution, params)
        best = max(logliks, key=lambda key: logliks[key])
        for row in rows:
            if row["target"] == target:
                row["best_by_loglik"] = row["distribution"] == best
    return pd.DataFrame(rows, columns=TABLE02_COLUMNS), fits


def plot_mle(train: pd.DataFrame, fits: dict) -> None:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(15, 5.5))
    colors = {"norm": "#1f77b4", "gamma": "#d62728", "lognorm": "#2ca02c"}
    for ax, target in zip(axes, TARGETS, strict=True):
        x = train[target].to_numpy(dtype=float)
        ax.hist(
            x,
            bins=60,
            density=True,
            alpha=0.35,
            color="grey",
            label="Histogram",
        )
        kde = stats.gaussian_kde(x)
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 400)
        ax.plot(grid, kde(grid), color="black", lw=1.6, label="KDE")
        for name in ("norm", "gamma", "lognorm"):
            distribution, params = fits[(target, name)]
            density = distribution.pdf(grid, *params)
            loglik = float(np.sum(distribution.logpdf(x, *params)))
            ax.plot(
                grid,
                density,
                color=colors[name],
                lw=2.0,
                label=f"{name} MLE (loglik={loglik:,.0f})",
            )
        ax.set_title(f"{target} distribution fits (MLE)")
        ax.set_xlabel(target)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("MLE distribution fits on the random 80% training partition")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig03_mle_distribution_fits.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    df = load_data()
    train, holdout = training_partition(df)

    table01 = descriptive_table(train)
    table01.to_csv(RESULTS_DIR / "table01_eda_outliers_vif.csv", index=False)
    plot_univariate(train)
    plot_correlation(train)

    table02, fits = fit_mle_table(train)
    table02.to_csv(RESULTS_DIR / "table02_mle_loglik.csv", index=False)
    plot_mle(train, fits)

    print(f"Descriptive train rows: {len(train)}; holdout rows: {len(holdout)}")
    print(table01.to_string(index=False))
    print()
    print(table02.to_string(index=False))


if __name__ == "__main__":
    main()
