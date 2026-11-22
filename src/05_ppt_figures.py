"""Stage 05: publication-quality, slide-ready PEMS summary figures."""

from __future__ import annotations

import json
from importlib import import_module

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pems_common import (
    FIGURES_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SENSORS,
    TARGETS,
    chronological_masks,
    ensure_directories,
    random_split_indices,
)

load_data = import_module("00_load_data").load_data
regression_utils = import_module("03_train_regression_cv")

PPT_DIR = FIGURES_DIR / "ppt"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#4D4D4D"
PROTOCOL_COLORS = {"random_5fold_holdout": BLUE, "chronological_2015": ORANGE}
VARIANT_COLORS = {"Raw": BLUE, "IQR": ORANGE, "3-sigma": GREEN}
REGIME_COLORS = [BLUE, ORANGE, GREEN]
MODEL_ORDER = ["OLS", "Ridge", "Lasso", "PCR", "Poly2_Ridge", "MLP"]
MODEL_LABELS = ["OLS", "Ridge", "Lasso", "PCR", "Poly2 Ridge", "MLP"]
REGIME_NAMES = ["Low load", "Mid load", "High load"]


def configure_style() -> None:
    """Set the large-format, high-contrast slide style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Write a 300-DPI PNG with a clean white background."""
    fig.savefig(
        PPT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Wrote {PPT_DIR / filename}")


def plot_vif() -> None:
    table = pd.read_csv(RESULTS_DIR / "table01_eda_outliers_vif.csv")
    table = table[table["column"].isin(SENSORS)].sort_values("vif", ascending=True)
    values = table["vif"].to_numpy(dtype=float)
    labels = table["column"].tolist()
    colors = [
        VERMILLION if value > 10 else ORANGE if value > 5 else BLUE
        for value in values
    ]

    fig, ax = plt.subplots(figsize=(11, 6.8))
    positions = np.arange(len(values))
    ax.barh(positions, values, color=colors, edgecolor="white", linewidth=1.2)
    for position, value in zip(positions, values, strict=True):
        ax.text(
            value * 1.06,
            position,
            f"{value:.1f}×",
            va="center",
            ha="left",
            fontsize=10,
            color=GREY,
        )
    ax.set_xscale("log")
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Variance inflation factor (log scale)")
    ax.set_ylabel("Sensor")
    ax.set_title("Sensor multicollinearity: VIF on the training partition")
    ax.axvline(5, color=GREY, linestyle="--", linewidth=1.2, label="VIF = 5")
    ax.axvline(10, color=VERMILLION, linestyle="--", linewidth=1.2, label="VIF = 10")
    ax.set_xlim(1.0, float(np.max(values)) * 1.45)
    ax.grid(axis="x", which="both")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "ppt01_sensor_vif_log.png")


def plot_mle_survival(df: pd.DataFrame) -> None:
    table = pd.read_csv(RESULTS_DIR / "table02_mle_loglik.csv")
    train_idx, _ = random_split_indices(len(df))
    train = df.iloc[train_idx]
    fit_colors = {"norm": GREY, "gamma": ORANGE, "lognorm": BLUE}
    fit_styles = {"norm": "--", "gamma": "-.", "lognorm": "-"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, target in zip(axes, TARGETS, strict=True):
        values = np.sort(train[target].to_numpy(dtype=float))
        n_values = len(values)
        empirical = np.arange(n_values, 0, -1) / n_values
        ax.step(
            values,
            empirical,
            where="post",
            color="black",
            linewidth=2.2,
            label="Empirical",
        )
        grid = np.linspace(values[0], values[-1], 900)
        best_name = ""
        for _, row in table[table["target"] == target].iterrows():
            distribution = row["distribution"]
            params = json.loads(row["parameters"])
            survival = 1.0 - getattr(stats, distribution).cdf(grid, **params)
            is_best = bool(row["best_by_loglik"])
            label = f"{distribution.capitalize()}"
            if is_best:
                best_name = distribution
                label += " (best loglik)"
            ax.plot(
                grid,
                survival,
                color=fit_colors[distribution],
                linestyle=fit_styles[distribution],
                linewidth=2.0 if is_best else 1.5,
                label=label,
            )
        q90 = float(np.quantile(train[target], 0.90))
        q95 = float(np.quantile(train[target], 0.95))
        for threshold, name, height in ((q90, "Q90", 0.97), (q95, "Q95", 0.79)):
            ax.axvline(threshold, color=VERMILLION, linestyle=":", linewidth=1.3)
            ax.text(
                threshold,
                height,
                f" {name} = {threshold:.2f}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="left",
                fontsize=9,
                color=VERMILLION,
            )
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.25)
        ax.set_xlabel(f"{target} concentration")
        ax.set_ylabel("Tail survival probability  P(X > x)")
        ax.set_title(
            f"{target}: empirical vs MLE tail survival\nbest fit = {best_name}",
            fontsize=15,
        )
        ax.grid(which="both", alpha=0.45)
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.suptitle(
        "Tail survival (CCDF) on the random 80% training partition",
        fontsize=15,
    )
    fig.tight_layout()
    save_figure(fig, "ppt02_mle_tail_survival.png")


def assign_regimes(df: pd.DataFrame) -> pd.DataFrame:
    train_idx, _ = random_split_indices(len(df))
    train = df.iloc[train_idx].copy()
    scaler = StandardScaler().fit(train[SENSORS])
    scaled = scaler.transform(train[SENSORS])
    model = KMeans(
        n_clusters=3,
        random_state=RANDOM_STATE,
        n_init=20,
    ).fit(scaled)
    centroids = scaler.inverse_transform(model.cluster_centers_)
    tey_index = SENSORS.index("TEY")
    order = np.argsort(centroids[:, tey_index])
    relabel = {int(old): int(new) for new, old in enumerate(order)}
    train["regime"] = np.array(
        [relabel[int(label)] for label in model.labels_], dtype=int
    )
    return train


def plot_regimes(df: pd.DataFrame) -> None:
    train = assign_regimes(df)
    counts = train["regime"].value_counts().sort_index().tolist()
    labels = [
        f"{name}\n(n={count:,})"
        for name, count in zip(REGIME_NAMES, counts, strict=True)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    for ax, target, scale in zip(
        axes, TARGETS, ["log", "linear"], strict=True
    ):
        sns.violinplot(
            data=train,
            x="regime",
            y=target,
            hue="regime",
            palette=REGIME_COLORS,
            inner="box",
            cut=0,
            linewidth=1.1,
            legend=False,
            ax=ax,
        )
        ax.set_yscale(scale)
        ax.set_xticks(range(3), labels)
        ax.set_xlabel("K-Means operating regime (ordered by TEY centroid)")
        ax.set_ylabel(f"{target} concentration")
        ax.set_title(
            f"{target} across operating regimes"
            f"{' (log scale)' if scale == 'log' else ''}"
        )
        ax.grid(axis="y", which="both", alpha=0.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Emissions by K-Means TEY regime — random 80% training partition",
        fontsize=15,
    )
    fig.tight_layout()
    save_figure(fig, "ppt03_regime_emissions.png")


def plot_multiyear_drift(df: pd.DataFrame) -> None:
    variables = ["CO", "NOX", "TEY", "AT"]
    colors = {
        "CO": VERMILLION,
        "NOX": BLUE,
        "TEY": GREEN,
        "AT": PURPLE,
    }
    years = np.sort(df["year"].unique())
    grouped = df.groupby("year")[variables]
    medians = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, variable in zip(axes.ravel(), variables, strict=True):
        ax.axvspan(2010.5, 2013.5, color=BLUE, alpha=0.09)
        ax.axvspan(2013.5, 2014.5, color=ORANGE, alpha=0.15)
        ax.axvspan(2014.5, 2015.5, color=VERMILLION, alpha=0.15)
        ax.fill_between(
            years,
            q25[variable].to_numpy(),
            q75[variable].to_numpy(),
            color=colors[variable],
            alpha=0.20,
            label="Yearly IQR",
        )
        ax.plot(
            years,
            medians[variable].to_numpy(),
            marker="o",
            markersize=7,
            linewidth=2.4,
            color=colors[variable],
            label="Yearly median",
        )
        for year in years:
            ax.axvline(year, color="white", linewidth=0.8, alpha=0.5)
        ax.set_xlim(2010.5, 2015.5)
        ax.set_xticks(years)
        ax.set_xlabel("Year")
        ax.set_ylabel(variable)
        ax.set_title(f"{variable}: yearly median and IQR")
        ax.grid(axis="y", alpha=0.5)
        ax.grid(axis="x", visible=False)

    zone_handles = [
        Patch(facecolor=BLUE, alpha=0.14, label="Train 2011–2013"),
        Patch(facecolor=ORANGE, alpha=0.25, label="Validation 2014"),
        Patch(facecolor=VERMILLION, alpha=0.25, label="Locked test 2015"),
        Line2D([], [], color=GREY, marker="o", label="Yearly median"),
    ]
    fig.suptitle(
        "Multi-year drift in emissions and operating conditions (2011–2015)",
        fontsize=15,
    )
    fig.legend(
        handles=zone_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "Colored ribbons show the yearly interquartile range (Q25–Q75).",
        ha="center",
        va="bottom",
        fontsize=10,
        color=GREY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.89))
    save_figure(fig, "ppt04_multiyear_drift_2011_2015.png")


def chronological_frames(df: pd.DataFrame):
    years = df["year"].to_numpy()
    train_mask, validation_mask, test_mask = chronological_masks(years)
    final_train = df.iloc[np.flatnonzero(train_mask | validation_mask)].copy()
    test = df.iloc[np.flatnonzero(test_mask)].copy()
    return final_train, test


def plot_cv_vs_2015(df: pd.DataFrame) -> None:
    table04 = pd.read_csv(RESULTS_DIR / "table04_regression_5fold_cv.csv")
    table05 = pd.read_csv(RESULTS_DIR / "table05_regression_chronological_2015.csv")
    final_train, test = chronological_frames(df)

    fig = plt.figure(figsize=(19, 15))
    grid = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 1.35], hspace=0.55, wspace=0.22
    )
    selected = {}
    for target in TARGETS:
        random_row = table04[table04["target"] == target].sort_values(
            "cv_rmse_mean"
        ).iloc[0]
        chrono_row = table05[table05["target"] == target].sort_values(
            "val_2014_rmse"
        ).iloc[0]
        selected[target] = (random_row["model"], chrono_row["model"])

    def metric_panel(ax, target, metric, ylabel, title):
        random_values = (
            table04[table04["target"] == target]
            .set_index("model")
            .loc[MODEL_ORDER, metric]
            .to_numpy(dtype=float)
        )
        chrono_values = (
            table05[table05["target"] == target]
            .set_index("model")
            .loc[MODEL_ORDER, metric]
            .to_numpy(dtype=float)
        )
        x_positions = np.arange(len(MODEL_ORDER))
        width = 0.37
        random_bars = ax.bar(
            x_positions - width / 2,
            random_values,
            width,
            color=PROTOCOL_COLORS["random_5fold_holdout"],
            label="Random holdout",
        )
        chrono_bars = ax.bar(
            x_positions + width / 2,
            chrono_values,
            width,
            color=PROTOCOL_COLORS["chronological_2015"],
            label="Locked 2015 test",
        )
        for model, bars in (
            (selected[target][0], random_bars),
            (selected[target][1], chrono_bars),
        ):
            bars[MODEL_ORDER.index(model)].set_edgecolor("black")
            bars[MODEL_ORDER.index(model)].set_linewidth(2.2)
        ax.set_xticks(
            x_positions,
            MODEL_LABELS,
            rotation=20,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.5)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=GREY, linewidth=0.8)
        ax.legend(loc="best")

    metric_panel(
        fig.add_subplot(grid[0, 0]),
        "CO",
        "test_rmse",
        "RMSE (lower is better)",
        "CO: test RMSE across models",
    )
    metric_panel(
        fig.add_subplot(grid[0, 1]),
        "CO",
        "test_r2",
        "R² (higher is better)",
        "CO: test R² across models",
    )
    metric_panel(
        fig.add_subplot(grid[1, 0]),
        "NOX",
        "test_rmse",
        "RMSE (lower is better)",
        "NOX: test RMSE across models",
    )
    metric_panel(
        fig.add_subplot(grid[1, 1]),
        "NOX",
        "test_r2",
        "R² (higher is better)",
        "NOX: test R² across models",
    )

    for column, target in enumerate(TARGETS):
        ax = fig.add_subplot(grid[2, column])
        row = table05[table05["target"] == target].sort_values(
            "val_2014_rmse"
        ).iloc[0]
        params = json.loads(row["best_params"])
        pipeline = regression_utils.build_pipeline(row["model"], params)
        pipeline.fit(
            final_train[SENSORS].to_numpy(dtype=float),
            final_train[target].to_numpy(dtype=float),
        )
        prediction = pipeline.predict(test[SENSORS].to_numpy(dtype=float))
        actual = test[target].to_numpy(dtype=float)
        hexbin = ax.hexbin(
            actual,
            prediction,
            gridsize=48,
            cmap="viridis",
            bins="log",
            mincnt=1,
            linewidths=0,
        )
        low = float(min(actual.min(), prediction.min()))
        high = float(max(actual.max(), prediction.max()))
        ax.plot([low, high], [low, high], "k--", linewidth=1.3)
        threshold = float(np.quantile(final_train[target], 0.95))
        ax.axvline(threshold, color=VERMILLION, linestyle=":", linewidth=1.5)
        ax.axhline(threshold, color=VERMILLION, linestyle=":", linewidth=1.5)
        true_positive = int(np.sum((actual > threshold) & (prediction > threshold)))
        false_negative = int(np.sum((actual > threshold) & (prediction <= threshold)))
        false_positive = int(np.sum((actual <= threshold) & (prediction > threshold)))
        true_negative = int(np.sum((actual <= threshold) & (prediction <= threshold)))
        corner_labels = [
            (0.02, 0.96, f"FP\n{false_positive:,}", "left", "top"),
            (0.98, 0.96, f"TP\n{true_positive:,}", "right", "top"),
            (0.02, 0.04, f"TN\n{true_negative:,}", "left", "bottom"),
            (0.98, 0.04, f"FN\n{false_negative:,}", "right", "bottom"),
        ]
        for x_text, y_text, text, horizontal, vertical in corner_labels:
            ax.text(
                x_text,
                y_text,
                text,
                transform=ax.transAxes,
                ha=horizontal,
                va=vertical,
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
                color=GREY,
            )
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(
            f"{target} 2015 parity — {row['model']}\n"
            f"R²={row['test_r2']:.3f}, RMSE={row['test_rmse']:.3f}, "
            f"Q95 τ={threshold:.2f}"
        )
        ax.grid(False)
        colorbar = fig.colorbar(hexbin, ax=ax, pad=0.01)
        colorbar.set_label("count (log scale)", fontsize=11)
        colorbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Random holdout vs locked 2015: model metrics and selected-model parity\n"
        "Black outline = model selected by training-only CV/2014 criterion; "
        "red dotted lines = training Q95",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "ppt05_cv_vs_2015_parity.png")


def plot_ablation_and_alerts() -> None:
    table06 = pd.read_csv(
        RESULTS_DIR / "table06_outlier_ablation_untrimmed_test.csv"
    )
    table07 = pd.read_csv(
        RESULTS_DIR / "table07_logistic_exceedance_metrics.csv"
    )
    protocols = ["random_5fold_holdout", "chronological_2015"]
    protocol_labels = {"random_5fold_holdout": "Random", "chronological_2015": "2015"}
    groups = [(protocol, target) for protocol in protocols for target in TARGETS]
    group_labels = [
        f"{protocol_labels[protocol]}\n{target}" for protocol, target in groups
    ]

    fig = plt.figure(figsize=(22, 12))
    grid = fig.add_gridspec(
        2, 6, height_ratios=[1.0, 1.05], hspace=0.45, wspace=0.65
    )
    metric_specs = [
        ("test_rmse", "Overall RMSE", "Overall RMSE"),
        ("q95_rmse", "Q95 tail RMSE", "Q95 tail RMSE"),
        ("q99_rmse", "Q99 tail RMSE", "Q99 tail RMSE"),
    ]
    for column, (metric, title, ylabel) in enumerate(metric_specs):
        ax = fig.add_subplot(grid[0, column * 2 : column * 2 + 2])
        x_positions = np.arange(len(groups))
        width = 0.26
        for offset, variant in enumerate(["Raw", "IQR", "3-sigma"]):
            values = []
            for protocol, target in groups:
                subset = table06[
                    (table06["protocol"] == protocol)
                    & (table06["target"] == target)
                    & (table06["variant"] == variant)
                ]
                values.append(float(subset[metric].iloc[0]))
            ax.bar(
                x_positions + (offset - 1) * width,
                values,
                width,
                color=VARIANT_COLORS[variant],
                label=variant,
            )
        ax.set_xticks(x_positions, group_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.5)
        ax.grid(axis="x", visible=False)
        if column == 0:
            ax.legend(title="Training rule", loc="upper left")

    alert = table07[table07["protocol"] == "chronological_2015"].copy()
    alert["run"] = alert["target"] + " " + alert["quantile"]
    alert_labels = [
        f"{row['run']}\nAUROC {row['test_roc_auc']:.3f}"
        for _, row in alert.iterrows()
    ]
    ax_metrics = fig.add_subplot(grid[1, 0:3])
    x_positions = np.arange(len(alert))
    width = 0.25
    metric_colors = {"test_precision": BLUE, "test_recall": ORANGE, "test_f1": GREEN}
    for offset, (metric, label) in enumerate(
        [
            ("test_precision", "Precision"),
            ("test_recall", "Recall"),
            ("test_f1", "F1"),
        ]
    ):
        ax_metrics.bar(
            x_positions + (offset - 1) * width,
            alert[metric].to_numpy(dtype=float),
            width,
            color=metric_colors[metric],
            label=label,
        )
    ax_metrics.set_xticks(x_positions, alert_labels)
    ax_metrics.set_ylim(0, 1.08)
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_title("Locked 2015 Q90/Q95 alert performance")
    ax_metrics.grid(axis="y", alpha=0.5)
    ax_metrics.grid(axis="x", visible=False)
    ax_metrics.legend(loc="upper right", ncol=3)

    ax_table = fig.add_subplot(grid[1, 3:6])
    ax_table.axis("off")
    cell_text = []
    for _, row in alert.iterrows():
        cell_text.append(
            [
                row["run"],
                f"{row['threshold']:.2f}",
                f"{row['train_prevalence'] * 100:.1f}%",
                f"{row['test_prevalence'] * 100:.1f}%",
                f"{int(row['tn']):,}",
                f"{int(row['fp']):,}",
                f"{int(row['fn']):,}",
                f"{int(row['tp']):,}",
            ]
        )
    table = ax_table.table(
        cellText=cell_text,
        colLabels=[
            "Alert",
            "τ",
            "Train prev.",
            "Test prev.",
            "TN",
            "FP",
            "FN",
            "TP",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    ax_table.set_title("Confusion breakdown — locked 2015 (positive = exceedance)")
    ax_table.text(
        0.5,
        -0.05,
        "Alert probability fixed at 0.5; thresholds and prevalence are training-derived.",
        transform=ax_table.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=GREY,
    )

    fig.suptitle(
        "Training-only outlier ablation on untrimmed tests, paired with "
        "Q90/Q95 alert trade-offs",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "ppt06_ablation_and_alerts.png")


def main() -> None:
    ensure_directories()
    PPT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    df = load_data()

    plot_vif()
    plot_mle_survival(df)
    plot_regimes(df)
    plot_multiyear_drift(df)
    plot_cv_vs_2015(df)
    plot_ablation_and_alerts()

    expected = sorted(path.name for path in PPT_DIR.glob("ppt0*.png"))
    print(f"PPT figures: {len(expected)}")
    for name in expected:
        size = (PPT_DIR / name).stat().st_size
        print(f"  {name}: {size:,} bytes")


if __name__ == "__main__":
    main()
