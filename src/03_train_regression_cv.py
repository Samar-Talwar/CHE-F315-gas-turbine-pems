"""Stage 03: syllabus regression models under CV and locked chronology."""

from __future__ import annotations

import warnings
from importlib import import_module

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from pems_common import (
    FIGURES_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SENSORS,
    TARGETS,
    chronological_masks,
    ensure_directories,
    json_params,
    random_split_indices,
    regression_metrics,
    tail_metrics,
    train_quantile_thresholds,
)

load_data = import_module("00_load_data").load_data

warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODEL_ORDER = ["OLS", "Ridge", "Lasso", "PCR", "Poly2_Ridge", "MLP"]
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
MLP_GRID = [
    (hidden, alpha)
    for hidden in [(32,), (64, 32)]
    for alpha in (1e-4, 1e-3)
]
VARIANTS = ["Raw", "IQR", "3-sigma"]

TABLE04_COLUMNS = [
    "protocol",
    "target",
    "model",
    "best_params",
    "cv_rmse_mean",
    "cv_rmse_std",
    "cv_r2_mean",
    "n_train",
    "n_test",
    "test_r2",
    "test_rmse",
    "test_mse",
    "q95_rmse",
    "q95_n",
    "q99_rmse",
    "q99_n",
]
TABLE05_COLUMNS = [
    "protocol",
    "target",
    "model",
    "best_params",
    "val_2014_r2",
    "val_2014_rmse",
    "n_initial_train",
    "n_val",
    "n_train",
    "n_test",
    "test_r2",
    "test_rmse",
    "test_mse",
    "q95_rmse",
    "q95_n",
    "q99_rmse",
    "q99_n",
]
TABLE06_COLUMNS = [
    "protocol",
    "target",
    "model",
    "variant",
    "n_train_original",
    "n_train_retained",
    "n_test",
    "test_r2",
    "test_rmse",
    "test_mse",
    "q95_rmse",
    "q95_n",
    "q99_rmse",
    "q99_n",
]
TABLE_LOG_COLUMNS = [
    "protocol",
    "target",
    "model",
    "best_params",
    "selection_space",
    "selection_rmse",
    "n_train",
    "n_test",
    "test_r2",
    "test_rmse",
    "test_mse",
    "q95_rmse",
    "q95_n",
    "q99_rmse",
    "q99_n",
]


def model_configs(model: str) -> list[dict]:
    if model == "OLS":
        return [{}]
    if model in {"Ridge", "Lasso", "Poly2_Ridge"}:
        return [{"alpha": alpha} for alpha in ALPHAS]
    if model == "PCR":
        return [{"n_components": n} for n in range(2, 10)]
    if model == "MLP":
        return [
            {"hidden_layer_sizes": hidden, "alpha": alpha}
            for hidden, alpha in MLP_GRID
        ]
    raise ValueError(f"Unknown model: {model}")


def build_pipeline(model: str, params: dict) -> Pipeline:
    if model == "OLS":
        return Pipeline([("model", LinearRegression())])
    if model == "Ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=params["alpha"], max_iter=20000)),
            ]
        )
    if model == "Lasso":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    Lasso(
                        alpha=params["alpha"],
                        max_iter=20000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if model == "PCR":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=params["n_components"],
                        random_state=RANDOM_STATE,
                    ),
                ),
                ("model", LinearRegression()),
            ]
        )
    if model == "Poly2_Ridge":
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=params["alpha"], max_iter=20000)),
            ]
        )
    if model == "MLP":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=params["hidden_layer_sizes"],
                        alpha=params["alpha"],
                        early_stopping=True,
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model}")


def _argmin_strict(values) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] < values[best]:
            best = index
    return best


def cv_select(model: str, X, y) -> dict:
    """Five-fold selection on a training partition only."""
    folds = list(
        KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X)
    )
    candidates = []
    for params in model_configs(model):
        fold_rmse = []
        fold_r2 = []
        for fit_index, score_index in folds:
            pipeline = build_pipeline(model, params)
            pipeline.fit(X[fit_index], y[fit_index])
            prediction = pipeline.predict(X[score_index])
            metrics = regression_metrics(y[score_index], prediction)
            fold_rmse.append(metrics["rmse"])
            fold_r2.append(metrics["r2"])
        candidates.append(
            {
                "params": params,
                "cv_rmse_mean": float(np.mean(fold_rmse)),
                "cv_rmse_std": float(np.std(fold_rmse, ddof=1)),
                "cv_r2_mean": float(np.mean(fold_r2)),
            }
        )
    best_index = _argmin_strict(
        [candidate["cv_rmse_mean"] for candidate in candidates]
    )
    return candidates[best_index]


def validation_select(model: str, X_train, y_train, X_val, y_val) -> dict:
    """Select by 2014 validation RMSE using 2011-2013 fits only."""
    candidates = []
    for params in model_configs(model):
        pipeline = build_pipeline(model, params)
        pipeline.fit(X_train, y_train)
        metrics = regression_metrics(y_val, pipeline.predict(X_val))
        candidates.append(
            {
                "params": params,
                "val_2014_rmse": metrics["rmse"],
                "val_2014_r2": metrics["r2"],
            }
        )
    best_index = _argmin_strict(
        [candidate["val_2014_rmse"] for candidate in candidates]
    )
    return candidates[best_index]


def fit_and_score(model: str, params: dict, X_train, y_train, X_test, y_test):
    pipeline = build_pipeline(model, params)
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, prediction)
    metrics.update(tail_metrics(y_test, prediction, train_quantile_thresholds(y_train)))
    return pipeline, prediction, metrics


def trimming_mask(frame: pd.DataFrame, target: str, variant: str) -> np.ndarray:
    """Training-only IQR or 3-sigma mask over sensors plus the modeled target."""
    mask = np.ones(len(frame), dtype=bool)
    columns = SENSORS + [target]
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if variant == "IQR":
            q25, q75 = np.quantile(values, [0.25, 0.75])
            iqr = q75 - q25
            low = q25 - 1.5 * iqr
            high = q75 + 1.5 * iqr
        elif variant == "3-sigma":
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            low = mean - 3.0 * std
            high = mean + 3.0 * std
        else:
            raise ValueError(f"Unknown variant: {variant}")
        mask &= (values >= low) & (values <= high)
    return mask


def run_random_protocol(df: pd.DataFrame):
    train_idx, test_idx = random_split_indices(len(df))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    rows04 = []
    records = {}
    predictions = {}
    for target in TARGETS:
        X_train = train[SENSORS].to_numpy(dtype=float)
        y_train = train[target].to_numpy(dtype=float)
        X_test = test[SENSORS].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)
        for model in MODEL_ORDER:
            selected = cv_select(model, X_train, y_train)
            params = selected["params"]
            pipeline, prediction, metrics = fit_and_score(
                model, params, X_train, y_train, X_test, y_test
            )
            row = {
                "protocol": "random_5fold_holdout",
                "target": target,
                "model": model,
                "best_params": json_params(params),
                "cv_rmse_mean": selected["cv_rmse_mean"],
                "cv_rmse_std": selected["cv_rmse_std"],
                "cv_r2_mean": selected["cv_r2_mean"],
                "n_train": len(train),
                "n_test": len(test),
            }
            row.update(
                {
                    "test_r2": metrics["r2"],
                    "test_rmse": metrics["rmse"],
                    "test_mse": metrics["mse"],
                    "q95_rmse": metrics["q95_rmse"],
                    "q95_n": metrics["q95_n"],
                    "q99_rmse": metrics["q99_rmse"],
                    "q99_n": metrics["q99_n"],
                }
            )
            rows04.append(row)
            records[(target, model)] = {
                "params": params,
                "selection_metric": "cv_rmse_mean",
                "selection_value": selected["cv_rmse_mean"],
            }
            predictions[(target, model)] = {
                "pipeline": pipeline,
                "prediction": prediction,
            }
            print(
                f"random {target} {model}: cv_rmse="
                f"{selected['cv_rmse_mean']:.5f} test_rmse={metrics['rmse']:.5f}"
            )
    return train, test, pd.DataFrame(rows04, columns=TABLE04_COLUMNS), (
        records,
        predictions,
    )


def run_chronological_protocol(df: pd.DataFrame):
    years = df["year"].to_numpy()
    train_mask, validation_mask, test_mask = chronological_masks(years)
    initial = df.iloc[np.flatnonzero(train_mask)].copy()
    validation = df.iloc[np.flatnonzero(validation_mask)].copy()
    final_train = df.iloc[np.flatnonzero(train_mask | validation_mask)].copy()
    test = df.iloc[np.flatnonzero(test_mask)].copy()
    rows05 = []
    records = {}
    predictions = {}
    for target in TARGETS:
        X_initial = initial[SENSORS].to_numpy(dtype=float)
        y_initial = initial[target].to_numpy(dtype=float)
        X_validation = validation[SENSORS].to_numpy(dtype=float)
        y_validation = validation[target].to_numpy(dtype=float)
        X_final = final_train[SENSORS].to_numpy(dtype=float)
        y_final = final_train[target].to_numpy(dtype=float)
        X_test = test[SENSORS].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)
        for model in MODEL_ORDER:
            selected = validation_select(
                model, X_initial, y_initial, X_validation, y_validation
            )
            params = selected["params"]
            pipeline, prediction, metrics = fit_and_score(
                model, params, X_final, y_final, X_test, y_test
            )
            row = {
                "protocol": "chronological_2015",
                "target": target,
                "model": model,
                "best_params": json_params(params),
                "val_2014_r2": selected["val_2014_r2"],
                "val_2014_rmse": selected["val_2014_rmse"],
                "n_initial_train": len(initial),
                "n_val": len(validation),
                "n_train": len(final_train),
                "n_test": len(test),
            }
            row.update(
                {
                    "test_r2": metrics["r2"],
                    "test_rmse": metrics["rmse"],
                    "test_mse": metrics["mse"],
                    "q95_rmse": metrics["q95_rmse"],
                    "q95_n": metrics["q95_n"],
                    "q99_rmse": metrics["q99_rmse"],
                    "q99_n": metrics["q99_n"],
                }
            )
            rows05.append(row)
            records[(target, model)] = {
                "params": params,
                "selection_metric": "val_2014_rmse",
                "selection_value": selected["val_2014_rmse"],
            }
            predictions[(target, model)] = {
                "pipeline": pipeline,
                "prediction": prediction,
            }
            print(
                f"chronological {target} {model}: val_rmse="
                f"{selected['val_2014_rmse']:.5f} "
                f"test_rmse={metrics['rmse']:.5f}"
            )
    return (
        initial,
        validation,
        final_train,
        test,
        pd.DataFrame(rows05, columns=TABLE05_COLUMNS),
        (records, predictions),
    )


def run_ablation(
    protocol: str,
    target: str,
    train_part: pd.DataFrame,
    test_part: pd.DataFrame,
    best_model: str,
    params: dict,
) -> tuple[list[dict], dict]:
    """Refit the frozen Raw model under Raw/IQR/3-sigma training variants."""
    X_test = test_part[SENSORS].to_numpy(dtype=float)
    y_test = test_part[target].to_numpy(dtype=float)
    y_train_raw = train_part[target].to_numpy(dtype=float)
    thresholds = train_quantile_thresholds(y_train_raw)
    rows = []
    raw_result = {}
    for variant in VARIANTS:
        if variant == "Raw":
            retained = train_part
        else:
            mask = trimming_mask(train_part, target, variant)
            retained = train_part.loc[mask]
        pipeline = build_pipeline(best_model, params)
        pipeline.fit(
            retained[SENSORS].to_numpy(dtype=float),
            retained[target].to_numpy(dtype=float),
        )
        prediction = pipeline.predict(X_test)
        metrics = regression_metrics(y_test, prediction)
        metrics.update(tail_metrics(y_test, prediction, thresholds))
        row = {
            "protocol": protocol,
            "target": target,
            "model": best_model,
            "variant": variant,
            "n_train_original": len(train_part),
            "n_train_retained": len(retained),
            "n_test": len(test_part),
            "test_r2": metrics["r2"],
            "test_rmse": metrics["rmse"],
            "test_mse": metrics["mse"],
            "q95_rmse": metrics["q95_rmse"],
            "q95_n": metrics["q95_n"],
            "q99_rmse": metrics["q99_rmse"],
            "q99_n": metrics["q99_n"],
        }
        rows.append(row)
        if variant == "Raw":
            raw_result = {
                "y_test": y_test,
                "prediction": prediction,
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
            }
    return rows, raw_result


def select_reference(records: dict, target: str) -> tuple[str, dict]:
    models = [model for model in MODEL_ORDER if (target, model) in records]
    values = [records[(target, model)]["selection_value"] for model in models]
    best_index = _argmin_strict(values)
    return models[best_index], records[(target, models[best_index])]


def log1p_variant(df: pd.DataFrame, protocol: str) -> dict:
    """Supplementary feed-forward MLP trained on log1p(CO)."""
    years = df["year"].to_numpy()
    if protocol == "random_5fold_holdout":
        train_idx, test_idx = random_split_indices(len(df))
        initial = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        selection_space = "cv_rmse_mean"
        selected = cv_select(
            "MLP",
            initial[SENSORS].to_numpy(dtype=float),
            np.log1p(initial["CO"].to_numpy(dtype=float)),
        )
        params = selected["params"]
        selection_rmse = selected["cv_rmse_mean"]
        final_train = initial
        n_train = len(initial)
    else:
        train_mask, validation_mask, test_mask = chronological_masks(years)
        initial = df.iloc[np.flatnonzero(train_mask)].copy()
        validation = df.iloc[np.flatnonzero(validation_mask)].copy()
        test = df.iloc[np.flatnonzero(test_mask)].copy()
        initial_log = np.log1p(initial["CO"].to_numpy(dtype=float))
        validation_log = np.log1p(validation["CO"].to_numpy(dtype=float))
        selected = validation_select(
            "MLP",
            initial[SENSORS].to_numpy(dtype=float),
            initial_log,
            validation[SENSORS].to_numpy(dtype=float),
            validation_log,
        )
        params = selected["params"]
        selection_space = "val_2014_rmse"
        selection_rmse = selected["val_2014_rmse"]
        final_train = df.iloc[np.flatnonzero(train_mask | validation_mask)].copy()
        n_train = len(final_train)
    pipeline = build_pipeline("MLP", params)
    pipeline.fit(
        final_train[SENSORS].to_numpy(dtype=float),
        np.log1p(final_train["CO"].to_numpy(dtype=float)),
    )
    y_test = test["CO"].to_numpy(dtype=float)
    prediction = np.expm1(
        pipeline.predict(test[SENSORS].to_numpy(dtype=float))
    )
    metrics = regression_metrics(y_test, prediction)
    metrics.update(
        tail_metrics(
            y_test,
            prediction,
            train_quantile_thresholds(final_train["CO"].to_numpy(dtype=float)),
        )
    )
    return {
        "protocol": protocol,
        "target": "CO",
        "model": "MLP_log1p",
        "best_params": json_params(params),
        "selection_space": selection_space,
        "selection_rmse": float(selection_rmse),
        "n_train": int(n_train),
        "n_test": len(test),
        "test_r2": metrics["r2"],
        "test_rmse": metrics["rmse"],
        "test_mse": metrics["mse"],
        "q95_rmse": metrics["q95_rmse"],
        "q95_n": metrics["q95_n"],
        "q99_rmse": metrics["q99_rmse"],
        "q99_n": metrics["q99_n"],
    }


def plot_parity(parity_data: dict) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    panel_map = [
        (0, 0, "random_5fold_holdout", "CO"),
        (0, 1, "random_5fold_holdout", "NOX"),
        (1, 0, "chronological_2015", "CO"),
        (1, 1, "chronological_2015", "NOX"),
    ]
    for row, column, protocol, target in panel_map:
        ax = axes[row, column]
        result = parity_data[(protocol, target)]
        y_true = result["y_test"]
        y_pred = result["prediction"]
        ax.scatter(y_true, y_pred, s=7, alpha=0.28, color="#4c72b0")
        low = float(min(np.min(y_true), np.min(y_pred)))
        high = float(max(np.max(y_true), np.max(y_pred)))
        ax.plot([low, high], [low, high], "k--", lw=1.2, label="identity")
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.set_title(
            f"{protocol} — {target}   "
            f"R²={result['r2']:.3f}, RMSE={result['rmse']:.3f}"
        )
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Preselected model parity: random holdout vs locked 2015 test",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig06_parity_cv_vs_chronological.png", dpi=150)
    plt.close(fig)


def plot_ablation(table06: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    protocols = ["random_5fold_holdout", "chronological_2015"]
    metrics = ["test_rmse", "q95_rmse", "q99_rmse"]
    metric_labels = ["Overall", "Q95", "Q99"]
    colors = {"Raw": "#4c72b0", "IQR": "#dd8452", "3-sigma": "#55a868"}
    positions = np.arange(len(metrics))
    width = 0.25
    for row, protocol in enumerate(protocols):
        for column, target in enumerate(TARGETS):
            ax = axes[row, column]
            subset = table06[
                (table06["protocol"] == protocol) & (table06["target"] == target)
            ]
            for offset, variant in enumerate(VARIANTS):
                values = (
                    subset[subset["variant"] == variant][metrics]
                    .iloc[0]
                    .to_numpy(dtype=float)
                )
                ax.bar(
                    positions + (offset - 1) * width,
                    values,
                    width,
                    color=colors[variant],
                    label=variant,
                )
            ax.set_xticks(positions)
            ax.set_xticklabels(metric_labels)
            ax.set_ylabel("RMSE on untrimmed test rows")
            ax.set_title(f"{protocol} — {target}")
            if row == 0 and column == 0:
                ax.legend(title="training variant", fontsize=8)
    fig.suptitle(
        "Training-only outlier ablation evaluated on identical untrimmed test rows",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig07_outlier_ablation_tails.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    df = load_data()

    (
        random_train,
        random_test,
        table04,
        (random_records, _),
    ) = run_random_protocol(df)
    (
        _,
        _,
        final_train,
        chronological_test,
        table05,
        (chronological_records, _),
    ) = run_chronological_protocol(df)

    table04.to_csv(RESULTS_DIR / "table04_regression_5fold_cv.csv", index=False)
    table05.to_csv(
        RESULTS_DIR / "table05_regression_chronological_2015.csv", index=False
    )

    ablation_rows = []
    parity_data = {}
    ablation_inputs = [
        (
            "random_5fold_holdout",
            random_train,
            random_test,
            random_records,
        ),
        (
            "chronological_2015",
            final_train,
            chronological_test,
            chronological_records,
        ),
    ]
    for protocol, train_part, test_part, records in ablation_inputs:
        for target in TARGETS:
            best_model, selected = select_reference(records, target)
            rows, raw_result = run_ablation(
                protocol,
                target,
                train_part,
                test_part,
                best_model,
                selected["params"],
            )
            ablation_rows.extend(rows)
            parity_data[(protocol, target)] = raw_result
            print(f"ablation {protocol} {target}: reference model={best_model}")

    table06 = pd.DataFrame(ablation_rows, columns=TABLE06_COLUMNS)
    table06.to_csv(
        RESULTS_DIR / "table06_outlier_ablation_untrimmed_test.csv", index=False
    )

    log_rows = [
        log1p_variant(df, "random_5fold_holdout"),
        log1p_variant(df, "chronological_2015"),
    ]
    pd.DataFrame(log_rows, columns=TABLE_LOG_COLUMNS).to_csv(
        RESULTS_DIR / "table04b_log1p_co_variant.csv", index=False
    )

    plot_parity(parity_data)
    plot_ablation(table06)

    print()
    print(table04.to_string(index=False))
    print()
    print(table05.to_string(index=False))
    print()
    print(table06.to_string(index=False))


if __name__ == "__main__":
    main()
