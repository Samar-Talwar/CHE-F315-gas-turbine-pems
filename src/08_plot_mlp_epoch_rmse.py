"""Plot per-epoch training and validation RMSE for the selected MLP models.

The benchmark MLP uses StandardScaler -> (64, 32) ReLU MLP -> linear output,
Adam (learning_rate_init=0.001), with early_stopping=True and seed 42.  To expose
the curves, this script trains one epoch at a time with warm_start=True and applies
the same patience of 10 epochs:

* early-stop split = internal 10% of the training partition for both protocols;
* the 20% random holdout and 2014 validation are scored for the curves only;
* 2015 is never read.

Scalers are fitted on training rows only, and CO/NOx use separate runs.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from pems_common import (
    RANDOM_STATE,
    RESULTS_DIR,
    SENSORS,
    TARGETS,
    chronological_masks,
    random_split_indices,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_PATH = ROOT / "report" / "CHE_F315_MLP_Epoch_RMSE.png"
CSV_PATH = ROOT / "report" / "mlp_epoch_rmse.csv"
MAX_EPOCHS = 400
PATIENCE = 10
TOL = 1e-4
COLORS = {
    "random": "#0072B2",
    "chronological": "#D55E00",
}
LABELS = {
    "random": "Random 80/20",
    "chronological": "Chronological 2011–13 → 2014",
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(ROOT / "data" / "processed" / "gas_turbine_modeling.csv")
    assert data.shape == (36733, 12)
    assert data["year"].value_counts().sort_index().to_dict() == {
        2011: 7411,
        2012: 7628,
        2013: 7152,
        2014: 7158,
        2015: 7384,
    }
    return data


def selected_params(table_name: str, target: str) -> dict:
    table = pd.read_csv(RESULTS_DIR / table_name)
    row = table[(table["target"] == target) & (table["model"] == "MLP")].iloc[0]
    return json.loads(row["best_params"])


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_curve(
    protocol: str,
    target: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: dict,
) -> tuple[dict, list[dict]]:
    """Train one epoch at a time and log train/validation RMSE."""
    scaler = StandardScaler().fit(train[SENSORS])
    x_train = scaler.transform(train[SENSORS])
    y_train = train[target].to_numpy(dtype=float)
    x_validation = scaler.transform(validation[SENSORS])
    y_validation = validation[target].to_numpy(dtype=float)

    fit_index, monitor_index = train_test_split(
        np.arange(len(train)),
        test_size=0.1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    x_fit = x_train[fit_index]
    y_fit = y_train[fit_index]
    x_monitor = x_train[monitor_index]
    y_monitor = y_train[monitor_index]

    model = MLPRegressor(
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        activation="relu",
        solver="adam",
        alpha=params["alpha"],
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=1,
        shuffle=True,
        random_state=RANDOM_STATE,
        tol=TOL,
        early_stopping=False,
        n_iter_no_change=PATIENCE,
        warm_start=True,
    )

    records = []
    best_monitor = float("inf")
    best_validation = float("inf")
    best_epoch = 0
    wait = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.fit(x_fit, y_fit)
        train_rmse = rmse(y_fit, model.predict(x_fit))
        validation_rmse = rmse(y_validation, model.predict(x_validation))
        monitor_rmse = rmse(y_monitor, model.predict(x_monitor))
        records.append(
            {
                "protocol": protocol,
                "target": target,
                "epoch": epoch,
                "train_rmse": train_rmse,
                "validation_rmse": validation_rmse,
            }
        )
        if validation_rmse < best_validation:
            best_validation = validation_rmse
            best_epoch = epoch
        if monitor_rmse < best_monitor - TOL:
            best_monitor = monitor_rmse
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    return (
        {
            "protocol": protocol,
            "target": target,
            "records": pd.DataFrame(records),
            "best_epoch": best_epoch,
            "best_validation_rmse": best_validation,
            "epochs_run": len(records),
        },
        records,
    )


def main() -> None:
    data = load_data()
    report_dir = ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    train_idx, holdout_idx = random_split_indices(len(data))
    random_train = data.iloc[train_idx].copy()
    random_validation = data.iloc[holdout_idx].copy()

    years = data["year"].to_numpy()
    train_mask, validation_mask, _ = chronological_masks(years)
    chronological_train = data.iloc[np.flatnonzero(train_mask)].copy()
    chronological_validation = data.iloc[np.flatnonzero(validation_mask)].copy()

    runs = {}
    all_records = []
    for target in TARGETS:
        random_params = selected_params(
            "table04_regression_5fold_cv.csv", target
        )
        chronological_params = selected_params(
            "table05_regression_chronological_2015.csv", target
        )
        runs[("random", target)], records = run_curve(
            "random",
            target,
            random_train,
            random_validation,
            random_params,
        )
        all_records.extend(records)
        runs[("chronological", target)], records = run_curve(
            "chronological",
            target,
            chronological_train,
            chronological_validation,
            chronological_params,
        )
        all_records.extend(records)
        print(
            f"{target}: random best val {runs[('random', target)]['best_validation_rmse']:.4f} "
            f"at epoch {runs[('random', target)]['best_epoch']}; "
            f"chronological best val {runs[('chronological', target)]['best_validation_rmse']:.4f} "
            f"at epoch {runs[('chronological', target)]['best_epoch']}"
        )

    pd.DataFrame(all_records).to_csv(CSV_PATH, index=False)

    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for ax, target in zip(axes, TARGETS, strict=True):
        panel_runs = [
            runs[("random", target)],
            runs[("chronological", target)],
        ]
        panel_max = max(run["epochs_run"] for run in panel_runs)
        for protocol in ("random", "chronological"):
            run = runs[(protocol, target)]
            frame = run["records"]
            color = COLORS[protocol]
            ax.plot(
                frame["epoch"],
                frame["train_rmse"],
                color=color,
                linewidth=1.7,
                linestyle="-",
                label=f"{LABELS[protocol]} train",
            )
            ax.plot(
                frame["epoch"],
                frame["validation_rmse"],
                color=color,
                linewidth=2.3,
                linestyle="--",
                label=f"{LABELS[protocol]} validation",
            )
        ax.set_xlim(0, panel_max * 1.03)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE (emission units)")
        ax.set_title(f"{target} MLP learning curve")
        ax.legend(loc="upper right", frameon=True, fontsize=9.2)
    fig.suptitle(
        "Epoch vs. RMSE — selected MLP (9 → 64 → 32 → 1)",
        fontsize=16,
        y=0.985,
    )
    fig.text(
        0.5,
        0.93,
        "StandardScaler fit on training rows only · ReLU · Adam lr=0.001 · seed 42",
        ha="center",
        fontsize=9.2,
        color="#2B5B84",
    )
    fig.text(
        0.5,
        0.02,
        "Early stop: internal 10% of training for both protocols. "
        "Validation curves are logged only; 2015 is never used.",
        ha="center",
        fontsize=8.8,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.885))
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {FIG_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
