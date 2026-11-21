"""Shared constants and helpers for the CHE F315 gas-turbine PEMS pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RANDOM_STATE = 42

SENSORS = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
TARGETS = ["CO", "NOX"]
ANALYSIS = SENSORS + TARGETS

COLUMNS = ["year"] + ANALYSIS

EXPECTED_YEAR_COUNTS = {
    2011: 7411,
    2012: 7628,
    2013: 7152,
    2014: 7158,
    2015: 7384,
}
EXPECTED_ROWS = 36733

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "gas_turbine_full.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "gas_turbine_modeling.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def ensure_directories() -> None:
    """Create the output directories used by every stage."""
    for directory in (PROCESSED_PATH.parent, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def random_split_indices(n_rows: int):
    """Return the canonical random 80/20 row-index split."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(n_rows)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return train_idx, test_idx


def chronological_masks(years):
    """Boolean masks for the locked 2011-2013 / 2014 / 2015 chronology."""
    years = np.asarray(years)
    train = years <= 2013
    validation = years == 2014
    test = years == 2015
    return train, validation, test


def regression_metrics(y_true, y_pred) -> dict:
    """MSE, RMSE and R2 with the formulae fixed by the specification."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residuals**2) / denominator)
    return {"mse": mse, "rmse": rmse, "r2": r2}


def tail_metrics(y_true, y_pred, thresholds) -> dict:
    """RMSE and count above each train-derived quantile threshold."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = {}
    for q, tau in thresholds.items():
        mask = y_true > tau
        n_tail = int(np.sum(mask))
        if n_tail == 0:
            rmse = float("nan")
        else:
            rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
        result[f"{q}_rmse"] = rmse
        result[f"{q}_n"] = n_tail
    return result


def train_quantile_thresholds(y_train) -> dict:
    """Linear-interpolation Q95/Q99 thresholds fitted on training targets."""
    y_train = np.asarray(y_train, dtype=float)
    return {
        "q95": float(np.quantile(y_train, 0.95)),
        "q99": float(np.quantile(y_train, 0.99)),
    }


def json_params(params: dict) -> str:
    """Serialize hyperparameters as a sorted-key JSON object."""

    def _convert(value):
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    clean = {str(k): _convert(v) for k, v in params.items()}
    return json.dumps(clean, sort_keys=True)
