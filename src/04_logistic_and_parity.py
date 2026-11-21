"""Stage 04: Q90/Q95 exceedance alerts with logistic regression."""

from __future__ import annotations

from importlib import import_module

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
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

QUANTILES = [90, 95]
TABLE07_COLUMNS = [
    "protocol",
    "target",
    "model",
    "quantile",
    "threshold",
    "n_train",
    "n_test",
    "train_prevalence",
    "test_prevalence",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "test_pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
]


def build_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_run(
    protocol: str,
    target: str,
    quantile: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict, dict]:
    y_train = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    threshold = float(np.quantile(y_train, quantile / 100.0))
    z_train = (y_train > threshold).astype(int)
    z_test = (y_test > threshold).astype(int)

    classifier = build_classifier()
    classifier.fit(
        train[SENSORS].to_numpy(dtype=float),
        z_train,
    )
    probability = classifier.predict_proba(
        test[SENSORS].to_numpy(dtype=float)
    )[:, 1]
    alert = (probability >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(z_test, alert, labels=[0, 1]).ravel()
    row = {
        "protocol": protocol,
        "target": target,
        "model": "LogisticRegression",
        "quantile": f"Q{quantile}",
        "threshold": threshold,
        "n_train": len(train),
        "n_test": len(test),
        "train_prevalence": float(np.mean(z_train)),
        "test_prevalence": float(np.mean(z_test)),
        "test_accuracy": float(accuracy_score(z_test, alert)),
        "test_precision": float(precision_score(z_test, alert, zero_division=0)),
        "test_recall": float(recall_score(z_test, alert, zero_division=0)),
        "test_f1": float(f1_score(z_test, alert, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(z_test, probability)),
        "test_pr_auc": float(average_precision_score(z_test, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    details = {
        "z_test": z_test,
        "probability": probability,
        "confusion": np.array([[tn, fp], [fn, tp]]),
        "threshold": threshold,
        "roc_auc": row["test_roc_auc"],
        "quantile": quantile,
    }
    return row, details


def plot_diagnostics(rows: list[dict], details: dict) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 4, figsize=(21, 19))
    protocols = ["random_5fold_holdout", "chronological_2015"]
    for row_index, protocol in enumerate(protocols):
        for column_index, target in enumerate(TARGETS):
            for q_offset, quantile in enumerate(QUANTILES):
                column = column_index * 2 + q_offset
                key = (protocol, target, quantile)
                detail = details[key]
                confusion_ax = axes[row_index, column]
                ConfusionMatrixDisplay(
                    confusion_matrix=detail["confusion"],
                    display_labels=["normal", "exceed"],
                ).plot(
                    ax=confusion_ax,
                    colorbar=False,
                    values_format="d",
                    cmap="Blues",
                )
                confusion_ax.set_title(
                    f"{protocol}\n{target} Q{quantile} τ={detail['threshold']:.3f} "
                    f"AUROC={detail['roc_auc']:.3f}",
                    fontsize=9,
                )

                roc_ax = axes[row_index + 2, column]
                fpr, tpr, _ = roc_curve(detail["z_test"], detail["probability"])
                roc_ax.plot(fpr, tpr, color="#4c72b0", lw=2)
                roc_ax.plot([0, 1], [0, 1], "k--", lw=1)
                roc_ax.set_xlabel("false positive rate")
                roc_ax.set_ylabel("true positive rate")
                roc_ax.set_title(
                    f"ROC {protocol}\n{target} Q{quantile} "
                    f"AUROC={detail['roc_auc']:.3f} τ={detail['threshold']:.3f}",
                    fontsize=9,
                )
    fig.suptitle(
        "Q90/Q95 exceedance alerts: confusion matrices and ROC curves "
        "(threshold fixed at 0.5)",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig08_logistic_confusion_roc.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    df = load_data()
    years = df["year"].to_numpy()
    train_mask, validation_mask, test_mask = chronological_masks(years)

    train_idx, test_idx = random_split_indices(len(df))
    partitions = {
        "random_5fold_holdout": (
            df.iloc[train_idx].copy(),
            df.iloc[test_idx].copy(),
        ),
        "chronological_2015": (
            df.iloc[np.flatnonzero(train_mask | validation_mask)].copy(),
            df.iloc[np.flatnonzero(test_mask)].copy(),
        ),
    }

    rows = []
    details = {}
    for protocol in ("random_5fold_holdout", "chronological_2015"):
        train, test = partitions[protocol]
        for target in TARGETS:
            for quantile in QUANTILES:
                row, detail = evaluate_run(
                    protocol, target, quantile, train, test
                )
                rows.append(row)
                details[(protocol, target, quantile)] = detail
                print(
                    f"{protocol} {target} Q{quantile}: "
                    f"AUROC={row['test_roc_auc']:.3f} F1={row['test_f1']:.3f}"
                )

    table07 = pd.DataFrame(rows, columns=TABLE07_COLUMNS)
    table07.to_csv(
        RESULTS_DIR / "table07_logistic_exceedance_metrics.csv", index=False
    )
    plot_diagnostics(rows, details)
    print()
    print(table07.to_string(index=False))


if __name__ == "__main__":
    main()
