import ast
import json
from pathlib import Path

import pandas as pd

BANNED_MODULES = {
    "xgboost", "lightgbm", "catboost", "shap", "imblearn",
    "torch", "tensorflow", "keras"
}

BANNED_SYMBOLS = {
    "SVC", "SVR", "NuSVC", "NuSVR", "LinearSVC", "LinearSVR",
    "RandomForestRegressor", "RandomForestClassifier",
    "ExtraTreesRegressor", "ExtraTreesClassifier",
    "GradientBoostingRegressor", "GradientBoostingClassifier",
    "HistGradientBoostingRegressor", "HistGradientBoostingClassifier",
    "DecisionTreeRegressor", "DecisionTreeClassifier",
    "KNeighborsRegressor", "KNeighborsClassifier",
    "GaussianProcessRegressor", "GaussianProcessClassifier",
    "DBSCAN", "OPTICS", "HDBSCAN", "GaussianMixture", "BayesianGaussianMixture",
    "LSTM", "GRU", "Conv1d", "Conv2d"
}

REQUIRED_FIGURES = [
    "fig01_univariate_kde_boxplots.png",
    "fig02_correlation_heatmap_11x11.png",
    "fig03_mle_distribution_fits.png",
    "fig04_pca_scree_biplot.png",
    "fig05_kmeans_regimes.png",
    "fig06_parity_cv_vs_chronological.png",
    "fig07_outlier_ablation_tails.png",
    "fig08_logistic_confusion_roc.png",
]

REQUIRED_TABLES = [
    "table01_eda_outliers_vif.csv",
    "table02_mle_loglik.csv",
    "table03_kmeans_centroids.csv",
    "table04_regression_5fold_cv.csv",
    "table05_regression_chronological_2015.csv",
    "table06_outlier_ablation_untrimmed_test.csv",
    "table07_logistic_exceedance_metrics.csv",
]


def _scan_ast_for_banned(code_str: str, origin: str):
    tree = ast.parse(code_str, filename=origin)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in BANNED_MODULES, f"Banned module '{alias.name}' in {origin}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in BANNED_MODULES, f"Banned module '{node.module}' in {origin}"
            for alias in node.names:
                assert alias.name not in BANNED_SYMBOLS, f"Banned algorithm '{alias.name}' in {origin}"
        elif isinstance(node, ast.Name):
            assert node.id not in BANNED_SYMBOLS, f"Banned algorithm reference '{node.id}' in {origin}"


def test_strict_syllabus_compliance_in_src():
    src_files = sorted(Path("src").glob("*.py"))
    assert len(src_files) >= 5, "Expected modular scripts src/00_*.py through src/04_*.py"
    for py_file in src_files:
        _scan_ast_for_banned(py_file.read_text(encoding="utf-8"), str(py_file))


def test_required_outputs_and_metrics_integrity():
    for fname in REQUIRED_TABLES:
        fpath = Path("results") / fname
        assert fpath.exists() and fpath.stat().st_size > 50, f"Missing or empty table: {fpath}"
        df = pd.read_csv(fpath)
        assert len(df) >= 2, f"Too few rows in {fpath}"
        assert not df.isna().all().all(), f"All-NaN table in {fpath}"

    for fname in REQUIRED_FIGURES:
        fpath = Path("figures") / fname
        assert fpath.exists() and fpath.stat().st_size > 5000, f"Missing or corrupt figure: {fpath}"


def test_chronological_drift_and_model_sanity():
    cv_df = pd.read_csv("results/table04_regression_5fold_cv.csv")
    chr_df = pd.read_csv("results/table05_regression_chronological_2015.csv")
    for df in (cv_df, chr_df):
        for col in ("target", "model", "test_r2", "test_rmse", "test_mse", "q95_rmse", "q99_rmse"):
            assert col in df.columns, f"Missing required metric column '{col}'"
        assert set(df["target"].str.upper()) >= {"CO", "NOX"}, "Both CO and NOX targets must be evaluated"


def test_executed_notebook_valid_and_syllabus_clean():
    nb_path = Path("notebooks/CHE_F315_Gas_Turbine_PEMS.ipynb")
    assert nb_path.exists(), "Final submission notebook notebooks/CHE_F315_Gas_Turbine_PEMS.ipynb missing"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) >= 7, "Notebook must cover all 7 Prof. Riju De presentation components"
    for idx, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        _scan_ast_for_banned(src, f"notebook_cell_{idx}")
        assert cell.get("execution_count") is not None, f"Cell {idx} is unexecuted"
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", f"Execution error in notebook cell {idx}: {out}"
