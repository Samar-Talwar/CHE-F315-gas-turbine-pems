# CHE F315 BITS Goa — Gas Turbine PEMS

## Verified data contract

`data/raw/gas_turbine_full.csv` has **36,733 rows × 12 columns**, in this exact order:

`year, AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO, NOX`

The `year` column exists. Its counts are 2011=7,411; 2012=7,628; 2013=7,152; 2014=7,158; 2015=7,384. There are no missing values. Preserve row order and duplicate rows. `year` is split metadata, never a predictor.

## Non-negotiable rules

1. **Syllabus only.** Permitted algorithms are OLS/linear regression, L1 Lasso, L2 Ridge, distribution MLE with `scipy.stats`, PCA/PCR, K-Means, `sklearn.neural_network.MLPRegressor` (feed-forward ANN), and `LogisticRegression`. Supporting preprocessing, model selection, and metrics are allowed. No SVM or any other supervised/unsupervised learner.
2. **Static bans are absolute.** Never import or reference any module or symbol in `BANNED_MODULES` or `BANNED_SYMBOLS` in `tests/verify_rubric.py`. This includes aliases, dead code, and notebook cells; the AST check rejects names even when unused. Do not weaken or edit the rubric tests.
3. **Independent targets.** Use only `AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP` as source predictors. CO must never be in `X`, a feature union, or a preprocessing input for a NOX model; NOX must never be in `X`, a feature union, or a preprocessing input for a CO model. The same isolation applies to exceedance classifiers.
4. **Zero leakage.** Fit `StandardScaler`, `PolynomialFeatures`, PCA, K-Means, MLE parameters used for decisions, feature selection, imputation, outlier limits, and Q90/Q95/Q99 thresholds only on the relevant training fold/partition. Validation and test data may only be transformed or scored. Use fold-local pipelines during CV and recompute every fitted quantity after any refit.
5. **Untrimmed evaluation.** Raw/IQR/3-sigma trimming is a training-only ablation. Never remove emissions spikes—or any other row—from a validation or test set. Tail thresholds must come from training targets, never test targets.
6. **Locked chronology.** For temporal evaluation, tune on 2011–2013 train → 2014 validation; 2015 is a locked test set and must not influence preprocessing, feature/model selection, trimming, thresholds, or tuning. Any final 2011–2014 refit must follow the completed 2014 selection.
7. **Reproducibility.** Use seed 42, relative paths from the repository root, deterministic output schemas, and the exact artifacts and evaluation protocols in `PROJECT_SPEC.md`. Create `results/`, `figures/`, and `data/processed/` as needed.
8. **Notebook quality.** `notebooks/CHE_F315_Gas_Turbine_PEMS.ipynb` must contain at least seven AST-valid Python code cells covering the seven presentation components. Execute every cell in order; no unexecuted cells or error outputs.

## Required gates

Run both commands from the repository root after every stage and before finishing; fix regressions rather than bypassing checks:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```
