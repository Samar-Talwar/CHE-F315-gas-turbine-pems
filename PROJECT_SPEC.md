# CHE F315 Gas Turbine PEMS — Implementation Specification

This is the executable contract for the five `src/` stages and the final notebook. All paths are relative to the repository root; all fitted objects are training-only, targets are independent, and seed 42 is used throughout.

## 1. Canonical data and evaluation contract

- Input: `data/raw/gas_turbine_full.csv`, shape **(36733, 12)**. Exact columns/order: `year, AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO, NOX`.
- `SENSORS = [AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP]`; `TARGETS = [CO, NOX]`; `ANALYSIS = SENSORS + TARGETS` (the 11 heatmap/EDA variables). `year` is used only for splitting and reporting.
- Validate exact year counts: 2011=7411, 2012=7628, 2013=7152, 2014=7158, 2015=7384. If `year` is absent, add it without sorting, using positional half-open slices `[0:7411]`, `[7411:15039]`, `[15039:22191]`, `[22191:29349]`, `[29349:36733]`, respectively. Any other row count is an error. Do not drop duplicates or impute/drop rows.
- For target `y`, `X = df[SENSORS]`; the opposite target and `year` are forbidden predictors and preprocessing inputs.

| Protocol | Training/tuning | Final fit | Locked evaluation |
|---|---|---|---|
| `random_5fold_holdout` | Random 80% train: call `train_test_split` on row indices with `test_size=0.2`, `shuffle=True`, `random_state=42` (29,386 train / 7,347 test). Tune with `KFold(n_splits=5, shuffle=True, random_state=42)` on train only. | Best configuration on all 29,386 training rows | Once on all 7,347 untrimmed holdout rows |
| `chronological_2015` | 2011–2013: 22,191 train; 2014: 7,158 validation. Select by 2014 RMSE, with every transform fit on 2011–2013 only. | After selection, refit the complete pipeline on 2011–2014: 29,349 rows | Once on all 7,384 untrimmed 2015 rows |

EDA/MLE/PCA/K-Means display fits use only the random 80% training partition, never its holdout; these descriptive fits are not consumed by the chronological pipeline. Each protocol otherwise fits its own transforms. Within CV, every pipeline step and threshold is re-fitted in that fold.

For predictions `ŷ` and truth `y`, report `test_mse = n⁻¹Σ(y−ŷ)²`, `test_rmse = sqrt(test_mse)`, and `test_r2 = 1 − Σ(y−ŷ)²/Σ(y−ȳ)²`. For q in {90,95,99}, `τq = np.quantile(y_train, q/100)` (linear interpolation), and `qXX_rmse = sqrt(mean((y−ŷ)² | y > τq))` on test rows. Retain `qXX_n`; a zero-denominator metric is `NaN`, never zero. Logistic labels use the same train-only linear quantile and strict `>` rule.

## 2. Stage contracts

### `src/00_load_data.py`

Implement a reusable loader and `main()`. Read without reordering rows, add/validate `year` as above, coerce the 11 analysis columns to finite floats, assert the exact schema/shape/counts, and fail loudly on nulls or infinities. Write `data/processed/gas_turbine_modeling.csv` with the exact original 12-column order and all 36,733 rows; print shape and year counts. Later stages consume this loader/output rather than maintaining separate cleaning rules.

### `src/01_eda_outliers_mle.py`

On the random training partition:

- For each of the 11 analysis variables report count, mean, sample standard deviation (`ddof=1`), min, Q01/Q25/median/Q75/Q99, max, missing count, and outlier counts. IQR limits are `Q25−1.5·IQR`, `Q75+1.5·IQR`; 3-sigma limits are `μ±3s`. VIF is reported only for the nine sensors: regress each sensor on the other eight by OLS and use `VIF_j = 1/(1−R_j²)`.
- Write `results/table01_eda_outliers_vif.csv`, exactly one row per analysis variable, columns: `column,n,mean,std,min,q01,q25,median,q75,q99,max,missing,iqr_low,iqr_high,iqr_outliers,sigma3_low,sigma3_high,sigma3_outliers,vif` (VIF blank for CO/NOX).
- Produce `figures/fig01_univariate_kde_boxplots.png` (KDE and boxplot for every analysis variable) and `figures/fig02_correlation_heatmap_11x11.png` (Pearson matrix in exact `ANALYSIS` order, annotated and scaled to [-1,1]).
- Fit `scipy.stats.norm`, `gamma`, and `lognorm` independently to CO and NOX by MLE. Use `norm.fit(x)`; use `gamma.fit(x, floc=0)` and `lognorm.fit(x, floc=0)` for the positive targets. For fitted density `f`, `loglik=Σlog f(x_i)`, `AIC=2k−2loglik`, `BIC=k·ln(n)−2loglik`, where k counts free fitted parameters. Select the best distribution by maximum log-likelihood, using AIC/BIC as diagnostics.
- Write `results/table02_mle_loglik.csv` (six rows) with `target,distribution,parameters,n,loglik,aic,bic,best_by_loglik`; serialize `parameters` as a sorted-key JSON object. Produce `figures/fig03_mle_distribution_fits.png` with target histograms/KDEs and all three fitted PDFs.

### `src/02_pca_kmeans.py`

Fit a train-only `StandardScaler` on `SENSORS`, then full PCA. For eigenvalue `λj`, explained variance is `λj/Σλ`; plot individual and cumulative values. PCR uses this same scaling/PCA pattern independently in each model fold.

Fit `KMeans(n_clusters=3, random_state=42, n_init=20)` on standardized sensors. Relabel clusters by ascending original-unit TEY centroid for reproducible regime IDs 0,1,2; inverse-transform centroids. Write `results/table03_kmeans_centroids.csv` with `cluster,n_members,inertia,AT,AP,AH,AFDP,GTEP,TIT,TAT,TEY,CDP` (three rows; total inertia repeated or recorded identically on each row).

Produce `figures/fig04_pca_scree_biplot.png` (scree/cumulative curve plus PC1–PC2 biplot with nine labeled loadings) and `figures/fig05_kmeans_regimes.png` (train PC1–PC2 scores colored by regime, centroid markers, and original-unit centroid summaries). These clusters are descriptive and are not extra regression predictors.

### `src/03_train_regression_cv.py`

Train every candidate independently for CO and NOX under both protocols. Standardize inside each pipeline. Candidates and search spaces are:

| `model` | Exact estimator/pipeline | Search space |
|---|---|---|
| `OLS` | `LinearRegression()` | none |
| `Ridge` | `StandardScaler → Ridge(max_iter=20000)` | `alpha`: {1e-4,1e-3,1e-2,1e-1,1,10,100} |
| `Lasso` | `StandardScaler → Lasso(max_iter=20000, random_state=42)` | same `alpha` grid |
| `PCR` | `StandardScaler → PCA → LinearRegression()` | `n_components`: integers 2 through 9 |
| `Poly2_Ridge` | `PolynomialFeatures(degree=2, include_bias=False) → StandardScaler → Ridge(max_iter=20000)` | same `alpha` grid |
| `MLP` | `StandardScaler → MLPRegressor(early_stopping=True, max_iter=1000, random_state=42)` | `hidden_layer_sizes`: {(32,), (64,32)}; `alpha`: {1e-4,1e-3} |

Use minimum RMSE for random five-fold selection and 2014 selection; break ties by the listed model/grid order, then simpler configuration. Never choose by holdout/2015 performance. Serialize selected parameters as sorted-key JSON in `best_params`.

Write both tables with uppercase target IDs and one row per target × model (12 rows each):

- `results/table04_regression_5fold_cv.csv`: `protocol,target,model,best_params,cv_rmse_mean,cv_rmse_std,cv_r2_mean,n_train,n_test,test_r2,test_rmse,test_mse,q95_rmse,q95_n,q99_rmse,q99_n`.
- `results/table05_regression_chronological_2015.csv`: `protocol,target,model,best_params,val_2014_r2,val_2014_rmse,n_initial_train,n_val,n_train,n_test,test_r2,test_rmse,test_mse,q95_rmse,q95_n,q99_rmse,q99_n`.

For each protocol/target, select a single Raw reference model without test information (minimum CV RMSE or 2014 RMSE). Freeze its hyperparameters for a training-only ablation with variants `Raw`, `IQR`, and `3-sigma`:

- Raw retains every training row. IQR removes a training row if any value in `SENSORS + [target]` is outside its train-fitted 1.5·IQR limits; 3-sigma uses train-fitted `μ±3s` limits. Neither method sees the opposite target.
- Re-fit every pipeline on the retained rows. Validation/test rows, including high-emission spikes, are never trimmed. Use the Raw training partition's Q95/Q99 thresholds for all three variants so tail comparisons use identical test rows.

Write `results/table06_outlier_ablation_untrimmed_test.csv` (2 protocols × 2 targets × 3 variants = 12 rows): `protocol,target,model,variant,n_train_original,n_train_retained,n_test,test_r2,test_rmse,test_mse,q95_rmse,q95_n,q99_rmse,q99_n`.

Produce `figures/fig06_parity_cv_vs_chronological.png` as four observed-vs-predicted panels (CO/NOX × random/chronological), using each protocol's preselected best model, identity lines, and `test_r2`/`test_rmse`. Produce `figures/fig07_outlier_ablation_tails.png` as grouped Raw/IQR/3-sigma bars for overall, Q95, and Q99 RMSE, faceted by target and protocol.

### `src/04_logistic_and_parity.py`

For each protocol, target, and q in {90,95}, compute `τq` from that protocol's training target only and define `z=1[y>τq]`. Fit `StandardScaler → LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=42)` on the same nine sensors; the opposite target is excluded. For chronological final fitting, use 2011–2014 only after model policy is fixed. Use probability 0.5 for alerts without test tuning.

Write `results/table07_logistic_exceedance_metrics.csv` (2 protocols × 2 targets × 2 quantiles = 8 rows): `protocol,target,model,quantile,threshold,n_train,n_test,train_prevalence,test_prevalence,test_accuracy,test_precision,test_recall,test_f1,test_roc_auc,test_pr_auc,tn,fp,fn,tp`. `model` is `LogisticRegression`; `quantile` is `Q90` or `Q95`; confusion counts follow positive=exceedance. Produce `figures/fig08_logistic_confusion_roc.png` with confusion matrices and ROC curves for all eight runs, annotated with AUROC and the training threshold. Include the existing regression parity figure in the final results discussion; do not choose any model using 2015.

## 3. Notebook and acceptance contract

`notebooks/CHE_F315_Gas_Turbine_PEMS.ipynb` must be a fully executed, AST-valid Python notebook with at least seven code cells and no error outputs. Organize exactly these seven Prof. Riju De presentation components, with concise Markdown stating method, leakage controls, and interpretation:

1. **Data/problem:** execute stage 00; show schema, year counts, target isolation, and both split timelines.
2. **EDA/outliers/VIF:** execute stage 01; discuss figures 01–02 and table 01.
3. **MLE:** present figure 03 and table 02, including log-likelihood/AIC/BIC choice.
4. **PCA/K-Means:** execute stage 02; present figures 04–05 and table 03.
5. **Regression CV vs chronology:** execute stage 03; present figure 06 and tables 04–05, without test-based selection.
6. **Outlier ablation:** present figure 07 and table 06; explicitly show unchanged untrimmed test counts and tail behavior.
7. **Q90/Q95 alerts and conclusions:** execute stage 04; present figure 08 and table 07, limitations, and deployment implications.

Use standard Python in code cells (no shell/magic-only execution), display the saved artifacts, and execute top-to-bottom with non-null `execution_count`. Required tables must have at least two data rows and non-all-NaN content; all required figures must be nonempty PNGs larger than 5,000 bytes. Finish by running `.venv/bin/pytest -q` and `.venv/bin/ruff check src tests` exactly.
