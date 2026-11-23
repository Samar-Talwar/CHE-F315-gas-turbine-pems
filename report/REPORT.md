# Gas Turbine PEMS — Results & Figures Report

**Course:** CHE F315: Machine Learning for Chemical Engineers  
**Instructor:** Professor Riju De  
**Campus:** BITS Pilani, K. K. Birla Goa Campus  
**Team:** Samar Talwar (2024A1PS0264G) & Soumya Mandoliya (2024A1PS0277G)  
**Date:** 23 November 2026

**Dataset:** UCI Gas Turbine Emission Dataset (ID 551) | 36,733 hourly rows (2011–2015) | 9 sensors → CO & NOx modeled independently | 0 missing values

## Executive Metrics

| Metric | Value |
|---|---|
| Random holdout CO | R² 0.776 / RMSE 1.103 |
| Random holdout NOx | R² 0.850 / RMSE 4.452 |
| Locked 2015 CO | R² 0.391 / RMSE 1.744 |
| Locked 2015 NOx | R² 0.259 / RMSE 9.583 |
| Peak overlap | TEY VIF 255.6 / CDP VIF 251.0 |
| Tail fit | Lognormal best for CO and NOx |
| Alert AUROC | 0.908–0.962 |

## Artifact Index

| Section | Table | Figures |
|---|---|---|
| 1 | [table01](../results/table01_eda_outliers_vif.csv) | [fig01](../figures/fig01_univariate_kde_boxplots.png) · [ppt01](../figures/ppt/ppt01_sensor_vif_log.png) · [fig02](../figures/fig02_correlation_heatmap_11x11.png) |
| 2 | [table02](../results/table02_mle_loglik.csv) | [fig03](../figures/fig03_mle_distribution_fits.png) · [ppt02](../figures/ppt/ppt02_mle_tail_survival.png) |
| 3 | [table03](../results/table03_kmeans_centroids.csv) | [fig04](../figures/fig04_pca_scree_biplot.png) · [fig05](../figures/fig05_kmeans_regimes.png) · [ppt03](../figures/ppt/ppt03_regime_emissions.png) |
| 4 | [table04](../results/table04_regression_5fold_cv.csv) · [table05](../results/table05_regression_chronological_2015.csv) · [table04b](../results/table04b_log1p_co_variant.csv) | [ppt04](../figures/ppt/ppt04_multiyear_drift_2011_2015.png) · [ppt05](../figures/ppt/ppt05_cv_vs_2015_parity.png) · [fig06](../figures/fig06_parity_cv_vs_chronological.png) |
| 5 | [table06](../results/table06_outlier_ablation_untrimmed_test.csv) | [fig07](../figures/fig07_outlier_ablation_tails.png) |
| 6 | [table07](../results/table07_logistic_exceedance_metrics.csv) | [fig08](../figures/fig08_logistic_confusion_roc.png) |

## Final Takeaways

- Best random holdout: CO R² 0.776 / RMSE 1.103; NOx R² 0.850 / RMSE 4.452.
- Locked 2015: CO R² 0.391 / RMSE 1.744; NOx R² 0.259 / RMSE 9.583, confirming plant drift.
- Keep raw spikes: IQR trimming raises random-CO Q99 error from 7.27 to 13.87.
- Alerts reach AUROC 0.908–0.962, but the 2015 NOx prevalence shift demands recalibration.

`ppt06_ablation_and_alerts.png` is the combined presentation view; this report uses its dedicated components `fig07` and `fig08`.
