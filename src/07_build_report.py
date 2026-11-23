"""Build the concise A4 results, metrics, tables and graphs report."""

from __future__ import annotations

import html as html_mod
import json
import re
from pathlib import Path

import pandas as pd
import pymupdf
from PIL import Image
from weasyprint import HTML

ROOT = Path("/home/samar_perelman$/code/che-f315-gas-turbine-pems")
OUT = ROOT / "report"
PDF_PATH = OUT / "CHE_F315_Gas_Turbine_PEMS_Report.pdf"
MD_PATH = OUT / "REPORT.md"
TMP_RENDER = Path("/tmp/opencode/report_pages")

REPORT_DATE = "23 November 2026"
COURSE = "CHE F315: Machine Learning for Chemical Engineers"
PROFESSOR = "Professor Riju De"
CAMPUS = "BITS Pilani, K. K. Birla Goa Campus"
TEAM = "Samar Talwar (2024A1PS0264G) & Soumya Mandoliya (2024A1PS0277G)"
TITLE = (
    "Gas Turbine Predictive Emission Monitoring System (CO & NOx) "
    "— Results & Figures Report"
)

NAVY = "#1B365D"
STEEL = "#2B5B84"
TINT = "#F2F6FA"
CHARCOAL = "#1F2937"
GREY = "#5B6573"
LINE = "#D7E0EA"
WHITE = "#FFFFFF"

CAPTIONS: list[str] = []


def load_tables():
    return {
        "t1": pd.read_csv(ROOT / "results" / "table01_eda_outliers_vif.csv"),
        "t2": pd.read_csv(ROOT / "results" / "table02_mle_loglik.csv"),
        "t3": pd.read_csv(ROOT / "results" / "table03_kmeans_centroids.csv"),
        "t4": pd.read_csv(ROOT / "results" / "table04_regression_5fold_cv.csv"),
        "t4b": pd.read_csv(ROOT / "results" / "table04b_log1p_co_variant.csv"),
        "t5": pd.read_csv(
            ROOT / "results" / "table05_regression_chronological_2015.csv"
        ),
        "t6": pd.read_csv(
            ROOT / "results" / "table06_outlier_ablation_untrimmed_test.csv"
        ),
        "t7": pd.read_csv(
            ROOT / "results" / "table07_logistic_exceedance_metrics.csv"
        ),
    }


def esc(value) -> str:
    return html_mod.escape(str(value))


def f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def caption(text: str) -> str:
    CAPTIONS.append(text)
    return f'<p class="caption">{esc(text)}</p>'


def table(
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
    text_cols: set[int],
    best_rows: set[int] | None = None,
    target_rows: set[int] | None = None,
) -> str:
    best_rows = best_rows or set()
    target_rows = target_rows or set()
    total = sum(widths)
    colgroup = "".join(
        f'<col style="width:{100.0 * width / total:.3f}%">' for width in widths
    )
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = []
    for row_index, row in enumerate(rows):
        classes = []
        if row_index in best_rows:
            classes.append("best-row")
        if row_index in target_rows:
            classes.append("target-row")
        cells = []
        for col_index, value in enumerate(row):
            cell_class = "" if col_index in text_cols else ' class="num"'
            cells.append(f"<td{cell_class}>{esc(value)}</td>")
        body.append(f'<tr class="{" ".join(classes)}">{"".join(cells)}</tr>')
    return (
        f'<table class="data" style="table-layout:fixed;width:100%">'
        f"<colgroup>{colgroup}</colgroup>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def table_block(content: str, line: str) -> str:
    return f'<div class="table-block">{content}{caption(line)}</div>'


def figure_block(path: str, line: str, tall: bool = False, hero: bool = False) -> str:
    assert (ROOT / path).is_file(), path
    image = (ROOT / path).resolve().as_uri()
    css_class = "figure-block"
    if tall:
        css_class += " tall"
    if hero:
        css_class += " hero"
    return (
        f'<div class="{css_class}">'
        f'<img src="{image}" alt="{esc(line)}">'
        f"{caption(line)}</div>"
    )


def format_params(raw: str) -> str:
    params = json.loads(raw)
    if not params:
        return "—"
    if "hidden_layer_sizes" in params:
        hidden = ",".join(str(value) for value in params["hidden_layer_sizes"])
        return f"({hidden}), α={params['alpha']:g}"
    if "alpha" in params:
        return f"α={params['alpha']:g}"
    if "n_components" in params:
        return f"{params['n_components']} PCs"
    return raw


def build_tables():
    t1, t2, t3, t4, t4b, t5, t6, t7 = (
        TABLES[key] for key in ("t1", "t2", "t3", "t4", "t4b", "t5", "t6", "t7")
    )

    rows_1a = []
    rows_1b = []
    for _, row in t1.iterrows():
        vif = "—" if pd.isna(row["vif"]) else f"{row['vif']:.1f}"
        rows_1a.append(
            [
                row["column"],
                f"{int(row['n']):,}",
                f"{row['mean']:.2f}",
                f"{row['std']:.2f}",
                f"{row['min']:.2f}",
                f"{row['q25']:.2f}",
                f"{row['median']:.2f}",
                f"{row['q75']:.2f}",
                f"{row['max']:.2f}",
                f"{int(row['missing'])}",
            ]
        )
        rows_1b.append(
            [
                row["column"],
                f"{row['iqr_low']:.2f}",
                f"{row['iqr_high']:.2f}",
                f"{int(row['iqr_outliers']):,}",
                f"{row['sigma3_low']:.2f}",
                f"{row['sigma3_high']:.2f}",
                f"{int(row['sigma3_outliers']):,}",
                vif,
            ]
        )

    rows_2 = []
    best_2 = set()
    for index, (_, row) in enumerate(t2.iterrows()):
        if bool(row["best_by_loglik"]):
            best_2.add(index)
        rows_2.append(
            [
                row["target"],
                row["distribution"],
                f"{row['loglik']:,.3f}",
                f"{row['aic']:,.3f}",
                f"{row['bic']:,.3f}",
                "YES" if bool(row["best_by_loglik"]) else "",
            ]
        )

    rows_3 = []
    for _, row in t3.iterrows():
        rows_3.append(
            [
                f"{int(row['cluster'])} "
                + ("Low" if int(row["cluster"]) == 0 else "Mid" if int(row["cluster"]) == 1 else "High"),
                f"{int(row['n_members']):,}",
                f"{row['inertia']:,.2f}",
                f"{row['AT']:.2f}",
                f"{row['AP']:.2f}",
                f"{row['AH']:.2f}",
                f"{row['AFDP']:.2f}",
                f"{row['GTEP']:.2f}",
                f"{row['TIT']:.2f}",
                f"{row['TAT']:.2f}",
                f"{row['TEY']:.2f}",
                f"{row['CDP']:.2f}",
            ]
        )

    rows_4 = []
    best_4 = set()
    for index, (_, row) in enumerate(t4.iterrows()):
        if row["model"] == "MLP":
            best_4.add(index)
        rows_4.append(
            [
                row["target"],
                row["model"],
                format_params(row["best_params"]),
                f(row["cv_rmse_mean"]),
                f(row["cv_r2_mean"]),
                f(row["test_r2"]),
                f(row["test_rmse"]),
                f(row["q95_rmse"]),
                f(row["q99_rmse"]),
            ]
        )

    rows_5 = []
    best_5 = set()
    for index, (_, row) in enumerate(t5.iterrows()):
        selected = (row["target"] == "CO" and row["model"] == "Poly2_Ridge") or (
            row["target"] == "NOX" and row["model"] == "OLS"
        )
        if selected:
            best_5.add(index)
        rows_5.append(
            [
                row["target"],
                row["model"],
                format_params(row["best_params"]),
                f(row["val_2014_r2"]),
                f(row["val_2014_rmse"]),
                f(row["test_r2"]),
                f(row["test_rmse"]),
                f(row["q95_rmse"]),
                f(row["q99_rmse"]),
            ]
        )

    rows_4b = []
    for _, row in t4b.iterrows():
        rows_4b.append(
            [
                "Random 80/20" if row["protocol"] == "random_5fold_holdout" else "Locked 2015",
                f(row["selection_rmse"]),
                f(row["test_r2"]),
                f(row["test_rmse"]),
                f(row["q95_rmse"]),
                f(row["q99_rmse"]),
            ]
        )

    rows_6 = []
    for _, row in t6.iterrows():
        rows_6.append(
            [
                "Random" if row["protocol"] == "random_5fold_holdout" else "2015",
                row["target"],
                row["model"],
                row["variant"],
                f"{int(row['n_train_retained']):,} / {int(row['n_train_original']):,}",
                f"{int(row['n_test']):,}",
                f(row["test_r2"]),
                f(row["test_rmse"]),
                f(row["q95_rmse"]),
                f(row["q99_rmse"]),
            ]
        )

    rows_7 = []
    for _, row in t7.iterrows():
        rows_7.append(
            [
                "Random" if row["protocol"] == "random_5fold_holdout" else "2015",
                row["target"],
                row["quantile"],
                f"{row['threshold']:.2f}",
                f"{pct(row['train_prevalence'])} → {pct(row['test_prevalence'])}",
                f(row["test_recall"]),
                f(row["test_precision"]),
                f(row["test_f1"]),
                f(row["test_roc_auc"]),
                f(row["test_pr_auc"]),
            ]
        )

    return {
        "1a": table(
            ["Variable", "n", "Mean", "Std", "Min", "Q25", "Median", "Q75", "Max", "Missing"],
            rows_1a,
            [8, 6, 10, 9, 9, 9, 10, 9, 9, 6],
            {0},
            target_rows={9, 10},
        ),
        "1b": table(
            ["Variable", "IQR Low", "IQR High", "IQR Out", "3σ Low", "3σ High", "3σ Out", "VIF"],
            rows_1b,
            [9, 12, 12, 8, 12, 12, 8, 8],
            {0},
            target_rows={9, 10},
        ),
        "2": table(
            ["Target", "Distribution", "Log-likelihood", "AIC", "BIC", "Best"],
            rows_2,
            [7, 13, 21, 21, 21, 8],
            {0, 1, 5},
            best_rows=best_2,
        ),
        "3": table(
            ["Regime", "n", "Inertia", "AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"],
            rows_3,
            [7, 7, 11, 8, 10, 8, 8, 9, 9, 9, 9, 9],
            {0},
        ),
        "4": table(
            ["Target", "Model", "Params", "CV RMSE", "CV R²", "Test R²", "Test RMSE", "Q95", "Q99"],
            rows_4,
            [7, 11, 17, 9, 8, 8, 9, 8, 8],
            {0, 1, 2},
            best_rows=best_4,
        ),
        "5": table(
            ["Target", "Model", "Params", "Val R²", "Val RMSE", "Test R²", "Test RMSE", "Q95", "Q99"],
            rows_5,
            [7, 11, 17, 8, 9, 8, 9, 8, 8],
            {0, 1, 2},
            best_rows=best_5,
        ),
        "4b": table(
            ["Protocol", "Selection RMSE", "Test R²", "Test RMSE", "Q95", "Q99"],
            rows_4b,
            [15, 16, 13, 15, 12, 12],
            {0},
        ),
        "6": table(
            ["Protocol", "Target", "Model", "Variant", "Train retained", "Test n", "R²", "RMSE", "Q95", "Q99"],
            rows_6,
            [8, 6, 11, 7, 15, 7, 8, 8, 9, 9],
            {0, 1, 2, 3},
        ),
        "7": table(
            ["Protocol", "Target", "Q", "τ", "Train → test", "Recall", "Precision", "F1", "AUROC", "PR-AUC"],
            rows_7,
            [9, 6, 5, 7, 15, 8, 9, 7, 8, 8],
            {0, 1, 2},
        ),
    }


def header_html() -> str:
    badge = (
        "UCI Gas Turbine Emission Dataset (ID 551) | 36,733 hourly rows (2011–2015) | "
        "9 sensors → CO & NOx modeled independently | 0 missing values"
    )
    return f"""
    <header class="report-header">
      <div class="eyebrow">CHE F315 · RESULTS, METRICS, TABLES &amp; GRAPHS</div>
      <h1>{esc(TITLE)}</h1>
      <div class="meta"><strong>Course:</strong> {esc(COURSE)} &nbsp;|&nbsp;
        <strong>Instructor:</strong> {esc(PROFESSOR)}</div>
      <div class="meta"><strong>Campus:</strong> {esc(CAMPUS)} &nbsp;|&nbsp;
        <strong>Date:</strong> {esc(REPORT_DATE)}</div>
      <div class="meta"><strong>Team:</strong> {esc(TEAM)}</div>
      <div class="badge">{esc(badge)}</div>
    </header>
    """


def kpi_html() -> str:
    cards = [
        ("RANDOM HOLDOUT — CO", "R² 0.776", "RMSE 1.103"),
        ("RANDOM HOLDOUT — NOx", "R² 0.850", "RMSE 4.452"),
        ("LOCKED 2015 — CO", "R² 0.391", "RMSE 1.744"),
        ("LOCKED 2015 — NOx", "R² 0.259", "RMSE 9.583"),
        ("PEAK SENSOR OVERLAP", "TEY 255.6", "CDP 251.0 VIF"),
        ("TAIL FIT & ALERTS", "Lognormal", "AUROC 0.908–0.962"),
    ]
    cells = []
    for label, value, sub in cards:
        cells.append(
            f'<td><div class="kpi"><div class="kpi-label">{esc(label)}</div>'
            f'<div class="kpi-value">{esc(value)}</div>'
            f'<div class="kpi-sub">{esc(sub)}</div></div></td>'
        )
    return (
        '<table class="kpi-grid"><tr>'
        + "".join(cells[:3])
        + "</tr><tr>"
        + "".join(cells[3:])
        + "</tr></table>"
    )


def build_html(tables: dict) -> str:
    CAPTIONS.clear()

    s1 = (
        "<h2>1 · Dataset Summary, Distributions, Multicollinearity &amp; Correlations</h2>"
        + table_block(
            tables["1a"],
            "CO is right-skewed (mean 2.37 vs median 1.71); NOx is steadier and all 11 variables have zero missing values.",
        )
        + table_block(
            tables["1b"],
            "CO/NOx hold 2,136/770 IQR spikes; TEY (255.6), CDP (251.0), and TIT (38.4) show the strongest overlap.",
        )
        + figure_block(
            "figures/fig01_univariate_kde_boxplots.png",
            "Most hours are low-emission, but CO reaches 44.1 and rare spikes pull the mean above the median.",
            tall=True,
        )
        + figure_block(
            "figures/ppt/ppt01_sensor_vif_log.png",
            "TEY 255.6, CDP 251.0, and TIT 38.4 exceed the VIF-10 flag, making simple linear coefficients unstable.",
            tall=True,
        )
        + figure_block(
            "figures/fig02_correlation_heatmap_11x11.png",
            "The heatmap confirms that load, firing, and exhaust sensors rise together in one dense correlation block.",
        )
    )

    s2 = (
        "<h2>2 · Parametric Emission Distribution Fitting (MLE)</h2>"
        + table_block(
            tables["2"],
            "Lognormal has the highest log-likelihood and lowest AIC/BIC for both CO and NOx, confirming a heavy tail.",
        )
        + figure_block(
            "figures/fig03_mle_distribution_fits.png",
            "Fitted densities show the normal curve missing the high-emission tail that lognormal follows more closely.",
        )
        + figure_block(
            "figures/ppt/ppt02_mle_tail_survival.png",
            "Log-scale tail survival shows lognormal tracking rare CO/NOx spikes better than normal or gamma.",
        )
    )

    s3 = (
        "<h2>3 · Operating Modes (PCA &amp; K-Means Clustering)</h2>"
        + table_block(
            tables["3"],
            "Three regimes (7,010/15,148/7,228 hours) rise from 112.4 to 154.8 TEY with total inertia 124,978.17.",
        )
        + figure_block(
            "figures/fig04_pca_scree_biplot.png",
            "PC1+PC2 explain 74.3% of sensor variation and expose the common load direction behind the overlap.",
        )
        + figure_block(
            "figures/fig05_kmeans_regimes.png",
            "K-Means separates low, mid, and high load in PCA space and summarizes each original-unit sensor centroid.",
        )
        + figure_block(
            "figures/ppt/ppt03_regime_emissions.png",
            "Low-load hours show much higher median CO (about 4.26 vs 1.18), while NOx stays comparatively flat.",
        )
    )

    s4 = (
        "<h2>4 · Regression Results — Random CV vs. Locked Future Year (2015)</h2>"
        + table_block(
            tables["4"],
            "Random holdout: MLP wins both targets (CO R² 0.776, NOx R² 0.850), beating OLS (0.557 and 0.495).",
        )
        + table_block(
            tables["5"],
            "2014 selects Poly2 Ridge for CO (1.344 RMSE) and OLS for NOx (8.550); 2015 is scored once.",
        )
        + table_block(
            tables["4b"],
            "Log1p(CO) MLP does not beat the raw-target model (1.129 vs 1.103), so the primary CO result stands.",
        )
        + figure_block(
            "figures/ppt/ppt04_multiyear_drift_2011_2015.png",
            "Yearly medians drift: CO rises 1.16 to 2.53 while NOx falls 65.5 to 56.8, explaining the 2015 error rise.",
            tall=True,
        )
        + figure_block(
            "figures/ppt/ppt05_cv_vs_2015_parity.png",
            "Random to 2015: CO RMSE rises 58.1% and NOx 115.3%; parity widens sharply at high emissions.",
            tall=True,
        )
        + figure_block(
            "figures/fig06_parity_cv_vs_chronological.png",
            "Four parity panels confirm tight random-holdout agreement and visibly wider 2015 scatter for both targets.",
            tall=True,
        )
    )

    s5 = (
        "<h2>5 · Outlier Removal Ablation Study</h2>"
        + table_block(
            tables["6"],
            "Trimming never changes test rows, but IQR raises overall and Q99 error for both targets.",
        )
        + figure_block(
            "figures/fig07_outlier_ablation_tails.png",
            "Training-only IQR trimming worsens overall and tail RMSE; raw training wins on identical untrimmed test hours.",
        )
    )

    final_box = """
    <div class="final-box">
      <div class="final-title">FINAL RESULTS — FOUR TAKEAWAYS</div>
      <ul>
        <li>Best random holdout: CO R² 0.776 / RMSE 1.103; NOx R² 0.850 / RMSE 4.452.</li>
        <li>Locked 2015: CO R² 0.391 / RMSE 1.744; NOx R² 0.259 / RMSE 9.583, confirming plant drift.</li>
        <li>Keep raw spikes: IQR trimming raises random-CO Q99 error from 7.27 to 13.87.</li>
        <li>Alerts reach AUROC 0.908–0.962, but the 2015 NOx prevalence shift demands recalibration.</li>
      </ul>
    </div>
    """

    s6 = (
        "<h2>6 · High-Emission Compliance Alerts (Q90 &amp; Q95 Logistic Classification)</h2>"
        + table_block(
            tables["7"],
            "Q90/Q95 alarms rank spikes well (AUROC 0.908–0.962), but 2015 NOx prevalence falls to 3.4–5.3%.",
        )
        + figure_block(
            "figures/fig08_logistic_confusion_roc.png",
            "Confusion matrices and ROC curves show high recall but many false alarms on locked-2015 NOx.",
            tall=True,
            hero=True,
        )
        + final_box
    )

    css = f"""
    @page {{
      size: A4 portrait;
      margin: 16mm 14mm 18mm 14mm;
      @top-center {{
        content: "CHE F315 — Gas Turbine PEMS Results Report";
        font-family: "Liberation Sans", "DejaVu Sans", sans-serif;
        font-size: 7.4pt; color: {NAVY}; font-weight: 700;
        padding-bottom: 2mm; border-bottom: 0.3mm solid {LINE};
      }}
      @bottom-left {{
        content: "Samar Talwar & Soumya Mandoliya | BITS Pilani Goa";
        font-family: "Liberation Sans", "DejaVu Sans", sans-serif;
        font-size: 7.1pt; color: {GREY};
      }}
      @bottom-right {{
        content: "Page " counter(page) " of " counter(pages);
        font-family: "Liberation Sans", "DejaVu Sans", sans-serif;
        font-size: 7.1pt; color: {GREY};
      }}
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: {WHITE}; }}
    body {{
      font-family: "Liberation Sans", "DejaVu Sans", sans-serif;
      color: {CHARCOAL}; font-size: 8.4pt; line-height: 1.3;
    }}
    .report-header {{
      background: {TINT}; border-left: 4.5mm solid {NAVY};
      padding: 4.5mm 5mm 4mm 5mm; margin-bottom: 5mm;
    }}
    .eyebrow {{
      font-size: 7.2pt; letter-spacing: 0.6pt; font-weight: 700;
      color: {STEEL}; text-transform: uppercase; margin-bottom: 1.5mm;
    }}
    h1 {{
      margin: 0 0 2.5mm 0; font-size: 16.3pt; line-height: 1.15;
      color: {NAVY}; font-weight: 700;
    }}
    .meta {{ font-size: 7.7pt; color: {CHARCOAL}; margin: 0.8mm 0; }}
    .meta strong {{ color: {NAVY}; }}
    .badge {{
      margin-top: 2.2mm; padding-top: 2mm; border-top: 0.3mm solid {LINE};
      font-size: 7.3pt; color: {STEEL}; font-weight: 700;
    }}
    .kpi-grid {{ width: 100%; border-collapse: separate; border-spacing: 2mm 2mm; margin: 0 0 2mm 0; }}
    .kpi-grid td {{ width: 33.333%; vertical-align: top; padding: 0; }}
    .kpi {{
      background: {TINT}; border-top: 1.2mm solid {NAVY};
      padding: 2.4mm 3mm 2.2mm 3mm; min-height: 18mm;
    }}
    .kpi-label {{ font-size: 6.3pt; letter-spacing: 0.25pt; color: {STEEL}; font-weight: 700; }}
    .kpi-value {{ font-size: 14.5pt; line-height: 1.05; color: {NAVY}; font-weight: 700; margin-top: 1mm; }}
    .kpi-sub {{ font-size: 7.1pt; color: {CHARCOAL}; margin-top: 0.8mm; }}
    h2 {{
      font-size: 11.4pt; line-height: 1.2; color: {NAVY}; margin: 4.5mm 0 2.2mm 0;
      padding-bottom: 1.1mm; border-bottom: 0.35mm solid {NAVY};
      break-after: avoid; page-break-after: avoid;
    }}
    .table-block, .figure-block {{
      break-inside: avoid; page-break-inside: avoid;
      margin: 0 0 4.4mm 0;
    }}
    .figure-block {{ text-align: center; }}
    .figure-block img {{
      display: block; margin: 0 auto; max-width: 100%; max-height: 94mm;
      width: auto; height: auto;
    }}
    .figure-block.tall img {{ max-height: 120mm; }}
    .figure-block.hero img {{ max-height: 180mm; }}
    .caption {{
      margin: 1.3mm 0 0 0; padding-left: 2mm;
      border-left: 0.7mm solid {STEEL};
      font-size: 7.15pt; line-height: 1.25; color: {CHARCOAL}; text-align: left;
    }}
    table.data {{
      width: 100%; border-collapse: collapse; table-layout: fixed;
      font-size: 7pt; line-height: 1.18;
    }}
    table.data thead {{ display: table-header-group; }}
    table.data th {{
      background: {NAVY}; color: {WHITE}; font-weight: 700;
      padding: 1.25mm 0.7mm; border: 0.2mm solid {NAVY}; text-align: left;
      font-size: 6.9pt;
    }}
    table.data td {{
      padding: 1.05mm 0.75mm; border-bottom: 0.2mm solid {LINE};
      vertical-align: top; overflow-wrap: anywhere;
    }}
    table.data tr:nth-child(even) td {{ background: #F8FAFD; }}
    table.data .num {{ text-align: right; white-space: nowrap; }}
    table.data tr.best-row td {{ background: #E6EFF8 !important; font-weight: 700; }}
    table.data tr.target-row td {{ background: {TINT}; font-weight: 700; }}
    .final-box {{
      break-inside: avoid; page-break-inside: avoid;
      background: {TINT}; border-left: 4.5mm solid {NAVY};
      padding: 3mm 4mm 2.5mm 4mm; margin-top: 6mm;
    }}
    .final-title {{
      font-size: 7.2pt; letter-spacing: 0.5pt; font-weight: 700;
      color: {NAVY}; text-transform: uppercase; margin-bottom: 1.5mm;
    }}
    .final-box ul {{ margin: 0; padding-left: 4.5mm; }}
    .final-box li {{ font-size: 8.1pt; line-height: 1.42; margin: 0.4mm 0; }}
    """

    body = (
        header_html()
        + kpi_html()
        + s1
        + s2
        + s3
        + s4
        + s5
        + s6
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{css}</style></head><body>{body}</body></html>"
    )


def build_markdown(tables: dict) -> None:
    # Companion index only; the PDF is the presentation artifact.
    lines = [
        "# Gas Turbine PEMS — Results & Figures Report",
        "",
        f"**Course:** {COURSE}  ",
        f"**Instructor:** {PROFESSOR}  ",
        f"**Campus:** {CAMPUS}  ",
        f"**Team:** {TEAM}  ",
        f"**Date:** {REPORT_DATE}",
        "",
        "**Dataset:** UCI Gas Turbine Emission Dataset (ID 551) | 36,733 hourly rows (2011–2015) | 9 sensors → CO & NOx modeled independently | 0 missing values",
        "",
        "## Executive Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        "| Random holdout CO | R² 0.776 / RMSE 1.103 |",
        "| Random holdout NOx | R² 0.850 / RMSE 4.452 |",
        "| Locked 2015 CO | R² 0.391 / RMSE 1.744 |",
        "| Locked 2015 NOx | R² 0.259 / RMSE 9.583 |",
        "| Peak overlap | TEY VIF 255.6 / CDP VIF 251.0 |",
        "| Tail fit | Lognormal best for CO and NOx |",
        "| Alert AUROC | 0.908–0.962 |",
        "",
        "## Artifact Index",
        "",
        "| Section | Table | Figures |",
        "|---|---|---|",
        "| 1 | [table01](../results/table01_eda_outliers_vif.csv) | [fig01](../figures/fig01_univariate_kde_boxplots.png) · [ppt01](../figures/ppt/ppt01_sensor_vif_log.png) · [fig02](../figures/fig02_correlation_heatmap_11x11.png) |",
        "| 2 | [table02](../results/table02_mle_loglik.csv) | [fig03](../figures/fig03_mle_distribution_fits.png) · [ppt02](../figures/ppt/ppt02_mle_tail_survival.png) |",
        "| 3 | [table03](../results/table03_kmeans_centroids.csv) | [fig04](../figures/fig04_pca_scree_biplot.png) · [fig05](../figures/fig05_kmeans_regimes.png) · [ppt03](../figures/ppt/ppt03_regime_emissions.png) |",
        "| 4 | [table04](../results/table04_regression_5fold_cv.csv) · [table05](../results/table05_regression_chronological_2015.csv) · [table04b](../results/table04b_log1p_co_variant.csv) | [ppt04](../figures/ppt/ppt04_multiyear_drift_2011_2015.png) · [ppt05](../figures/ppt/ppt05_cv_vs_2015_parity.png) · [fig06](../figures/fig06_parity_cv_vs_chronological.png) |",
        "| 5 | [table06](../results/table06_outlier_ablation_untrimmed_test.csv) | [fig07](../figures/fig07_outlier_ablation_tails.png) |",
        "| 6 | [table07](../results/table07_logistic_exceedance_metrics.csv) | [fig08](../figures/fig08_logistic_confusion_roc.png) |",
        "",
        "## Final Takeaways",
        "",
        "- Best random holdout: CO R² 0.776 / RMSE 1.103; NOx R² 0.850 / RMSE 4.452.",
        "- Locked 2015: CO R² 0.391 / RMSE 1.744; NOx R² 0.259 / RMSE 9.583, confirming plant drift.",
        "- Keep raw spikes: IQR trimming raises random-CO Q99 error from 7.27 to 13.87.",
        "- Alerts reach AUROC 0.908–0.962, but the 2015 NOx prevalence shift demands recalibration.",
        "",
        "`ppt06_ablation_and_alerts.png` is the combined presentation view; this report uses its dedicated components `fig07` and `fig08`.",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pdf(tables: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    html = build_html(tables)
    HTML(string=html, base_url=str(ROOT) + "/").write_pdf(PDF_PATH)
    print(f"Wrote {PDF_PATH}")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def verify_pdf() -> None:
    document = pymupdf.open(PDF_PATH)
    assert len(document) >= 6, f"Unexpectedly short report: {len(document)} pages"
    TMP_RENDER.mkdir(parents=True, exist_ok=True)
    for old in TMP_RENDER.glob("page_*.png"):
        old.unlink()
    all_text = []
    image_count = 0
    for index, page in enumerate(document, start=1):
        rect = page.rect
        assert abs(rect.width - 595.276) < 1.0, (index, rect)
        assert abs(rect.height - 841.89) < 1.0, (index, rect)
        text = page.get_text("text")
        assert len(text.strip()) > 20
        all_text.append(normalize(text))
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            x0, y0, x1, y1 = block["bbox"]
            assert x0 >= 30 and x1 <= 566, (index, "horizontal", block["bbox"])
            assert y0 >= 10 and y1 <= 835, (index, "vertical", block["bbox"])
        images = page.get_image_info()
        image_count += len(images)
        for item in images:
            x0, y0, x1, y1 = item["bbox"]
            assert x0 >= 35 and x1 <= 561, (index, "image horizontal", item["bbox"])
            assert y0 >= 35 and y1 <= 805, (index, "image vertical", item["bbox"])
        pix = page.get_pixmap(dpi=150)
        pix.save(TMP_RENDER / f"page_{index:02d}.png")
    joined = " ".join(all_text)
    missing = [line for line in CAPTIONS if normalize(line) not in joined]
    assert not missing, f"Captions not found in PDF: {missing}"
    assert image_count == 13, f"Expected 13 figures, found {image_count}"

    thumbs = []
    for page_path in sorted(TMP_RENDER.glob("page_*.png")):
        with Image.open(page_path) as image:
            image = image.convert("RGB")
            image.thumbnail((300, 424))
            canvas = Image.new("RGB", (312, 436), "white")
            canvas.paste(image, ((312 - image.width) // 2, (436 - image.height) // 2))
            thumbs.append(canvas)
    columns = 4
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 312, rows * 436), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 312, (index // columns) * 436))
    contact = Path("/tmp/opencode/report_contact.png")
    sheet.save(contact)
    print(f"Rendered {len(document)} pages to {TMP_RENDER}")
    print(f"Contact sheet: {contact}")
    print(f"Verified {len(CAPTIONS)} captions and {image_count} figures")


def main() -> None:
    global TABLES
    TABLES = load_tables()
    report_tables = build_tables()
    build_pdf(report_tables)
    build_markdown(TABLES)
    verify_pdf()
    print(f"Report package ready in {OUT}")


if __name__ == "__main__":
    main()
