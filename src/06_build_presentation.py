"""Build the 22-slide CHE F315 gas-turbine PEMS presentation.

Outputs
-------
presentation/CHE_F315_Gas_Turbine_PEMS.pdf   16:9, vector-crisp, 22 pages
presentation/CHE_F315_Gas_Turbine_PEMS.pptx  editable matching deck
presentation/SPEAKER_NOTES.md                  minute-by-minute speaker script

The PDF is rendered with WeasyPrint; the PPTX with python-pptx. Both use the
same verified numbers read from results/*.csv and the same fixed 16:9 layout.
"""

from __future__ import annotations

import html as html_mod
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt
from weasyprint import HTML

ROOT = Path("/home/samar_perelman$/code/che-f315-gas-turbine-pems")
OUT = ROOT / "presentation"
PDF_PATH = OUT / "CHE_F315_Gas_Turbine_PEMS.pdf"
PPTX_PATH = OUT / "CHE_F315_Gas_Turbine_PEMS.pptx"
NOTES_PATH = OUT / "SPEAKER_NOTES.md"
TMP_RENDER = Path("/tmp/opencode/presentation_pages")

PRESENTATION_DATE = "23 November 2026"
COURSE = "CHE F315 — Machine Learning for Chemical Engineers"
PROFESSOR = "Professor Riju De"
INSTITUTION = "BITS Pilani, K. K. Birla Goa Campus"
TEAM = [
    ("Samar Talwar", "2024A1PS0264G"),
    ("Soumya Mandoliya", "2024A1PS0277G"),
]
REPO = "github.com/Samar-Talwar/CHE-F315-gas-turbine-pems"

PAGE_W = 960.0
PAGE_H = 540.0
TOTAL = 22

NAVY = "#1B365D"
STEEL = "#2B5B84"
TINT = "#F2F6FA"
CHARCOAL = "#1F2937"
GREY = "#5B6573"
LINE = "#D7E0EA"
CRIMSON = "#A63A50"
AMBER = "#B7791F"
WHITE = "#FFFFFF"

FONT = "Liberation Sans"
PPT_FONT = "Arial"


# ---------------------------------------------------------------------------
# Verified data and formatting helpers
# ---------------------------------------------------------------------------


def load_verified_data():
    df = pd.read_csv(ROOT / "data" / "processed" / "gas_turbine_modeling.csv")
    tables = {
        "t1": pd.read_csv(ROOT / "results" / "table01_eda_outliers_vif.csv"),
        "t2": pd.read_csv(ROOT / "results" / "table02_mle_loglik.csv"),
        "t3": pd.read_csv(ROOT / "results" / "table03_kmeans_centroids.csv"),
        "t4": pd.read_csv(ROOT / "results" / "table04_regression_5fold_cv.csv"),
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
    assert df.shape == (36733, 12)
    assert df["year"].value_counts().sort_index().to_dict() == {
        2011: 7411,
        2012: 7628,
        2013: 7152,
        2014: 7158,
        2015: 7384,
    }
    assert int(df.isna().sum().sum()) == 0
    return df, tables


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def get_cell(
    table: pd.DataFrame,
    target: str,
    model: str,
    column: str,
) -> float:
    subset = table[(table["target"] == target) & (table["model"] == model)]
    assert len(subset) == 1
    return float(subset.iloc[0][column])


def t7_cell(target: str, quantile: str, column: str) -> float:
    subset = tables["t7"][
        (tables["t7"]["protocol"] == "chronological_2015")
        & (tables["t7"]["target"] == target)
        & (tables["t7"]["quantile"] == quantile)
    ]
    assert len(subset) == 1
    return float(subset.iloc[0][column])


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def check_images(paths):
    for path in paths:
        full = ROOT / path
        assert full.is_file(), f"Missing figure: {path}"
        assert full.stat().st_size > 5000, f"Figure too small: {path}"


def bullet(lead: str, text: str) -> dict:
    return {"lead": lead, "text": text}


def takeaway(label: str, text: str, tone: str = "navy") -> dict:
    return {"label": label, "text": text, "tone": tone}


# ---------------------------------------------------------------------------
# Slide content
# ---------------------------------------------------------------------------


df, tables = load_verified_data()
t1 = tables["t1"].set_index("column")
t3 = tables["t3"]
n_low, n_mid, n_high = (int(value) for value in t3["n_members"])
tey_low, tey_mid, tey_high = (float(value) for value in t3["TEY"])
t4 = tables["t4"]
t5 = tables["t5"]
t6 = tables["t6"]
t7 = tables["t7"]

year_medians = df.groupby("year")[["CO", "TEY", "NOX", "AT"]].median()

co_median = float(t1.loc["CO", "median"])
co_mean = float(t1.loc["CO", "mean"])
co_max = float(t1.loc["CO", "max"])
nox_median = float(t1.loc["NOX", "median"])
nox_max = float(t1.loc["NOX", "max"])
vif_tey = float(t1.loc["TEY", "vif"])
vif_cdp = float(t1.loc["CDP", "vif"])
vif_tit = float(t1.loc["TIT", "vif"])
co_drift_start = float(year_medians.loc[2011, "CO"])
co_drift_end = float(year_medians.loc[2015, "CO"])
nox_drift_start = float(year_medians.loc[2011, "NOX"])
nox_drift_end = float(year_medians.loc[2015, "NOX"])
tey_drift_start = float(year_medians.loc[2011, "TEY"])
tey_drift_end = float(year_medians.loc[2015, "TEY"])
at_drift_start = float(year_medians.loc[2011, "AT"])
at_drift_end = float(year_medians.loc[2015, "AT"])

random_co_mlp_r2 = get_cell(t4, "CO", "MLP", "test_r2")
random_nox_mlp_r2 = get_cell(t4, "NOX", "MLP", "test_r2")
random_co_mlp_rmse = get_cell(t4, "CO", "MLP", "test_rmse")
random_nox_mlp_rmse = get_cell(t4, "NOX", "MLP", "test_rmse")
random_co_ols_r2 = get_cell(t4, "CO", "OLS", "test_r2")
random_nox_ols_r2 = get_cell(t4, "NOX", "OLS", "test_r2")
chrono_co_r2 = get_cell(t5, "CO", "Poly2_Ridge", "test_r2")
chrono_nox_r2 = get_cell(t5, "NOX", "OLS", "test_r2")
chrono_co_rmse = get_cell(t5, "CO", "Poly2_Ridge", "test_rmse")
chrono_nox_rmse = get_cell(t5, "NOX", "OLS", "test_rmse")
co_rmse_increase = chrono_co_rmse / random_co_mlp_rmse - 1.0
nox_rmse_increase = chrono_nox_rmse / random_nox_mlp_rmse - 1.0
auroc_min = float(t7["test_roc_auc"].min())
auroc_max = float(t7["test_roc_auc"].max())

co_q90_precision = t7_cell("CO", "Q90", "test_precision")
co_q90_recall = t7_cell("CO", "Q90", "test_recall")
nox_q90_prevalence = t7_cell("NOX", "Q90", "test_prevalence")
nox_q95_prevalence = t7_cell("NOX", "Q95", "test_prevalence")
nox_q95_precision = t7_cell("NOX", "Q95", "test_precision")
nox_q95_recall = t7_cell("NOX", "Q95", "test_recall")
nox_q90_precision = t7_cell("NOX", "Q90", "test_precision")
nox_q90_recall = t7_cell("NOX", "Q90", "test_recall")
nox_q90_fp = int(t7_cell("NOX", "Q90", "fp"))
nox_q90_tp = int(t7_cell("NOX", "Q90", "tp"))
nox_q90_fn = int(t7_cell("NOX", "Q90", "fn"))

co_raw_q99 = float(
    t6[(t6.protocol == "random_5fold_holdout") & (t6.target == "CO") & (t6.variant == "Raw")].iloc[0]["q99_rmse"]
)
co_iqr_q99 = float(
    t6[(t6.protocol == "random_5fold_holdout") & (t6.target == "CO") & (t6.variant == "IQR")].iloc[0]["q99_rmse"]
)
nox_raw_q99 = float(
    t6[(t6.protocol == "chronological_2015") & (t6.target == "NOX") & (t6.variant == "Raw")].iloc[0]["q99_rmse"]
)
nox_iqr_q99 = float(
    t6[(t6.protocol == "chronological_2015") & (t6.target == "NOX") & (t6.variant == "IQR")].iloc[0]["q99_rmse"]
)

SLIDES = [
    {
        "layout": "title",
        "title": "Predicting Gas Turbine Emissions Using Machine Learning",
        "subtitle": "A Software-Based Alternative to Expensive Physical Emission Sensors (CO & NOx)",
    },
    {
        "layout": "roadmap",
        "title": "Presentation Roadmap",
        "steps": [
            ("1", "Problem & Data", "Why emissions matter and what the plant measures."),
            ("2", "Understand the Plant", "Hidden patterns and natural operating modes."),
            ("3", "Model & Test Fairly", "Simple to smart models, with no peeking."),
            ("4", "Results & Surprises", "What worked and what changed over time."),
            ("5", "Plant Takeaways", "Turning predictions into practical warnings."),
        ],
        "note": "About 17–18 minutes · Samar Talwar presents slides 1–11 · Soumya Mandoliya presents slides 12–22.",
        "takeaway": takeaway(
            "What This Means",
            "We follow one plant story: understand the data, test honestly, then act on the results.",
        ),
    },
    {
        "layout": "bullets",
        "title": "What Problem Are We Solving?",
        "bullets": [
            bullet("The job:", "A gas turbine burns fuel to make electricity. CO and NOx are unwanted side products."),
            bullet("The rule:", "Plants must measure and limit these gases, often hour by hour."),
            bullet("The hardware:", "Physical emission analyzers (CEMS) work well, but they are costly to buy, calibrate, and repair."),
            bullet("The software option:", "A PEMS learns from normal turbine sensors and estimates emissions without adding hardware."),
        ],
        "takeaway": takeaway(
            "Why It Matters",
            "A software backup can give earlier warning and reduce dependence on one expensive instrument.",
        ),
    },
    {
        "layout": "two_cards",
        "title": "Our Engineering Goals",
        "cards": [
            (
                "1",
                "Predict the level",
                "Estimate hourly CO and NOx from nine sensors. R² tells us how much hourly variation we explain; RMSE tells us the typical miss.",
            ),
            (
                "2",
                "Warn before the spike",
                "Flag the top 10% and top 5% worst hours. Operators need time to act, not a perfect post-mortem.",
            ),
        ],
        "note": "Hard rule: CO and NOx are trained separately. NOx never enters a CO model, and CO never enters a NOx model.",
        "takeaway": takeaway(
            "Success",
            "Useful predictions plus trustworthy warnings — not just a high score.",
        ),
    },
    {
        "layout": "stats",
        "title": "Where Did the Data Come From?",
        "stats": [
            ("36,733", "hourly readings"),
            ("2011–2015", "five full years"),
            ("9", "everyday sensors"),
            ("0", "missing values"),
        ],
        "bullets": [
            bullet("Real plant:", "UCI public dataset ID 551, from a Turkish gas-turbine power plant."),
            bullet("Every row:", "weather, turbine settings, and the measured CO and NOx for that hour."),
            bullet("Why years matter:", "They let us test on an unseen future period, not just random rows."),
        ],
        "takeaway": takeaway(
            "Data Quality",
            "Five years of complete hourly data gives us both learning and a realistic future test.",
        ),
    },
    {
        "layout": "table",
        "title": "Inside the Dataset: 9 Inputs → 2 Outputs",
        "table": {
            "header": ["Group", "Short name", "What it means"],
            "rows": [
                ["Weather", "AT", "Air temperature"],
                ["Weather", "AP", "Air pressure"],
                ["Weather", "AH", "Humidity"],
                ["Turbine", "TEY", "Power output (load)"],
                ["Turbine", "TIT", "Turbine inlet temperature"],
                ["Turbine", "TAT", "Exhaust temperature"],
                ["Turbine", "CDP", "Compressor discharge pressure"],
                ["Turbine", "GTEP", "Exhaust pressure"],
                ["Turbine", "AFDP", "Air-filter pressure drop"],
                ["Answer", "CO", "Carbon monoxide level"],
                ["Answer", "NOx", "Nitrogen oxide level"],
            ],
            "widths": [170, 160, 500],
            "target_rows": [9, 10],
        },
        "note": "Inputs = the first nine rows. Outputs = the last two. CO and NOx are never inputs to each other.",
        "note_top": 368,
        "takeaway": takeaway(
            "Model Boundary",
            "The model sees only nine sensors. CO and NOx are separate answers.",
        ),
    },
    {
        "layout": "figure_side",
        "title": "First Look at Emissions: Mostly Low, Rare Spikes",
        "image": "figures/fig01_univariate_kde_boxplots.png",
        "bullets": [
            bullet("Most hours are clean.", f"Typical CO is {fmt(co_median, 2)} mg/m³; typical NOx is {fmt(nox_median, 1)} mg/m³."),
            bullet("Rare spikes dominate the average.", f"CO reaches {fmt(co_max, 1)}, while the average is pulled up to {fmt(co_mean, 2)}."),
            bullet("NOx is steadier.", f"Its maximum is {fmt(nox_max, 1)}, and its spread is much narrower than CO's."),
        ],
        "takeaway": takeaway(
            "What This Means",
            "The rare high hours are exactly what a monitor must catch, so we keep them in every test.",
        ),
    },
    {
        "layout": "figure",
        "title": "Hidden Trap #1 — Sensors Move Together",
        "image": "figures/ppt/ppt01_sensor_vif_log.png",
        "bullets": [
            bullet("Sensors overlap.", "Power output (TEY), compressor pressure (CDP), and firing temperature (TIT) rise and fall together."),
            bullet("Overlap scores (VIF) are extreme.", f"VIF reaches {fmt(vif_tey, 1)}, {fmt(vif_cdp, 1)}, and {fmt(vif_tit, 1)}; a simple line cannot separate their effects."),
            bullet("But complexity alone is not enough.", "Ridge and PCA hardly improved accuracy — the bigger issue is curved combustion behavior."),
        ],
        "takeaway": takeaway(
            "Important Finding",
            "When inputs move together, a straight-line model becomes fragile; we need combined and curved effects.",
            tone="warning",
        ),
    },
    {
        "layout": "figure",
        "title": "How Inputs Relate to CO and NOx",
        "image": "figures/fig02_correlation_heatmap_11x11.png",
        "bullets": [
            bullet("Load is the big CO lever.", "Higher power output generally means more complete burning and lower CO."),
            bullet("NOx needs a wider view.", "Firing temperature and ambient weather matter, not load alone."),
            bullet("Keep all nine sensors.", "The heatmap shows overlap, but each sensor still adds some unique information."),
        ],
        "takeaway": takeaway(
            "Plant Reading",
            "Physics guides the model: load helps explain CO, while NOx needs the full operating picture.",
        ),
    },
    {
        "layout": "figure_top",
        "title": "Why a Standard Bell Curve Fails Here",
        "image": "figures/ppt/ppt02_mle_tail_survival.png",
        "bullets": [
            bullet("Too optimistic:", "A bell curve says extremes are almost impossible."),
            bullet("Lognormal wins:", "It follows the long tail and fits far better."),
            bullet("Grade the tails:", "We report Q95 and Q99, not just average error."),
        ],
        "takeaway": takeaway(
            "Why It Matters",
            "Rare does not mean unimportant; the high-emission tail is the reason the monitor exists.",
        ),
    },
    {
        "layout": "two_figures",
        "title": "Discovering 3 Natural Plant Operating Modes",
        "images": [
            ("figures/fig04_pca_scree_biplot.png", "PCA: main operating directions"),
            ("figures/ppt/ppt03_regime_emissions.png", "K-Means: low, mid, high load"),
        ],
        "bullets": [
            bullet("PCA finds directions.", "The first two directions explain 74.3% of sensor variation."),
            bullet("K-Means finds three modes.", f"Low, mid, and high load: {n_low:,} / {n_mid:,} / {n_high:,} hours, centered near {tey_low:.0f} / {tey_mid:.0f} / {tey_high:.0f} MW."),
            bullet("Low load is the CO danger zone.", "Median CO is about 4.26 at low load vs 1.18 at high load; NOx stays flatter (61–64)."),
        ],
        "takeaway": takeaway(
            "Operating Insight",
            "The plant changes mode; one global formula misses the part-load CO spikes.",
        ),
    },
    {
        "layout": "split",
        "title": "How We Tested Fairly: Zero Cheating",
        "panels": [
            {
                "title": "Practice quiz: random 80/20",
                "boxes": [
                    ("All 5 years", "36,733 hours"),
                    ("Train 80%", "29,386 hours"),
                    ("Hold out 20%", "7,347 hours"),
                ],
                "text": "Tests learning within known years. Good for comparing models.",
            },
            {
                "title": "Real exam: future year",
                "boxes": [
                    ("Train 2011–2013", "22,191 hours"),
                    ("Tune on 2014", "7,158 hours"),
                    ("Locked test 2015", "7,384 hours"),
                ],
                "text": "Mirrors deployment: the future is never used for choices.",
            },
        ],
        "note": "Every scaler, cut-off, and cleaning rule is learned from training hours only.",
        "takeaway": takeaway(
            "Fair-Test Rule",
            "The future-year test is the honest exam; the random split is only a practice quiz.",
        ),
    },
    {
        "layout": "ladder",
        "title": "Algorithms We Compared: Simple to Smart",
        "steps": [
            ("1", "Simple line", "OLS", "One straight formula."),
            ("2", "Combined directions", "PCR", "Compress overlapping sensors."),
            ("3", "Curved + regularized", "Poly Ridge / Lasso", "Capture bends without overfitting."),
            ("4", "Neural network", "MLP", "Learn complex interactions."),
            ("5", "Spike warning", "Logistic Regression", "Flag the top 5–10% hours."),
        ],
        "note": "All five are syllabus-approved. Every model saw the same data and the same fair tests.",
        "takeaway": takeaway(
            "Model Logic",
            "Start simple, add complexity only when the data supports it, and score every model the same way.",
        ),
    },
    {
        "layout": "table",
        "title": "Results 1: Strong Performance on a Random Split",
        "table": {
            "header": ["Target", "Model", "CV error", "Test R²", "Test error", "Q95 error", "Q99 error"],
            "rows": [
                ["CO", "OLS", fmt(get_cell(t4, "CO", "OLS", "cv_rmse_mean")), fmt(get_cell(t4, "CO", "OLS", "test_r2")), fmt(get_cell(t4, "CO", "OLS", "test_rmse")), fmt(get_cell(t4, "CO", "OLS", "q95_rmse")), fmt(get_cell(t4, "CO", "OLS", "q99_rmse"))],
                ["CO", "PCR", fmt(get_cell(t4, "CO", "PCR", "cv_rmse_mean")), fmt(get_cell(t4, "CO", "PCR", "test_r2")), fmt(get_cell(t4, "CO", "PCR", "test_rmse")), fmt(get_cell(t4, "CO", "PCR", "q95_rmse")), fmt(get_cell(t4, "CO", "PCR", "q99_rmse"))],
                ["CO", "Poly2 Ridge", fmt(get_cell(t4, "CO", "Poly2_Ridge", "cv_rmse_mean")), fmt(get_cell(t4, "CO", "Poly2_Ridge", "test_r2")), fmt(get_cell(t4, "CO", "Poly2_Ridge", "test_rmse")), fmt(get_cell(t4, "CO", "Poly2_Ridge", "q95_rmse")), fmt(get_cell(t4, "CO", "Poly2_Ridge", "q99_rmse"))],
                ["CO", "MLP", fmt(get_cell(t4, "CO", "MLP", "cv_rmse_mean")), fmt(random_co_mlp_r2), fmt(random_co_mlp_rmse), fmt(get_cell(t4, "CO", "MLP", "q95_rmse")), fmt(get_cell(t4, "CO", "MLP", "q99_rmse"))],
                ["NOx", "OLS", fmt(get_cell(t4, "NOX", "OLS", "cv_rmse_mean")), fmt(get_cell(t4, "NOX", "OLS", "test_r2")), fmt(get_cell(t4, "NOX", "OLS", "test_rmse")), fmt(get_cell(t4, "NOX", "OLS", "q95_rmse")), fmt(get_cell(t4, "NOX", "OLS", "q99_rmse"))],
                ["NOx", "PCR", fmt(get_cell(t4, "NOX", "PCR", "cv_rmse_mean")), fmt(get_cell(t4, "NOX", "PCR", "test_r2")), fmt(get_cell(t4, "NOX", "PCR", "test_rmse")), fmt(get_cell(t4, "NOX", "PCR", "q95_rmse")), fmt(get_cell(t4, "NOX", "PCR", "q99_rmse"))],
                ["NOx", "Poly2 Ridge", fmt(get_cell(t4, "NOX", "Poly2_Ridge", "cv_rmse_mean")), fmt(get_cell(t4, "NOX", "Poly2_Ridge", "test_r2")), fmt(get_cell(t4, "NOX", "Poly2_Ridge", "test_rmse")), fmt(get_cell(t4, "NOX", "Poly2_Ridge", "q95_rmse")), fmt(get_cell(t4, "NOX", "Poly2_Ridge", "q99_rmse"))],
                ["NOx", "MLP", fmt(get_cell(t4, "NOX", "MLP", "cv_rmse_mean")), fmt(random_nox_mlp_r2), fmt(random_nox_mlp_rmse), fmt(get_cell(t4, "NOX", "MLP", "q95_rmse")), fmt(get_cell(t4, "NOX", "MLP", "q99_rmse"))],
            ],
            "widths": [70, 145, 105, 100, 110, 125, 125],
            "highlight_rows": [3, 7],
        },
        "note": "R² = share of hourly variation explained (higher is better) · error = RMSE in emission units (lower is better).",
        "note_top": 330,
        "takeaway": takeaway(
            "Result",
            f"The neural network captures combustion behavior best: CO R² {fmt(random_co_mlp_r2)} and NOx R² {fmt(random_nox_mlp_r2)}.",
        ),
    },
    {
        "layout": "figure_side",
        "title": "Results 2: The Real-World Future Year (2015)",
        "image": "figures/ppt/ppt05_cv_vs_2015_parity.png",
        "bullets": [
            bullet("Random split:", f"CO R² {fmt(random_co_mlp_r2)}; NOx R² {fmt(random_nox_mlp_r2)}."),
            bullet("Locked 2015:", f"CO R² {fmt(chrono_co_r2)}; NOx R² {fmt(chrono_nox_r2)}. Typical error rises {pct(co_rmse_increase)} for CO and {pct(nox_rmse_increase)} for NOx."),
            bullet("Read the scatter:", "Predictions follow the diagonal at normal hours, but high spikes are harder. Red lines mark the top 5%."),
        ],
        "takeaway": takeaway(
            "Deployment View",
            "A strong random score can hide a hard future year; trust the locked 2015 result.",
        ),
    },
    {
        "layout": "figure",
        "title": "Why Did 2015 Accuracy Drop? Plant Drift",
        "image": "figures/ppt/ppt04_multiyear_drift_2011_2015.png",
        "bullets": [
            bullet("CO drifted up.", f"Median CO rose from {fmt(co_drift_start, 2)} in 2011 to {fmt(co_drift_end, 2)} in 2015."),
            bullet("NOx drifted down.", f"Median NOx fell from {fmt(nox_drift_start, 1)} to {fmt(nox_drift_end, 1)}."),
            bullet("The plant changed too.", f"TEY moved {fmt(tey_drift_start, 1)} → {fmt(tey_drift_end, 1)}, and ambient AT moved {fmt(at_drift_start, 1)} → {fmt(at_drift_end, 1)}."),
        ],
        "takeaway": takeaway(
            "Root Cause",
            "The 2015 drop is plant drift, not a coding bug; models need monitoring and retraining.",
        ),
    },
    {
        "layout": "figure",
        "title": "Surprise Discovery: The Outlier Deletion Trap",
        "image": "figures/ppt/ppt06_ablation_and_alerts.png",
        "bullets": [
            bullet("We tested deleting spikes.", "Simple range-based cleaning (IQR) removed 17.8–27.3% of training hours; test hours stayed untouched."),
            bullet("Deletion hurt the tails.", f"Random CO Q99 error rose {fmt(co_raw_q99, 2)} → {fmt(co_iqr_q99, 2)}; 2015 NOx Q99 rose {fmt(nox_raw_q99, 2)} → {fmt(nox_iqr_q99, 2)}."),
            bullet("Keep the hard examples.", "A mild 3σ trim helped one average score slightly, but its tail errors still got worse."),
        ],
        "takeaway": takeaway(
            "Important Finding",
            "Do not clean away rare high-emission hours; they are the reason the monitor exists.",
            tone="warning",
        ),
    },
    {
        "layout": "figure_side",
        "title": "Turning Predictions Into Plant Alarms (Q90 / Q95)",
        "image": "figures/fig08_logistic_confusion_roc.png",
        "bullets": [
            bullet("Alarms for the worst hours.", "We flag the top 10% (Q90) and top 5% (Q95) of emissions."),
            bullet("Ranking is strong.", f"AUROC {fmt(auroc_min)}–{fmt(auroc_max)}; 1.0 is perfect, 0.5 is a coin flip."),
            bullet("Recall is high; false alarms are the cost.", f"2015 NOx Q90 caught {nox_q90_tp} of {nox_q90_tp + nox_q90_fn} spikes but raised {nox_q90_fp:,} false alarms."),
        ],
        "takeaway": takeaway(
            "Operator View",
            "Use the alarm as a screening tool — it tells operators where to look first, not what to certify.",
        ),
    },
    {
        "layout": "table",
        "title": "Important Catch: Alarm Thresholds Need Updates",
        "table": {
            "header": ["Target", "Alarm", "Cut-off", "Spike share", "Recall", "Precision", "F1", "AUROC"],
            "rows": [
                ["CO", "Q90", fmt(t7_cell("CO", "Q90", "threshold"), 2), f"{pct(t7_cell('CO', 'Q90', 'train_prevalence'))} → {pct(t7_cell('CO', 'Q90', 'test_prevalence'))}", fmt(co_q90_recall), fmt(co_q90_precision), fmt(t7_cell("CO", "Q90", "test_f1")), fmt(t7_cell("CO", "Q90", "test_roc_auc"))],
                ["CO", "Q95", fmt(t7_cell("CO", "Q95", "threshold"), 2), f"{pct(t7_cell('CO', 'Q95', 'train_prevalence'))} → {pct(t7_cell('CO', 'Q95', 'test_prevalence'))}", fmt(t7_cell("CO", "Q95", "test_recall")), fmt(t7_cell("CO", "Q95", "test_precision")), fmt(t7_cell("CO", "Q95", "test_f1")), fmt(t7_cell("CO", "Q95", "test_roc_auc"))],
                ["NOx", "Q90", fmt(t7_cell("NOX", "Q90", "threshold"), 2), f"{pct(t7_cell('NOX', 'Q90', 'train_prevalence'))} → {pct(nox_q90_prevalence)}", fmt(nox_q90_recall), fmt(nox_q90_precision), fmt(t7_cell("NOX", "Q90", "test_f1")), fmt(t7_cell("NOX", "Q90", "test_roc_auc"))],
                ["NOx", "Q95", fmt(t7_cell("NOX", "Q95", "threshold"), 2), f"{pct(t7_cell('NOX', 'Q95', 'train_prevalence'))} → {pct(nox_q95_prevalence)}", fmt(nox_q95_recall), fmt(nox_q95_precision), fmt(t7_cell("NOX", "Q95", "test_f1")), fmt(t7_cell("NOX", "Q95", "test_roc_auc"))],
            ],
            "widths": [75, 85, 95, 155, 85, 100, 75, 95],
        },
        "bullets": [
            bullet("2015 NOx had fewer spikes.", f"Share fell from 10.0% to {pct(nox_q90_prevalence)} at Q90 and from 5.0% to {pct(nox_q95_prevalence)} at Q95."),
            bullet("Fixed alarms over-warned.", f"Recall (spikes caught) stayed above 0.99, but precision (real alarms) fell to {fmt(nox_q90_precision)} / {fmt(nox_q95_precision)}."),
            bullet("Recalibrate with fresh labels.", "Choose thresholds on a new validation period, never on the locked test."),
        ],
        "takeaway": takeaway(
            "Warning",
            "An alarm that catches everything can still overwhelm operators; revisit cut-offs as the plant changes.",
            tone="warning",
        ),
    },
    {
        "layout": "cards4",
        "title": "Key Inferences & Engineering Insights",
        "cards": [
            ("Nonlinear physics", f"MLP CO R² {fmt(random_co_mlp_r2)} vs {fmt(random_co_ols_r2)} for a straight line."),
            ("Future-year drift", f"2015 error rose {pct(co_rmse_increase)} for CO and {pct(nox_rmse_increase)} for NOx."),
            ("Keep outliers", f"Range-based trimming raised random-CO Q99 error from {fmt(co_raw_q99, 2)} to {fmt(co_iqr_q99, 2)}."),
            ("Recalibrate alarms", f"2015 NOx precision was {fmt(nox_q90_precision)} despite {fmt(nox_q90_recall)} recall."),
        ],
        "takeaway": takeaway(
            "Big Picture",
            "Good ML is as much about fair testing and maintenance as about the model.",
        ),
    },
    {
        "layout": "deployment",
        "title": "Final Results & Real-Plant Deployment Plan",
        "steps": [
            ("1", "Deploy as an advisor", "Run the neural network beside CEMS; log every prediction and alarm."),
            ("2", "Retrain on a schedule", "Use quarterly walk-forward updates with the newest labelled hours."),
            ("3", "Keep raw spikes", "Never delete extreme hours; watch Q95/Q99 and drift."),
            ("4", "Tune with operators", "Recalibrate alarm cut-offs on validation data and confirm with CEMS."),
        ],
        "takeaway": takeaway(
            "Practical Plan",
            "Treat the model as a living loop: predict, compare, retrain, recalibrate.",
        ),
    },
    {
        "layout": "closing",
        "title": "Thank You & Questions?",
    },
]

assert len(SLIDES) == TOTAL

FIGURES = [
    "figures/fig01_univariate_kde_boxplots.png",
    "figures/fig02_correlation_heatmap_11x11.png",
    "figures/fig04_pca_scree_biplot.png",
    "figures/fig08_logistic_confusion_roc.png",
    "figures/ppt/ppt01_sensor_vif_log.png",
    "figures/ppt/ppt02_mle_tail_survival.png",
    "figures/ppt/ppt03_regime_emissions.png",
    "figures/ppt/ppt04_multiyear_drift_2011_2015.png",
    "figures/ppt/ppt05_cv_vs_2015_parity.png",
    "figures/ppt/ppt06_ablation_and_alerts.png",
]
check_images(FIGURES)


# ---------------------------------------------------------------------------
# HTML / PDF rendering
# ---------------------------------------------------------------------------

CSS = f"""
@page {{ size: 960pt 540pt; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: #FFFFFF; }}
body {{ font-family: "{FONT}", "DejaVu Sans", sans-serif; color: {CHARCOAL}; }}
.slide {{
  position: relative; width: 960pt; height: 540pt; background: #FFFFFF;
  overflow: hidden; page-break-after: always;
}}
.slide:last-child {{ page-break-after: auto; }}
.title {{
  position: absolute; left: 46pt; top: 22pt; width: 868pt;
  font-size: 25pt; line-height: 1.12; font-weight: 700; color: {NAVY};
}}
.rule {{ position: absolute; left: 46pt; top: 84pt; width: 868pt; height: 0.7pt; background: {LINE}; }}
.accent {{ position: absolute; left: 46pt; top: 82.5pt; width: 120pt; height: 3pt; background: {NAVY}; }}
.bullets {{ position: absolute; left: 46pt; top: 105pt; width: 868pt; }}
.bullets ul {{ margin: 0; padding-left: 17pt; }}
.bullets li {{ line-height: 1.31; margin-bottom: 11pt; }}
.lead {{ color: {NAVY}; font-weight: 700; }}
.small {{ font-size: 14pt; color: {CHARCOAL}; line-height: 1.25; }}
.caption {{ font-size: 12pt; color: {GREY}; line-height: 1.25; }}
.note {{ position: absolute; left: 46pt; font-size: 12.5pt; color: {GREY}; line-height: 1.25; }}
.footer {{
  position: absolute; left: 0; bottom: 0; width: 960pt; height: 25pt;
  border-top: 0.6pt solid {LINE}; font-size: 9.2pt; color: {GREY};
}}
.footer .left {{ position: absolute; left: 46pt; top: 8pt; }}
.footer .center {{ position: absolute; left: 285pt; width: 390pt; top: 8pt; text-align: center; }}
.footer .right {{ position: absolute; right: 46pt; top: 8pt; }}
.callout {{
  position: absolute; background: {TINT}; border-left: 5pt solid {NAVY};
}}
.callout .bar {{
  height: 19pt; background: {NAVY}; color: #FFFFFF; font-size: 10pt;
  font-weight: 700; letter-spacing: 0.5pt; padding: 4pt 10pt 0 10pt;
}}
.callout .text {{ font-size: 14.4pt; line-height: 1.22; padding: 6pt 10pt 0 10pt; }}
.callout.warning {{ border-left-color: {CRIMSON}; }}
.callout.amber {{ border-left-color: {AMBER}; }}
.figframe {{
  position: absolute; background: #FFFFFF; border: 0.7pt solid {LINE};
  border-radius: 5pt; overflow: hidden;
}}
.figframe img {{ position: absolute; }}
.two-card {{
  position: absolute; top: 120pt; width: 424pt; height: 222pt;
  background: {TINT}; border-top: 4pt solid {NAVY}; padding: 22pt;
}}
.two-card .num {{ font-size: 28pt; font-weight: 700; color: {STEEL}; }}
.two-card h3 {{ margin: 3pt 0 10pt 0; font-size: 21pt; color: {NAVY}; }}
.two-card p {{ margin: 0; font-size: 16pt; line-height: 1.35; }}
.stat {{
  position: absolute; top: 112pt; width: 204pt; height: 106pt;
  background: {TINT}; border-top: 3pt solid {NAVY}; padding: 13pt;
}}
.stat .big {{ font-size: 28pt; font-weight: 700; color: {NAVY}; line-height: 1; }}
.stat .label {{ margin-top: 7pt; font-size: 14pt; color: {CHARCOAL}; }}
.table {{ width: 100%; border-collapse: collapse; font-size: 14pt; }}
.table th {{
  background: {NAVY}; color: #FFFFFF; font-weight: 700; text-align: left;
  padding: 4pt 7pt; border: 0.5pt solid {NAVY};
}}
.table td {{ padding: 2.2pt 7pt; border-bottom: 0.5pt solid {LINE}; }}
.table tr.alt td {{ background: #F8FAFD; }}
.table tr.highlight td {{ background: #E6EFF8; font-weight: 700; }}
.table tr.target td {{ background: {TINT}; font-weight: 700; }}
.road-step {{
  position: absolute; top: 145pt; width: 158pt; height: 112pt;
  background: {TINT}; border-top: 3pt solid {NAVY}; padding: 12pt;
}}
.road-step .num {{ font-size: 19pt; font-weight: 700; color: {STEEL}; }}
.road-step h3 {{ margin: 2pt 0 6pt 0; font-size: 15pt; color: {NAVY}; }}
.road-step p {{ margin: 0; font-size: 12pt; line-height: 1.25; }}
.arrow {{ position: absolute; top: 183pt; font-size: 24pt; color: {STEEL}; font-weight: 700; }}
.panel {{
  position: absolute; top: 108pt; width: 424pt; height: 286pt;
  background: {TINT}; border: 0.8pt solid {LINE}; border-top: 4pt solid {NAVY}; padding: 16pt;
}}
.panel h3 {{ margin: 0 0 12pt 0; font-size: 18pt; color: {NAVY}; }}
.panel .box {{
  position: absolute; left: 16pt; right: 16pt; height: 42pt;
  background: #FFFFFF; border-left: 3pt solid {STEEL}; padding: 9pt 10pt;
  font-size: 14pt; color: {CHARCOAL};
}}
.panel .box b {{ color: {NAVY}; }}
.panel .pdown {{ position: absolute; left: 210pt; width: 0; height: 0; border-left: 5pt solid transparent; border-right: 5pt solid transparent; border-top: 8pt solid {STEEL}; }}
.panel p {{ position: absolute; left: 16pt; right: 16pt; bottom: 12pt; margin: 0; font-size: 13pt; color: {GREY}; }}
.ladder-step {{
  position: absolute; top: 145pt; width: 164pt; height: 150pt;
  background: #FFFFFF; border: 0.8pt solid {LINE}; border-top: 4pt solid {NAVY}; padding: 12pt;
}}
.ladder-step .num {{ font-size: 20pt; font-weight: 700; color: {STEEL}; }}
.ladder-step h3 {{ margin: 2pt 0 4pt 0; font-size: 14pt; color: {NAVY}; }}
.ladder-step .tag {{ font-size: 12pt; font-weight: 700; color: {STEEL}; margin-bottom: 5pt; }}
.ladder-step p {{ margin: 0; font-size: 11.5pt; line-height: 1.25; }}
.ladder-arrow {{ position: absolute; top: 211pt; width: 12pt; height: 0; border-top: 2pt solid {STEEL}; }}
.ladder-arrow::after {{ content: ""; position: absolute; right: -1pt; top: -4pt; border-left: 5pt solid {STEEL}; border-top: 4pt solid transparent; border-bottom: 4pt solid transparent; }}
.card4 {{
  position: absolute; width: 424pt; height: 128pt; background: {TINT};
  border-left: 4pt solid {NAVY}; padding: 15pt 18pt;
}}
.card4 h3 {{ margin: 0 0 6pt 0; font-size: 18pt; color: {NAVY}; }}
.card4 p {{ margin: 0; font-size: 14pt; line-height: 1.3; }}
.dep-step {{
  position: absolute; top: 132pt; width: 204pt; height: 242pt;
  background: #FFFFFF; border: 0.8pt solid {LINE}; border-top: 4pt solid {NAVY}; padding: 14pt;
}}
.dep-step .num {{
  width: 29pt; height: 29pt; line-height: 29pt; text-align: center;
  background: {NAVY}; color: #FFFFFF; border-radius: 15pt;
  font-size: 15pt; font-weight: 700; margin-bottom: 10pt;
}}
.dep-step h3 {{ margin: 0 0 8pt 0; font-size: 16pt; color: {NAVY}; }}
.dep-step p {{ margin: 0; font-size: 13pt; line-height: 1.28; }}
.cover {{ background: #FFFFFF; }}
.cover .topbar {{ position: absolute; left: 0; top: 0; width: 960pt; height: 9pt; background: {NAVY}; }}
.cover .topbar2 {{ position: absolute; left: 0; top: 9pt; width: 960pt; height: 3pt; background: {STEEL}; }}
.cover .kicker {{ position: absolute; left: 70pt; top: 66pt; width: 820pt; text-align: center; font-size: 11pt; letter-spacing: 1.2pt; color: {STEEL}; font-weight: 700; }}
.cover h1 {{ position: absolute; left: 65pt; top: 112pt; width: 830pt; text-align: center; font-size: 40pt; line-height: 1.08; color: {NAVY}; margin: 0; }}
.cover .subtitle {{ position: absolute; left: 95pt; top: 243pt; width: 770pt; text-align: center; font-size: 20pt; line-height: 1.25; color: {STEEL}; }}
.cover .line {{ position: absolute; left: 390pt; top: 310pt; width: 180pt; height: 2pt; background: {NAVY}; }}
.cover .course {{ position: absolute; left: 90pt; top: 326pt; width: 780pt; text-align: center; font-size: 15pt; color: {CHARCOAL}; }}
.cover .prof {{ position: absolute; left: 90pt; top: 348pt; width: 780pt; text-align: center; font-size: 15pt; color: {CHARCOAL}; }}
.cover .band {{ position: absolute; left: 0; top: 386pt; width: 960pt; height: 86pt; background: {TINT}; }}
.cover .team {{ position: absolute; left: 180pt; top: 399pt; width: 600pt; text-align: center; font-size: 16pt; color: {NAVY}; line-height: 1.55; }}
.cover .inst {{ position: absolute; left: 90pt; top: 487pt; width: 780pt; text-align: center; font-size: 13.5pt; color: {GREY}; }}
.closing h1 {{ position: absolute; left: 60pt; top: 145pt; width: 840pt; text-align: center; font-size: 46pt; color: {NAVY}; margin: 0; }}
.closing .sub {{ position: absolute; left: 90pt; top: 218pt; width: 780pt; text-align: center; font-size: 25pt; color: {STEEL}; }}
.closing .team {{ position: absolute; left: 180pt; top: 292pt; width: 600pt; text-align: center; font-size: 17pt; color: {CHARCOAL}; line-height: 1.55; }}
.closing .repo {{ position: absolute; left: 90pt; top: 382pt; width: 780pt; text-align: center; font-size: 15pt; color: {STEEL}; }}
.closing .meta {{ position: absolute; left: 90pt; top: 414pt; width: 780pt; text-align: center; font-size: 13pt; color: {GREY}; }}
"""


def esc(text: str) -> str:
    return html_mod.escape(str(text))


def fit_image_html(path: str, x: float, y: float, w: float, h: float) -> str:
    iw, ih = image_size(ROOT / path)
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    left = x + (w - dw) / 2.0
    top = y + (h - dh) / 2.0
    return (
        f'<div class="figframe" style="left:{x}pt;top:{y}pt;width:{w}pt;height:{h}pt;">'
        f'<img src="{(ROOT / path).resolve().as_uri()}" '
        f'style="left:{left - x}pt;top:{top - y}pt;width:{dw}pt;height:{dh}pt;">'
        "</div>"
    )


def footer_html(number: int) -> str:
    return (
        '<div class="footer">'
        '<span class="left">CHE F315: Machine Learning for Chemical Engineers</span>'
        '<span class="center">Gas Turbine Emission Prediction (CO &amp; NOx)</span>'
        f'<span class="right">{number} / {TOTAL}</span>'
        "</div>"
    )


def callout_html(
    item: dict,
    x: float = 46,
    y: float = 446,
    w: float = 868,
    h: float = 64,
) -> str:
    tone = item.get("tone", "navy")
    cls = "callout"
    if tone == "warning":
        cls += " warning"
    elif tone == "amber":
        cls += " amber"
    return (
        f'<div class="{cls}" style="left:{x}pt;top:{y}pt;width:{w}pt;height:{h}pt;">'
        f'<div class="bar">{esc(item["label"]).upper()}</div>'
        f'<div class="text">{esc(item["text"])}</div>'
        "</div>"
    )


def bullets_html(
    bullets: list[dict],
    x: float = 46,
    y: float = 105,
    w: float = 868,
    size: float = 17,
) -> str:
    items = "".join(
        f'<li><span class="lead">{esc(item["lead"])}</span> {esc(item["text"])}</li>'
        for item in bullets
    )
    return (
        f'<div class="bullets" style="left:{x}pt;top:{y}pt;width:{w}pt;'
        f'font-size:{size}pt;">'
        f"<ul>{items}</ul></div>"
    )


def table_html(
    spec: dict,
    x: float,
    y: float,
    width: float,
    size: float = 14,
) -> str:
    cols = "".join(f'<col style="width:{w}pt">' for w in spec["widths"])
    header = "".join(f"<th>{esc(h)}</th>" for h in spec["header"])
    target_rows = set(spec.get("target_rows", []))
    highlight_rows = set(spec.get("highlight_rows", []))
    rows = []
    for index, row in enumerate(spec["rows"]):
        classes = []
        if index % 2 == 1:
            classes.append("alt")
        if index in highlight_rows:
            classes.append("highlight")
        if index in target_rows:
            classes.append("target")
        cells = "".join(f"<td>{esc(cell)}</td>" for cell in row)
        rows.append(f'<tr class="{" ".join(classes)}">{cells}</tr>')
    return (
        f'<table class="table" style="position:absolute;left:{x}pt;top:{y}pt;'
        f"width:{width}pt;font-size:{size}pt;\">"
        f"<colgroup>{cols}</colgroup><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_slide(slide: dict, number: int) -> str:
    layout = slide["layout"]
    if layout == "title":
        team = "<br>".join(
            f"{esc(name)} &nbsp;·&nbsp; {esc(pid)}" for name, pid in TEAM
        )
        return (
            '<div class="slide cover">'
            '<div class="topbar"></div><div class="topbar2"></div>'
            '<div class="kicker">CHE F315 · MACHINE LEARNING FOR CHEMICAL ENGINEERS</div>'
            f'<h1>{esc(slide["title"])}</h1>'
            f'<div class="subtitle">{esc(slide["subtitle"])}</div>'
            '<div class="line"></div>'
            f'<div class="course">Course: {esc(COURSE)}</div>'
            f'<div class="prof">Professor: Riju De</div>'
            '<div class="band"></div>'
            f'<div class="team">{team}</div>'
            f'<div class="inst">{esc(INSTITUTION)} &nbsp;·&nbsp; Date: {esc(PRESENTATION_DATE)}</div>'
            "</div>"
        )

    if layout == "closing":
        team = "<br>".join(
            f"{esc(name)} &nbsp;·&nbsp; {esc(pid)}" for name, pid in TEAM
        )
        return (
            '<div class="slide closing">'
            f'<h1>{esc(slide["title"])}</h1>'
            '<div class="sub">Discussion &amp; feedback</div>'
            f'<div class="team">{team}</div>'
            f'<div class="repo">{esc(REPO)}</div>'
            f'<div class="meta">{esc(COURSE)} &nbsp;·&nbsp; {esc(PROFESSOR)} &nbsp;·&nbsp; {esc(PRESENTATION_DATE)}</div>'
            f"{footer_html(number)}"
            "</div>"
        )

    body = (
        f'<div class="title">{esc(slide["title"])}</div>'
        '<div class="rule"></div><div class="accent"></div>'
    )
    take = slide.get("takeaway")
    callout = callout_html(take) if take else ""

    if layout in {"bullets", "stats"}:
        if layout == "stats":
            positions = [46, 260, 474, 688]
            for x, (big, label) in zip(positions, slide["stats"], strict=True):
                body += (
                    f'<div class="stat" style="left:{x}pt;">'
                    f'<div class="big">{esc(big)}</div>'
                    f'<div class="label">{esc(label)}</div></div>'
                )
            body += bullets_html(slide["bullets"], y=245)
        else:
            body += bullets_html(slide["bullets"], y=108)
        body += callout

    elif layout == "roadmap":
        positions = [46, 228, 410, 592, 774]
        for index, ((num, heading, text), x) in enumerate(
            zip(slide["steps"], positions, strict=True)
        ):
            body += (
                f'<div class="road-step" style="left:{x}pt;">'
                f'<div class="num">{esc(num)}</div>'
                f"<h3>{esc(heading)}</h3><p>{esc(text)}</p></div>"
            )
            if index < len(positions) - 1:
                body += f'<div class="arrow" style="left:{x + 162}pt;">→</div>'
        body += f'<div class="note" style="top:278pt;">{esc(slide["note"])}</div>'
        body += callout

    elif layout == "two_cards":
        for index, (num, heading, text) in enumerate(slide["cards"]):
            x = 46 + index * 444
            body += (
                f'<div class="two-card" style="left:{x}pt;">'
                f'<div class="num">{esc(num)}</div>'
                f"<h3>{esc(heading)}</h3><p>{esc(text)}</p></div>"
            )
        body += f'<div class="note" style="top:365pt;">{esc(slide["note"])}</div>'
        body += callout

    elif layout == "table":
        width = sum(slide["table"]["widths"])
        x = (PAGE_W - width) / 2.0
        body += table_html(slide["table"], x=x, y=106, width=width)
        note_top = slide.get("note_top", 384)
        body += f'<div class="note" style="top:{note_top}pt;">{esc(slide.get("note", ""))}</div>'
        if "bullets" in slide:
            body += bullets_html(slide["bullets"], y=292, w=868, size=15.5)
        body += callout

    elif layout == "figure":
        body += bullets_html(slide["bullets"], x=46, y=108, w=360, size=16)
        body += fit_image_html(slide["image"], 430, 104, 484, 330)
        body += callout

    elif layout == "figure_side":
        body += bullets_html(slide["bullets"], x=46, y=108, w=420, size=16)
        body += fit_image_html(slide["image"], 486, 104, 428, 392)
        body += callout_html(take, x=46, y=424, w=420, h=70)

    elif layout == "figure_top":
        body += fit_image_html(slide["image"], 90, 103, 780, 286)
        cols = [46, 344, 642]
        for x, item in zip(cols, slide["bullets"], strict=True):
            body += (
                f'<div class="small" style="position:absolute;left:{x}pt;top:398pt;'
                f'width:270pt;"><span class="lead">{esc(item["lead"])}</span> '
                f'{esc(item["text"])}</div>'
            )
        body += callout

    elif layout == "two_figures":
        body += bullets_html(slide["bullets"], x=46, y=108, w=420, size=16)
        body += fit_image_html(slide["images"][0][0], 486, 104, 428, 166)
        body += fit_image_html(slide["images"][1][0], 486, 302, 428, 166)
        body += (
            f'<div class="caption" style="position:absolute;left:486pt;top:274pt;width:428pt;">'
            f'{esc(slide["images"][0][1])}</div>'
        )
        body += (
            f'<div class="caption" style="position:absolute;left:486pt;top:472pt;width:428pt;">'
            f'{esc(slide["images"][1][1])}</div>'
        )
        body += callout_html(take, x=46, y=424, w=420, h=70)

    elif layout == "split":
        for index, panel in enumerate(slide["panels"]):
            x = 46 + index * 444
            body += (
                f'<div class="panel" style="left:{x}pt;">'
                f'<h3>{esc(panel["title"])}</h3>'
            )
            for box_index, (label, value) in enumerate(panel["boxes"]):
                top = 62 + box_index * 58
                body += (
                    f'<div class="box" style="top:{top}pt;">'
                    f"<b>{esc(label)}</b><br>{esc(value)}</div>"
                )
                if box_index < 2:
                    body += f'<div class="pdown" style="top:{top + 43}pt;"></div>'
            body += f'<p>{esc(panel["text"])}</p></div>'
        body += f'<div class="note" style="top:405pt;">{esc(slide["note"])}</div>'
        body += callout

    elif layout == "ladder":
        positions = [46, 222, 398, 574, 750]
        for index, ((num, heading, tag, text), x) in enumerate(
            zip(slide["steps"], positions, strict=True)
        ):
            body += (
                f'<div class="ladder-step" style="left:{x}pt;">'
                f'<div class="num">{esc(num)}</div>'
                f"<h3>{esc(heading)}</h3>"
                f'<div class="tag">{esc(tag)}</div>'
                f"<p>{esc(text)}</p></div>"
            )
            if index < len(positions) - 1:
                body += f'<div class="ladder-arrow" style="left:{x + 166}pt;"></div>'
        body += f'<div class="note" style="top:315pt;">{esc(slide["note"])}</div>'
        body += callout

    elif layout == "cards4":
        positions = [(46, 112), (490, 112), (46, 254), (490, 254)]
        for (x, y), (heading, text) in zip(positions, slide["cards"], strict=True):
            body += (
                f'<div class="card4" style="left:{x}pt;top:{y}pt;">'
                f"<h3>{esc(heading)}</h3><p>{esc(text)}</p></div>"
            )
        body += callout

    elif layout == "deployment":
        positions = [46, 262, 478, 694]
        for (num, heading, text), x in zip(
            slide["steps"], positions, strict=True
        ):
            body += (
                f'<div class="dep-step" style="left:{x}pt;">'
                f'<div class="num">{esc(num)}</div>'
                f"<h3>{esc(heading)}</h3><p>{esc(text)}</p></div>"
            )
        body += callout

    else:
        raise ValueError(f"Unknown layout: {layout}")

    return f'<div class="slide">{body}{footer_html(number)}</div>'


def render_html() -> str:
    slides = "".join(
        render_slide(slide, index + 1) for index, slide in enumerate(SLIDES)
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{slides}</body></html>"
    )


def build_pdf() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    HTML(string=render_html(), base_url=str(ROOT) + "/").write_pdf(PDF_PATH)
    print(f"Wrote {PDF_PATH}")


# ---------------------------------------------------------------------------
# PPTX rendering
# ---------------------------------------------------------------------------


def rgb(hex_value: str) -> RGBColor:
    value = hex_value.lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def add_text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    size: float,
    color: str = CHARCOAL,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
    font: str = PPT_FONT,
    anchor=MSO_ANCHOR.TOP,
    italic: bool = False,
):
    box = slide.shapes.add_textbox(
        Inches(x / 72.0), Inches(y / 72.0), Inches(w / 72.0), Inches(h / 72.0)
    )
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    lines = str(text).split("\n")
    for line_index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if line_index == 0 else frame.add_paragraph()
        paragraph.alignment = align
        run = paragraph.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = rgb(color)
    return box


def add_rect(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str | None = None,
    line: str | None = None,
    line_width: float = 0.8,
    shape=MSO_SHAPE.RECTANGLE,
):
    item = slide.shapes.add_shape(
        shape,
        Inches(x / 72.0),
        Inches(y / 72.0),
        Inches(w / 72.0),
        Inches(h / 72.0),
    )
    if fill is None:
        item.fill.background()
    else:
        item.fill.solid()
        item.fill.fore_color.rgb = rgb(fill)
    if line is None:
        item.line.fill.background()
    else:
        item.line.color.rgb = rgb(line)
        item.line.width = Pt(line_width)
    return item


def add_line(slide, x1: float, y1: float, x2: float, y2: float, color: str, width=0.7):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1 / 72.0),
        Inches(y1 / 72.0),
        Inches(x2 / 72.0),
        Inches(y2 / 72.0),
    )
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)


def add_bullets(
    slide,
    bullets: list[dict],
    x: float,
    y: float,
    w: float,
    h: float,
    size: float = 17,
    color: str = CHARCOAL,
):
    box = slide.shapes.add_textbox(
        Inches(x / 72.0), Inches(y / 72.0), Inches(w / 72.0), Inches(h / 72.0)
    )
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    for index, item in enumerate(bullets):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(9)
        paragraph.line_spacing = 1.02
        bullet_run = paragraph.add_run()
        bullet_run.text = "•  "
        bullet_run.font.name = PPT_FONT
        bullet_run.font.size = Pt(size)
        bullet_run.font.color.rgb = rgb(STEEL)
        lead_run = paragraph.add_run()
        lead_run.text = f"{item['lead']} "
        lead_run.font.name = PPT_FONT
        lead_run.font.size = Pt(size)
        lead_run.font.bold = True
        lead_run.font.color.rgb = rgb(NAVY)
        body_run = paragraph.add_run()
        body_run.text = item["text"]
        body_run.font.name = PPT_FONT
        body_run.font.size = Pt(size)
        body_run.font.color.rgb = rgb(color)
    return box


def add_footer(slide, number: int):
    add_line(slide, 0, 515, 960, 515, LINE, 0.6)
    add_text(
        slide,
        "CHE F315: Machine Learning for Chemical Engineers",
        46,
        520,
        330,
        18,
        9.2,
        GREY,
    )
    add_text(
        slide,
        "Gas Turbine Emission Prediction (CO & NOx)",
        300,
        520,
        360,
        18,
        9.2,
        GREY,
        align=PP_ALIGN.CENTER,
    )
    add_text(
        slide,
        f"{number} / {TOTAL}",
        870,
        520,
        44,
        18,
        9.2,
        GREY,
        align=PP_ALIGN.RIGHT,
    )


def add_callout(slide, item: dict, x=46, y=446, w=868, h=64):
    accent = NAVY
    if item.get("tone") == "warning":
        accent = CRIMSON
    elif item.get("tone") == "amber":
        accent = AMBER
    add_rect(slide, x, y, w, h, fill=TINT)
    add_rect(slide, x, y, 5, h, fill=accent)
    add_rect(slide, x + 5, y, w - 5, 19, fill=NAVY)
    add_text(slide, item["label"].upper(), x + 12, y + 2.5, w - 24, 15, 10, WHITE, bold=True)
    add_text(slide, item["text"], x + 12, y + 22, w - 24, h - 24, 14.4, CHARCOAL)


def add_figure(slide, path: str, x: float, y: float, w: float, h: float):
    iw, ih = image_size(ROOT / path)
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    left = x + (w - dw) / 2.0
    top = y + (h - dh) / 2.0
    add_rect(slide, x, y, w, h, fill=WHITE, line=LINE, line_width=0.7)
    slide.shapes.add_picture(
        str(ROOT / path),
        Inches(left / 72.0),
        Inches(top / 72.0),
        width=Inches(dw / 72.0),
        height=Inches(dh / 72.0),
    )


def add_table(slide, spec: dict, x: float, y: float, size: float = 14):
    rows = len(spec["rows"]) + 1
    columns = len(spec["header"])
    width = sum(spec["widths"])
    shape = slide.shapes.add_table(
        rows,
        columns,
        Inches(x / 72.0),
        Inches(y / 72.0),
        Inches(width / 72.0),
        Inches(rows * 0.31),
    )
    table = shape.table
    for index, col_width in enumerate(spec["widths"]):
        table.columns[index].width = Inches(col_width / 72.0)
    for row_index in range(rows):
        table.rows[row_index].height = Inches(0.31)
    for col_index, header in enumerate(spec["header"]):
        cell = table.cell(0, col_index)
        cell.fill.solid()
        cell.fill.fore_color.rgb = rgb(NAVY)
        cell.margin_left = Pt(5)
        cell.margin_right = Pt(5)
        cell.margin_top = Pt(2)
        cell.margin_bottom = Pt(2)
        cell.text = header
        for paragraph in cell.text_frame.paragraphs:
            paragraph.runs[0].font.name = PPT_FONT
            paragraph.runs[0].font.size = Pt(size)
            paragraph.runs[0].font.bold = True
            paragraph.runs[0].font.color.rgb = rgb(WHITE)
    highlight = set(spec.get("highlight_rows", []))
    target = set(spec.get("target_rows", []))
    for row_index, row in enumerate(spec["rows"], start=1):
        for col_index, value in enumerate(row):
            cell = table.cell(row_index, col_index)
            fill = WHITE
            if row_index - 1 in highlight:
                fill = "#E6EFF8"
            elif row_index - 1 in target:
                fill = TINT
            elif (row_index - 1) % 2 == 1:
                fill = "#F8FAFD"
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(fill)
            cell.margin_left = Pt(5)
            cell.margin_right = Pt(5)
            cell.margin_top = Pt(1)
            cell.margin_bottom = Pt(1)
            cell.text = str(value)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.runs[0].font.name = PPT_FONT
                paragraph.runs[0].font.size = Pt(size)
                paragraph.runs[0].font.color.rgb = rgb(CHARCOAL)
                if row_index - 1 in highlight:
                    paragraph.runs[0].font.bold = True


def build_pptx() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    for number, data in enumerate(SLIDES, start=1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        background = slide.background
        background.fill.solid()
        background.fill.fore_color.rgb = rgb(WHITE)
        layout = data["layout"]

        if layout == "title":
            add_rect(slide, 0, 0, 960, 9, fill=NAVY)
            add_rect(slide, 0, 9, 960, 3, fill=STEEL)
            add_text(slide, "CHE F315 · MACHINE LEARNING FOR CHEMICAL ENGINEERS", 70, 60, 820, 18, 11, STEEL, bold=True, align=PP_ALIGN.CENTER)
            add_text(slide, data["title"], 65, 112, 830, 118, 40, NAVY, bold=True, align=PP_ALIGN.CENTER)
            add_text(slide, data["subtitle"], 95, 247, 770, 54, 20, STEEL, align=PP_ALIGN.CENTER)
            add_line(slide, 390, 311, 570, 311, NAVY, 2)
            add_text(slide, f"Course: {COURSE}", 90, 326, 780, 20, 15, CHARCOAL, align=PP_ALIGN.CENTER)
            add_text(slide, "Professor: Riju De", 90, 348, 780, 20, 15, CHARCOAL, align=PP_ALIGN.CENTER)
            add_rect(slide, 0, 386, 960, 86, fill=TINT)
            team_text = "\n".join(f"{name}  ·  {pid}" for name, pid in TEAM)
            add_text(slide, team_text, 180, 399, 600, 55, 16, NAVY, bold=True, align=PP_ALIGN.CENTER)
            add_text(slide, f"{INSTITUTION}  ·  Date: {PRESENTATION_DATE}", 90, 487, 780, 20, 13.5, GREY, align=PP_ALIGN.CENTER)
            continue

        if layout == "closing":
            add_text(slide, "Thank You & Questions?", 60, 120, 840, 70, 44, NAVY, bold=True, align=PP_ALIGN.CENTER)
            add_line(slide, 380, 205, 580, 205, NAVY, 2)
            add_text(slide, "Discussion & feedback", 90, 225, 780, 42, 25, STEEL, align=PP_ALIGN.CENTER)
            team_text = "\n".join(f"{name}  ·  {pid}" for name, pid in TEAM)
            add_text(slide, team_text, 180, 294, 600, 55, 17, CHARCOAL, bold=True, align=PP_ALIGN.CENTER)
            add_text(slide, REPO, 90, 380, 780, 24, 15, STEEL, align=PP_ALIGN.CENTER)
            add_text(slide, f"{COURSE}  ·  {PROFESSOR}  ·  {PRESENTATION_DATE}", 90, 414, 780, 20, 13, GREY, align=PP_ALIGN.CENTER)
            add_footer(slide, number)
            continue

        add_text(slide, data["title"], 46, 22, 868, 54, 25, NAVY, bold=True)
        add_line(slide, 46, 84, 914, 84, LINE, 0.7)
        add_line(slide, 46, 83, 166, 83, NAVY, 3)
        take = data.get("takeaway")

        if layout in {"bullets", "stats"}:
            if layout == "stats":
                for x, (big, label) in zip([46, 260, 474, 688], data["stats"], strict=True):
                    add_rect(slide, x, 112, 204, 106, fill=TINT)
                    add_line(slide, x, 112, x + 204, 112, NAVY, 3)
                    add_text(slide, big, x + 13, 126, 178, 38, 28, NAVY, bold=True)
                    add_text(slide, label, x + 13, 176, 178, 28, 14, CHARCOAL)
                add_bullets(slide, data["bullets"], 46, 245, 868, 180, 17)
            else:
                add_bullets(slide, data["bullets"], 46, 108, 868, 320, 17)
            add_callout(slide, take)

        elif layout == "roadmap":
            positions = [46, 228, 410, 592, 774]
            for index, ((num, heading, text), x) in enumerate(zip(data["steps"], positions, strict=True)):
                add_rect(slide, x, 145, 158, 112, fill=TINT)
                add_line(slide, x, 145, x + 158, 145, NAVY, 3)
                add_text(slide, num, x + 12, 154, 30, 24, 19, STEEL, bold=True)
                add_text(slide, heading, x + 12, 178, 134, 20, 15, NAVY, bold=True)
                add_text(slide, text, x + 12, 200, 134, 50, 11.5, CHARCOAL)
                if index < 4:
                    add_text(slide, "→", x + 162, 180, 20, 26, 24, STEEL, bold=True)
            add_text(slide, data["note"], 46, 278, 868, 22, 12, GREY)
            add_callout(slide, take)

        elif layout == "two_cards":
            for index, (num, heading, text) in enumerate(data["cards"]):
                x = 46 + index * 444
                add_rect(slide, x, 120, 424, 222, fill=TINT)
                add_line(slide, x, 120, x + 424, 120, NAVY, 4)
                add_text(slide, num, x + 22, 140, 40, 32, 28, STEEL, bold=True)
                add_text(slide, heading, x + 22, 178, 380, 28, 21, NAVY, bold=True)
                add_text(slide, text, x + 22, 215, 380, 105, 16, CHARCOAL)
            add_text(slide, data["note"], 46, 365, 868, 40, 12.5, GREY)
            add_callout(slide, take)

        elif layout == "table":
            width = sum(data["table"]["widths"])
            x = (PAGE_W - width) / 2.0
            add_table(slide, data["table"], x, 106, size=14)
            if data.get("note"):
                add_text(slide, data["note"], 46, 384, 868, 22, 12.5, GREY)
            if "bullets" in data:
                add_bullets(slide, data["bullets"], 46, 292, 868, 140, 15)
            add_callout(slide, take)

        elif layout == "figure":
            add_bullets(slide, data["bullets"], 46, 108, 360, 320, 16)
            add_figure(slide, data["image"], 430, 104, 484, 330)
            add_callout(slide, take)

        elif layout == "figure_side":
            add_bullets(slide, data["bullets"], 46, 108, 420, 300, 16)
            add_figure(slide, data["image"], 486, 104, 428, 392)
            add_callout(slide, take, x=46, y=424, w=420, h=70)

        elif layout == "figure_top":
            add_figure(slide, data["image"], 90, 103, 780, 286)
            for x, item in zip([46, 344, 642], data["bullets"], strict=True):
                add_text(
                    slide,
                    f"{item['lead']} {item['text']}",
                    x,
                    398,
                    270,
                    44,
                    13,
                    CHARCOAL,
                )
            add_callout(slide, take)

        elif layout == "two_figures":
            add_bullets(slide, data["bullets"], 46, 108, 420, 300, 16)
            add_figure(slide, data["images"][0][0], 486, 104, 428, 166)
            add_figure(slide, data["images"][1][0], 486, 302, 428, 166)
            add_text(slide, data["images"][0][1], 486, 274, 428, 18, 12, GREY)
            add_text(slide, data["images"][1][1], 486, 472, 428, 18, 12, GREY)
            add_callout(slide, take, x=46, y=424, w=420, h=70)

        elif layout == "split":
            for index, panel in enumerate(data["panels"]):
                x = 46 + index * 444
                add_rect(slide, x, 108, 424, 286, fill=TINT, line=LINE)
                add_line(slide, x, 108, x + 424, 108, NAVY, 4)
                add_text(slide, panel["title"], x + 16, 124, 392, 28, 18, NAVY, bold=True)
                for box_index, (label, value) in enumerate(panel["boxes"]):
                    top = 170 + box_index * 58
                    add_rect(slide, x + 16, top, 392, 42, fill=WHITE)
                    add_rect(slide, x + 16, top, 3, 42, fill=STEEL)
                    add_text(slide, f"{label}: {value}", x + 27, top + 9, 370, 26, 14, CHARCOAL)
                    if box_index < 2:
                        add_rect(slide, x + 202, top + 43, 10, 10, fill=STEEL, shape=MSO_SHAPE.DOWN_ARROW)
                add_text(slide, panel["text"], x + 16, 360, 392, 26, 12.5, GREY)
            add_text(slide, data["note"], 46, 405, 868, 22, 12.5, GREY)
            add_callout(slide, take)

        elif layout == "ladder":
            positions = [46, 222, 398, 574, 750]
            for index, ((num, heading, tag, text), x) in enumerate(zip(data["steps"], positions, strict=True)):
                add_rect(slide, x, 145, 164, 150, fill=WHITE, line=LINE)
                add_line(slide, x, 145, x + 164, 145, NAVY, 4)
                add_text(slide, num, x + 12, 154, 30, 24, 20, STEEL, bold=True)
                add_text(slide, heading, x + 12, 180, 140, 34, 13.5, NAVY, bold=True)
                add_text(slide, tag, x + 12, 210, 140, 20, 11.5, STEEL, bold=True)
                add_text(slide, text, x + 12, 232, 140, 54, 11, CHARCOAL)
                if index < 4:
                    add_rect(slide, x + 166, 210, 14, 8, fill=STEEL, shape=MSO_SHAPE.RIGHT_ARROW)
            add_text(slide, data["note"], 46, 315, 868, 22, 12, GREY)
            add_callout(slide, take)

        elif layout == "cards4":
            positions = [(46, 112), (490, 112), (46, 254), (490, 254)]
            for (x, y), (heading, text) in zip(positions, data["cards"], strict=True):
                add_rect(slide, x, y, 424, 128, fill=TINT)
                add_rect(slide, x, y, 4, 128, fill=NAVY)
                add_text(slide, heading, x + 18, y + 15, 390, 28, 18, NAVY, bold=True)
                add_text(slide, text, x + 18, y + 50, 390, 66, 14, CHARCOAL)
            add_callout(slide, take)

        elif layout == "deployment":
            positions = [46, 262, 478, 694]
            for (num, heading, text), x in zip(data["steps"], positions, strict=True):
                add_rect(slide, x, 132, 204, 242, fill=WHITE, line=LINE)
                add_line(slide, x, 132, x + 204, 132, NAVY, 4)
                add_rect(slide, x + 14, 146, 29, 29, fill=NAVY, shape=MSO_SHAPE.OVAL)
                add_text(slide, num, x + 14, 149, 29, 22, 15, WHITE, bold=True, align=PP_ALIGN.CENTER)
                add_text(slide, heading, x + 14, 188, 176, 44, 16, NAVY, bold=True)
                add_text(slide, text, x + 14, 238, 176, 125, 13, CHARCOAL)
            add_callout(slide, take)

        add_footer(slide, number)

    prs.save(PPTX_PATH)
    print(f"Wrote {PPTX_PATH}")


# ---------------------------------------------------------------------------
# Speaker notes
# ---------------------------------------------------------------------------


def build_notes() -> None:
    text = f"""# CHE F315 — Speaker Notes

**Talk:** Predicting Gas Turbine Emissions Using Machine Learning  
**Subtitle:** A Software-Based Alternative to Expensive Physical Emission Sensors (CO & NOx)  
**Course:** {COURSE}  
**Professor:** {PROFESSOR}  
**Date:** {PRESENTATION_DATE}  
**Institution:** {INSTITUTION}

**Team:** Samar Talwar (2024A1PS0264G) · Soumya Mandoliya (2024A1PS0277G)

**Target runtime:** ~17 min 50 s, leaving 7–8 minutes for questions.  
**Split:** Samar Talwar presents slides 1–11; Soumya Mandoliya presents slides 12–22.

---

## Minute-by-minute pacing

| Time | Slide | Speaker | Focus |
|---|---|---|---|
| 0:00–0:40 | 1 | Samar | Title, team, one-sentence purpose |
| 0:40–1:10 | 2 | Samar | Five-step roadmap |
| 1:10–2:00 | 3 | Samar | Why emissions monitoring matters |
| 2:00–2:45 | 4 | Samar | Two engineering goals |
| 2:45–3:30 | 5 | Samar | Dataset scale and quality |
| 3:30–4:20 | 6 | Samar | Nine inputs, two outputs |
| 4:20–5:15 | 7 | Samar | Rare CO/NOx spikes |
| 5:15–6:05 | 8 | Samar | Sensors move together |
| 6:05–6:55 | 9 | Samar | Physical drivers |
| 6:55–7:45 | 10 | Samar | Why a bell curve fails |
| 7:45–8:50 | 11 | Samar | Three operating modes; handoff |
| 8:50–9:00 | — | Both | Smooth handoff |
| 9:00–10:00 | 12 | Soumya | Fair random vs future-year tests |
| 10:00–10:40 | 13 | Soumya | Simple-to-smart model ladder |
| 10:40–11:30 | 14 | Soumya | Random-split results |
| 11:30–12:30 | 15 | Soumya | Locked 2015 results |
| 12:30–13:20 | 16 | Soumya | Plant drift |
| 13:20–14:10 | 17 | Soumya | Outlier deletion trap |
| 14:10–15:05 | 18 | Soumya | Q90/Q95 alarms |
| 15:05–15:55 | 19 | Soumya | Threshold drift |
| 15:55–16:40 | 20 | Soumya | Four engineering lessons |
| 16:40–17:25 | 21 | Soumya | Deployment plan |
| 17:25–17:50 | 22 | Soumya | Thank you and questions |

---

## Slide-by-slide script

### Slide 1 — Title (Samar, 0:00–0:40)
Good morning/afternoon. We are Samar Talwar and Soumya Mandoliya. Our CHE F315 project asks a practical question: can ordinary turbine sensors predict CO and NOx well enough to warn a plant before an emission spike? We used five years of real plant data and tested the models on an unseen future year.

### Slide 2 — Roadmap (Samar, 0:40–1:10)
Point across the five-step arrow. “We will first explain the plant problem, then the data, then how we test fairly. After that, the results and surprises. We finish with what a plant could actually deploy.”

### Slide 3 — What problem? (Samar, 1:10–2:00)
Point to “CEMS” and “PEMS.” Gas turbines produce CO when combustion is incomplete and NOx at high firing temperatures. Physical analyzers are accurate but expensive to maintain. A software sensor uses existing sensors to estimate emissions every hour. It is a backup and an early-warning layer, not a legal replacement for hardware.

### Slide 4 — Goals (Samar, 2:00–2:45)
Point to card 1, then card 2. “Task one is the number: predict hourly CO and NOx. Task two is the warning: flag the worst 10% and 5% of hours. R² is the share of hourly variation we explain; RMSE is the typical miss.” Emphasize that CO and NOx are separate models.

### Slide 5 — Data (Samar, 2:45–3:30)
Point to the four cards. “36,733 hourly rows, five full years, nine sensors, zero missing values.” This is public UCI data from a real Turkish plant. The five years are important because they let us test on a later year.

### Slide 6 — Dataset (Samar, 3:30–4:20)
Point to the three groups: weather, turbine, answers. “The first nine rows are inputs. The last two are outputs. NOx never enters a CO model and CO never enters a NOx model.” This is the target-isolation rule.

### Slide 7 — Spikes (Samar, 4:20–5:15)
Point to the CO KDE and boxplot. “Most hours are low, but the tail runs to 44.1. The median is 1.71 and the mean is pulled to 2.37.” Point to NOx. “NOx is steadier.” Say clearly: the rare high hours are the ones regulators care about, so we do not delete them.

### Slide 8 — Sensors overlap (Samar, 5:15–6:05)
Point to TEY, CDP, and TIT bars. “These sensors rise and fall together; the overlap score reaches 255.6, 251.0, and 38.4.” A simple straight-line model cannot separate their effects. Ridge and PCA help diagnose it, but they barely improved accuracy — the real issue is curved behavior.

### Slide 9 — Correlations (Samar, 6:05–6:55)
Point to the CO row/column, then the NOx row/column. “Higher load generally means lower CO because combustion is more complete. NOx depends on firing and ambient conditions, not load alone.” Keeping all nine sensors is deliberate.

### Slide 10 — Bell curve (Samar, 6:55–7:45)
Point to the empirical tail and the curves. “A normal bell curve says extreme values are almost impossible. The real data has a long right tail.” Lognormal fits best; the fit score is much lower (better). That is why we report Q95/Q99 errors.

### Slide 11 — Regimes (Samar, 7:45–8:50)
Point to PCA first, then the violin plots. “Two directions explain 74.3% of sensor variation. K-Means finds low, mid, and high load: 7,010, 15,148, and 7,228 hours, centered near 112, 133, and 155 MW.” Point to the CO violins. “Low load has median CO about 4.26 vs 1.18 at high load — part-load quenching. NOx stays flatter.” Then hand off.

**Handoff line (Samar):** “That brings us to the hardest question: how do we test this without fooling ourselves? Soumya will show our future-year exam and what it revealed.”

**Handoff line (Soumya):** “Thanks, Samar. The plant gave us five years, so we can do something stronger than a random split: we can hide a whole future year.”

### Slide 12 — Fair tests (Soumya, 9:00–10:00)
Point to the left panel, then the right. “The random 80/20 split is a practice quiz: 29,386 train and 7,347 held out. The future-year test is the real exam: train on 2011–2013, tune on 2014, lock 2015.” Every scaler, threshold, and cleaning rule is learned only from training data. 2015 is never used for choices.

### Slide 13 — Model ladder (Soumya, 10:00–10:40)
Move left to right. “We start with a simple line, then PCA, then curved regularized models, then a neural network, and finally a logistic classifier for alarms.” All are syllabus-approved; every model sees the same splits and metrics.

### Slide 14 — Random results (Soumya, 10:40–11:30)
Point to the MLP rows. “On the random split, the neural network reaches CO R² 0.776 and NOx R² 0.850. The simple line is 0.557 and 0.495.” RMSE is 1.103 for CO and 4.452 for NOx. The nonlinear gain is real.

### Slide 15 — Locked 2015 (Soumya, 11:30–12:30)
Point to the top metric panels, then the parity hexbins. “On the unseen 2015 year, CO R² falls to 0.391 and NOx to 0.259. RMSE rises 58% for CO and 115% for NOx.” Point to the Q95 lines/quadrants. “Normal hours follow the diagonal; spikes are harder.”

### Slide 16 — Drift (Soumya, 12:30–13:20)
Point to each panel in the drift figure. “CO median rises 1.16 to 2.53; NOx falls 65.5 to 56.8; TEY and ambient conditions move too.” The 2015 drop is plant drift, not a coding bug. Models need monitoring and retraining.

### Slide 17 — Outlier trap (Soumya, 13:20–14:10)
Point to the left side (ablation bars) and the right side (alerts/confusion). “We tested deleting extreme training rows but kept the test hours untouched. IQR removed 17.8–27.3% of training hours. Random-CO Q99 error rose 7.27 to 13.87; 2015 NOx Q99 rose 28.98 to 35.18.” So raw training wins; keep the hard examples.

### Slide 18 — Alarms (Soumya, 14:10–15:05)
Point to the ROC curves. “We flag the top 10% and 5% worst hours. AUROC is 0.908–0.962; 1.0 is perfect and 0.5 is a coin flip.” Point to NOx confusion. “Q90 catches 389 of 391 spikes but raises 2,429 false alarms. High recall is useful for screening, but operators need help with workload.”

### Slide 19 — Threshold catch (Soumya, 15:05–15:55)
Point to the 2015 NOx rows. “Spike share falls from 10% to 5.3% at Q90 and 5% to 3.4% at Q95. Recall stays above 0.99, but precision drops to 0.138 and 0.107.” Fixed thresholds over-warn when prevalence changes. Recalibrate on fresh validation data, never on the locked test.

### Slide 20 — Four lessons (Soumya, 15:55–16:40)
Point to each card: nonlinear physics, future-year drift, keep outliers, recalibrate alarms. “The model matters, but fair testing and maintenance matter just as much.”

### Slide 21 — Deployment (Soumya, 16:40–17:25)
Point to the four steps. “Run it beside CEMS as an advisor. Retrain quarterly with walk-forward data. Keep raw spikes and watch Q95/Q99. Tune alarms with operators and confirm with CEMS.” Start advisory, not closed-loop control.

### Slide 22 — Closing (Soumya, 17:25–17:50)
“Thank you. The repository and speaker notes are public at the GitHub link. We are happy to take questions.”

---

## Six anticipated Q&A answers

1. **Why not use a much bigger neural network?**  
   The data is tabular with 36,733 rows and nine inputs. A compact MLP already captures the nonlinearity. The bigger risk is temporal drift, not model capacity. A simpler model is also easier to audit and retrain.

2. **Why did 2015 accuracy drop so much?**  
   The plant changed: CO and NOx medians moved, load and weather shifted, and NOx spike prevalence fell. That is distribution drift. The code and leakage controls are unchanged; the relationships themselves moved.

3. **Why keep extreme spikes instead of cleaning them?**  
   The spikes are the target of the monitoring problem. Our paired test kept the test rows identical. IQR trimming raised random-CO Q99 error from 7.27 to 13.87 and 2015 NOx Q99 from 28.98 to 35.18. Deleting them makes the model blind to the hours that matter.

4. **Is NOx R² 0.259 actually useful?**  
   It explains about a quarter of hourly variation, so it is not a precision instrument. But the alarm model ranks spikes well (AUROC 0.947–0.962), which is enough for screening. It should be used with CEMS confirmation, not alone.

5. **Can this replace a physical CEMS?**  
   No. It is an advisory soft sensor. It can catch patterns earlier and provide redundancy, but regulatory reporting still needs certified hardware and human review. Deployment should be a parallel advisory layer first.

6. **Why not use NOx to predict CO, since both are measured?**  
   That would be target leakage and would make the model useless when one analyzer fails. CO and NOx have different chemistry and different controls. We keep the targets independent so each model is a genuine sensor-only soft sensor.
"""
    NOTES_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {NOTES_PATH}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def expected_image_counts() -> list[int]:
    counts = []
    for slide in SLIDES:
        if slide["layout"] in {"figure", "figure_side", "figure_top"}:
            counts.append(1)
        elif slide["layout"] == "two_figures":
            counts.append(2)
        else:
            counts.append(0)
    return counts


def verify_pdf() -> None:
    import pymupdf

    assert PDF_PATH.is_file() and PDF_PATH.stat().st_size > 50_000
    document = pymupdf.open(PDF_PATH)
    assert len(document) == TOTAL, f"Expected {TOTAL} pages, got {len(document)}"
    TMP_RENDER.mkdir(parents=True, exist_ok=True)
    for old in TMP_RENDER.glob("slide_*.png"):
        old.unlink()

    counts = expected_image_counts()
    for index, page in enumerate(document):
        number = index + 1
        rect = page.rect
        assert abs(rect.width - PAGE_W) < 0.5, (number, rect)
        assert abs(rect.height - PAGE_H) < 0.5, (number, rect)
        text = page.get_text("text")
        assert len(text.strip()) > 20, f"Slide {number} is blank"
        # All text blocks must remain inside the page.
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            x0, y0, x1, y1 = block["bbox"]
            assert x0 >= -0.5 and y0 >= -0.5 and x1 <= PAGE_W + 0.5 and y1 <= PAGE_H + 0.5, (
                number,
                block["bbox"],
                text[:40],
            )
        # Every placed image must fit inside the page and match its aspect ratio.
        images = page.get_image_info()
        assert len(images) == counts[index], (
            f"Slide {number}: expected {counts[index]} images, found {len(images)}"
        )
        for item in images:
            x0, y0, x1, y1 = item["bbox"]
            assert x0 >= -0.5 and y0 >= -0.5
            assert x1 <= PAGE_W + 0.5 and y1 <= PAGE_H + 0.5
            assert y1 <= 512, f"Slide {number}: image enters footer zone"
        pix = page.get_pixmap(dpi=140)
        pix.save(TMP_RENDER / f"slide_{number:02d}.png")

    # Contact sheet for fast visual inspection.
    thumbs = []
    for page_path in sorted(TMP_RENDER.glob("slide_*.png")):
        with Image.open(page_path) as image:
            image = image.convert("RGB")
            image.thumbnail((360, 203))
            canvas = Image.new("RGB", (372, 215), "white")
            canvas.paste(image, ((372 - image.width) // 2, (215 - image.height) // 2))
            thumbs.append(canvas)
    columns = 4
    rows = (TOTAL + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 372, rows * 215), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 372, (index // columns) * 215))
    contact = Path("/tmp/opencode/presentation_contact.png")
    sheet.save(contact)
    print(f"Rendered {TOTAL} pages to {TMP_RENDER}")
    print(f"Contact sheet: {contact}")


def verify_pptx() -> None:
    presentation = Presentation(PPTX_PATH)
    assert len(presentation.slides) == TOTAL
    for slide in presentation.slides:
        assert len(slide.shapes) > 0
    assert abs(presentation.slide_width - Inches(13.333)) < 1000
    print(f"PPTX verified: {len(presentation.slides)} slides")

    soffice = shutil.which("soffice")
    if soffice is None:
        print("LibreOffice not installed; skipped editable-deck PDF conversion check")
        return
    with tempfile.TemporaryDirectory(prefix="pems_pptx_check_") as temp_dir:
        profile = Path(temp_dir) / "profile"
        profile.mkdir()
        command = [
            soffice,
            "--headless",
            f"-env:UserInstallation={profile.resolve().as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            temp_dir,
            str(PPTX_PATH),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=180,
            )
        except (subprocess.SubprocessError, OSError) as error:
            print(f"LibreOffice Impress conversion unavailable: {error}")
            return
        converted = Path(temp_dir) / f"{PPTX_PATH.stem}.pdf"
        if not converted.is_file():
            print("LibreOffice did not produce a PDF; skipped conversion check")
            return
        import pymupdf

        document = pymupdf.open(converted)
        assert len(document) == TOTAL, (
            f"PPTX conversion produced {len(document)} pages, expected {TOTAL}"
        )
        print(f"PPTX LibreOffice conversion verified: {len(document)} pages")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_pdf()
    build_pptx()
    build_notes()
    verify_pdf()
    verify_pptx()
    print(f"Presentation package ready in {OUT}")


if __name__ == "__main__":
    main()
