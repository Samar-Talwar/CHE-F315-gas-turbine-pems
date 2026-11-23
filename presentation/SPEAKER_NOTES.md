# CHE F315 — Speaker Notes

**Talk:** Predicting Gas Turbine Emissions Using Machine Learning  
**Subtitle:** A Software-Based Alternative to Expensive Physical Emission Sensors (CO & NOx)  
**Course:** CHE F315 — Machine Learning for Chemical Engineers  
**Professor:** Professor Riju De  
**Date:** 23 November 2026  
**Institution:** BITS Pilani, K. K. Birla Goa Campus

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
