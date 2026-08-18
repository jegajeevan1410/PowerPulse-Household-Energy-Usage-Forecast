# PowerPulse: Household Energy Usage Forecast

Machine learning pipeline that predicts household electricity consumption from historical usage data, built on the UCI Individual Household Electric Power Consumption dataset (2.07 million minute-level readings, Dec 2006 – Nov 2010).

**Final model:** XGBoost — RMSE 0.1534 kW · MAE 0.0642 kW · R² 0.9707

---

## Problem

Accurate energy consumption forecasting supports better planning, cost reduction, and resource optimisation. This project builds a regression model to predict `Global_active_power` for a single household, and extracts actionable insights into usage patterns for both consumers and energy providers.

---

## Dataset

| Property | Value |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) |
| Location | Single household, Sceaux, France |
| Period | December 2006 – November 2010 (47 months) |
| Granularity | 1 minute |
| Records | 2,075,259 |
| Target | `Global_active_power` (kW) |

The dataset is not included in this repository due to size. Download it from the link above and place `household_power_consumption.txt` in the project root.

---

## Approach

**1. Data cleaning**
Missing values (25,979 rows, 1.25%) occurred as complete measurement blocks spanning entire days, making interpolation unsuitable — these rows were dropped. Validated against negative values, duplicate timestamps, and out-of-range voltage.

**2. Leakage removal**
`Global_intensity` correlated with the target at r ≈ 1.00, since power is the product of voltage and current. Keeping it would inflate R² to 0.999 while making the model useless — knowing intensity at time *T* presupposes having measured consumption at time *T*. It was removed.

**3. Feature engineering**
Built lag features (1 min, 1 hour, 1 day), rolling statistics (60-min and 1440-min mean, 60-min std), previous-day average, and calendar features. All rolling windows shifted by one step to exclude the current observation. Ranked 19 candidates by Random Forest importance and retained the top 15.

**4. Chronological split**
First 80% train (1,638,272 rows), final 20% test (409,568 rows). Random shuffling would leak future observations into training.

**5. Modelling**
Trained four models against a persistence baseline, then tuned the strongest with `RandomizedSearchCV` and `TimeSeriesSplit`.

---

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence Baseline | 0.0719 | 0.2216 | 0.9389 |
| Linear Regression | 0.0943 | 0.1960 | 0.9522 |
| Random Forest | **0.0614** | 0.1624 | 0.9672 |
| **XGBoost** | 0.0642 | 0.1534 | 0.9707 |
| Neural Network (MLP) | 0.0661 | **0.1529** | **0.9709** |
| XGBoost (tuned) | 0.0617 | 0.1575 | 0.9691 |

![Actual vs Predicted](images/actual_vs_predicted_2days.png)

![Feature Importance](images/feature_importance.png)

XGBoost was selected as the final model. Its RMSE deficit against the Neural Network (0.0005) is within noise, and it trains on the full 1.64M rows in 57 seconds — roughly 3× faster than Random Forest on less than a fifth of the data — while requiring no feature scaling and producing interpretable importances.

Hyperparameter tuning did not improve performance. The search selected a shallower configuration that lowered MAE but raised RMSE, and the untuned model was retained.

---

## Key Findings

**The previous minute dominates.** `lag_1min` carries 0.68 feature importance under XGBoost. Consumption changes little between consecutive minutes, which sets a natural ceiling on achievable improvement.

**Error is bimodal.** During steady consumption, errors fall to ~0.002 kW. At appliance switch-on events, they reach ~3.0 kW — a gap of roughly 1,500×. Every one of the five worst predictions was an under-prediction where load jumped from ~1.4 kW to over 4 kW within a single minute. This is a structural limit, not a tuning deficiency: appliance activation is an occupant decision that leaves no trace in prior readings. It also explains the persistent MAE/RMSE gap, since RMSE's squared term is dominated by these rare transitions.

**Occupancy is detectable from consumption alone.** Daily averages fell to ~0.4 kW throughout August 2010 against an annual mean near 1.1 kW — a vacancy signature the model tracked without any explicit indicator.

**Voltage responds to load.** A −0.40 correlation with consumption reflects line voltage sagging under household demand.

**Evening peak drives the load curve.** Consumption rises from a ~0.3 kW overnight baseline to peaks above 5 kW between 18:00 and 21:00.

---

## Recommendations

- **Households:** shift deferrable loads (dishwasher, washing machine, water heater) outside 18:00–21:00 to reduce peak-tariff exposure. `Sub_metering_3` — water heating and climate control — offers the highest single-appliance savings potential.
- **Providers:** the model is suitable for very short-term load balancing. For day-ahead planning, retrain on hourly aggregates where lag dominance diminishes and calendar and weather features become materially useful.
- **Anomaly detection:** because residuals are near-zero during steady operation, a threshold on sustained deviation flags equipment faults or unauthorised usage without a separate model.

---

## Limitations

- Forecast horizon is one minute ahead; performance would degrade substantially at longer horizons.
- Findings derive from a single household in one climate — generalisation is untested.
- Sudden load onset is inherently unpredictable from consumption history.
- No weather, tariff, or occupancy covariates were available.

---

## Repository Structure

```
├── notebooks/
│   └── PowerPulse_Analysis.ipynb    # full pipeline
├── report/
│   └── PowerPulse_Report.md         # detailed write-up
├── models/
│   ├── final_xgb_model.pkl
│   └── features.pkl
├── images/                          # plots
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/jegajeevan1410/PowerPulse-Household-Energy-Forecast.git
cd PowerPulse-Household-Energy-Forecast
pip install -r requirements.txt
```

Download `household_power_consumption.txt` from the UCI link above and place it in the project root, then run the notebook.

**Inference with the saved model:**

```python
import joblib
import pandas as pd

model = joblib.load("models/final_xgb_model.pkl")
features = joblib.load("models/features.pkl")

prediction = model.predict(X_new[features])
```

Column order must match `features.pkl`.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · joblib

---

## Author

**Jegajeevan**
[GitHub](https://github.com/jegajeevan1410) · [LinkedIn](https://linkedin.com/in/jegajeevan1410)

