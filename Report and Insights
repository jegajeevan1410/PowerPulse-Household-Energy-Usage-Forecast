# PowerPulse: Household Energy Usage Forecast

**Domain:** Energy and Utilities
**Objective:** Predict household Global Active Power consumption from historical usage data, and derive actionable insights into energy usage patterns.

---

## 1. Approach

The project followed a five-stage pipeline:

1. **Data Understanding and Exploration** — loaded the UCI Individual Household Electric Power Consumption dataset, inspected structure and quality, and ran EDA to identify distributions, correlations, outliers, and temporal patterns.
2. **Data Preprocessing** — handled missing and inconsistent values, parsed timestamps into calendar features, engineered lag and rolling-window features, and scaled the data for linear and neural models.
3. **Feature Engineering** — ranked candidate features by Random Forest importance and selected the top 15 predictors.
4. **Model Selection and Training** — trained Linear Regression, Random Forest, XGBoost, and a Neural Network against a persistence baseline, then performed hyperparameter tuning on the strongest candidate.
5. **Model Evaluation** — compared models on RMSE, MAE, and R², analysed feature importance and residuals, and selected a final model.

---

## 2. Data Analysis

### 2.1 Dataset

| Property | Value |
|---|---|
| Source | UCI Individual Household Electric Power Consumption |
| Location | Single household, Sceaux, France (7 km from Paris) |
| Period | December 2006 – November 2010 (47 months) |
| Granularity | 1 minute |
| Raw records | 2,075,259 |
| Target variable | `Global_active_power` (kW) |

### 2.2 Data Quality

Missing values accounted for 25,979 records (**1.25%**), matching the dataset documentation. Inspection confirmed that `Date` and `Time` were always present — only the seven measurement columns went missing, and they did so together as complete rows. Because gaps occurred in long contiguous blocks (for example, all of 28 April 2007), interpolation would have propagated a single value across an entire day. These rows were therefore dropped rather than filled.

Additional checks confirmed no negative values, no duplicate timestamps, and voltage values within the expected 220–255 V range for the French grid.

### 2.3 Exploratory Findings

**Consumption patterns.** Usage follows a clear daily rhythm: a flat overnight baseline of roughly 0.3 kW, a morning rise, and a pronounced evening peak between 18:00 and 21:00. Weekday and weekend profiles differ only marginally.

**Correlation structure.** `Global_intensity` correlated with the target at r ≈ 1.00. This is expected — power is the product of voltage and current — and makes the variable unusable as a predictor, since knowing it at time *T* presupposes having already measured consumption at time *T*. It was removed as target leakage.

`Voltage` showed a negative correlation of −0.40 with consumption, consistent with line voltage sagging under increased household load. `Sub_metering_3` (water heater and air conditioner) was the strongest legitimate correlate at 0.64.

**Outliers.** Boxplots showed heavy right tails across all power variables. These were retained rather than removed: high readings represent genuine appliance activity — precisely the peaks the model is intended to predict — not measurement error. Sub-meterings 1 and 2 read zero most of the time, so any non-zero reading registers statistically as an outlier while being physically normal.

### 2.4 Feature Engineering

Nineteen candidate features were constructed and ranked by Random Forest importance; the top 15 were retained, accounting for over 99% of cumulative importance.

| Group | Features |
|---|---|
| Lag | `lag_1min`, `lag_60min`, `lag_1day` |
| Rolling | `roll_mean_60`, `roll_mean_1440`, `roll_std_60`, `prev_day_avg` |
| Sub-metering | `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3` |
| Electrical | `Voltage`, `Global_reactive_power` |
| Calendar | `hour`, `dayofweek`, `is_weekend` |

All rolling windows were computed with a one-step shift so that the current observation is excluded from its own window, preventing leakage.

Discarded features (`minute`, `day`, `weekofyear`, `year`, `month`, `season`, `is_peak_hour`) each scored below 0.001 importance. Removing them changed Linear Regression R² by only 0.0001 (0.9523 → 0.9522), confirming they were redundant once lag features were present.

External weather data was not incorporated. The dataset contains no weather variables, and while temperature would plausibly improve winter predictions given electric heating load, the calendar features act as partial proxies. Integrating hourly meteorological data is noted as future work.

---

## 3. Model Selection and Evaluation

### 3.1 Experimental Setup

The data was split chronologically — first 80% for training, final 20% for testing — rather than randomly. Random shuffling would place future observations in the training set and past observations in the test set, producing optimistic but meaningless scores.

| Split | Records | Period |
|---|---|---|
| Train | 1,638,272 | 17 Dec 2006 – 5 Feb 2010 |
| Test | 409,568 | 5 Feb 2010 – 26 Nov 2010 |

Features were standardised using a scaler fitted on training data only. Tree-based models used unscaled inputs. All models used `random_state=42` for reproducibility.

### 3.2 Baseline

A persistence baseline was established — predicting each minute's consumption as equal to the previous minute's. Any model must beat this to demonstrate genuine learning.

### 3.3 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence Baseline | 0.0719 | 0.2216 | 0.9389 |
| Linear Regression | 0.0943 | 0.1960 | 0.9522 |
| Random Forest | **0.0614** | 0.1624 | 0.9672 |
| **XGBoost** | 0.0642 | 0.1534 | 0.9707 |
| Neural Network (MLP) | 0.0661 | **0.1529** | **0.9709** |
| XGBoost (tuned) | 0.0617 | 0.1575 | 0.9691 |

**Linear Regression** beat the baseline on RMSE and R² but lost on MAE. The divergence is informative: MAE weights all errors equally while RMSE penalises large errors quadratically. The baseline is more accurate during steady consumption but fails badly at transitions, whereas the linear model uses sub-metering and voltage signals to partially anticipate them.

**Random Forest** was the first model to beat the baseline on all three metrics. Trained on a 300,000-row sample in 187 seconds.

**XGBoost** achieved the best RMSE and R² among tree models while training on the full 1.64 million rows in 57 seconds — approximately three times faster than Random Forest on less than one-fifth of the data.

**Neural Network** matched XGBoost within noise (RMSE 0.1529 vs 0.1534) despite training on only 300,000 rows and stopping before convergence at the iteration limit.

**Hyperparameter tuning** used `RandomizedSearchCV` with 15 candidate configurations and `TimeSeriesSplit(n_splits=3)` for time-aware cross-validation. The search selected a more conservative configuration (`max_depth` 6 rather than 8, `learning_rate` 0.05 rather than 0.1). This reduced MAE (0.0642 → 0.0617) but increased RMSE (0.1534 → 0.1575) — the shallower model handles typical minutes better while smoothing over sharp peaks. Because the search optimised RMSE on a 400,000-row chronological slice rather than the full training set, the selected parameters did not transfer. Tuning did not improve overall performance, and the untuned configuration was retained.

### 3.4 Final Model

**XGBoost (untuned)** was selected. Its RMSE advantage over Random Forest is meaningful, its 0.0005 RMSE deficit against the Neural Network is within noise, and it offers decisive practical advantages: it trains on the complete dataset in under a minute, requires no feature scaling, and produces interpretable feature importances.

### 3.5 Feature Importance

| Feature | Linear Regression (coefficient) | XGBoost (importance) |
|---|---|---|
| `lag_1min` | 0.844 | 0.682 |
| `roll_mean_60` | 0.067 | 0.122 |
| `Sub_metering_1` | 0.104 | 0.065 |
| `Sub_metering_3` | 0.094 | 0.053 |
| `Sub_metering_2` | 0.089 | 0.043 |
| `Voltage` | −0.031 | 0.006 |
| `Global_reactive_power` | 0.022 | 0.003 |
| `is_weekend` | −0.003 | 0.002 |

Both models agree that the previous minute's reading dominates. Notably, XGBoost distributes importance more evenly — `lag_1min` falls from 0.96 under Random Forest to 0.68, with `roll_mean_60` rising to 0.12 — indicating that gradient boosting extracts more signal from recent-trend features rather than relying on simple persistence.

The negative `Voltage` coefficient is physically meaningful: higher household current draw produces measurable voltage drop. Calendar features contribute almost nothing once lag features are present, since recent consumption already encodes time-of-day behaviour implicitly.

### 3.6 Error Analysis

Prediction quality is highly bimodal.

**Steady periods** — errors are negligible:

| Predicted | Actual | Error |
|---|---|---|
| 0.3162 | 0.3160 | 0.0002 |
| 0.4212 | 0.4240 | 0.0028 |
| 0.3505 | 0.3480 | 0.0025 |

**Switch-on events** — the five largest errors in the test set:

| Predicted | Actual | Error |
|---|---|---|
| 1.423 | 4.428 | 3.005 |
| 1.526 | 4.518 | 2.992 |
| 1.263 | 4.214 | 2.951 |
| 1.352 | 4.260 | 2.908 |
| 0.669 | 3.566 | 2.897 |

Every one of the worst cases is an **under-prediction** where consumption jumped from roughly 1.4 kW to over 4 kW within a single minute. The error magnitude gap between the two regimes is roughly 1,500×.

This is not a tuning deficiency but a structural limit. Appliance activation is an occupant decision that leaves no trace in prior consumption readings — no amount of historical data encodes the moment someone chooses to switch on an oven. This asymmetry also explains the persistent gap between MAE (0.0642) and RMSE (0.1534): the squared term in RMSE is dominated by these rare but large transition errors.

### 3.7 Visual Validation

Time-series plots confirmed the quantitative results at two scales. At minute resolution, predicted and actual traces overlap almost completely across full daily cycles, with visible divergence only at sharp peaks. At daily-average resolution across the ten-month test period, the model tracks seasonal trend faithfully, including a prolonged low-consumption period in August 2010.

![Actual vs Predicted — 2 Day Sample](../images/actual_vs_predicted_2days.png)

![Daily Average Comparison](../images/daily_average_comparison.png)

---

## 4. Insights and Recommendations

### 4.1 Energy Usage Insights

**Evening peak is the dominant load event.** Consumption concentrates between 18:00 and 21:00, rising from an overnight baseline near 0.3 kW to peaks exceeding 5 kW.

**Water heating and climate control drive the largest share.** `Sub_metering_3` correlates most strongly with total consumption among the sub-meters (r = 0.64) and shows the highest sub-metering importance in the linear model.

**Occupancy is directly detectable.** Daily-average consumption fell to approximately 0.4 kW throughout August 2010, against an annual mean near 1.1 kW — a signature consistent with household vacancy. The model tracked this without any explicit vacation indicator.

**Winter consumption substantially exceeds summer.** The seasonal pattern in the daily-average series is consistent with electric heating load.

**Voltage responds measurably to load.** The −0.40 correlation between voltage and consumption reflects local grid response to household demand.

### 4.2 Recommendations by Business Use Case

**Household energy management.** Shifting deferrable loads — dishwasher, washing machine, water heater — outside the 18:00–21:00 window would reduce peak-tariff exposure. Because `Sub_metering_3` accounts for a large share of consumption, water heater scheduling and thermostat setback offer the highest single-appliance savings potential.

**Demand forecasting for providers.** The model achieves R² 0.97 at one-minute horizon, sufficient for very short-term load balancing. For day-ahead planning, the pipeline should be retrained on hourly-aggregated data, where lag dominance diminishes and calendar and weather features become materially more useful.

**Anomaly detection.** The residual distribution provides a ready detection mechanism. Because errors are near-zero during steady operation, sustained deviation beyond a threshold — for instance three standard deviations — would flag equipment faults, unauthorised usage, or metering errors without requiring a separate model.

**Occupancy-aware automation.** The August signature demonstrates that vacancy is inferrable from consumption alone. This supports automatic vacation-mode triggering for water heating and climate systems.

### 4.3 Limitations

- **Forecast horizon is one minute.** The model estimates the next minute given the current one. It does not forecast hours or days ahead, and would degrade substantially at longer horizons where `lag_1min` is unavailable.
- **Single household.** All findings derive from one dwelling in Sceaux, France. Generalisation to other households, dwelling types, or climates is untested.
- **Sudden load onset is not predictable.** As Section 3.6 establishes, appliance switch-on events cannot be anticipated from consumption history.
- **No external covariates.** Weather, tariff schedules, and occupancy data were unavailable.

### 4.4 Future Work

1. Retrain on hourly and daily aggregations to support longer forecast horizons.
2. Integrate historical weather data to test whether temperature improves winter predictions.
3. Evaluate sequence models (LSTM, Temporal Fusion Transformer) for multi-step forecasting.
4. Formalise the residual-threshold approach into a deployable anomaly detection module.
5. Deploy the model behind a web interface for interactive scenario testing.

---

## 5. Reproducibility

| Component | Detail |
|---|---|
| Language | Python 3 |
| Libraries | pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn |
| Random seed | `random_state=42` across all models and splits |
| Validation | `TimeSeriesSplit(n_splits=3)` for hyperparameter search |
| Model artifact | `final_xgb_model.pkl` (joblib) |
| Feature list | `features.pkl` (preserves column order for inference) |

---

## Appendix: Final Feature Set

```
lag_1min, lag_60min, lag_1day,
roll_mean_60, roll_mean_1440, roll_std_60, prev_day_avg,
Sub_metering_1, Sub_metering_2, Sub_metering_3,
Voltage, Global_reactive_power,
hour, dayofweek, is_weekend
```
