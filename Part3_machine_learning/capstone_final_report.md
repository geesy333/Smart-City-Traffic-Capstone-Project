# Smart City Traffic Intelligence Capstone: End-to-End Analytics, Data Engineering & Machine Learning Report

**Corridor:** Westbound Interstate 94 (I-94), Minneapolis–St Paul  
**Dataset:** Metro Interstate Traffic Volume (48,204 raw hourly records, 2012–2017)  

---

## 1. Executive Summary
This capstone delivers a production-ready traffic intelligence system across three integrated stages: exploratory SQL/BI analytics (Part 1), an automated Python ETL and feature engineering pipeline (Part 2), and supervised machine learning models for continuous traffic volume forecasting and congestion detection (Part 3). Across all modeling experiments using strict chronological validation, **temporal commuter rhythms and short-term traffic momentum—not weather fluctuations—emerge as the dominant drivers of corridor congestion**.

---

## 2. Part 1 & Part 2 Recap: Data Foundation & Feature Engineering
- **SQL & Statistical Audit (Part 1):** Analysis of 48,204 hourly records established baseline demand (mean `3,259.82` vehicles/hr, median `3,380.00` vehicles/hr), confirmed weak correlation between temperature and traffic ($r = 0.1303$), and quantified congestion probability ($P(\text{Congestion} > 5,500) = 14.73\%$).
- **Automated Cleaning Pipeline (`pipeline.py`):** Standardised 1,730 categorical text entries, dropped 17 exact duplicates, and imputed 10 physically impossible temperature sensor failures (`<= 0 K` or `> 330 K`) and 1 rainfall anomaly (`> 9000 mm`) using monthly medians, yielding 48,187 verified records.
- **Feature Engineering (`feature_engineering.py`):** Constructed 20 domain-specific temporal, cyclical (`hour_sin`, `hour_cos`, `month_sin`, `month_cos`), meteorological (`temp_celsius`, `is_freezing`, `total_precipitation_1h`), and leak-free autoregressive lag features (`traffic_lag_1h`, `traffic_lag_24h`, `traffic_rolling_mean_3h`), producing 48,163 modeling-ready observations.

---

## 3. Part 3: Supervised Machine Learning Results
All models were evaluated on a strict **chronological 80/20 split** (Train: 38,530 rows | Test: 9,633 rows) to prevent future time-series data leakage.

### A. Continuous Traffic Volume Forecasting (`train_regression.py`)
| Model | RMSE (Vehicles/Hr) | MAE (Vehicles/Hr) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor (Best)** | **250.20** | **154.49** | **0.9838** |
| Gradient Boosting Regressor | 264.61 | 171.29 | 0.9819 |
| Ridge Regression (Baseline) | 546.13 | 401.15 | 0.9230 |

- **Key Takeaway:** Tree-based ensemble modeling reduced prediction error by **54.2%** compared to the linear baseline, explaining **98.38%** of out-of-sample traffic variance (`01_regression_actual_vs_predicted.png`).

### B. Binary Congestion Classification (`train_classification.py`, Target: `traffic_volume > 5,500`)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier (Best)** | **0.9713** | **0.8700** | **0.9305** | **0.8992** | **0.9940** |
| Random Forest Classifier | 0.9615 | 0.7957 | 0.9683 | 0.8735 | 0.9940 |
| Logistic Regression (Baseline) | 0.8726 | 0.5196 | 0.9622 | 0.6748 | 0.9701 |

- **Confusion Matrix & Diagnostic Analysis (`02_classification_confusion_roc.png`):** On 9,633 unseen test hours, the Gradient Boosting Classifier correctly detected **1,231 of 1,323** congested hours (`93.05%` recall) with only 184 false positives (`97.13%` overall accuracy, `0.9940` ROC-AUC).

### C. Top Predictive Drivers (`model_evaluation.py`)
1. `traffic_lag_1h` (**54.23%**) — Immediate prior-hour traffic momentum.
2. `hour` (**16.52%**) — Diurnal commuter schedule.
3. `hour_sin` (**10.91%**) & `hour_cos` (**9.06%**) — Cyclical daily periodicity.
4. `traffic_rolling_mean_3h` (**2.18%**) — Short-term 3-hour rolling corridor trend.

---

## 4. Operational Recommendations for Smart-City Deployment
1. **Proactive Ramp Metering & Dynamic Pricing:** Integrate the Gradient Boosting Congestion Classifier into traffic operations centers to trigger automated ramp metering and variable tolling 60 minutes ahead of predicted congestion spikes (`F1 = 0.8992`, `93.05%` recall).
2. **Real-Time Sensor Health Guardrails:** Deploy the `pipeline.py` validation layer upstream of live inference pipelines to automatically catch and impute `0 K` sensor faults before they corrupt real-time predictions.
3. **Scheduled Model Retraining:** Retrain the Random Forest and Gradient Boosting models quarterly on newly ingested traffic logs to adapt to seasonal construction and multi-year corridor growth.
