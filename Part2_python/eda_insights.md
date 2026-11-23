# Part 2: Python Data Engineering & Exploratory Data Analysis (EDA) Insights

## 1. Data Cleaning & Pipeline Audit Summary (`pipeline.py`)
- **Raw Input:** 48,204 hourly records across 9 columns (`Metro_Interstate_Traffic_Volume.csv`).
- **Schema & Timestamp Validation:** All 9 expected columns verified; 0 invalid timestamps found.
- **Categorical Standardisation:** Trimmed whitespace and normalised casing across 1,730 categorical text entries (`holiday`, `weather_main`, `weather_description`).
- **Duplicate Removal:** Identified and removed 17 exact duplicate rows, bringing the dataset to 48,187 unique records.
- **Outlier Imputation:** Detected and imputed 10 physically impossible temperature readings (`<= 0 K` or `> 330 K`, including `0 K` / `-273.15 °C` sensor failures) and 1 impossible rainfall reading (`> 9000 mm`) using monthly medians.

## 2. Feature Engineering Summary (`feature_engineering.py`)
Engineered 20 domain-specific temporal, meteorological, cyclical, and lag features (48,163 rows × 29 columns after dropping the initial 24 rows lacking 24-hour historical lag context):
- **Temporal & Calendar Indicators:** `hour`, `day_of_week`, `day_name`, `month`, `year`, `is_weekend`, `is_holiday`, and `is_rush_hour` (weekdays 07:00–09:00 and 16:00–18:00).
- **Cyclical Encodings:** `hour_sin`, `hour_cos`, `month_sin`, and `month_cos` to preserve circular continuity between hour 23 and hour 0.
- **Meteorological & Target Indicators:** `temp_celsius`, `is_freezing`, `has_precipitation`, `total_precipitation_1h`, and binary congestion target `is_congested` (`traffic_volume > 5,500` vehicles/hr).
- **Historical Lag & Rolling Features:** `traffic_lag_1h`, `traffic_lag_24h`, and `traffic_rolling_mean_3h` (leak-free shifted rolling average).

## 3. Visual & Statistical Insights (`visualizations.py`)
1. **Weekday vs. Weekend Diurnal Profile (`01_hourly_weekday_vs_weekend.png`):**
   - Weekdays follow a pronounced **bimodal commuter curve** with sharp peaks during the morning rush (07:00–08:00) and evening rush (16:00–17:00).
   - Weekends display a single, broader midday plateau peaking gently between 12:00 and 16:00 with significantly lower morning volume.
2. **Weather Condition Distributions (`02_weather_traffic_boxplot.png`):**
   - Median and interquartile traffic volumes remain relatively stable across standard weather conditions (`Clouds`, `Clear`, `Rain`, `Mist`), while extreme or rare events (`Squall`, `Fog`) exhibit lower median volumes and narrower distributions.
3. **Cleaned Temperature vs. Traffic Relationship (`03_temperature_vs_traffic.png`):**
   - Imputing the `0 K` (`-273.15 °C`) sensor anomalies restored a continuous ambient temperature range (`-30 °C` to `+37 °C`), revealing a mild positive linear slope ($r \approx 0.13$) driven by higher daytime commuter activity during warmer daylight hours.
4. **Feature Correlation Structure (`04_feature_correlation_heatmap.png`):**
   - Short-term autoregressive lags (`traffic_lag_1h`, `traffic_rolling_mean_3h`, and `traffic_lag_24h`) and rush-hour indicators (`is_rush_hour`, `hour_sin`) exhibit the strongest linear relationships with `traffic_volume`, providing high-signal predictors for Part 3 machine learning models.
