import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "ml_pipeline.log") -> None:
    """Configure console and file logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)


def prepare_ml_dataset(filepath: str) -> pd.DataFrame:
    """Clean raw data and engineer leak-free features for supervised modeling."""
    df = pd.read_csv(filepath)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.dropna(subset=["date_time"]).drop_duplicates().reset_index(drop=True)
    df = df.sort_values("date_time").reset_index(drop=True)

    # Standardise categorical text
    df["holiday"] = df["holiday"].fillna("None").astype(str).str.strip()
    df["weather_main"] = df["weather_main"].fillna("None").astype(str).str.strip().str.title()

    # Impute physically impossible temperature (<= 0 K or > 330 K) & rainfall (> 9000 mm)
    df["_month"] = df["date_time"].dt.month
    invalid_temp = (df["temp"] <= 0) | (df["temp"] > 330)
    for m in sorted(df["_month"].unique()):
        mask = (df["_month"] == m) & invalid_temp
        if mask.any():
            df.loc[mask, "temp"] = df.loc[(df["_month"] == m) & (~invalid_temp), "temp"].median()

    invalid_rain = (df["rain_1h"] < 0) | (df["rain_1h"] > 9000)
    for m in sorted(df["_month"].unique()):
        mask = (df["_month"] == m) & invalid_rain
        if mask.any():
            df.loc[mask, "rain_1h"] = df.loc[(df["_month"] == m) & (~invalid_rain), "rain_1h"].median()

    # Temporal & cyclical features
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["_month"]
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = (df["holiday"] != "None").astype(int)
    df["is_rush_hour"] = ((df["is_weekend"] == 0) & (df["hour"].isin([7, 8, 9, 16, 17, 18]))).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Weather features & binary congestion target
    df["temp_celsius"] = df["temp"] - 273.15
    df["is_freezing"] = (df["temp_celsius"] <= 0.0).astype(int)
    df["has_precipitation"] = ((df["rain_1h"] > 0) | (df["snow_1h"] > 0)).astype(int)
    df["total_precipitation_1h"] = df["rain_1h"] + df["snow_1h"]
    df["is_congested"] = (df["traffic_volume"] > 5500).astype(int)

    # Leak-free autoregressive lags
    df["traffic_lag_1h"] = df["traffic_volume"].shift(1)
    df["traffic_lag_24h"] = df["traffic_volume"].shift(24)
    df["traffic_rolling_mean_3h"] = df["traffic_volume"].shift(1).rolling(window=3, min_periods=1).mean()

    df = df.dropna(subset=["traffic_lag_1h", "traffic_lag_24h"]).drop(columns=["_month"]).reset_index(drop=True)
    logger.info("Prepared ML dataset: %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


def main() -> None:
    setup_logging("ml_pipeline.log")
    logger.info("Starting Part 3 Task 1: Supervised Regression Modeling.")

    df = prepare_ml_dataset("Metro_Interstate_Traffic_Volume.csv")
    df.to_csv("ml_ready_traffic_data.csv", index=False)

    numeric_features = [
        "temp_celsius",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday",
        "is_rush_hour",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "is_freezing",
        "has_precipitation",
        "total_precipitation_1h",
        "traffic_lag_1h",
        "traffic_lag_24h",
        "traffic_rolling_mean_3h",
    ]
    categorical_features = ["weather_main"]

    X = df[numeric_features + categorical_features]
    y = df["traffic_volume"]

    # Chronological 80/20 train-test split (prevents future time-series leakage)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info("Chronological split complete: Train=%d rows, Test=%d rows.", len(X_train), len(X_test))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "Ridge_Regression_Baseline": Ridge(alpha=1.0),
        "Random_Forest_Regressor": RandomForestRegressor(
            n_estimators=150, max_depth=16, min_samples_leaf=4, random_state=42, n_jobs=-1
        ),
        "Gradient_Boosting_Regressor": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42
        ),
    }

    os.makedirs("models", exist_ok=True)
    results = []
    best_r2 = -np.inf
    best_model_name = ""
    best_pipeline = None

    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        results.append({"Model": name, "RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2_Score": round(r2, 4)})
        logger.info("%s -> RMSE: %.2f | MAE: %.2f | R2: %.4f", name, rmse, mae, r2)

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipe

    results_df = pd.DataFrame(results).sort_values("R2_Score", ascending=False)
    results_df.to_csv("regression_metrics.csv", index=False)

    best_model_path = os.path.join("models", "best_regression_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    logger.info(
        "Best Regression Model: '%s' (R2=%.4f) saved to '%s'.",
        best_model_name,
        best_r2,
        best_model_path,
    )


if __name__ == "__main__":
    main()
