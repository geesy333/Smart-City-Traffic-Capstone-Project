import logging
import sys
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "pipeline.log") -> None:
    """Append feature engineering logs to pipeline.log and stream to console."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, meteorological, cyclical, and lag features for traffic modelling."""
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)

    # 1. Temporal features
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["day_name"] = df["date_time"].dt.day_name()
    df["month"] = df["date_time"].dt.month
    df["year"] = df["date_time"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = (df["holiday"] != "None").astype(int)

    # Rush hour indicator (Morning 07:00-09:00, Evening 16:00-18:00 on weekdays)
    is_weekday = df["is_weekend"] == 0
    is_peak_window = df["hour"].isin([7, 8, 9, 16, 17, 18])
    df["is_rush_hour"] = (is_weekday & is_peak_window).astype(int)
    logger.info("Created temporal features: hour, day_of_week, month, year, is_weekend, is_holiday, is_rush_hour.")

    # 2. Cyclical encodings for hour and month (preserves 23 -> 0 and 12 -> 1 continuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    logger.info("Created cyclical trigonometric features: hour_sin, hour_cos, month_sin, month_cos.")

    # 3. Meteorological & domain interaction features
    df["temp_celsius"] = df["temp"] - 273.15
    df["is_freezing"] = (df["temp_celsius"] <= 0.0).astype(int)
    df["has_precipitation"] = ((df["rain_1h"] > 0) | (df["snow_1h"] > 0)).astype(int)
    df["total_precipitation_1h"] = df["rain_1h"] + df["snow_1h"]

    # Traffic congestion target label (> 5500 vehicles/hr matching Part 1 definition)
    df["is_congested"] = (df["traffic_volume"] > 5500).astype(int)
    logger.info("Created weather & target features: temp_celsius, is_freezing, has_precipitation, total_precipitation_1h, is_congested.")

    # 4. Lag and rolling window features (leak-free historical context)
    df["traffic_lag_1h"] = df["traffic_volume"].shift(1)
    df["traffic_lag_24h"] = df["traffic_volume"].shift(24)
    df["traffic_rolling_mean_3h"] = df["traffic_volume"].shift(1).rolling(window=3, min_periods=1).mean()

    before_drop = len(df)
    df = df.dropna(subset=["traffic_lag_1h", "traffic_lag_24h"]).reset_index(drop=True)
    dropped_lag_rows = before_drop - len(df)
    logger.warning(
        "Dropped %d initial rows lacking 24-hour historical lag context.",
        dropped_lag_rows,
    )

    return df


def main() -> None:
    setup_logging("pipeline.log")
    logger.info("Starting Part 2 feature engineering pipeline.")

    input_path = "cleaned_traffic_data.csv"
    output_path = "engineered_traffic_data.csv"

    try:
        df = pd.read_csv(input_path)
        logger.info("Loaded '%s' (%d rows, %d columns).", input_path, df.shape[0], df.shape[1])

        engineered_df = engineer_features(df)
        engineered_df.to_csv(output_path, index=False)
        logger.info(
            "Feature engineering completed successfully. Saved '%s' (%d rows, %d columns).",
            output_path,
            engineered_df.shape[0],
            engineered_df.shape[1],
        )
    except FileNotFoundError:
        logger.error("Cleaned dataset '%s' not found. Run pipeline.py first.", input_path, exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Feature engineering failed due to an unexpected error.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
