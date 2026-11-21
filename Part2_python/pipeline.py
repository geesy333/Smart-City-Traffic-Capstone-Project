import logging
import os
import sys
import pandas as pd
import numpy as np

# Configure module-level logger (never use root logger directly)
logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]


def setup_logging(log_file: str = "pipeline.log") -> None:
    """Configure console and file handlers on the entry-point logger."""
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


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV file with specific exception handling."""
    try:
        df = pd.read_csv(filepath)
        logger.info(
            "Successfully loaded raw dataset from '%s' (%d rows, %d columns).",
            filepath,
            df.shape[0],
            df.shape[1],
        )
        return df
    except FileNotFoundError:
        logger.error("Raw dataset file not found at '%s'.", filepath, exc_info=True)
        raise
    except pd.errors.EmptyDataError:
        logger.error("Raw dataset file '%s' is empty.", filepath, exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Failed to parse CSV file '%s'.", filepath, exc_info=True)
        raise
    except Exception:
        logger.error("Unexpected error while loading '%s'.", filepath, exc_info=True)
        raise


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that all expected columns exist before processing."""
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.error("Schema validation failed. Missing columns: %s", missing_cols)
        raise ValueError(f"Missing required columns: {missing_cols}")
    logger.info("Schema validation passed: all %d expected columns verified.", len(EXPECTED_COLUMNS))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean categorical values, timestamps, duplicates, missing values, and outliers."""
    df = df.copy()

    # Step 1: Parse and validate date_time
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    invalid_dates = int(df["date_time"].isna().sum())
    if invalid_dates > 0:
        df = df.dropna(subset=["date_time"])
        logger.warning(
            "Dropped %d rows due to unparseable or missing date_time values.",
            invalid_dates,
        )
    else:
        logger.info("Date/time parsing completed: 0 invalid timestamps found.")

    # Step 2: Standardise categorical text fields
    cat_cols = ["holiday", "weather_main", "weather_description"]
    modified_cat_rows = 0
    for col in cat_cols:
        df[col] = df[col].fillna("None").astype(str)
        cleaned_series = df[col].str.strip()
        if col == "weather_main":
            cleaned_series = cleaned_series.str.title()
        elif col == "weather_description":
            cleaned_series = cleaned_series.str.lower()
        diff_count = int((df[col] != cleaned_series).sum())
        modified_cat_rows += diff_count
        df[col] = cleaned_series

    if modified_cat_rows > 0:
        logger.warning(
            "Standardised whitespace/casing across %d categorical field entries.",
            modified_cat_rows,
        )
    else:
        logger.info("Categorical standardisation completed: all text labels already consistent.")

    # Step 3: Remove exact duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.warning(
            "Dropped %d exact duplicate rows from the dataset.",
            dup_count,
        )
    else:
        logger.info("Duplicate check completed: 0 duplicate rows found.")

    # Step 4: Handle missing numerical values
    num_cols = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]
    for col in num_cols:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            median_val = float(df[col].median())
            df[col] = df[col].fillna(median_val)
            logger.warning(
                "Imputed %d missing values in '%s' using column median (%.2f).",
                missing_count,
                col,
                median_val,
            )
    logger.info("Missing value check completed across numerical columns.")

    # Step 5: Detect and impute physically impossible temperature outliers (<= 0 K or > 330 K)
    df["_month"] = df["date_time"].dt.month
    invalid_temp_mask = (df["temp"] <= 0) | (df["temp"] > 330)
    invalid_temp_count = int(invalid_temp_mask.sum())

    if invalid_temp_count > 0:
        for month in sorted(df["_month"].unique()):
            month_mask = (df["_month"] == month) & invalid_temp_mask
            if month_mask.any():
                valid_month_temps = df.loc[(df["_month"] == month) & (~invalid_temp_mask), "temp"]
                month_median = float(valid_month_temps.median())
                df.loc[month_mask, "temp"] = month_median
                logger.debug(
                    "Month %02d temperature outlier imputation applied median %.2f K to %d rows.",
                    month,
                    month_median,
                    int(month_mask.sum()),
                )
        logger.warning(
            "Imputed %d rows with physically impossible temperature readings (<= 0 K or > 330 K) using monthly medians.",
            invalid_temp_count,
        )
    else:
        logger.info("Temperature outlier check completed: 0 invalid readings found.")

    # Step 6: Detect and impute physically impossible rainfall outliers (< 0 mm or > 9000 mm)
    invalid_rain_mask = (df["rain_1h"] < 0) | (df["rain_1h"] > 9000)
    invalid_rain_count = int(invalid_rain_mask.sum())

    if invalid_rain_count > 0:
        for month in sorted(df["_month"].unique()):
            month_mask = (df["_month"] == month) & invalid_rain_mask
            if month_mask.any():
                valid_month_rain = df.loc[(df["_month"] == month) & (~invalid_rain_mask), "rain_1h"]
                month_rain_median = float(valid_month_rain.median())
                df.loc[month_mask, "rain_1h"] = month_rain_median
        logger.warning(
            "Imputed %d rows with physically impossible rainfall readings (< 0 or > 9000 mm) using monthly medians.",
            invalid_rain_count,
        )
    else:
        logger.info("Rainfall outlier check completed: 0 invalid readings found.")

    # Step 7: Validate non-negative traffic volume
    invalid_vol_mask = df["traffic_volume"] < 0
    invalid_vol_count = int(invalid_vol_mask.sum())
    if invalid_vol_count > 0:
        df = df[~invalid_vol_mask].reset_index(drop=True)
        logger.warning(
            "Dropped %d rows with negative traffic_volume readings.",
            invalid_vol_count,
        )
    else:
        logger.info("Traffic volume range check completed: all values >= 0.")

    df = df.drop(columns=["_month"])
    return df


def main() -> None:
    setup_logging("pipeline.log")
    logger.info("Starting Part 2 data cleaning pipeline.")

    default_paths = [
        "Metro_Interstate_Traffic_Volume.csv",
        os.path.join("..", "Data", "Metro_Interstate_Traffic_Volume.csv"),
        os.path.join("Data", "Metro_Interstate_Traffic_Volume.csv"),
    ]
    raw_csv_path = next((p for p in default_paths if os.path.exists(p)), default_paths[0])

    try:
        raw_df = load_raw_data(raw_csv_path)
        validate_schema(raw_df)
        cleaned_df = clean_data(raw_df)

        output_path = "cleaned_traffic_data.csv"
        cleaned_df.to_csv(output_path, index=False)
        logger.info(
            "Pipeline completed successfully. Cleaned dataset saved to '%s' (%d rows, %d columns).",
            output_path,
            cleaned_df.shape[0],
            cleaned_df.shape[1],
        )
    except Exception:
        logger.error("Data cleaning pipeline terminated due to an unrecoverable error.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
