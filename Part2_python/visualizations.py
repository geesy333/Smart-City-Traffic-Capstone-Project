import logging
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "pipeline.log") -> None:
    """Append visualization logs to pipeline.log and stream to console."""
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


def generate_visualizations(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """Generate and save publication-ready static charts for Part 2 EDA."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # Chart 1: Hourly Traffic Profile (Weekday vs. Weekend)
    plt.figure(figsize=(10, 5.5))
    hourly_summary = (
        df.groupby(["hour", "is_weekend"])["traffic_volume"]
        .mean()
        .reset_index()
    )
    hourly_summary["Day Type"] = hourly_summary["is_weekend"].map({0: "Weekday", 1: "Weekend"})

    sns.lineplot(
        data=hourly_summary,
        x="hour",
        y="traffic_volume",
        hue="Day Type",
        marker="o",
        linewidth=2.5,
        palette=["#1f77b4", "#ff7f0e"],
    )
    plt.axvspan(7, 9, color="#1f77b4", alpha=0.10, label="Morning Rush (07:00-09:00)")
    plt.axvspan(16, 18, color="#d62728", alpha=0.10, label="Evening Rush (16:00-18:00)")
    plt.title("Average Hourly I-94 Traffic Volume: Weekday vs. Weekend", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (0 - 23)", fontsize=11)
    plt.ylabel("Average Traffic Volume (Vehicles / Hour)", fontsize=11)
    plt.xticks(range(0, 24))
    plt.legend(frameon=True)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "01_hourly_weekday_vs_weekend.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    logger.info("Saved Chart 1 to '%s'.", chart1_path)

    # Chart 2: Traffic Volume Distribution by Weather Condition
    plt.figure(figsize=(11, 5.5))
    order = (
        df.groupby("weather_main")["traffic_volume"]
        .median()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(
        data=df,
        x="weather_main",
        y="traffic_volume",
        order=order,
        palette="viridis",
    )
    plt.title("Traffic Volume Distribution Across Weather Conditions (Cleaned Data)", fontsize=13, fontweight="bold")
    plt.xlabel("Primary Weather Condition", fontsize=11)
    plt.ylabel("Traffic Volume (Vehicles / Hour)", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "02_weather_traffic_boxplot.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    logger.info("Saved Chart 2 to '%s'.", chart2_path)

    # Chart 3: Cleaned Temperature (Celsius) vs. Traffic Volume Trend
    plt.figure(figsize=(10, 5.5))
    sample_df = df.sample(n=min(6000, len(df)), random_state=42)
    sns.regplot(
        data=sample_df,
        x="temp_celsius",
        y="traffic_volume",
        scatter_kws={"alpha": 0.18, "s": 14, "color": "#2b5c8f"},
        line_kws={"color": "#d62728", "linewidth": 2.2, "label": "Linear Trend"},
    )
    plt.title("Cleaned Ambient Temperature (°C) vs. Traffic Volume", fontsize=13, fontweight="bold")
    plt.xlabel("Temperature (°C) [Outliers Imputed]", fontsize=11)
    plt.ylabel("Traffic Volume (Vehicles / Hour)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "03_temperature_vs_traffic.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    logger.info("Saved Chart 3 to '%s'.", chart3_path)

    # Chart 4: Correlation Heatmap of Engineered Features
    plt.figure(figsize=(10, 8))
    selected_cols = [
        "traffic_volume",
        "traffic_lag_1h",
        "traffic_lag_24h",
        "traffic_rolling_mean_3h",
        "is_rush_hour",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "temp_celsius",
        "clouds_all",
        "total_precipitation_1h",
    ]
    corr_matrix = df[selected_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix of Engineered Traffic & Weather Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "04_feature_correlation_heatmap.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    logger.info("Saved Chart 4 to '%s'.", chart4_path)


def main() -> None:
    setup_logging("pipeline.log")
    logger.info("Starting Part 2 static visualization pipeline.")

    input_path = "engineered_traffic_data.csv"
    try:
        df = pd.read_csv(input_path)
        logger.info("Loaded '%s' (%d rows, %d columns).", input_path, df.shape[0], df.shape[1])
        generate_visualizations(df, output_dir="figures")
        logger.info("All Part 2 visualizations generated successfully.")
    except FileNotFoundError:
        logger.error("Engineered dataset '%s' not found. Run feature_engineering.py first.", input_path, exc_info=True)
        sys.exit(1)
    except Exception:
        logger.error("Visualization generation failed due to an unexpected error.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
