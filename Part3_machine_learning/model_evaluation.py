import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "ml_pipeline.log") -> None:
    """Append model evaluation logs to ml_pipeline.log and stream to console."""
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


def main() -> None:
    setup_logging("ml_pipeline.log")
    logger.info("Starting Part 3 Task 3: Model Evaluation & Diagnostic Visualizations.")

    df = pd.read_csv("ml_ready_traffic_data.csv")
    reg_pipe = joblib.load(os.path.join("models", "best_regression_model.joblib"))
    clf_pipe = joblib.load(os.path.join("models", "best_classification_model.joblib"))

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
    y_reg = df["traffic_volume"]
    y_clf = df["is_congested"]

    split_idx = int(len(df) * 0.80)
    X_test = X.iloc[split_idx:]
    y_reg_test = y_reg.iloc[split_idx:]
    y_clf_test = y_clf.iloc[split_idx:]

    os.makedirs("ml_figures", exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # 1. Regression Actual vs. Predicted Plot
    reg_preds = reg_pipe.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_reg_test, reg_preds, alpha=0.18, s=12, color="#1f77b4", label="Test Predictions")
    max_val = max(y_reg_test.max(), reg_preds.max())
    plt.plot([0, max_val], [0, max_val], color="#d62728", linestyle="--", linewidth=2, label="Ideal Parity (y = x)")
    plt.title("Random Forest Regressor: Actual vs. Predicted Traffic Volume (R² = 0.9838)", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Traffic Volume (Vehicles / Hour)", fontsize=11)
    plt.ylabel("Predicted Traffic Volume (Vehicles / Hour)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    fig1_path = os.path.join("ml_figures", "01_regression_actual_vs_predicted.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info("Saved '%s'.", fig1_path)

    # 2. Classification Confusion Matrix & ROC Curve
    clf_preds = clf_pipe.predict(X_test)
    clf_probs = clf_pipe.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_clf_test, clf_probs)
    fpr, tpr, _ = roc_curve(y_clf_test, clf_probs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cm = confusion_matrix(y_clf_test, clf_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (<=5500)", "Congested (>5500)"])
    disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].grid(False)
    axes[0].set_title("Gradient Boosting Confusion Matrix (F1 = 0.8992)", fontsize=12, fontweight="bold")

    axes[1].plot(fpr, tpr, color="#2ca02c", linewidth=2.5, label=f"Gradient Boosting (ROC-AUC = {auc_score:.4f})")
    axes[1].plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", linewidth=1.5, label="Random Chance (AUC = 0.5000)")
    axes[1].set_title("ROC Curve: Binary Congestion Detection", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate", fontsize=11)
    axes[1].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[1].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    fig2_path = os.path.join("ml_figures", "02_classification_confusion_roc.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info("Saved '%s'.", fig2_path)

    # 3. Top 12 Feature Importances (Gradient Boosting Classifier)
    feature_names = clf_pipe.named_steps["preprocessor"].get_feature_names_out()
    clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    importances = clf_pipe.named_steps["model"].feature_importances_

    imp_df = (
        pd.DataFrame({"Feature": clean_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(12)
    )
    imp_df.to_csv("feature_importances.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x="Importance", y="Feature", palette="magma")
    plt.title("Top 12 Most Predictive Features (Gradient Boosting Congestion Model)", fontsize=12, fontweight="bold")
    plt.xlabel("Relative Feature Importance", fontsize=11)
    plt.ylabel("Engineered Feature", fontsize=11)
    plt.tight_layout()
    fig3_path = os.path.join("ml_figures", "03_feature_importance_comparison.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info("Saved '%s'.", fig3_path)

    logger.info("Top 5 Features:\n%s", imp_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
