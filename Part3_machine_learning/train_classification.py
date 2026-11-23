import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "ml_pipeline.log") -> None:
    """Append classification logs to ml_pipeline.log and stream to console."""
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
    logger.info("Starting Part 3 Task 2: Binary Congestion Classification Modeling.")

    input_path = "ml_ready_traffic_data.csv"
    try:
        df = pd.read_csv(input_path)
        logger.info("Loaded '%s' (%d rows, %d columns).", input_path, df.shape[0], df.shape[1])
    except FileNotFoundError:
        logger.error("Dataset '%s' not found. Run train_regression.py first.", input_path, exc_info=True)
        sys.exit(1)

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
    y = df["is_congested"]

    # Chronological 80/20 split matching regression pipeline
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(
        "Chronological split complete: Train=%d rows (Congested=%.2f%%), Test=%d rows (Congested=%.2f%%).",
        len(X_train),
        y_train.mean() * 100,
        len(X_test),
        y_test.mean() * 100,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    classifiers = {
        "Logistic_Regression_Baseline": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random_Forest_Classifier": RandomForestClassifier(
            n_estimators=150, max_depth=16, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient_Boosting_Classifier": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42
        ),
    }

    os.makedirs("models", exist_ok=True)
    results = []
    best_f1 = -np.inf
    best_clf_name = ""
    best_clf_pipeline = None

    for name, clf in classifiers.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds))
        rec = float(recall_score(y_test, preds))
        f1 = float(f1_score(y_test, preds))
        auc = float(roc_auc_score(y_test, probs))

        results.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(auc, 4),
            }
        )
        logger.info(
            "%s -> Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | ROC-AUC: %.4f",
            name,
            acc,
            prec,
            rec,
            f1,
            auc,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_clf_name = name
            best_clf_pipeline = pipe

    results_df = pd.DataFrame(results).sort_values("F1_Score", ascending=False)
    results_df.to_csv("classification_metrics.csv", index=False)

    best_preds = best_clf_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)
    logger.info("Best Classifier (%s) Confusion Matrix:\n%s", best_clf_name, cm)
    logger.info("Detailed Classification Report:\n%s", classification_report(y_test, best_preds))

    best_clf_path = os.path.join("models", "best_classification_model.joblib")
    joblib.dump(best_clf_pipeline, best_clf_path)
    logger.info(
        "Best Classification Model: '%s' (F1=%.4f) saved to '%s'.",
        best_clf_name,
        best_f1,
        best_clf_path,
    )


if __name__ == "__main__":
    main()
