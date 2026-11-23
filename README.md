# Smart City Traffic Intelligence Capstone Project

An end-to-end data analytics, data engineering, and machine learning solution analyzing westbound Interstate 94 (I-94) traffic volume between Minneapolis and St. Paul (48,204 hourly observations, 2012–2017).

---

## Project Overview & Key Findings

- **Primary Congestion Driver:** Commuter schedules—rather than weather fluctuations—drive heavy congestion (`traffic_volume > 5,500` vehicles/hour, occurring in `14.73%` of recorded hours), with sharp weekday peaks during morning (`07:00–08:00`) and evening (`16:00–17:00`) rush hours.
- **Weather & Temperature Relationship:** Ambient temperature exhibits only a weak positive linear correlation with traffic volume ($r = 0.1303$), driven by daytime rush hours occurring during warmer daylight hours.
- **Data Quality Remediation:** Automated ingestion validation detected and imputed `0 K` (`-273.15 °C`) temperature sensor malfunctions and rainfall outliers using monthly medians, and removed exact duplicate logs.
- **Predictive Modeling Performance (Chronological 80/20 Split):**
  - **Traffic Volume Forecasting (`RandomForestRegressor`):** Achieved **$R^2 = 0.9838$** (`RMSE = 250.20`, `MAE = 154.49` vehicles/hour), reducing baseline Ridge Regression error by 54.2%.
  - **Congestion Classification (`GradientBoostingClassifier`):** Achieved **97.13% Accuracy**, **0.8992 F1-Score**, **93.05% Recall**, and **0.9940 ROC-AUC**.

---

## Repository Structure

```text
Smart-City-Traffic-Capstone-Project/
├── Data/
│   └── Metro_Interstate_Traffic_Volume.csv       # Raw hourly traffic & weather dataset (48,204 rows)
├── Part1_data_analytics/
│   ├── task1_traffic_analysis.sql                # SQLite analytical queries (annual trends, holidays)
│   ├── traffic_analysis.db                       # SQLite relational database
│   ├── Part1_Tasks2_3_Stats_Probability.ipynb    # Descriptive statistics, correlation & probability modeling
│   ├── traffic_dashboard.pbix                    # Interactive Power BI Traffic Intelligence Dashboard
│   └── insights_report.pdf                       # Executive Part 1 analytics & BI report
├── Part2_python/
│   ├── pipeline.py                               # Automated ETL, schema validation & outlier cleaning script
│   ├── pipeline.log                              # Execution & audit trail log
│   ├── feature_engineering.py                    # Temporal, cyclical, weather & lag feature creation
│   ├── visualizations.py                         # Automated 300-DPI exploratory visualization generator
│   ├── figures/                                  # Static EDA figures (hourly, weather, temp, heatmap)
│   └── eda_insights.md                           # Exploratory data analysis summary
├── Part3_machine_learning/
│   ├── train_regression.py                       # Time-aware regression pipeline (Ridge, RF, GBR)
│   ├── train_classification.py                   # Time-aware congestion classification pipeline
│   ├── model_evaluation.py                       # Diagnostic charts & feature importance extraction
│   ├── models/                                   # Serialized best models (.joblib)
│   ├── ml_figures/                               # Parity plot, Confusion Matrix, ROC curve & importance plot
│   ├── regression_metrics.csv                    # Regression benchmark metrics
│   ├── classification_metrics.csv                # Classification benchmark metrics
│   ├── feature_importances.csv                   # Ranked feature importances
│   ├── ml_pipeline.log                           # ML training & evaluation log
│   └── capstone_final_report.md                  # End-to-end executive capstone report
└── README.md                                     # Project overview & reproducibility guide

How to Reproduce the Pipelines

Install Dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn joblib

Run Part 2 Data Engineering & EDA:

cd Part2_python
python pipeline.py
python feature_engineering.py
python visualizations.py

Run Part 3 Machine Learning Training & Evaluation:

cd ../Part3_machine_learning
python train_regression.py
python train_classification.py
python model_evaluation.py
