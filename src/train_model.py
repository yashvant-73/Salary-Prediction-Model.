from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Salary Data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "salary_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURES = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]
NUMERIC_FEATURES = ["Age", "Years of Experience"]
CATEGORICAL_FEATURES = ["Gender", "Education Level", "Job Title"]
TARGET = "Salary"


def make_pipeline(regressor):
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def main():
    df = pd.read_csv(DATA_PATH)
    missing_columns = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")
    df = df[FEATURES + [TARGET]].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    print(f"Dataset rows: {len(df)}; exact duplicate rows: {df.duplicated().sum()}")
    # Exact duplicates are removed so repeated copies cannot appear on both sides of the split.
    df = df.drop_duplicates().dropna(subset=[TARGET])
    if len(df) < 10:
        raise ValueError("At least 10 rows with numeric salary are required.")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted = {}
    for name, regressor in models.items():
        pipeline = make_pipeline(regressor)
        cv_scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=-1,
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        row = {
            "model": name,
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
            "r2": float(r2_score(y_test, predictions)),
            "cv_mae_mean": float(-cv_scores["test_mae"].mean()),
            "cv_r2_mean": float(cv_scores["test_r2"].mean()),
        }
        results.append(row)
        fitted[name] = pipeline
    # Select on cross-validated MAE; held-out test metrics remain an unbiased final report.
    best = min(results, key=lambda item: item["cv_mae_mean"])
    selected = fitted[best["model"]]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, MODEL_PATH)
    metrics = {
        "selected_model": best["model"],
        "feature_columns": FEATURES,
        "target": TARGET,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "duplicate_rows_removed": int(pd.read_csv(DATA_PATH).duplicated().sum()),
        "salary_min": float(y.min()),
        "salary_max": float(y.max()),
        "job_title_counts": df["Job Title"].value_counts().to_dict(),
        "comparison": results,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Selected by mean CV MAE: {best['model']}")
    print(f"Saved pipeline: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
