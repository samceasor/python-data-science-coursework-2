"""
PYTHON FOR DATA SCIENCE COURSEWORK 2
Project: Rainfall Occurrence Classification in Kampala

Run collect_data.py first, then run:
    python src/analysis.py

The script:
1. Loads and profiles the raw data.
2. Cleans and preprocesses it.
3. Creates a rainfall-occurrence target.
4. Trains Logistic Regression and Decision Tree models.
5. Compares them with a majority-class baseline.
6. Saves tables, plots, prepared data and metrics to outputs/.
"""

from pathlib import Path
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "kampala_weather_raw.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42


def make_rain_target(df, threshold=0.1):
    """Create a binary rainfall-occurrence target from precipitation."""
    result = df.copy()
    result["rain_occurred"] = (
        result["precipitation_sum"] >= threshold
    ).astype(int)
    return result


def validate_columns(df, required):
    """Check that all required columns are present before analysis."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_weather_data(df):
    """Clean types, duplicates and impossible values reproducibly."""
    cleaned = df.copy()
    cleaned["time"] = pd.to_datetime(cleaned["time"], errors="coerce")

    numeric_cols = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
        "pressure_msl_mean",
    ]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    before_duplicates = len(cleaned)
    cleaned = cleaned.drop_duplicates().copy()
    duplicate_count = before_duplicates - len(cleaned)

    # Convert impossible values to missing values. The modelling pipeline
    # imputes missing predictor values without altering the raw CSV.
    cleaned.loc[
        ~cleaned["relative_humidity_2m_mean"].between(0, 100),
        "relative_humidity_2m_mean",
    ] = np.nan
    cleaned.loc[
        cleaned["precipitation_sum"] < 0, "precipitation_sum"
    ] = np.nan
    cleaned.loc[
        cleaned["windspeed_10m_max"] < 0, "windspeed_10m_max"
    ] = np.nan

    return cleaned, duplicate_count


def add_features(df):
    """Create calendar features from the date column."""
    result = df.copy()
    result["month"] = result["time"].dt.month
    result["day_of_year"] = result["time"].dt.dayofyear
    return result


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Fit a model and calculate classification metrics."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    else:
        auc = np.nan

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred, zero_division=0),
        "roc_auc": auc,
    }
    return model, pred, metrics


def save_confusion_matrix(model, X_test, y_test, title, output_name):
    """Save one model's confusion matrix as a separate image."""
    display = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["No rain", "Rain"],
        labels=[0, 1],
        values_format="d",
    )
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(OUT / output_name, dpi=200)
    plt.close(display.figure_)


def main():
    if not RAW.exists():
        raise FileNotFoundError(
            "Raw data not found. Run 'python collect_data.py' first."
        )

    raw = pd.read_csv(RAW)
    required = [
        "time",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
        "pressure_msl_mean",
    ]
    validate_columns(raw, required)

    # Profile the raw data before cleaning.
    profile = pd.DataFrame(
        {
            "column": raw.columns,
            "dtype": [str(value) for value in raw.dtypes],
            "missing_values": [int(value) for value in raw.isna().sum()],
            "unique_values": [
                int(value) for value in raw.nunique(dropna=True)
            ],
        }
    )
    profile.to_csv(OUT / "dataset_profile.csv", index=False)

    before_shape = raw.shape
    cleaned, duplicate_count = clean_weather_data(raw)
    after_clean_shape = cleaned.shape

    cleaned = cleaned.dropna(subset=["time"]).sort_values("time")
    prepared = make_rain_target(cleaned)
    prepared = add_features(prepared)

    # Precipitation is used to define the target, so it is deliberately
    # excluded from predictors to avoid target leakage.
    features = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "windspeed_10m_max",
        "pressure_msl_mean",
        "month",
        "day_of_year",
    ]
    X = prepared[features]
    y = prepared["rain_occurred"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    tree = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    baseline = DummyClassifier(strategy="most_frequent")
    models = [
        ("Majority baseline", baseline),
        ("Logistic Regression", logistic),
        ("Decision Tree", tree),
    ]

    all_metrics = []
    fitted = {}
    predictions = {}

    for name, model in models:
        fitted_model, pred, metrics = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        all_metrics.append(metrics)
        fitted[name] = fitted_model
        predictions[name] = pred

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT / "model_metrics.csv", index=False)

    # Select the best non-baseline model by F1-score and save its matrix.
    comparison = metrics_df[metrics_df["model"] != "Majority baseline"]
    best_name = (
        comparison.sort_values("f1_score", ascending=False)
        .iloc[0]["model"]
    )
    best_model = fitted[best_name]
    best_cm = confusion_matrix(
        y_test, predictions[best_name], labels=[0, 1]
    )
    best_display = ConfusionMatrixDisplay(
        confusion_matrix=best_cm,
        display_labels=["No rain", "Rain"],
    )
    best_display.plot(values_format="d")
    best_display.ax_.set_title(f"Confusion Matrix - {best_name}")
    best_display.figure_.tight_layout()
    best_display.figure_.savefig(OUT / "confusion_matrix.png", dpi=200)
    plt.close(best_display.figure_)

    # Save separate matrices so the two model results are easy to inspect.
    save_confusion_matrix(
        fitted["Logistic Regression"],
        X_test,
        y_test,
        "Confusion Matrix - Logistic Regression",
        "confusion_matrix_logistic.png",
    )
    save_confusion_matrix(
        fitted["Decision Tree"],
        X_test,
        y_test,
        "Confusion Matrix - Decision Tree",
        "confusion_matrix_decision_tree.png",
    )

    # Visualize the proportion of rainy days by month.
    plt.figure()
    prepared.groupby("month")["rain_occurred"].mean().plot(kind="bar")
    plt.ylabel("Proportion of rainy days")
    plt.xlabel("Month")
    plt.title("Rainfall Occurrence by Month")
    plt.tight_layout()
    plt.savefig(OUT / "rainfall_by_month.png", dpi=200)
    plt.close()

    # Save the prepared dataset and reproducibility information.
    prepared.to_csv(OUT / "kampala_weather_prepared.csv", index=False)

    reproducibility = pd.DataFrame(
        {
            "item": [
                "Python",
                "pandas",
                "NumPy",
                "scikit-learn",
                "Matplotlib",
                "random_state",
                "raw_shape",
                "clean_shape",
                "duplicates_removed",
                "best_model_by_f1",
            ],
            "value": [
                platform.python_version(),
                pd.__version__,
                np.__version__,
                __import__("sklearn").__version__,
                __import__("matplotlib").__version__,
                RANDOM_STATE,
                str(before_shape),
                str(after_clean_shape),
                duplicate_count,
                best_name,
            ],
        }
    )
    reproducibility.to_csv(OUT / "reproducibility.csv", index=False)

    print("\n=== DATASET ===")
    print("Raw shape:", before_shape)
    print("Clean shape:", after_clean_shape)
    print("Duplicates removed:", duplicate_count)
    print("\n=== TARGET CLASS DISTRIBUTION (%) ===")
    print((y.value_counts(normalize=True).sort_index() * 100).round(2))
    print("\n=== MODEL RESULTS ===")
    print(metrics_df.to_string(index=False))
    print("\nBest model by F1:", best_name)
    print("\nOutputs saved in:", OUT.resolve())


if __name__ == "__main__":
    main()

