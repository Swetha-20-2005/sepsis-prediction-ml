"""Train and evaluate a Random Forest on a synthetic demonstration dataset."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/sepsis_synthetic.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "sepsis_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURES = ["HR", "Temp", "Resp", "Lactate", "WBC"]
TARGET = "SepsisLabel"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python generate_data.py` first."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("SepsisLabel must contain only 0 and 1.")
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    X = df[FEATURES]
    y = df[TARGET]
    if y.nunique() < 2:
        raise ValueError("Target must contain both classes (0 and 1).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=2
        ))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset_type": "synthetic demonstration data",
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Model trained and saved to:", MODEL_PATH)
    print("\nEvaluation on held-out synthetic test data:")
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"{key}: {metrics[key]:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print("\nReminder: these results reflect synthetic data only, not clinical performance.")

if __name__ == "__main__":
    main()
