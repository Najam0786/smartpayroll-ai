# src/models/attrition/train.py
"""
Attrition prediction model training.
Trains XGBoost with SMOTE for class imbalance.
Tracks all experiments with MLflow.
"""

import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a trained model and return metrics.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        model_name: Name for logging

    Returns:
        dict of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_proba), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
    }

    print(f"\n{'='*55}")
    print(f"{model_name} — Results")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    return metrics


def train_baseline(X_train, y_train, X_test, y_test) -> dict:
    """Train Logistic Regression baseline."""
    print("\n[1/2] Training Logistic Regression baseline...")

    with mlflow.start_run(run_name="logistic-regression-baseline"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
        })

        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model, X_test, y_test, "Logistic Regression"
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    return {"model": model, "metrics": metrics}


def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    """Train XGBoost with SMOTE oversampling."""
    print("\n[2/2] Training XGBoost with SMOTE...")

    with mlflow.start_run(run_name="xgboost-smote"):
        params = {
            "model": "XGBoost",
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "scale_pos_weight": 5,
            "sampling": "SMOTE",
        }
        mlflow.log_params(params)

        # Apply SMOTE to training data only
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE — 0: {(y_resampled==0).sum()}, "
              f"1: {(y_resampled==1).sum()}")

        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=5,
            random_state=42,
            eval_metric="auc",
            verbosity=0,
        )
        model.fit(X_resampled, y_resampled)

        metrics = evaluate_model(
            model, X_test, y_test, "XGBoost + SMOTE"
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    return {"model": model, "metrics": metrics}


def run_training():
    """Run full training pipeline."""
    from src.data.ingest import load_hr_data
    from src.data.clean import clean_hr_data
    from src.features.feature_engineering import (
        encode_categorical, select_features, split_data
    )

    mlflow.set_experiment("smartpayroll-attrition")

    # Load and prepare data
    df = clean_hr_data(load_hr_data())
    df_encoded = encode_categorical(df)
    X, y = select_features(df_encoded)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    baseline = train_baseline(X_train, y_train, X_test, y_test)
    xgb = train_xgboost(X_train, y_train, X_test, y_test)

    # Compare
    print(f"\n{'='*55}")
    print("MODEL COMPARISON")
    print(f"{'='*55}")
    for name, result in [("Baseline (LR)", baseline),
                          ("XGBoost+SMOTE", xgb)]:
        m = result["metrics"]
        print(f"{name:15s} | AUC: {m['auc_roc']:.3f} "
              f"| F1: {m['f1']:.3f} "
              f"| Recall: {m['recall']:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()