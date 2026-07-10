"""
model_monitor.py

Sub-Objective 2.3 & 2.4

Model Evaluation + MLOps Monitoring

This module:
1. Loads trained models
2. Evaluates Logistic Regression & Random Forest
3. Calculates evaluation metrics
4. Selects the best model
5. Saves metrics for monitoring
6. Saves best model
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from model_training import train_models

# ---------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"

MODEL_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

BEST_MODEL = MODEL_DIR / "best_model.pkl"
METRICS_FILE = METRICS_DIR / "model_metrics.json"


# ---------------------------------------------------------------------
# Evaluate Model
# ---------------------------------------------------------------------


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {

        "accuracy":
            round(
                accuracy_score(y_test, predictions),
                4,
            ),

        "precision":
            round(
                precision_score(y_test, predictions),
                4,
            ),

        "recall":
            round(
                recall_score(y_test, predictions),
                4,
            ),

        "f1_score":
            round(
                f1_score(y_test, predictions),
                4,
            ),

        "roc_auc":
            round(
                roc_auc_score(y_test, probabilities),
                4,
            ),

        "confusion_matrix":
            confusion_matrix(
                y_test,
                predictions,
            ).tolist(),

        "classification_report":
            classification_report(
                y_test,
                predictions,
                output_dict=True,
            ),
    }

    return metrics


# ---------------------------------------------------------------------
# Save Metrics
# ---------------------------------------------------------------------


def save_metrics(metrics):

    with open(METRICS_FILE, "w") as fp:

        json.dump(
            metrics,
            fp,
            indent=4,
        )


# ---------------------------------------------------------------------
# Compare Models
# ---------------------------------------------------------------------


def compare_models(logistic_metrics, rf_metrics):

    if rf_metrics["accuracy"] >= logistic_metrics["accuracy"]:

        best = "Random Forest"

        shutil.copy(
            MODEL_DIR / "random_forest.pkl",
            BEST_MODEL,
        )

    else:

        best = "Logistic Regression"

        shutil.copy(
            MODEL_DIR / "logistic_regression.pkl",
            BEST_MODEL,
        )

    return best


# ---------------------------------------------------------------------
# Monitor Models
# ---------------------------------------------------------------------


def monitor_models():
    """
    Complete Monitoring Pipeline
    """

    training_output = train_models()

    logistic_model = training_output["logistic_model"]

    random_forest_model = training_output["random_forest_model"]

    X_test = training_output["X_test"]

    y_test = training_output["y_test"]

    logistic_metrics = evaluate_model(
        logistic_model,
        X_test,
        y_test,
    )

    random_forest_metrics = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
    )

    best_model = compare_models(
        logistic_metrics,
        random_forest_metrics,
    )

    output = {

        "timestamp":
            datetime.utcnow().isoformat(),

        "dataset": "Loan Default Dataset",

        "training_rows":
            training_output["train_rows"],

        "testing_rows":
            training_output["test_rows"],

        "best_model":
            best_model,

        "logistic_regression":
            logistic_metrics,

        "random_forest":
            random_forest_metrics,
    }

    save_metrics(output)

    return output


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    metrics = monitor_models()

    print("\n========================================")
    print("MODEL EVALUATION COMPLETE")
    print("========================================\n")

    print("Best Model :", metrics["best_model"])

    print("\nLogistic Regression")

    print(
        metrics["logistic_regression"]
    )

    print("\nRandom Forest")

    print(
        metrics["random_forest"]
    )

    print("\nMetrics Saved")

    print(METRICS_FILE)

    print("\nBest Model Saved")

    print(BEST_MODEL)
