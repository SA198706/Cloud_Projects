"""
ml_pipeline.py

Sub-Objective 2

Complete Machine Learning Pipeline

Pipeline Steps
--------------
1. Load Dataset
2. Preprocess Dataset
3. Train Logistic Regression
4. Train Random Forest
5. Evaluate Models
6. Select Best Model
7. Save Metrics
8. Save Best Model

This module acts as the single entry point for the ML pipeline.
"""

from pathlib import Path
from datetime import datetime
import json
import logging

from model_training import train_models
from model_monitor import monitor_models

# -------------------------------------------------------------------
# Directories
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PIPELINE_LOG = LOG_DIR / "ml_pipeline.log"

# -------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------

logger = logging.getLogger("ml_pipeline")
logger.setLevel(logging.INFO)

if not logger.handlers:

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(PIPELINE_LOG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# -------------------------------------------------------------------
# ML Pipeline
# -------------------------------------------------------------------


def run_ml_pipeline():
    """
    Executes the complete Machine Learning pipeline.
    """

    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info("Machine Learning Pipeline Started")
    logger.info("=" * 70)

    try:

        # -------------------------------------------------------------
        # Step 1 : Train Models
        # -------------------------------------------------------------

        logger.info("Training models...")

        training_output = train_models()

        logger.info(
            "Training completed successfully."
        )

        logger.info(
            f"Training Rows : {training_output['train_rows']}"
        )

        logger.info(
            f"Testing Rows : {training_output['test_rows']}"
        )

        # -------------------------------------------------------------
        # Step 2 : Evaluate Models
        # -------------------------------------------------------------

        logger.info("Evaluating models...")

        monitoring_output = monitor_models()

        logger.info("Evaluation completed.")

        logger.info(
            f"Best Model : {monitoring_output['best_model']}"
        )

        # -------------------------------------------------------------
        # Step 3 : Pipeline Summary
        # -------------------------------------------------------------

        end_time = datetime.now()

        duration = round(
            (end_time - start_time).total_seconds(),
            2,
        )

        summary = {

            "pipeline_status": "SUCCESS",

            "started_at":
                start_time.isoformat(),

            "completed_at":
                end_time.isoformat(),

            "duration_seconds":
                duration,

            "training_rows":
                monitoring_output["training_rows"],

            "testing_rows":
                monitoring_output["testing_rows"],

            "best_model":
                monitoring_output["best_model"],

            "logistic_accuracy":
                monitoring_output[
                    "logistic_regression"
                ]["accuracy"],

            "random_forest_accuracy":
                monitoring_output[
                    "random_forest"
                ]["accuracy"],
        }

        logger.info(json.dumps(summary, indent=4))

        logger.info("=" * 70)
        logger.info("Machine Learning Pipeline Completed")
        logger.info("=" * 70)

        return summary

    except Exception as ex:

        logger.exception(
            "Machine Learning Pipeline Failed."
        )

        return {

            "pipeline_status": "FAILED",

            "error": str(ex),

            "timestamp": datetime.now().isoformat(),
        }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":

    result = run_ml_pipeline()

    print("\n")

    print("=" * 60)

    print("ML PIPELINE SUMMARY")

    print("=" * 60)

    print(json.dumps(result, indent=4))

    print("=" * 60)
