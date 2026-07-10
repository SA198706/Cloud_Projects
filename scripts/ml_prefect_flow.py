"""
ml_prefect_flow.py

Workflow

Load Data
    ↓
Preprocess Data
    ↓
Run EDA
    ↓
Train ML Models
    ↓
Evaluate Models
    ↓
Save Metrics
    ↓
Complete

Every stage is an individual Prefect Task so that
execution is visible in Prefect Cloud.
"""

from prefect import flow, task, get_run_logger

from preprocessing import load_data, preprocess
from eda import run_eda
from ml_pipeline import run_ml_pipeline


# --------------------------------------------------------------------
# Load Dataset
# --------------------------------------------------------------------

@task(
    name="Load Dataset",
    retries=2,
    retry_delay_seconds=10,
)
def load_dataset_task():

    logger = get_run_logger()

    df = load_data()

    logger.info(
        f"Dataset loaded successfully. Rows={len(df)}"
    )

    return df


# --------------------------------------------------------------------
# Preprocess Dataset
# --------------------------------------------------------------------

@task(
    name="Preprocess Dataset"
)
def preprocess_task(df):

    logger = get_run_logger()

    result = preprocess(df)

    logger.info(
        f"Rows after preprocessing : {result['rows']}"
    )

    logger.info(
        f"Default Rate : {result['default_rate']:.4f}"
    )

    logger.info(
        f"Missing Values After Cleaning : {result['missing_after']}"
    )

    return result


# --------------------------------------------------------------------
# Run EDA
# --------------------------------------------------------------------

@task(
    name="Exploratory Data Analysis"
)
def eda_task(preprocessed):

    logger = get_run_logger()

    eda_result = run_eda(
        preprocessed["df_clean"]
    )

    logger.info(
        f"EDA Completed."
    )

    logger.info(
        f"Generated {eda_result['plots_generated']} plots."
    )

    top_feature = list(
        eda_result["correlation_with_target"].items()
    )[0]

    logger.info(
        f"Top Correlated Feature : "
        f"{top_feature[0]} = {top_feature[1]:.4f}"
    )

    return eda_result


# --------------------------------------------------------------------
# Machine Learning Pipeline
# --------------------------------------------------------------------

@task(
    name="Machine Learning Pipeline"
)
def ml_pipeline_task():

    logger = get_run_logger()

    logger.info(
        "Starting Machine Learning Pipeline..."
    )

    result = run_ml_pipeline()

    logger.info(
        "Machine Learning Pipeline Completed."
    )

    logger.info(
        f"Pipeline Status : {result['pipeline_status']}"
    )

    if result["pipeline_status"] == "SUCCESS":

        logger.info(
            f"Best Model : {result['best_model']}"
        )

        logger.info(
            f"Random Forest Accuracy : "
            f"{result['random_forest_accuracy']}"
        )

        logger.info(
            f"Logistic Accuracy : "
            f"{result['logistic_accuracy']}"
        )

    return result


# --------------------------------------------------------------------
# Final Summary
# --------------------------------------------------------------------

@task(
    name="Pipeline Summary"
)
def summary_task(pre_result, eda_result, ml_result):

    logger = get_run_logger()

    logger.info("=" * 60)

    logger.info("PIPELINE EXECUTION SUMMARY")

    logger.info("=" * 60)

    logger.info(
        f"Rows Processed : {pre_result['rows']}"
    )

    logger.info(
        f"Default Rate : {pre_result['default_rate']:.4f}"
    )

    logger.info(
        f"Plots Generated : "
        f"{eda_result['plots_generated']}"
    )

    logger.info(
        f"Pipeline Status : "
        f"{ml_result['pipeline_status']}"
    )

    if ml_result["pipeline_status"] == "SUCCESS":

        logger.info(
            f"Best Model : "
            f"{ml_result['best_model']}"
        )

        logger.info(
            f"Random Forest Accuracy : "
            f"{ml_result['random_forest_accuracy']}"
        )

        logger.info(
            f"Logistic Accuracy : "
            f"{ml_result['logistic_accuracy']}"
        )

    logger.info("=" * 60)

    return {

        "rows": pre_result["rows"],

        "default_rate": pre_result["default_rate"],

        "plots_generated": eda_result["plots_generated"],

        "pipeline_status": ml_result["pipeline_status"],

        "best_model":
            ml_result.get("best_model", "N/A"),
    }


# --------------------------------------------------------------------
# Prefect Flow
# --------------------------------------------------------------------

@flow(
    name="Loan Default ML Pipeline",
    log_prints=True,
)
def loan_default_ml_pipeline():

    df = load_dataset_task()

    pre_result = preprocess_task(df)

    eda_result = eda_task(pre_result)

    ml_result = ml_pipeline_task()

    summary = summary_task(
        pre_result,
        eda_result,
        ml_result,
    )

    print("\n")

    print("=" * 60)

    print("PIPELINE EXECUTED SUCCESSFULLY")

    print("=" * 60)

    print(summary)

    print("=" * 60)

    return summary


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        # Deploy to Prefect Cloud with a 2-minute cron schedule.
        # Requires: prefect cloud login (run once first)
        # Keep this process running — it polls for scheduled work.
        loan_default_ml_pipeline.serve(
            name="loan-default-ml-deployment",
            cron="*/2 * * * *",
            tags=["loan", "mlops", "aimlczg549"],
        )
    else:
        # One-off local run (no deployment)
        loan_default_ml_pipeline()
