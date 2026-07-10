"""
deploy_to_prefect_cloud.py
AIMLCZG549 — Sub-Objective 1.5 / 2 / DataOps + MLOps

Deploys the full loan-default ML pipeline to Prefect Cloud with a
2-minute cron schedule. Each task is independently observable in the
Prefect Cloud dashboard (logs, duration, success/failure).

Before running:
    1. prefect cloud login --key <your-api-key>
       (get a key at https://app.prefect.cloud/my/api-keys)
    2. python deploy_to_prefect_cloud.py

Leave this process running — it registers the deployment in Prefect
Cloud and executes scheduled runs locally (every 2 minutes). Stop
with Ctrl+C.
"""

from prefect import flow, task, get_run_logger
from preprocessing import load_data, preprocess
from eda import run_eda
from ml_pipeline import run_ml_pipeline


@task(name="Load Dataset", retries=2, retry_delay_seconds=10)
def load_dataset_task():
    logger = get_run_logger()
    df = load_data()
    logger.info(f"Dataset loaded: {len(df)} rows")
    return df


@task(name="Preprocess Dataset")
def preprocess_task(df):
    logger = get_run_logger()
    result = preprocess(df)
    logger.info(f"Rows: {result['rows']} | Default rate: {result['default_rate']:.4f}")
    logger.info(f"Missing values after cleaning: {result['missing_after']}")
    return result


@task(name="Exploratory Data Analysis")
def eda_task(preprocessed):
    logger = get_run_logger()
    eda_result = run_eda(preprocessed["df_clean"])
    top_feature = list(eda_result["correlation_with_target"].items())[0]
    logger.info(f"EDA complete — {eda_result['plots_generated']} plots generated")
    logger.info(f"Top correlated feature: {top_feature[0]} = {top_feature[1]:.4f}")
    return eda_result


@task(name="Machine Learning Pipeline")
def ml_pipeline_task():
    logger = get_run_logger()
    logger.info("Starting ML pipeline (train + evaluate + select best model)...")
    result = run_ml_pipeline()
    logger.info(f"Status: {result['pipeline_status']}")
    if result["pipeline_status"] == "SUCCESS":
        logger.info(f"Best model: {result['best_model']}")
        logger.info(f"Logistic Regression accuracy: {result['logistic_accuracy']}")
        logger.info(f"Random Forest accuracy:       {result['random_forest_accuracy']}")
    return result


@task(name="Pipeline Summary")
def summary_task(pre_result, eda_result, ml_result):
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Rows processed:    {pre_result['rows']}")
    logger.info(f"Default rate:      {pre_result['default_rate']:.4f}")
    logger.info(f"Plots generated:   {eda_result['plots_generated']}")
    logger.info(f"Pipeline status:   {ml_result['pipeline_status']}")
    if ml_result["pipeline_status"] == "SUCCESS":
        logger.info(f"Best model:        {ml_result['best_model']}")
        logger.info(f"LR accuracy:       {ml_result['logistic_accuracy']}")
        logger.info(f"RF accuracy:       {ml_result['random_forest_accuracy']}")
    logger.info("=" * 60)
    return {
        "rows": pre_result["rows"],
        "default_rate": pre_result["default_rate"],
        "plots_generated": eda_result["plots_generated"],
        "pipeline_status": ml_result["pipeline_status"],
        "best_model": ml_result.get("best_model", "N/A"),
        "logistic_accuracy": ml_result.get("logistic_accuracy"),
        "random_forest_accuracy": ml_result.get("random_forest_accuracy"),
    }


@flow(name="Loan Default ML Pipeline", log_prints=True)
def loan_default_ml_pipeline():
    """
    Full end-to-end pipeline:
      Load → Preprocess → EDA → Train+Evaluate ML → Summary
    Runs every 2 minutes as a Prefect Cloud deployment.
    """
    df = load_dataset_task()
    pre_result = preprocess_task(df)
    eda_result = eda_task(pre_result)
    ml_result = ml_pipeline_task()
    summary = summary_task(pre_result, eda_result, ml_result)
    return summary


if __name__ == "__main__":
    print("Deploying 'Loan Default ML Pipeline' to Prefect Cloud...")
    print("Schedule: every 2 minutes (cron: */2 * * * *)")
    print("Press Ctrl+C to stop the server.\n")

    loan_default_ml_pipeline.serve(
        name="loan-default-ml-deployment",
        cron="*/2 * * * *",
        tags=["loan", "mlops", "aimlczg549"],
    )
