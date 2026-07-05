"""
prefect_flow.py
Wraps the existing preprocessing.py + eda.py pipeline into a Prefect flow.
Each @task becomes independently observable (its own logs, retries,
duration, success/failure) inside Prefect Cloud's UI.
"""
from prefect import flow, task, get_run_logger
from preprocessing import load_data, preprocess
from eda import run_eda


@task(name="load-raw-data", retries=2, retry_delay_seconds=10)
def load_data_task():
    logger = get_run_logger()
    df = load_data()
    logger.info(f"Loaded {len(df)} raw loan applications")
    return df


@task(name="preprocess-data")
def preprocess_task(df):
    logger = get_run_logger()
    result = preprocess(df)
    logger.info(f"Preprocessing complete: {result['rows']} rows, "
                f"missing values before imputation: {result['missing_before']}, "
                f"default rate: {result['default_rate']:.4f}")
    return result


@task(name="run-eda")
def eda_task(pre_result):
    logger = get_run_logger()
    eda_out = run_eda(pre_result['df_clean'])
    top_feature = list(eda_out['correlation_with_target'].items())[0]
    logger.info(f"EDA complete: {eda_out['plots_generated']} plots generated, "
                f"top correlated feature = {top_feature[0]} ({top_feature[1]:.3f})")
    return eda_out


@flow(name="loan-default-pipeline", log_prints=True)
def loan_pipeline():
    df = load_data_task()
    pre = preprocess_task(df)
    eda_out = eda_task(pre)
    print(f"Pipeline run complete | rows={pre['rows']} "
          f"| default_rate={pre['default_rate']:.4f} "
          f"| plots={eda_out['plots_generated']}")
    return {"rows": pre['rows'], "default_rate": pre['default_rate']}


if __name__ == "__main__":
    # For a one-off local run:
    #  loan_pipeline()
    #
    # For a long-running scheduled deployment (every 2 minutes),
    # run this file and leave it running — it registers with
    # Prefect Cloud and polls for scheduled work:
    #loan_pipeline.serve(
    #    name="loan-pipeline-every-2min",
    #    cron="*/2 * * * *"
    #)
    loan_pipeline.serve(name="loan-pipeline-deployment")
