"""
deploy_prefect.py
Sub-Objective 1.5 (DataOps) — deploy the loan pipeline to Prefect Cloud.

This wraps the existing `run_pipeline_once()` job in a Prefect flow and
deploys it with `.serve()`. `.serve()` is the simplest deployment method:
it registers the deployment in Prefect Cloud AND starts a local process
that runs the flow on schedule (every 2 minutes). No work pools or workers
to configure.

Run it from the `scripts/` folder so the imports below resolve:

    cd scripts
    prefect cloud login            # one-time: connect to your workspace
    python api_access/deploy_prefect.py   # (or wherever this file lives)

Leave the terminal running during your demo — that process is what
executes the scheduled runs. Stop it with Ctrl+C.
"""
import itertools

from prefect import flow, get_run_logger

# Reuse the pipeline job the team already wrote (Sub-Objective 1.5).
# NOTE: run from the scripts/ directory so this import works.
from pipeline_job import run_pipeline_once

# Simple incrementing run id for the life of the serving process.
_run_counter = itertools.count(1)


@flow(name="loan-default-pipeline")
def loan_pipeline_flow():
    """One scheduled pipeline run (preprocessing + EDA + logging)."""
    logger = get_run_logger()
    run_id = next(_run_counter)
    logger.info(f"Starting pipeline run #{run_id}")
    record = run_pipeline_once(run_id=run_id)
    logger.info(f"Run #{run_id} finished with status={record.get('status')}")
    return record


if __name__ == "__main__":
    # interval=120 seconds == every 2 minutes (matches the assignment).
    loan_pipeline_flow.serve(
        name="loan-pipeline-deployment",
        interval=120,
        tags=["loan", "dataops"],
    )
