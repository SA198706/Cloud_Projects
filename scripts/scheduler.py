"""
scheduler.py
Sub-Objective 1.5 (DataOps): Automates preprocessing + EDA to run on a
recurring schedule (every 2 minutes), simulating a production DataOps
job (e.g., a Cloud Scheduler / Airflow DAG / cron-triggered Lambda in
a real deployment).

In production this would be deployed as:
  - AWS: EventBridge rule (rate: 2 minutes) -> Lambda/Step Function
  - Azure: Azure Data Factory scheduled trigger / Azure Functions Timer Trigger
  - GCP: Cloud Scheduler -> Cloud Function / Vertex AI Pipeline
  - Databricks: Jobs with a cron schedule

Here it's demonstrated locally with APScheduler for the assignment.
"""
from apscheduler.schedulers.blocking import BlockingScheduler
from pipeline_job import run_pipeline_once, logger

run_counter = {"n": 0}


def scheduled_job():
    run_counter["n"] += 1
    result = run_pipeline_once(run_id=run_counter["n"])
    print(f"[Run #{result['run_id']}] {result['status']} "
          f"| default_rate={result.get('default_rate')} "
          f"| duration={result['duration_sec']}s")


if __name__ == '__main__':
    scheduler = BlockingScheduler()
    scheduler.add_job(scheduled_job, 'interval', minutes=2, id='loan_pipeline_job',
                       next_run_time=None)  # first run fires after 2 min by default

    logger.info("Scheduler starting: pipeline will run every 2 minutes")
    print("DataOps scheduler started — pipeline runs every 2 minutes. Ctrl+C to stop.")

    # Run once immediately on startup, then let the interval trigger take over
    scheduled_job()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
