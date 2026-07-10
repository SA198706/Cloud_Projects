"""
pipeline_job.py
Sub-Objective 1.5 (DataOps): One complete pipeline run.
Wraps preprocessing.py + eda.py into a single job, captures key metrics
and logs them (both a human-readable log file and a machine-readable
JSONL file that the dashboard reads from).
"""
import json
import time
import logging
from datetime import datetime, timezone

from pathlib import Path
from preprocessing import load_data, preprocess
from eda import run_eda

LOG_DIR = str(Path(__file__).resolve().parent.parent / "logs")
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
JSONL_LOG = f'{LOG_DIR}/pipeline_log.jsonl'
TEXT_LOG = f'{LOG_DIR}/pipeline.log'

logging.basicConfig(
    filename=TEXT_LOG,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('loan_pipeline')


def run_pipeline_once(run_id: int) -> dict:
    start = time.time()
    ts = datetime.now(timezone.utc).isoformat()
    logger.info(f"Run #{run_id} started")

    try:
        # Simulate a new incoming batch of loan applications each run
        # (in production this would be the last N minutes of new records)
        full_df = load_data()
        df = full_df.sample(frac=0.15, random_state=run_id).reset_index(drop=True)
        pre = preprocess(df)
        eda_out = run_eda(pre['df_clean'])

        duration = round(time.time() - start, 3)
        top_driver = list(eda_out['correlation_with_target'].items())[0]

        record = {
            "run_id": run_id,
            "timestamp_utc": ts,
            "status": "SUCCESS",
            "duration_sec": duration,
            "rows_processed": pre['rows'],
            "missing_values_imputed": sum(pre['missing_before'].values()),
            "default_rate": round(pre['default_rate'], 4),
            "top_correlated_feature": top_driver[0],
            "top_correlation_value": round(top_driver[1], 4),
            "plots_generated": eda_out['plots_generated'],
            "default_rate_by_grade": {k: round(v, 4) for k, v in eda_out['default_rate_by_grade'].items()},
        }
        logger.info(f"Run #{run_id} SUCCESS in {duration}s | rows={pre['rows']} "
                     f"| default_rate={record['default_rate']} | imputed={record['missing_values_imputed']}")

    except Exception as e:
        duration = round(time.time() - start, 3)
        record = {
            "run_id": run_id,
            "timestamp_utc": ts,
            "status": "FAILED",
            "duration_sec": duration,
            "error": str(e)
        }
        logger.error(f"Run #{run_id} FAILED after {duration}s: {e}")

    with open(JSONL_LOG, 'a') as f:
        f.write(json.dumps(record) + '\n')

    return record


if __name__ == '__main__':
    result = run_pipeline_once(run_id=0)
    print(json.dumps(result, indent=2))
