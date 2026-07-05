# Loan Default Prediction — Data Pipeline Design

## Business Problem
A bank approves loans without fully knowing repayment risk upfront. This pipeline
predicts the probability that a loan applicant will default, using historical
loan data, so underwriting can adjust approvals, pricing, and loss reserves.

## Architecture
```
generate_data.py  -->  loan_data.csv (raw, with injected missing values)
        |
        v
preprocessing.py  -->  summary stats, dtype check, missing-value check,
                        median/mode imputation, StandardScaler normalization
        |
        v
eda.py            -->  correlation coefficients, binning (dti/income tiers),
                        label encoding, Random Forest feature importance,
                        8 univariate/bivariate charts (plots/)
        |
        v
pipeline_job.py   -->  wraps preprocessing + EDA into one job, captures
                        run metrics, writes structured logs
        |
        v
scheduler.py      -->  APScheduler triggers pipeline_job every 2 minutes
                        (maps to AWS EventBridge / Azure Data Factory
                        trigger / GCP Cloud Scheduler / Databricks Jobs
                        cron in a real cloud deployment)
        |
        v
logs/pipeline_log.jsonl  -->  feeds the Cloud dashboard (see dashboard
                               visualization: run history, default-rate
                               trend, success/failure status)
```

## Files
- `scripts/generate_data.py` — builds the synthetic Lending-Club-style dataset
  (8,000 rows, realistic feature correlations, injected missingness)
- `scripts/preprocessing.py` — Sub-Objective 1.3 (data pre-processing)
- `scripts/eda.py` — Sub-Objective 1.4 (EDA)
- `scripts/pipeline_job.py` — single pipeline execution + logging
- `scripts/scheduler.py` — Sub-Objective 1.5 (DataOps): runs the job every
  2 minutes using APScheduler, logs every run to `logs/pipeline_log.jsonl`
- `data/loan_data.csv` — raw generated data; `loan_data_clean.csv` — after
  preprocessing
- `plots/` — 8 EDA visualizations (class distribution, interest rate
  distribution, income distribution, default rate by grade, DTI by default
  status, default rate by purpose, correlation heatmap, feature importance)
- `logs/pipeline_log.jsonl` — machine-readable run history (used by the
  dashboard); `logs/pipeline.log` — human-readable log

## Key EDA Findings (for your report)
- Default rate: ~26.5% (calibrated to reflect realistic lending risk)
- Strongest predictor: `int_rate` (correlation 0.24 with default; also the
  top Random Forest feature — makes sense, since rate reflects assigned risk grade)
- Default rate rises monotonically with grade: A=17% -> G=60%, validating
  that the bank's own grading system is directionally sound
- `dti` (debt-to-income) and `revol_util` are secondary but meaningful drivers
- Missingness was injected into `annual_inc`, `dti`, `revol_util`, `emp_length`
  (3-5% each) and resolved via median (numeric) / mode (categorical) imputation

## Running it
```bash
pip install apscheduler scikit-learn pandas numpy matplotlib seaborn
python scripts/generate_data.py      # generates data/loan_data.csv
python scripts/preprocessing.py      # standalone preprocessing check
python scripts/eda.py                # standalone EDA + plots
python scripts/scheduler.py          # starts the 2-minute recurring job
```

## Cloud Deployment Note (Sub-Objective 3 tie-in)
In a real cloud deployment, `scheduler.py`'s local APScheduler would be
replaced by the platform's native scheduler (AWS EventBridge, Azure Data
Factory trigger, GCP Cloud Scheduler, or Databricks Jobs), and `pipeline_job.py`
would run as a container/notebook job. The pipeline/job status, endpoint
status, compute type, and model version pulled via that platform's API
satisfy Sub-Objective 3 (API access) and can be displayed alongside the
DataOps dashboard shown here.
