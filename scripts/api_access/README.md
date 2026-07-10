# Sub-Objective 3 — API Access (2 marks)

Retrieves and displays key application details from **Prefect Cloud's built-in
REST API**, satisfying:

- **3.1 Retrieve Key Application Details** — queries the deployed flow, its
  deployment, schedule, work pool, and run history via the API.
- **3.2 Display Application Details** — prints **10** application details
  (well over the required minimum of 4).

## Setup

```bash
pip install -r requirements.txt
```

Get a Prefect Cloud API key: **Prefect Cloud → your profile → API Keys → Create
API Key**. Then:

```bash
export PREFECT_API_KEY="pnu_xxxxxxxxxxxx"
python part3_api_access.py
```

Account and workspace IDs are pre-filled from the team's Prefect Cloud URL.
Override them if needed:

```bash
export PREFECT_ACCOUNT_ID="..."
export PREFECT_WORKSPACE_ID="..."
```

Filter to a single flow:

```bash
python part3_api_access.py --flow-name loan-default-pipeline
```

## What it displays (for the report screenshot)

Flow name, deployment name, deployment ID, active/paused status, schedule
(the every-2-minute cron), work pool, work queue, tags, created & updated
timestamps, run counts by state (completed/failed/crashed/running), and the
most recent flow runs with their status and start time.

## Notes

- Requires Part 2's flow to already be **deployed to the Prefect Cloud
  workspace** — the API can only report on deployments that exist.
- Uses only Prefect's built-in REST API (no third-party services).
