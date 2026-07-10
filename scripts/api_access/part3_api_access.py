#!/usr/bin/env python3
"""
Sub-Objective 3: API Access
AIMLCZG549 - API-driven Cloud Native Solutions

3.1 Retrieve Key Application Details via Prefect Cloud's built-in REST API.
3.2 Display at least four application details retrieved via the API.

The pipeline / ML job is deployed as a Prefect flow deployment. This script
authenticates to Prefect Cloud and pulls live details about the flow, its
deployment, and its recent runs -- proving the application's state is
accessible programmatically (not just via the UI).

Usage:
    export PREFECT_API_KEY="pnu_xxx..."      # your Prefect Cloud API key
    python part3_api_access.py

    # optional: filter to one deployment/flow by name
    python part3_api_access.py --flow-name loan-default-pipeline
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Configuration -- account & workspace IDs come from your Prefect Cloud URL:
# https://app.prefect.cloud/account/<ACCOUNT_ID>/workspace/<WORKSPACE_ID>/...
# ---------------------------------------------------------------------------
ACCOUNT_ID = os.getenv("PREFECT_ACCOUNT_ID", "fffd6623-b09b-4ba1-b8cf-48df25d9dfaf")
WORKSPACE_ID = os.getenv("PREFECT_WORKSPACE_ID", "bbea7d63-6223-45d3-9a61-71738def52bf")
API_KEY = os.getenv("PREFECT_API_KEY", "")

BASE_URL = (
    f"https://api.prefect.cloud/api/accounts/{ACCOUNT_ID}"
    f"/workspaces/{WORKSPACE_ID}"
)


def _headers():
    if not API_KEY:
        sys.exit(
            "ERROR: set your Prefect Cloud API key first:\n"
            '    export PREFECT_API_KEY="pnu_xxxxxxxx"'
        )
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _post(path, body=None):
    """POST to a Prefect read/filter endpoint and return parsed JSON."""
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=body or {})
    resp.raise_for_status()
    return resp.json()


def _get(path):
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers())
    resp.raise_for_status()
    return resp.json()


def _fmt_time(value):
    """Pretty-print an ISO timestamp; tolerate None / bad values."""
    if not value:
        return "-"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, AttributeError):
        return str(value)


def get_deployments(flow_name=None):
    """3.1 -- list deployments (the deployed application/pipeline)."""
    body = {}
    if flow_name:
        body = {"flows": {"name": {"any_": [flow_name]}}}
    return _post("/deployments/filter", body)


def get_flow(flow_id):
    return _get(f"/flows/{flow_id}")


def get_flow_runs(deployment_id, limit=5):
    """Recent flow runs for a deployment, newest first."""
    body = {
        "deployments": {"id": {"any_": [deployment_id]}},
        "sort": "START_TIME_DESC",
        "limit": limit,
    }
    return _post("/flow_runs/filter", body)


def count_runs_by_state(deployment_id):
    """Success / failure counts -- useful monitoring detail."""
    counts = {}
    for state in ("COMPLETED", "FAILED", "CRASHED", "RUNNING"):
        body = {
            "deployments": {"id": {"any_": [deployment_id]}},
            "flow_runs": {"state": {"type": {"any_": [state]}}},
        }
        try:
            counts[state] = _post("/flow_runs/count", body)
        except requests.HTTPError:
            counts[state] = "n/a"
    return counts


def print_deployment_report(dep):
    """3.2 -- display >= 4 application details retrieved via the API."""
    dep_id = dep.get("id")
    flow = get_flow(dep["flow_id"]) if dep.get("flow_id") else {}
    # /flow_runs/filter returns a JSON array of run objects
    runs = get_flow_runs(dep_id)
    counts = count_runs_by_state(dep_id)

    schedule = "-"
    schedules = dep.get("schedules") or []
    if schedules:
        sched = schedules[0].get("schedule", {})
        schedule = sched.get("cron") or sched.get("interval") or str(sched)

    print("=" * 70)
    print("  APPLICATION DETAILS RETRIEVED VIA PREFECT CLOUD API")
    print("=" * 70)
    # --- the four+ required details ---
    print(f"  1. Flow name .............. {flow.get('name', '-')}")
    print(f"  2. Deployment name ........ {dep.get('name', '-')}")
    print(f"  3. Deployment ID .......... {dep_id}")
    print(f"  4. Status (paused?) ....... {'PAUSED' if dep.get('paused') else 'ACTIVE / READY'}")
    print(f"  5. Schedule ............... {schedule}")
    print(f"  6. Work pool .............. {dep.get('work_pool_name', '-')}")
    print(f"  7. Work queue ............. {dep.get('work_queue_name', '-')}")
    print(f"  8. Tags ................... {', '.join(dep.get('tags', [])) or '-'}")
    print(f"  9. Created ................ {_fmt_time(dep.get('created'))}")
    print(f" 10. Last updated .......... {_fmt_time(dep.get('updated'))}")
    print("-" * 70)
    print("  RUN MONITORING (via API):")
    print(f"     Completed: {counts.get('COMPLETED')}   "
          f"Failed: {counts.get('FAILED')}   "
          f"Crashed: {counts.get('CRASHED')}   "
          f"Running: {counts.get('RUNNING')}")
    print("-" * 70)
    print("  RECENT FLOW RUNS:")
    if not runs:
        print("     (no runs found)")
    for r in runs:
        state = (r.get("state") or {}).get("type", "-")
        print(f"     - {r.get('name', '-'):<28} {state:<12} "
              f"start={_fmt_time(r.get('start_time'))}")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description="Prefect Cloud API access (Sub-Objective 3)")
    parser.add_argument("--flow-name", help="filter to a single flow name", default=None)
    args = parser.parse_args()

    print(f"\nConnecting to Prefect Cloud workspace:\n  {BASE_URL}\n")

    deployments = get_deployments(args.flow_name)
    if not deployments:
        print("No deployments found. Make sure Part 2 deployed a flow to this workspace.")
        return

    print(f"Found {len(deployments)} deployment(s).\n")
    for dep in deployments:
        print_deployment_report(dep)


if __name__ == "__main__":
    main()
