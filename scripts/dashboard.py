"""
dashboard.py
Sub-Objective 1.5 — DataOps Cloud Dashboard

Reads the JSONL run-history produced by pipeline_job.py / scheduler.py
and prints a text-based dashboard to the terminal.  In a production cloud
deployment this output would be forwarded to CloudWatch, Stackdriver, or
a Grafana datasource; here it demonstrates the same information in-process.

Usage (run from the project root OR the scripts/ folder):
    python scripts/dashboard.py                # summary + last 10 runs
    python scripts/dashboard.py --tail 20      # show last N runs
    python scripts/dashboard.py --watch        # refresh every 30 s
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the JSONL log path regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_LOG_CANDIDATES = [
    _SCRIPT_DIR / "logs" / "pipeline_log.jsonl",          # scripts/logs/
    _SCRIPT_DIR.parent / "logs" / "pipeline_log.jsonl",   # project-root logs/
]


def _find_log() -> Path | None:
    for p in _LOG_CANDIDATES:
        if p.exists():
            return p
    return None


def _read_runs(log_path: Path) -> list[dict]:
    runs = []
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return runs


def _fmt_ts(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso


def _status_icon(status: str) -> str:
    return {"SUCCESS": "[OK]", "FAILED": "[FAIL]"}.get(status, f"[{status}]")


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------

def render_dashboard(runs: list[dict], tail: int = 10) -> None:
    total = len(runs)
    successes = sum(1 for r in runs if r.get("status") == "SUCCESS")
    failures = total - successes
    success_rate = (successes / total * 100) if total else 0

    recent = runs[-tail:]

    # Derive trends from the last 10 successful runs
    def_rates = [r["default_rate"] for r in runs if r.get("status") == "SUCCESS"]
    avg_default = sum(def_rates) / len(def_rates) if def_rates else None
    durations = [r["duration_sec"] for r in runs if r.get("status") == "SUCCESS"]
    avg_dur = sum(durations) / len(durations) if durations else None

    # Determine most common top-correlated feature
    features: dict[str, int] = {}
    for r in runs:
        feat = r.get("top_correlated_feature")
        if feat:
            features[feat] = features.get(feat, 0) + 1
    top_feat = max(features, key=features.get) if features else "N/A"

    print()
    print("=" * 72)
    print("  DATAOPS PIPELINE DASHBOARD  —  Loan Default Prediction")
    print("=" * 72)
    print(f"  Log file   : {_find_log()}")
    print(f"  Refreshed  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("-" * 72)
    print("  OVERALL STATISTICS")
    print(f"    Total pipeline runs     : {total}")
    print(f"    Successful              : {successes}")
    print(f"    Failed                  : {failures}")
    print(f"    Success rate            : {success_rate:.1f}%")
    if avg_default is not None:
        print(f"    Avg default rate        : {avg_default:.4f}")
    if avg_dur is not None:
        print(f"    Avg run duration (s)    : {avg_dur:.2f}")
    print(f"    Top correlated feature  : {top_feat}")
    print("-" * 72)
    print(f"  RECENT RUNS  (last {min(tail, total)} of {total})")
    print(
        f"  {'#':<5} {'Timestamp (UTC)':<22} {'Status':<8} "
        f"{'Rows':<8} {'Default%':<10} {'Dur(s)':<8} {'Top Feature'}"
    )
    print("  " + "-" * 70)
    for r in recent:
        icon = _status_icon(r.get("status", "?"))
        ts = _fmt_ts(r.get("timestamp_utc", ""))
        rows = r.get("rows_processed", "-")
        dr = f"{r['default_rate']:.4f}" if r.get("default_rate") is not None else "-"
        dur = f"{r['duration_sec']:.2f}" if r.get("duration_sec") is not None else "-"
        feat = r.get("top_correlated_feature", "-")
        err = f"  ERROR: {r['error']}" if r.get("error") else ""
        run_id = r.get("run_id", "?")
        print(
            f"  {run_id:<5} {ts:<22} {icon:<8} "
            f"{str(rows):<8} {dr:<10} {dur:<8} {feat}{err}"
        )

    # Default rate by grade for the most recent successful run
    latest_ok = next(
        (r for r in reversed(runs) if r.get("status") == "SUCCESS"), None
    )
    if latest_ok and latest_ok.get("default_rate_by_grade"):
        print("-" * 72)
        print("  DEFAULT RATE BY GRADE  (latest run)")
        for grade, rate in sorted(latest_ok["default_rate_by_grade"].items()):
            bar_len = int(rate * 40)
            bar = "█" * bar_len
            print(f"    {grade}: {bar:<40} {rate:.3f}")

    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DataOps pipeline run-history dashboard"
    )
    parser.add_argument(
        "--tail", type=int, default=10,
        help="Number of recent runs to display (default: 10)"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Refresh the dashboard every 30 seconds until Ctrl+C"
    )
    args = parser.parse_args()

    log_path = _find_log()

    def _render() -> None:
        if not log_path or not log_path.exists():
            print(
                "\n[dashboard] No run log found yet.\n"
                "Start the scheduler first:\n"
                "    python scripts/scheduler.py\n"
            )
            return
        runs = _read_runs(log_path)
        if not runs:
            print("\n[dashboard] Log file is empty — no runs recorded yet.\n")
            return
        render_dashboard(runs, tail=args.tail)

    if args.watch:
        print("Dashboard in watch mode — refreshing every 30 s. Ctrl+C to stop.")
        try:
            while True:
                _render()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
            sys.exit(0)
    else:
        _render()


if __name__ == "__main__":
    main()
