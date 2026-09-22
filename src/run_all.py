#!/usr/bin/env python3
"""
run_all.py — one-command re-run of the full research pipeline.

Runs, in order:
    src/mt5_data.py
    src/run_research.py
    src/threshold_sweep.py
    src/backtest.py

For each run it:
  - streams + captures stdout/stderr of every step
  - stops the chain the moment a step fails (non-zero exit code)
  - ALWAYS writes a snapshot of the run, success or failure, so you never
    silently lose a run from your history
  - saves a full-detail JSON file per run under reports/snapshots/
  - appends a compact row (or rows, one per symbol/timeframe) to
    reports/snapshots/snapshot_log.csv so you can track how results evolve
    over time without opening every JSON file

Usage:
    python src/run_all.py
    python src/run_all.py --skip-download        # reuse existing data/raw CSVs
    python src/run_all.py --continue-on-error     # don't stop chain on a failed step
    python src/run_all.py --python /path/to/python

Place this file in src/ (next to mt5_data.py etc.) and run it from the
project root, matching how the other scripts are invoked in the README
(`python src/data_quality.py`, `python src/backtest.py`, ...).

NOTE ON STAT EXTRACTION:
The parsing in `extract_stats()` is a best-effort regex match against the
kind of console output you described (per-symbol blocks with a win rate
and a return %). It has NOT been matched against your actual backtest.py
print statements yet. If the real output format differs, the run will
still succeed and the full raw stdout is saved in the JSON snapshot either
way — you just won't get the parsed win_rate/return columns in the CSV
until the regexes in STAT_PATTERNS are adjusted to match your real output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SNAPSHOT_DIR = PROJECT_ROOT / "reports" / "snapshots"
LOG_DIR = SNAPSHOT_DIR / "logs"
CSV_LOG_PATH = SNAPSHOT_DIR / "snapshot_log.csv"

# Steps to run in order. Each is a script under src/.
# `optional_after_skip_download` steps still run even with --skip-download.
PIPELINE_STEPS = [
    {"name": "mt5_data", "script": "mt5_data.py", "skip_if_no_download": True},
    {"name": "run_research", "script": "run_research.py", "skip_if_no_download": False},
    {"name": "threshold_sweep", "script": "threshold_sweep.py", "skip_if_no_download": False},
    {"name": "backtest", "script": "backtest.py", "skip_if_no_download": False},
]

# --- Best-effort stat extraction from backtest.py's stdout -----------------
# Adjust these once you share backtest.py's actual print format.
# Assumes blocks look roughly like:
#   XAUUSD H4
#   Win rate: 100.00%
#   Trades: 2
#   Net return: +4.7%
SYMBOL_BLOCK_RE = re.compile(
    r"^(?P<symbol>[A-Z]{6})\s+(?P<timeframe>[A-Z0-9]+)\s*$", re.MULTILINE
)
STAT_PATTERNS = {
    "win_rate": re.compile(r"win\s*rate[:\s]+([\-+]?\d+(?:\.\d+)?)\s*%", re.IGNORECASE),
    "trades": re.compile(r"trades?[:\s]+(\d+)", re.IGNORECASE),
    "net_return_pct": re.compile(r"net\s*return[:\s]+([\-+]?\d+(?:\.\d+)?)\s*%", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    name: str
    script: str
    command: str
    status: str  # "success" | "failed" | "skipped"
    returncode: int | None
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""


@dataclass
class RunSnapshot:
    run_id: str
    started_at: str
    finished_at: str = ""
    overall_status: str = "running"
    steps: list = field(default_factory=list)
    parsed_stats: list = field(default_factory=list)  # list of per-symbol dicts


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run_step(python_exe: str, name: str, script: str) -> StepResult:
    script_path = SRC_DIR / script
    command = f"{python_exe} {script_path}"
    print(f"\n{'=' * 70}\n▶ Running {name}  ({command})\n{'=' * 70}")

    if not script_path.exists():
        print(f"  !! {script_path} not found — skipping this step.")
        return StepResult(
            name=name, script=script, command=command,
            status="skipped", returncode=None, duration_seconds=0.0,
            stderr=f"{script_path} not found",
        )

    start = time.time()
    proc = subprocess.run(
        [python_exe, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    duration = time.time() - start

    # Echo output live-ish (after the fact, since we captured it) so you can
    # still see it in the terminal, not just in the saved log.
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    status = "success" if proc.returncode == 0 else "failed"
    print(f"  {'✔' if status == 'success' else '✘'} {name} finished "
          f"({status}, {duration:.1f}s, exit code {proc.returncode})")

    return StepResult(
        name=name, script=script, command=command,
        status=status, returncode=proc.returncode, duration_seconds=duration,
        stdout=proc.stdout, stderr=proc.stderr,
    )


def extract_stats(backtest_stdout: str) -> list[dict]:
    """Best-effort parse of per-symbol win rate / trades / return from
    backtest.py's stdout. Returns a list of dicts, one per symbol block
    found. Falls back to an empty list if nothing matches — the raw
    stdout is preserved in the JSON snapshot regardless."""
    results = []
    blocks = list(SYMBOL_BLOCK_RE.finditer(backtest_stdout))
    if not blocks:
        return results

    for i, match in enumerate(blocks):
        block_start = match.end()
        block_end = blocks[i + 1].start() if i + 1 < len(blocks) else len(backtest_stdout)
        block_text = backtest_stdout[block_start:block_end]

        stats = {"symbol": match.group("symbol"), "timeframe": match.group("timeframe")}
        for key, pattern in STAT_PATTERNS.items():
            m = pattern.search(block_text)
            stats[key] = m.group(1) if m else None
        results.append(stats)

    return results


def write_json_snapshot(snapshot: RunSnapshot) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{snapshot.run_id}.json"
    payload = {
        "run_id": snapshot.run_id,
        "started_at": snapshot.started_at,
        "finished_at": snapshot.finished_at,
        "overall_status": snapshot.overall_status,
        "parsed_stats": snapshot.parsed_stats,
        "steps": [
            {
                "name": s.name,
                "script": s.script,
                "command": s.command,
                "status": s.status,
                "returncode": s.returncode,
                "duration_seconds": round(s.duration_seconds, 2),
                "stdout_tail": s.stdout[-4000:],
                "stderr_tail": s.stderr[-2000:],
            }
            for s in snapshot.steps
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def append_csv_log(snapshot: RunSnapshot) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "started_at", "overall_status", "failed_step",
        "symbol", "timeframe", "win_rate", "trades", "net_return_pct",
    ]
    is_new = not CSV_LOG_PATH.exists()

    failed_step = next((s.name for s in snapshot.steps if s.status == "failed"), "")

    rows = []
    if snapshot.parsed_stats:
        for stats in snapshot.parsed_stats:
            rows.append({
                "run_id": snapshot.run_id,
                "started_at": snapshot.started_at,
                "overall_status": snapshot.overall_status,
                "failed_step": failed_step,
                "symbol": stats.get("symbol", ""),
                "timeframe": stats.get("timeframe", ""),
                "win_rate": stats.get("win_rate", ""),
                "trades": stats.get("trades", ""),
                "net_return_pct": stats.get("net_return_pct", ""),
            })
    else:
        # No parsed stats (e.g. run failed before backtest, or parsing
        # didn't match) — still log one row so the run isn't lost.
        rows.append({
            "run_id": snapshot.run_id,
            "started_at": snapshot.started_at,
            "overall_status": snapshot.overall_status,
            "failed_step": failed_step,
            "symbol": "", "timeframe": "", "win_rate": "", "trades": "", "net_return_pct": "",
        })

    with CSV_LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def print_summary(snapshot: RunSnapshot, json_path: Path) -> None:
    print(f"\n{'#' * 70}")
    print(f"# RUN SUMMARY — {snapshot.run_id}")
    print(f"{'#' * 70}")
    for s in snapshot.steps:
        icon = {"success": "✔", "failed": "✘", "skipped": "–"}[s.status]
        print(f"  {icon} {s.name:<18} {s.status:<8} {s.duration_seconds:>6.1f}s")
    print(f"\nOverall: {snapshot.overall_status.upper()}")
    if snapshot.parsed_stats:
        print("\nParsed stats:")
        for stats in snapshot.parsed_stats:
            print(f"  {stats.get('symbol')} {stats.get('timeframe')}: "
                  f"win_rate={stats.get('win_rate')}%  "
                  f"trades={stats.get('trades')}  "
                  f"net_return={stats.get('net_return_pct')}%")
    else:
        print("\nNo stats parsed automatically — check the JSON snapshot's "
              "stdout_tail for backtest.py, and adjust STAT_PATTERNS in "
              "this script to match the real output format.")
    print(f"\nJSON snapshot: {json_path}")
    print(f"CSV log:       {CSV_LOG_PATH}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-download", action="store_true",
                         help="Skip mt5_data.py and reuse existing data/raw CSVs.")
    parser.add_argument("--continue-on-error", action="store_true",
                         help="Keep running later steps even if an earlier step fails. "
                              "Default is to stop the chain on the first failure "
                              "(a snapshot is always saved either way).")
    parser.add_argument("--python", default=sys.executable,
                         help="Python executable to use for each step (default: current interpreter).")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    snapshot = RunSnapshot(run_id=run_id, started_at=datetime.now().isoformat(timespec="seconds"))

    print(f"Starting pipeline run {run_id}")
    if args.skip_download:
        print("(--skip-download set: reusing existing data/raw CSVs)")

    chain_broken = False
    backtest_stdout = ""

    for step_cfg in PIPELINE_STEPS:
        if chain_broken and not args.continue_on_error:
            snapshot.steps.append(StepResult(
                name=step_cfg["name"], script=step_cfg["script"],
                command="(not run — earlier step failed)",
                status="skipped", returncode=None, duration_seconds=0.0,
            ))
            continue

        if args.skip_download and step_cfg["skip_if_no_download"]:
            print(f"\n▶ Skipping {step_cfg['name']} (--skip-download)")
            snapshot.steps.append(StepResult(
                name=step_cfg["name"], script=step_cfg["script"],
                command="(skipped: --skip-download)",
                status="skipped", returncode=None, duration_seconds=0.0,
            ))
            continue

        result = run_step(args.python, step_cfg["name"], step_cfg["script"])
        snapshot.steps.append(result)

        if step_cfg["name"] == "backtest" and result.status == "success":
            backtest_stdout = result.stdout

        if result.status == "failed":
            chain_broken = True
            print(f"\n⚠ Step '{step_cfg['name']}' failed — stopping the chain here.")
            if not args.continue_on_error:
                break

    snapshot.finished_at = datetime.now().isoformat(timespec="seconds")
    any_failed = any(s.status == "failed" for s in snapshot.steps)
    snapshot.overall_status = "failed" if any_failed else "success"

    if backtest_stdout:
        snapshot.parsed_stats = extract_stats(backtest_stdout)

    json_path = write_json_snapshot(snapshot)
    append_csv_log(snapshot)
    print_summary(snapshot, json_path)

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())