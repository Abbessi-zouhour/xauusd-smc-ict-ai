"""
One command to check in on the project: pull fresh data, re-run the
research pipeline, sweep RR thresholds, backtest, then snapshot the
key numbers so this run can be compared against the last one instead
of eyeballing two giant terminal dumps.

Usage:
    python -m src.checkup                 # full chain, including MT5 download
    python -m src.checkup --skip-download  # skip MT5, use existing data/raw/*

Every run appends one row per symbol/timeframe to
reports/progress_history.csv, then prints what changed since the
previous row for that same symbol/timeframe (if any). A few days
between runs at H4/H1 will often show nothing changed -- that's
expected, not a bug (see run_research.py's own sample-size warnings).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
HISTORY_PATH = REPORTS_DIR / "progress_history.csv"

SYMBOLS = ["xauusd", "xagusd"]
TIMEFRAME_NAMES = ["m15", "h1", "h4", "d1"]

HISTORY_COLUMNS = [
    "run_at",
    "symbol",
    "timeframe",
    "total_setups",
    "valid_2r_setups",
    "n_labeled",
    "wins",
    "losses",
    "win_rate",
    "expectancy_r",
]


# ---------------------------------------------------------------------
# Step 1: pull fresh data (optional, skippable, never fatal)
# ---------------------------------------------------------------------

def download_data() -> None:

    print()
    print("#" * 70)
    print("# STEP 1: MT5 DATA DOWNLOAD")
    print("#" * 70)

    try:
        from src import mt5_data
    except ImportError as error:
        print(
            f"Skipping download -- couldn't import src.mt5_data "
            f"({error}). This is expected if the MetaTrader5 package "
            "isn't installed here, or if you're running this "
            "somewhere other than the machine with the MT5 terminal. "
            "Continuing with whatever is already in data/raw/."
        )
        return

    try:
        mt5_data.main()
    except Exception as error:
        print(
            f"Download step raised an error ({error}) -- continuing "
            "with whatever is already in data/raw/. If some symbols/"
            "timeframes downloaded before the error, they were still "
            "saved; only what failed is stale."
        )


# ---------------------------------------------------------------------
# Steps 2-4: research, threshold sweep, backtest
# ---------------------------------------------------------------------

def run_research_step() -> None:

    print()
    print("#" * 70)
    print("# STEP 2: RESEARCH PIPELINE")
    print("#" * 70)

    from src import run_research

    run_research.main()


def run_threshold_sweep_step() -> None:

    print()
    print("#" * 70)
    print("# STEP 3: RR THRESHOLD SWEEP")
    print("#" * 70)

    from src import threshold_sweep

    threshold_sweep.run_threshold_sweep()


def run_backtest_step() -> None:

    print()
    print("#" * 70)
    print("# STEP 4: BACKTEST")
    print("#" * 70)

    from src import backtest

    backtest.main()


# ---------------------------------------------------------------------
# Step 5: snapshot key numbers
# ---------------------------------------------------------------------

def _summarize_setups_file(symbol: str, timeframe: str) -> dict | None:
    """
    Recompute the same headline numbers run_research.py already
    printed, straight from its saved CSV -- so this script has a
    single source of truth (the file) rather than trying to scrape
    printed console output.
    """

    path = PROCESSED_DIR / f"{symbol}_{timeframe}_setups.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if "valid_2r_setup" not in df.columns or "outcome" not in df.columns:
        return None

    total_setups = len(df)
    valid_2r = int((df["valid_2r_setup"] == True).sum())  # noqa: E712

    decided = df[df["outcome"].isin([1, -1])]
    n_labeled = len(decided)
    wins = int((decided["outcome"] == 1).sum())
    losses = int((decided["outcome"] == -1).sum())

    if n_labeled == 0:
        win_rate = float("nan")
        expectancy_r = float("nan")
    else:
        win_rate = wins / n_labeled

        realized_r = decided.apply(
            lambda row: (
                float(row["available_rr"])
                if row["outcome"] == 1
                else -1.0
            ),
            axis=1,
        )
        expectancy_r = float(realized_r.mean())

    return {
        "total_setups": total_setups,
        "valid_2r_setups": valid_2r,
        "n_labeled": n_labeled,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "expectancy_r": expectancy_r,
    }


def take_snapshot() -> pd.DataFrame:

    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = []

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAME_NAMES:

            summary = _summarize_setups_file(symbol, timeframe)

            if summary is None:
                continue

            rows.append(
                {
                    "run_at": run_at,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    **summary,
                }
            )

    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def append_to_history(snapshot: pd.DataFrame) -> pd.DataFrame:

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)
        history = pd.concat([history, snapshot], ignore_index=True)
    else:
        history = snapshot.copy()

    history.to_csv(HISTORY_PATH, index=False)

    return history


# ---------------------------------------------------------------------
# Step 6: diff against the previous run
# ---------------------------------------------------------------------

def print_diff(history: pd.DataFrame) -> None:

    print()
    print("#" * 70)
    print("# WHAT CHANGED SINCE LAST CHECKUP")
    print("#" * 70)

    if history.empty:
        print("No history yet.")
        return

    history = history.sort_values("run_at")

    any_comparison = False

    for (symbol, timeframe), group in history.groupby(
        ["symbol", "timeframe"], sort=False
    ):

        if len(group) < 2:
            continue

        any_comparison = True

        previous = group.iloc[-2]
        current = group.iloc[-1]

        delta_setups = (
            current["total_setups"] - previous["total_setups"]
        )
        delta_valid = (
            current["valid_2r_setups"] - previous["valid_2r_setups"]
        )
        delta_labeled = current["n_labeled"] - previous["n_labeled"]

        print(f"\n{symbol.upper()} {timeframe.upper()}:")
        print(
            f"  Total setups:    {previous['total_setups']:>6} -> "
            f"{current['total_setups']:>6} ({delta_setups:+d})"
        )
        print(
            f"  Valid RR>=2:     {previous['valid_2r_setups']:>6} -> "
            f"{current['valid_2r_setups']:>6} ({delta_valid:+d})"
        )
        print(
            f"  Labeled trades:  {previous['n_labeled']:>6} -> "
            f"{current['n_labeled']:>6} ({delta_labeled:+d})"
        )

        if pd.notna(current["win_rate"]):
            prev_wr = (
                f"{previous['win_rate']*100:.1f}%"
                if pd.notna(previous["win_rate"])
                else "n/a"
            )
            print(
                f"  Win rate:        {prev_wr:>6} -> "
                f"{current['win_rate']*100:.1f}%"
            )

        if pd.notna(current["expectancy_r"]):
            prev_exp = (
                f"{previous['expectancy_r']:+.2f}R"
                if pd.notna(previous["expectancy_r"])
                else "n/a"
            )
            print(
                f"  Expectancy:      {prev_exp:>6} -> "
                f"{current['expectancy_r']:+.2f}R"
            )

        if delta_labeled == 0:
            print(
                "  (no new resolved trades since last checkup -- "
                "nothing new to read into here)"
            )

    if not any_comparison:
        print(
            "\nThis is the first checkup with saved history for at "
            "least one symbol/timeframe -- nothing to diff against "
            "yet. Run this again after more data accumulates to see "
            "a comparison."
        )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:

    skip_download = "--skip-download" in sys.argv

    print("#" * 70)
    print("# PROJECT CHECKUP")
    print("#" * 70)
    print(
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    if skip_download:
        print("(--skip-download: using existing data/raw/* as-is)")

    if not skip_download:
        download_data()

    run_research_step()
    run_threshold_sweep_step()
    run_backtest_step()

    print()
    print("#" * 70)
    print("# STEP 5: SNAPSHOT")
    print("#" * 70)

    snapshot = take_snapshot()

    if snapshot.empty:
        print(
            "No processed setups files found -- nothing to snapshot. "
            "Did run_research complete successfully above?"
        )
        return

    history = append_to_history(snapshot)

    print(f"Snapshot saved to: {HISTORY_PATH}")

    print_diff(history)

    print()
    print("Checkup completed. No live orders were executed.")


if __name__ == "__main__":
    main()