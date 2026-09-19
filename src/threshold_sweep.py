"""
RR threshold sweep for the XAUUSD SMC / ICT pipeline.

run_research.py answers one question: "at RR >= 2.0, what's the
win rate?" With only a handful of setups clearing that bar per
timeframe, that single number is close to meaningless on its own
-- it doesn't say whether 2.0 is a good place to draw the line,
or whether a slightly different value would look completely
different.

This script answers a more useful question: across a RANGE of RR
thresholds, how do win rate, expectancy, and sample size move
together? That's what "statistically meaningful" means here --
not a bigger single number, but a curve with confidence intervals
attached, so a difference between two thresholds can be judged
against how much of it is just sampling noise.

Steps:
    1. Run the full detection pipeline once per timeframe with
       an effectively-disabled RR filter (minimum_rr is tiny),
       so every structurally valid setup gets a computed
       available_rr instead of being discarded before we can
       see it.
    2. Label EVERY setup that has complete levels, not just the
       ones that would have cleared a 2R filter (only_valid_2r
       =False) -- otherwise we'd still only be looking at
       whatever the old filter let through.
    3. Pool labeled setups across timeframes (with a "timeframe"
       column kept so you can still slice by timeframe).
    4. For a range of RR thresholds, filter to
       available_rr >= threshold and compute win rate,
       Wilson-score 95% CI, expectancy, and sample size.

This is descriptive research, same as run_research.py. It does
not execute trades and does not pick a threshold for you.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.data_loader import load_ohlcv_csv
from src.ict import detect_ict
from src.labels import label_setup_outcomes
from src.liquidity import detect_liquidity
from src.setups import detect_setups
from src.structure import detect_market_structure, detect_swings


# ---------------------------------------------------------------------
# Same detection settings as run_research.py, so this sweep is
# measuring the effect of the RR threshold alone, not a different
# pipeline. If you change one, change the other.
# ---------------------------------------------------------------------

MAX_TARGET_ATR_MULTIPLE = 8.0
MIN_RISK_ATR_MULTIPLE = 0.5
ENTRY_MODE = "fvg_midpoint"
FVG_RETRACEMENT = 0.5

# Effectively "no filter" -- detect_setups requires minimum_rr > 0,
# and calculate_available_rr needs *some* value to compute
# valid_2r_setup against, but we ignore that column here and filter
# on the raw available_rr ourselves at each threshold below.
NO_FILTER_MINIMUM_RR = 0.01

# Thresholds to sweep. Add/remove values as needed -- this does
# not need to match run_research.py's minimum_rr=2.0.
RR_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TIMEFRAMES = {
    "m15": "xauusd_m15.csv",
    "h1": "xauusd_h1.csv",
    "h4": "xauusd_h4.csv",
    "d1": "xauusd_d1.csv",
}


# ---------------------------------------------------------------------
# Pipeline, one timeframe, unfiltered
# ---------------------------------------------------------------------

def _detect_all_setups(timeframe: str, filename: str) -> pd.DataFrame:
    """
    Run structure -> liquidity -> ICT -> setups -> labels for one
    timeframe, WITHOUT discarding setups below the usual RR bar.

    Returns every row that has a complete entry/stop/target/label,
    tagged with which timeframe it came from and its available_rr,
    so the caller can filter by whatever threshold it wants.
    """

    input_path = RAW_DIR / filename

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {input_path}"
        )

    df = load_ohlcv_csv(input_path)

    df = detect_swings(df, left_bars=3, right_bars=3)
    df = detect_market_structure(df, right_bars=3)
    df = detect_liquidity(df, lookback=20, tolerance=0.10)
    df = detect_ict(df)

    df = detect_setups(
        df,
        sequence_window=3,
        minimum_rr=NO_FILTER_MINIMUM_RR,
        stop_buffer=0.0,
        max_target_atr_multiple=MAX_TARGET_ATR_MULTIPLE,
        min_risk_atr_multiple=MIN_RISK_ATR_MULTIPLE,
        entry_mode=ENTRY_MODE,
        fvg_retracement=FVG_RETRACEMENT,
    )

    # only_valid_2r=False: label every setup with complete levels,
    # not just the ones that would have cleared a 2R filter. The
    # RR threshold is applied afterwards, in the sweep itself.
    df = label_setup_outcomes(
        df,
        max_holding_bars=200,
        only_valid_2r=False,
    )

    df = df.copy()
    df["timeframe"] = timeframe

    return df


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def _wilson_interval(
    wins: int,
    decided: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """
    Wilson score 95% confidence interval for a win rate.

    Preferred over the naive p +/- 1.96*sqrt(p(1-p)/n) interval
    at small n: it doesn't produce nonsensical bounds outside
    [0, 1] and it's the standard choice for binomial proportions
    with few trials, which is exactly the regime this project is
    in right now.
    """

    if decided == 0:
        return (math.nan, math.nan)

    p = wins / decided
    denom = 1 + z ** 2 / decided

    center = p + z ** 2 / (2 * decided)
    spread = z * math.sqrt(
        (p * (1 - p) + z ** 2 / (4 * decided)) / decided
    )

    low = (center - spread) / denom
    high = (center + spread) / denom

    return (max(0.0, low), min(1.0, high))


def _summarize_at_threshold(
    labeled: pd.DataFrame,
    threshold: float,
) -> dict:
    """
    Win rate, expectancy, and sample size for setups whose
    available_rr clears `threshold`, restricted to setups that
    actually resolved (outcome != timeout).
    """

    subset = labeled[
        labeled["available_rr"] >= threshold
    ]

    decided = subset[
        subset["outcome"].isin([1, -1])
    ]

    n = len(decided)
    wins = int((decided["outcome"] == 1).sum())
    losses = int((decided["outcome"] == -1).sum())

    timeouts = int(
        (subset["outcome"] == 0).sum()
    )

    if n == 0:
        return {
            "threshold": threshold,
            "n": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": timeouts,
            "win_rate": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "expectancy_r": math.nan,
        }

    win_rate = wins / n
    ci_low, ci_high = _wilson_interval(wins, n)

    # Realized R per decided trade: the full available_rr for a
    # win (exit at the structural target), -1R for a loss (exit
    # at the stop) -- exact given how labels.py resolves outcomes,
    # not an approximation.
    realized_r = decided["outcome"].map(
        {1: 1.0, -1: -1.0}
    ) * decided["available_rr"].where(
        decided["outcome"] == 1, 1.0
    )

    expectancy_r = float(realized_r.mean())

    return {
        "threshold": threshold,
        "n": n,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": win_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "expectancy_r": expectancy_r,
    }


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def run_threshold_sweep() -> pd.DataFrame:

    print("=" * 70)
    print("RR THRESHOLD SWEEP")
    print("=" * 70)

    all_labeled = []

    for timeframe, filename in TIMEFRAMES.items():

        print(f"\nDetecting setups on {timeframe.upper()}...")

        try:
            labeled = _detect_all_setups(timeframe, filename)
        except FileNotFoundError as error:
            print(f"  Skipping {timeframe}: {error}")
            continue

        has_outcome = labeled["outcome"].notna()
        print(
            f"  {has_outcome.sum():,} setups with a resolved "
            f"or timed-out outcome"
        )

        all_labeled.append(labeled)

    if not all_labeled:
        raise RuntimeError(
            "No timeframe data found -- nothing to sweep."
        )

    pooled = pd.concat(all_labeled, ignore_index=True)

    labeled_pooled = pooled[pooled["outcome"].notna()]

    print()
    print("=" * 70)
    print("POOLED (all timeframes) -- THRESHOLD SWEEP")
    print("=" * 70)
    print(
        f"{'RR >=':>6} | {'n':>4} | {'W':>3} | {'L':>3} | "
        f"{'T':>3} | {'win rate':>9} | {'95% CI':>17} | "
        f"{'expectancy':>10}"
    )
    print("-" * 70)

    rows = []

    for threshold in RR_THRESHOLDS:

        stats = _summarize_at_threshold(
            labeled_pooled, threshold,
        )
        rows.append(stats)

        if stats["n"] == 0:
            print(
                f"{threshold:>6.2f} | {'0':>4} |   - |   - | "
                f"{stats['timeouts']:>3} | {'n/a':>9} | "
                f"{'n/a':>17} | {'n/a':>10}"
            )
            continue

        ci = (
            f"[{stats['ci_low']*100:5.1f}, "
            f"{stats['ci_high']*100:5.1f}]"
        )

        print(
            f"{threshold:>6.2f} | {stats['n']:>4} | "
            f"{stats['wins']:>3} | {stats['losses']:>3} | "
            f"{stats['timeouts']:>3} | "
            f"{stats['win_rate']*100:>8.1f}% | {ci:>17} | "
            f"{stats['expectancy_r']:>+9.2f}R"
        )

    print("-" * 70)
    print(
        "n = decided trades (wins + losses) at or above that RR.\n"
        "95% CI is the Wilson score interval on win rate -- if two\n"
        "thresholds' intervals overlap heavily, don't read much into\n"
        "the difference between their point estimates; that's noise,\n"
        "not signal, until n grows."
    )

    # ---------------------------------------------------------
    # Also print the breakdown per timeframe, same thresholds,
    # so you can see whether any one timeframe is driving the
    # pooled numbers.
    # ---------------------------------------------------------
    for timeframe in TIMEFRAMES:

        tf_labeled = labeled_pooled[
            labeled_pooled["timeframe"] == timeframe
        ]

        if tf_labeled.empty:
            continue

        print()
        print("-" * 70)
        print(f"{timeframe.upper()} only")
        print("-" * 70)

        for threshold in RR_THRESHOLDS:

            stats = _summarize_at_threshold(
                tf_labeled, threshold,
            )

            if stats["n"] == 0:
                continue

            ci = (
                f"[{stats['ci_low']*100:5.1f}, "
                f"{stats['ci_high']*100:5.1f}]"
            )

            print(
                f"  RR >= {threshold:>4.2f}  n={stats['n']:>3}  "
                f"win_rate={stats['win_rate']*100:>5.1f}%  "
                f"95% CI={ci}  expectancy={stats['expectancy_r']:+.2f}R"
            )

    # ---------------------------------------------------------
    # Save the pooled, unfiltered, labeled dataset so it can be
    # sliced further outside this script (e.g. in a notebook)
    # without rerunning detection.
    # ---------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "xauusd_threshold_sweep_pooled.csv"
    pooled.to_csv(out_path, index=False)

    print()
    print(f"Saved pooled unfiltered dataset to: {out_path}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_threshold_sweep()