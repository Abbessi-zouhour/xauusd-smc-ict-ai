"""
Run the SMC / ICT research pipeline for each symbol in SYMBOLS.

This script:
1. Loads historical OHLCV data.
2. Detects market structure.
3. Detects liquidity.
4. Detects ICT concepts.
5. Detects SMC/ICT setups.
6. Calculates structural targets.
7. Calculates actual available RR.
8. Applies the minimum RR filter.
9. Saves processed datasets.

This is historical research/backtesting only.
No live trading or order execution is performed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_ohlcv_csv
from src.labels import label_setup_outcomes, summarize_outcomes
from src.liquidity import detect_liquidity
from src.ict import detect_ict
from src.setups import detect_setups
from src.structure import detect_market_structure, detect_swings


# ---------------------------------------------------------------------
# Minimum sample size before ML training is meaningful.
#
# This is a research heuristic, not a hard rule: below this many
# labeled (non-timeout) 2R setups, an XGBoost model is very likely
# to fit noise rather than signal. Treat pipeline output below this
# threshold as "not yet ready for Phase 7 (ML)".
# ---------------------------------------------------------------------

MIN_SETUPS_FOR_ML = 200

# ---------------------------------------------------------------------
# Target distance cap, in multiples of ATR at the setup candle.
#
# Without this, the "nearest unbroken swing" target is usually close
# (small reward), so the 2R filter ends up selecting almost only the
# rare cases where the nearest swing happens to be implausibly far
# away -- a tiny stop paired with a long-shot target, not a real edge.
# Capping the target to a realistic multiple of recent volatility
# excludes those long-shot outliers. Set to None to reproduce the
# original unbounded behavior.
# ---------------------------------------------------------------------

MAX_TARGET_ATR_MULTIPLE = 8.0

# ---------------------------------------------------------------------
# Minimum risk floor, in multiples of ATR at the setup candle.
#
# stop_loss is the high/low of a single candle (calculate_setup_levels).
# When that candle's range is tiny relative to normal volatility, RR
# gets inflated by a noise-sized denominator rather than a genuinely
# favorable setup -- and a noise-sized stop is *more* likely to be hit
# by ordinary price action, not less. This floor rejects setups whose
# risk is too small to be a meaningful stop. Set to None to reproduce
# the original unbounded behavior (not recommended -- see README notes
# on this fix).
# ---------------------------------------------------------------------

MIN_RISK_ATR_MULTIPLE = 0.5

# ---------------------------------------------------------------------
# ATR-multiple fallback target, in multiples of ATR at the setup
# candle.
#
# The nearest confirmed structural swing beyond entry is very often
# only just beyond entry, since swings are frequent -- so the
# structural-target search tends to produce a small reward relative
# to risk, and the 2R filter ends up selecting almost exclusively
# the rare setups that happen to have a distant swing nearby (a
# tiny, unrepresentative sample). When set, find_structural_targets()
# still prefers a real structural swing whenever one clears minimum_rr
# on its own; this value only kicks in as a fallback target -- sized
# to at least this many ATRs, and at least minimum_rr * risk -- for
# setups where no structural swing does. Capped by
# MAX_TARGET_ATR_MULTIPLE above, same as a structural target would be.
# Set to None to reproduce the original behavior (structural swings
# only, no fallback -- this is what produced the very small 2R-filtered
# samples noted in checkup.py's output).
# ---------------------------------------------------------------------

ATR_FALLBACK_MULTIPLE = 4.0

SPREAD = {
    "xauusd": 0.30,
    "xagusd": 0.03,
}


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------
# Timeframes
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Symbols and timeframes
#
# Each symbol is processed completely separately -- its own section,
# its own FINAL SUMMARY, its own output CSVs. They are NEVER pooled
# together: XAUUSD and XAGUSD are different markets, and averaging
# their setups into one number would blur two different things into
# something meaningless. Add more symbols here once you've pulled
# their raw data with src.mt5_data (SYMBOLS there must match these
# keys).
# ---------------------------------------------------------------------

SYMBOLS = ["xauusd", "xagusd"]

TIMEFRAME_NAMES = ["m15", "h1", "h4", "d1"]


def _raw_filenames(symbol: str) -> dict[str, str]:
    return {
        tf: f"{symbol}_{tf}.csv"
        for tf in TIMEFRAME_NAMES
    }


# ---------------------------------------------------------------------
# Process one timeframe
# ---------------------------------------------------------------------

def process_timeframe(
    symbol: str,
    timeframe: str,
    filename: str,
) -> pd.DataFrame:
    """Process one symbol/timeframe through the complete research pipeline."""

    print()
    print("=" * 70)
    print(f"Processing {symbol.upper()} {timeframe.upper()}")
    print("=" * 70)

    input_path = RAW_DIR / filename

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {input_path}"
        )

    # ---------------------------------------------------------
    # 1. Load historical OHLCV data
    # ---------------------------------------------------------

    df = load_ohlcv_csv(input_path)

    print(f"Loaded {len(df):,} candles")

    # ---------------------------------------------------------
    # 2. Market structure
    # ---------------------------------------------------------

    df = detect_swings(
        df,
        left_bars=3,
        right_bars=3,
    )

    df = detect_market_structure(
        df,
        right_bars=3,
    )

    # ---------------------------------------------------------
    # 3. Liquidity
    # ---------------------------------------------------------

    df = detect_liquidity(
        df,
        lookback=20,
        tolerance=0.10,
    )

    # ---------------------------------------------------------
    # 4. ICT concepts
    # ---------------------------------------------------------

    df = detect_ict(
        df,
    )
    print()
    print("DETECTION COLUMNS")
    print("-" * 40)

    detection_columns = [
        column
        for column in df.columns
        if (
            "sweep" in column.lower()
            or "fvg" in column.lower()
            or "displacement" in column.lower()
        )
    ]

    print(detection_columns)
    # ---------------------------------------------------------
    # 5. SMC / ICT setups
    # ---------------------------------------------------------

    symbol_spread = SPREAD.get(symbol)

    if symbol_spread is None:
        print(
            f"WARNING: no SPREAD entry for {symbol!r} -- "
            "defaulting to 0.0 (no cost modeled). Add this "
            "symbol to the SPREAD dict once you've checked its "
            "live bid/ask in MT5."
        )
        symbol_spread = 0.0

    df = detect_setups(
        df,
        sequence_window=3,
        minimum_rr=2.0,
        stop_buffer=0.0,
        max_target_atr_multiple=MAX_TARGET_ATR_MULTIPLE,
        min_risk_atr_multiple=MIN_RISK_ATR_MULTIPLE,
        entry_mode="fvg_midpoint",
        fvg_retracement=0.5,
        spread=symbol_spread,
        atr_fallback_multiple=ATR_FALLBACK_MULTIPLE,
    )

    # ---------------------------------------------------------
    # 5b. Label outcomes for setups that pass the 2R filter
    #
    # A setup clearing the 2R filter only means the structural
    # target was theoretically far enough away. This step checks
    # what actually happened afterwards: did price reach the
    # target before the stop, hit the stop first, or time out.
    # ---------------------------------------------------------

    df = label_setup_outcomes(
        df,
        max_holding_bars=200,
        only_valid_2r=True,
    )

    # ---------------------------------------------------------
    # 6. Save processed dataset
    # ---------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DIR
        / f"{symbol}_{timeframe}_setups.csv"
    )

    # timestamp is carried as the DataFrame's index throughout
    # the pipeline (see data_loader.load_ohlcv_csv), not as a
    # regular column. index=False here would silently drop it --
    # every row would still LOOK complete (a full OHLC candle,
    # entry/stop/target, outcome) but there'd be no way to find
    # that candle on an actual chart to sanity-check it, which is
    # exactly the kind of "looks right in the CSV" false
    # confidence that's easy to miss. reset_index() turns the
    # index back into a proper named "timestamp" column before
    # saving.
    df.reset_index().to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 7. Setup statistics
    # ---------------------------------------------------------

    bullish_count = int(
        (
            df["setup_direction"]
            == "bullish"
        ).sum()
    )

    bearish_count = int(
        (
            df["setup_direction"]
            == "bearish"
        ).sum()
    )

    total_setups = int(df["setup_direction"].isin(["bullish", "bearish"]).sum())

    valid_rr_count = int(
        df["valid_2r_setup"].sum()
    )

    rr_available_count = int(
        df["rr_ratio"].notna().sum()
    )

    if rr_available_count > 0:
        rr_filter_rate = (
            valid_rr_count
            / rr_available_count
            * 100
        )
    else:
        rr_filter_rate = 0.0

    print()
    print("SETUP STATISTICS")
    print("-" * 40)

    print(
        f"Bullish setups:       {bullish_count:,}"
    )

    print(
        f"Bearish setups:       {bearish_count:,}"
    )

    print(
        f"Total setups:         {total_setups:,}"
    )

    print(
        f"RR available:         {rr_available_count:,}"
    )

    print(
        f"Valid RR >= 2:        {valid_rr_count:,}"
    )

    print(
        f"RR filter rate:       {rr_filter_rate:.2f}%"
    )

    # ---------------------------------------------------------
    # 8. RR distribution
    # ---------------------------------------------------------

    rr_values = df.loc[
        df["rr_ratio"].notna(),
        "rr_ratio",
    ]

    if not rr_values.empty:

        print()
        print("RR DISTRIBUTION")
        print("-" * 40)

        print(
            f"Minimum RR:           "
            f"{rr_values.min():.2f}"
        )

        print(
            f"Median RR:            "
            f"{rr_values.median():.2f}"
        )

        print(
            f"Mean RR:              "
            f"{rr_values.mean():.2f}"
        )

        print(
            f"Maximum RR:           "
            f"{rr_values.max():.2f}"
        )

    # ---------------------------------------------------------
    # 9. Target / RR examples
    # ---------------------------------------------------------

    print()
    print("TARGET / RR COLUMNS")
    print("-" * 40)

    display_columns = [
        "setup_direction",
        "entry",
        "stop_loss",
        "structural_target",
        "available_reward",
        "available_risk",
        "available_rr",
        "valid_2r_setup",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    setup_examples = (
        df[available_columns]
        .dropna(
            subset=["setup_direction"]
        )
        .tail(5)
    )

    if setup_examples.empty:
        print("No setups detected.")

    else:
        print(
            setup_examples.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # 9b. Outcome / expectancy statistics
    #
    # This is the number that actually matters: of the setups
    # that passed the 2R filter, how many resolved as wins,
    # losses, or timeouts once price is walked forward.
    # ---------------------------------------------------------

    summary = summarize_outcomes(df)

    print()
    print("OUTCOME STATISTICS (2R-filtered setups)")
    print("-" * 40)

    print(
        f"Labeled setups:       "
        f"{summary['total_labeled']:,}"
    )

    print(
        f"Wins (TP first):      "
        f"{summary['wins']:,}"
    )

    print(
        f"Losses (SL first):    "
        f"{summary['losses']:,}"
    )

    print(
        f"Timeouts:             "
        f"{summary['timeouts']:,}"
    )

    if not pd.isna(summary["win_rate"]):

        print(
            f"Win rate:             "
            f"{summary['win_rate'] * 100:.2f}%"
        )

        # Expectancy in R, assuming a 2R target and 1R stop.
        # This is a lower bound: wins that ran further than the
        # structural target aren't captured here, only whether
        # TP was reached first.
        expectancy_r = (
            summary["win_rate"] * 2.0
            - (1 - summary["win_rate"]) * 1.0
        )

        print(
            f"Expectancy (approx):  "
            f"{expectancy_r:+.2f}R per setup"
        )

    else:
        print("Win rate:             n/a (no labeled setups)")

    # ---------------------------------------------------------
    # 10. Output
    # ---------------------------------------------------------

    print()
    print(f"Saved: {output_path}")

    if valid_rr_count < MIN_SETUPS_FOR_ML:

        print(
            f"WARNING: only {valid_rr_count} setups passed "
            f"the 2R filter on {timeframe.upper()}. Below "
            f"~{MIN_SETUPS_FOR_ML} labeled examples, an XGBoost "
            f"model is likely to fit noise rather than signal. "
            f"Treat this timeframe as descriptive research for "
            f"now, not yet ready for Phase 7 (ML)."
        )

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_symbol(symbol: str) -> None:
    """Run the complete research pipeline for one symbol.

    Fully self-contained: its own per-timeframe sections, its own
    FINAL SUMMARY. Never mixed with another symbol's numbers -- see
    the SYMBOLS comment above for why.
    """

    print()
    print("#" * 70)
    print(f"# {symbol.upper()} SMC / ICT RESEARCH PIPELINE")
    print("#" * 70)

    filenames = _raw_filenames(symbol)

    all_results: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []

    for timeframe, filename in filenames.items():

        input_path = RAW_DIR / filename

        if not input_path.exists():
            print(
                f"\nSkipping {symbol.upper()} {timeframe.upper()}: "
                f"missing raw dataset {input_path} "
                "(run src.mt5_data first)."
            )
            missing_files.append(timeframe)
            continue

        try:

            all_results[timeframe] = process_timeframe(
                symbol,
                timeframe,
                filename,
            )

        except Exception as exc:

            print()
            print(
                f"ERROR processing "
                f"{symbol.upper()} {timeframe.upper()}: {exc}"
            )

            raise

    if not all_results:
        print(
            f"\nNo data found for {symbol.upper()} at all -- "
            "skipping this symbol entirely."
        )
        return

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_wins = 0
    total_losses = 0
    total_timeouts = 0

    for timeframe, df in all_results.items():

        bullish_count = int(
            (
                df["setup_direction"]
                == "bullish"
            ).sum()
        )

        bearish_count = int(
            (
                df["setup_direction"]
                == "bearish"
            ).sum()
        )

        total = (
            bullish_count
            + bearish_count
        )

        valid_2r = int(
            df["valid_2r_setup"].sum()
        )

        outcome_summary = summarize_outcomes(df)

        total_wins += outcome_summary["wins"]
        total_losses += outcome_summary["losses"]
        total_timeouts += outcome_summary["timeouts"]

        print(
            f"{timeframe.upper():<5} | "
            f"setups={total:<5} | "
            f"RR>=2={valid_2r:<5} | "
            f"wins={outcome_summary['wins']:<3} | "
            f"losses={outcome_summary['losses']:<3} | "
            f"timeouts={outcome_summary['timeouts']:<3}"
        )

    print()
    print("-" * 70)

    combined_decided = total_wins + total_losses
    combined_total = total_wins + total_losses + total_timeouts

    print(
        f"COMBINED ({symbol.upper()}, all timeframes pooled): "
        f"{combined_total} labeled setups"
    )

    if combined_decided > 0:

        combined_win_rate = total_wins / combined_decided

        combined_expectancy = (
            combined_win_rate * 2.0
            - (1 - combined_win_rate) * 1.0
        )

        print(
            f"Combined win rate:    "
            f"{combined_win_rate * 100:.2f}% "
            f"({total_wins}W / {total_losses}L, "
            f"{total_timeouts} timeouts excluded)"
        )

        print(
            f"Combined expectancy:  "
            f"{combined_expectancy:+.2f}R per setup"
        )

    if combined_total < MIN_SETUPS_FOR_ML:

        print()
        print(
            f"WARNING: only {combined_total} setups total "
            f"passed the 2R filter across ALL timeframes "
            f"combined. This is very likely too small a "
            f"sample for a reliable win-rate estimate, let "
            f"alone for training an ML model. Before building "
            f"features.py / model.py, consider either (a) "
            f"loosening the target definition (e.g. a fallback "
            f"ATR-multiple or liquidity-pool target when the "
            f"nearest structural swing is unusable), or (b) "
            f"downloading more history / more symbols to grow "
            f"the sample."
        )

    print()
    print(f"{symbol.upper()} research pipeline completed.")

    if missing_files:
        print(
            f"({symbol.upper()} timeframes skipped due to missing "
            f"raw data: {', '.join(t.upper() for t in missing_files)})"
        )


def main() -> None:
    """Run the complete research pipeline for every symbol in SYMBOLS.

    Each symbol is fully independent -- see run_symbol()'s docstring.
    """

    for symbol in SYMBOLS:
        run_symbol(symbol)

    print()
    print("=" * 70)
    print("All symbols processed. No live orders were executed.")
    print("=" * 70)


if __name__ == "__main__":
    main()