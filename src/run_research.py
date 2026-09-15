"""
Run the XAUUSD SMC / ICT research pipeline.

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
from src.liquidity import detect_liquidity
from src.ict import detect_ict
from src.setups import detect_setups
from src.structure import detect_market_structure, detect_swings


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------
# Timeframes
# ---------------------------------------------------------------------

TIMEFRAMES = {
    "m15": "xauusd_m15.csv",
    "h1": "xauusd_h1.csv",
    "h4": "xauusd_h4.csv",
    "d1": "xauusd_d1.csv",
}


# ---------------------------------------------------------------------
# Process one timeframe
# ---------------------------------------------------------------------

def process_timeframe(
    timeframe: str,
    filename: str,
) -> pd.DataFrame:
    """Process one timeframe through the complete research pipeline."""

    print()
    print("=" * 70)
    print(f"Processing {timeframe.upper()}")
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

    df = detect_setups(
        df,
        sequence_window=3,
        minimum_rr=2.0,
        stop_buffer=0.0,
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
        / f"xauusd_{timeframe}_setups.csv"
    )

    df.to_csv(
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

    total_setups = (
        bullish_count
        + bearish_count
    )

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
    # 10. Output
    # ---------------------------------------------------------

    print()
    print(f"Saved: {output_path}")

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete research pipeline."""

    print()
    print("=" * 70)
    print("XAUUSD SMC / ICT RESEARCH PIPELINE")
    print("=" * 70)

    all_results: dict[str, pd.DataFrame] = {}

    for timeframe, filename in TIMEFRAMES.items():

        try:

            all_results[timeframe] = process_timeframe(
                timeframe,
                filename,
            )

        except Exception as exc:

            print()
            print(
                f"ERROR processing "
                f"{timeframe.upper()}: {exc}"
            )

            raise

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

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

        print(
            f"{timeframe.upper():<5} | "
            f"setups={total:<5} | "
            f"RR>=2={valid_2r:<5}"
        )

    print()
    print("Research pipeline completed.")
    print("No live orders were executed.")


if __name__ == "__main__":
    main()