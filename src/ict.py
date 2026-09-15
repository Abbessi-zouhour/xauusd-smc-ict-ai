"""
ICT (Inner Circle Trader) concepts for XAUUSD research/backtesting.

Implemented concepts:
- Fair Value Gaps (FVG)
- Bullish / bearish displacement
- Liquidity sweep + displacement context
- FVG boundaries
- FVG age / active state

Design principles:
- No future-looking values are used for a signal at candle i.
- Signals are based only on candles available up to candle i.
- Functions return DataFrames so they can be composed into the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

REQUIRED_COLUMNS = {"open", "high", "low", "close"}


def _validate_ohlc(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame contains OHLC columns."""
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required OHLC columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty.")


# ============================================================
# Fair Value Gaps
# ============================================================

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect three-candle Fair Value Gaps.

    Bullish FVG:
        Current candle low > candle two bars earlier high.

        candle[i-2].high < candle[i].low

    Bearish FVG:
        Current candle high < candle two bars earlier low.

        candle[i-2].low > candle[i].high

    The FVG is recorded on candle i, when the third candle
    confirms the gap.

    Added columns:
        bullish_fvg
        bearish_fvg
        fvg_type
        fvg_top
        fvg_bottom
        fvg_size

    Returns:
        Copy of the input DataFrame with FVG columns.
    """

    _validate_ohlc(df)

    result = df.copy()

    # Previous two-candle boundaries
    high_2 = result["high"].shift(2)
    low_2 = result["low"].shift(2)

    # Bullish FVG:
    # candle[i-2] high < candle[i] low
    bullish = result["low"] > high_2

    # Bearish FVG:
    # candle[i-2] low > candle[i] high
    bearish = result["high"] < low_2

    result["bullish_fvg"] = bullish.fillna(False)
    result["bearish_fvg"] = bearish.fillna(False)

    result["fvg_type"] = np.select(
        [
            result["bullish_fvg"],
            result["bearish_fvg"],
        ],
        [
            "bullish",
            "bearish",
        ],
        default=None,
    )

    # FVG boundaries
    #
    # Bullish:
    # bottom = candle[i-2].high
    # top    = candle[i].low
    #
    # Bearish:
    # bottom = candle[i].high
    # top    = candle[i-2].low

    result["fvg_bottom"] = np.nan
    result["fvg_top"] = np.nan

    bullish_mask = result["bullish_fvg"]
    bearish_mask = result["bearish_fvg"]

    result.loc[bullish_mask, "fvg_bottom"] = high_2[bullish_mask]
    result.loc[bullish_mask, "fvg_top"] = result.loc[
        bullish_mask, "low"
    ]

    result.loc[bearish_mask, "fvg_bottom"] = result.loc[
        bearish_mask, "high"
    ]
    result.loc[bearish_mask, "fvg_top"] = low_2[bearish_mask]

    result["fvg_size"] = (
        result["fvg_top"] - result["fvg_bottom"]
    )

    return result


# ============================================================
# Displacement
# ============================================================

def detect_displacement(
    df: pd.DataFrame,
    body_ratio_threshold: float = 0.60,
    range_multiplier: float = 1.5,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Detect strong directional candles interpreted as displacement.

    A candle is considered bullish displacement when:

        1. Close > Open
        2. Body / range >= body_ratio_threshold
        3. Candle range >= ATR * range_multiplier

    A candle is considered bearish displacement when:

        1. Close < Open
        2. Body / range >= body_ratio_threshold
        3. Candle range >= ATR * range_multiplier

    Parameters:
        body_ratio_threshold:
            Minimum fraction of candle range occupied by the body.

        range_multiplier:
            Minimum candle range relative to ATR.

        atr_period:
            Rolling ATR period.

    Added columns:
        candle_range
        candle_body
        body_ratio
        atr
        bullish_displacement
        bearish_displacement
        displacement

    Returns:
        Copy of the input DataFrame.
    """

    _validate_ohlc(df)

    if body_ratio_threshold <= 0 or body_ratio_threshold > 1:
        raise ValueError(
            "body_ratio_threshold must be between 0 and 1."
        )

    if range_multiplier <= 0:
        raise ValueError("range_multiplier must be greater than 0.")

    if atr_period <= 0:
        raise ValueError("atr_period must be greater than 0.")

    result = df.copy()

    result["candle_range"] = (
        result["high"] - result["low"]
    )

    result["candle_body"] = (
        result["close"] - result["open"]
    ).abs()

    result["body_ratio"] = np.where(
        result["candle_range"] > 0,
        result["candle_body"] / result["candle_range"],
        0.0,
    )

    # True Range
    previous_close = result["close"].shift(1)

    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["atr"] = true_range.rolling(
        window=atr_period,
        min_periods=atr_period,
    ).mean()

    valid_range = (
        result["candle_range"] > 0
    )

    strong_body = (
        result["body_ratio"] >= body_ratio_threshold
    )

    strong_range = (
        result["candle_range"]
        >= result["atr"] * range_multiplier
    )

    result["bullish_displacement"] = (
        (result["close"] > result["open"])
        & valid_range
        & strong_body
        & strong_range
    )

    result["bearish_displacement"] = (
        (result["close"] < result["open"])
        & valid_range
        & strong_body
        & strong_range
    )

    result["displacement"] = np.select(
        [
            result["bullish_displacement"],
            result["bearish_displacement"],
        ],
        [
            "bullish",
            "bearish",
        ],
        default=None,
    )

    return result


# ============================================================
# FVG + Displacement
# ============================================================

def detect_fvg_with_displacement(
    df: pd.DataFrame,
    body_ratio_threshold: float = 0.60,
    range_multiplier: float = 1.5,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Detect FVGs and displacement together.

    This is useful for identifying stronger ICT-style
    displacement/FVG combinations.

    Added column:
        displacement_fvg

    Values:
        bullish
        bearish
        None
    """

    result = detect_displacement(
        df,
        body_ratio_threshold=body_ratio_threshold,
        range_multiplier=range_multiplier,
        atr_period=atr_period,
    )

    result = detect_fvg(result)

    result["displacement_fvg"] = np.select(
        [
            result["bullish_fvg"]
            & result["bullish_displacement"],
            result["bearish_fvg"]
            & result["bearish_displacement"],
        ],
        [
            "bullish",
            "bearish",
        ],
        default=None,
    )

    return result


# ============================================================
# Liquidity Sweep + Displacement
# ============================================================

def detect_sweep_displacement(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine previously detected liquidity sweeps with
    displacement.

    Expected liquidity columns:
        bullish_liquidity_sweep
        bearish_liquidity_sweep

    If those columns do not exist, a ValueError is raised.

    Added columns:
        bullish_sweep_displacement
        bearish_sweep_displacement
        sweep_displacement

    This does NOT assume that every sweep is a valid setup.
    It only marks the observed combination.
    """

    _validate_ohlc(df)

    required = {
        "bullish_liquidity_sweep",
        "bearish_liquidity_sweep",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing liquidity columns: "
            f"{sorted(missing)}. "
            "Run detect_liquidity() first."
        )

    result = df.copy()

    if "bullish_displacement" not in result.columns:
        result = detect_displacement(result)

    if "bearish_displacement" not in result.columns:
        result = detect_displacement(result)

    result["bullish_sweep_displacement"] = (
        result["bullish_liquidity_sweep"].fillna(False)
        & result["bullish_displacement"].fillna(False)
    )

    result["bearish_sweep_displacement"] = (
        result["bearish_liquidity_sweep"].fillna(False)
        & result["bearish_displacement"].fillna(False)
    )

    result["sweep_displacement"] = np.select(
        [
            result["bullish_sweep_displacement"],
            result["bearish_sweep_displacement"],
        ],
        [
            "bullish",
            "bearish",
        ],
        default=None,
    )

    return result


# ============================================================
# Complete ICT Pipeline
# ============================================================

def detect_ict(
    df: pd.DataFrame,
    body_ratio_threshold: float = 0.60,
    range_multiplier: float = 1.5,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Run the main ICT detection pipeline.

    Pipeline:

        OHLC
          ↓
        Displacement
          ↓
        Fair Value Gaps
          ↓
        ICT features

    Liquidity sweeps can be combined later using
    detect_sweep_displacement() after liquidity detection.
    """

    result = detect_displacement(
        df,
        body_ratio_threshold=body_ratio_threshold,
        range_multiplier=range_multiplier,
        atr_period=atr_period,
    )

    result = detect_fvg(result)

    result["ict_fvg_displacement"] = np.select(
        [
            result["bullish_fvg"]
            & result["bullish_displacement"],
            result["bearish_fvg"]
            & result["bearish_displacement"],
        ],
        [
            "bullish",
            "bearish",
        ],
        default=None,
    )

    return result