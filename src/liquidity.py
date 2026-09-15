import numpy as np
import pandas as pd


def detect_previous_levels(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Detect rolling previous highs and lows.

    shift(1) is critical:
    the current candle must NOT be included when
    calculating its previous liquidity levels.
    """

    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    result = df.copy()

    result["previous_high"] = (
        result["high"]
        .rolling(lookback)
        .max()
        .shift(1)
    )

    result["previous_low"] = (
        result["low"]
        .rolling(lookback)
        .min()
        .shift(1)
    )

    return result


def detect_equal_levels(
    df: pd.DataFrame,
    tolerance: float = 0.10,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Detect approximate equal highs and equal lows.

    tolerance is expressed in price units.

    Example:
        tolerance=0.10 means two levels within
        $0.10 are considered approximately equal.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be > 0")

    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    result = df.copy()

    rolling_high = (
        result["high"]
        .rolling(lookback)
    )

    rolling_low = (
        result["low"]
        .rolling(lookback)
    )

    high_range = (
        rolling_high.max()
        - rolling_high.min()
    )

    low_range = (
        rolling_low.max()
        - rolling_low.min()
    )

    result["equal_high_zone"] = (
        high_range <= tolerance
    )

    result["equal_low_zone"] = (
        low_range <= tolerance
    )

    return result


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    tolerance: float = 0.10,
) -> pd.DataFrame:
    """
    Detect basic liquidity sweeps.

    Bullish sweep:
        price trades below a previous low,
        then closes back above that previous low.

    Bearish sweep:
        price trades above a previous high,
        then closes back below that previous high.

    These are research definitions and will be
    refined later using swing/liquidity context.
    """

    if (
        "previous_high" not in df.columns
        or "previous_low" not in df.columns
    ):
        raise ValueError(
            "Run detect_previous_levels() first."
        )

    result = df.copy()

    result["bullish_liquidity_sweep"] = False
    result["bearish_liquidity_sweep"] = False

    previous_low = result["previous_low"]
    previous_high = result["previous_high"]

    result["bullish_liquidity_sweep"] = (
        previous_low.notna()
        & (result["low"] < previous_low - tolerance)
        & (result["close"] > previous_low)
    )

    result["bearish_liquidity_sweep"] = (
        previous_high.notna()
        & (result["high"] > previous_high + tolerance)
        & (result["close"] < previous_high)
    )

    return result


def detect_liquidity(
    df: pd.DataFrame,
    lookback: int = 20,
    tolerance: float = 0.10,
) -> pd.DataFrame:
    """
    Run the complete liquidity detection pipeline.
    """

    result = detect_previous_levels(
        df,
        lookback=lookback,
    )

    result = detect_equal_levels(
        result,
        tolerance=tolerance,
        lookback=lookback,
    )

    result = detect_liquidity_sweeps(
        result,
        tolerance=tolerance,
    )

    return result