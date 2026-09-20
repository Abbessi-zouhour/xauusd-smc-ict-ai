"""
Structural target detection for XAUUSD SMC/ICT research.

This module identifies a market-derived structural target for a detected
setup and calculates the available reward-to-risk ratio.

Bullish:
    Entry
      ↓
    Stop Loss
      ↓
    Structural Target above entry

Bearish:
    Structural Target below entry
      ↓
    Entry
      ↓
    Stop Loss

Minimum acceptable RR:
    RR >= 2.0

Important:
    Targets are selected only from confirmed swing information
    available BEFORE the setup candle.

This module is for historical research and backtesting.
It does not execute trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_input(df: pd.DataFrame) -> None:
    """Validate the columns required for structural targets."""

    required = {
        "high",
        "low",
        "close",
        "entry",
        "stop_loss",
        "setup_direction",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )


# ---------------------------------------------------------------------
# Bullish structural target
# ---------------------------------------------------------------------

def _find_previous_bullish_target(
    df: pd.DataFrame,
    position: int,
    entry: float,
    max_distance: float | None = None,
) -> tuple[float, int]:
    """
    Find the nearest previous confirmed swing high above entry.

    Only information available BEFORE the setup candle is used.

    max_distance:
        Optional cap, in absolute price units, on how far above
        entry the target may be. Candidates further away than
        this are excluded, since a structurally "valid" but
        implausibly distant target is unlikely to be reached
        within any reasonable holding period. Pass None (default)
        to keep the original unbounded behavior.

    Returns:
        (target_price, target_position)

    If no valid target exists:
        (np.nan, -1)
    """

    # "confirmed_swing_high_price" only becomes non-NaN on the row
    # where a swing high actually became knowable (pivot + right_bars).
    # Using the raw "swing_high_price" column here would leak
    # information from up to right_bars candles in the future,
    # since that column is written at the pivot row itself.
    price_col = (
        "confirmed_swing_high_price"
        if "confirmed_swing_high_price" in df.columns
        else "swing_high_price"
    )

    # Only candles BEFORE the setup candle.
    previous_rows = df.iloc[:position]

    if previous_rows.empty:
        return np.nan, -1

    candidate_mask = (
        previous_rows[price_col].notna()
        & (
            previous_rows[price_col]
            > entry
        )
    )

    if max_distance is not None:
        candidate_mask &= (
            previous_rows[price_col] - entry
        ) <= max_distance

    candidates = previous_rows[candidate_mask]

    if candidates.empty:
        return np.nan, -1

    # Nearest structural level above entry.
    target_price = candidates[
        price_col
    ].min()

    # Find the most recent occurrence of that price.
    matching_positions = np.flatnonzero(
        (
            previous_rows[price_col]
            .to_numpy()
            == target_price
        )
    )

    if len(matching_positions) == 0:
        return np.nan, -1

    target_position = int(
        matching_positions[-1]
    )

    return float(target_price), target_position


# ---------------------------------------------------------------------
# Bearish structural target
# ---------------------------------------------------------------------

def _find_previous_bearish_target(
    df: pd.DataFrame,
    position: int,
    entry: float,
    max_distance: float | None = None,
) -> tuple[float, int]:
    """
    Find the nearest previous confirmed swing low below entry.

    Only information available BEFORE the setup candle is used.

    max_distance:
        Optional cap, in absolute price units, on how far below
        entry the target may be. See _find_previous_bullish_target
        for the rationale. Pass None (default) to keep the
        original unbounded behavior.

    Returns:
        (target_price, target_position)

    If no valid target exists:
        (np.nan, -1)
    """

    # See the matching comment in _find_previous_bullish_target --
    # this must read the confirmed-price column, not the raw
    # pivot-row column, or it leaks up to right_bars candles of
    # future information into the target.
    price_col = (
        "confirmed_swing_low_price"
        if "confirmed_swing_low_price" in df.columns
        else "swing_low_price"
    )

    # Only candles BEFORE the setup candle.
    previous_rows = df.iloc[:position]

    if previous_rows.empty:
        return np.nan, -1

    candidate_mask = (
        previous_rows[price_col].notna()
        & (
            previous_rows[price_col]
            < entry
        )
    )

    if max_distance is not None:
        candidate_mask &= (
            entry - previous_rows[price_col]
        ) <= max_distance

    candidates = previous_rows[candidate_mask]

    if candidates.empty:
        return np.nan, -1

    # Nearest structural level below entry.
    target_price = candidates[
        price_col
    ].max()

    # Find the most recent occurrence of that price.
    matching_positions = np.flatnonzero(
        (
            previous_rows[price_col]
            .to_numpy()
            == target_price
        )
    )

    if len(matching_positions) == 0:
        return np.nan, -1

    target_position = int(
        matching_positions[-1]
    )

    return float(target_price), target_position


# ---------------------------------------------------------------------
# Structural target detection
# ---------------------------------------------------------------------

def find_structural_targets(
    df: pd.DataFrame,
    max_target_atr_multiple: float | None = None,
) -> pd.DataFrame:
    """
    Find a structural target for every detected setup.

    Bullish:
        nearest previous confirmed swing high above entry.

    Bearish:
        nearest previous confirmed swing low below entry.

    No future candles are used.

    max_target_atr_multiple:
        Optional research filter. When set, a candidate target
        is only accepted if its distance from entry is no more
        than (max_target_atr_multiple * atr) at the setup candle,
        where 'atr' must already be present as a column (see
        src.ict.detect_displacement). This excludes structurally
        "valid" targets that are so far away, relative to recent
        volatility, that reaching them within any realistic
        holding period is implausible -- without this, the
        2R filter tends to select almost exclusively these
        long-shot targets, since nearby swings rarely clear 2R
        on their own. When atr is missing or non-positive for a
        given row, no cap is applied for that row (treated as
        unknown, not as "no limit"). Pass None (default) to
        reproduce the original unbounded behavior.

    Additional audit columns:
        target_index
        target_age
    """

    _validate_input(df)

    result = df.copy()

    result["structural_target"] = np.nan
    result["target_index"] = -1
    result["target_age"] = np.nan

    has_atr = "atr" in result.columns

    for position in range(len(result)):

        direction = result.iloc[
            position
        ]["setup_direction"]

        entry = result.iloc[
            position
        ]["entry"]

        if pd.isna(entry):
            continue

        entry = float(entry)

        max_distance = None

        if max_target_atr_multiple is not None and has_atr:

            atr_value = result.iloc[position]["atr"]

            if pd.notna(atr_value) and atr_value > 0:
                max_distance = (
                    max_target_atr_multiple * float(atr_value)
                )

        # -------------------------------------------------------------
        # Bullish setup
        # -------------------------------------------------------------

        if direction == "bullish":

            target, target_position = (
                _find_previous_bullish_target(
                    result,
                    position,
                    entry,
                    max_distance=max_distance,
                )
            )

        # -------------------------------------------------------------
        # Bearish setup
        # -------------------------------------------------------------

        elif direction == "bearish":

            target, target_position = (
                _find_previous_bearish_target(
                    result,
                    position,
                    entry,
                    max_distance=max_distance,
                )
            )

        else:
            continue

        # -------------------------------------------------------------
        # Store target
        # -------------------------------------------------------------

        if not pd.isna(target):

            result.iat[
                position,
                result.columns.get_loc(
                    "structural_target"
                ),
            ] = target

            result.iat[
                position,
                result.columns.get_loc(
                    "target_index"
                ),
            ] = target_position

            result.iat[
                position,
                result.columns.get_loc(
                    "target_age"
                ),
            ] = position - target_position

    return result


# ---------------------------------------------------------------------
# Available RR
# ---------------------------------------------------------------------

def calculate_available_rr(
    df: pd.DataFrame,
    minimum_rr: float = 2.0,
    max_target_atr_multiple: float | None = None,
    min_risk_atr_multiple: float | None = None,
    spread: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate the actual reward-to-risk ratio.

    Bullish:
        risk = entry - stop_loss
        reward = structural_target - entry

    Bearish:
        risk = stop_loss - entry
        reward = entry - structural_target

    A setup is valid when:
        available_rr >= minimum_rr

    max_target_atr_multiple:
        See find_structural_targets(). Passed through unchanged.

    min_risk_atr_multiple:
        Optional research filter, symmetric to
        max_target_atr_multiple but on the risk side. The
        stop_loss placed by setups.calculate_setup_levels() is
        the high/low of a single candle, which can be tiny
        relative to normal volatility. A tiny risk denominator
        inflates available_rr without the setup actually being
        any more likely to reach its target -- if anything a
        noise-sized stop is more likely to be swept immediately.
        When set, a setup is only accepted if
        available_risk >= min_risk_atr_multiple * atr (atr must
        already be a column, see src.ict.detect_displacement).
        When atr is missing or non-positive for a given row, no
        floor is applied for that row. Pass None (default) to
        reproduce the original unbounded behavior.

    spread:
        Fixed round-trip cost, in price units, applied to every
        setup before anything else in this function (the RR
        filter, min_risk_atr_multiple, and every downstream
        consumer of available_risk/available_reward/available_rr
        all see cost-adjusted numbers). We only have OHLC, not a
        historical bid/ask series, so this approximates a
        broker's spread as one fixed value rather than something
        that varies with volatility or price level over the
        dataset's history -- a real spread does neither perfectly,
        but a fixed value in price units is a much closer
        approximation to how CFD/forex-style spreads are usually
        quoted than a spread that scales with price level would
        be (see run_research.py / threshold_sweep.py's SPREAD
        constant for the source of a reasonable default).

        Mechanically: effective_risk = raw_risk + spread,
        effective_reward = raw_reward - spread. A winning setup's
        realized R shrinks (available_rr uses the *effective*
        numbers). A losing setup's realized R is still exactly
        -1.0 by construction, because R itself is now measured
        against the cost-inclusive risk unit -- the loss isn't
        being made artificially worse, "1R" is just being defined
        more honestly. Setups whose reward can't even cover the
        spread (effective_reward <= 0) are excluded entirely, same
        as any other invalid reward. Pass 0.0 (default) to
        reproduce the original no-cost behavior.
    """

    if minimum_rr <= 0:
        raise ValueError(
            "minimum_rr must be greater than 0."
        )

    if spread < 0:
        raise ValueError(
            "spread must be >= 0."
        )

    result = find_structural_targets(
        df,
        max_target_atr_multiple=max_target_atr_multiple,
    )

    result["available_reward"] = np.nan
    result["available_risk"] = np.nan
    result["available_rr"] = np.nan

    # -----------------------------------------------------------------
    # Direction masks
    # -----------------------------------------------------------------

    bullish = (
        result["setup_direction"]
        == "bullish"
    )

    bearish = (
        result["setup_direction"]
        == "bearish"
    )

    # -----------------------------------------------------------------
    # Bullish RR
    # -----------------------------------------------------------------

    bullish_valid = (
        bullish
        & result["entry"].notna()
        & result["stop_loss"].notna()
        & result["structural_target"].notna()
    )

    bullish_risk = (
        result["entry"]
        - result["stop_loss"]
    )

    bullish_reward = (
        result["structural_target"]
        - result["entry"]
    )

    result.loc[
        bullish_valid,
        "available_risk",
    ] = bullish_risk[
        bullish_valid
    ]

    result.loc[
        bullish_valid,
        "available_reward",
    ] = bullish_reward[
        bullish_valid
    ]

    # -----------------------------------------------------------------
    # Bearish RR
    # -----------------------------------------------------------------

    bearish_valid = (
        bearish
        & result["entry"].notna()
        & result["stop_loss"].notna()
        & result["structural_target"].notna()
    )

    bearish_risk = (
        result["stop_loss"]
        - result["entry"]
    )

    bearish_reward = (
        result["entry"]
        - result["structural_target"]
    )

    result.loc[
        bearish_valid,
        "available_risk",
    ] = bearish_risk[
        bearish_valid
    ]

    result.loc[
        bearish_valid,
        "available_reward",
    ] = bearish_reward[
        bearish_valid
    ]

    # -----------------------------------------------------------------
    # Spread cost
    #
    # Applied here, after raw risk/reward are computed from price
    # levels but BEFORE min_risk_atr_multiple and the RR filter --
    # both of those, and every downstream consumer of
    # available_risk/available_reward/available_rr, should see
    # cost-adjusted numbers. See the spread parameter's docstring
    # above for the mechanics and why a fixed value is used.
    # -----------------------------------------------------------------

    result["available_risk_gross"] = result["available_risk"]
    result["available_reward_gross"] = result["available_reward"]

    if spread > 0:

        has_levels = result["available_risk"].notna()

        result.loc[
            has_levels,
            "available_risk",
        ] = (
            result.loc[has_levels, "available_risk"]
            + spread
        )

        result.loc[
            has_levels,
            "available_reward",
        ] = (
            result.loc[has_levels, "available_reward"]
            - spread
        )

    # -----------------------------------------------------------------
    # Validate risk and reward
    # -----------------------------------------------------------------

    valid_risk = (
        result["available_risk"] > 0
    )

    if min_risk_atr_multiple is not None and "atr" in result.columns:

        atr = result["atr"]
        has_atr = atr.notna() & (atr > 0)

        min_risk = min_risk_atr_multiple * atr

        risk_floor_ok = (
            ~has_atr
            | (result["available_risk"] >= min_risk)
        )

        valid_risk = valid_risk & risk_floor_ok

    valid_reward = (
        result["available_reward"] > 0
    )

    valid_rr = (
        valid_risk
        & valid_reward
    )

    # -----------------------------------------------------------------
    # Actual RR
    # -----------------------------------------------------------------

    result.loc[
        valid_rr,
        "available_rr",
    ] = (
        result.loc[
            valid_rr,
            "available_reward",
        ]
        / result.loc[
            valid_rr,
            "available_risk",
        ]
    )

    # -----------------------------------------------------------------
    # Minimum RR filter
    # -----------------------------------------------------------------

    result["valid_2r_setup"] = (
        result["available_rr"]
        >= minimum_rr
    )

    return result


# ---------------------------------------------------------------------
# Minimum RR filter
# ---------------------------------------------------------------------

def apply_minimum_rr_filter(
    df: pd.DataFrame,
    minimum_rr: float = 2.0,
) -> pd.DataFrame:
    """
    Apply the minimum RR filter.

    Examples:
        1.5R -> False
        2.0R -> True
        2.5R -> True
        4.0R -> True
    """

    if minimum_rr <= 0:
        raise ValueError(
            "minimum_rr must be greater than 0."
        )

    if "available_rr" not in df.columns:

        result = calculate_available_rr(
            df,
            minimum_rr=minimum_rr,
        )

    else:

        result = df.copy()

        result["valid_2r_setup"] = (
            result["available_rr"]
            >= minimum_rr
        )

    return result


# ---------------------------------------------------------------------
# Complete target pipeline
# ---------------------------------------------------------------------

def detect_targets(
    df: pd.DataFrame,
    minimum_rr: float = 2.0,
    max_target_atr_multiple: float | None = None,
    min_risk_atr_multiple: float | None = None,
    spread: float = 0.0,
) -> pd.DataFrame:
    """
    Complete structural-target pipeline.

    Pipeline:

        Setup
          ↓
        Structural Target
          ↓
        Available Reward
          ↓
        Available Risk
          ↓
        Spread cost
          ↓
        Actual RR
          ↓
        Minimum RR filter
    """

    return calculate_available_rr(
        df,
        minimum_rr=minimum_rr,
        max_target_atr_multiple=max_target_atr_multiple,
        min_risk_atr_multiple=min_risk_atr_multiple,
        spread=spread,
    )