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

    if "swing_high_price" not in df.columns:
        return np.nan, -1

    # Only candles BEFORE the setup candle.
    previous_rows = df.iloc[:position]

    if previous_rows.empty:
        return np.nan, -1

    candidate_mask = (
        previous_rows["swing_high_price"].notna()
        & (
            previous_rows["swing_high_price"]
            > entry
        )
    )

    if max_distance is not None:
        candidate_mask &= (
            previous_rows["swing_high_price"] - entry
        ) <= max_distance

    candidates = previous_rows[candidate_mask]

    if candidates.empty:
        return np.nan, -1

    # Nearest structural level above entry.
    target_price = candidates[
        "swing_high_price"
    ].min()

    # Find the most recent occurrence of that price.
    matching_positions = np.flatnonzero(
        (
            previous_rows["swing_high_price"]
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

    if "swing_low_price" not in df.columns:
        return np.nan, -1

    # Only candles BEFORE the setup candle.
    previous_rows = df.iloc[:position]

    if previous_rows.empty:
        return np.nan, -1

    candidate_mask = (
        previous_rows["swing_low_price"].notna()
        & (
            previous_rows["swing_low_price"]
            < entry
        )
    )

    if max_distance is not None:
        candidate_mask &= (
            entry - previous_rows["swing_low_price"]
        ) <= max_distance

    candidates = previous_rows[candidate_mask]

    if candidates.empty:
        return np.nan, -1

    # Nearest structural level below entry.
    target_price = candidates[
        "swing_low_price"
    ].max()

    # Find the most recent occurrence of that price.
    matching_positions = np.flatnonzero(
        (
            previous_rows["swing_low_price"]
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
    """

    if minimum_rr <= 0:
        raise ValueError(
            "minimum_rr must be greater than 0."
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
    # Validate risk and reward
    # -----------------------------------------------------------------

    valid_risk = (
        result["available_risk"] > 0
    )

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
        Actual RR
          ↓
        Minimum RR filter
    """

    return calculate_available_rr(
        df,
        minimum_rr=minimum_rr,
        max_target_atr_multiple=max_target_atr_multiple,
    )