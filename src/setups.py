"""
SMC / ICT setup detection for XAUUSD research.

Setup sequence:

    Liquidity Sweep
          ↓
    Displacement
          ↓
         FVG
          ↓
        Setup
          ↓
    Structural Stop
          ↓
    Structural Target
          ↓
    Actual Reward/Risk
          ↓
        RR >= 2

This module is for historical research and backtesting.
It does not execute trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ict import detect_ict
from src.liquidity import detect_liquidity
from src.targets import calculate_available_rr


def _validate_input(df: pd.DataFrame) -> None:
    """Validate the OHLC columns required by the setup detector."""

    required = {
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty.")


def _ensure_ict_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure liquidity and ICT columns exist.

    Supports both the current liquidity column names and
    the legacy names used by older tests.
    """

    result = df.copy()

    # ---------------------------------------------------------
    # Backward compatibility for old test/data column names
    # ---------------------------------------------------------
    if (
        "bullish_liquidity_sweep" not in result.columns
        and "bullish_sweep" in result.columns
    ):
        result["bullish_liquidity_sweep"] = (
            result["bullish_sweep"]
        )

    if (
        "bearish_liquidity_sweep" not in result.columns
        and "bearish_sweep" in result.columns
    ):
        result["bearish_liquidity_sweep"] = (
            result["bearish_sweep"]
        )

    # ---------------------------------------------------------
    # Calculate liquidity only if still missing
    # ---------------------------------------------------------
    liquidity_columns = {
        "previous_high",
        "previous_low",
        "bullish_liquidity_sweep",
        "bearish_liquidity_sweep",
    }

    if not liquidity_columns.issubset(result.columns):
        result = detect_liquidity(result)

    # ---------------------------------------------------------
    # ICT columns
    # ---------------------------------------------------------
    ict_columns = {
        "bullish_fvg",
        "bearish_fvg",
        "bullish_displacement",
        "bearish_displacement",
    }

    if not ict_columns.issubset(result.columns):
        result = detect_ict(result)

    return result


def _find_sequence(
    sweep: pd.Series,
    displacement: pd.Series,
    fvg: pd.Series,
    window: int,
) -> pd.Series:
    """
    Detect:

        Sweep → Displacement → FVG

    within a maximum total sequence window.

    The setup is marked on the FVG candle.
    """

    if window < 2:
        raise ValueError(
            "window must be at least 2."
        )

    result = pd.Series(
        False,
        index=sweep.index,
        dtype=bool,
    )

    sweep_values = (
        sweep.fillna(False)
        .astype(bool)
        .to_numpy()
    )

    displacement_values = (
        displacement.fillna(False)
        .astype(bool)
        .to_numpy()
    )

    fvg_values = (
        fvg.fillna(False)
        .astype(bool)
        .to_numpy()
    )

    sweep_positions = np.flatnonzero(
        sweep_values
    )

    for sweep_pos in sweep_positions:

        max_fvg_pos = min(
            sweep_pos + window,
            len(sweep) - 1,
        )

        if sweep_pos + 1 > max_fvg_pos:
            continue

        # -----------------------------------------------------
        # Find first displacement after sweep
        # -----------------------------------------------------
        displacement_positions = np.flatnonzero(
            displacement_values[
                sweep_pos + 1 : max_fvg_pos + 1
            ]
        )

        if len(displacement_positions) == 0:
            continue

        displacement_pos = (
            sweep_pos
            + 1
            + displacement_positions[0]
        )

        # -----------------------------------------------------
        # FVG must happen AFTER displacement
        # -----------------------------------------------------
        fvg_start = displacement_pos + 1

        if fvg_start > max_fvg_pos:
            continue

        fvg_positions = np.flatnonzero(
            fvg_values[
                fvg_start : max_fvg_pos + 1
            ]
        )

        if len(fvg_positions) == 0:
            continue

        first_fvg_pos = (
            fvg_start
            + fvg_positions[0]
        )

        # Setup is marked on the FVG candle
        result.iloc[first_fvg_pos] = True

    return result


def detect_setup_context(
    df: pd.DataFrame,
    sequence_window: int = 3,
) -> pd.DataFrame:
    """
    Detect bullish and bearish SMC/ICT setup sequences.
    """

    _validate_input(df)

    # ---------------------------------------------------------
    # Keep backward-compatible behavior:
    # sequence_window < 2 produces no valid
    # three-stage sequence.
    # ---------------------------------------------------------
    if sequence_window < 2:

        result = _ensure_ict_columns(df)

        result["bullish_setup"] = False
        result["bearish_setup"] = False

        result["setup_direction"] = pd.Series(
            None,
            index=result.index,
            dtype="object",
        )

        return result

    result = _ensure_ict_columns(df)

    # ---------------------------------------------------------
    # Bullish sequence
    #
    # Liquidity sweep
    #       ↓
    # Displacement
    #       ↓
    # FVG
    # ---------------------------------------------------------
    result["bullish_setup"] = _find_sequence(
        result["bullish_liquidity_sweep"],
        result["bullish_displacement"],
        result["bullish_fvg"],
        sequence_window,
    )

    # ---------------------------------------------------------
    # Bearish sequence
    #
    # Liquidity sweep
    #       ↓
    # Displacement
    #       ↓
    # FVG
    # ---------------------------------------------------------
    result["bearish_setup"] = _find_sequence(
        result["bearish_liquidity_sweep"],
        result["bearish_displacement"],
        result["bearish_fvg"],
        sequence_window,
    )

    # ---------------------------------------------------------
    # Setup direction
    # ---------------------------------------------------------
    result["setup_direction"] = pd.Series(
        None,
        index=result.index,
        dtype="object",
    )

    result.loc[
        result["bullish_setup"],
        "setup_direction",
    ] = "bullish"

    result.loc[
        result["bearish_setup"],
        "setup_direction",
    ] = "bearish"

    return result


def calculate_setup_levels(
    df: pd.DataFrame,
    reward_multiple: float | None = None,
    sl_buffer: float = 0.0,
    stop_buffer: float | None = None,
) -> pd.DataFrame:
    """
    Calculate entry and structural stop-loss.

    Entry:
        Close of the setup candle.

    Bullish stop:
        Low of the setup candle.

    Bearish stop:
        High of the setup candle.

    IMPORTANT:
        reward_multiple is retained only for backward
        compatibility with the previous API.

        It is NOT used to construct the take-profit.

        The target is determined by structural market
        information in targets.py.

    sl_buffer:
        Legacy parameter name.

    stop_buffer:
        Preferred parameter name.
    """

    # ---------------------------------------------------------
    # Backward-compatible validation
    # ---------------------------------------------------------
    if reward_multiple is not None:

        if reward_multiple <= 0:
            raise ValueError(
                "reward_multiple must be greater than 0."
            )

    if stop_buffer is not None:

        if stop_buffer < 0:
            raise ValueError(
                "stop_buffer must be >= 0."
            )

        if sl_buffer != 0.0:
            raise ValueError(
                "Use either sl_buffer or stop_buffer, "
                "not both."
            )

        sl_buffer = stop_buffer

    if sl_buffer < 0:
        raise ValueError(
            "sl_buffer must be >= 0."
        )

    result = df.copy()

    result["entry"] = np.nan
    result["stop_loss"] = np.nan

    bullish = (
        result["setup_direction"]
        == "bullish"
    )

    bearish = (
        result["setup_direction"]
        == "bearish"
    )

    # ---------------------------------------------------------
    # Entry
    # ---------------------------------------------------------
    result.loc[
        bullish,
        "entry",
    ] = result.loc[
        bullish,
        "close",
    ]

    result.loc[
        bearish,
        "entry",
    ] = result.loc[
        bearish,
        "close",
    ]

    # ---------------------------------------------------------
    # Structural Stop Loss
    # ---------------------------------------------------------
    result.loc[
        bullish,
        "stop_loss",
    ] = (
        result.loc[
            bullish,
            "low",
        ]
        - sl_buffer
    )

    result.loc[
        bearish,
        "stop_loss",
    ] = (
        result.loc[
            bearish,
            "high",
        ]
        + sl_buffer
    )

    # ---------------------------------------------------------
    # Risk
    # ---------------------------------------------------------
    result["risk"] = np.nan

    bullish_risk = (
        result["entry"]
        - result["stop_loss"]
    )

    bearish_risk = (
        result["stop_loss"]
        - result["entry"]
    )

    bullish_valid = (
        bullish
        & (bullish_risk > 0)
    )

    bearish_valid = (
        bearish
        & (bearish_risk > 0)
    )

    result.loc[
        bullish_valid,
        "risk",
    ] = bullish_risk[
        bullish_valid
    ]

    result.loc[
        bearish_valid,
        "risk",
    ] = bearish_risk[
        bearish_valid
    ]

    return result


def detect_setups(
    df: pd.DataFrame,
    reward_multiple: float | None = None,
    sequence_window: int = 3,
    minimum_rr: float = 2.0,
    sl_buffer: float = 0.0,
    stop_buffer: float | None = None,
) -> pd.DataFrame:
    """
    Complete setup detection pipeline.

    The old reward_multiple parameter is accepted for
    compatibility but is NOT used to create TP.

    Actual target:

        Setup
          ↓
        Structural target
          ↓
        Actual RR
          ↓
        RR >= minimum_rr
    """

    if minimum_rr <= 0:
        raise ValueError(
            "minimum_rr must be greater than 0."
        )

    # ---------------------------------------------------------
    # 1. Detect Sweep → Displacement → FVG
    # ---------------------------------------------------------
    result = detect_setup_context(
        df,
        sequence_window=sequence_window,
    )

    # ---------------------------------------------------------
    # 2. Calculate Entry + Structural Stop
    # ---------------------------------------------------------
    result = calculate_setup_levels(
        result,
        reward_multiple=reward_multiple,
        sl_buffer=sl_buffer,
        stop_buffer=stop_buffer,
    )

    # ---------------------------------------------------------
    # 3. Find Structural Target + Actual RR
    # ---------------------------------------------------------
    result = calculate_available_rr(
        result,
        minimum_rr=minimum_rr,
    )

    # ---------------------------------------------------------
    # 4. Structural TP
    # ---------------------------------------------------------
    result["take_profit"] = (
        result["structural_target"]
    )

    # ---------------------------------------------------------
    # 5. Reward
    # ---------------------------------------------------------
    result["reward"] = (
        result["available_reward"]
    )

    # ---------------------------------------------------------
    # 6. Actual RR
    # ---------------------------------------------------------
    result["rr_ratio"] = (
        result["available_rr"]
    )

    result["minimum_rr"] = minimum_rr

    # ---------------------------------------------------------
    # 7. Valid setup
    # ---------------------------------------------------------
    result["valid_setup"] = (
        result["setup_direction"].notna()
        & result["risk"].notna()
        & result["structural_target"].notna()
        & result["available_rr"].notna()
    )

    # ---------------------------------------------------------
    # 8. Minimum 2R filter
    # ---------------------------------------------------------
    result["valid_2r_setup"] = (
        result["valid_setup"]
        & (
            result["rr_ratio"]
            >= minimum_rr
        )
    )

    return result