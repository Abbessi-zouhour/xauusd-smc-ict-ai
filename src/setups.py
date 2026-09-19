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
) -> tuple[pd.Series, pd.Series]:
    """
    Detect:

        Sweep → Displacement → FVG

    within a maximum total sequence window.

    The setup is marked on the FVG candle.

    Returns:
        (setup_mask, sequence_start)

        setup_mask:
            Boolean series, True on the FVG candle that
            completes a sweep -> displacement -> FVG sequence.

        sequence_start:
            Integer position of the sweep candle that started
            the sequence, recorded on the same FVG row as
            setup_mask. -1 where there is no setup. This is
            what actually invalidates the setup thesis (the
            sweep's extreme), which can be earlier than -- and
            further away than -- the confirmation candle itself.
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

    sequence_start = pd.Series(
        -1,
        index=sweep.index,
        dtype="int64",
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
        sequence_start.iloc[first_fvg_pos] = sweep_pos

    return result, sequence_start


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

        result["setup_sequence_start"] = -1

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
    result["bullish_setup"], bullish_start = _find_sequence(
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
    result["bearish_setup"], bearish_start = _find_sequence(
        result["bearish_liquidity_sweep"],
        result["bearish_displacement"],
        result["bearish_fvg"],
        sequence_window,
    )

    # ---------------------------------------------------------
    # Sequence start (position of the sweep candle that began
    # the matched sequence). Used to compute the real
    # invalidation extreme in calculate_setup_levels(), rather
    # than only looking at the confirmation candle.
    # ---------------------------------------------------------
    result["setup_sequence_start"] = -1

    result.loc[
        result["bullish_setup"],
        "setup_sequence_start",
    ] = bullish_start[result["bullish_setup"]]

    result.loc[
        result["bearish_setup"],
        "setup_sequence_start",
    ] = bearish_start[result["bearish_setup"]]

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
    entry_mode: str = "close",
    fvg_retracement: float = 0.5,
) -> pd.DataFrame:
    """
    Calculate entry and structural stop-loss.

    Entry:
        entry_mode="close" (default, original behavior):
            Close of the setup (FVG confirmation) candle.
            This is a "chase" entry -- by the time it fills,
            the sweep + displacement + FVG sequence has already
            happened, so it's far from the sweep's extreme.
            Combined with the honest stop below (the sweep's
            actual invalidation point), that produces a large,
            realistic risk with little room for a nearby target
            to give a good RR.

        entry_mode="fvg_midpoint":
            A retracement entry inside the FVG zone itself,
            at fvg_retracement of the way from fvg_top toward
            fvg_bottom (0.5 = the zone's midpoint, the common
            ICT "optimal trade entry" convention). This is a
            limit-style entry: it assumes price pulls back into
            the imbalance before continuing, which is much
            closer to the sweep's extreme than the confirmation
            candle's close, without weakening the stop.
            Requires "fvg_top" and "fvg_bottom" columns (see
            src.ict.detect_fvg). Rows where a setup exists but
            these columns are missing/NaN fall back to the
            "close" entry for that row.

            IMPORTANT: an entry away from the setup candle's own
            close is not guaranteed to ever be filled. Use
            label_setup_outcomes() (labels.py) to label
            outcomes -- it already scans forward for the first
            bar where price actually reaches this entry level,
            and excludes setups that never got filled rather
            than assuming a fill that never happened.

    Bullish stop:
        The lowest low reached between the liquidity-sweep
        candle and the confirmation (FVG) candle, inclusive --
        i.e. the real invalidation point of the setup thesis.
        Falls back to the low of the setup candle alone when
        the sweep's position is unknown (setup_sequence_start
        missing/-1), preserving the old behavior for callers
        that construct setup_direction manually.

    Bearish stop:
        Mirror of the bullish case: the highest high reached
        between the sweep candle and the confirmation candle.

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

    if entry_mode not in ("close", "fvg_midpoint"):
        raise ValueError(
            'entry_mode must be "close" or "fvg_midpoint".'
        )

    if not (0.0 <= fvg_retracement <= 1.0):
        raise ValueError(
            "fvg_retracement must be between 0 and 1."
        )

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

    if entry_mode == "fvg_midpoint":

        has_zone = (
            "fvg_top" in result.columns
            and "fvg_bottom" in result.columns
        )

        if has_zone:

            zone_ok = (
                result["fvg_top"].notna()
                & result["fvg_bottom"].notna()
            )

            # Bullish: retrace DOWN from fvg_top toward
            # fvg_bottom by fvg_retracement.
            bullish_retrace = bullish & zone_ok

            result.loc[
                bullish_retrace,
                "entry",
            ] = (
                result.loc[bullish_retrace, "fvg_top"]
                - fvg_retracement
                * (
                    result.loc[bullish_retrace, "fvg_top"]
                    - result.loc[bullish_retrace, "fvg_bottom"]
                )
            )

            # Bearish: retrace UP from fvg_bottom toward
            # fvg_top by fvg_retracement.
            bearish_retrace = bearish & zone_ok

            result.loc[
                bearish_retrace,
                "entry",
            ] = (
                result.loc[bearish_retrace, "fvg_bottom"]
                + fvg_retracement
                * (
                    result.loc[bearish_retrace, "fvg_top"]
                    - result.loc[bearish_retrace, "fvg_bottom"]
                )
            )

            # Rows without a usable FVG zone keep the "close"
            # entry already assigned above.

    # ---------------------------------------------------------
    # Structural Stop Loss
    #
    # Uses the extreme reached between the sweep candle and the
    # confirmation candle, not just the confirmation candle's
    # own wick -- that extreme is the actual point that
    # invalidates the setup thesis. A stop placed only on the
    # confirmation candle can be far tighter than that real
    # invalidation point, which artificially inflates RR and
    # gets setups stopped out by ordinary noise.
    # ---------------------------------------------------------
    has_sequence_start = (
        "setup_sequence_start" in result.columns
    )

    highs = result["high"].to_numpy()
    lows = result["low"].to_numpy()

    stop_loss_col = result.columns.get_loc("stop_loss")
    direction_col = result.columns.get_loc("setup_direction")

    if has_sequence_start:
        start_col = result.columns.get_loc(
            "setup_sequence_start"
        )

    for pos in range(len(result)):

        direction = result.iat[pos, direction_col]

        if direction not in ("bullish", "bearish"):
            continue

        start_pos = pos

        if has_sequence_start:
            start_value = result.iat[pos, start_col]

            if (
                pd.notna(start_value)
                and int(start_value) >= 0
                and int(start_value) <= pos
            ):
                start_pos = int(start_value)

        if direction == "bullish":
            stop = lows[start_pos : pos + 1].min() - sl_buffer
        else:
            stop = highs[start_pos : pos + 1].max() + sl_buffer

        result.iat[pos, stop_loss_col] = stop

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
    max_target_atr_multiple: float | None = None,
    min_risk_atr_multiple: float | None = None,
    entry_mode: str = "close",
    fvg_retracement: float = 0.5,
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

    max_target_atr_multiple:
        Optional research filter passed through to
        targets.calculate_available_rr(). See
        targets.find_structural_targets() for the rationale.

    min_risk_atr_multiple:
        Optional research filter passed through to
        targets.calculate_available_rr(). Rejects setups whose
        stop_loss (the high/low of a single candle) is smaller
        than min_risk_atr_multiple * atr, since a noise-sized
        stop inflates available_rr without making the setup any
        more likely to actually reach its target. See
        targets.calculate_available_rr() for the rationale.

    entry_mode / fvg_retracement:
        Passed through to calculate_setup_levels(). See that
        function's docstring -- "fvg_midpoint" requires
        label_setup_outcomes() to check for an actual fill
        before a setup counts as a trade.
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
        entry_mode=entry_mode,
        fvg_retracement=fvg_retracement,
    )

    # ---------------------------------------------------------
    # 3. Find Structural Target + Actual RR
    # ---------------------------------------------------------
    result = calculate_available_rr(
        result,
        minimum_rr=minimum_rr,
        max_target_atr_multiple=max_target_atr_multiple,
        min_risk_atr_multiple=min_risk_atr_multiple,
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