"""
Outcome labeling for XAUUSD SMC/ICT setups.

This module answers the question that setups.py / targets.py
deliberately leave open:

    Given a detected setup with entry / stop_loss / take_profit,
    did price actually reach the target before the stop
    once we walk forward through the following candles?

Pipeline position:

    detect_setups()  (setups.py)
          |
          v
    entry, stop_loss, take_profit, valid_2r_setup
          |
          v
    label_setup_outcomes()   <-- this module
          |
          v
    outcome, bars_held, mfe_r, mae_r, exit_price
          |
          v
    features.py / model.py (supervised training)

Design principles (matching the rest of the project):
    - Only bars AFTER the setup candle are used to determine
      the outcome of that setup. Nothing here looks backward
      into the future relative to when the setup formed -- it
      looks forward from it, which is what a real trade would do.
    - If both the stop and the target are touched within the
      SAME candle (high/low straddle both levels), the outcome
      is resolved conservatively: the stop is assumed to have
      been hit first, since OHLC data cannot tell us the true
      intracandle path.
    - If neither level is reached within max_holding_bars, the
      setup is labeled as a timeout (outcome = 0) rather than
      forced into a win or a loss.

This module is for historical research and backtesting.
It does not execute trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Validation
# ============================================================

REQUIRED_COLUMNS = {
    "high",
    "low",
    "close",
    "entry",
    "stop_loss",
    "take_profit",
    "setup_direction",
}


def _validate_input(df: pd.DataFrame) -> None:
    """Validate the columns required for outcome labeling."""

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty.")


# ============================================================
# Single-setup forward scan
# ============================================================

def _scan_bullish_outcome(
    highs: np.ndarray,
    lows: np.ndarray,
    setup_pos: int,
    entry: float,
    stop_loss: float,
    take_profit: float,
    risk: float,
    max_holding_bars: int,
) -> tuple[int, int, float, float, float]:
    """
    Walk forward from the bar AFTER the setup candle.

    Returns:
        (outcome, bars_held, mfe_r, mae_r, exit_price)

    outcome:
         1  -> take_profit reached first
        -1  -> stop_loss reached first
         0  -> neither reached within max_holding_bars
    """

    total_bars = len(highs)

    scan_start = setup_pos + 1
    scan_end = min(
        setup_pos + max_holding_bars,
        total_bars - 1,
    )

    mfe_r = 0.0
    mae_r = 0.0

    for i in range(scan_start, scan_end + 1):

        bar_high = highs[i]
        bar_low = lows[i]

        # Track excursions in R multiples, using only
        # information from bars already scanned.
        favorable = (bar_high - entry) / risk
        adverse = (entry - bar_low) / risk

        mfe_r = max(mfe_r, favorable)
        mae_r = max(mae_r, adverse)

        hit_target = bar_high >= take_profit
        hit_stop = bar_low <= stop_loss

        if hit_target and hit_stop:
            # Both levels touched on the same candle.
            # Cannot determine intracandle order from OHLC
            # alone -- resolve conservatively as a loss.
            return -1, i - setup_pos, mfe_r, mae_r, stop_loss

        if hit_stop:
            return -1, i - setup_pos, mfe_r, mae_r, stop_loss

        if hit_target:
            return 1, i - setup_pos, mfe_r, mae_r, take_profit

    return 0, scan_end - setup_pos, mfe_r, mae_r, np.nan


def _scan_bearish_outcome(
    highs: np.ndarray,
    lows: np.ndarray,
    setup_pos: int,
    entry: float,
    stop_loss: float,
    take_profit: float,
    risk: float,
    max_holding_bars: int,
) -> tuple[int, int, float, float, float]:
    """
    Mirror of _scan_bullish_outcome() for bearish setups.
    """

    total_bars = len(highs)

    scan_start = setup_pos + 1
    scan_end = min(
        setup_pos + max_holding_bars,
        total_bars - 1,
    )

    mfe_r = 0.0
    mae_r = 0.0

    for i in range(scan_start, scan_end + 1):

        bar_high = highs[i]
        bar_low = lows[i]

        favorable = (entry - bar_low) / risk
        adverse = (bar_high - entry) / risk

        mfe_r = max(mfe_r, favorable)
        mae_r = max(mae_r, adverse)

        hit_target = bar_low <= take_profit
        hit_stop = bar_high >= stop_loss

        if hit_target and hit_stop:
            # Conservative resolution -- see bullish scan.
            return -1, i - setup_pos, mfe_r, mae_r, stop_loss

        if hit_stop:
            return -1, i - setup_pos, mfe_r, mae_r, stop_loss

        if hit_target:
            return 1, i - setup_pos, mfe_r, mae_r, take_profit

    return 0, scan_end - setup_pos, mfe_r, mae_r, np.nan


# ============================================================
# Full pipeline
# ============================================================

def label_setup_outcomes(
    df: pd.DataFrame,
    max_holding_bars: int = 200,
    only_valid_2r: bool = True,
) -> pd.DataFrame:
    """
    Label every detected setup with its historical outcome.

    Added columns:
        outcome        ->  1 (TP first), -1 (SL first), 0 (timeout)
        bars_held      ->  number of bars until resolution
        mfe_r          ->  max favorable excursion, in R multiples
        mae_r          ->  max adverse excursion, in R multiples
        exit_price     ->  price at which the setup was resolved

    Parameters:
        max_holding_bars:
            Maximum number of forward bars to scan before
            giving up and marking the setup as a timeout.

        only_valid_2r:
            When True (default), only rows where
            valid_2r_setup is True are labeled. Other rows
            are left as NaN / <NA>. When False, every row
            with a complete entry/stop/target is labeled,
            regardless of the 2R filter -- useful for
            inspecting the raw distribution of outcomes
            before filtering.

    Rows without a detected setup (setup_direction is None,
    or entry/stop_loss/take_profit missing) are left unlabeled.
    """

    _validate_input(df)

    if max_holding_bars < 1:
        raise ValueError(
            "max_holding_bars must be >= 1."
        )

    result = df.copy()

    result["outcome"] = pd.array(
        [pd.NA] * len(result),
        dtype="Int64",
    )
    result["bars_held"] = pd.array(
        [pd.NA] * len(result),
        dtype="Int64",
    )
    result["mfe_r"] = np.nan
    result["mae_r"] = np.nan
    result["exit_price"] = np.nan

    highs = result["high"].to_numpy()
    lows = result["low"].to_numpy()

    has_levels = (
        result["entry"].notna()
        & result["stop_loss"].notna()
        & result["take_profit"].notna()
        & result["setup_direction"].notna()
    )

    if only_valid_2r and "valid_2r_setup" in result.columns:
        candidate_mask = has_levels & result["valid_2r_setup"].fillna(False)
    else:
        candidate_mask = has_levels

    candidate_positions = np.flatnonzero(
        candidate_mask.to_numpy()
    )

    for pos in candidate_positions:

        row = result.iloc[pos]

        entry = float(row["entry"])
        stop_loss = float(row["stop_loss"])
        take_profit = float(row["take_profit"])
        direction = row["setup_direction"]

        if direction == "bullish":
            risk = entry - stop_loss
        elif direction == "bearish":
            risk = stop_loss - entry
        else:
            continue

        if risk <= 0:
            continue

        if direction == "bullish":
            outcome, bars_held, mfe_r, mae_r, exit_price = (
                _scan_bullish_outcome(
                    highs,
                    lows,
                    pos,
                    entry,
                    stop_loss,
                    take_profit,
                    risk,
                    max_holding_bars,
                )
            )
        else:
            outcome, bars_held, mfe_r, mae_r, exit_price = (
                _scan_bearish_outcome(
                    highs,
                    lows,
                    pos,
                    entry,
                    stop_loss,
                    take_profit,
                    risk,
                    max_holding_bars,
                )
            )

        result.iat[
            pos, result.columns.get_loc("outcome")
        ] = outcome

        result.iat[
            pos, result.columns.get_loc("bars_held")
        ] = bars_held

        result.iat[
            pos, result.columns.get_loc("mfe_r")
        ] = mfe_r

        result.iat[
            pos, result.columns.get_loc("mae_r")
        ] = mae_r

        result.iat[
            pos, result.columns.get_loc("exit_price")
        ] = exit_price

    return result


# ============================================================
# Summary helper
# ============================================================

def summarize_outcomes(df: pd.DataFrame) -> dict:
    """
    Summarize labeled outcomes for a quick sanity check.

    Returns a plain dict so it can be logged, printed, or
    written to a report without pulling in extra dependencies.
    """

    if "outcome" not in df.columns:
        raise ValueError(
            "Run label_setup_outcomes() first."
        )

    labeled = df[df["outcome"].notna()]

    total = len(labeled)

    if total == 0:
        return {
            "total_labeled": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": np.nan,
            "avg_bars_held": np.nan,
        }

    wins = int((labeled["outcome"] == 1).sum())
    losses = int((labeled["outcome"] == -1).sum())
    timeouts = int((labeled["outcome"] == 0).sum())

    decided = wins + losses

    win_rate = (
        wins / decided if decided > 0 else np.nan
    )

    return {
        "total_labeled": total,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": win_rate,
        "avg_bars_held": float(
            labeled["bars_held"].astype(float).mean()
        ),
    }