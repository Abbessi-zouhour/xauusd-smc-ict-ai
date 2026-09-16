import numpy as np
import pandas as pd

from src.targets import (
    find_structural_targets,
    calculate_available_rr,
)


def make_bullish_two_targets():
    """
    Entry = 100, ATR at setup candle = 2.0.

    Nearby swing high: 103  (distance = 3  -> 1.5x ATR)
    Far swing high:    130  (distance = 30 -> 15x ATR)

    Nearest unfiltered target is still 103 (it's the minimum
    candidate above entry), so this dataframe alone doesn't
    distinguish capped vs uncapped behavior -- see the second
    fixture below for that.
    """

    return pd.DataFrame(
        {
            "open": [96, 98, 99, 100],
            "high": [98, 100, 101, 102],
            "low": [94, 96, 97, 99],
            "close": [97, 99, 100, 100],
            "entry": [np.nan, np.nan, np.nan, 100.0],
            "stop_loss": [np.nan, np.nan, np.nan, 95.0],
            "setup_direction": [None, None, None, "bullish"],
            "swing_high_price": [130.0, 103.0, np.nan, np.nan],
            "swing_low_price": [np.nan, np.nan, np.nan, np.nan],
            "atr": [np.nan, np.nan, np.nan, 2.0],
        }
    )


def make_bullish_only_far_target():
    """
    Entry = 100, ATR at setup candle = 2.0.

    Only candidate swing high above entry is 130
    (distance = 30 -> 15x ATR), well beyond an 8x cap.
    """

    return pd.DataFrame(
        {
            "open": [96, 98, 99, 100],
            "high": [98, 100, 101, 102],
            "low": [94, 96, 97, 99],
            "close": [97, 99, 100, 100],
            "entry": [np.nan, np.nan, np.nan, 100.0],
            "stop_loss": [np.nan, np.nan, np.nan, 95.0],
            "setup_direction": [None, None, None, "bullish"],
            "swing_high_price": [130.0, np.nan, np.nan, np.nan],
            "swing_low_price": [np.nan, np.nan, np.nan, np.nan],
            "atr": [np.nan, np.nan, np.nan, 2.0],
        }
    )


def test_uncapped_behavior_unchanged_by_default():
    df = make_bullish_only_far_target()

    result = find_structural_targets(df)

    assert result.loc[3, "structural_target"] == 130.0


def test_cap_excludes_implausibly_distant_target():
    df = make_bullish_only_far_target()

    result = find_structural_targets(
        df,
        max_target_atr_multiple=8.0,
    )

    assert pd.isna(result.loc[3, "structural_target"])


def test_cap_still_finds_nearby_target():
    df = make_bullish_two_targets()

    result = find_structural_targets(
        df,
        max_target_atr_multiple=8.0,
    )

    # 103 is within 8x ATR (16.0), 130 is not.
    # Nearest-within-cap should still be 103.
    assert result.loc[3, "structural_target"] == 103.0


def test_cap_with_missing_atr_is_not_applied():
    df = make_bullish_only_far_target()
    df["atr"] = np.nan

    result = find_structural_targets(
        df,
        max_target_atr_multiple=8.0,
    )

    # No ATR available -> cap cannot be evaluated -> unbounded
    # behavior for that row, not silently excluded.
    assert result.loc[3, "structural_target"] == 130.0


def test_cap_flows_through_calculate_available_rr():
    df = make_bullish_only_far_target()

    result = calculate_available_rr(
        df,
        max_target_atr_multiple=8.0,
    )

    assert pd.isna(result.loc[3, "available_rr"])
    assert bool(result.loc[3, "valid_2r_setup"]) is False