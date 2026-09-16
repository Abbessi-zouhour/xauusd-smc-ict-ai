import numpy as np
import pandas as pd
import pytest

from src.labels import (
    label_setup_outcomes,
    summarize_outcomes,
)


def make_bullish_df(outcome="win"):
    """
    Setup candle at index 2.

    Entry = 100
    Stop  = 95   (risk = 5)
    Target = 110 (reward = 10, 2R)

    outcome:
        "win"      -> TP hit at bar 4
        "loss"     -> SL hit at bar 4
        "timeout"  -> neither hit
        "same_bar" -> both hit on bar 4 (conservative loss)
    """

    base = {
        "high":  [101, 102, 101, 103, 106],
        "low":   [99,  100, 99,  98,  104],
        "close": [100, 101, 100, 100, 105],
        "entry":         [np.nan, np.nan, 100.0, np.nan, np.nan],
        "stop_loss":     [np.nan, np.nan, 95.0,  np.nan, np.nan],
        "take_profit":   [np.nan, np.nan, 110.0, np.nan, np.nan],
        "setup_direction": [None, None, "bullish", None, None],
        "valid_2r_setup": [False, False, True, False, False],
    }

    if outcome == "win":
        base["high"][4] = 111
        base["low"][4] = 108
    elif outcome == "loss":
        base["high"][4] = 103
        base["low"][4] = 90
    elif outcome == "timeout":
        base["high"] = [101, 102, 101, 103, 104]
        base["low"] = [99, 100, 99, 98, 102]
    elif outcome == "same_bar":
        base["high"][4] = 111
        base["low"][4] = 90

    return pd.DataFrame(base)


def make_bearish_df(outcome="win"):
    """
    Setup candle at index 2.

    Entry = 100
    Stop  = 105  (risk = 5)
    Target = 90  (reward = 10, 2R)
    """

    base = {
        "high":  [101, 100, 101, 103, 96],
        "low":   [98,  99,  99,  92,  94],
        "close": [100, 99,  100, 95,  95],
        "entry":         [np.nan, np.nan, 100.0, np.nan, np.nan],
        "stop_loss":     [np.nan, np.nan, 105.0, np.nan, np.nan],
        "take_profit":   [np.nan, np.nan, 90.0,  np.nan, np.nan],
        "setup_direction": [None, None, "bearish", None, None],
        "valid_2r_setup": [False, False, True, False, False],
    }

    if outcome == "win":
        base["high"][4] = 96
        base["low"][4] = 89
    elif outcome == "loss":
        base["high"][4] = 106
        base["low"][4] = 97

    return pd.DataFrame(base)


def test_bullish_tp_hit_first():
    df = make_bullish_df("win")

    result = label_setup_outcomes(df, max_holding_bars=10)

    assert result.loc[2, "outcome"] == 1
    assert result.loc[2, "bars_held"] == 2
    assert result.loc[2, "exit_price"] == 110.0


def test_bullish_sl_hit_first():
    df = make_bullish_df("loss")

    result = label_setup_outcomes(df, max_holding_bars=10)

    assert result.loc[2, "outcome"] == -1
    assert result.loc[2, "exit_price"] == 95.0


def test_bullish_timeout():
    df = make_bullish_df("timeout")

    result = label_setup_outcomes(df, max_holding_bars=10)

    assert result.loc[2, "outcome"] == 0
    assert pd.isna(result.loc[2, "exit_price"])


def test_same_bar_resolves_conservatively_as_loss():
    df = make_bullish_df("same_bar")

    result = label_setup_outcomes(df, max_holding_bars=10)

    assert result.loc[2, "outcome"] == -1


def test_bearish_tp_hit_first():
    df = make_bearish_df("win")

    result = label_setup_outcomes(df, max_holding_bars=10)

    assert result.loc[2, "outcome"] == 1
    assert result.loc[2, "exit_price"] == 90.0


def test_bearish_sl_hit_first():
    df = make_bearish_df("loss")

    result = label_setup_outcomes(df, max_holding_bars=10)

    assert result.loc[2, "outcome"] == -1
    assert result.loc[2, "exit_price"] == 105.0


def test_only_valid_2r_filters_out_unfiltered_setups():
    df = make_bullish_df("win")
    df.loc[2, "valid_2r_setup"] = False

    result = label_setup_outcomes(
        df,
        max_holding_bars=10,
        only_valid_2r=True,
    )

    assert pd.isna(result.loc[2, "outcome"])


def test_only_valid_2r_false_labels_anyway():
    df = make_bullish_df("win")
    df.loc[2, "valid_2r_setup"] = False

    result = label_setup_outcomes(
        df,
        max_holding_bars=10,
        only_valid_2r=False,
    )

    assert result.loc[2, "outcome"] == 1


def test_invalid_max_holding_bars():
    df = make_bullish_df("win")

    with pytest.raises(ValueError):
        label_setup_outcomes(df, max_holding_bars=0)


def test_missing_columns_raise():
    df = pd.DataFrame({"high": [1, 2], "low": [1, 2]})

    with pytest.raises(ValueError):
        label_setup_outcomes(df)


def test_summarize_outcomes():
    df = make_bullish_df("win")

    labeled = label_setup_outcomes(df, max_holding_bars=10)
    summary = summarize_outcomes(labeled)

    assert summary["total_labeled"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["win_rate"] == 1.0


def test_summarize_outcomes_no_labels():
    df = make_bullish_df("win")
    df["valid_2r_setup"] = False

    labeled = label_setup_outcomes(df, max_holding_bars=10)
    summary = summarize_outcomes(labeled)

    assert summary["total_labeled"] == 0
    assert np.isnan(summary["win_rate"])