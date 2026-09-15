import numpy as np
import pandas as pd
import pytest

from src.setups import (
    detect_setup_context,
    calculate_setup_levels,
    detect_setups,
)


def make_base_dataframe():

    return pd.DataFrame(
        {
            "open": [
                100,
                101,
                102,
                103,
                104,
            ],

            "high": [
                101,
                103,
                104,
                105,
                105,
            ],

            "low": [
                99,
                100,
                101,
                102,
                103,
            ],

            "close": [
                100.5,
                102.5,
                103.5,
                104.5,
                104.5,
            ],

            "bullish_sweep": [
                True,
                False,
                False,
                False,
                False,
            ],

            "bearish_sweep": [
                False,
                False,
                False,
                False,
                False,
            ],

            "bullish_displacement": [
                False,
                True,
                False,
                False,
                False,
            ],

            "bearish_displacement": [
                False,
                False,
                False,
                False,
                False,
            ],

            "bullish_fvg": [
                False,
                False,
                True,
                False,
                False,
            ],

            "bearish_fvg": [
                False,
                False,
                False,
                False,
                False,
            ],

            "previous_low": [
                99,
                100,
                101,
                102,
                102,
            ],

            "previous_high": [
                101,
                103,
                104,
                105,
                106,
            ],

            "swing_high_price": [
                np.nan,
                110.0,
                np.nan,
                np.nan,
                np.nan,
            ],

            "swing_low_price": [
                90.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )


def test_bullish_setup_detection():

    df = make_base_dataframe()

    result = detect_setup_context(df)

    assert bool(
        result.loc[2, "bullish_setup"]
    ) is True

    assert result.loc[
        2,
        "setup_direction",
    ] == "bullish"


def test_bearish_setup_not_triggered():

    df = make_base_dataframe()

    result = detect_setup_context(df)

    assert bool(
        result["bearish_setup"].any()
    ) is False


def test_bullish_setup_levels():

    df = make_base_dataframe()

    df["setup_direction"] = pd.Series(
    None,
    index=df.index,
    dtype="object",
)
    df.loc[2, "setup_direction"] = "bullish"

    result = calculate_setup_levels(
        df,
        reward_multiple=2.0,
    )

    assert result.loc[2, "entry"] == 103.5

    assert result.loc[2, "stop_loss"] == 101.0

    assert result.loc[2, "risk"] == 2.5

    # TP is intentionally NOT created here.
    # Structural target is calculated separately.
    assert (
        "take_profit"
        not in result.columns
    )


def test_two_r_filter():

    df = make_base_dataframe()

    result = detect_setups(
        df,
        reward_multiple=2.0,
        sequence_window=3,
    )

    assert "structural_target" in result.columns
    assert "rr_ratio" in result.columns
    assert "valid_2r_setup" in result.columns

    # The old implementation artificially created
    # exactly 2R. The new implementation must use
    # the structural target instead.
    assert result.loc[
        2,
        "take_profit",
    ] == result.loc[
        2,
        "structural_target",
    ]


def test_no_setup_without_all_conditions():

    df = make_base_dataframe()

    # Remove displacement confirmation.
    df.loc[
        1,
        "bullish_displacement",
    ] = False

    result = detect_setup_context(df)

    assert bool(
        result["bullish_setup"].any()
    ) is False


def test_sequence_window():

    df = make_base_dataframe()

    # Candle 0 -> sweep
    # Candle 1 -> displacement
    # Candle 2 -> FVG
    #
    # Total distance = 2 candles.
    #
    # Therefore window=1 cannot complete
    # the sequence.

    result = detect_setup_context(
        df,
        sequence_window=1,
    )

    assert bool(
        result["bullish_setup"].any()
    ) is False


def test_detect_setups_pipeline():

    df = make_base_dataframe()

    result = detect_setups(
        df,
        reward_multiple=2.0,
        sequence_window=3,
    )

    expected_columns = {
        "entry",
        "stop_loss",
        "risk",
        "structural_target",
        "available_reward",
        "available_risk",
        "available_rr",
        "take_profit",
        "reward",
        "rr_ratio",
        "valid_setup",
        "valid_2r_setup",
    }

    assert expected_columns.issubset(
        result.columns
    )


def test_invalid_reward_multiple():

    df = make_base_dataframe()

    with pytest.raises(ValueError):

        calculate_setup_levels(
            df,
            reward_multiple=0,
        )


def test_negative_sl_buffer():

    df = make_base_dataframe()

    with pytest.raises(ValueError):

        calculate_setup_levels(
            df,
            sl_buffer=-1,
        )