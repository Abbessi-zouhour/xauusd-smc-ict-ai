import numpy as np
import pandas as pd
import pytest

from src.targets import (
    find_structural_targets,
    calculate_available_rr,
    apply_minimum_rr_filter,
    detect_targets,
)


def make_bullish_dataframe(
    target_price=110.0,
):
    """
    Synthetic bullish setup.

    Entry = 100
    Stop  = 95
    Risk  = 5

    Default target = 110
    Reward = 10
    RR = 2R
    """

    return pd.DataFrame(
        {
            "open": [96, 98, 99, 100],
            "high": [98, 100, 101, 102],
            "low": [94, 96, 97, 99],
            "close": [97, 99, 100, 100],

            "entry": [
                np.nan,
                np.nan,
                np.nan,
                100.0,
            ],

            "stop_loss": [
                np.nan,
                np.nan,
                np.nan,
                95.0,
            ],

            "setup_direction": [
                None,
                None,
                None,
                "bullish",
            ],

            "swing_high_price": [
                110.0,
                np.nan,
                np.nan,
                np.nan,
            ],

            "swing_low_price": [
                np.nan,
                90.0,
                92.0,
                np.nan,
            ],
        }
    )


def make_bearish_dataframe(
    target_price=88.0,
):
    """
    Synthetic bearish setup.

    Entry = 100
    Stop  = 105
    Risk  = 5

    Default target = 88
    Reward = 12
    RR = 2.4R
    """

    return pd.DataFrame(
        {
            "open": [104, 102, 101, 100],
            "high": [106, 104, 103, 101],
            "low": [102, 100, 99, 98],
            "close": [103, 101, 100, 100],

            "entry": [
                np.nan,
                np.nan,
                np.nan,
                100.0,
            ],

            "stop_loss": [
                np.nan,
                np.nan,
                np.nan,
                105.0,
            ],

            "setup_direction": [
                None,
                None,
                None,
                "bearish",
            ],

            "swing_high_price": [
                110.0,
                108.0,
                106.0,
                np.nan,
            ],

            "swing_low_price": [
                88.0,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )


def test_bullish_structural_target():
    df = make_bullish_dataframe()

    result = find_structural_targets(df)

    assert result.loc[3, "structural_target"] == 110.0


def test_bearish_structural_target():
    df = make_bearish_dataframe()

    result = find_structural_targets(df)

    assert result.loc[3, "structural_target"] == 88.0


def test_bullish_available_rr():
    df = make_bullish_dataframe()

    result = calculate_available_rr(df)

    # Risk   = 100 - 95 = 5
    # Reward = 110 - 100 = 10
    # RR     = 2R

    assert result.loc[3, "available_risk"] == 5.0
    assert result.loc[3, "available_reward"] == 10.0
    assert result.loc[3, "available_rr"] == 2.0


def test_bearish_available_rr():
    df = make_bearish_dataframe()

    result = calculate_available_rr(df)

    # Risk   = 105 - 100 = 5
    # Reward = 100 - 88 = 12
    # RR     = 2.4R

    assert result.loc[3, "available_risk"] == 5.0
    assert result.loc[3, "available_reward"] == 12.0
    assert result.loc[3, "available_rr"] == 2.4


def test_exactly_2r_is_valid():
    df = make_bullish_dataframe()

    result = calculate_available_rr(df)

    assert result.loc[3, "available_rr"] == 2.0
    assert bool(
        result.loc[3, "valid_2r_setup"]
    ) is True


def test_above_2r_is_valid():
    df = make_bearish_dataframe()

    result = calculate_available_rr(df)

    # Available RR = 2.4R
    # Therefore it passes the 2R minimum.

    assert result.loc[3, "available_rr"] == 2.4
    assert bool(
        result.loc[3, "valid_2r_setup"]
    ) is True


def test_below_2r_is_rejected():
    df = make_bullish_dataframe()

    # Modify the only previous swing high
    # to create a 1.4R setup.
    df.loc[0, "swing_high_price"] = 107.0

    result = calculate_available_rr(df)

    # Risk   = 5
    # Reward = 7
    # RR     = 1.4R

    assert result.loc[3, "available_rr"] == 1.4
    assert bool(
        result.loc[3, "valid_2r_setup"]
    ) is False


def test_minimum_rr_can_be_changed():
    df = make_bearish_dataframe()

    result = calculate_available_rr(
        df,
        minimum_rr=2.5,
    )

    # Available RR = 2.4R
    # Therefore it fails a 2.5R requirement.

    assert result.loc[3, "available_rr"] == 2.4
    assert bool(
        result.loc[3, "valid_2r_setup"]
    ) is False


def test_apply_minimum_rr_filter():
    df = make_bullish_dataframe()

    df["available_rr"] = [
        np.nan,
        np.nan,
        np.nan,
        3.0,
    ]

    result = apply_minimum_rr_filter(
        df,
        minimum_rr=2.0,
    )

    assert bool(
        result.loc[3, "valid_2r_setup"]
    ) is True


def test_invalid_minimum_rr():
    df = make_bullish_dataframe()

    with pytest.raises(ValueError):
        calculate_available_rr(
            df,
            minimum_rr=0,
        )


def test_pipeline():
    df = make_bullish_dataframe()

    result = detect_targets(
        df,
        minimum_rr=2.0,
    )

    expected_columns = {
        "structural_target",
        "available_reward",
        "available_risk",
        "available_rr",
        "valid_2r_setup",
    }

    assert expected_columns.issubset(
        result.columns
    )

    assert result.loc[3, "available_rr"] == 2.0

    assert bool(
        result.loc[3, "valid_2r_setup"]
    ) is True