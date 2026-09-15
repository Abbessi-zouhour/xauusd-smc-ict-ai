import pandas as pd

from src.liquidity import (
    detect_equal_levels,
    detect_liquidity,
    detect_liquidity_sweeps,
    detect_previous_levels,
)


def create_liquidity_data():

    timestamps = pd.date_range(
        "2026-01-01",
        periods=30,
        freq="h",
        tz="UTC",
    )

    data = {
        "open": [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            104.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
            108.0,
            109.0,
            108.0,
            107.0,
            108.0,
            109.0,
            110.0,
            111.0,
            112.0,
            111.0,
            110.0,
            109.0,
            108.0,
            109.0,
            110.0,
            111.0,
            112.0,
            113.0,
        ],
        "high": [
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
            106.0,
            105.0,
            106.0,
            107.0,
            108.0,
            109.0,
            110.0,
            111.0,
            110.0,
            109.0,
            110.0,
            111.0,
            112.0,
            113.0,
            114.0,
            113.0,
            112.0,
            111.0,
            110.0,
            111.0,
            112.0,
            113.0,
            114.0,
            115.0,
        ],
        "low": [
            99.0,
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            103.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
            108.0,
            107.0,
            106.0,
            107.0,
            108.0,
            109.0,
            110.0,
            111.0,
            110.0,
            109.0,
            108.0,
            107.0,
            108.0,
            109.0,
            110.0,
            111.0,
            112.0,
        ],
        "close": [
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            105.0,
            104.0,
            105.0,
            106.0,
            107.0,
            108.0,
            109.0,
            110.0,
            109.0,
            108.0,
            109.0,
            110.0,
            111.0,
            112.0,
            113.0,
            112.0,
            111.0,
            110.0,
            109.0,
            110.0,
            111.0,
            112.0,
            113.0,
            114.0,
        ],
    }

    return pd.DataFrame(
        data,
        index=timestamps,
    )


def test_previous_levels():

    df = create_liquidity_data()

    result = detect_previous_levels(
        df,
        lookback=5,
    )

    assert "previous_high" in result.columns
    assert "previous_low" in result.columns

    # The first candles cannot have a complete
    # previous lookback window.
    assert result["previous_high"].iloc[0] != result["previous_high"].iloc[0]


def test_equal_levels():

    df = create_liquidity_data()

    result = detect_equal_levels(
        df,
        tolerance=5.0,
        lookback=5,
    )

    assert "equal_high_zone" in result.columns
    assert "equal_low_zone" in result.columns

    assert result["equal_high_zone"].dtype == bool
    assert result["equal_low_zone"].dtype == bool


def test_liquidity_sweeps():

    df = create_liquidity_data()

    result = detect_previous_levels(
        df,
        lookback=5,
    )

    result = detect_liquidity_sweeps(
        result,
        tolerance=0.10,
    )

    assert "bullish_liquidity_sweep" in result.columns
    assert "bearish_liquidity_sweep" in result.columns

    assert result["bullish_liquidity_sweep"].dtype == bool
    assert result["bearish_liquidity_sweep"].dtype == bool


def test_complete_liquidity_pipeline():

    df = create_liquidity_data()

    result = detect_liquidity(
        df,
        lookback=5,
        tolerance=0.10,
    )

    expected_columns = [
        "previous_high",
        "previous_low",
        "equal_high_zone",
        "equal_low_zone",
        "bullish_liquidity_sweep",
        "bearish_liquidity_sweep",
    ]

    for column in expected_columns:
        assert column in result.columns