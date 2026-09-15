import pandas as pd

from src.structure import (
    detect_market_structure,
    detect_swings,
)


def create_test_data():

    timestamps = pd.date_range(
        "2026-01-01",
        periods=30,
        freq="h",
        tz="UTC",
    )

    data = {
        "open": [
            100, 101, 102, 103, 104,
            105, 106, 107, 108, 109,
            108, 107, 106, 105, 104,
            105, 106, 107, 108, 109,
            110, 111, 112, 113, 114,
            115, 116, 117, 118, 119,
        ],

        "high": [
            102, 103, 104, 105, 106,
            107, 108, 109, 110, 111,
            110, 109, 108, 107, 106,
            107, 108, 109, 110, 111,
            112, 113, 114, 115, 116,
            117, 118, 119, 120, 121,
        ],

        "low": [
            99, 100, 101, 102, 103,
            104, 105, 106, 107, 108,
            107, 106, 105, 104, 103,
            104, 105, 106, 107, 108,
            109, 110, 111, 112, 113,
            114, 115, 116, 117, 118,
        ],

        "close": [
            101, 102, 103, 104, 105,
            106, 107, 108, 109, 110,
            109, 108, 107, 106, 105,
            106, 107, 108, 109, 110,
            111, 112, 113, 114, 115,
            116, 117, 118, 119, 120,
        ],
    }

    return pd.DataFrame(
        data,
        index=timestamps,
    )


def test_swings():

    df = create_test_data()

    result = detect_swings(
        df,
        left_bars=3,
        right_bars=3,
    )

    assert "swing_high" in result.columns
    assert "swing_low" in result.columns

    assert result["swing_high"].dtype == bool
    assert result["swing_low"].dtype == bool


def test_market_structure():

    df = create_test_data()

    df = detect_swings(
        df,
        left_bars=3,
        right_bars=3,
    )

    result = detect_market_structure(
        df,
        right_bars=3,
    )

    assert "bos_bullish" in result.columns
    assert "bos_bearish" in result.columns

    assert "mss_bullish" in result.columns
    assert "mss_bearish" in result.columns

    assert "structure" in result.columns