import pandas as pd

from src.ict import (
    detect_fvg,
    detect_displacement,
    detect_fvg_with_displacement,
)


def test_bullish_fvg():
    df = pd.DataFrame(
        {
            "open": [100, 102, 105],
            "high": [101, 103, 110],
            "low": [99, 102, 105],
            "close": [100.5, 102.5, 109],
        }
    )

    result = detect_fvg(df)

    assert bool(result.loc[2, "bullish_fvg"]) is True
    assert result.loc[2, "fvg_type"] == "bullish"
    assert result.loc[2, "fvg_bottom"] == 101
    assert result.loc[2, "fvg_top"] == 105


def test_bearish_fvg():
    df = pd.DataFrame(
        {
            "open": [105, 103, 100],
            "high": [106, 104, 101],
            "low": [104, 102, 95],
            "close": [104.5, 102.5, 96],
        }
    )

    result = detect_fvg(df)

    assert bool(result.loc[2, "bearish_fvg"]) is True
    assert result.loc[2, "fvg_type"] == "bearish"
    assert result.loc[2, "fvg_bottom"] == 101
    assert result.loc[2, "fvg_top"] == 104


def test_no_fvg_when_candles_overlap():
    df = pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [103, 104, 105],
            "low": [99, 100, 101],
            "close": [102, 103, 104],
        }
    )

    result = detect_fvg(df)

    assert not result["bullish_fvg"].any()
    assert not result["bearish_fvg"].any()


def test_displacement_columns_exist():
    df = pd.DataFrame(
        {
            "open": [100] * 20,
            "high": [101] * 20,
            "low": [99] * 20,
            "close": [100.5] * 20,
        }
    )

    result = detect_displacement(df)

    assert "bullish_displacement" in result.columns
    assert "bearish_displacement" in result.columns
    assert "atr" in result.columns
    assert "body_ratio" in result.columns


def test_fvg_with_displacement_columns():
    df = pd.DataFrame(
        {
            "open": [100, 102, 105],
            "high": [101, 103, 110],
            "low": [99, 102, 105],
            "close": [100.5, 102.5, 109],
        }
    )

    result = detect_fvg_with_displacement(df)

    assert "bullish_fvg" in result.columns
    assert "bearish_fvg" in result.columns
    assert "bullish_displacement" in result.columns
    assert "bearish_displacement" in result.columns
    assert "displacement_fvg" in result.columns