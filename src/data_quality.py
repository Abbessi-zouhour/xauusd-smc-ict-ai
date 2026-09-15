from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")

TIMEFRAMES = {
    "M15": "xauusd_m15.csv",
    "H1": "xauusd_h1.csv",
    "H4": "xauusd_h4.csv",
    "D1": "xauusd_d1.csv",
}


def validate_dataframe(df: pd.DataFrame, timeframe: str) -> dict:
    """Validate one XAUUSD dataset."""

    result = {
        "timeframe": timeframe,
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "duplicates": int(df.index.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "invalid_ohlc": 0,
        "monotonic": df.index.is_monotonic_increasing,
    }

    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    result["invalid_ohlc"] = int(invalid_ohlc.sum())

    return result


def load_and_validate(timeframe: str, filename: str) -> dict:
    """Load and validate one timeframe."""

    file_path = DATA_DIR / filename

    if not file_path.exists():
        return {
            "timeframe": timeframe,
            "error": f"File not found: {file_path}",
        }

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp")

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return validate_dataframe(df, timeframe)


def main():
    print("=" * 70)
    print("XAUUSD EXNESS DATA QUALITY REPORT")
    print("=" * 70)

    results = []

    for timeframe, filename in TIMEFRAMES.items():
        result = load_and_validate(
            timeframe,
            filename,
        )

        results.append(result)

        print(f"\n[{timeframe}]")

        for key, value in result.items():
            print(f"{key}: {value}")

    report = pd.DataFrame(results)

    Path("reports").mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = Path("reports/data_quality.csv")

    report.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print(f"Report saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()