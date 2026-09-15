from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["open", "high", "low", "close"]


def load_ohlcv_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file and validate its structure.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Rename common column variations
    rename_map = {
        "datetime": "timestamp",
        "date": "timestamp",
        "time": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }

    df = df.rename(columns=rename_map)

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Convert timestamp if available
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True,
        )

        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"])

        df = df.set_index("timestamp")

    # Convert OHLCV columns to numeric
    numeric_columns = [
        col
        for col in ["open", "high", "low", "close", "volume"]
        if col in df.columns
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # Basic OHLC validation
    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    if invalid_ohlc.any():
        raise ValueError(
            f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships."
        )

    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a basic summary of the dataset."""

    print("\n===== DATASET SUMMARY =====")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        print(f"Start: {df.index.min()}")
        print(f"End:   {df.index.max()}")

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())