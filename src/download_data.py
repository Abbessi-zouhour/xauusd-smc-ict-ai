import yfinance as yf
from pathlib import Path


def download_xauusd(
    period: str = "2y",
    interval: str = "1h",
) -> None:
    """
    Download historical gold data for research/backtesting.

    Yahoo Finance ticker:
        GC=F = Gold Futures

    Note:
        This is gold futures data, not broker-specific XAUUSD spot data.
    """

    print(f"Downloading gold data...")
    print(f"Period: {period}")
    print(f"Interval: {interval}")

    data = yf.download(
        "GC=F",
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=True,
    )

    if data.empty:
        raise RuntimeError("No data was downloaded.")

    # Handle MultiIndex columns returned by yfinance
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Normalize column names
    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    # Rename Yahoo columns
    data = data.rename(
        columns={
            "datetime": "timestamp",
            "date": "timestamp",
        }
    )

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(data.columns)}"
        )

    data = data[required]

    # Remove invalid rows
    data = data.dropna()

    # Sort chronologically
    data = data.sort_values("timestamp")

    # Save
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "xauusd_gold_futures_1h.csv"

    data.to_csv(output_file, index=False)

    print("\nDownload successful!")
    print(f"Rows: {len(data):,}")
    print(f"Start: {data['timestamp'].min()}")
    print(f"End:   {data['timestamp'].max()}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    download_xauusd()