from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


# ---------------------------------------------------------------------
# History start dates.
#
# These were previously quite conservative (M15 back to 2024, H1 to
# 2022, H4/D1 to 2018) -- that's a large part of why the RR-filtered
# sample sizes are so small. mt5.copy_rates_range() just returns
# whatever the broker actually has and silently gives you less if you
# ask for more than exists, so there's no harm in requesting further
# back than real history goes. Push these as far back as you want;
# most brokers keep D1/H4 history for a couple decades and M15/H1 for
# several years, but it varies by broker -- check what you actually
# get back after running this (see the "Start:"/"End:" printout).
# ---------------------------------------------------------------------

TIMEFRAMES = {
    "m15": {
        "mt5": mt5.TIMEFRAME_M15,
        "start": datetime(2015, 1, 1, tzinfo=timezone.utc),
    },
    "h1": {
        "mt5": mt5.TIMEFRAME_H1,
        "start": datetime(2010, 1, 1, tzinfo=timezone.utc),
    },
    "h4": {
        "mt5": mt5.TIMEFRAME_H4,
        "start": datetime(2000, 1, 1, tzinfo=timezone.utc),
    },
    "d1": {
        "mt5": mt5.TIMEFRAME_D1,
        "start": datetime(2000, 1, 1, tzinfo=timezone.utc),
    },
}


def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError(
            f"MT5 initialization failed: {mt5.last_error()}"
        )

    print("MT5 initialized successfully.")


def find_xauusd_symbol():
    candidates = [
        "XAUUSD",
        "XAUUSDm",
        "XAUUSD.",
        "XAUUSDc",
        "XAUUSD.a",
    ]

    for symbol in candidates:
        info = mt5.symbol_info(symbol)

        if info is not None:
            if not info.visible:
                mt5.symbol_select(symbol, True)

            print(f"Using symbol: {symbol}")
            return symbol

    symbols = mt5.symbols_get("*XAUUSD*")

    if symbols:
        for info in symbols:
            if info.visible:
                print(f"Using discovered symbol: {info.name}")
                return info.name

        symbol = symbols[0].name
        mt5.symbol_select(symbol, True)

        print(f"Using discovered symbol: {symbol}")

        return symbol

    raise RuntimeError(
        "Could not find an XAUUSD symbol in MT5."
    )


def download_history(
    symbol,
    timeframe,
    start,
    end,
):
    print(
        f"Downloading {symbol} "
        f"from {start} to {end}..."
    )

    rates = mt5.copy_rates_range(
        symbol,
        timeframe,
        start,
        end,
    )

    if rates is None:
        raise RuntimeError(
            f"MT5 download failed: {mt5.last_error()}"
        )

    if len(rates) == 0:
        raise RuntimeError(
            "MT5 returned no historical data."
        )

    df = pd.DataFrame(rates)

    df["timestamp"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True,
    )

    df = df.drop(columns=["time"])

    df = df.rename(
        columns={
            "tick_volume": "volume",
        }
    )

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df = df[columns]

    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")

    return df


def save_data(df, timeframe):
    output_dir = Path("data/raw")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir /
        f"xauusd_{timeframe}.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved {len(df):,} candles "
        f"to {output_file}"
    )


def main():
    initialize_mt5()

    try:
        symbol = find_xauusd_symbol()

        end = datetime.now(timezone.utc)

        for timeframe_name, settings in TIMEFRAMES.items():

            print("\n" + "=" * 70)
            print(
                f"{timeframe_name.upper()} HISTORY"
            )
            print("=" * 70)

            df = download_history(
                symbol=symbol,
                timeframe=settings["mt5"],
                start=settings["start"],
                end=end,
            )

            save_data(
                df,
                timeframe_name,
            )

            print(
                f"Start: {df['timestamp'].min()}"
            )

            print(
                f"End:   {df['timestamp'].max()}"
            )

    finally:
        mt5.shutdown()
        print("\nMT5 connection closed.")


if __name__ == "__main__":
    main()