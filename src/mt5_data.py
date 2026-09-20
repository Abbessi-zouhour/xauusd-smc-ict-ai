import time
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


# ---------------------------------------------------------------------
# Symbols to download.
#
# Each entry is a short key used for filenames (data/raw/{key}_{tf}.csv)
# plus a list of likely broker-side ticker spellings to try, in order,
# and a wildcard fallback search if none of those match exactly --
# brokers append suffixes like "m", ".", "c", ".a" inconsistently
# (see how XAUUSDm showed up on this account).
#
# Add more symbols here to pull more data. Keep in mind: a second
# symbol only helps your sample size if you analyze it on its own
# first (see run_research.py / threshold_sweep.py's SYMBOLS list) --
# pooling a different market's setups into gold's stats just blurs
# two different things together.
# ---------------------------------------------------------------------

SYMBOLS = {
    "xauusd": {
        "candidates": [
            "XAUUSD",
            "XAUUSDm",
            "XAUUSD.",
            "XAUUSDc",
            "XAUUSD.a",
        ],
        "search_pattern": "*XAUUSD*",
    },
    "xagusd": {
        "candidates": [
            "XAGUSD",
            "XAGUSDm",
            "XAGUSD.",
            "XAGUSDc",
            "XAGUSD.a",
        ],
        "search_pattern": "*XAGUSD*",
    },
}


def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError(
            f"MT5 initialization failed: {mt5.last_error()}"
        )

    print("MT5 initialized successfully.")


def find_symbol(symbol_key: str) -> str:
    """
    Resolve a SYMBOLS key (e.g. "xauusd") to the actual ticker this
    broker uses (e.g. "XAUUSDm"), trying the known candidate
    spellings first and falling back to a wildcard search.
    """

    settings = SYMBOLS.get(symbol_key)

    if settings is None:
        raise ValueError(
            f"Unknown symbol key: {symbol_key!r}. "
            f"Known keys: {sorted(SYMBOLS)}"
        )

    for symbol in settings["candidates"]:
        info = mt5.symbol_info(symbol)

        if info is not None:
            if not info.visible:
                mt5.symbol_select(symbol, True)

            print(f"Using symbol: {symbol}")
            return symbol

    symbols = mt5.symbols_get(settings["search_pattern"])

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
        f"Could not find a {symbol_key.upper()} symbol in MT5. "
        "Check it's visible in Market Watch, or that this broker "
        "offers it at all."
    )


# IPC-related error codes MT5 returns when the connection to the
# terminal drops or the terminal is still busy from a previous
# heavy request -- these are worth retrying after a short pause.
# -2 "Invalid params" is NOT retried here: that means the terminal
# genuinely doesn't have the requested history cached, and retrying
# won't change that (see download_history's fallback for how that
# case is already handled).
_RETRYABLE_ERROR_CODES = {-10001, -10002, -10003, -1}

_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 5


def download_history(
    symbol,
    timeframe,
    start,
    end,
):
    last_error = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):

        try:
            return _download_history_once(
                symbol, timeframe, start, end,
            )

        except RuntimeError as error:
            error_code = mt5.last_error()[0]

            if (
                error_code not in _RETRYABLE_ERROR_CODES
                or attempt == _MAX_ATTEMPTS
            ):
                raise

            last_error = error

            print(
                f"  Attempt {attempt}/{_MAX_ATTEMPTS} failed "
                f"({mt5.last_error()}) -- likely a dropped/busy "
                f"terminal connection, not a data problem. "
                f"Retrying in {_RETRY_DELAY_SECONDS}s..."
            )

            time.sleep(_RETRY_DELAY_SECONDS)

    # Unreachable given the raise above, but keeps type checkers
    # and linters happy.
    raise last_error


def _download_history_once(
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

    if rates is None or len(rates) == 0:

        # copy_rates_range only reads what the terminal has
        # already cached locally for that date range -- it does
        # NOT force a fresh pull from the broker for history it
        # hasn't loaded yet (that's what "Invalid params" / a
        # None result usually means here, not a bad request).
        #
        # copy_rates_from_pos doesn't need a start date: it asks
        # for the most recent `count` bars counting backward from
        # the latest one, which is a request the terminal can
        # always attempt to satisfy from whatever it actually has
        # (or can fetch), and won't fail just because `start` is
        # further back than what's cached. We take whatever comes
        # back and clip it to [start, end] afterward.
        print(
            "  copy_rates_range returned nothing "
            f"({mt5.last_error()}); "
            "falling back to copy_rates_from_pos "
            "(most recent bars available)..."
        )

        rates = mt5.copy_rates_from_pos(
            symbol,
            timeframe,
            0,
            200_000,
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

    # Clip to the requested window in case the fallback pulled
    # bars outside [start, end].
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    df = df[
        (df["timestamp"] >= start_ts)
        & (df["timestamp"] <= end_ts)
    ]

    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")

    if len(df) == 0:
        raise RuntimeError(
            "MT5 returned no historical data "
            "inside the requested window."
        )

    return df


def save_data(df, symbol_key, timeframe):
    output_dir = Path("data/raw")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir /
        f"{symbol_key}_{timeframe}.csv"
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

    results = {}

    try:
        end = datetime.now(timezone.utc)

        for i, symbol_key in enumerate(SYMBOLS):

            if i > 0:
                # Give the terminal a moment to recover after the
                # previous symbol's (potentially large) history
                # pulls -- this is the likely cause if you saw
                # "IPC send failed" on a symbol immediately after
                # a big one succeeded.
                print(
                    f"\nPausing {_RETRY_DELAY_SECONDS}s before "
                    "the next symbol..."
                )
                time.sleep(_RETRY_DELAY_SECONDS)

            print("\n" + "#" * 70)
            print(f"# {symbol_key.upper()}")
            print("#" * 70)

            try:
                symbol = find_symbol(symbol_key)
            except RuntimeError as error:
                print(f"FAILED: {error}\nSkipping {symbol_key.upper()} entirely.")
                results[symbol_key] = {
                    tf: None for tf in TIMEFRAMES
                }
                continue

            results[symbol_key] = {}

            for timeframe_name, settings in TIMEFRAMES.items():

                print("\n" + "=" * 70)
                print(
                    f"{symbol_key.upper()} {timeframe_name.upper()} HISTORY"
                )
                print("=" * 70)

                try:
                    df = download_history(
                        symbol=symbol,
                        timeframe=settings["mt5"],
                        start=settings["start"],
                        end=end,
                    )
                except RuntimeError as error:
                    print(
                        f"FAILED: {error}\n"
                        f"Skipping {symbol_key.upper()} "
                        f"{timeframe_name.upper()} "
                        "-- see the step-by-step in the chat for "
                        "loading more history into the MT5 terminal, "
                        "then re-run this script. Other symbols/"
                        "timeframes will still be attempted."
                    )
                    results[symbol_key][timeframe_name] = None
                    continue

                save_data(
                    df,
                    symbol_key,
                    timeframe_name,
                )

                print(
                    f"Start: {df['timestamp'].min()}"
                )

                print(
                    f"End:   {df['timestamp'].max()}"
                )

                results[symbol_key][timeframe_name] = (
                    df["timestamp"].min(),
                    df["timestamp"].max(),
                    len(df),
                )

    finally:
        mt5.shutdown()
        print("\nMT5 connection closed.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for symbol_key in SYMBOLS:

        print(f"\n{symbol_key.upper()}:")

        for timeframe_name in TIMEFRAMES:

            result = results.get(symbol_key, {}).get(timeframe_name)

            if result is None:
                print(f"  {timeframe_name.upper():>4}: FAILED (see above)")
                continue

            start, end_ts, count = result

            print(
                f"  {timeframe_name.upper():>4}: {count:>8,} candles  "
                f"{start} -> {end_ts}"
            )


if __name__ == "__main__":
    main()