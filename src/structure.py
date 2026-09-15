import pandas as pd


def detect_swings(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
) -> pd.DataFrame:
    """
    Detect confirmed swing highs and swing lows.

    A swing high is a candle whose high is greater than
    the highs of the candles immediately surrounding it.

    A swing is only considered confirmed after `right_bars`
    candles have formed. This is important for avoiding
    look-ahead bias in backtesting.
    """

    if left_bars < 1 or right_bars < 1:
        raise ValueError(
            "left_bars and right_bars must be >= 1"
        )

    required = ["open", "high", "low", "close"]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    result = df.copy()

    result["swing_high"] = False
    result["swing_low"] = False

    result["swing_high_price"] = float("nan")
    result["swing_low_price"] = float("nan")

    result["swing_high_confirmed"] = False
    result["swing_low_confirmed"] = False

    highs = result["high"].to_numpy()
    lows = result["low"].to_numpy()

    total_bars = len(result)

    for pivot in range(
        left_bars,
        total_bars - right_bars,
    ):

        left_highs = highs[
            pivot - left_bars:pivot
        ]

        right_highs = highs[
            pivot + 1:pivot + right_bars + 1
        ]

        left_lows = lows[
            pivot - left_bars:pivot
        ]

        right_lows = lows[
            pivot + 1:pivot + right_bars + 1
        ]

        is_swing_high = (
            highs[pivot] > left_highs.max()
            and highs[pivot] >= right_highs.max()
        )

        is_swing_low = (
            lows[pivot] < left_lows.min()
            and lows[pivot] <= right_lows.min()
        )

        pivot_timestamp = result.index[pivot]

        confirmation_index = (
            pivot + right_bars
        )

        confirmation_timestamp = result.index[
            confirmation_index
        ]

        if is_swing_high:

            result.at[
                pivot_timestamp,
                "swing_high"
            ] = True

            result.at[
                pivot_timestamp,
                "swing_high_price"
            ] = highs[pivot]

            result.at[
                confirmation_timestamp,
                "swing_high_confirmed"
            ] = True

        if is_swing_low:

            result.at[
                pivot_timestamp,
                "swing_low"
            ] = True

            result.at[
                pivot_timestamp,
                "swing_low_price"
            ] = lows[pivot]

            result.at[
                confirmation_timestamp,
                "swing_low_confirmed"
            ] = True

    return result


def detect_market_structure(
    df: pd.DataFrame,
    right_bars: int = 3,
) -> pd.DataFrame:
    """
    Detect basic BOS and MSS events using confirmed swings.

    BOS:
        Price closes beyond the most recently confirmed
        structural swing in the direction of the break.

    MSS:
        A break occurring after the previous structural
        direction was opposite.

    This is a research definition and will be refined
    as the strategy develops.
    """

    result = df.copy()

    result["bos_bullish"] = False
    result["bos_bearish"] = False

    result["mss_bullish"] = False
    result["mss_bearish"] = False

    result["structure"] = "neutral"

    last_swing_high = None
    last_swing_low = None

    current_structure = "neutral"

    for i in range(len(result)):

        timestamp = result.index[i]

        # Update structural levels only when swings
        # have become confirmed.
        if result.iloc[i]["swing_high_confirmed"]:

            pivot_index = i - right_bars

            if pivot_index >= 0:
                pivot_timestamp = result.index[
                    pivot_index
                ]

                last_swing_high = result.loc[
                    pivot_timestamp,
                    "swing_high_price",
                ]

        if result.iloc[i]["swing_low_confirmed"]:

            pivot_index = i - right_bars

            if pivot_index >= 0:
                pivot_timestamp = result.index[
                    pivot_index
                ]

                last_swing_low = result.loc[
                    pivot_timestamp,
                    "swing_low_price",
                ]

        close = result.iloc[i]["close"]

        bullish_break = (
            last_swing_high is not None
            and close > last_swing_high
        )

        bearish_break = (
            last_swing_low is not None
            and close < last_swing_low
        )

        if bullish_break:

            result.at[
                timestamp,
                "bos_bullish"
            ] = True

            if current_structure == "bearish":
                result.at[
                    timestamp,
                    "mss_bullish"
                ] = True

            current_structure = "bullish"

        elif bearish_break:

            result.at[
                timestamp,
                "bos_bearish"
            ] = True

            if current_structure == "bullish":
                result.at[
                    timestamp,
                    "mss_bearish"
                ] = True

            current_structure = "bearish"

        result.at[
            timestamp,
            "structure"
        ] = current_structure

    return result