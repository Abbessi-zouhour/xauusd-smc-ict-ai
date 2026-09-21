"""
Rule-based backtest over already-labeled setups.

This does NOT retrain or predict anything -- it takes the setups
run_research.py already detected and labeled (win/loss/timeout, via
labels.label_setup_outcomes) and replays them in chronological order
against a simulated account, so "expectancy per setup" turns into
"what would actually have happened to a balance."

IMPORTANT, read before trusting any number this prints:

    - Every symbol/timeframe here has, at most, a couple dozen
      trades. Drawdown, profit factor, and streak statistics need
      far more than that to be reliable -- a single lucky or unlucky
      trade can swing "max drawdown" by double digits. Treat this as
      a literal replay of exactly what already happened in this
      specific historical sample, not a forecast or a guarantee.
    - Trades are simulated strictly sequentially (each trade's
      outcome is applied to the balance before the next one starts),
      even if two setups' real holding periods would have actually
      overlapped in calendar time. At this trade count that's a
      reasonable simplification, but it means this is NOT a full
      portfolio/margin simulation -- it doesn't model simultaneous
      open positions, margin usage, or correlation between
      concurrently-open trades.
    - Symbols are NEVER pooled into one equity curve here, same
      rule as run_research.py / threshold_sweep.py: XAUUSD and
      XAGUSD are different markets and averaging their trade
      sequences together would misrepresent both.
    - TIMEFRAMES below defaults to just H4 -- the only timeframe
      that has shown a positive, RR-threshold-consistent pattern
      replicated across BOTH symbols in threshold_sweep.py. M15 has
      shown a replicated NEGATIVE pattern across both symbols;
      backtesting it would just be dressing up a losing setup
      definition as an equity curve. Add "h1"/"d1"/"m15" here only
      once/if their own evidence justifies it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------
# What to backtest. Kept in sync in spirit with run_research.py's
# SYMBOLS, but TIMEFRAMES is deliberately narrower -- see the module
# docstring for why.
# ---------------------------------------------------------------------

SYMBOLS = ["xauusd", "xagusd"]
TIMEFRAMES = ["h4"]

INITIAL_BALANCE = 10_000.0

# Fraction of the CURRENT balance risked on each trade (compounding).
# 1% is a conventional conservative default for a discretionary/
# systematic strategy with an unproven edge -- change it here, not
# by hand-editing the simulation loop.
RISK_PCT = 0.01

REQUIRED_COLUMNS = {
    "timestamp",
    "setup_direction",
    "valid_2r_setup",
    "outcome",
    "available_rr",
}


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

def load_trades(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load the tradable subset of one symbol/timeframe's processed
    setups: only setups that (a) cleared the 2R filter and (b)
    actually resolved with a win or loss (outcome is 1 or -1).

    Setups that never got filled, or that timed out without hitting
    either level (outcome 0 or NaN), are excluded -- we don't have a
    clean, non-speculative exit price for them (see
    labels._scan_bullish_outcome / _scan_bearish_outcome: exit_price
    is NaN for a timeout), so including them would mean inventing a
    P&L rather than reporting one.
    """

    path = PROCESSED_DIR / f"{symbol}_{timeframe}_setups.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} -- run `python -m src.run_research` first."
        )

    df = pd.read_csv(path, parse_dates=["timestamp"])

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing expected columns: {sorted(missing)}. "
            "Was this file produced by the current run_research.py?"
        )

    trades = df[
        (df["valid_2r_setup"] == True)  # noqa: E712
        & df["outcome"].isin([1, -1])
    ].copy()

    excluded = int((df["valid_2r_setup"] == True).sum()) - len(trades)  # noqa: E712

    if excluded > 0:
        print(
            f"  ({excluded} setup(s) cleared the 2R filter but never "
            "filled or timed out without resolving -- excluded, not "
            "counted as trades)"
        )

    trades = trades.sort_values("timestamp").reset_index(drop=True)

    return trades


# ---------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------

def simulate_equity(
    trades: pd.DataFrame,
    initial_balance: float,
    risk_pct: float,
) -> pd.DataFrame:
    """
    Replay trades in chronological order against a simulated
    account. Risk per trade is a fixed PERCENTAGE of the balance at
    the time of that trade (compounding, both up and down).

    Realized R per trade:
        win  -> that specific trade's available_rr (the actual
                structural reward/risk ratio, not a flat assumed
                number)
        loss -> exactly -1.0 (the stop is the risk unit by
                construction -- see targets.calculate_available_rr)
    """

    balance = initial_balance
    rows = []

    for _, trade in trades.iterrows():

        risk_amount = balance * risk_pct

        if trade["outcome"] == 1:
            realized_r = float(trade["available_rr"])
        else:
            realized_r = -1.0

        pnl = risk_amount * realized_r
        balance += pnl

        rows.append(
            {
                "timestamp": trade["timestamp"],
                "setup_direction": trade["setup_direction"],
                "available_rr": trade.get("available_rr"),
                "realized_r": realized_r,
                "risk_amount": risk_amount,
                "pnl": pnl,
                "balance": balance,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------

def compute_stats(
    equity: pd.DataFrame,
    initial_balance: float,
) -> dict:

    if equity.empty:
        return {}

    n = len(equity)

    final_balance = equity["balance"].iloc[-1]
    total_return_pct = (final_balance / initial_balance - 1) * 100

    running_max = equity["balance"].cummax()
    drawdown_pct = (equity["balance"] - running_max) / running_max * 100
    max_drawdown_pct = drawdown_pct.min()

    wins = int((equity["realized_r"] > 0).sum())
    losses = int((equity["realized_r"] < 0).sum())
    win_rate = wins / n

    gross_profit = equity.loc[equity["pnl"] > 0, "pnl"].sum()
    gross_loss = -equity.loc[equity["pnl"] < 0, "pnl"].sum()

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    current_sign = 0

    for r in equity["realized_r"]:

        sign = 1 if r > 0 else -1

        if sign == current_sign:
            current_streak += 1
        else:
            current_streak = 1
            current_sign = sign

        if sign == 1:
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)

    return {
        "n_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "final_balance": final_balance,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "profit_factor": profit_factor,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_r": equity["realized_r"].mean(),
    }


# ---------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------

def plot_equity_curve(
    equity: pd.DataFrame,
    symbol: str,
    timeframe: str,
    out_path: Path,
) -> None:

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        equity["timestamp"],
        equity["balance"],
        marker="o",
        markersize=4,
        linewidth=1.5,
    )

    starting_balance = equity["balance"].iloc[0] - equity["pnl"].iloc[0]

    ax.axhline(
        starting_balance,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Starting balance",
    )

    ax.set_title(
        f"{symbol.upper()} {timeframe.upper()} equity curve "
        f"({len(equity)} trades)\n"
        "Small sample -- see backtest.py module docstring "
        "before trusting this shape",
        fontsize=11,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Balance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Per-symbol/timeframe run
# ---------------------------------------------------------------------

def run_backtest_for(symbol: str, timeframe: str) -> dict | None:

    print("=" * 70)
    print(f"{symbol.upper()} {timeframe.upper()} BACKTEST")
    print("=" * 70)

    try:
        trades = load_trades(symbol, timeframe)
    except FileNotFoundError as error:
        print(f"Skipping: {error}")
        print()
        return None

    if trades.empty:
        print(
            "No tradable (2R-filtered, filled, resolved) setups "
            "found -- skipping."
        )
        print()
        return None

    print(f"{len(trades)} tradable setups found.")

    equity = simulate_equity(trades, INITIAL_BALANCE, RISK_PCT)
    stats = compute_stats(equity, INITIAL_BALANCE)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORTS_DIR / f"{symbol}_{timeframe}_equity_curve.csv"
    equity.to_csv(csv_path, index=False)

    chart_path = REPORTS_DIR / f"{symbol}_{timeframe}_equity_curve.png"
    plot_equity_curve(equity, symbol, timeframe, chart_path)

    print(f"Trades:              {stats['n_trades']}")
    print(
        f"Win rate:             {stats['win_rate']*100:.1f}% "
        f"({stats['wins']}W / {stats['losses']}L)"
    )
    print(
        f"Final balance:        ${stats['final_balance']:,.2f} "
        f"(start ${INITIAL_BALANCE:,.2f}, {RISK_PCT*100:.0f}% risk/trade)"
    )
    print(f"Total return:         {stats['total_return_pct']:+.1f}%")
    print(f"Max drawdown:         {stats['max_drawdown_pct']:.1f}%")
    print(f"Profit factor:        {stats['profit_factor']:.2f}")
    print(f"Longest win streak:   {stats['max_win_streak']}")
    print(f"Longest loss streak:  {stats['max_loss_streak']}")
    print(f"Average realized R:   {stats['avg_r']:+.2f}R")
    print(f"Saved equity curve CSV to:   {csv_path}")
    print(f"Saved equity curve chart to: {chart_path}")

    if stats["n_trades"] < 30:
        print(
            f"NOTE: only {stats['n_trades']} trades -- drawdown/streak/"
            "profit-factor numbers above are descriptive of this exact "
            "history, not statistically reliable estimates of what to "
            "expect going forward."
        )

    print()

    return stats


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:

    print()
    print("#" * 70)
    print("# RULE-BASED BACKTEST (no ML -- replays already-labeled setups)")
    print("#" * 70)
    print()

    all_stats: dict[tuple[str, str], dict] = {}

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:

            stats = run_backtest_for(symbol, timeframe)

            if stats:
                all_stats[(symbol, timeframe)] = stats

    if not all_stats:
        print("No backtests produced any results.")
        return

    print("=" * 70)
    print("SUMMARY (each row is its own independent equity curve --")
    print("never combine these into one number; see module docstring)")
    print("=" * 70)
    print(
        f"{'Symbol':<8} {'TF':<4} {'Trades':>6} {'WinRate':>8} "
        f"{'Return':>9} {'MaxDD':>8} {'PF':>6} {'AvgR':>7}"
    )
    print("-" * 70)

    for (symbol, timeframe), stats in all_stats.items():

        print(
            f"{symbol.upper():<8} {timeframe.upper():<4} "
            f"{stats['n_trades']:>6} "
            f"{stats['win_rate']*100:>7.1f}% "
            f"{stats['total_return_pct']:>+8.1f}% "
            f"{stats['max_drawdown_pct']:>7.1f}% "
            f"{stats['profit_factor']:>6.2f} "
            f"{stats['avg_r']:>+6.2f}R"
        )

    print()
    print("Backtest completed. No live orders were executed.")


if __name__ == "__main__":
    main()