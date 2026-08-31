"""Entry point: fetch data, engineer indicators + price-action signals, then
exhaustively combo-backtest every indicator condition crossed with every
price-action signal and report the top combinations by real PnL.
"""

import time

import pandas as pd
from rich.console import Console
from rich.table import Table

from .combo_backtester import ComboBacktester, build_best_of_table
from .data_fetcher import DataFetcher
from .indicator_engine import IndicatorEngine
from .price_action_engine import PriceActionEngine

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

SYMBOL = "ETHUSD"
INTERVAL = "5m"
TOTAL_DAYS = 365

INCLUDE_INDICATORS = True
# Largest indicator window is 100 bars (EMA_100/SMA_100); need more history than that.
MIN_INDICATOR_BARS = 150

BACKTEST_INITIAL_CAPITAL = 1000.0
BACKTEST_RISK_PER_TRADE_PCT = 2.0
BACKTEST_STOP_LOSS_PCT = 0.5  # Fixed 0.5% Stop Loss
BACKTEST_TAKE_PROFIT_PCT = 1  # Fixed 1% Take Profit Target (1:2 risk:reward)
BACKTEST_SL_TP_MODE = "fixed"  # Fixed %% SL/TP mode (was "atr" - fixed % needs this to take effect)
# 3 days held constant across timeframes: 72 1h candles at INTERVAL="1h", scaled to
# 864 5m candles here (3 * 24 * 60 / 5) - keep this in sync if INTERVAL changes again.
BACKTEST_MAX_HOLD_BARS = 864
BACKTEST_FEE_PCT = 0.05



RUN_COMBO_BACKTEST = True
COMBO_MIN_SIZE = 3
# Ceiling on combo size - search stops early on its own once nothing clears COMBO_MIN_FIRES.
COMBO_MAX_SIZE = 100
COMBO_MIN_FIRES = 15
# Safety cap: at COMBO_MIN_FIRES=15 (~35k candles/year on 15m), the AND filter barely
# prunes anything - size 2 already clears ~99.5% of all possible combos, size 3 ~97%,
# so candidate counts explode combinatorially level over level with no cap (seen live:
# size 3 -> 8.5M survivors, size 4 raw estimate -> 812M). Keeping only the top-scoring
# survivors per level (each combo's own bracket-aware quick score, see
# ComboBacktester._extend_and_filter_batch - not a sum of its members' solo PnL) bounds
# memory/time regardless of how permissive COMBO_MIN_FIRES is - never a random cut, and
# reported via trimmed_levels.
COMBO_MAX_SURVIVORS_PER_LEVEL = 100000
# Multi-candle patterns: every condition also gets a "k candles ago" copy at
# each depth here (e.g. a big mother candle 2 bars back + inside bar 1 bar
# back + breakout now), not just conditions that all fire on the same candle.
COMBO_LAG_DEPTHS = (1, 2, 3)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def build_dataset():
    fetcher = DataFetcher(symbol=SYMBOL, interval=INTERVAL, total_days=TOTAL_DAYS)
    df = fetcher.fetch()

    if df.empty:
        raise RuntimeError("No data fetched - aborting.")

    if INCLUDE_INDICATORS:
        if len(df) < MIN_INDICATOR_BARS:
            raise RuntimeError(
                f"Only {len(df)} candles fetched (TOTAL_DAYS={TOTAL_DAYS}, INTERVAL={INTERVAL!r}), but "
                f"IndicatorEngine needs at least {MIN_INDICATOR_BARS} bars of history (its largest rolling "
                f"window is 100 bars, e.g. EMA_100/SMA_100) - increase TOTAL_DAYS and re-run."
            )
        df = IndicatorEngine(df).build()

    df = PriceActionEngine(df).build()
    return df


def column_groups(df):
    """Split columns into OHLCV / indicator / price-action buckets for reporting."""
    ohlcv_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
    price_action_cols = [c for c in df.columns if c.startswith("sig_")]
    indicator_cols = [c for c in df.columns if c not in ohlcv_cols and c not in price_action_cols]
    return ohlcv_cols, indicator_cols, price_action_cols


def print_report(df):
    console = Console(width=220)
    console.print("\n[bold]DATASET REPORT[/bold]")

    console.print(f"[dim]Symbol / Interval[/dim]      {SYMBOL} / {INTERVAL}")
    console.print(f"[dim]Rows (candles)[/dim]         {len(df)}")
    console.print(f"[dim]Columns[/dim]                {len(df.columns)}")
    console.print(f"[dim]Date range[/dim]             {df.index.min()}  ->  {df.index.max()}")
    console.print(f"[dim]Missing values (total)[/dim] {int(df.isna().sum().sum())}")

    ohlcv_cols, indicator_cols, price_action_cols = column_groups(df)
    column_groups_table = Table(title="Columns", show_lines=False)
    column_groups_table.add_column("Bucket", style="bold")
    column_groups_table.add_column("Count", justify="right")
    for label, cols in [
        ("OHLCV", ohlcv_cols),
        ("Indicator", indicator_cols),
        ("Price Action", price_action_cols),
    ]:
        column_groups_table.add_row(label, str(len(cols)))
    console.print(column_groups_table)


def main():
    run_start = time.monotonic()
    df = build_dataset()
    print_report(df)

    precomputed = {}  # avoids re-running the combo search just to export it

    if RUN_COMBO_BACKTEST:
        combo_bt = ComboBacktester(
            df,
            initial_capital=BACKTEST_INITIAL_CAPITAL,
            risk_per_trade_pct=BACKTEST_RISK_PER_TRADE_PCT,
            stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
            take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
            sl_tp_mode=BACKTEST_SL_TP_MODE,
            max_hold_bars=BACKTEST_MAX_HOLD_BARS,
            fee_pct=BACKTEST_FEE_PCT,
            min_combo_size=COMBO_MIN_SIZE,
            max_combo_size=COMBO_MAX_SIZE,

            min_fires=COMBO_MIN_FIRES,
            lag_depths=COMBO_LAG_DEPTHS,
            n_workers=None,  # every CPU core minus one
            max_raw_candidates_per_level=None,  # no ceiling, examine every candidate at every level
            max_survivors_per_level=COMBO_MAX_SURVIVORS_PER_LEVEL,  # bounds runaway growth, see comment above
            max_search_seconds=None,  # no wall-clock limit
        )
        precomputed["combo_backtester"] = combo_bt
        # Every combo is inserted into data/combo_results.db as it's computed
        # (see ComboBacktester.run/_simulate_tasks) - by the time this call
        # returns, results are already safely on disk regardless of what
        # happens next.
        precomputed["combo_profitable"] = combo_bt.print_report()

    print_run_summary(run_start, df, precomputed)


def print_run_summary(run_start, df, precomputed):
    """Short end-of-run summary: total time, row/column counts, best combo
    breakdown. Every combo is already safe in data/combo_results.db by the
    time this runs (inserted incrementally as each size finishes, see
    ComboBacktester._generate_tasks) - no separate JSON export."""
    console = Console(width=220)
    elapsed = time.monotonic() - run_start

    console.print("\n[bold]RUN SUMMARY[/bold]")
    console.print(f"Total time          : {elapsed:.1f}s")
    console.print(f"Dataset             : {len(df):,} rows x {len(df.columns)} columns ({SYMBOL} {INTERVAL})")

    profitable = precomputed.get("combo_profitable")
    if profitable is not None and not profitable.empty:
        console.print(f"Profitable combos   : {len(profitable):,} (best PnL: ${profitable.iloc[0]['total_pnl']:+.2f})")
        best = profitable.iloc[0]
        console.print(
            f"Best result         : ${BACKTEST_INITIAL_CAPITAL:,.0f} -> ${best['final_equity']:,.2f} "
            f"({best['return_pct']:+.1f}%) using [{best['direction']}] {best['combo']} "
            f"(size {best['size']}, {best['trades']} trades, {best['win_rate_pct']:.1f}% win rate)"
        )

        console.print()
        console.print(build_best_of_table(profitable, COMBO_MIN_FIRES))
        console.print(
            f"[dim]*Balanced Best ranks return% x win_rate% x sqrt(trades) among combos with >={COMBO_MIN_FIRES} "
            "trades - rewards profit and win rate together with enough trades to trust it, instead of chasing a "
            "single extreme.[/dim]"
        )


if __name__ == "__main__":
    main()
