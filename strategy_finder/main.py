"""Entry point: fetch data, engineer indicators + price-action signals, then
exhaustively combo-backtest every indicator condition crossed with every
price-action signal and report the top combinations by real PnL.
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd
from rich.console import Console
from rich.table import Table

from .combo_backtester import ComboBacktester
from .data_fetcher import DataFetcher
from .indicator_engine import IndicatorEngine
from .price_action_engine import PriceActionEngine
from .report_exporter import ReportExporter

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

SYMBOL = "ETHUSD"
INTERVAL = "15m"
TOTAL_DAYS = 365

INCLUDE_INDICATORS = True
# Largest indicator window is 100 bars (EMA_100/SMA_100); need more history than that.
MIN_INDICATOR_BARS = 150

BACKTEST_INITIAL_CAPITAL = 1000.0
BACKTEST_RISK_PER_TRADE_PCT = 2.0
BACKTEST_STOP_LOSS_PCT = 1  # 1% stop-loss
BACKTEST_TAKE_PROFIT_PCT = 2  # 3% target -> 1:3 reward:risk
BACKTEST_MAX_HOLD_BARS = 20
BACKTEST_FEE_PCT = 0.05

RUN_COMBO_BACKTEST = True
COMBO_MIN_SIZE = 3
# Ceiling on combo size - search stops early on its own once nothing clears COMBO_MIN_FIRES.
COMBO_MAX_SIZE = 100
COMBO_MIN_FIRES = 15
# Multi-candle patterns: every condition also gets a "k candles ago" copy at
# each depth here (e.g. a big mother candle 2 bars back + inside bar 1 bar
# back + breakout now), not just conditions that all fire on the same candle.
COMBO_LAG_DEPTHS = (1, 2, 3)

JSON_EXPORT_PATH = "data/report.json"

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
            max_hold_bars=BACKTEST_MAX_HOLD_BARS,
            fee_pct=BACKTEST_FEE_PCT,
            min_combo_size=COMBO_MIN_SIZE,
            max_combo_size=COMBO_MAX_SIZE,
            min_fires=COMBO_MIN_FIRES,
            lag_depths=COMBO_LAG_DEPTHS,
            n_workers=None,  # every CPU core minus one
            max_raw_candidates_per_level=None,  # no ceiling, examine every candidate at every level
            max_survivors_per_level=None,  # no cap, every combo that clears min_fires carries forward
            max_search_seconds=None,  # no wall-clock limit
        )
        precomputed["combo_backtester"] = combo_bt
        precomputed["combo_profitable"] = combo_bt.print_report()

    output_paths = []
    ReportExporter(df, export_config(), precomputed=precomputed).save(JSON_EXPORT_PATH)
    output_paths.append(JSON_EXPORT_PATH)

    print_run_summary(run_start, df, precomputed, output_paths)


def print_run_summary(run_start, df, precomputed, output_paths):
    """Short end-of-run summary: total time, row/column counts, where the
    output landed. The detailed combo breakdown lives in data/report.json,
    not the console."""
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

    for path in output_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            console.print(f"Saved               : {path} ({size_mb:.1f} MB)")


def export_config():
    """Every setting ReportExporter needs, gathered from this module's constants."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "total_days": TOTAL_DAYS,
        "backtest_initial_capital": BACKTEST_INITIAL_CAPITAL,
        "backtest_risk_per_trade_pct": BACKTEST_RISK_PER_TRADE_PCT,
        "backtest_stop_loss_pct": BACKTEST_STOP_LOSS_PCT,
        "backtest_take_profit_pct": BACKTEST_TAKE_PROFIT_PCT,
        "backtest_max_hold_bars": BACKTEST_MAX_HOLD_BARS,
        "backtest_fee_pct": BACKTEST_FEE_PCT,
        "run_combo_backtest": RUN_COMBO_BACKTEST,
        "combo_min_size": COMBO_MIN_SIZE,
        "combo_max_size": COMBO_MAX_SIZE,
        "combo_min_fires": COMBO_MIN_FIRES,
        "combo_lag_depths": list(COMBO_LAG_DEPTHS),
    }


if __name__ == "__main__":
    main()
