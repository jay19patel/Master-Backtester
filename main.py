"""Entry point: fetch data, engineer indicators + price-action signals, then
exhaustively combo-backtest every indicator condition crossed with every
price-action signal and report the top combinations by real PnL.
"""

from datetime import datetime, timezone

import pandas as pd
from rich.console import Console
from rich.table import Table

from combo_backtester import ComboBacktester
from data_fetcher import DataFetcher
from indicator_engine import IndicatorEngine
from price_action_engine import PriceActionEngine
from report_exporter import ReportExporter

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

SYMBOL = "ETHUSD"
INTERVAL = "1h"
TOTAL_DAYS = 50

INCLUDE_INDICATORS = True
# Largest rolling window IndicatorEngine uses is 100 bars (EMA_100/SMA_100) -
# pandas_ta returns None instead of a column when there isn't enough history,
# which crashes with a confusing AttributeError deep inside IndicatorEngine.
# This catches it early with an actionable message instead.
MIN_INDICATOR_BARS = 150

BACKTEST_INITIAL_CAPITAL = 1000.0
BACKTEST_RISK_PER_TRADE_PCT = 2.0
BACKTEST_STOP_LOSS_PCT = 1  # 1% stop-loss
BACKTEST_TAKE_PROFIT_PCT = 3  # 3% target -> 1:3 reward:risk
BACKTEST_MAX_HOLD_BARS = 20
BACKTEST_FEE_PCT = 0.05

RUN_COMBO_BACKTEST = True
COMBO_MIN_SIZE = 3
# Safety ceiling, not a target - the Apriori search (see ComboBacktester) tries
# every size exhaustively (1, 1+2, 1+2+3, ...) and stops on its own the moment
# no combo of a size can clear COMBO_MIN_FIRES anymore. This just bounds how
# far it's allowed to go if conditions turn out to be highly correlated.
COMBO_MAX_SIZE = 50
COMBO_MIN_FIRES = 15
COMBO_CONSOLE_TOP_N = 20
COMBO_JSON_TOP_N = 2000  # report.json/dashboard cap - a browser can't reasonably hold huge row counts
COMBO_N_WORKERS = None  # None = cpu_count - 1

RUN_JSON_EXPORT = True
JSON_EXPORT_PATH = "report.json"

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
    df = build_dataset()
    print_report(df)

    # Computed once here and handed to ReportExporter so the JSON export
    # doesn't re-run the (expensive) combo search a second time just to save it.
    precomputed = {}

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
            console_top_n=COMBO_CONSOLE_TOP_N,
            n_workers=COMBO_N_WORKERS,
        )
        precomputed["combo_backtester"] = combo_bt
        precomputed["combo_profitable"] = combo_bt.print_report()

    if RUN_JSON_EXPORT:
        ReportExporter(df, export_config(), precomputed=precomputed).save(JSON_EXPORT_PATH)


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
        "combo_json_top_n": COMBO_JSON_TOP_N,
    }


if __name__ == "__main__":
    main()
