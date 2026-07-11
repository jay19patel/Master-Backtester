"""Entry point: fetch data, engineer indicators, add oracle labels, and report stats.

Pipeline order (also the console report order):
    1. Indicator relevance   - which technical indicators connect to the oracle signal
    2. Price Action relevance - same methodology/schema, applied to sig_* signals instead
    3. Backtest               - each sig_* signal traded alone, real PnL
    4. Combo Backtest         - combinations of indicators + sig_* signals, real PnL
    5. Oracle Ceiling         - the oracle's own label backtested (a theoretical benchmark)
"""

from datetime import datetime, timezone

import pandas as pd
from rich.console import Console
from rich.table import Table

from backtester import Backtester
from combo_backtester import ComboBacktester
from data_fetcher import DataFetcher
from indicator_engine import IndicatorEngine
from oracle_backtester import OracleBacktester
from oracle_labeler import OracleLabeler
from price_action_engine import PriceActionEngine
from relevance_analyzer import RelevanceAnalyzer
from report_exporter import ReportExporter

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

SYMBOL = "BTCUSD"
INTERVAL = "1h"
TOTAL_DAYS = 365
ORACLE_LOOKAHEAD = 20
ORACLE_MIN_REWARD_RISK_RATIO = 2.0  # 1:2 minimum - the winning side must be at least double the losing side

INCLUDE_INDICATORS = True
RUN_RELEVANCE_ANALYSIS = True

RUN_BACKTEST = True
BACKTEST_INITIAL_CAPITAL = 100.0
BACKTEST_RISK_PER_TRADE_PCT = 2.0
BACKTEST_STOP_LOSS_PCT = 1  # 0.5% stop-loss
BACKTEST_TAKE_PROFIT_PCT = 2  # 1% target -> 1:2 reward:risk
BACKTEST_MAX_HOLD_BARS = 20
BACKTEST_FEE_PCT = 0.05

RUN_COMBO_BACKTEST = True
COMBO_MIN_SIZE = 1
COMBO_MAX_SIZE = 2  # pool covers ALL indicators + ALL sig_* signals; capped at pairs to keep runtime sane
COMBO_MIN_FIRES = 15
COMBO_CONSOLE_TOP_N = 10

RUN_ORACLE_BACKTEST = True
# Oracle Ceiling internally compares standalone vs a portfolio-risk-managed run
# of the oracle's own label - these are that risk-management setup's parameters.
PORTFOLIO_MAX_CONCURRENT_TRADES = 3
PORTFOLIO_RISK_CAP_PCT = 6.0
PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT = 10.0
PORTFOLIO_DRAWDOWN_RECOVERY_PCT = 5.0
PORTFOLIO_THROTTLED_RISK_PCT = 1.0

RUN_JSON_EXPORT = True
JSON_EXPORT_PATH = "report.json"

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def build_dataset():
    fetcher = DataFetcher(symbol=SYMBOL, interval=INTERVAL, total_days=TOTAL_DAYS)
    df = fetcher.fetch()

    if df.empty:
        raise RuntimeError("No data fetched - aborting.")

    if INCLUDE_INDICATORS:
        df = IndicatorEngine(df).build()

    df = PriceActionEngine(df).build()

    df = OracleLabeler(
        df, lookahead=ORACLE_LOOKAHEAD, min_reward_risk_ratio=ORACLE_MIN_REWARD_RISK_RATIO
    ).label()
    return df


def column_groups(df):
    """Split columns into OHLCV / indicator / price-action / oracle buckets for reporting."""
    oracle_cols = [c for c in df.columns if c.startswith("oracle_")]
    ohlcv_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
    price_action_cols = [c for c in df.columns if c.startswith("sig_")]
    indicator_cols = [
        c for c in df.columns if c not in oracle_cols and c not in ohlcv_cols and c not in price_action_cols
    ]
    return ohlcv_cols, indicator_cols, price_action_cols, oracle_cols


def print_report(df):
    console = Console(width=220)
    console.print("\n[bold]DATASET REPORT[/bold]")

    console.print(f"[dim]Symbol / Interval[/dim]      {SYMBOL} / {INTERVAL}")
    console.print(f"[dim]Rows (candles)[/dim]         {len(df)}")
    console.print(f"[dim]Columns[/dim]                {len(df.columns)}")
    console.print(f"[dim]Date range[/dim]             {df.index.min()}  ->  {df.index.max()}")
    console.print(f"[dim]Missing values (total)[/dim] {int(df.isna().sum().sum())}")

    if "oracle_signal" in df.columns:
        console.print(f"\n[bold]Oracle signal[/bold] (min reward:risk = 1:{ORACLE_MIN_REWARD_RISK_RATIO:.0f})")
        counts = df["oracle_signal"].value_counts(dropna=False).sort_index()
        signal_table = Table(show_lines=False)
        signal_table.add_column("Signal", style="bold")
        signal_table.add_column("Count", justify="right")
        signal_table.add_column("Percent", justify="right")
        for value, count in counts.items():
            name = value if pd.notna(value) else "UNKNOWN (tail, no future data yet)"
            pct = count / len(df) * 100
            signal_table.add_row(name, str(int(count)), f"{pct:.1f}%")
        console.print(signal_table)

    ohlcv_cols, indicator_cols, price_action_cols, oracle_cols = column_groups(df)
    column_groups_table = Table(title="Columns", show_lines=False)
    column_groups_table.add_column("Bucket", style="bold")
    column_groups_table.add_column("Count", justify="right")
    for label, cols in [
        ("OHLCV", ohlcv_cols),
        ("Oracle", oracle_cols),
        ("Indicator", indicator_cols),
        ("Price Action", price_action_cols),
    ]:
        column_groups_table.add_row(label, str(len(cols)))
    console.print(column_groups_table)


def main():
    df = build_dataset()
    print_report(df)

    if RUN_RELEVANCE_ANALYSIS:
        _, indicator_cols, price_action_cols, _ = column_groups(df)
        RelevanceAnalyzer(df, columns=indicator_cols, label="Indicator").print_report()
        RelevanceAnalyzer(df, columns=price_action_cols, label="Price Action").print_report()

    if RUN_BACKTEST:
        Backtester(
            df,
            initial_capital=BACKTEST_INITIAL_CAPITAL,
            risk_per_trade_pct=BACKTEST_RISK_PER_TRADE_PCT,
            stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
            take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
            max_hold_bars=BACKTEST_MAX_HOLD_BARS,
            fee_pct=BACKTEST_FEE_PCT,
        ).print_report()

    if RUN_COMBO_BACKTEST:
        ComboBacktester(
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
        ).print_report()

    if RUN_ORACLE_BACKTEST:
        OracleBacktester(
            df,
            initial_capital=BACKTEST_INITIAL_CAPITAL,
            risk_per_trade_pct=BACKTEST_RISK_PER_TRADE_PCT,
            stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
            take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
            max_hold_bars=BACKTEST_MAX_HOLD_BARS,
            fee_pct=BACKTEST_FEE_PCT,
            portfolio_kwargs=dict(
                max_concurrent_trades=PORTFOLIO_MAX_CONCURRENT_TRADES,
                portfolio_risk_cap_pct=PORTFOLIO_RISK_CAP_PCT,
                drawdown_throttle_trigger_pct=PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT,
                drawdown_recovery_pct=PORTFOLIO_DRAWDOWN_RECOVERY_PCT,
                throttled_risk_pct=PORTFOLIO_THROTTLED_RISK_PCT,
            ),
        ).print_report()

    if RUN_JSON_EXPORT:
        ReportExporter(df, export_config()).save(JSON_EXPORT_PATH)


def export_config():
    """Every setting ReportExporter needs, gathered from this module's constants."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "total_days": TOTAL_DAYS,
        "oracle_lookahead": ORACLE_LOOKAHEAD,
        "oracle_min_reward_risk_ratio": ORACLE_MIN_REWARD_RISK_RATIO,
        "run_relevance_analysis": RUN_RELEVANCE_ANALYSIS,
        "run_backtest": RUN_BACKTEST,
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
        "run_oracle_backtest": RUN_ORACLE_BACKTEST,
        "portfolio_max_concurrent_trades": PORTFOLIO_MAX_CONCURRENT_TRADES,
        "portfolio_risk_cap_pct": PORTFOLIO_RISK_CAP_PCT,
        "portfolio_drawdown_throttle_trigger_pct": PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT,
        "portfolio_drawdown_recovery_pct": PORTFOLIO_DRAWDOWN_RECOVERY_PCT,
        "portfolio_throttled_risk_pct": PORTFOLIO_THROTTLED_RISK_PCT,
    }


if __name__ == "__main__":
    main()
