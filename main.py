"""Entry point: fetch data, engineer indicators, add oracle labels, and report stats."""

from datetime import datetime, timezone

import pandas as pd

from backtester import Backtester
from condition_finder import ConditionFinder
from data_fetcher import DataFetcher
from indicator_engine import IndicatorEngine
from oracle_backtester import OracleBacktester
from oracle_labeler import OracleLabeler
from portfolio_manager import PortfolioManager
from relevance_analyzer import RelevanceAnalyzer
from report_exporter import ReportExporter
from signal_combo_backtester import SignalComboBacktester
from signal_evaluator import evaluate_signals

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

SYMBOL = "BTCUSD"
INTERVAL = "1h"
TOTAL_DAYS = 365
ORACLE_LOOKAHEAD = 20
ORACLE_MIN_REWARD_RISK_RATIO = 2.0  # 1:2 minimum - the winning side must be at least double the losing side

# Indicators are needed now (the EMA vs signal heatmap depends on EMA_20). The
# relevance analysis is a separate, heavier report - keep it off unless asked for.
INCLUDE_INDICATORS = True
RUN_RELEVANCE_ANALYSIS = True
RUN_CONDITION_SEARCH = True

CONDITION_MIN_SUPPORT = 100
CONDITION_MAX_COMBO_SIZE = 4  # 3 or 4 indicators combined together, not just pairs

RUN_PRICE_ACTION = True
PRICE_ACTION_FORWARD_BARS = 10
PRICE_ACTION_MIN_FIRES = 15

RUN_BACKTEST = True
BACKTEST_INITIAL_CAPITAL = 100.0
BACKTEST_RISK_PER_TRADE_PCT = 2.0
BACKTEST_STOP_LOSS_PCT = 1  # 0.5% stop-loss
BACKTEST_TAKE_PROFIT_PCT = 2  # 1% target -> 1:2 reward:risk
BACKTEST_MAX_HOLD_BARS = 20
BACKTEST_FEE_PCT = 0.05

RUN_PORTFOLIO = True
PORTFOLIO_MAX_CONCURRENT_TRADES = 3
PORTFOLIO_RISK_CAP_PCT = 6.0
PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT = 10.0
PORTFOLIO_DRAWDOWN_RECOVERY_PCT = 5.0
PORTFOLIO_THROTTLED_RISK_PCT = 1.0
PORTFOLIO_BREAKEVEN_TRIGGER_R = 1.0
PORTFOLIO_TRAIL_TRIGGER_R = 1.5
PORTFOLIO_TRAIL_DISTANCE_R = 0.5

RUN_SIGNAL_COMBO_BACKTEST = True
SIGNAL_COMBO_MIN_SIZE = 2
SIGNAL_COMBO_MAX_SIZE = 4
SIGNAL_COMBO_MIN_FIRES = 15

RUN_ORACLE_BACKTEST = True

RUN_JSON_EXPORT = True
JSON_EXPORT_PATH = "report.json"

EMA_COLUMN = "EMA_20"

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def build_dataset():
    fetcher = DataFetcher(symbol=SYMBOL, interval=INTERVAL, total_days=TOTAL_DAYS)
    df = fetcher.fetch()

    if df.empty:
        raise RuntimeError("No data fetched - aborting.")

    if INCLUDE_INDICATORS:
        df = IndicatorEngine(df).build()

    df = OracleLabeler(
        df, lookahead=ORACLE_LOOKAHEAD, min_reward_risk_ratio=ORACLE_MIN_REWARD_RISK_RATIO
    ).label()
    return df


def column_groups(df):
    """Split columns into OHLCV / oracle / indicator buckets for reporting."""
    oracle_cols = [c for c in df.columns if c.startswith("oracle_")]
    ohlcv_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
    indicator_cols = [c for c in df.columns if c not in oracle_cols and c not in ohlcv_cols]
    return ohlcv_cols, indicator_cols, oracle_cols


def ema_signal_crosstab(df, ema_col=EMA_COLUMN):
    """Cross-tab of "price vs EMA" trend state against oracle_signal.

    Returns None if either column is missing, else (counts, row_pct, row_order, col_order).
    """
    if ema_col not in df.columns or "oracle_signal" not in df.columns:
        return None

    known = df[df["oracle_signal"].notna() & df[ema_col].notna()].copy()

    above_label = f"Above {ema_col}"
    below_label = f"Below {ema_col}"
    known["ema_state"] = (known["Close"] >= known[ema_col]).map({True: above_label, False: below_label})

    row_order = [name for name in [above_label, below_label] if name in known["ema_state"].unique()]
    col_order = [s for s in ["BUY", "HOLD", "SELL"] if s in known["oracle_signal"].unique()]

    counts = pd.crosstab(known["ema_state"], known["oracle_signal"]).reindex(
        index=row_order, columns=col_order, fill_value=0
    )
    row_pct = counts.div(counts.sum(axis=1), axis=0) * 100
    return counts, row_pct, row_order, col_order


def print_ema_signal_relationship(df):
    """Show how price-vs-EMA trend state lines up with oracle_signal, so you can see
    which trend state tends to produce which kind of signal."""
    crosstab = ema_signal_crosstab(df)
    if crosstab is None:
        return
    counts, row_pct, row_order, col_order = crosstab

    print(f"\n--- {EMA_COLUMN} trend vs Signal relationship ---")
    print(f"(of the candles where price was above/below {EMA_COLUMN}, what % got each signal)")

    print("\nCounts:")
    print(counts)

    print("\nRow % (within each trend state):")
    print(row_pct.round(1))

    print("\nIn words:")
    for state_name in row_order:
        total = counts.loc[state_name].sum()
        parts = ", ".join(f"{col} {row_pct.loc[state_name, col]:.1f}%" for col in col_order)
        print(f"  When price was {state_name:<16} ({total:>5} candles) -> {parts}")


def print_signal_meaning(df):
    """Explain in plain language + real averages what a BUY/HOLD/SELL signal means."""
    if "oracle_signal" not in df.columns:
        return

    known = df[df["oracle_signal"].notna()]
    stats_cols = ["oracle_upside_pct", "oracle_downside_pct", "oracle_upside_gap", "oracle_downside_gap"]
    avg = known.groupby("oracle_signal")[stats_cols].mean()
    counts = known["oracle_signal"].value_counts()

    print(f"\n--- What each signal means (avg over the next {ORACLE_LOOKAHEAD} candles) ---")
    for signal in ["BUY", "HOLD", "SELL"]:
        if signal not in avg.index:
            continue
        row = avg.loc[signal]
        n = int(counts.get(signal, 0))
        print(f"\n{signal} ({n} candles, {n / len(known) * 100:.1f}% of known candles):")
        print(f"  Avg potential upside   : +{row['oracle_upside_pct']:.2f}%  (price gap {row['oracle_upside_gap']:.4f})")
        print(f"  Avg potential downside : {row['oracle_downside_pct']:.2f}%  (price gap {row['oracle_downside_gap']:.4f})")

        if signal == "BUY":
            print(
                f"  Meaning: going forward {ORACLE_LOOKAHEAD} candles, the upside "
                f"(+{row['oracle_upside_pct']:.2f}%) was at least {ORACLE_MIN_REWARD_RISK_RATIO:.0f}x "
                f"bigger than the downside ({row['oracle_downside_pct']:.2f}%) -> a long trade had a "
                f"clear, favorable reward:risk edge."
            )
        elif signal == "SELL":
            print(
                f"  Meaning: going forward {ORACLE_LOOKAHEAD} candles, the downside "
                f"({row['oracle_downside_pct']:.2f}%) was at least {ORACLE_MIN_REWARD_RISK_RATIO:.0f}x "
                f"bigger than the upside (+{row['oracle_upside_pct']:.2f}%) -> a short trade had a "
                f"clear, favorable reward:risk edge."
            )
        else:
            print(
                "  Meaning: neither side cleared the "
                f"{ORACLE_MIN_REWARD_RISK_RATIO:.0f}x bar over the other -> no clean edge, best to stay flat."
            )


def print_report(df):
    print("\n" + "=" * 70)
    print("DATASET REPORT")
    print("=" * 70)
    print(f"Symbol / Interval      : {SYMBOL} / {INTERVAL}")
    print(f"Rows (candles)         : {len(df)}")
    print(f"Columns                : {len(df.columns)}")
    print(f"Date range             : {df.index.min()}  ->  {df.index.max()}")
    print(f"Missing values (total) : {int(df.isna().sum().sum())}")

    if "oracle_signal" in df.columns:
        print(f"\n--- Oracle signal (min reward:risk = 1:{ORACLE_MIN_REWARD_RISK_RATIO:.0f}) ---")
        counts = df["oracle_signal"].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            name = value if pd.notna(value) else "UNKNOWN (tail, no future data yet)"
            pct = count / len(df) * 100
            print(f"  {name:<28}: {count:>6}  ({pct:5.1f}%)")

    print_signal_meaning(df)
    print_ema_signal_relationship(df)

    ohlcv_cols, indicator_cols, oracle_cols = column_groups(df)

    print(f"\n--- OHLCV columns ({len(ohlcv_cols)}) ---")
    for i, col in enumerate(ohlcv_cols, start=1):
        print(f"  {i:>3}. {col}")

    print(f"\n--- Oracle columns ({len(oracle_cols)}) ---")
    for i, col in enumerate(oracle_cols, start=1):
        print(f"  {i:>3}. {col}")

    print(f"\n--- Indicator columns ({len(indicator_cols)}) ---")
    for i, col in enumerate(indicator_cols, start=1):
        print(f"  {i:>3}. {col}")

    print("=" * 70 + "\n")


def print_price_action_report(df):
    """Run every PriceActionEngine sig_* strategy and rank them by forward-return
    expectancy (in ATR units) over the next `PRICE_ACTION_FORWARD_BARS` candles."""
    print("\n" + "=" * 100)
    print(
        f"PRICE ACTION SIGNAL EVALUATION (forward={PRICE_ACTION_FORWARD_BARS} candles, "
        f"min fires={PRICE_ACTION_MIN_FIRES})"
    )
    print("=" * 100)
    print("hit_rate    : of the candles this signal fired on, % that were profitable")
    print("              `forward_bars` later (long signals measured long, shorts measured short)")
    print("avg_R       : average forward move in ATR units (e.g. 0.5 = moved half an ATR in its favor)")
    print("expectancy_R: hit_rate x avg_win - (1-hit_rate) x avg_loss -> the number that actually")
    print("              matters: positive = this signal has a real edge, negative = it loses on average")

    result = evaluate_signals(df, forward_bars=PRICE_ACTION_FORWARD_BARS, min_fires=PRICE_ACTION_MIN_FIRES)

    header = f"{'Signal':<26} {'fires':>7} {'hit_rate':>9} {'avg_R':>8} {'expectancy_R':>13}"
    print("\n" + header)
    print("-" * len(header))
    for _, row in result.iterrows():
        if row["hit_rate"] is None:
            print(f"{row['signal']:<26} {row['fires']:>7}   insufficient data (< {PRICE_ACTION_MIN_FIRES} fires)")
        else:
            print(
                f"{row['signal']:<26} {row['fires']:>7} {row['hit_rate']:>9.1%} "
                f"{row['avg_R']:>8.3f} {row['expectancy_R']:>13.3f}"
            )
    print("=" * 100 + "\n")
    return result


def main():
    df = build_dataset()
    print_report(df)
    if RUN_RELEVANCE_ANALYSIS:
        RelevanceAnalyzer(df).print_report()
    if RUN_CONDITION_SEARCH:
        ConditionFinder(
            df, min_support=CONDITION_MIN_SUPPORT, max_combo_size=CONDITION_MAX_COMBO_SIZE
        ).print_report()
    if RUN_PRICE_ACTION:
        print_price_action_report(df)
    backtest_result = None
    if RUN_BACKTEST:
        backtest_result = Backtester(
            df,
            initial_capital=BACKTEST_INITIAL_CAPITAL,
            risk_per_trade_pct=BACKTEST_RISK_PER_TRADE_PCT,
            stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
            take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
            max_hold_bars=BACKTEST_MAX_HOLD_BARS,
            fee_pct=BACKTEST_FEE_PCT,
        ).print_report()

    if RUN_PORTFOLIO and backtest_result is not None:
        profitable_signals = backtest_result.loc[backtest_result["total_pnl"] > 0, "signal"].tolist()
        if profitable_signals:
            print(
                f"[Portfolio] Trading the {len(profitable_signals)} signal(s) that were profitable "
                f"standalone, together on one account: {profitable_signals}"
            )
            PortfolioManager(
                df,
                signals=profitable_signals,
                initial_capital=BACKTEST_INITIAL_CAPITAL,
                risk_per_trade_pct=BACKTEST_RISK_PER_TRADE_PCT,
                stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
                take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
                max_hold_bars=BACKTEST_MAX_HOLD_BARS,
                fee_pct=BACKTEST_FEE_PCT,
                max_concurrent_trades=PORTFOLIO_MAX_CONCURRENT_TRADES,
                portfolio_risk_cap_pct=PORTFOLIO_RISK_CAP_PCT,
                drawdown_throttle_trigger_pct=PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT,
                drawdown_recovery_pct=PORTFOLIO_DRAWDOWN_RECOVERY_PCT,
                throttled_risk_pct=PORTFOLIO_THROTTLED_RISK_PCT,
                breakeven_trigger_r=PORTFOLIO_BREAKEVEN_TRIGGER_R,
                trail_trigger_r=PORTFOLIO_TRAIL_TRIGGER_R,
                trail_distance_r=PORTFOLIO_TRAIL_DISTANCE_R,
            ).print_report()
        else:
            print("[Portfolio] No standalone-profitable signals found - skipping portfolio run.\n")

    if RUN_SIGNAL_COMBO_BACKTEST:
        SignalComboBacktester(
            df,
            initial_capital=BACKTEST_INITIAL_CAPITAL,
            risk_per_trade_pct=BACKTEST_RISK_PER_TRADE_PCT,
            stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
            take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
            max_hold_bars=BACKTEST_MAX_HOLD_BARS,
            fee_pct=BACKTEST_FEE_PCT,
            min_combo_size=SIGNAL_COMBO_MIN_SIZE,
            max_combo_size=SIGNAL_COMBO_MAX_SIZE,
            min_fires=SIGNAL_COMBO_MIN_FIRES,
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
        "ema_column": EMA_COLUMN,
        "run_relevance_analysis": RUN_RELEVANCE_ANALYSIS,
        "run_condition_search": RUN_CONDITION_SEARCH,
        "condition_min_support": CONDITION_MIN_SUPPORT,
        "condition_max_combo_size": CONDITION_MAX_COMBO_SIZE,
        "run_price_action": RUN_PRICE_ACTION,
        "price_action_forward_bars": PRICE_ACTION_FORWARD_BARS,
        "price_action_min_fires": PRICE_ACTION_MIN_FIRES,
        "run_backtest": RUN_BACKTEST,
        "backtest_initial_capital": BACKTEST_INITIAL_CAPITAL,
        "backtest_risk_per_trade_pct": BACKTEST_RISK_PER_TRADE_PCT,
        "backtest_stop_loss_pct": BACKTEST_STOP_LOSS_PCT,
        "backtest_take_profit_pct": BACKTEST_TAKE_PROFIT_PCT,
        "backtest_max_hold_bars": BACKTEST_MAX_HOLD_BARS,
        "backtest_fee_pct": BACKTEST_FEE_PCT,
        "run_portfolio": RUN_PORTFOLIO,
        "portfolio_max_concurrent_trades": PORTFOLIO_MAX_CONCURRENT_TRADES,
        "portfolio_risk_cap_pct": PORTFOLIO_RISK_CAP_PCT,
        "portfolio_drawdown_throttle_trigger_pct": PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT,
        "portfolio_drawdown_recovery_pct": PORTFOLIO_DRAWDOWN_RECOVERY_PCT,
        "portfolio_throttled_risk_pct": PORTFOLIO_THROTTLED_RISK_PCT,
        "run_signal_combo_backtest": RUN_SIGNAL_COMBO_BACKTEST,
        "signal_combo_min_size": SIGNAL_COMBO_MIN_SIZE,
        "signal_combo_max_size": SIGNAL_COMBO_MAX_SIZE,
        "signal_combo_min_fires": SIGNAL_COMBO_MIN_FIRES,
        "run_oracle_backtest": RUN_ORACLE_BACKTEST,
    }


if __name__ == "__main__":
    main()
