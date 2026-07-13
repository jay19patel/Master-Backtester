"""Entry point: fetch data, build the features the active strategy(ies) need,
backtest each one independently AND together on one portfolio-managed
account, and save the results (including the full trade-by-trade order log).

This branch is intentionally lean - 4 files:
    data_fetcher.py      - fetch/cache OHLCV data
    backtester.py         - realistic single-strategy simulation engine
    portfolio_manager.py  - risk-managed simulation of the active strategies together
                             (concurrent-position cap, portfolio risk cap, drawdown throttle)
    strategies.py         - all 25 candidate strategies + the feature engineering they need

main.py itself is just the glue: fetch -> build features -> backtest each
ACTIVE strategy standalone (with its full trade log) -> all of them together
under PortfolioManager -> save results.json / results.csv / trades.csv.
"""

import json
from datetime import datetime, timezone

import pandas as pd
from rich.console import Console
from rich.table import Table

from backtester import Backtester
from data_fetcher import DataFetcher
from portfolio_manager import PortfolioManager
from strategies import STRATEGIES, build_direction_array, build_features

SYMBOL = "ETHUSD"
INTERVAL = "1h"
TOTAL_DAYS = 365

# Which of the 25 candidate strategies (defined in strategies.py) to actually
# trade. strategy_01 is the best LONG-only strategy (direction=1) and
# strategy_02 is the best SHORT-only strategy (direction=-1) of the 25 - every
# strategy here trades in only ONE fixed direction (see strategies.py), so
# pairing the best of each side is how you get both LONG and SHORT signals
# while keeping each side's most profitable strategy.
ACTIVE_STRATEGY_NAMES = ["strategy_01", "strategy_02"]

BACKTEST_INITIAL_CAPITAL = 100.0
BACKTEST_RISK_PER_TRADE_PCT = 2.0
BACKTEST_STOP_LOSS_PCT = 1  # 1% stop
BACKTEST_TAKE_PROFIT_PCT = 3  # 3% target -> 1:3 reward:risk
BACKTEST_MAX_HOLD_BARS = 20
BACKTEST_FEE_PCT = 0.05
BACKTEST_MAX_LEVERAGE = 2.0  # hard cap: position notional never exceeds 2x current equity

# PortfolioManager risk-management knobs for the "all active strategies together" run.
PORTFOLIO_MAX_CONCURRENT_TRADES = 5
PORTFOLIO_RISK_CAP_PCT = 10.0
PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT = 10.0
PORTFOLIO_DRAWDOWN_RECOVERY_PCT = 5.0
PORTFOLIO_THROTTLED_RISK_PCT = 1.0

RESULTS_JSON_PATH = "results.json"
RESULTS_CSV_PATH = "results.csv"
TRADES_CSV_PATH = "trades.csv"


def build_dataset():
    df = DataFetcher(symbol=SYMBOL, interval=INTERVAL, total_days=TOTAL_DAYS).fetch()
    if df.empty:
        raise RuntimeError("No data fetched - aborting.")
    return build_features(df)


def build_strategy_direction_arrays(df):
    """Computed ONCE and reused for both the standalone and portfolio-managed
    runs below, since building a direction array does real work (rolling
    medians etc.) per strategy. Only ACTIVE_STRATEGY_NAMES are built."""
    active = [s for s in STRATEGIES if s["name"] in ACTIVE_STRATEGY_NAMES]
    missing = set(ACTIVE_STRATEGY_NAMES) - {s["name"] for s in active}
    if missing:
        raise ValueError(f"ACTIVE_STRATEGY_NAMES references unknown strategy name(s): {sorted(missing)}")
    return [{"name": s["name"], "combo": s["combo"], "direction_array": build_direction_array(df, s)} for s in active]


def run_standalone(df, strategy_arrays):
    """Each active strategy traded alone on its own account - shows each
    strategy's raw, un-managed edge."""
    bt = Backtester(
        df,
        initial_capital=BACKTEST_INITIAL_CAPITAL,
        risk_per_trade_pct=BACKTEST_RISK_PER_TRADE_PCT,
        stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
        take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
        max_hold_bars=BACKTEST_MAX_HOLD_BARS,
        fee_pct=BACKTEST_FEE_PCT,
        max_leverage=BACKTEST_MAX_LEVERAGE,
    )

    rows = []
    for strategy in strategy_arrays:
        result = bt.run_strategy(strategy["name"], strategy["direction_array"])
        if result is None:
            continue
        result["combo"] = strategy["combo"]
        rows.append(result)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("total_pnl", ascending=False).reset_index(drop=True)
    return result_df


def run_portfolio(df, strategy_arrays):
    """All active strategies traded together on ONE shared, risk-managed account -
    shows what capital growth looks like when position sizing/concurrency/
    drawdown are actively managed across the whole basket, not per-strategy."""
    pm = PortfolioManager(
        df,
        strategy_arrays,
        initial_capital=BACKTEST_INITIAL_CAPITAL,
        risk_per_trade_pct=BACKTEST_RISK_PER_TRADE_PCT,
        stop_loss_pct=BACKTEST_STOP_LOSS_PCT,
        take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
        max_hold_bars=BACKTEST_MAX_HOLD_BARS,
        fee_pct=BACKTEST_FEE_PCT,
        max_leverage=BACKTEST_MAX_LEVERAGE,
        max_concurrent_trades=PORTFOLIO_MAX_CONCURRENT_TRADES,
        portfolio_risk_cap_pct=PORTFOLIO_RISK_CAP_PCT,
        drawdown_throttle_trigger_pct=PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT,
        drawdown_recovery_pct=PORTFOLIO_DRAWDOWN_RECOVERY_PCT,
        throttled_risk_pct=PORTFOLIO_THROTTLED_RISK_PCT,
    )
    return pm, pm.print_report()


def print_report(result_df):
    console = Console(width=220)
    console.print(
        f"\n[bold]STRATEGY BACKTEST[/bold]: {len(ACTIVE_STRATEGY_NAMES)} active strategy(ies), each traded on its "
        f"own ${BACKTEST_INITIAL_CAPITAL:.0f} starting balance"
    )
    console.print(
        f"Stop-loss / Target: {BACKTEST_STOP_LOSS_PCT}% / {BACKTEST_TAKE_PROFIT_PCT}% | "
        f"Risk per trade: {BACKTEST_RISK_PER_TRADE_PCT}% | Fee (round trip): {BACKTEST_FEE_PCT * 2}%"
    )

    if result_df.empty:
        console.print("\nNo strategy produced any trades.")
        return

    table = Table(title=f"All {len(result_df)} active strategies, ranked by total PnL", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Strategy", style="bold")
    table.add_column("trades", justify="right")
    table.add_column("win_rate%", justify="right")
    table.add_column("final_$", justify="right")
    table.add_column("total_pnl", justify="right")
    table.add_column("return%", justify="right")
    table.add_column("max_dd%", justify="right")

    for i, row in result_df.iterrows():
        pnl_style = "green" if row["total_pnl"] > 0 else "red"
        table.add_row(
            str(i + 1),
            row["name"],
            str(row["trades"]),
            f"{row['win_rate_pct']:.1f}",
            f"{row['final_equity']:.2f}",
            f"[{pnl_style}]{row['total_pnl']:+.2f}[/{pnl_style}]",
            f"[{pnl_style}]{row['return_pct']:+.1f}[/{pnl_style}]",
            f"{row['max_drawdown_pct']:.1f}",
        )
    console.print(table)

    best = result_df.iloc[0]
    console.print(
        f"\n[bold]Best:[/bold] {best['name']} ({best['combo']}) -> "
        f"${BACKTEST_INITIAL_CAPITAL:.0f} became ${best['final_equity']:.2f} "
        f"({best['return_pct']:+.1f}%) over {best['trades']} trades, {best['win_rate_pct']:.1f}% win rate."
    )


def print_trade_log(name, trade_log):
    """Full order-by-order history for one strategy: every entry/exit price,
    stop-loss, target, timing, and outcome - not just aggregate stats."""
    console = Console(width=220)
    console.print(f"\n[bold]TRADE LOG[/bold]: {name} - every order taken, in sequence")

    table = Table(show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Direction", style="bold")
    table.add_column("Entry time")
    table.add_column("Exit time")
    table.add_column("Held", justify="right")
    table.add_column("Entry $", justify="right")
    table.add_column("Stop $", justify="right")
    table.add_column("Target $", justify="right")
    table.add_column("Exit $", justify="right")
    table.add_column("Lev", justify="right")
    table.add_column("Exit reason")
    table.add_column("Planned RR", justify="right")
    table.add_column("Actual RR", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("Equity after", justify="right")

    for i, t in enumerate(trade_log, 1):
        pnl_style = "green" if t["pnl"] > 0 else "red"
        dir_style = "green" if t["direction"] == "LONG" else "red"
        rr_style = "green" if t["rr_achieved"] > 0 else "red"
        table.add_row(
            str(i),
            f"[{dir_style}]{t['direction']}[/{dir_style}]",
            str(t["entry_time"]),
            str(t["exit_time"]),
            f"{t['holding_bars']} bars ({t['holding_time']})",
            f"{t['entry_price']:.4f}",
            f"{t['stop_price']:.4f}",
            f"{t['target_price']:.4f}",
            f"{t['exit_price']:.4f}",
            f"{t['leverage']:.2f}x",
            t["exit_reason"],
            f"1:{t['planned_rr']:.1f}",
            f"[{rr_style}]{t['rr_achieved']:+.2f}R[/{rr_style}]",
            f"[{pnl_style}]{t['pnl']:+.2f}[/{pnl_style}]",
            f"{t['equity_after']:.2f}",
        )
    console.print(table)


def save_results(standalone_df, portfolio_trades, portfolio_equity, portfolio_curve):
    portfolio_wins = [t for t in portfolio_trades if t["pnl"] > 0]
    portfolio_section = {
        "trades": len(portfolio_trades),
        "win_rate_pct": round(len(portfolio_wins) / len(portfolio_trades) * 100, 1) if portfolio_trades else None,
        "final_equity": round(portfolio_equity, 2),
        "total_pnl": round(portfolio_equity - BACKTEST_INITIAL_CAPITAL, 2),
        "max_drawdown_pct": round(PortfolioManager._max_drawdown_pct(portfolio_curve), 1),
        "equity_curve": [round(v, 2) for v in portfolio_curve],
        "trade_log": portfolio_trades,
        "config": {
            "max_concurrent_trades": PORTFOLIO_MAX_CONCURRENT_TRADES,
            "portfolio_risk_cap_pct": PORTFOLIO_RISK_CAP_PCT,
            "drawdown_throttle_trigger_pct": PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT,
            "drawdown_recovery_pct": PORTFOLIO_DRAWDOWN_RECOVERY_PCT,
            "throttled_risk_pct": PORTFOLIO_THROTTLED_RISK_PCT,
            "max_leverage": BACKTEST_MAX_LEVERAGE,
        },
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "total_days": TOTAL_DAYS,
        "config": {
            "initial_capital": BACKTEST_INITIAL_CAPITAL,
            "risk_per_trade_pct": BACKTEST_RISK_PER_TRADE_PCT,
            "stop_loss_pct": BACKTEST_STOP_LOSS_PCT,
            "take_profit_pct": BACKTEST_TAKE_PROFIT_PCT,
            "max_hold_bars": BACKTEST_MAX_HOLD_BARS,
            "fee_pct": BACKTEST_FEE_PCT,
            "max_leverage": BACKTEST_MAX_LEVERAGE,
        },
        "standalone": standalone_df.drop(columns=["equity_curve"]).to_dict(orient="records"),
        "portfolio_managed": portfolio_section,
    }
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    standalone_df.drop(columns=["equity_curve", "trade_log"]).to_csv(RESULTS_CSV_PATH, index=False)

    # The ONE combined order book: portfolio_trades is the actual shared-account,
    # direction-exclusive trade log (one LONG + one SHORT max at a time) - NOT
    # the standalone per-strategy logs, which ran on separate hypothetical
    # accounts and don't reflect the direction-exclusivity rule at all.
    pd.DataFrame(portfolio_trades).to_csv(TRADES_CSV_PATH, index=False)

    print(f"[main] Saved -> {RESULTS_JSON_PATH}, {RESULTS_CSV_PATH}, {TRADES_CSV_PATH}")


def main():
    df = build_dataset()
    strategy_arrays = build_strategy_direction_arrays(df)

    standalone_df = run_standalone(df, strategy_arrays)
    print_report(standalone_df)
    for _, row in standalone_df.iterrows():
        print_trade_log(row["name"], row["trade_log"])

    _, (portfolio_trades, portfolio_equity, portfolio_curve) = run_portfolio(df, strategy_arrays)

    save_results(standalone_df, portfolio_trades, portfolio_equity, portfolio_curve)


if __name__ == "__main__":
    main()
