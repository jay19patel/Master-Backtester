"""OracleBacktester: backtests the oracle's own BUY/SELL label directly, using
the exact same realistic bracket/fee/position-sizing/risk-management machinery
as Backtester and PortfolioManager.

IMPORTANT - this is NOT a tradeable strategy. `oracle_signal` is built from the
next `oracle_lookahead` candles' actual future High/Low (see oracle_labeler.py) -
it already knows what happens next. Backtesting it does not show what a real
strategy would have made; it shows the THEORETICAL CEILING - the best a perfect
predictor of the oracle's own label could possibly do under this exact
stop/target/fee/risk setup. Every real (non-lookahead) signal in this project
should be judged against this ceiling, never expected to beat it.
"""

from rich.console import Console
from rich.table import Table

from backtester import Backtester
from portfolio_manager import PortfolioManager

# Named with a sig_ prefix so PortfolioManager sees an existing signal column
# and skips rebuilding PriceActionEngine (which this analysis has no use for).
SIGNAL_COLUMN = "sig_oracle_ceiling"


class OracleBacktester:
    """Backtests oracle_signal standalone and portfolio-risk-managed.

    Usage:
        OracleBacktester(df).print_report()
    """

    def __init__(
        self,
        df,
        initial_capital=100.0,
        risk_per_trade_pct=2.0,
        stop_loss_pct=0.5,
        take_profit_pct=1.0,
        max_hold_bars=20,
        fee_pct=0.05,
        portfolio_kwargs=None,
    ):
        self.df = df.copy()
        direction = self.df["oracle_signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0}).fillna(0)
        self.df[SIGNAL_COLUMN] = direction.astype(int)

        self.initial_capital = initial_capital
        self.backtest_kwargs = dict(
            initial_capital=initial_capital,
            risk_per_trade_pct=risk_per_trade_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_hold_bars=max_hold_bars,
            fee_pct=fee_pct,
        )
        self.portfolio_kwargs = portfolio_kwargs or {}

    def run(self):
        """Returns (standalone_dict, managed_dict). managed_dict includes an
        'equity_curve' list (per-candle, for charting)."""
        bt = Backtester(self.df, **self.backtest_kwargs)
        trades, final_equity = bt.simulate_direction_array(self.df[SIGNAL_COLUMN].to_numpy())
        wins = [t for t in trades if t["pnl"] > 0]
        standalone = {
            "trades": len(trades),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
            "final_equity": round(final_equity, 2),
            "total_pnl": round(final_equity - self.initial_capital, 2),
        }

        pm = PortfolioManager(self.df, signals=[SIGNAL_COLUMN], **self.backtest_kwargs, **self.portfolio_kwargs)
        pf_trades, pf_equity, pf_curve = pm.run()
        pf_wins = [t for t in pf_trades if t["pnl"] > 0]
        managed = {
            "trades": len(pf_trades),
            "win_rate_pct": round(len(pf_wins) / len(pf_trades) * 100, 1) if pf_trades else None,
            "final_equity": round(pf_equity, 2),
            "total_pnl": round(pf_equity - self.initial_capital, 2),
            "max_drawdown_pct": round(PortfolioManager._max_drawdown_pct(pf_curve), 1),
            "equity_curve": [round(v, 2) for v in pf_curve],
        }
        return standalone, managed

    def print_report(self):
        standalone, managed = self.run()
        console = Console(width=220)

        console.print("\n[bold yellow]ORACLE CEILING BACKTEST[/bold yellow] - NOT a tradeable strategy (uses future-looking data)")
        console.print(
            "oracle_signal is built from the NEXT lookahead candles' real high/low - it already knows the "
            "future. This is the theoretical BEST CASE under this bracket/fee setup - a ceiling to judge "
            "real signals against, not something you can trade live."
        )

        table = Table(show_lines=False)
        table.add_column("Mode", style="bold")
        table.add_column("trades", justify="right")
        table.add_column("win_rate%", justify="right")
        table.add_column("final_$", justify="right")
        table.add_column("total_pnl", justify="right")
        table.add_column("max_dd%", justify="right")

        for label, stats in [("Standalone (no risk mgmt)", standalone), ("Portfolio-managed (drawdown throttle etc.)", managed)]:
            pnl_style = "green" if stats["total_pnl"] > 0 else "red"
            table.add_row(
                label,
                str(stats["trades"]),
                f"{stats['win_rate_pct']}",
                f"{stats['final_equity']:.2f}",
                f"[{pnl_style}]{stats['total_pnl']:+.2f}[/{pnl_style}]",
                f"{stats.get('max_drawdown_pct', '-')}",
            )
        console.print(table)
        return standalone, managed
