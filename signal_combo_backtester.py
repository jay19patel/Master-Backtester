"""SignalComboBacktester: combines multiple PriceActionEngine sig_* signals into
a single confluence trade signal (they must ALL agree on direction on the same
candle) and backtests EACH combination the exact same realistic way Backtester
backtests one signal alone - so you can see whether waiting for 2, 3 or 4
signals to line up beats trading any single one of them.

This answers a different question than ConditionFinder: ConditionFinder scores
combinations of static indicator conditions against the oracle's BUY/SELL label
(a precision check). This scores combinations of actual tradeable PriceAction
signals against REAL simulated trades (win rate, PnL) - the same realistic
bracket, fees, and position sizing as Backtester.

Performance note: requiring several independent, sparse signals to fire on the
EXACT same candle is combinatorially rare, so most combinations end up with
zero or very few fires. A cheap vectorized fire-count check runs first for
every combination; only the ones clearing `min_fires` go through the much
more expensive bar-by-bar trade simulation, which keeps combo sizes up to 4
practical even though the search space itself is large.
"""

import itertools

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from backtester import Backtester
from price_action_engine import PriceActionEngine


class SignalComboBacktester:
    """Backtests every combination of `min_combo_size..max_combo_size` sig_*
    signals (all must agree on direction on the same candle) using the same
    realistic bracket/fee/sizing mechanics as Backtester.

    Usage:
        SignalComboBacktester(df).print_report()
    """

    def __init__(
        self,
        df,
        signals=None,
        initial_capital=100.0,
        risk_per_trade_pct=2.0,
        stop_loss_pct=0.5,
        take_profit_pct=1.0,
        max_hold_bars=20,
        fee_pct=0.05,
        min_combo_size=2,
        max_combo_size=4,
        min_fires=15,
    ):
        has_signals = any(c.startswith("sig_") for c in df.columns)
        self.df = df.copy() if has_signals else PriceActionEngine(df.copy()).build()
        self.signal_names = signals or [c for c in self.df.columns if c.startswith("sig_")]
        self.min_combo_size = min_combo_size
        self.max_combo_size = max_combo_size
        self.min_fires = min_fires

        self.backtester = Backtester(
            self.df,
            initial_capital=initial_capital,
            risk_per_trade_pct=risk_per_trade_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_hold_bars=max_hold_bars,
            fee_pct=fee_pct,
        )

    def _combo_direction_array(self, combo):
        """+1 where EVERY signal in the combo == 1 (all agree long), -1 where
        every signal == -1 (all agree short), else 0."""
        stacked = np.vstack([self.df[name].to_numpy() for name in combo])
        all_long = np.all(stacked == 1, axis=0)
        all_short = np.all(stacked == -1, axis=0)
        return np.where(all_long, 1, np.where(all_short, -1, 0))

    def run(self):
        """Backtest every signal combination. Returns a DataFrame, best PnL first."""
        rows = []
        names = self.signal_names

        for size in range(self.min_combo_size, self.max_combo_size + 1):
            for combo in itertools.combinations(names, size):
                combo_arr = self._combo_direction_array(combo)
                fires = int(np.count_nonzero(combo_arr))
                if fires < self.min_fires:
                    continue  # cheap vectorized skip before the expensive simulation

                trades, final_equity = self.backtester.simulate_direction_array(combo_arr)
                n_trades = len(trades)
                if n_trades < self.min_fires:
                    continue

                wins = [t for t in trades if t["pnl"] > 0]
                total_pnl = final_equity - self.backtester.initial_capital

                rows.append(
                    {
                        "combo": " + ".join(name.replace("sig_", "") for name in combo),
                        "size": size,
                        "fires": fires,
                        "trades": n_trades,
                        "win_rate_pct": round(len(wins) / n_trades * 100, 1),
                        "final_equity": round(final_equity, 2),
                        "total_pnl": round(total_pnl, 2),
                        "return_pct": round(total_pnl / self.backtester.initial_capital * 100, 1),
                    }
                )

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("total_pnl", ascending=False).reset_index(drop=True)
        return result

    def print_report(self):
        result = self.run()
        console = Console(width=220)

        n_signals = len(self.signal_names)
        console.print(
            f"[bold]SIGNAL COMBINATION BACKTEST[/bold] - {n_signals} base signals, combo sizes "
            f"{self.min_combo_size}-{self.max_combo_size} (must ALL agree on direction, same candle), "
            f"min {self.min_fires} fires kept"
        )

        if result.empty:
            console.print("No combination cleared the minimum fire count - agreement across this many signals is too rare in this data.")
            return result

        console.print(
            f"{len(result)} combinations cleared the minimum - ranked by total PnL on a "
            f"${self.backtester.initial_capital:.0f} account, same bracket/fees as the standalone backtest"
        )

        table = Table(title=f"{len(result)} qualifying combinations, best PnL first", show_lines=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Combo", style="bold")
        table.add_column("size", justify="right")
        table.add_column("fires", justify="right")
        table.add_column("trades", justify="right")
        table.add_column("win_rate%", justify="right")
        table.add_column("final_$", justify="right")
        table.add_column("total_pnl", justify="right")
        table.add_column("return%", justify="right")

        for i, row in result.iterrows():
            pnl_style = "green" if row["total_pnl"] > 0 else "red"
            table.add_row(
                str(i + 1),
                row["combo"],
                str(row["size"]),
                str(row["fires"]),
                str(row["trades"]),
                f"{row['win_rate_pct']:.1f}",
                f"{row['final_equity']:.2f}",
                f"[{pnl_style}]{row['total_pnl']:+.2f}[/{pnl_style}]",
                f"[{pnl_style}]{row['return_pct']:+.1f}[/{pnl_style}]",
            )

        console.print(table)

        best = result.iloc[0]
        console.print(
            f"\n[bold]Best combo:[/bold] {best['combo']} -> ${self.backtester.initial_capital:.0f} became "
            f"${best['final_equity']:.2f} ({best['return_pct']:+.1f}%) over {best['trades']} trades, "
            f"{best['win_rate_pct']:.1f}% win rate."
        )
        return result
