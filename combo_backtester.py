"""ComboBacktester: searches combinations that freely MIX every indicator
column together with PriceActionEngine sig_* signals, and backtests each
combination for REAL money (win rate, PnL) using the exact same realistic
engine as Backtester - entry at next open, stop/target bracket walked
bar-by-bar, risk-based position sizing, fees, no overlapping trades.

This replaces two earlier, separate searches: ConditionFinder (indicator
conditions only, scored against the oracle label - a precision check, not real
money) and SignalComboBacktester (price-action signals only). Mixing both
pools means a combo can be, for example, "RSI_14<median AND trend_volume>median
AND golden_pullback(S)" - indicator conditions plus a price-action signal
together, and nothing is excluded: every one of the ~136 indicator columns and
all 24 sig_* signals can appear in any combination.

Directional tagging: every indicator column gets an automatic long/short pair
by comparing its value to its own trailing rolling median (`value >
median` -> long pool, `value < median` -> short pool). The median is computed
over a strictly causal rolling window (`condition_window`, default 100 bars),
so it only ever looks at past data - no lookahead. This uniform rule covers
oscillators, moving averages, volume/volatility measures, interaction features
like trend_volume, anything - without hand-curating a condition per indicator.
Every sig_* price-action signal is split into a long half (`col == 1`) and a
short half (`col == -1`).

Performance: requiring several conditions/signals to ALL be true on the same
candle is combinatorially rare, and the full pool is large (~160
conditions/signals per direction). Combo sizes are capped at 1-2 by default
(every single condition plus every pair) to keep the search to tens of
thousands of combinations instead of tens of millions - a cheap vectorized
fire-count check runs first for every combination; only the ones clearing
`min_fires` go through the much more expensive bar-by-bar trade simulation.
"""

import itertools

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from backtester import Backtester

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class ComboBacktester:
    """Searches combined indicator-condition + price-action-signal combinations
    and backtests each one for real PnL. Requires sig_* columns to already be
    present in `df` (build them with PriceActionEngine before constructing this).

    Usage:
        ComboBacktester(df).print_report()
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
        min_combo_size=1,
        max_combo_size=2,
        min_fires=15,
        console_top_n=25,
        condition_window=100,
    ):
        self.df = df
        self.min_combo_size = min_combo_size
        self.max_combo_size = max_combo_size
        self.min_fires = min_fires
        self.console_top_n = console_top_n
        self.condition_window = condition_window
        self.backtester = Backtester(
            df,
            initial_capital=initial_capital,
            risk_per_trade_pct=risk_per_trade_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_hold_bars=max_hold_bars,
            fee_pct=fee_pct,
        )
        self.stats = {}

    # ------------------------------------------------------------------
    # Build the long/short condition + signal pools
    # ------------------------------------------------------------------
    def _build_pools(self):
        """Every indicator column (all ~136 of them) gets an automatic
        long/short condition pair: is it currently above or below its own
        trailing rolling median? Every sig_* price-action signal is split into
        its long (`== 1`) and short (`== -1`) half. Nothing is hand-curated or
        excluded - the full pool covers every indicator crossed with every
        price-action signal."""
        df = self.df
        long_pool = {}
        short_pool = {}

        excluded = set(OHLCV_COLUMNS) | {"oracle_signal"}
        indicator_cols = [
            c
            for c in df.columns
            if c not in excluded
            and not c.startswith("oracle_")
            and not c.startswith("sig_")
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        for col in indicator_cols:
            rolling_median = df[col].rolling(self.condition_window, min_periods=self.condition_window).median()
            long_pool[f"{col}>median"] = (df[col] > rolling_median).to_numpy()
            short_pool[f"{col}<median"] = (df[col] < rolling_median).to_numpy()

        for col in [c for c in df.columns if c.startswith("sig_")]:
            arr = df[col].to_numpy()
            name = col.replace("sig_", "")
            long_pool[f"{name}(L)"] = arr == 1
            short_pool[f"{name}(S)"] = arr == -1

        return long_pool, short_pool

    # ------------------------------------------------------------------
    # Search + backtest
    # ------------------------------------------------------------------
    def run(self):
        """Backtest every qualifying combination. Returns a DataFrame (both
        winning and losing combos that cleared min_fires), best PnL first."""
        long_pool, short_pool = self._build_pools()
        self.stats = {
            "long_pool_size": len(long_pool),
            "short_pool_size": len(short_pool),
            "combos_tested": 0,
            "combos_cleared_min_fires": 0,
            "combos_simulated": 0,
        }

        rows = []
        for direction, pool in ((1, long_pool), (-1, short_pool)):
            names = list(pool.keys())
            for size in range(self.min_combo_size, self.max_combo_size + 1):
                for combo in itertools.combinations(names, size):
                    self.stats["combos_tested"] += 1

                    combined_mask = pool[combo[0]]
                    for name in combo[1:]:
                        combined_mask = combined_mask & pool[name]

                    fires = int(np.count_nonzero(combined_mask))
                    if fires < self.min_fires:
                        continue
                    self.stats["combos_cleared_min_fires"] += 1

                    direction_array = np.where(combined_mask, direction, 0)
                    trades, final_equity = self.backtester.simulate_direction_array(direction_array)
                    self.stats["combos_simulated"] += 1

                    n_trades = len(trades)
                    if n_trades < self.min_fires:
                        continue

                    wins = [t for t in trades if t["pnl"] > 0]
                    total_pnl = final_equity - self.backtester.initial_capital

                    rows.append(
                        {
                            "combo": " AND ".join(combo),
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

        console.print(
            f"[bold]COMBO BACKTEST[/bold] - long pool {self.stats['long_pool_size']}, "
            f"short pool {self.stats['short_pool_size']} conditions/signals, combo sizes "
            f"{self.min_combo_size}-{self.max_combo_size}, min {self.min_fires} fires kept"
        )
        console.print(
            f"Search funnel: {self.stats['combos_tested']:,} tested -> "
            f"{self.stats['combos_cleared_min_fires']:,} cleared min fires -> "
            f"{self.stats['combos_simulated']:,} simulated"
        )

        if result.empty:
            console.print("No combination cleared the minimum fire count.")
            return result

        profitable = result[result["total_pnl"] > 0].reset_index(drop=True)
        console.print(
            f"{len(profitable)} of {len(result)} simulated combinations were profitable "
            f"on a ${self.backtester.initial_capital:.0f} account - ranked by total PnL"
        )

        if profitable.empty:
            console.print("None were profitable under these realistic assumptions.")
            return profitable

        # The full profitable list (all of it, however many that is) is what
        # gets saved to report.json / shown in the dashboard - the console only
        # needs a readable top slice, not a multi-thousand-row dump.
        shown = profitable.head(self.console_top_n)
        title = f"Top {len(shown)} of {len(profitable)} profitable combinations, best PnL first"
        if len(profitable) > len(shown):
            title += f" (see report.json / dashboard for all {len(profitable)})"

        table = Table(title=title, show_lines=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Combo", style="bold")
        table.add_column("size", justify="right")
        table.add_column("fires", justify="right")
        table.add_column("trades", justify="right")
        table.add_column("win_rate%", justify="right")
        table.add_column("final_$", justify="right")
        table.add_column("total_pnl", justify="right")
        table.add_column("return%", justify="right")

        for i, row in shown.iterrows():
            table.add_row(
                str(i + 1),
                row["combo"],
                str(row["size"]),
                str(row["fires"]),
                str(row["trades"]),
                f"{row['win_rate_pct']:.1f}",
                f"{row['final_equity']:.2f}",
                f"[green]{row['total_pnl']:+.2f}[/green]",
                f"[green]{row['return_pct']:+.1f}[/green]",
            )
        console.print(table)

        best = profitable.iloc[0]
        console.print(
            f"\n[bold]Best combo:[/bold] {best['combo']} -> ${self.backtester.initial_capital:.0f} became "
            f"${best['final_equity']:.2f} ({best['return_pct']:+.1f}%) over {best['trades']} trades, "
            f"{best['win_rate_pct']:.1f}% win rate."
        )
        return profitable
