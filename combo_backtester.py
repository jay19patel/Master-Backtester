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
conditions/signals per direction) - exhaustively trying every combination
stops being feasible past size 2 (C(160,3) is already 670K per direction,
size 5 is billions). Sizes 1-2 are still searched exhaustively (every single
condition, every pair - nothing skipped). Beyond that, a greedy beam search
takes over: the best `beam_width` combos from the previous size (by total
PnL) are each extended with every remaining pool item to form the next size,
and only the best `beam_width` of those survive to extend again. This is how
a 5-way combo like "golden_pullback(L) AND squeeze_breakout(L) AND
trend_confluence(L) AND fib_extension(L) AND return_20>median" becomes
reachable without exhaustively testing all ~200 billion 5-way combinations -
it's built up one profitable step at a time instead. A cheap vectorized
fire-count check runs first for every combination; only the ones clearing
`min_fires` go through the much more expensive bar-by-bar trade simulation.

Diversity: left unchecked, a beam search collapses onto whichever single
condition is individually strongest (e.g. swing_low_at_pivot>median beats
almost everything else alone), because every combo built on top of it also
outperforms combos built on weaker foundations - so the ENTIRE beam ends up
containing that one condition at every size beyond 2. `beam_diversity_cap`
caps how many frontier survivors may share any single condition at each step,
so weaker-but-different foundations stay in the running too. This is what
makes `diverse_top_combos()` (below) able to find combos that don't all
reduce to "condition X plus assorted extra filters".
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
        beam_width=150,
        beam_diversity_cap=None,
    ):
        self.df = df
        self.min_combo_size = min_combo_size
        self.max_combo_size = max_combo_size
        self.min_fires = min_fires
        self.console_top_n = console_top_n
        self.condition_window = condition_window
        self.beam_width = beam_width
        # Caps how many frontier survivors may share any single condition, so
        # one dominant condition (e.g. swing_low_at_pivot>median, which beats
        # almost everything else on its own) can't crowd out the ENTIRE beam -
        # without this, every combo the beam ever finds beyond size 2 ends up
        # containing that one condition, leaving nothing independent to find.
        self.beam_diversity_cap = beam_diversity_cap or max(5, beam_width // 10)
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
    def _evaluate(self, combo, size, combined_mask, direction):
        """Fire-count pre-filter, then (if it clears) the real bar-by-bar
        simulation. Returns (row_dict_or_None). Always increments
        combos_tested; increments combos_cleared_min_fires/combos_simulated
        only as it clears each stage."""
        self.stats["combos_tested"] += 1
        fires = int(np.count_nonzero(combined_mask))
        if fires < self.min_fires:
            return None
        self.stats["combos_cleared_min_fires"] += 1

        direction_array = np.where(combined_mask, direction, 0)
        trades, final_equity = self.backtester.simulate_direction_array(direction_array)
        self.stats["combos_simulated"] += 1

        n_trades = len(trades)
        if n_trades < self.min_fires:
            return None

        wins = [t for t in trades if t["pnl"] > 0]
        total_pnl = final_equity - self.backtester.initial_capital
        return {
            "combo": " AND ".join(combo),
            "size": size,
            "fires": fires,
            "trades": n_trades,
            "win_rate_pct": round(len(wins) / n_trades * 100, 1),
            "final_equity": round(final_equity, 2),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / self.backtester.initial_capital * 100, 1),
        }

    def _select_diverse_frontier(self, candidates):
        """candidates: list of (combo_names, mask, row) sorted best-PnL-first
        already. Walks best-first, skipping any candidate where a condition it
        uses has already hit `beam_diversity_cap` selections - so the frontier
        can't collapse onto variations of one dominant condition, keeping
        room for genuinely different strategies to survive into later sizes."""
        selected = []
        condition_counts = {}
        for combo_names, mask, row in candidates:
            if len(selected) >= self.beam_width:
                break
            if any(condition_counts.get(name, 0) >= self.beam_diversity_cap for name in combo_names):
                continue
            selected.append((combo_names, mask, row))
            for name in combo_names:
                condition_counts[name] = condition_counts.get(name, 0) + 1
        return selected

    def run(self):
        """Backtest every qualifying combination. Sizes 1-2 are searched
        exhaustively; sizes 3+ (if max_combo_size allows) are built with a
        greedy beam search on top of the exhaustive size-2 results, since
        exhaustively trying every larger combination is computationally
        infeasible with a ~160-item pool. Returns a DataFrame (both winning
        and losing combos that cleared min_fires), best PnL first."""
        long_pool, short_pool = self._build_pools()
        self.stats = {
            "long_pool_size": len(long_pool),
            "short_pool_size": len(short_pool),
            "combos_tested": 0,
            "combos_cleared_min_fires": 0,
            "combos_simulated": 0,
        }

        rows = []
        exhaustive_max_size = min(self.max_combo_size, 2)

        for direction, pool in ((1, long_pool), (-1, short_pool)):
            names = list(pool.keys())
            frontier = []  # [(combo_names_tuple, combined_mask, row_dict), ...] to extend beyond size 2

            for size in range(self.min_combo_size, exhaustive_max_size + 1):
                size_survivors = []
                for combo in itertools.combinations(names, size):
                    combined_mask = pool[combo[0]]
                    for name in combo[1:]:
                        combined_mask = combined_mask & pool[name]

                    row = self._evaluate(combo, size, combined_mask, direction)
                    if row is None:
                        continue
                    rows.append(row)
                    size_survivors.append((combo, combined_mask, row))

                if size == exhaustive_max_size:
                    size_survivors.sort(key=lambda item: item[2]["total_pnl"], reverse=True)
                    frontier = self._select_diverse_frontier(size_survivors)

            # Greedy beam expansion for sizes beyond the exhaustive cutoff.
            for size in range(exhaustive_max_size + 1, self.max_combo_size + 1):
                if not frontier:
                    break
                candidates = []
                seen_keys = set()
                for combo_names, mask, _ in frontier:
                    used = set(combo_names)
                    for name in names:
                        if name in used:
                            continue
                        new_combo = combo_names + (name,)
                        key = tuple(sorted(new_combo))
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        combined_mask = mask & pool[name]
                        row = self._evaluate(new_combo, size, combined_mask, direction)
                        if row is None:
                            continue
                        rows.append(row)
                        candidates.append((new_combo, combined_mask, row))

                candidates.sort(key=lambda item: item[2]["total_pnl"], reverse=True)
                frontier = self._select_diverse_frontier(candidates)

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("total_pnl", ascending=False).reset_index(drop=True)
        return result

    def print_report(self):
        result = self.run()
        console = Console(width=220)

        search_note = (
            f"sizes 1-2 exhaustive, sizes 3-{self.max_combo_size} via greedy beam search "
            f"(top {self.beam_width} carried forward each step)"
            if self.max_combo_size > 2
            else f"sizes {self.min_combo_size}-{self.max_combo_size} exhaustive"
        )
        console.print(
            f"[bold]COMBO BACKTEST[/bold] - long pool {self.stats['long_pool_size']}, "
            f"short pool {self.stats['short_pool_size']} conditions/signals, combo {search_note}, "
            f"min {self.min_fires} fires kept"
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

        # report.json / the dashboard get a larger (but still capped, see
        # ReportExporter's combo_json_top_n) slice of this same ranked list -
        # the console only needs a readable top slice, not a multi-thousand-row dump.
        shown = profitable.head(self.console_top_n)
        title = f"Top {len(shown)} of {len(profitable)} profitable combinations, best PnL first"
        if len(profitable) > len(shown):
            title += " (see report.json / dashboard for more)"

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


def diverse_top_combos(result, size=5, n=10):
    """The plain "top N by total PnL" list is usually just one dominant
    condition (e.g. swing_low_at_pivot>median) wearing N different extra
    filters - not N genuinely different strategies. This picks the best `n`
    combos - restricted to combo size(s) `size` (an int for one exact size, or
    a list/tuple to consider several sizes together) - such that NO TWO
    selected combos share even one underlying condition. It walks all
    candidates best-PnL-first (across every allowed size at once) and skips
    any that overlap a condition already used by a combo already picked. What
    's left is `n` mutually independent strategies, each still the best
    available given everything picked before it. Allowing multiple sizes
    gives the greedy walk far more candidates to pick from, so it can reach a
    larger `n` than any single size's pool of truly disjoint combos allows."""
    sizes = [size] if isinstance(size, int) else list(size)
    candidates = result[result["size"].isin(sizes)].sort_values("total_pnl", ascending=False)

    selected_rows = []
    used_conditions = set()
    for _, row in candidates.iterrows():
        conditions = set(row["combo"].split(" AND "))
        if conditions & used_conditions:
            continue
        selected_rows.append(row)
        used_conditions |= conditions
        if len(selected_rows) >= n:
            break

    return pd.DataFrame(selected_rows).reset_index(drop=True)
