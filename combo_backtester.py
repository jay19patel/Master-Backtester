"""ComboBacktester: exhaustively tests combinations of every size (1, 1+2,
1+2+3, ...) of indicator conditions + PriceActionEngine sig_* signals
together, and backtests each combination for REAL money (win rate, PnL) using
the exact same realistic engine as Backtester - entry at next open,
stop/target bracket walked bar-by-bar, risk-based position sizing, fees, no
overlapping trades.

Directional tagging: every indicator column gets an automatic long/short pair
by comparing its value to its own trailing rolling median (`value >
median` -> long pool, `value < median` -> short pool). The median is computed
over a strictly causal rolling window (`condition_window`, default 100 bars),
so it only ever looks at past data - no lookahead. This uniform rule covers
oscillators, moving averages, volume/volatility measures, interaction features
like trend_volume, anything - without hand-curating a condition per indicator.
Every sig_* price-action signal is split into a long half (`col == 1`) and a
short half (`col == -1`).

Quality filter: a condition/signal that is constant, entirely NaN, or never
fires at all in a given direction is dropped from that direction's pool before
the search starts - it can only ever produce empty/duplicate combinations, so
keeping it around just wastes search space. A condition whose own solo fire
count is already below `min_fires` is dropped too: ANDing it with anything
else can only keep the same fire count or shrink it, so no combination built
on top of it could ever clear the threshold either - dropping it loses no
result, only search space.

Exhaustive, not approximate - Apriori-style level-wise search: with a pool of
100+ conditions, generating every raw combination the plain way (C(pool, k)
for every k) explodes fast past k=3-4. Instead, size-k combinations are only
built by EXTENDING size-(k-1) combinations that already cleared `min_fires`.
This loses zero results: ANDing in one more condition can only keep the fire
count the same or shrink it (never grow it), so any k-combo that clears
min_fires necessarily has every one of its (k-1)-subsets also clearing
min_fires - meaning each of those subsets is guaranteed to already be sitting
in the previous level, ready to be extended. Nothing reachable is skipped,
unlike a PnL-ranked beam search. The search self-terminates the moment a
level produces zero survivors (no combo of that size can clear the fire
threshold anymore) - `max_combo_size` is just a safety ceiling, not a target
every run is expected to reach. `max_candidates_per_level` is a second safety
valve for when min_fires isn't yet pruning much at a given size (common on
large datasets, since a low-ish min_fires can still be cleared by many
independent conditions ANDed together) - if a level would blow past it, a
uniform RANDOM sample is kept instead (never "top by fire count": the
highest-firing combos are the least selective/predictive ones, so ranking by
fires would systematically discard the rarer combos most likely to carry
genuine edge). The console/report say plainly whenever this happens.

Performance: a cheap vectorized fire-count prefilter runs for every
combination first (boolean AND across numpy arrays, one incremental AND per
extra condition - not recomputed from scratch); only combinations clearing
`min_fires` go through the much more expensive bar-by-bar trade simulation.
That simulation step is spread across a process pool (`n_workers`, default =
cpu_count - 1) - each worker is initialized once with the OHLC arrays and
condition pools (not a full DataFrame), then just receives which
condition-name tuples to evaluate, so the expensive part of the search runs
on every core instead of one.
"""

import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from backtester import simulate_trades

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Populated once per worker process by _init_worker - avoids re-pickling the
# OHLC arrays / condition pools on every single task.
_WORKER_STATE = {}


def _init_worker(pools_by_direction, ohlc, params):
    _WORKER_STATE["pools"] = pools_by_direction
    _WORKER_STATE["ohlc"] = ohlc
    _WORKER_STATE["params"] = params


def _evaluate_batch(tasks):
    """tasks: list of (direction, combo_names_tuple). Returns (rows,
    cleared_min_fires_count, simulated_count) for this batch."""
    pools = _WORKER_STATE["pools"]
    ohlc = _WORKER_STATE["ohlc"]
    params = _WORKER_STATE["params"]
    min_fires = params["min_fires"]

    rows = []
    cleared = 0
    simulated = 0

    for direction, combo in tasks:
        pool = pools[direction]
        mask = pool[combo[0]]
        for name in combo[1:]:
            mask = mask & pool[name]

        fires = int(np.count_nonzero(mask))
        if fires < min_fires:
            continue
        cleared += 1

        direction_array = np.where(mask, direction, 0)
        trades, final_equity = simulate_trades(
            direction_array,
            ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"],
            params["initial_capital"], params["risk_per_trade_pct"], params["stop_loss_pct"],
            params["take_profit_pct"], params["max_hold_bars"], params["fee_pct"],
        )
        simulated += 1

        n_trades = len(trades)
        if n_trades < min_fires:
            continue

        wins = [t for t in trades if t["pnl"] > 0]
        total_pnl = final_equity - params["initial_capital"]
        rows.append({
            "combo": " AND ".join(combo),
            "size": len(combo),
            "fires": fires,
            "trades": n_trades,
            "win_rate_pct": round(len(wins) / n_trades * 100, 1),
            "final_equity": round(final_equity, 2),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / params["initial_capital"] * 100, 1),
        })

    return rows, cleared, simulated


class ComboBacktester:
    """Exhaustively searches indicator-condition + price-action-signal
    combinations and backtests each one for real PnL, in parallel.

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
        max_combo_size=8,
        min_fires=15,
        console_top_n=20,
        condition_window=100,
        n_workers=None,
        max_candidates_per_level=5_000,
    ):
        self.df = df
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_bars = max_hold_bars
        self.fee_pct = fee_pct
        self.min_combo_size = min_combo_size
        self.max_combo_size = max_combo_size
        self.min_fires = min_fires
        self.console_top_n = console_top_n
        self.condition_window = condition_window
        self.n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
        # Bounds peak memory for a level to roughly max_candidates_per_level *
        # len(df) bytes (one boolean mask per surviving combo) - a level only
        # gets this big if min_fires is too low relative to dataset size for
        # the Apriori pruning to bite yet, or conditions are highly
        # correlated. Kept combos are a random sample (not top-by-fires - see
        # module docstring); print_report()/the console clearly say when a
        # level was capped or truncated.
        self.max_candidates_per_level = max_candidates_per_level
        self.stats = {}
        self.max_size_reached = {}
        self.capped_levels = []
        self.truncated_levels = []

    # ------------------------------------------------------------------
    # Build the long/short condition + signal pools
    # ------------------------------------------------------------------
    def _build_pools(self):
        """Every indicator column gets an automatic long/short condition
        pair: is it currently above or below its own trailing rolling median?
        Every sig_* price-action signal is split into its long (`== 1`) and
        short (`== -1`) half. A condition/signal that is constant, never
        fires, or fires fewer than `min_fires` times on its own is dropped -
        see the module docstring for why that loses no reachable result."""
        df = self.df

        # swing_high_at_pivot / swing_low_at_pivot are built with a CENTERED
        # rolling window (see PriceActionEngine.add_swings) - knowing bar i is
        # a pivot requires seeing `swing_right` bars AFTER it, so using them
        # directly as a same-bar condition is lookahead bias (real-time you
        # can't know this yet). The causal equivalents (last_swing_high/
        # last_swing_low price levels, bars_since_swing_high/_low freshness)
        # are NOT excluded, since those ARE properly lagged and safe to trade on.
        excluded = set(OHLCV_COLUMNS) | {"swing_high_at_pivot", "swing_low_at_pivot"}
        indicator_cols = [
            c
            for c in df.columns
            if c not in excluded and not c.startswith("sig_") and pd.api.types.is_numeric_dtype(df[c])
        ]

        long_pool = {}
        short_pool = {}
        dropped = []

        for col in indicator_cols:
            series = df[col]
            if series.nunique(dropna=True) <= 1:
                dropped.append(col)
                continue
            rolling_median = series.rolling(self.condition_window, min_periods=self.condition_window).median()
            above = (series > rolling_median).to_numpy()
            below = (series < rolling_median).to_numpy()
            kept = False
            if int(np.count_nonzero(above)) >= self.min_fires:
                long_pool[f"{col}>median"] = above
                kept = True
            if int(np.count_nonzero(below)) >= self.min_fires:
                short_pool[f"{col}<median"] = below
                kept = True
            if not kept:
                dropped.append(col)

        for col in [c for c in df.columns if c.startswith("sig_")]:
            arr = df[col].to_numpy()
            name = col.replace("sig_", "")
            long_mask = arr == 1
            short_mask = arr == -1
            kept = False
            if int(np.count_nonzero(long_mask)) >= self.min_fires:
                long_pool[f"{name}(L)"] = long_mask
                kept = True
            if int(np.count_nonzero(short_mask)) >= self.min_fires:
                short_pool[f"{name}(S)"] = short_mask
                kept = True
            if not kept:
                dropped.append(col)

        self.dropped_columns = dropped
        return long_pool, short_pool

    # ------------------------------------------------------------------
    # Search + backtest
    # ------------------------------------------------------------------
    def _generate_tasks(self, pools_by_direction):
        """Apriori-style level-wise search: build every combination of every
        size, but only by EXTENDING combinations from the previous size that
        already cleared min_fires. This is exact, not approximate - ANDing in
        one more condition can only keep the fire count the same or shrink it
        (never grow it), so any k-combo that clears min_fires necessarily has
        every one of its (k-1)-subsets also clearing min_fires, meaning every
        one of those subsets is guaranteed to already be present at the
        previous level. Extending every previous-level survivor with every
        remaining pool item is therefore guaranteed to reach it - nothing is
        skipped, unlike a PnL-ranked beam search. The search self-terminates
        the moment a level produces zero survivors (no combo of that size can
        clear the threshold), so `max_combo_size` is just a safety ceiling,
        not a target every run is expected to reach."""
        tasks = []
        tested = 0
        cleared = 0
        max_size_reached = {}
        capped_levels = []
        truncated_levels = []
        # If a level's raw candidate count blows past this before the top-K
        # heap even gets a chance to matter, stop scanning further parents at
        # that level instead of grinding through millions of near-useless
        # AND+count_nonzero calls. This only bites when min_fires is too low
        # relative to dataset size for the pruning to be effective yet -
        # logged clearly (truncated_levels), never silent.
        max_examined_per_level = max(self.max_candidates_per_level * 20, 200_000)

        for direction, pool in pools_by_direction.items():
            names_list = list(pool.keys())

            # Size 1: every pool entry already clears min_fires by construction (_build_pools).
            current = [((name,), pool[name]) for name in names_list]
            if self.min_combo_size <= 1:
                tasks.extend((direction, names) for names, _ in current)
            tested += len(current)
            cleared += len(current)
            size_reached = 1

            size = 1
            while current and size < self.max_combo_size:
                size += 1
                seen = set()
                # Reservoir sample (Algorithm R), bounded to max_candidates_per_level -
                # NOT "top by fire count": the highest-firing combos are the LEAST
                # selective/predictive ones (an almost-always-true condition has no
                # edge), so ranking by fires when capping would systematically throw
                # away the rarer, potentially profitable combos and keep only noise.
                # A uniform random sample stays representative of the whole survivor
                # set instead. Size 2 is never capped - C(pool_size, 2) is inherently
                # small (a few thousand at most), so it stays exactly as exhaustive
                # as the plain pairs-only search this replaced; only size 3+ can
                # actually blow up combinatorially.
                level_cap = float("inf") if size <= 2 else self.max_candidates_per_level
                reservoir = []
                examined = 0
                survivors_seen = 0
                truncated = False

                for combo_names, mask in current:
                    if truncated:
                        break
                    used = set(combo_names)
                    for name in names_list:
                        if name in used:
                            continue
                        new_names = combo_names + (name,)
                        key = frozenset(new_names)
                        if key in seen:
                            continue
                        seen.add(key)
                        examined += 1
                        tested += 1
                        if examined > max_examined_per_level:
                            truncated = True
                            break

                        new_mask = mask & pool[name]
                        fires = int(np.count_nonzero(new_mask))
                        if fires < self.min_fires:
                            continue
                        survivors_seen += 1
                        if len(reservoir) < level_cap:
                            reservoir.append((new_names, new_mask))
                        else:
                            j = random.randint(0, survivors_seen - 1)
                            if j < level_cap:
                                reservoir[j] = (new_names, new_mask)

                next_level = reservoir
                if truncated:
                    truncated_levels.append((direction, size, examined))
                if survivors_seen > len(next_level):
                    capped_levels.append((direction, size, survivors_seen, len(next_level)))

                cleared += len(next_level)
                if size >= self.min_combo_size:
                    tasks.extend((direction, names) for names, _ in next_level)
                if next_level:
                    size_reached = size
                current = next_level

            max_size_reached[direction] = size_reached

        self.max_size_reached = max_size_reached
        self.capped_levels = capped_levels
        self.truncated_levels = truncated_levels
        return tasks, tested, cleared

    def run(self):
        """Backtest every qualifying combination, at every size, exhaustively
        and in parallel. Returns a DataFrame (both winning and losing combos
        that cleared min_fires), best PnL first."""
        long_pool, short_pool = self._build_pools()
        pools_by_direction = {1: long_pool, -1: short_pool}

        tasks, tested, cleared_prefilter = self._generate_tasks(pools_by_direction)

        self.stats = {
            "long_pool_size": len(long_pool),
            "short_pool_size": len(short_pool),
            "dropped_columns": len(self.dropped_columns),
            "combos_tested": tested,
            "combos_cleared_min_fires": cleared_prefilter,
            "combos_simulated": 0,
        }

        if not tasks:
            return pd.DataFrame()

        ohlc = {
            "open": self.df["Open"].to_numpy(),
            "high": self.df["High"].to_numpy(),
            "low": self.df["Low"].to_numpy(),
            "close": self.df["Close"].to_numpy(),
        }
        params = {
            "initial_capital": self.initial_capital,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_hold_bars": self.max_hold_bars,
            "fee_pct": self.fee_pct,
            "min_fires": self.min_fires,
        }

        chunk_size = max(50, len(tasks) // (self.n_workers * 8))
        chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]

        rows = []
        with ProcessPoolExecutor(
            max_workers=self.n_workers, initializer=_init_worker, initargs=(pools_by_direction, ohlc, params)
        ) as executor:
            for batch_rows, _cleared, simulated in executor.map(_evaluate_batch, chunks):
                # _cleared is ignored here - every task handed to the worker already
                # cleared min_fires during _generate_tasks, so re-adding it would
                # double-count combos_cleared_min_fires.
                rows.extend(batch_rows)
                self.stats["combos_simulated"] += simulated

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("total_pnl", ascending=False).reset_index(drop=True)
        return result

    def print_report(self):
        result = self.run()
        console = Console(width=220)

        console.print(
            f"[bold]COMBO BACKTEST[/bold] - long pool {self.stats['long_pool_size']}, "
            f"short pool {self.stats['short_pool_size']} conditions/signals "
            f"({self.stats['dropped_columns']} dropped - constant, never fired, or below {self.min_fires} fires), "
            f"exhaustive Apriori search sizes {self.min_combo_size}-{self.max_combo_size} "
            f"(ceiling), {self.n_workers} parallel workers"
        )
        console.print(
            f"Search funnel: {self.stats['combos_tested']:,} tested -> "
            f"{self.stats['combos_cleared_min_fires']:,} cleared min fires -> "
            f"{self.stats['combos_simulated']:,} simulated"
        )
        for direction, label in ((1, "long"), (-1, "short")):
            reached = self.max_size_reached.get(direction)
            if reached is not None:
                note = " (hit the max_combo_size ceiling)" if reached == self.max_combo_size else " (search died out naturally)"
                console.print(f"  {label} pool: largest surviving combo size = {reached}{note}")
        for direction, size, original_count, kept_count in self.capped_levels:
            label = "long" if direction == 1 else "short"
            console.print(
                f"  [yellow]capped[/yellow] {label} size-{size}: {original_count:,} candidates cleared min_fires, "
                f"kept a random sample of {kept_count:,} (max_candidates_per_level={self.max_candidates_per_level:,})"
            )
        for direction, size, examined in self.truncated_levels:
            label = "long" if direction == 1 else "short"
            console.print(
                f"  [red]truncated[/red] {label} size-{size}: stopped after examining {examined:,} raw candidates "
                f"without finishing - min_fires={self.min_fires} isn't pruning fast enough at this size for this "
                f"dataset (not fully exhaustive at/beyond this size - consider raising min_fires)"
            )

        if result.empty:
            console.print("No combination cleared the minimum fire count.")
            return result

        profitable = result[result["total_pnl"] > 0].reset_index(drop=True)
        console.print(
            f"{len(profitable)} of {len(result)} simulated combinations were profitable "
            f"on a ${self.initial_capital:.0f} account - ranked by total PnL"
        )

        if profitable.empty:
            console.print("None were profitable under these realistic assumptions.")
            return profitable

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

        # One row per distinct combo size actually present, best PnL first within
        # that size - lets you see the strongest pattern at EACH complexity level
        # (3-condition, 4-condition, ... up to whatever size the search reached),
        # not just the overall top N pooled together (which can otherwise be
        # dominated by one or two sizes).
        best_idx = profitable.groupby("size")["total_pnl"].idxmax()
        best_per_size = profitable.loc[best_idx].sort_values("size").reset_index(drop=True)

        size_table = Table(title="Best combo at each size", show_lines=False)
        size_table.add_column("size", justify="right", style="dim")
        size_table.add_column("Combo", style="bold")
        size_table.add_column("fires", justify="right")
        size_table.add_column("trades", justify="right")
        size_table.add_column("win_rate%", justify="right")
        size_table.add_column("final_$", justify="right")
        size_table.add_column("total_pnl", justify="right")
        size_table.add_column("return%", justify="right")

        for _, row in best_per_size.iterrows():
            size_table.add_row(
                str(row["size"]),
                row["combo"],
                str(row["fires"]),
                str(row["trades"]),
                f"{row['win_rate_pct']:.1f}",
                f"{row['final_equity']:.2f}",
                f"[green]{row['total_pnl']:+.2f}[/green]",
                f"[green]{row['return_pct']:+.1f}[/green]",
            )
        console.print(size_table)

        # Every row here already cleared min_fires (>= self.min_fires trades),
        # so a high win rate isn't just noise from a handful of lucky trades.
        top_winrate = profitable.sort_values(
            ["win_rate_pct", "total_pnl"], ascending=[False, False]
        ).head(5).reset_index(drop=True)

        winrate_table = Table(title="Top 5 by win rate", show_lines=False)
        winrate_table.add_column("#", justify="right", style="dim")
        winrate_table.add_column("Combo", style="bold")
        winrate_table.add_column("size", justify="right")
        winrate_table.add_column("fires", justify="right")
        winrate_table.add_column("trades", justify="right")
        winrate_table.add_column("win_rate%", justify="right")
        winrate_table.add_column("final_$", justify="right")
        winrate_table.add_column("total_pnl", justify="right")
        winrate_table.add_column("return%", justify="right")

        for i, row in top_winrate.iterrows():
            winrate_table.add_row(
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
        console.print(winrate_table)

        best = profitable.iloc[0]
        console.print(
            f"\n[bold]Best combo overall:[/bold] {best['combo']} (size {best['size']}) -> "
            f"${self.initial_capital:.0f} became ${best['final_equity']:.2f} ({best['return_pct']:+.1f}%) "
            f"over {best['trades']} trades, {best['win_rate_pct']:.1f}% win rate."
        )
        return profitable
