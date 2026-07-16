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
short half (`col == -1`). Every result row carries its `direction` (Long/Short)
explicitly - not just implied by the combo text.

Quality filter: a condition/signal that is constant, entirely NaN, or never
fires at all in a given direction is dropped from that direction's pool before
the search starts - it can only ever produce empty/duplicate combinations, so
keeping it around just wastes search space. A condition whose own solo fire
count is already below `min_fires` is dropped too: ANDing it with anything
else can only keep the same fire count or shrink it, so no combination built
on top of it could ever clear the threshold either - dropping it loses no
result, only search space.

Exhaustive, not approximate, NEVER randomly sampled - Apriori-style level-wise
search: with a pool of 100+ conditions, generating every raw combination the
plain way (C(pool, k) for every k) explodes fast past k=3-4. Instead, size-k
combinations are only built by EXTENDING size-(k-1) combinations that already
cleared `min_fires`. This loses zero results: ANDing in one more condition can
only keep the fire count the same or shrink it (never grow it), so any k-combo
that clears min_fires necessarily has every one of its (k-1)-subsets also
clearing min_fires - meaning each of those subsets is guaranteed to already be
sitting in the previous level, ready to be extended. Nothing reachable is
skipped.

No sampling, ever - either a level completes 100% or the search stops clean:
if a level's candidate count would exceed `max_raw_candidates_per_level`
(before filtering) or its survivor count would exceed
`max_survivors_per_level` (after filtering), the search STOPS at the last
fully-completed size for that direction - it does NOT keep a random/partial
slice and pretend to continue. Every size that IS reported is a complete,
exhaustive count; sizes beyond the stop point are simply not explored. This is
reported plainly (`stopped_levels`) so you know exactly how far the search
went and why.

Performance - full CPU parallelism, not just for simulation: both the
fire-count prefilter (cheap boolean AND per candidate) AND the bar-by-bar
trade simulation (expensive) are distributed across a process pool
(`n_workers`, default = every CPU core). Each worker is initialized once with
the OHLC arrays and condition pools (not a full DataFrame, and masks are never
shipped between processes - only lightweight condition-name tuples travel over
IPC), so the search scales across every core for both phases. A live progress
bar (rich) shows candidates examined per level and combos simulated, so
there's always a clear sense of how far a run has gotten.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from backtester import simulate_trades

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Populated once per worker process by _init_worker - avoids re-pickling the
# OHLC arrays / condition pools on every single task. Masks are NEVER shipped
# between processes - only condition-name tuples do, so IPC payloads stay tiny
# even at large combo sizes.
_WORKER_STATE = {}


def _init_worker(pools_by_direction, ohlc, params):
    _WORKER_STATE["pools"] = pools_by_direction
    _WORKER_STATE["ohlc"] = ohlc
    _WORKER_STATE["params"] = params


def _mask_for(pool, combo):
    mask = pool[combo[0]]
    for name in combo[1:]:
        mask = mask & pool[name]
    return mask


def _filter_batch(tasks):
    """Generation-phase worker: tasks is a list of (direction, combo_names).
    Recomputes each combo's mask from the resident pool and checks min_fires -
    this is the cheap prefilter, run in parallel across every core. Returns
    (survivors, examined_count) where survivors is [(direction, combo_names,
    fires), ...] for combos that cleared min_fires."""
    pools = _WORKER_STATE["pools"]
    min_fires = _WORKER_STATE["params"]["min_fires"]

    survivors = []
    for direction, combo in tasks:
        fires = int(np.count_nonzero(_mask_for(pools[direction], combo)))
        if fires >= min_fires:
            survivors.append((direction, combo, fires))
    return survivors, len(tasks)


def _evaluate_batch(tasks):
    """Simulation-phase worker: tasks is a list of (direction, combo_names),
    every one already confirmed (during generation) to clear min_fires.
    Returns (rows, simulated_count) for this batch."""
    pools = _WORKER_STATE["pools"]
    ohlc = _WORKER_STATE["ohlc"]
    params = _WORKER_STATE["params"]
    min_fires = params["min_fires"]

    rows = []
    simulated = 0

    for direction, combo in tasks:
        mask = _mask_for(pools[direction], combo)
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
            "direction": "Long" if direction == 1 else "Short",
            "combo": " AND ".join(combo),
            "conditions": list(combo),
            "size": len(combo),
            "fires": int(np.count_nonzero(mask)),
            "trades": n_trades,
            "win_rate_pct": round(len(wins) / n_trades * 100, 1),
            "final_equity": round(final_equity, 2),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / params["initial_capital"] * 100, 1),
        })

    return rows, simulated


class ComboBacktester:
    """Exhaustively searches indicator-condition + price-action-signal
    combinations and backtests each one for real PnL, in parallel, with no
    random sampling.

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
        max_raw_candidates_per_level=20_000_000,
        max_survivors_per_level=2_000_000,
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
        # Every core, not cpu_count-1: this machine's other work can wait
        # while a search runs - maximize throughput, not responsiveness.
        self.n_workers = n_workers or os.cpu_count() or 1
        # Two independent safety ceilings, checked BEFORE any sampling would
        # ever be needed: max_raw_candidates_per_level bounds how many
        # (parent, extra-condition) pairs get generated+deduped+checked at a
        # level (the pre-filter cost); max_survivors_per_level bounds how many
        # confirmed combos get carried into the next level / final simulation
        # (the expensive cost). If EITHER would be exceeded, the search stops
        # cleanly at the last fully-completed size for that direction and
        # says so - it never keeps a partial or random slice.
        self.max_raw_candidates_per_level = max_raw_candidates_per_level
        self.max_survivors_per_level = max_survivors_per_level
        self.stats = {}
        self.max_size_reached = {}
        self.stopped_levels = []

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
    def _generate_tasks(self, pools_by_direction, executor, progress):
        """Apriori-style level-wise search, parallelized across every core.
        At each level: build the raw (parent, extra-condition) candidate list
        by extending every previous-level survivor, dedup it, then farm the
        fire-count check out to the process pool. If the raw candidate count
        or the survivor count would blow past its ceiling, stop THIS
        direction here (last complete size is size-1 fewer) - never sample."""
        tasks = []
        tested = 0
        cleared = 0
        max_size_reached = {}
        stopped_levels = []

        for direction, pool in pools_by_direction.items():
            names_list = list(pool.keys())
            label = "long" if direction == 1 else "short"

            # Size 1: every pool entry already clears min_fires by construction (_build_pools).
            current_combos = [(name,) for name in names_list]
            if self.min_combo_size <= 1:
                tasks.extend((direction, c) for c in current_combos)
            tested += len(current_combos)
            cleared += len(current_combos)
            size_reached = 1

            size = 1
            while current_combos and size < self.max_combo_size:
                size += 1

                seen = set()
                raw_candidates = []
                for combo_names in current_combos:
                    used = set(combo_names)
                    for name in names_list:
                        if name in used:
                            continue
                        new_names = combo_names + (name,)
                        key = frozenset(new_names)
                        if key in seen:
                            continue
                        seen.add(key)
                        raw_candidates.append((direction, new_names))

                raw_count = len(raw_candidates)
                if raw_count > self.max_raw_candidates_per_level:
                    stopped_levels.append((direction, size, "raw_candidates", raw_count, self.max_raw_candidates_per_level))
                    current_combos = []
                    break

                tested += raw_count
                chunk_size = max(200, raw_count // (self.n_workers * 8) or 1)
                chunks = [raw_candidates[i : i + chunk_size] for i in range(0, raw_count, chunk_size)]

                task_id = progress.add_task(f"[{label}] size {size}: filtering", total=raw_count)
                survivors = []
                for batch_survivors, examined in executor.map(_filter_batch, chunks):
                    survivors.extend(batch_survivors)
                    progress.update(task_id, advance=examined)
                progress.remove_task(task_id)

                if len(survivors) > self.max_survivors_per_level:
                    stopped_levels.append((direction, size, "survivors", len(survivors), self.max_survivors_per_level))
                    current_combos = []
                    break

                cleared += len(survivors)
                next_combos = [combo_names for (_, combo_names, _fires) in survivors]
                if size >= self.min_combo_size:
                    tasks.extend((direction, combo_names) for combo_names in next_combos)
                if next_combos:
                    size_reached = size
                current_combos = next_combos

            max_size_reached[direction] = size_reached

        self.max_size_reached = max_size_reached
        self.stopped_levels = stopped_levels
        return tasks, tested, cleared

    def run(self):
        """Backtest every qualifying combination, at every size, exhaustively
        and in parallel - no random sampling. Returns a DataFrame (both
        winning and losing combos that cleared min_fires), best PnL first."""
        long_pool, short_pool = self._build_pools()
        pools_by_direction = {1: long_pool, -1: short_pool}

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

        console = Console(width=220)
        progress_columns = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        rows = []
        with Progress(*progress_columns, console=console) as progress, ProcessPoolExecutor(
            max_workers=self.n_workers, initializer=_init_worker, initargs=(pools_by_direction, ohlc, params)
        ) as executor:
            tasks, tested, cleared_prefilter = self._generate_tasks(pools_by_direction, executor, progress)

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

            chunk_size = max(50, len(tasks) // (self.n_workers * 8))
            chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]

            task_id = progress.add_task("Simulating trades", total=len(tasks))
            for chunk, (batch_rows, simulated) in zip(chunks, executor.map(_evaluate_batch, chunks)):
                rows.extend(batch_rows)
                self.stats["combos_simulated"] += simulated
                progress.update(task_id, advance=len(chunk))

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
            f"(ceiling), {self.n_workers} parallel workers (all cores)"
        )
        console.print(
            f"Search funnel: {self.stats['combos_tested']:,} tested -> "
            f"{self.stats['combos_cleared_min_fires']:,} cleared min fires -> "
            f"{self.stats['combos_simulated']:,} simulated"
        )
        for direction, label in ((1, "long"), (-1, "short")):
            reached = self.max_size_reached.get(direction)
            if reached is not None:
                note = " (hit the max_combo_size ceiling)" if reached == self.max_combo_size else " (search died out naturally - no combo of the next size clears min_fires)"
                console.print(f"  {label} pool: fully exhaustive through size {reached}{note}")
        for direction, size, reason, count, limit in self.stopped_levels:
            label = "long" if direction == 1 else "short"
            what = "raw candidates to check" if reason == "raw_candidates" else "combos cleared min_fires"
            console.print(
                f"  [red]stopped[/red] {label} at size {size}: would need to process {count:,} {what}, "
                f"over the {limit:,} ceiling - NOT sampled or truncated, search simply stops here so every "
                f"size actually reported stays 100% exhaustive. Raise the relevant limit, raise min_fires, "
                f"or lower max_combo_size to go further."
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

        table = self._make_table(title, shown, numbered=True)
        console.print(table)

        # One row per distinct combo size actually present, best PnL first within
        # that size - lets you see the strongest pattern at EACH complexity level
        # (3-condition, 4-condition, ... up to whatever size the search reached),
        # not just the overall top N pooled together (which can otherwise be
        # dominated by one or two sizes).
        best_idx = profitable.groupby("size")["total_pnl"].idxmax()
        best_per_size = profitable.loc[best_idx].sort_values("size").reset_index(drop=True)
        console.print(self._make_table("Best combo at each size", best_per_size, numbered=False, size_col=True))

        # Every row here already cleared min_fires (>= self.min_fires trades),
        # so a high win rate isn't just noise from a handful of lucky trades.
        top_winrate = profitable.sort_values(
            ["win_rate_pct", "total_pnl"], ascending=[False, False]
        ).head(5).reset_index(drop=True)
        console.print(self._make_table("Top 5 by win rate", top_winrate, numbered=True))

        best = profitable.iloc[0]
        console.print(
            f"\n[bold]Best combo overall:[/bold] [{best['direction']}] {best['combo']} (size {best['size']}) -> "
            f"${self.initial_capital:.0f} became ${best['final_equity']:.2f} ({best['return_pct']:+.1f}%) "
            f"over {best['trades']} trades, {best['win_rate_pct']:.1f}% win rate."
        )
        return profitable

    @staticmethod
    def _make_table(title, rows_df, numbered=True, size_col=True):
        table = Table(title=title, show_lines=False)
        if numbered:
            table.add_column("#", justify="right", style="dim")
        if size_col:
            table.add_column("size", justify="right", style="dim")
        table.add_column("Dir", style="bold")
        table.add_column("Combo", style="bold")
        table.add_column("fires", justify="right")
        table.add_column("trades", justify="right")
        table.add_column("win_rate%", justify="right")
        table.add_column("final_$", justify="right")
        table.add_column("total_pnl", justify="right")
        table.add_column("return%", justify="right")

        for i, row in rows_df.iterrows():
            dir_style = "green" if row["direction"] == "Long" else "red"
            cells = []
            if numbered:
                cells.append(str(i + 1))
            if size_col:
                cells.append(str(row["size"]))
            cells.extend([
                f"[{dir_style}]{row['direction']}[/{dir_style}]",
                row["combo"],
                str(row["fires"]),
                str(row["trades"]),
                f"{row['win_rate_pct']:.1f}",
                f"{row['final_equity']:.2f}",
                f"[green]{row['total_pnl']:+.2f}[/green]",
                f"[green]{row['return_pct']:+.1f}[/green]",
            ])
            table.add_row(*cells)
        return table
