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

Exhaustive where possible, quality-guided (never random) where not -
Apriori-style level-wise search: size-k combinations are only built by
EXTENDING size-(k-1) combinations that already cleared `min_fires` (ANDing in
one more condition can only shrink the fire count, never grow it, so this
misses nothing a plain C(pool, k) enumeration would find - it's just far
cheaper to compute). With a pool of 100+ conditions this still explodes past
size 4-5, though: every size-1 condition is simulated immediately (not
deferred) to get each one's own real standalone total_pnl - this becomes a
per-condition quality score. When a level's survivor count would exceed
`max_survivors_per_level`, instead of a random reservoir OR simply stopping,
the survivors are ranked by the SUM of their member conditions' quality scores
and only the top `max_survivors_per_level` continue - so the combos that keep
getting explored at deeper sizes are the ones built from conditions with
already-proven, individually-real edge, not an arbitrary or random slice. This
is reported plainly (`trimmed_levels`) with exactly how many were trimmed at
each size, so it's never mistaken for a complete/exhaustive count.

Performance - full CPU parallelism, for EVERY phase, including candidate
generation itself: the size-1 warm-up simulation, extending survivors into
new candidates, the fire-count prefilter, and the bar-by-bar trade simulation
are all distributed across a process pool (`n_workers`, default = every CPU
core). Two bottlenecks had to be eliminated to get here, both single-threaded
work left running in the main process while workers sat idle at 0% CPU:
(1) building the (parent, extra-condition) candidate list was originally a
plain Python loop in the main process - now each worker extends its own
slice of parent combos directly (one incremental AND per extra condition);
(2) deduping cross-parent duplicates (two different parents reaching the same
child) via `frozenset(combo)` was, at scale (hundreds of millions of raw
candidates), an even bigger single-threaded cost than (1) - eliminated
entirely by giving every pool name a fixed CANONICAL rank and only ever
extending with names ranked after a combo's own highest member (the same
trick `itertools.combinations` uses internally), which makes every
(parent, extra-condition) pair produce a distinct child - no two parents can
ever reach the same k-combo, so there is nothing left to dedup. Each worker
is initialized once with the OHLC arrays, condition pools, and the canonical
name ranks (not a full DataFrame, and masks are never shipped between
processes - only lightweight condition-name tuples travel over IPC). A live
progress bar (rich) shows candidates examined per level and combos simulated,
so there's always a clear sense of how far a run has gotten.
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


def _init_worker(pools_by_direction, ohlc, params, names_list_by_direction, name_rank_by_direction):
    _WORKER_STATE["pools"] = pools_by_direction
    _WORKER_STATE["ohlc"] = ohlc
    _WORKER_STATE["params"] = params
    _WORKER_STATE["names_list"] = names_list_by_direction
    _WORKER_STATE["name_rank"] = name_rank_by_direction


def _mask_for(pool, combo):
    mask = pool[combo[0]]
    for name in combo[1:]:
        mask = mask & pool[name]
    return mask


def _extend_and_filter_batch(args):
    """Generation-phase worker: args is (direction, parent_combos_chunk).
    Every combo is kept in a fixed CANONICAL order (each pool name has a
    permanent rank, assigned once in _generate_tasks) - a parent only ever
    extends with names ranked AFTER its own highest-ranked member. This is the
    same trick itertools.combinations uses internally: it makes every
    (parent, extra-condition) pair produce a distinct child, so no two
    different parents can ever reach the same k-combo - eliminating the
    need for ANY dedup pass (which, at scale, was itself a single-threaded
    bottleneck worse than the candidate generation it replaced). Extends with
    one incremental AND onto the parent's own mask, not recomputed from
    scratch. Returns (survivors, examined_count)."""
    direction, parent_chunk = args
    pool = _WORKER_STATE["pools"][direction]
    min_fires = _WORKER_STATE["params"]["min_fires"]
    rank = _WORKER_STATE["name_rank"][direction]
    names_list = _WORKER_STATE["names_list"][direction]

    survivors = []
    examined = 0
    for combo_names in parent_chunk:
        parent_mask = _mask_for(pool, combo_names)
        start = rank[combo_names[-1]] + 1
        for name in names_list[start:]:
            examined += 1
            fires = int(np.count_nonzero(parent_mask & pool[name]))
            if fires >= min_fires:
                survivors.append((direction, combo_names + (name,), fires))
    return survivors, examined


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
    combinations and backtests each one for real PnL, in parallel. Never
    randomly samples - when a level is too large to fully explore, it keeps
    the combos built from the highest-quality individual conditions.

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
        # max_raw_candidates_per_level is a defensive-only ceiling now (raw
        # candidates at a level = previous survivors * pool size, and
        # survivors are always trimmed to max_survivors_per_level, so this
        # almost never triggers in practice - it only guards the pathological
        # case of an enormous pool with a tiny min_fires making even the FIRST
        # extension explode). max_survivors_per_level is the real, expected-to-
        # trigger lever: how many combos carry forward into the next size when
        # more than that cleared min_fires - see the module docstring for how
        # the kept ones are chosen (quality-ranked, never random).
        self.max_raw_candidates_per_level = max_raw_candidates_per_level
        self.max_survivors_per_level = max_survivors_per_level
        self.stats = {}
        self.max_size_reached = {}
        self.trimmed_levels = []

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
    def _simulate_tasks(self, tasks, executor, progress, description):
        """Runs _evaluate_batch over `tasks` in parallel with a progress bar.
        Returns (rows, simulated_count). Shared by the size-1 warm-up pass and
        the final simulation pass so both report progress the same way."""
        if not tasks:
            return [], 0
        chunk_size = max(50, len(tasks) // (self.n_workers * 8) or 1)
        chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]
        task_id = progress.add_task(description, total=len(tasks))
        rows = []
        simulated = 0
        for chunk, (batch_rows, batch_simulated) in zip(chunks, executor.map(_evaluate_batch, chunks)):
            rows.extend(batch_rows)
            simulated += batch_simulated
            progress.update(task_id, advance=len(chunk))
        progress.remove_task(task_id)
        return rows, simulated

    def _generate_tasks(self, pools_by_direction, names_list_by_direction, name_rank_by_direction, executor, progress):
        """Apriori-style level-wise search, parallelized across every core -
        including candidate generation itself, via a CANONICAL ordering (each
        pool name has a fixed rank; a combo only ever extends with names
        ranked after its own highest member - the same trick
        itertools.combinations uses internally). This makes every
        (parent, extra-condition) pair produce a distinct child, so no two
        parents can ever reach the same k-combo - there is NO dedup step,
        which at scale was itself a single-threaded bottleneck worse than the
        generation loop it followed. Size-1 conditions are simulated
        immediately (not deferred) so each one has a real, own total_pnl to
        use as a quality score. At each later level: farm candidate
        generation + the fire-count check out to the process pool, then - if
        there are more survivors than max_survivors_per_level - keep the ones
        whose member conditions have the highest SUMMED quality score (never
        a random slice) and continue extending from those."""
        tasks = []           # (direction, combo_names) for sizes 2+ awaiting final simulation
        warm_rows = []        # already-simulated size-1 rows (reused directly, not re-simulated)
        tested = 0
        cleared = 0
        simulated_count = 0
        max_size_reached = {}
        trimmed_levels = []

        for direction, pool in pools_by_direction.items():
            names_list = names_list_by_direction[direction]
            name_rank = name_rank_by_direction[direction]
            label = "long" if direction == 1 else "short"

            # Size 1: simulate NOW so every condition has a real quality score
            # (its own standalone total_pnl) to rank deeper combos by later.
            size1_tasks = [(direction, (name,)) for name in names_list]
            size1_rows, size1_simulated = self._simulate_tasks(
                size1_tasks, executor, progress, f"[{label}] size 1: simulating (quality scores)"
            )
            simulated_count += size1_simulated
            tested += len(size1_tasks)
            cleared += len(size1_tasks)

            condition_score = {row["conditions"][0]: row["total_pnl"] for row in size1_rows}
            if self.min_combo_size <= 1:
                warm_rows.extend(size1_rows)

            current_combos = [(name,) for name in names_list]
            size_reached = 1

            def combo_score(combo_names, _scores=condition_score):
                return sum(_scores.get(name, 0.0) for name in combo_names)

            size = 1
            while current_combos and size < self.max_combo_size:
                size += 1

                # Exact count (cheap: one pass over the current frontier) of
                # how many extensions canonical ordering will actually try -
                # each combo only extends with names ranked after its own max.
                pool_size = len(names_list)
                raw_estimate = sum(pool_size - name_rank[c[-1]] - 1 for c in current_combos)
                if raw_estimate > self.max_raw_candidates_per_level:
                    trimmed_levels.append((direction, size, "raw_candidates", raw_estimate, self.max_raw_candidates_per_level, 0))
                    current_combos = []
                    break

                # Distribute candidate generation + fire-count filtering across
                # every core - each worker gets a slice of PARENT combos and
                # extends+filters them independently. No dedup needed (see
                # docstring): canonical ordering means every survivor here is
                # already unique.
                n_chunks = max(1, min(len(current_combos), self.n_workers * 4))
                chunk_size = max(1, -(-len(current_combos) // n_chunks))
                parent_chunks = [current_combos[i : i + chunk_size] for i in range(0, len(current_combos), chunk_size)]
                tasks_for_workers = [(direction, chunk) for chunk in parent_chunks]

                task_id = progress.add_task(f"[{label}] size {size}: extending + filtering", total=raw_estimate)
                survivors = []
                examined_total = 0
                for batch_survivors, examined in executor.map(_extend_and_filter_batch, tasks_for_workers):
                    survivors.extend(batch_survivors)
                    examined_total += examined
                    progress.update(task_id, advance=examined)
                progress.remove_task(task_id)
                tested += examined_total

                original_count = len(survivors)
                if original_count > self.max_survivors_per_level:
                    survivors.sort(key=lambda item: combo_score(item[1]), reverse=True)
                    survivors = survivors[: self.max_survivors_per_level]
                    trimmed_levels.append(
                        (direction, size, "survivors", original_count, self.max_survivors_per_level, len(survivors))
                    )

                cleared += len(survivors)
                next_combos = [combo_names for (_, combo_names, _fires) in survivors]
                if size >= self.min_combo_size:
                    tasks.extend((direction, combo_names) for combo_names in next_combos)
                if next_combos:
                    size_reached = size
                current_combos = next_combos

            max_size_reached[direction] = size_reached

        self.max_size_reached = max_size_reached
        self.trimmed_levels = trimmed_levels
        return tasks, warm_rows, tested, cleared, simulated_count

    def run(self):
        """Backtest every qualifying combination, at every size. Returns a
        DataFrame (both winning and losing combos that cleared min_fires),
        best PnL first."""
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

        # Fixed canonical order per direction, assigned once - see
        # _extend_and_filter_batch's docstring for why this eliminates the
        # need for any dedup pass during the search.
        names_list_by_direction = {d: list(pool.keys()) for d, pool in pools_by_direction.items()}
        name_rank_by_direction = {
            d: {name: i for i, name in enumerate(names)} for d, names in names_list_by_direction.items()
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

        with Progress(*progress_columns, console=console) as progress, ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=_init_worker,
            initargs=(pools_by_direction, ohlc, params, names_list_by_direction, name_rank_by_direction),
        ) as executor:
            tasks, warm_rows, tested, cleared_prefilter, simulated_count = self._generate_tasks(
                pools_by_direction, names_list_by_direction, name_rank_by_direction, executor, progress
            )

            self.stats = {
                "long_pool_size": len(long_pool),
                "short_pool_size": len(short_pool),
                "dropped_columns": len(self.dropped_columns),
                "combos_tested": tested,
                "combos_cleared_min_fires": cleared_prefilter,
                "combos_simulated": simulated_count,
            }

            rows = list(warm_rows)
            if tasks:
                final_rows, final_simulated = self._simulate_tasks(tasks, executor, progress, "Simulating trades")
                rows.extend(final_rows)
                self.stats["combos_simulated"] += final_simulated

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
            f"Apriori search sizes {self.min_combo_size}-{self.max_combo_size} "
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
                console.print(f"  {label} pool: reached size {reached}{note}")
        for direction, size, reason, original_count, limit, kept_count in self.trimmed_levels:
            label = "long" if direction == 1 else "short"
            if reason == "raw_candidates":
                console.print(
                    f"  [red]stopped[/red] {label} at size {size}: {original_count:,} raw candidates to check, "
                    f"over the {limit:,} ceiling - this direction stops here (raise max_raw_candidates_per_level "
                    f"or min_fires to go further)."
                )
            else:
                console.print(
                    f"  [yellow]trimmed[/yellow] {label} size-{size}: {original_count:,} combos cleared min_fires, "
                    f"kept the {kept_count:,} built from the highest-quality conditions (by each condition's own "
                    f"standalone total_pnl) - NOT random, but not exhaustive either at/beyond this size."
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
