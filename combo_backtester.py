"""ComboBacktester: exhaustively tests combinations of every size (1, 1+2,
1+2+3, ...) of indicator conditions + PriceActionEngine sig_* signals, and
backtests each for real PnL using the same engine as Backtester.

Directional tagging: each indicator gets a long/short pair by comparing its
value to its own trailing rolling median (`condition_window`, causal only).
Each sig_* signal splits into long (`== 1`) / short (`== -1`) halves.

Quality filter: constant/all-NaN/never-firing conditions, and any condition
whose solo fire count is already below `min_fires`, are dropped up front -
ANDing more conditions in can only shrink fire count, never grow it, so this
loses no reachable result.

Apriori-style level-wise search: size-k combos only extend size-(k-1) combos
that already cleared `min_fires`. Size-1 conditions are simulated immediately
to get a real standalone `total_pnl` as a quality score. When a level's
survivor count exceeds `max_survivors_per_level`, only the top-scoring combos
(by summed member quality score) continue - never random - reported via
`trimmed_levels` so it's never mistaken for exhaustive.

Performance: every phase (warm-up simulation, candidate extension, fire-count
prefilter, trade simulation) runs across a process pool (`n_workers`). Pool
names get a fixed canonical rank so a combo only ever extends with names
ranked after its own highest member (same trick `itertools.combinations` uses
internally) - this makes every (parent, extra-condition) pair produce a
distinct child, so no dedup pass is needed.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from backtester import simulate_trades

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Populated once per worker by _init_worker - avoids re-pickling OHLC/pools per task.
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
    Extends each parent only with names ranked after its own highest member
    (fixed canonical rank per pool name - same trick itertools.combinations
    uses internally), so no two parents ever reach the same k-combo and no
    dedup pass is needed. Returns (survivors, examined_count)."""
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
    combinations and backtests each for real PnL, in parallel. When a level
    is too large to fully explore, keeps the highest-quality combos - never
    a random sample.

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
        max_survivors_per_level=20_000,
        max_search_seconds=None,
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
        self.n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)  # leave a core free
        # Safety nets: raw-candidate ceiling, survivors-per-level cap, optional
        # time budget. Whichever trips first, combos already found are kept.
        self.max_raw_candidates_per_level = max_raw_candidates_per_level
        self.max_survivors_per_level = max_survivors_per_level
        self.max_search_seconds = max_search_seconds
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

        # swing_high/low_at_pivot use a centered window (needs future bars to
        # confirm a pivot) - lookahead bias, so excluded. Causal equivalents
        # (last_swing_high/low, bars_since_swing_*) are fine and kept.
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
    def _simulate_tasks(self, tasks, executor, progress, description, console=None, heartbeat_seconds=10):
        """Runs _evaluate_batch over `tasks` in parallel with a progress bar.
        Returns (rows, simulated_count). Shared by the size-1 warm-up pass and
        the final simulation pass. If `console` is given, also prints a
        permanent line every `heartbeat_seconds` (useful once redirected to a
        log file, where the live bar shows nothing)."""
        if not tasks:
            return [], 0
        chunk_size = max(50, len(tasks) // (self.n_workers * 8) or 1)
        chunks = [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]
        task_id = progress.add_task(description, total=len(tasks))
        rows = []
        simulated = 0
        done = 0
        start = time.monotonic()
        last_print = start
        for chunk_idx, (chunk, (batch_rows, batch_simulated)) in enumerate(zip(chunks, executor.map(_evaluate_batch, chunks)), start=1):
            rows.extend(batch_rows)
            simulated += batch_simulated
            done += len(chunk)
            progress.update(task_id, advance=len(chunk))
            now = time.monotonic()
            if console is not None and now - last_print >= heartbeat_seconds:
                pct = done / len(tasks) * 100
                console.print(
                    f"    [dim]{description}[/dim]: {done:,}/{len(tasks):,} ({pct:.1f}%), "
                    f"{chunk_idx:,}/{len(chunks):,} chunks, {len(rows):,} profitable-eligible rows so far "
                    f"({now - start:.0f}s elapsed)"
                )
                last_print = now
        progress.remove_task(task_id)
        return rows, simulated

    def _generate_tasks(self, pools_by_direction, names_list_by_direction, name_rank_by_direction, executor, progress, console):
        """Apriori-style level-wise search, parallelized across every core
        including candidate generation. Size-1 conditions are simulated
        immediately so each has a real quality score. At each later level,
        candidate generation + fire-count filtering run on the process pool;
        if survivors exceed max_survivors_per_level, only the top-scoring
        (summed quality score) continue - never a random slice."""
        tasks = []      # (direction, combo_names) for sizes 2+ awaiting final simulation
        warm_rows = []  # already-simulated size-1 rows
        tested = 0
        cleared = 0
        simulated_count = 0
        max_size_reached = {}
        trimmed_levels = []

        for direction, pool in pools_by_direction.items():
            names_list = names_list_by_direction[direction]
            name_rank = name_rank_by_direction[direction]
            label = "long" if direction == 1 else "short"
            direction_start = time.monotonic()  # own time budget per direction

            console.print(f"[dim][{label}][/dim] {len(names_list):,} conditions - simulating size 1 for quality scores...")

            size1_tasks = [(direction, (name,)) for name in names_list]
            size1_rows, size1_simulated = self._simulate_tasks(
                size1_tasks, executor, progress, f"[{label}] size 1: simulating (quality scores)", console=console
            )
            simulated_count += size1_simulated
            tested += len(size1_tasks)
            cleared += len(size1_tasks)
            console.print(
                f"[dim][{label}][/dim] size 1 done: {len(size1_rows):,} of {len(names_list):,} produced a real "
                f"trade ({time.monotonic() - direction_start:.1f}s elapsed)"
            )

            condition_score = {row["conditions"][0]: row["total_pnl"] for row in size1_rows}
            if self.min_combo_size <= 1:
                warm_rows.extend(size1_rows)

            current_combos = [(name,) for name in names_list]
            size_reached = 1

            def combo_score(combo_names, _scores=condition_score):
                return sum(_scores.get(name, 0.0) for name in combo_names)

            size = 1
            while current_combos and size < self.max_combo_size:
                if self.max_search_seconds is not None:
                    elapsed = time.monotonic() - direction_start
                    if elapsed > self.max_search_seconds:
                        trimmed_levels.append((direction, size, "time_budget", 0, self.max_search_seconds, 0))
                        break
                size += 1
                level_start = time.monotonic()

                # Exact count of extensions canonical ordering will try this level.
                pool_size = len(names_list)
                raw_estimate = sum(pool_size - name_rank[c[-1]] - 1 for c in current_combos)
                if raw_estimate > self.max_raw_candidates_per_level:
                    trimmed_levels.append((direction, size, "raw_candidates", raw_estimate, self.max_raw_candidates_per_level, 0))
                    current_combos = []
                    break

                # Each worker extends+filters its own slice of parent combos - no
                # dedup needed (canonical ordering keeps every survivor unique).
                # Chunks target a small fixed work quantum so the time-budget
                # check below can fire mid-level, not just between levels.
                avg_ops_per_parent = max(1.0, raw_estimate / max(1, len(current_combos)))
                target_ops_per_chunk = 50_000
                parents_per_chunk = max(1, int(target_ops_per_chunk / avg_ops_per_parent))
                parent_chunks = [
                    current_combos[i : i + parents_per_chunk] for i in range(0, len(current_combos), parents_per_chunk)
                ]
                tasks_for_workers = [(direction, chunk) for chunk in parent_chunks]

                task_id = progress.add_task(f"[{label}] size {size}: extending + filtering", total=raw_estimate)
                survivors = []
                examined_total = 0
                time_budget_hit_mid_level = False
                total_chunks = len(tasks_for_workers)
                heartbeat_seconds = 10
                last_print = level_start
                for chunk_idx, (batch_survivors, examined) in enumerate(
                    executor.map(_extend_and_filter_batch, tasks_for_workers), start=1
                ):
                    survivors.extend(batch_survivors)
                    examined_total += examined
                    progress.update(task_id, advance=examined)
                    now = time.monotonic()
                    if now - last_print >= heartbeat_seconds:
                        pct = examined_total / raw_estimate * 100 if raw_estimate else 100
                        console.print(
                            f"    [dim][{label}][/dim] size {size}: {examined_total:,}/{raw_estimate:,} examined "
                            f"({pct:.1f}%), {chunk_idx:,}/{total_chunks:,} chunks, {len(survivors):,} cleared so far "
                            f"({now - level_start:.0f}s elapsed)"
                        )
                        last_print = now
                    if self.max_search_seconds is not None and time.monotonic() - direction_start > self.max_search_seconds:
                        time_budget_hit_mid_level = True
                        break
                progress.remove_task(task_id)
                tested += examined_total

                original_count = len(survivors)
                if time_budget_hit_mid_level:
                    # Partial level: real survivors, but not fully examined.
                    if original_count > self.max_survivors_per_level:
                        survivors.sort(key=lambda item: combo_score(item[1]), reverse=True)
                        survivors = survivors[: self.max_survivors_per_level]
                    trimmed_levels.append(
                        (direction, size, "time_budget", original_count, self.max_search_seconds, len(survivors))
                    )
                elif original_count > self.max_survivors_per_level:
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

                level_elapsed = time.monotonic() - level_start
                kept_note = f", kept top {len(survivors):,}" if original_count > len(survivors) else ""
                console.print(
                    f"[dim][{label}][/dim] size {size} done: {examined_total:,} examined -> "
                    f"{original_count:,} cleared min_fires{kept_note} ({level_elapsed:.1f}s, "
                    f"{time.monotonic() - direction_start:.1f}s total for {label})"
                )

                if time_budget_hit_mid_level:
                    break

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

        # Fixed canonical rank per name, assigned once (see _extend_and_filter_batch).
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
                pools_by_direction, names_list_by_direction, name_rank_by_direction, executor, progress, console
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
                final_rows, final_simulated = self._simulate_tasks(
                    tasks, executor, progress, "Simulating trades", console=console
                )
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
            if reason == "time_budget":
                if original_count > 0:
                    console.print(
                        f"  [red]time budget[/red] {label} size-{size}: ran out of the {limit:,}s budget partway "
                        f"through this size - {original_count:,} combos were found before time ran out "
                        f"(kept {kept_count:,}, quality-ranked if capped), but this size was NOT fully examined. "
                        f"Raise max_search_seconds for a longer run."
                    )
                else:
                    console.print(
                        f"  [red]time budget[/red] {label} stopped before size {size}: this direction had "
                        f"{limit:,}s and used it all on previous sizes - whatever was found through the previous "
                        f"size is kept, size {size}+ simply weren't reached. Raise max_search_seconds for a longer run."
                    )
            elif reason == "raw_candidates":
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

        # Best combo per distinct size - shows the top pattern at each complexity level.
        best_idx = profitable.groupby("size")["total_pnl"].idxmax()
        best_per_size = profitable.loc[best_idx].sort_values("size").reset_index(drop=True)
        console.print(self._make_table("Best combo at each size", best_per_size, numbered=False, size_col=True))

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
