"""ComboBacktester: exhaustively tests combinations of every size (1, 1+2,
1+2+3, ...) of indicator conditions + PriceActionEngine sig_* signals, and
backtests each for real PnL using the same engine as Backtester.

Directional tagging: each indicator gets a long/short pair by comparing its
value to its own trailing rolling median (`condition_window`, causal only).
Each sig_* signal splits into long (`== 1`) / short (`== -1`) halves.

Multi-candle patterns (`lag_depths`): every condition/signal above is ALSO
added shifted back by each depth in `lag_depths` (default 1/2/3 candles),
suffixed `[-1]`, `[-2]`, `[-3]` - e.g. `RSI_14>median[-2]` means "RSI_14 was
above its median 2 candles ago". Since combos AND same-row masks together,
this is what lets a combo express a real multi-candle pattern (a big-range
"mother" candle 2 bars back + an inside bar 1 bar back + today's breakout),
not just conditions that all happen on the same candle. Each lag depth is
kept independently only if it alone still clears `min_fires`.

Quality filter: constant/all-NaN/never-firing conditions, and any condition
whose solo fire count is already below `min_fires`, are dropped up front -
ANDing more conditions in can only shrink fire count, never grow it, so this
loses no reachable result.

Apriori-style level-wise search: size-k combos only extend size-(k-1) combos
that already cleared `min_fires`. `max_survivors_per_level` and
`max_raw_candidates_per_level` are optional (None = no cap, fully exhaustive);
when set and a level exceeds them, only the top-scoring combos continue -
never random - reported via `trimmed_levels` so a capped run is never
mistaken for exhaustive.

Trim scoring (see `_extend_and_filter_batch`): each candidate's own fire
mask (already computed to check `min_fires`) is scored directly - NOT by summing its
members' standalone solo PnL. Two mediocre-alone conditions can have a real
combined edge that a sum-of-parts score would rank low and prune before it
ever reaches a larger size; scoring the combo's own fires avoids that. The
score is a t-stat-style measure (mean bracket return / stdev, scaled by
sqrt(fire count)) of a cheap proxy trade using the SAME SL/TP bracket as the
real engine (`_dense_bracket_returns` mirrors simulate_trades' atr/swing/
fixed modes) - just without fees, position sizing, or overlap-skipping,
which need the real sequential engine. Computed with plain vectorized numpy
on data already in memory, so it adds negligible cost next to the fire-count
check it rides alongside. It rewards both a real directional edge and enough
fires to trust it, and is only used to pick which combos continue the
search - final reported results still come from the real `simulate_trades`
engine.

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
from numpy.lib.stride_tricks import sliding_window_view
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from . import db as results_db
from .backtester import simulate_trades

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Populated once per worker by _init_worker - avoids re-pickling OHLC/pools per task.
_WORKER_STATE = {}


def _init_worker(pools_by_direction, ohlc, params, names_list_by_direction, name_rank_by_direction, bracket_returns_by_direction):
    _WORKER_STATE["pools"] = pools_by_direction
    _WORKER_STATE["ohlc"] = ohlc
    _WORKER_STATE["params"] = params
    _WORKER_STATE["names_list"] = names_list_by_direction
    _WORKER_STATE["name_rank"] = name_rank_by_direction
    _WORKER_STATE["fwd"] = bracket_returns_by_direction
    _WORKER_STATE["fwd_sq"] = {d: r * r for d, r in bracket_returns_by_direction.items()}
    # Dense (pool_size x n_candles) bool matrix per direction, row order matching
    # names_list - lets _extend_and_filter_batch AND+score an entire parent's
    # candidate extensions in one matrix op instead of one call per candidate.
    _WORKER_STATE["pool_matrix"] = {
        d: np.array([pools_by_direction[d][name] for name in names])
        for d, names in names_list_by_direction.items()
    }


def _mask_for(pool, combo):
    mask = pool[combo[0]]
    for name in combo[1:]:
        mask = mask & pool[name]
    return mask


def _dense_bracket_returns(
    direction, open_, high, low, close, max_hold_bars,
    sl_tp_mode="atr", atr=None, swing_high=None, swing_low=None,
    stop_loss_pct=None, take_profit_pct=None,
):
    """Vectorized, direction-signed %% return using the SAME SL/TP bracket
    logic as simulate_trades' 'atr'/'swing'/'fixed' modes, computed for
    EVERY candle as if a signal fired there (ignoring fees, position sizing,
    and trade overlap - those need the real sequential engine). Used ONLY to
    rank candidates for beam trimming (see module docstring), never to
    report results.

    This matters: a naive fixed-horizon close-to-close return ignores the
    actual bracket shape entirely (e.g. ATR mode's 1:2 risk:reward with
    early stop-outs), so it can rank a combo very differently from what
    simulate_trades later finds - the exact mismatch that let a real
    high-quality combo get trimmed out in favor of worse ones. Mirroring the
    bracket mechanics closes that gap while staying a single vectorized
    pass (no per-trade Python loop), computed once per direction per run."""
    n = len(open_)
    is_long = direction == 1
    entry_i = np.arange(n) + 1
    entry_i_c = np.minimum(entry_i, n - 1)
    valid = entry_i < n
    entry_price = open_[entry_i_c]

    if sl_tp_mode == "atr" and atr is not None:
        entry_atr = atr[entry_i_c]
        entry_atr = np.nan_to_num(entry_atr, nan=entry_price * 0.01)
        stop_dist = np.maximum(entry_atr * 1.5, entry_price * 0.003)
        target_dist = np.maximum(entry_atr * 3.0, entry_price * 0.006)
    elif sl_tp_mode == "swing" and swing_high is not None and swing_low is not None:
        recent_sh = swing_high[entry_i_c]
        recent_sl = swing_low[entry_i_c]
        stop_dist = (
            np.maximum(entry_price - recent_sl, entry_price * 0.005)
            if is_long
            else np.maximum(recent_sh - entry_price, entry_price * 0.005)
        )
        stop_dist = np.nan_to_num(stop_dist, nan=entry_price * 0.01)
        target_dist = stop_dist * 2.0
    else:
        stop_dist = entry_price * ((stop_loss_pct or 0.5) / 100)
        target_dist = entry_price * ((take_profit_pct or 1.0) / 100)

    stop_price = entry_price - stop_dist if is_long else entry_price + stop_dist
    target_price = entry_price + target_dist if is_long else entry_price - target_dist

    window_len = max_hold_bars + 1
    pad = np.full(window_len, np.nan)
    high_windows = sliding_window_view(np.concatenate([high, pad]), window_len)[entry_i_c]
    low_windows = sliding_window_view(np.concatenate([low, pad]), window_len)[entry_i_c]

    if is_long:
        hit_stop = low_windows <= stop_price[:, None]
        hit_target = high_windows >= target_price[:, None]
    else:
        hit_stop = high_windows >= stop_price[:, None]
        hit_target = low_windows <= target_price[:, None]
    any_hit = hit_stop | hit_target

    has_hit = any_hit.any(axis=1)
    first_col = np.argmax(any_hit, axis=1)
    stop_wins = hit_stop[np.arange(n), first_col]  # same-bar tie -> stop wins, matching simulate_trades
    is_stop = has_hit & stop_wins
    is_target = has_hit & ~stop_wins

    time_exit_i = np.minimum(entry_i + max_hold_bars, n - 1)
    exit_price = np.where(is_stop, stop_price, np.where(is_target, target_price, close[time_exit_i]))

    ret = (exit_price - entry_price) / entry_price * direction
    return np.where(valid, ret, 0.0)


def build_best_of_table(profitable, min_trades):
    """Rich table of the best combo along different axes - raw profit,
    long/short split, win rate, and a balanced pick - not just the single
    top-PnL row. Shared by ComboBacktester.print_report() and main.py's
    end-of-run summary so both show the same breakdown. `profitable` is a
    DataFrame of only total_pnl > 0 rows, sorted best PnL first."""
    long_rows = profitable[profitable["direction"] == "Long"]
    short_rows = profitable[profitable["direction"] == "Short"]
    eligible = profitable[profitable["trades"] >= min_trades]

    best_win_rate = eligible.sort_values("win_rate_pct", ascending=False).iloc[0] if not eligible.empty else None
    best_balanced = None
    if not eligible.empty:
        # Rewards profit AND win rate together, scaled by sqrt(trades) so a
        # combo needs enough trades to trust it - not just one extreme metric.
        composite = (eligible["return_pct"] / 100) * (eligible["win_rate_pct"] / 100) * np.sqrt(eligible["trades"])
        best_balanced = eligible.loc[composite.idxmax()]

    table = Table(title="Best Of", show_lines=False)
    table.add_column("Category", style="bold")
    table.add_column("Dir")
    table.add_column("Combo", overflow="fold")
    table.add_column("Size", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("Return%", justify="right")
    table.add_column("Final $", justify="right")

    def add_row(label, row):
        if row is None:
            table.add_row(label, "-", "[dim]no qualifying combo[/dim]", "-", "-", "-", "-", "-")
            return
        pnl_style = "green" if row["total_pnl"] > 0 else "red"
        table.add_row(
            label, row["direction"], row["combo"], str(row["size"]), str(row["trades"]),
            f"{row['win_rate_pct']:.1f}", f"[{pnl_style}]{row['return_pct']:+.1f}[/{pnl_style}]",
            f"{row['final_equity']:.2f}",
        )

    add_row("Max Profit (overall)", profitable.iloc[0])
    add_row("Best Long", long_rows.iloc[0] if not long_rows.empty else None)
    add_row("Best Short", short_rows.iloc[0] if not short_rows.empty else None)
    add_row(f"Best Win Rate (>={min_trades} trades)", best_win_rate)
    add_row("Decent Balanced Best*", best_balanced)
    return table


def _extend_and_filter_batch(args):
    """Generation-phase worker: args is (direction, parent_combos_chunk).
    Extends each parent only with names ranked after its own highest member
    (fixed canonical rank per pool name - same trick itertools.combinations
    uses internally), so no two parents ever reach the same k-combo and no
    dedup pass is needed.

    All of a parent's candidate extensions are AND-ed and scored together in
    one matrix op (pool_matrix slice & parent_mask -> fires via a row sum,
    score via two matrix-vector products against the bracket-return array
    and its square) instead of one Python/numpy call per candidate. With
    hundreds of thousands to millions of candidates per level, per-call
    overhead - not the actual math - was what made scoring each one
    separately ~6x slower than the unscored version; batching removes that
    overhead while keeping the exact same t-stat-style score (mean bracket
    return over its stdev, scaled by sqrt(fire count), via the identity
    var = E[x^2] - E[x]^2). Returns (survivors, examined_count)."""
    direction, parent_chunk = args
    pool = _WORKER_STATE["pools"][direction]
    min_fires = _WORKER_STATE["params"]["min_fires"]
    rank = _WORKER_STATE["name_rank"][direction]
    names_list = _WORKER_STATE["names_list"][direction]
    pool_matrix = _WORKER_STATE["pool_matrix"][direction]
    fwd = _WORKER_STATE["fwd"][direction]
    fwd_sq = _WORKER_STATE["fwd_sq"][direction]

    survivors = []
    examined = 0
    for combo_names in parent_chunk:
        parent_mask = _mask_for(pool, combo_names)
        start = rank[combo_names[-1]] + 1
        candidate_names = names_list[start:]
        examined += len(candidate_names)
        if not candidate_names:
            continue

        combo_matrix = pool_matrix[start:] & parent_mask[None, :]
        fires_arr = combo_matrix.sum(axis=1)
        keep = fires_arr >= min_fires
        if not keep.any():
            continue

        kept_idx = np.flatnonzero(keep)
        combo_f = combo_matrix[kept_idx].astype(np.float64)
        fires_kept = fires_arr[kept_idx].astype(np.float64)
        # errstate: some BLAS backends (e.g. Accelerate) raise transient
        # divide/overflow warnings on intermediate blocked-matmul steps that
        # cancel out before the final result - verified bit-exact against a
        # scalar per-candidate reference, so these are cosmetic, not real.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mean = (combo_f @ fwd) / fires_kept
            var = np.maximum((combo_f @ fwd_sq) / fires_kept - mean * mean, 0.0)
        std = np.sqrt(var)
        sqrt_n = np.sqrt(fires_kept)
        score = np.where(std > 0, mean / np.where(std > 0, std, 1.0) * sqrt_n, mean * sqrt_n)

        for j, idx in enumerate(kept_idx):
            survivors.append((direction, combo_names + (candidate_names[idx],), int(fires_arr[idx]), float(score[j])))
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

    sl_tp_mode = params.get("sl_tp_mode", "atr")
    atr = ohlc.get("atr")
    swing_high = ohlc.get("swing_high")
    swing_low = ohlc.get("swing_low")

    for direction, combo in tasks:
        mask = _mask_for(pools[direction], combo)
        direction_array = np.where(mask, direction, 0)
        trades, final_equity = simulate_trades(
            direction_array,
            ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"],
            params["initial_capital"], params["risk_per_trade_pct"], params["stop_loss_pct"],
            params["take_profit_pct"], params["max_hold_bars"], params["fee_pct"],
            atr=atr, swing_high=swing_high, swing_low=swing_low, sl_tp_mode=sl_tp_mode,
        )

        simulated += 1

        n_trades = len(trades)
        if n_trades < min_fires:
            continue

        wins = [t for t in trades if t["pnl"] > 0]
        total_pnl = final_equity - params["initial_capital"]

        avg_sl_pct = (
            sum(abs(t["entry_price"] - t["stop_price"]) / t["entry_price"] * 100 for t in trades) / n_trades
            if n_trades > 0
            else params["stop_loss_pct"]
        )
        avg_tp_pct = (
            sum(abs(t["target_price"] - t["entry_price"]) / t["entry_price"] * 100 for t in trades) / n_trades
            if n_trades > 0
            else params["take_profit_pct"]
        )

        rows.append({
            "direction": "Long" if direction == 1 else "Short",
            "combo": " AND ".join(combo),
            "conditions": list(combo),
            "size": len(combo),
            "fires": int(np.count_nonzero(mask)),
            "trades": n_trades,
            "win_rate_pct": round(len(wins) / n_trades * 100, 1),
            "avg_sl_pct": round(avg_sl_pct, 2),
            "avg_tp_pct": round(avg_tp_pct, 2),
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
        sl_tp_mode="atr",
        max_hold_bars=20,
        fee_pct=0.05,
        min_combo_size=1,
        max_combo_size=8,
        min_fires=15,
        condition_window=100,
        lag_depths=(1, 2, 3),
        n_workers=None,
        max_raw_candidates_per_level=None,
        max_survivors_per_level=None,
        max_search_seconds=None,
    ):
        self.df = df
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.sl_tp_mode = sl_tp_mode

        self.max_hold_bars = max_hold_bars
        self.fee_pct = fee_pct
        self.min_combo_size = min_combo_size
        self.max_combo_size = max_combo_size
        self.min_fires = min_fires
        self.condition_window = condition_window
        self.lag_depths = tuple(lag_depths)
        self.n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)  # leave a core free
        # Optional safety nets: raw-candidate ceiling, survivors-per-level cap,
        # time budget - each None by default (no cap, fully exhaustive search).
        # Whichever trips first (if set), combos already found are kept.
        self.max_raw_candidates_per_level = max_raw_candidates_per_level
        self.max_survivors_per_level = max_survivors_per_level
        self.max_search_seconds = max_search_seconds
        self.stats = {}
        self.max_size_reached = {}
        self.trimmed_levels = []

    # ------------------------------------------------------------------
    # Build the long/short condition + signal pools
    # ------------------------------------------------------------------
    def _add_with_lags(self, pool, base_name, mask):
        """Adds `mask` as-of-now plus a shifted-back copy for each depth in
        `self.lag_depths` (a condition/signal that was true k candles ago,
        named e.g. `RSI_14>median[-2]`), each kept independently only if it
        alone still clears `min_fires`. Returns whether anything was added,
        for the caller's dropped-column bookkeeping."""
        added = False
        if int(np.count_nonzero(mask)) >= self.min_fires:
            pool[base_name] = mask
            added = True
        for k in self.lag_depths:
            shifted = np.zeros_like(mask)
            shifted[k:] = mask[:-k]
            if int(np.count_nonzero(shifted)) >= self.min_fires:
                pool[f"{base_name}[-{k}]"] = shifted
                added = True
        return added

    def _build_pools(self):
        """Every indicator column gets an automatic long/short condition
        pair: is it currently above or below its own trailing rolling median?
        Every sig_* price-action signal is split into its long (`== 1`) and
        short (`== -1`) half. Each of those also gets a `[-k]` "k candles
        ago" copy per `self.lag_depths` - see the module docstring. A
        condition/signal that is constant, never fires, or fires fewer than
        `min_fires` times on its own (at every lag depth) is dropped - see
        the module docstring for why that loses no reachable result."""
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
            kept_above = self._add_with_lags(long_pool, f"{col}>median", above)
            kept_below = self._add_with_lags(short_pool, f"{col}<median", below)
            if not (kept_above or kept_below):
                dropped.append(col)

        for col in [c for c in df.columns if c.startswith("sig_")]:
            arr = df[col].to_numpy()
            name = col.replace("sig_", "")
            long_mask = arr == 1
            short_mask = arr == -1
            kept_long = self._add_with_lags(long_pool, f"{name}(L)", long_mask)
            kept_short = self._add_with_lags(short_pool, f"{name}(S)", short_mask)
            if not (kept_long or kept_short):
                dropped.append(col)

        self.dropped_columns = dropped
        return long_pool, short_pool

    # ------------------------------------------------------------------
    # Search + backtest
    # ------------------------------------------------------------------
    def _simulate_tasks(self, tasks, executor, progress, description, console=None, heartbeat_seconds=10, persist=True):
        """Runs _evaluate_batch over `tasks` in parallel with a progress bar.
        Returns (rows, simulated_count). Shared by the size-1 warm-up pass and
        the final simulation pass. When `persist` is True, every batch's rows
        are inserted into the SQLite results DB (results_db) as soon as that
        batch finishes - so a crash or kill mid-run loses at most the one
        in-flight batch, never the whole pass. `persist=False` is for combos
        below `min_combo_size` (e.g. the size-1 quality-score warm-up when
        min_combo_size > 1) - real trades, but not combos the run is meant to
        report, so they must not end up mixed into the results table."""
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
            if persist:
                results_db.insert_results(batch_rows)
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
        immediately for real reported rows. At each later level, candidate
        generation + fire-count filtering run on the process pool, scoring
        each survivor's own fires as it goes (see `_extend_and_filter_batch`); if survivors
        exceed max_survivors_per_level, only the top-scoring continue -
        never a random slice.

        Each level that clears `min_combo_size` is simulated and persisted to
        the results DB right here, as soon as that level finishes - not
        accumulated into one giant list simulated only after every size for
        every direction has been generated. On a search with no
        max_combo_size ceiling (the default), generation alone can run for a
        long time; without this, a kill anywhere before the very end loses
        100% of results regardless of how far the search got, since nothing
        would have been simulated yet. With it, a kill mid-run keeps every
        level that finished before that point."""
        rows = []       # every persisted real-trade row so far, across all sizes/directions
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
                size1_tasks, executor, progress, f"[{label}] size 1: simulating (quality scores)", console=console,
                persist=(self.min_combo_size <= 1),
            )
            simulated_count += size1_simulated
            tested += len(size1_tasks)
            cleared += len(size1_tasks)
            console.print(
                f"[dim][{label}][/dim] size 1 done: {len(size1_rows):,} of {len(names_list):,} produced a real "
                f"trade ({time.monotonic() - direction_start:.1f}s elapsed)"
            )

            if self.min_combo_size <= 1:
                rows.extend(size1_rows)

            current_combos = [(name,) for name in names_list]
            size_reached = 1

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
                if self.max_raw_candidates_per_level is not None and raw_estimate > self.max_raw_candidates_per_level:
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
                has_survivor_cap = self.max_survivors_per_level is not None
                if time_budget_hit_mid_level:
                    # Partial level: real survivors, but not fully examined.
                    if has_survivor_cap and original_count > self.max_survivors_per_level:
                        survivors.sort(key=lambda item: item[3], reverse=True)
                        survivors = survivors[: self.max_survivors_per_level]
                    trimmed_levels.append(
                        (direction, size, "time_budget", original_count, self.max_search_seconds, len(survivors))
                    )
                elif has_survivor_cap and original_count > self.max_survivors_per_level:
                    survivors.sort(key=lambda item: item[3], reverse=True)
                    survivors = survivors[: self.max_survivors_per_level]
                    trimmed_levels.append(
                        (direction, size, "survivors", original_count, self.max_survivors_per_level, len(survivors))
                    )

                cleared += len(survivors)
                next_combos = [combo_names for (_, combo_names, _fires, _score) in survivors]
                if size >= self.min_combo_size and next_combos:
                    level_tasks = [(direction, combo_names) for combo_names in next_combos]
                    level_rows, level_simulated = self._simulate_tasks(
                        level_tasks, executor, progress, f"[{label}] size {size}: simulating", console=console,
                        persist=True,
                    )
                    rows.extend(level_rows)
                    simulated_count += level_simulated
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
        return rows, tested, cleared, simulated_count

    def run(self):
        """Backtest every qualifying combination, at every size. Returns a
        DataFrame (both winning and losing combos that cleared min_fires),
        best PnL first."""
        # Each run starts from a clean results table - only the latest run's
        # combos are ever kept, and every row inserted from here on (see
        # _simulate_tasks) is this run's, not a leftover from a previous one.
        results_db.clear_results()

        long_pool, short_pool = self._build_pools()
        pools_by_direction = {1: long_pool, -1: short_pool}

        atr_col = "ATR_14" if "ATR_14" in self.df.columns else ([c for c in self.df.columns if c.startswith("ATR_")] or [None])[0]
        sh_col = "swing_high" if "swing_high" in self.df.columns else None
        sl_col = "swing_low" if "swing_low" in self.df.columns else None

        ohlc = {
            "open": self.df["Open"].to_numpy(),
            "high": self.df["High"].to_numpy(),
            "low": self.df["Low"].to_numpy(),
            "close": self.df["Close"].to_numpy(),
            "atr": self.df[atr_col].to_numpy() if atr_col and atr_col in self.df.columns else None,
            "swing_high": self.df[sh_col].to_numpy() if sh_col and sh_col in self.df.columns else None,
            "swing_low": self.df[sl_col].to_numpy() if sl_col and sl_col in self.df.columns else None,
        }
        params = {
            "initial_capital": self.initial_capital,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "sl_tp_mode": self.sl_tp_mode,
            "max_hold_bars": self.max_hold_bars,
            "fee_pct": self.fee_pct,
            "min_fires": self.min_fires,
        }


        # Bracket-aware per-candle return proxy (same SL/TP mechanics as the real
        # engine), computed once per direction and shared by every worker - used
        # only to score candidates for beam trimming (see _extend_and_filter_batch).
        bracket_returns_by_direction = {
            d: _dense_bracket_returns(
                d, ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], self.max_hold_bars,
                sl_tp_mode=self.sl_tp_mode, atr=ohlc.get("atr"),
                swing_high=ohlc.get("swing_high"), swing_low=ohlc.get("swing_low"),
                stop_loss_pct=self.stop_loss_pct, take_profit_pct=self.take_profit_pct,
            )
            for d in pools_by_direction
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
            initargs=(pools_by_direction, ohlc, params, names_list_by_direction, name_rank_by_direction, bracket_returns_by_direction),
        ) as executor:
            rows, tested, cleared_prefilter, simulated_count = self._generate_tasks(
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
                    f"kept the top {kept_count:,} by each combo's own bracket-aware quick score "
                    f"(not each member's solo PnL) - NOT random, but not exhaustive either at/beyond this size."
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

        # Full breakdown of every combo lives in data/combo_results.db (query
        # via strategy_finder.db) - console gets a "best of" table instead of
        # a single top-PnL line, so the report doesn't imply the top-PnL combo
        # is the only strategy worth considering (a high-winrate or
        # better-balanced combo can be more tradeable in practice than the
        # single highest-PnL one).
        console.print()
        console.print(build_best_of_table(profitable, self.min_fires))
        console.print(
            f"[dim]*Balanced Best ranks return% x win_rate% x sqrt(trades) among combos with >={self.min_fires} "
            "trades - rewards profit and win rate together with enough trades to trust it, instead of chasing a "
            "single extreme.[/dim]"
        )
        return profitable
