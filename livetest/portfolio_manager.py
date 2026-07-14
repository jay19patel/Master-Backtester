"""PortfolioManager: trades multiple strategies together on ONE shared account,
with portfolio-level risk controls a single-strategy backtest can't show at all.

This does NOT try to raise the win rate - it manages risk and lets winners run
further, which grows capital even at the same accuracy:

  - One LONG and one SHORT open at a time, max: a new signal is skipped if a
    position in that SAME direction is already open, no matter which strategy
    fired it - so signals aren't just OR'd into one account, they're combined
    into a single, direction-exclusive order book (one combined trade log,
    chronologically ordered across every strategy).
  - Concurrent-position cap: an extra hard ceiling on how many trades can be
    open at once (on top of the one-per-direction rule above), in case more
    than 2 strategies are ever traded together.
  - Portfolio-level risk cap: total risk currently "on the table" across every
    open position is capped as a % of equity - on top of the per-trade risk -
    so simultaneous strategies can't compound into an outsized bet.
  - Leverage cap: position notional is capped at `max_leverage` x current
    equity (matches Backtester) - a real exchange won't lend more than this,
    so risk-based sizing is capped here too if it would otherwise need more.
  - Drawdown throttle: risk-per-trade is automatically cut once the account is
    in a drawdown past a trigger, and restored once it recovers past a lower
    threshold (hysteresis, so it doesn't flip-flop every bar) - the classic
    real-world "trade smaller while you're losing" rule.
  - Breakeven-then-trail stop: once a trade is up `breakeven_trigger_r` (in R,
    i.e. multiples of the initial risk), its stop moves to entry - it can no
    longer become a loser. Once it reaches the ORIGINAL take-profit distance
    (never earlier - trailing too early would just cap winners short of the
    fixed target instead of extending them), the fixed target is dropped and
    the stop trails `trail_distance_r` behind the best price reached instead.
    A strong move can then run well past the original 1:2 target - this grows
    the AVERAGE WINNER without needing a better win rate. OFF by default (see
    `use_trailing_stop`).

Positions are tracked by absolute `entry_time` (not a bar-index into `self.df`)
specifically so a simulation can be RESUMED on a different (later-fetched) df
that doesn't contain the earlier bars at all - see `run_incremental()`, used
by the live engine, which only ever holds a small rolling window of candles in
memory rather than the entire history since inception.

Usage:
    strategies = [{"name": "strategy_01", "direction_array": arr1}, ...]
    PortfolioManager(df, strategies).print_report()               # backtest, fresh start
    PortfolioManager(df, strategies).run_incremental(prior_state)  # live, resumed
"""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table


class PortfolioManager:
    """Backtests a BASKET of strategies together on one shared, risk-managed account.

    Usage:
        PortfolioManager(df, strategies).print_report()
    """

    def __init__(
        self,
        df,
        strategies,
        initial_capital=100.0,
        risk_per_trade_pct=2.0,
        stop_loss_pct=0.5,
        take_profit_pct=1.0,
        max_hold_bars=20,
        fee_pct=0.05,
        max_concurrent_trades=3,
        portfolio_risk_cap_pct=6.0,
        drawdown_throttle_trigger_pct=10.0,
        drawdown_recovery_pct=5.0,
        throttled_risk_pct=1.0,
        max_leverage=2.0,
        bar_duration=pd.Timedelta(hours=1),
        use_trailing_stop=False,
        breakeven_trigger_r=1.0,
        trail_trigger_r=1.5,
        trail_distance_r=0.5,
    ):
        """
        strategies                   : list of {"name": str, "direction_array": array}
                                        (same convention as Backtester.run()) - a
                                        +1/-1/0 direction per candle, one array per
                                        strategy, all traded together on one account
        risk_per_trade_pct          : normal per-trade risk, % of current equity
        max_concurrent_trades       : hard cap on simultaneously open positions
        max_leverage                 : hard cap on any position's notional as a
                                        multiple of current equity (matches
                                        Backtester's leverage cap)
        portfolio_risk_cap_pct      : max total risk %, summed across all open
                                       positions, allowed at any one time
        drawdown_throttle_trigger_pct: switch to `throttled_risk_pct` once the
                                       account is down this % from its equity peak
        drawdown_recovery_pct        : switch back to normal risk once drawdown
                                       recovers below this %
        throttled_risk_pct           : reduced per-trade risk used while throttled
        use_trailing_stop            : OFF by default. When tested on real signals,
                                       an early breakeven stop cut off trades that
                                       would have gone on to hit the full target more
                                       often than it saved from a bigger loss, and a
                                       0.5R trailing giveback cost more than a fixed
                                       1:2 target captured - so the default is the
                                       plain fixed stop/target bracket (matches
                                       Backtester exactly). Turn this on to experiment
                                       - it can still help on a strategy whose winners
                                       genuinely tend to run well past the target.
        breakeven_trigger_r          : (only if use_trailing_stop) move stop to entry
                                       once favorable move reaches this many R
        trail_trigger_r              : (only if use_trailing_stop) start trailing the
                                       stop once favorable move reaches this many R -
                                       always clamped to at least the target's own R,
                                       so it can never cap a winner short of target
        trail_distance_r             : (only if use_trailing_stop) how many R the
                                       trailing stop sits behind the best price reached
        bar_duration                  : fixed candle spacing (default 1h), used to
                                       convert entry_time -> held_bars. NOT inferred
                                       from adjacent rows in self.df, since a resumed
                                       live run's df may start mid-history with no
                                       "previous" row to diff against.
        """
        self.df = df
        self.strategy_arrays = {s["name"]: np.asarray(s["direction_array"]) for s in strategies}
        self.strategy_names = list(self.strategy_arrays.keys())
        self.bar_duration = bar_duration

        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_bars = max_hold_bars
        self.fee_pct = fee_pct
        self.max_concurrent_trades = max_concurrent_trades
        self.portfolio_risk_cap_pct = portfolio_risk_cap_pct
        self.drawdown_throttle_trigger_pct = drawdown_throttle_trigger_pct
        self.drawdown_recovery_pct = drawdown_recovery_pct
        self.throttled_risk_pct = throttled_risk_pct
        self.max_leverage = max_leverage
        self.use_trailing_stop = use_trailing_stop
        self.breakeven_trigger_r = breakeven_trigger_r
        # Trailing must never engage BEFORE the original target (target_r) - otherwise
        # it truncates winners short of the fixed take-profit instead of extending them
        # past it. If the caller asks for an earlier trigger, the target still wins.
        target_r = take_profit_pct / stop_loss_pct
        self.trail_trigger_r = max(trail_trigger_r, target_r)
        self.trail_distance_r = trail_distance_r

    # ------------------------------------------------------------------
    # Core simulation - one shared account, walked bar by bar
    # ------------------------------------------------------------------
    def _simulate(self, start_i, equity, peak_equity, throttled, open_positions, pending_entries):
        """Shared core loop, parameterized so it can either start completely
        fresh (run()) or resume from persisted state on a different df that
        doesn't contain the earlier bars (run_incremental(), for live use).
        `open_positions` entries key off entry_time (absolute), never a bar
        index into self.df - a resumed run's df doesn't contain the bar the
        position was originally opened on."""
        df = self.df
        n = len(df)
        open_ = df["Open"].to_numpy()
        high = df["High"].to_numpy()
        low = df["Low"].to_numpy()
        close = df["Close"].to_numpy()
        sig_arrays = self.strategy_arrays

        trades = []
        equity_curve = [equity]

        for i in range(start_i, n):
            # 1. Open anything scheduled from the previous bar's signal.
            for strategy_name, direction in pending_entries:
                if len(open_positions) >= self.max_concurrent_trades:
                    continue  # slot full - this entry is missed, not queued (realistic)

                occupied_directions = {p["direction"] for p in open_positions}
                if direction in occupied_directions:
                    continue  # that direction already has an open position - one LONG and one SHORT max, never two of the same

                current_risk_pct = self.throttled_risk_pct if throttled else self.risk_per_trade_pct
                risk_dollars = equity * (current_risk_pct / 100)
                open_risk_dollars = sum(p["risk_dollars"] for p in open_positions)
                if open_risk_dollars + risk_dollars > equity * (self.portfolio_risk_cap_pct / 100):
                    continue  # would blow through the portfolio-level risk cap

                entry_price = open_[i]
                stop_dist = entry_price * (self.stop_loss_pct / 100)
                target_dist = entry_price * (self.take_profit_pct / 100)
                if direction == 1:
                    stop_price = entry_price - stop_dist
                    target_price = entry_price + target_dist
                else:
                    stop_price = entry_price + stop_dist
                    target_price = entry_price - target_dist

                position_size = risk_dollars / stop_dist
                max_position_size = (equity * self.max_leverage) / entry_price
                position_size = min(position_size, max_position_size)

                open_positions.append(
                    {
                        "strategy": strategy_name,
                        "direction": direction,
                        "entry_time": df.index[i],
                        "entry_price": entry_price,
                        "equity_at_entry": equity,  # for reporting leverage - the OTHER open position can move
                        # equity before this one closes, so equity-at-close is the wrong denominator
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "stop_dist": stop_dist,
                        "risk_dollars": risk_dollars,
                        "position_size": position_size,
                        "best_price": entry_price,
                        "breakeven_done": False,
                        "trailing": False,
                    }
                )
            pending_entries = []

            # 2. Manage / close open positions using this bar's High/Low.
            still_open = []
            just_closed_directions = set()
            for pos in open_positions:
                direction = pos["direction"]
                held_bars = (df.index[i] - pos["entry_time"]) / self.bar_duration
                exit_price, exit_reason = None, None

                if self.use_trailing_stop:
                    if direction == 1:
                        pos["best_price"] = max(pos["best_price"], high[i])
                        favorable_move = pos["best_price"] - pos["entry_price"]
                    else:
                        pos["best_price"] = min(pos["best_price"], low[i])
                        favorable_move = pos["entry_price"] - pos["best_price"]
                    r_multiple = favorable_move / pos["stop_dist"] if pos["stop_dist"] else 0

                    if not pos["breakeven_done"] and r_multiple >= self.breakeven_trigger_r:
                        pos["stop_price"] = pos["entry_price"]
                        pos["breakeven_done"] = True
                    if r_multiple >= self.trail_trigger_r:
                        pos["trailing"] = True
                    if pos["trailing"]:
                        trail_stop = pos["best_price"] - direction * self.trail_distance_r * pos["stop_dist"]
                        pos["stop_price"] = max(pos["stop_price"], trail_stop) if direction == 1 else min(
                            pos["stop_price"], trail_stop
                        )

                    # One evolving stop handles everything: initial risk stop, then
                    # breakeven once favorable, then a trailing stop once favorable
                    # enough to have passed the original target - it never exits at
                    # a fixed target price, so a strong trend can run well past 2R
                    # instead of being capped there.
                    hit_stop = low[i] <= pos["stop_price"] if direction == 1 else high[i] >= pos["stop_price"]
                    if hit_stop:
                        if pos["trailing"]:
                            exit_reason = "trail"
                        elif pos["stop_price"] == pos["entry_price"]:
                            exit_reason = "breakeven"
                        else:
                            exit_reason = "stop"
                        exit_price = pos["stop_price"]
                else:
                    # Plain fixed stop/target bracket - same mechanics as Backtester.
                    hit_stop = low[i] <= pos["stop_price"] if direction == 1 else high[i] >= pos["stop_price"]
                    hit_target = high[i] >= pos["target_price"] if direction == 1 else low[i] <= pos["target_price"]
                    if hit_stop:
                        exit_price, exit_reason = pos["stop_price"], "stop"
                    elif hit_target:
                        exit_price, exit_reason = pos["target_price"], "target"

                if exit_price is None and held_bars >= self.max_hold_bars:
                    exit_price, exit_reason = close[i], "time"

                if exit_price is None:
                    still_open.append(pos)
                    continue

                raw_pnl = pos["position_size"] * (exit_price - pos["entry_price"]) * direction
                fee = pos["position_size"] * pos["entry_price"] * (self.fee_pct / 100) * 2
                pnl = raw_pnl - fee
                equity += pnl
                just_closed_directions.add(pos["direction"])
                trades.append(
                    {
                        "strategy": pos["strategy"],
                        "entry_time": pos["entry_time"],
                        "exit_time": df.index[i],
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "entry_price": round(pos["entry_price"], 6),
                        "exit_price": round(exit_price, 6),
                        "stop_price": round(pos["stop_price"], 6),
                        "target_price": round(pos["target_price"], 6),
                        "position_size": round(pos["position_size"], 6),
                        # leverage vs equity AT ENTRY, not equity-at-close - the other
                        # open direction's P&L can move equity in between, which would
                        # otherwise make this drift away from the actual sizing decision
                        "leverage": round((pos["position_size"] * pos["entry_price"]) / pos["equity_at_entry"], 3),
                        "exit_reason": exit_reason,
                        "holding_bars": int(held_bars),
                        "holding_time": str(df.index[i] - pos["entry_time"]),
                        "planned_rr": round(self.take_profit_pct / self.stop_loss_pct, 3),
                        "rr_achieved": round((exit_price - pos["entry_price"]) * direction / pos["stop_dist"], 3),
                        "pnl": pnl,
                        "equity_after": equity,
                    }
                )

            open_positions = still_open
            equity_curve.append(equity)

            peak_equity = max(peak_equity, equity)
            drawdown_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity else 0
            if not throttled and drawdown_pct >= self.drawdown_throttle_trigger_pct:
                throttled = True
            elif throttled and drawdown_pct <= self.drawdown_recovery_pct:
                throttled = False

            # 3. New signals firing on this bar get queued for next bar's open.
            # One LONG and one SHORT open at a time, max - never two of the same
            # direction, regardless of which strategy fired them. A direction
            # that just closed ON THIS bar is not re-opened until the next bar
            # either - matches Backtester, which resumes scanning at exit_i + 1,
            # never re-testing the exit bar itself for a fresh entry.
            reserved_directions = {p["direction"] for p in open_positions} | just_closed_directions
            for name, arr in sig_arrays.items():
                direction = arr[i]
                if direction == 0 or direction in reserved_directions:
                    continue
                if i + 1 < n:
                    pending_entries.append((name, direction))
                    reserved_directions.add(direction)  # first strategy to fire this direction this bar wins

        # open_positions: still-open at the end of available data (live state).
        # pending_entries: a signal fired on the very LAST bar with nowhere to
        # enter yet (no next bar exists) - the live engine fills these once the
        # next candle's open arrives on its next fetch cycle.
        return trades, equity, equity_curve, open_positions, pending_entries, peak_equity, throttled

    def run(self):
        """Backtest mode: fresh start at initial_capital, no prior positions."""
        trades, equity, equity_curve, open_positions, pending_entries, _, _ = self._simulate(
            start_i=0,
            equity=self.initial_capital,
            peak_equity=self.initial_capital,
            throttled=False,
            open_positions=[],
            pending_entries=[],
        )
        return trades, equity, equity_curve, open_positions, pending_entries

    def run_incremental(self, prior_state):
        """Live mode: resume from persisted state instead of starting fresh.
        `prior_state` needs: balance, peak_equity, throttled, open_positions
        (each with entry_time/direction/entry_price/stop_price/target_price/
        stop_dist/risk_dollars/position_size/equity_at_entry/best_price/
        breakeven_done/trailing), pending_entries, last_processed_time (a
        pandas Timestamp, or None to process the entire df).

        Only bars strictly AFTER last_processed_time are simulated - self.df
        may be a small rolling window that doesn't contain the bars any
        already-open position was originally entered on, which is exactly why
        positions are keyed by entry_time rather than a bar index."""
        last_processed_time = prior_state.get("last_processed_time")
        if last_processed_time is None:
            start_i = 0
        else:
            after = self.df.index > last_processed_time
            if not after.any():
                return (
                    [],
                    prior_state["balance"],
                    [prior_state["balance"]],
                    prior_state.get("open_positions", []),
                    prior_state.get("pending_entries", []),
                    prior_state.get("peak_equity", prior_state["balance"]),
                    prior_state.get("throttled", False),
                )
            start_i = int(np.argmax(after))  # first True index

        return self._simulate(
            start_i=start_i,
            equity=prior_state["balance"],
            peak_equity=prior_state.get("peak_equity", prior_state["balance"]),
            throttled=prior_state.get("throttled", False),
            open_positions=prior_state.get("open_positions", []),
            pending_entries=prior_state.get("pending_entries", []),
        )

    @staticmethod
    def _max_drawdown_pct(equity_curve):
        peak = equity_curve[0]
        max_dd = 0.0
        for e in equity_curve:
            peak = max(peak, e)
            if peak:
                max_dd = max(max_dd, (peak - e) / peak * 100)
        return max_dd

    def print_report(self):
        trades, final_equity, equity_curve, open_positions, pending_entries = self.run()
        console = Console(width=220)

        console.print(
            f"\n[bold]PORTFOLIO BACKTEST[/bold]: {len(self.strategy_names)} strategies traded together on one "
            f"${self.initial_capital:.0f} account"
        )
        console.print(
            f"Risk per trade       : {self.risk_per_trade_pct:.1f}% normal, {self.throttled_risk_pct:.1f}% "
            f"while throttled (drawdown >= {self.drawdown_throttle_trigger_pct:.0f}%, "
            f"restored below {self.drawdown_recovery_pct:.0f}%)"
        )
        console.print(
            f"Concurrent positions : max {self.max_concurrent_trades} (1 LONG + 1 SHORT max in practice), "
            f"portfolio risk cap {self.portfolio_risk_cap_pct:.1f}% of equity, max leverage {self.max_leverage:.1f}x"
        )
        console.print(
            f"Stop / Target        : {self.stop_loss_pct:.2f}% / {self.take_profit_pct:.2f}% initial bracket "
            f"(1:{self.take_profit_pct / self.stop_loss_pct:.1f} reward:risk)"
            + (
                f", breakeven at {self.breakeven_trigger_r:.1f}R, trail from {self.trail_trigger_r:.1f}R "
                f"(trailing {self.trail_distance_r:.1f}R behind)"
                if self.use_trailing_stop
                else " (fixed bracket, no trailing)"
            )
        )

        n_trades = len(trades)
        if n_trades == 0:
            console.print("\nNo trades were taken.")
            return trades, final_equity, equity_curve, open_positions, pending_entries

        wins = [t for t in trades if t["pnl"] > 0]
        win_rate = len(wins) / n_trades * 100
        total_pnl = final_equity - self.initial_capital
        max_dd = self._max_drawdown_pct(equity_curve)
        pnl_style = "green" if total_pnl > 0 else "red"

        console.print(f"[dim]Total trades[/dim]  {n_trades}")
        console.print(f"[dim]Win rate[/dim]      {win_rate:.1f}%")
        console.print(f"[dim]Final equity[/dim]  ${final_equity:.2f}  (started at ${self.initial_capital:.0f})")
        console.print(
            f"[dim]Total PnL[/dim]     [{pnl_style}]${total_pnl:+.2f}  "
            f"({total_pnl / self.initial_capital * 100:+.1f}%)[/{pnl_style}]"
        )
        console.print(f"[dim]Max drawdown[/dim]  {max_dd:.1f}%")

        exit_reasons = pd.Series([t["exit_reason"] for t in trades]).value_counts()
        exit_table = Table(title="Exit reasons", show_lines=False)
        exit_table.add_column("Reason", style="bold")
        exit_table.add_column("Count", justify="right")
        exit_table.add_column("Percent", justify="right")
        for reason, count in exit_reasons.items():
            exit_table.add_row(reason, str(count), f"{count / n_trades * 100:.1f}%")
        console.print(exit_table)

        direction_table = Table(title="LONG vs SHORT breakdown", show_lines=False)
        direction_table.add_column("Direction", style="bold")
        direction_table.add_column("trades", justify="right")
        direction_table.add_column("win_rate%", justify="right")
        direction_table.add_column("total_profit", justify="right")
        direction_table.add_column("total_loss", justify="right")
        direction_table.add_column("total_pnl", justify="right")
        for label in ["LONG", "SHORT", "TOTAL"]:
            group = trades if label == "TOTAL" else [t for t in trades if t["direction"] == label]
            if not group:
                continue
            g_wins = [t for t in group if t["pnl"] > 0]
            g_losses = [t for t in group if t["pnl"] <= 0]
            g_pnl = sum(t["pnl"] for t in group)
            g_style = "green" if g_pnl > 0 else "red"
            row_style = "bold" if label == "TOTAL" else ""
            direction_table.add_row(
                f"[{row_style}]{label}[/{row_style}]" if row_style else label,
                str(len(group)),
                f"{len(g_wins) / len(group) * 100:.1f}",
                f"{sum(t['pnl'] for t in g_wins):.2f}",
                f"{sum(t['pnl'] for t in g_losses):.2f}",
                f"[{g_style}]{g_pnl:+.2f}[/{g_style}]",
            )
        console.print(direction_table)

        return trades, final_equity, equity_curve, open_positions, pending_entries
