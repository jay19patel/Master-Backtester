"""PortfolioManager: trades multiple signals together on ONE shared account, with
portfolio-level risk controls a single-signal backtest can't show at all.

This does NOT try to raise the win rate - it manages risk and lets winners run
further, which grows capital even at the same accuracy:

  - Concurrent-position cap: limits how many trades can be open at once, so
    several signals firing together can't all stack risk onto the account
    at the same time.
  - Portfolio-level risk cap: total risk currently "on the table" across every
    open position is capped as a % of equity - on top of the per-trade risk -
    so simultaneous signals can't compound into an outsized bet.
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
    the AVERAGE WINNER without needing a better win rate.

Usage:
    PortfolioManager(df, signals=["sig_choch", "sig_bos_retest"]).print_report()
"""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from price_action_engine import PriceActionEngine


class PortfolioManager:
    """Backtests a BASKET of signals together on one shared, risk-managed account.

    Usage:
        PortfolioManager(df, signals=[...]).print_report()
    """

    def __init__(
        self,
        df,
        signals,
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
        use_trailing_stop=False,
        breakeven_trigger_r=1.0,
        trail_trigger_r=1.5,
        trail_distance_r=0.5,
    ):
        """
        signals                     : list of sig_* column names to trade together
        risk_per_trade_pct          : normal per-trade risk, % of current equity
        max_concurrent_trades       : hard cap on simultaneously open positions
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
                                       - it can still help on a signal whose winners
                                       genuinely tend to run well past the target.
        breakeven_trigger_r          : (only if use_trailing_stop) move stop to entry
                                       once favorable move reaches this many R
        trail_trigger_r              : (only if use_trailing_stop) start trailing the
                                       stop once favorable move reaches this many R -
                                       always clamped to at least the target's own R,
                                       so it can never cap a winner short of target
        trail_distance_r             : (only if use_trailing_stop) how many R the
                                       trailing stop sits behind the best price reached
        """
        has_signals = any(c.startswith("sig_") for c in df.columns)
        self.df = df.copy() if has_signals else PriceActionEngine(df.copy()).build()
        self.signal_names = [s for s in signals if s in self.df.columns]

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
    def run(self):
        df = self.df
        n = len(df)
        open_ = df["Open"].to_numpy()
        high = df["High"].to_numpy()
        low = df["Low"].to_numpy()
        close = df["Close"].to_numpy()
        sig_arrays = {name: df[name].to_numpy() for name in self.signal_names}

        equity = self.initial_capital
        peak_equity = equity
        throttled = False

        open_positions = []
        pending_entries = []
        trades = []
        equity_curve = [equity]

        for i in range(n):
            # 1. Open anything scheduled from the previous bar's signal.
            for signal_name, direction in pending_entries:
                if len(open_positions) >= self.max_concurrent_trades:
                    continue  # slot full - this entry is missed, not queued (realistic)

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

                open_positions.append(
                    {
                        "signal": signal_name,
                        "direction": direction,
                        "entry_i": i,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "target_price": target_price,
                        "stop_dist": stop_dist,
                        "risk_dollars": risk_dollars,
                        "position_size": risk_dollars / stop_dist,
                        "best_price": entry_price,
                        "breakeven_done": False,
                        "trailing": False,
                    }
                )
            pending_entries = []

            # 2. Manage / close open positions using this bar's High/Low.
            still_open = []
            just_closed_signals = set()
            for pos in open_positions:
                direction = pos["direction"]
                held = i - pos["entry_i"]
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

                if exit_price is None and held >= self.max_hold_bars:
                    exit_price, exit_reason = close[i], "time"

                if exit_price is None:
                    still_open.append(pos)
                    continue

                raw_pnl = pos["position_size"] * (exit_price - pos["entry_price"]) * direction
                fee = pos["position_size"] * pos["entry_price"] * (self.fee_pct / 100) * 2
                pnl = raw_pnl - fee
                equity += pnl
                just_closed_signals.add(pos["signal"])
                trades.append(
                    {
                        "signal": pos["signal"],
                        "entry_time": df.index[pos["entry_i"]],
                        "exit_time": df.index[i],
                        "direction": "LONG" if direction == 1 else "SHORT",
                        "exit_reason": exit_reason,
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
            # A signal that just closed ON THIS bar is not re-checked until the
            # next bar either - matches Backtester, which resumes scanning at
            # exit_i + 1, never re-testing the exit bar itself for a fresh entry.
            active_signals = {p["signal"] for p in open_positions} | just_closed_signals
            for name, arr in sig_arrays.items():
                if name in active_signals:
                    continue
                direction = arr[i]
                if direction != 0 and i + 1 < n:
                    pending_entries.append((name, direction))

        return trades, equity, equity_curve

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
        trades, final_equity, equity_curve = self.run()
        console = Console(width=220)

        console.print(
            f"\n[bold]PORTFOLIO BACKTEST[/bold]: {len(self.signal_names)} signals traded together on one "
            f"${self.initial_capital:.0f} account"
        )
        console.print(f"Signals             : {', '.join(self.signal_names)}")
        console.print(
            f"Risk per trade       : {self.risk_per_trade_pct:.1f}% normal, {self.throttled_risk_pct:.1f}% "
            f"while throttled (drawdown >= {self.drawdown_throttle_trigger_pct:.0f}%, "
            f"restored below {self.drawdown_recovery_pct:.0f}%)"
        )
        console.print(
            f"Concurrent positions : max {self.max_concurrent_trades}, "
            f"portfolio risk cap {self.portfolio_risk_cap_pct:.1f}% of equity"
        )
        console.print(
            f"Stop / Target        : {self.stop_loss_pct:.2f}% / {self.take_profit_pct:.2f}% initial bracket, "
            f"breakeven at {self.breakeven_trigger_r:.1f}R, trail from {self.trail_trigger_r:.1f}R "
            f"(trailing {self.trail_distance_r:.1f}R behind)"
        )

        n_trades = len(trades)
        if n_trades == 0:
            console.print("\nNo trades were taken.")
            return trades, final_equity

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

        per_signal = pd.DataFrame(trades).groupby("signal")["pnl"].agg(["count", "sum"]).rename(
            columns={"count": "trades", "sum": "pnl"}
        )
        contrib_table = Table(title="Contribution per signal", show_lines=False)
        contrib_table.add_column("Signal", style="bold")
        contrib_table.add_column("trades", justify="right")
        contrib_table.add_column("pnl", justify="right")
        for name, row in per_signal.sort_values("pnl", ascending=False).iterrows():
            row_style = "green" if row["pnl"] > 0 else "red"
            contrib_table.add_row(name, str(int(row["trades"])), f"[{row_style}]{row['pnl']:+.2f}[/{row_style}]")
        console.print(contrib_table)

        return trades, final_equity
