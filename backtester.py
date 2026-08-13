"""Backtester: simulates trading each PriceActionEngine sig_* signal on its own,
from a fixed account balance, to see real PnL - not just a hit-rate number.

Realistic by construction: entry at next candle's Open (no lookahead), a
fixed-% stop/target bracket walked bar-by-bar (time exit at max_hold_bars if
neither is hit), risk-based position sizing (% of current equity), a
round-trip fee, and no overlapping trades per signal.

Breakeven win rate before fees = 1 / (1 + reward/risk).
"""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from price_action_engine import PriceActionEngine


def simulate_trades(
    sig, open_, high, low, close,
    initial_capital, risk_per_trade_pct, stop_loss_pct, take_profit_pct, max_hold_bars, fee_pct,
    index=None,
):
    """Core bar-by-bar bracket simulation. Shared by Backtester (full trade
    detail incl. timestamps) and ComboBacktester's workers (index=None skips
    entry_time/exit_time, keeping the payload numpy-only)."""
    n = len(open_)
    equity = initial_capital
    trades = []

    i = 0
    while i < n - 1:
        direction = sig[i]
        if direction == 0:
            i += 1
            continue

        entry_i = i + 1  # act on the NEXT candle's open - no lookahead
        if entry_i >= n:
            break
        entry_price = open_[entry_i]

        stop_dist = entry_price * (stop_loss_pct / 100)
        target_dist = entry_price * (take_profit_pct / 100)
        if direction == 1:
            stop_price = entry_price - stop_dist
            target_price = entry_price + target_dist
        else:
            stop_price = entry_price + stop_dist
            target_price = entry_price - target_dist

        exit_price, exit_reason, exit_i = None, "time", min(entry_i + max_hold_bars, n - 1)

        for j in range(entry_i, exit_i + 1):
            if direction == 1:
                hit_stop = low[j] <= stop_price
                hit_target = high[j] >= target_price
            else:
                hit_stop = high[j] >= stop_price
                hit_target = low[j] <= target_price

            # If both stop and target were touched in one candle, assume stop hit first.
            if hit_stop:
                exit_price, exit_reason, exit_i = stop_price, "stop", j
                break
            if hit_target:
                exit_price, exit_reason, exit_i = target_price, "target", j
                break

        if exit_price is None:
            exit_price, exit_reason = close[exit_i], "time"

        risk_dollars = equity * (risk_per_trade_pct / 100)
        position_size = risk_dollars / stop_dist

        raw_pnl = position_size * (exit_price - entry_price) * direction
        fee = position_size * entry_price * (fee_pct / 100) * 2  # both legs
        pnl = raw_pnl - fee

        equity += pnl
        trade = {
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "position_size": position_size,
            "exit_reason": exit_reason,
            "pnl": pnl,
            "equity_after": equity,
        }
        if index is not None:
            trade["entry_time"] = index[entry_i]
            trade["exit_time"] = index[exit_i]
        trades.append(trade)

        i = exit_i + 1  # no overlapping trades for the same signal

    return trades, equity


def simulate_trades_mfe_mae(sig, open_, high, low, close, take_profit_pct, stop_loss_pct, max_hold_bars):
    """Diagnostic sibling of simulate_trades(): for every signal fire, walks the
    full max_hold_bars window unconditionally (ignoring the fixed bracket) and
    records how far price actually moved - max favorable excursion (mfe_pct),
    max adverse excursion (mae_pct), and the close-of-window move (final_pct) -
    plus whether the fixed target would have been touched before the fixed
    stop (target_hit, same stop-wins-ties convention as simulate_trades).

    Lets a condition's real directional edge (final_pct > 0) be measured
    separately from whatever fixed TP/SL bracket happens to be configured -
    a condition can be directionally right most of the time yet still show a
    mediocre win rate purely because the fixed target sits farther than its
    typical move. Same no-lookahead entry (next candle's open) and
    non-overlapping-window consumption as simulate_trades, so trade
    counts/timing line up 1:1 with the real backtest."""
    n = len(open_)
    rows = []

    i = 0
    while i < n - 1:
        direction = sig[i]
        if direction == 0:
            i += 1
            continue

        entry_i = i + 1  # act on the NEXT candle's open - no lookahead
        if entry_i >= n:
            break
        entry_price = open_[entry_i]
        exit_bound = min(entry_i + max_hold_bars, n - 1)

        mfe_pct = 0.0
        mae_pct = 0.0
        target_touched_i = None
        stop_touched_i = None

        for j in range(entry_i, exit_bound + 1):
            if direction == 1:
                fav_pct = (high[j] - entry_price) / entry_price * 100
                adv_pct = (entry_price - low[j]) / entry_price * 100
            else:
                fav_pct = (entry_price - low[j]) / entry_price * 100
                adv_pct = (high[j] - entry_price) / entry_price * 100

            mfe_pct = max(mfe_pct, fav_pct)
            mae_pct = max(mae_pct, adv_pct)

            if target_touched_i is None and fav_pct >= take_profit_pct:
                target_touched_i = j
            if stop_touched_i is None and adv_pct >= stop_loss_pct:
                stop_touched_i = j

        final_pct = (close[exit_bound] - entry_price) / entry_price * 100 * direction
        # stop wins same-candle ties, matching simulate_trades' convention
        target_hit = target_touched_i is not None and (
            stop_touched_i is None or target_touched_i < stop_touched_i
        )

        rows.append({
            "direction": direction,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "final_pct": final_pct,
            "target_hit": target_hit,
        })

        i = exit_bound + 1  # no overlapping windows, matches simulate_trades

    return rows


class Backtester:
    """Backtests every sig_* column independently against a fixed starting balance.

    Usage:
        Backtester(df, initial_capital=100).print_report()
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
    ):
        """
        initial_capital    : starting account balance ($)
        risk_per_trade_pct : % of current equity risked per trade
        stop_loss_pct      : stop-loss distance from entry, % of entry price
        take_profit_pct    : take-profit distance from entry, % of entry price
        max_hold_bars      : force-close after this many candles if neither hit
        fee_pct            : round-trip fee/slippage, % of trade notional
        """
        has_signals = any(c.startswith("sig_") for c in df.columns)
        self.df = df.copy() if has_signals else PriceActionEngine(df.copy()).build()

        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_bars = max_hold_bars
        self.fee_pct = fee_pct

    @property
    def breakeven_win_rate_pct(self):
        reward_risk = self.take_profit_pct / self.stop_loss_pct
        return 1 / (1 + reward_risk) * 100

    @property
    def fee_adjusted_breakeven_win_rate_pct(self):
        """Fees raise the win rate needed to break even - a tight stop implies
        leverage, so the fee (charged on notional) bites harder per $ risked."""
        reward_risk = self.take_profit_pct / self.stop_loss_pct
        _, fee_pct_of_risk = self._fee_drag_stats()
        return (1 + fee_pct_of_risk / 100) / (1 + reward_risk) * 100

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------
    def _simulate_signal(self, col):
        return self.simulate_direction_array(self.df[col].to_numpy())

    def simulate_direction_array(self, sig):
        """Same as _simulate_signal but takes a raw +1/-1/0 array directly,
        so a combined/confluence signal can reuse it without a real column."""
        df = self.df
        return simulate_trades(
            sig,
            df["Open"].to_numpy(),
            df["High"].to_numpy(),
            df["Low"].to_numpy(),
            df["Close"].to_numpy(),
            self.initial_capital,
            self.risk_per_trade_pct,
            self.stop_loss_pct,
            self.take_profit_pct,
            self.max_hold_bars,
            self.fee_pct,
            index=df.index,
        )

    @staticmethod
    def _max_drawdown_pct(trades):
        peak = None
        max_dd = 0.0
        for t in trades:
            e = t["equity_after"]
            peak = e if peak is None else max(peak, e)
            if peak:
                max_dd = max(max_dd, (peak - e) / peak * 100)
        return max_dd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self):
        """Backtest every sig_* column. Returns a DataFrame, most profitable first."""
        sig_cols = [c for c in self.df.columns if c.startswith("sig_")]
        rows = []

        for col in sig_cols:
            trades, final_equity = self._simulate_signal(col)
            n_trades = len(trades)
            if n_trades == 0:
                continue

            wins = [t for t in trades if t["pnl"] > 0]
            losses = [t for t in trades if t["pnl"] <= 0]
            total_pnl = final_equity - self.initial_capital

            rows.append(
                {
                    "signal": col,
                    "trades": n_trades,
                    "win_rate_pct": round(len(wins) / n_trades * 100, 1),
                    "final_equity": round(final_equity, 2),
                    "total_pnl": round(total_pnl, 2),
                    "total_profit": round(sum(t["pnl"] for t in wins), 2),
                    "total_loss": round(sum(t["pnl"] for t in losses), 2),
                    "return_pct": round(total_pnl / self.initial_capital * 100, 1),
                    "avg_pnl_per_trade": round(total_pnl / n_trades, 3),
                    "max_drawdown_pct": round(self._max_drawdown_pct(trades), 1),
                }
            )

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("total_pnl", ascending=False).reset_index(drop=True)
        return result

    def _fee_drag_stats(self):
        """How much of each trade's risked $ gets eaten by fees - notional =
        risk / stop%, so a tighter stop means more leverage and bigger fee drag."""
        stop_pct = self.stop_loss_pct
        implied_leverage = self.risk_per_trade_pct / stop_pct if stop_pct else float("nan")
        fee_pct_of_risk = (self.fee_pct * 2) / stop_pct * 100 if stop_pct else float("nan")
        return implied_leverage, fee_pct_of_risk

    def run_no_fee_comparison(self):
        """Re-run every signal with fees stripped out, to isolate raw directional
        edge from fee/leverage drag. Returns a DataFrame merging both results."""
        fee_pct, self.fee_pct = self.fee_pct, 0.0
        no_fee_result = self.run().rename(columns={"total_pnl": "total_pnl_no_fees"})
        self.fee_pct = fee_pct

        with_fee_result = self.run()
        merged = with_fee_result.merge(
            no_fee_result[["signal", "total_pnl_no_fees"]], on="signal", how="left"
        )
        merged["fee_drag_$"] = (merged["total_pnl_no_fees"] - merged["total_pnl"]).round(2)
        return merged.sort_values("total_pnl_no_fees", ascending=False).reset_index(drop=True)

    @staticmethod
    def _signals_table(title, result):
        table = Table(title=title, show_lines=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Signal", style="bold")
        table.add_column("trades", justify="right")
        table.add_column("win_rate%", justify="right")
        table.add_column("final_$", justify="right")
        table.add_column("total_pnl", justify="right")
        table.add_column("total_profit", justify="right")
        table.add_column("total_loss", justify="right")
        table.add_column("return%", justify="right")
        table.add_column("avg_pnl", justify="right")
        table.add_column("max_dd%", justify="right")

        for i, row in result.iterrows():
            pnl_style = "green" if row["total_pnl"] > 0 else "red"
            table.add_row(
                str(i + 1),
                row["signal"],
                str(row["trades"]),
                f"{row['win_rate_pct']:.1f}",
                f"{row['final_equity']:.2f}",
                f"[{pnl_style}]{row['total_pnl']:+.2f}[/{pnl_style}]",
                f"{row['total_profit']:.2f}",
                f"{row['total_loss']:.2f}",
                f"[{pnl_style}]{row['return_pct']:+.1f}[/{pnl_style}]",
                f"{row['avg_pnl_per_trade']:.3f}",
                f"{row['max_drawdown_pct']:.1f}",
            )
        return table

    def print_report(self):
        result = self.run()
        console = Console(width=220)

        console.print(f"\n[bold]BACKTEST[/bold]: every signal traded on its own ${self.initial_capital:.0f} starting balance")
        console.print(f"Risk per trade      : {self.risk_per_trade_pct:.1f}% of current equity")
        console.print(
            f"Stop-loss / Target  : {self.stop_loss_pct:.2f}% / {self.take_profit_pct:.2f}% "
            f"(1:{self.take_profit_pct / self.stop_loss_pct:.1f} reward:risk)"
        )
        console.print(f"Max holding period  : {self.max_hold_bars} candles (else closed at market)")
        console.print(f"Fees (round trip)   : {self.fee_pct * 2:.2f}% of trade notional")
        console.print(
            f"Breakeven win rate  : {self.breakeven_win_rate_pct:.1f}% before fees, "
            f"{self.fee_adjusted_breakeven_win_rate_pct:.1f}% after fees "
            "(the real bar a signal's win rate must clear)"
        )

        if result.empty:
            console.print("\nNo signal produced any trades.")
            return result

        console.print(
            "\n(total_profit = sum of only the winning trades' PnL, total_loss = sum of only the losing trades' PnL)"
        )
        console.print(self._signals_table("All signals, ranked by total PnL", result))

        profitable = result[result["total_pnl"] > 0]
        if profitable.empty:
            console.print("\n[bold]Profitable signals: none.[/bold] Every signal lost money under these realistic assumptions.")
        else:
            console.print(self._signals_table(f"Profitable signals only ({len(profitable)} of {len(result)})", profitable))
            best = profitable.iloc[0]
            console.print(
                f"\n[bold]Best:[/bold] {best['signal']} turned ${self.initial_capital:.0f} into "
                f"${best['final_equity']:.2f} ({best['return_pct']:+.1f}%) over {best['trades']} trades, "
                f"{best['win_rate_pct']:.1f}% win rate."
            )
        return result
