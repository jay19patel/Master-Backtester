"""Backtester: simulates trading each PriceActionEngine sig_* signal on its own,
starting from a fixed account balance, so you can see which signals would have
actually grown (or shrunk) real money - not just an abstract hit-rate number.

What makes this "realistic" instead of a naive lookahead-free hit-rate check:

  - Entry is at the NEXT candle's Open, never the signal candle's own Close -
    you can only act once a candle has actually finished.
  - Every trade carries a fixed-% stop-loss AND take-profit off the entry price
    (default 0.5% stop / 1% target - a 1:2 reward:risk bracket). Whichever
    level price touches first, walking the candles bar by bar, decides the
    exit. If neither is touched within `max_hold_bars`, the trade is closed at
    that bar's Close (a time exit - closer to how a real strategy would give
    up on a stale setup).
  - Position size is risk-based: each trade risks a fixed `risk_per_trade_pct`
    of CURRENT equity (not the original $100), sized so that hitting the stop
    loses exactly that %. This is the standard way real strategies size
    positions - it also means the account can't blow up in one bad trade, and
    equity compounds realistically as it grows or shrinks.
  - A round-trip fee/slippage cost (`fee_pct`) is deducted from every trade.
  - Trades for the same signal never overlap - a new signal is ignored while a
    position from that same signal is still open, exactly like a single
    strategy running on a single account.

Reward:risk math worth knowing: with the default stop_loss_pct=0.5 and
take_profit_pct=1.0 (a 1:2 risk:reward bracket), the breakeven win rate before
fees is 1 / (1 + reward/risk) = 33.3%. Anything reliably above that, after
fees, is a real edge; anything at or below it is not.
"""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from price_action_engine import PriceActionEngine


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
        risk_per_trade_pct : % of CURRENT equity risked per trade (position size
                              is sized so hitting the stop loses exactly this much)
        stop_loss_pct       : stop-loss distance from entry, as a % of entry price
        take_profit_pct     : take-profit distance from entry, as a % of entry price
        max_hold_bars       : force-close the trade after this many candles if
                               neither the stop nor the target was touched
        fee_pct             : round-trip fee/slippage, % of trade notional
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
        """Fees are a fixed cost per trade (win or lose), so they don't just shave
        a little off the edge - they raise the win rate you actually need to clear.
        fee_pct_of_risk comes from leverage: a tight stop means a big notional for
        a small $ risk, and the fee is charged on that notional."""
        reward_risk = self.take_profit_pct / self.stop_loss_pct
        _, fee_pct_of_risk = self._fee_drag_stats()
        return (1 + fee_pct_of_risk / 100) / (1 + reward_risk) * 100

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------
    def _simulate_signal(self, col):
        return self.simulate_direction_array(self.df[col].to_numpy())

    def simulate_direction_array(self, sig):
        """Same trade simulation as _simulate_signal, but takes a raw +1/-1/0
        direction array directly instead of a column name - lets other modules
        (e.g. a combined/confluence signal built from several sig_* columns)
        reuse this exact, already-validated simulation without needing to
        write it to a real column on the DataFrame first."""
        df = self.df
        n = len(df)
        open_ = df["Open"].to_numpy()
        high = df["High"].to_numpy()
        low = df["Low"].to_numpy()
        close = df["Close"].to_numpy()

        equity = self.initial_capital
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

            stop_dist = entry_price * (self.stop_loss_pct / 100)
            target_dist = entry_price * (self.take_profit_pct / 100)
            if direction == 1:
                stop_price = entry_price - stop_dist
                target_price = entry_price + target_dist
            else:
                stop_price = entry_price + stop_dist
                target_price = entry_price - target_dist

            exit_price, exit_reason, exit_i = None, "time", min(entry_i + self.max_hold_bars, n - 1)

            for j in range(entry_i, exit_i + 1):
                if direction == 1:
                    hit_stop = low[j] <= stop_price
                    hit_target = high[j] >= target_price
                else:
                    hit_stop = high[j] >= stop_price
                    hit_target = low[j] <= target_price

                # If a single candle's range could have hit both, assume the
                # worse outcome (stop) happened first - the conservative,
                # realistic default when you don't know the exact intra-candle path.
                if hit_stop:
                    exit_price, exit_reason, exit_i = stop_price, "stop", j
                    break
                if hit_target:
                    exit_price, exit_reason, exit_i = target_price, "target", j
                    break

            if exit_price is None:
                exit_price, exit_reason = close[exit_i], "time"

            risk_dollars = equity * (self.risk_per_trade_pct / 100)
            position_size = risk_dollars / stop_dist

            raw_pnl = position_size * (exit_price - entry_price) * direction
            fee = position_size * entry_price * (self.fee_pct / 100) * 2  # both legs
            pnl = raw_pnl - fee

            equity += pnl
            trades.append(
                {
                    "entry_time": df.index[entry_i],
                    "exit_time": df.index[exit_i],
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "exit_reason": exit_reason,
                    "pnl": pnl,
                    "equity_after": equity,
                }
            )

            i = exit_i + 1  # no overlapping trades for the same signal

        return trades, equity

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
        """How much of each trade's risked $ gets eaten by fees, given the tight
        stop implies leverage - the risked amount is fixed (risk_per_trade_pct of
        equity) but the FEE is charged on notional, and notional = risk / stop%,
        so a tighter stop means more leverage and a bigger fee bite per $ risked."""
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
