"""Backtester: simulates trading a given +1/-1/0 direction array against a
fixed starting balance, one strategy at a time - so you can see which
strategies would have actually grown (or shrunk) real money, not just an
abstract hit-rate number.

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
  - Trades for the same strategy never overlap - a new signal is ignored while
    a position from that same strategy is still open, exactly like a single
    strategy running on a single account.

Reward:risk math worth knowing: with the default stop_loss_pct=0.5 and
take_profit_pct=1.0 (a 1:2 risk:reward bracket), the breakeven win rate before
fees is 1 / (1 + reward/risk) = 33.3%. Anything reliably above that, after
fees, is a real edge; anything at or below it is not.
"""

import numpy as np
import pandas as pd


class Backtester:
    """Backtests +1/-1/0 direction arrays against a fixed starting balance.

    Usage:
        bt = Backtester(df, initial_capital=100)
        trades, final_equity = bt.simulate_direction_array(direction_array)
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
        self.df = df
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

    def simulate_direction_array(self, sig):
        """sig: a +1 (long) / -1 (short) / 0 (flat) numpy array, one entry per
        candle. Returns (trades, final_equity)."""
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

            i = exit_i + 1  # no overlapping trades for the same strategy

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

    def run_strategy(self, name, direction_array):
        """Backtest one named strategy's direction array. Returns a result dict
        (None if it never traded)."""
        trades, final_equity = self.simulate_direction_array(direction_array)
        n_trades = len(trades)
        if n_trades == 0:
            return None

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = final_equity - self.initial_capital

        return {
            "name": name,
            "trades": n_trades,
            "win_rate_pct": round(len(wins) / n_trades * 100, 1),
            "final_equity": round(final_equity, 2),
            "total_pnl": round(total_pnl, 2),
            "total_profit": round(sum(t["pnl"] for t in wins), 2),
            "total_loss": round(sum(t["pnl"] for t in losses), 2),
            "return_pct": round(total_pnl / self.initial_capital * 100, 1),
            "avg_pnl_per_trade": round(total_pnl / n_trades, 3),
            "max_drawdown_pct": round(self._max_drawdown_pct(trades), 1),
            "equity_curve": [round(t["equity_after"], 2) for t in trades],
        }

    def run(self, strategies):
        """strategies: list of {"name", "direction_array"} (or anything with
        those two keys). Returns a DataFrame, most profitable first."""
        rows = []
        for strat in strategies:
            result = self.run_strategy(strat["name"], strat["direction_array"])
            if result is not None:
                rows.append(result)

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("total_pnl", ascending=False).reset_index(drop=True)
        return result
