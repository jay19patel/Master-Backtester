"""Run ONE selected strategy (a single combo from strategy_finder's search - a
direction plus a list of AND'ed conditions) standalone, and get its full trade
book back - not a combined long+short pairing (that was the old
finalbacktesting.py's job; this replaces it with a simpler single-strategy view
for the ui/ Streamlit app's "Backtest a Strategy" tab).

Usage:
    from backtesting.strategy_runner import run_strategy
    result = run_strategy("Long", ["ATR_pct<median", "vol_regime<median"], df)
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_finder.backtester import simulate_trades  # noqa: E402

_LAG_SUFFIX_RE = re.compile(r"\[-(\d+)\]$")


def resolve_condition_mask(df, condition, condition_window=100):
    """Reconstructs the exact boolean mask ComboBacktester used for one
    condition name, e.g. "RSI_7>median", "sweep(L)", or a multi-candle
    "k candles ago" copy like "RSI_7>median[-2]"."""
    lag_match = _LAG_SUFFIX_RE.search(condition)
    if lag_match:
        k = int(lag_match.group(1))
        base_mask = resolve_condition_mask(df, condition[: lag_match.start()], condition_window)
        return base_mask.shift(k, fill_value=False)
    if condition.endswith("(L)") or condition.endswith("(S)"):
        name = condition[:-3]
        is_long = condition.endswith("(L)")
        arr = df[f"sig_{name}"]
        return (arr == 1) if is_long else (arr == -1)
    if condition.endswith(">median"):
        col = condition[: -len(">median")]
        median = df[col].rolling(condition_window, min_periods=condition_window).median()
        return df[col] > median
    if condition.endswith("<median"):
        col = condition[: -len("<median")]
        median = df[col].rolling(condition_window, min_periods=condition_window).median()
        return df[col] < median
    raise ValueError(f"Cannot parse condition: {condition!r}")


def build_combo_mask(df, conditions, condition_window=100):
    mask = resolve_condition_mask(df, conditions[0], condition_window)
    for condition in conditions[1:]:
        mask = mask & resolve_condition_mask(df, condition, condition_window)
    return mask


def _max_drawdown_pct(trades):
    peak = None
    max_dd = 0.0
    for t in trades:
        e = t["equity_after"]
        peak = e if peak is None else max(peak, e)
        if peak:
            max_dd = max(max_dd, (peak - e) / peak * 100)
    return max_dd


def run_strategy(
    direction,
    conditions,
    df,
    condition_window=100,
    initial_capital=1000.0,
    risk_per_trade_pct=2.0,
    stop_loss_pct=1.0,
    take_profit_pct=2.0,
    max_hold_bars=20,
    fee_pct=0.05,
):
    """Rebuilds one combo's boolean mask against `df` and simulates it
    standalone. Returns the full trade book + equity curve + summary stats
    for just this one strategy."""
    mask = build_combo_mask(df, conditions, condition_window)
    dir_value = 1 if direction == "Long" else -1
    sig = np.where(mask, dir_value, 0)

    trades, final_equity = simulate_trades(
        sig,
        df["Open"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy(),
        initial_capital, risk_per_trade_pct, stop_loss_pct, take_profit_pct, max_hold_bars, fee_pct,
        index=df.index,
    )

    n_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    total_pnl = final_equity - initial_capital
    equity_curve = [initial_capital] + [t["equity_after"] for t in trades]

    summary = {
        "direction": direction,
        "combo": " AND ".join(conditions),
        "fires": int(np.count_nonzero(mask)),
        "trades": n_trades,
        "win_rate_pct": round(len(wins) / n_trades * 100, 1) if n_trades else 0.0,
        "final_equity": round(final_equity, 2),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / initial_capital * 100, 1) if initial_capital else 0.0,
        "max_drawdown_pct": round(_max_drawdown_pct(trades), 1),
    }

    return {"summary": summary, "trades": trades, "equity_curve": equity_curve}


def _cli_smoke_test():
    """Picks the current best combo from data/report.json and re-derives its
    stats independently - should match what the search already recorded."""
    import json
    import time

    from strategy_finder.main import BACKTEST_FEE_PCT, BACKTEST_INITIAL_CAPITAL, BACKTEST_MAX_HOLD_BARS
    from strategy_finder.main import BACKTEST_RISK_PER_TRADE_PCT, BACKTEST_STOP_LOSS_PCT, BACKTEST_TAKE_PROFIT_PCT
    from strategy_finder.main import build_dataset

    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "report.json")
    with open(report_path) as f:
        report = json.load(f)
    cb = report["combo_backtest"]
    best = max(cb["combinations"], key=lambda c: c["total_pnl"])
    conditions = [cb["condition_dictionary"][i] for i in best["conditions"]]
    print(f"Recorded in report.json: {best['direction']} {' AND '.join(conditions)}")
    print(f"  trades={best['trades']} win_rate={best['win_rate_pct']}% total_pnl={best['total_pnl']}")

    t0 = time.monotonic()
    df = build_dataset()
    print(f"Dataset rebuilt in {time.monotonic() - t0:.1f}s")

    result = run_strategy(
        best["direction"], conditions, df,
        initial_capital=BACKTEST_INITIAL_CAPITAL, risk_per_trade_pct=BACKTEST_RISK_PER_TRADE_PCT,
        stop_loss_pct=BACKTEST_STOP_LOSS_PCT, take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
        max_hold_bars=BACKTEST_MAX_HOLD_BARS, fee_pct=BACKTEST_FEE_PCT,
    )
    s = result["summary"]
    print(f"Re-derived independently: trades={s['trades']} win_rate={s['win_rate_pct']}% total_pnl={s['total_pnl']}")

    match = (s["trades"] == best["trades"] and abs(s["total_pnl"] - best["total_pnl"]) < 0.01
             and abs(s["win_rate_pct"] - best["win_rate_pct"]) < 0.1)
    print("MATCH" if match else "MISMATCH - investigate")


if __name__ == "__main__":
    _cli_smoke_test()
