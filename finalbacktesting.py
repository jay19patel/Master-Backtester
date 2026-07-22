"""finalbacktesting.py: finds the top Long combos and top Short combos (by
total PnL) from ComboBacktester's search, and exports everything
dashboard.html's "Final Backtest" tab needs to let you pick ANY one of each
side and see a COMBINED, single-account backtest - when the Long strategy's
signal fires, it takes a long trade; when the Short strategy's signal fires,
it takes a short trade; both on the same equity curve, not two separate
reports. The actual bar-by-bar simulation for whatever pair you pick runs
client-side in dashboard.html's JS (a straight port of backtester.py's
simulate_trades), so switching strategies is instant - no need to re-run this
script for every combination, only when you want the underlying combo search
refreshed.

Usage:
    python3 finalbacktesting.py

Output:
    final_backtest_data.json - OHLC candles + params + the top Long/Short
    combos (each with its own boolean signal mask) for dashboard.html to load.
"""

import json
import math

import numpy as np
import pandas as pd
from rich.console import Console

import main as cfg
from combo_backtester import ComboBacktester

OUTPUT_JSON = "final_backtest_data.json"
TOP_N_PER_SIDE = 20  # how many Long / how many Short combos the dashboard gets to pick from


# ----------------------------------------------------------------------
# Rebuilding a combo's boolean mask from its condition names
# ----------------------------------------------------------------------
def resolve_condition_mask(df, condition, condition_window):
    """Reconstructs the exact boolean mask ComboBacktester used for one
    condition name, e.g. "RSI_7>median" or "sweep_reversal(L)"."""
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


def mask_to_bitstring(mask):
    """Boolean array -> compact '0'/'1' string (far smaller in JSON than a
    number array with commas - ~1 byte/candle instead of ~2-3)."""
    return "".join("1" if v else "0" for v in mask.to_numpy())


def find_top_combos(df, console):
    """Runs ComboBacktester with main.py's exact settings (prints its normal
    report too) and returns (top_long_rows, top_short_rows) DataFrames."""
    combo_bt = ComboBacktester(
        df,
        initial_capital=cfg.BACKTEST_INITIAL_CAPITAL,
        risk_per_trade_pct=cfg.BACKTEST_RISK_PER_TRADE_PCT,
        stop_loss_pct=cfg.BACKTEST_STOP_LOSS_PCT,
        take_profit_pct=cfg.BACKTEST_TAKE_PROFIT_PCT,
        max_hold_bars=cfg.BACKTEST_MAX_HOLD_BARS,
        fee_pct=cfg.BACKTEST_FEE_PCT,
        min_combo_size=cfg.COMBO_MIN_SIZE,
        max_combo_size=cfg.COMBO_MAX_SIZE,
        min_fires=cfg.COMBO_MIN_FIRES,
        console_top_n=cfg.COMBO_CONSOLE_TOP_N,
        n_workers=cfg.COMBO_N_WORKERS,
        max_raw_candidates_per_level=cfg.COMBO_MAX_RAW_CANDIDATES_PER_LEVEL,
        max_survivors_per_level=cfg.COMBO_MAX_SURVIVORS_PER_LEVEL,
        max_search_seconds=cfg.COMBO_MAX_SEARCH_SECONDS,
    )
    profitable = combo_bt.print_report()
    if profitable is None or profitable.empty:
        raise RuntimeError("No profitable combination found at all - nothing to export.")

    long_rows = profitable[profitable["direction"] == "Long"].head(TOP_N_PER_SIDE)
    short_rows = profitable[profitable["direction"] == "Short"].head(TOP_N_PER_SIDE)
    return long_rows, short_rows


def json_safe_num(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return None if (math.isnan(value) or math.isinf(value)) else value


def rows_to_export(df, rows):
    out = []
    for _, row in rows.iterrows():
        mask = build_combo_mask(df, row["conditions"])
        out.append({
            "combo": row["combo"],
            "size": int(row["size"]),
            "fires": int(row["fires"]),
            "trades": int(row["trades"]),
            "win_rate_pct": json_safe_num(row["win_rate_pct"]),
            "total_pnl": json_safe_num(row["total_pnl"]),
            "return_pct": json_safe_num(row["return_pct"]),
            "mask": mask_to_bitstring(mask),
        })
    return out


def main():
    console = Console(width=220)
    df = cfg.build_dataset()

    console.print("\n[bold]FINAL BACKTEST EXPORT[/bold] - searching for the top Long and Short combos...\n")
    top_long, top_short = find_top_combos(df, console)

    console.print(
        f"\nExporting top {len(top_long)} Long and top {len(top_short)} Short combos "
        f"for dashboard.html's Final Backtest tab..."
    )

    payload = {
        "params": {
            "initial_capital": cfg.BACKTEST_INITIAL_CAPITAL,
            "risk_per_trade_pct": cfg.BACKTEST_RISK_PER_TRADE_PCT,
            "stop_loss_pct": cfg.BACKTEST_STOP_LOSS_PCT,
            "take_profit_pct": cfg.BACKTEST_TAKE_PROFIT_PCT,
            "max_hold_bars": cfg.BACKTEST_MAX_HOLD_BARS,
            "fee_pct": cfg.BACKTEST_FEE_PCT,
            "symbol": cfg.SYMBOL,
            "interval": cfg.INTERVAL,
        },
        "time": [t.isoformat() for t in df.index],
        "open": [json_safe_num(v) for v in df["Open"]],
        "high": [json_safe_num(v) for v in df["High"]],
        "low": [json_safe_num(v) for v in df["Low"]],
        "close": [json_safe_num(v) for v in df["Close"]],
        "long_combos": rows_to_export(df, top_long),
        "short_combos": rows_to_export(df, top_short),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f)

    console.print(f"[bold]Saved:[/bold] {OUTPUT_JSON}")
    console.print("Open dashboard.html -> \"Final Backtest\" tab to pick any Long + Short pair and see the combined result.")


if __name__ == "__main__":
    main()
