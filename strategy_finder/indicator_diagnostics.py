"""Deeper diagnostics on top of ComboBacktester's search, aimed at answering
four trader/data-scientist questions about the *kept* indicator set:

1. compute_contribution_stats: which indicators actually drive profitable
   combos (frequency + lift), vs. which are just along for the ride.
2. compute_redundancy_pairs: which indicator pairs are near-duplicates
   (correlated), so the search space carries less-than-it-looks-like signal.
3. find_perfect_predictors: any combo with a 100% win rate - scored with a
   Wilson lower bound + a size penalty, so a 9-trade fluke doesn't read the
   same as an 80-trade result.
4. compute_direction_target_diagnostic: separates "was the direction right"
   from "did the FIXED take-profit/stop-loss bracket actually capture it" -
   a condition can have real directional edge that a too-far fixed target
   fails to monetize. Also suggests a data-driven TP/SL from the condition's
   actual median excursion size.

All four are pure functions over already-computed data (a df, or the
condition_dictionary/combinations already saved in data/combo_results.db) -
no changes to how simulate_trades() actually trades.
"""

import math
import re

import numpy as np
import pandas as pd

from .backtester import simulate_trades_mfe_mae
from .combo_backtester import ComboBacktester

_LAG_SUFFIX_RE = re.compile(r"\[-(\d+)\]$")


def _base_name(condition):
    """Strips the '[-k]' lag suffix and (L)/(S) or >median/<median suffix,
    e.g. 'RSI_14>median[-2]' -> ('RSI_14', 'indicator')."""
    condition = _LAG_SUFFIX_RE.sub("", condition)
    if condition.endswith("(L)") or condition.endswith("(S)"):
        return condition[:-3], "price_action"
    if condition.endswith(">median") or condition.endswith("<median"):
        return condition[: -len(">median")], "indicator"
    return condition, "unknown"


# ----------------------------------------------------------------------
# 1. Per-indicator contribution (frequency + lift) across profitable combos
# ----------------------------------------------------------------------
def compute_contribution_stats(condition_dictionary, combinations):
    """For each base indicator/signal: how often it appears across all
    profitable combos, and how much a combo's total_pnl differs from the
    overall average when that indicator is present (its "lift")."""
    if not combinations:
        return {"overall_mean_pnl": 0.0, "total_combos": 0, "rows": []}

    base_of_idx = [_base_name(name) for name in condition_dictionary]

    stats = {}
    total_pnl_sum = 0.0
    for combo in combinations:
        pnl = combo["total_pnl"]
        total_pnl_sum += pnl
        seen = {base_of_idx[idx] for idx in combo["conditions"]}
        for base, kind in seen:
            s = stats.setdefault(base, {"count": 0, "pnl_sum": 0.0, "kind": kind})
            s["count"] += 1
            s["pnl_sum"] += pnl

    total_combos = len(combinations)
    overall_mean_pnl = total_pnl_sum / total_combos

    rows = []
    for base, s in stats.items():
        cnt = s["count"]
        mean_pnl_with = s["pnl_sum"] / cnt
        rows.append({
            "name": base,
            "kind": s["kind"],
            "appears_in": cnt,
            "pct_of_profitable_combos": round(cnt / total_combos * 100, 3),
            "mean_pnl_when_present": round(mean_pnl_with, 2),
            "lift_vs_overall_mean": round(mean_pnl_with - overall_mean_pnl, 2),
        })
    rows.sort(key=lambda r: -r["appears_in"])

    return {"overall_mean_pnl": round(overall_mean_pnl, 2), "total_combos": total_combos, "rows": rows}


# ----------------------------------------------------------------------
# 2. Redundant (highly correlated) indicator pairs
# ----------------------------------------------------------------------
def compute_redundancy_pairs(df, indicator_cols, threshold=0.9):
    """Flags numeric indicator pairs with |correlation| >= threshold, e.g.
    RSI_7 vs RSI_14 - the search space carries less independent signal than
    the raw column count suggests."""
    numeric_cols = [
        c for c in indicator_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1
    ]
    if len(numeric_cols) < 2:
        return []

    corr = df[numeric_cols].corr().abs()
    pairs = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r >= threshold:
                pairs.append({"a": a, "b": b, "correlation": round(float(r), 3)})
    pairs.sort(key=lambda p: -p["correlation"])
    return pairs


# ----------------------------------------------------------------------
# 3. Perfect-predictor scan (100% win rate), with statistical skepticism
# ----------------------------------------------------------------------
def _wilson_lower_bound(wins, n, z=1.96):
    """Lower bound of the Wilson score confidence interval for a win
    proportion - standard way to avoid over-trusting a small sample's raw
    win rate (e.g. 100% over 9 trades vs. 100% over 80 trades)."""
    if n == 0:
        return 0.0
    phat = wins / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)


def find_perfect_predictors(combinations, condition_dictionary, top_n=200):
    """Every combo with a 100% win rate, ranked by a Wilson lower-bound
    confidence (not just the raw 100%) with a plain reliability tag - larger
    combos (more ANDed conditions) are inherently more overfit-prone, so
    size caps how high the tag can go regardless of trade count."""
    rows = []
    for combo in combinations:
        if combo["win_rate_pct"] != 100.0:
            continue
        trades = combo["trades"]
        size = combo["size"]
        lb = _wilson_lower_bound(trades, trades)

        if lb >= 0.85 and size <= 5:
            reliability = "High"
        elif lb >= 0.6 and size <= 8:
            reliability = "Medium"
        else:
            reliability = "Low"

        rows.append({
            "direction": combo["direction"],
            "combo": " AND ".join(condition_dictionary[i] for i in combo["conditions"]),
            "size": size,
            "trades": trades,
            "total_pnl": combo["total_pnl"],
            "wilson_lower_bound_pct": round(lb * 100, 1),
            "reliability": reliability,
        })

    rows.sort(key=lambda r: (-r["wilson_lower_bound_pct"], -r["trades"]))
    return rows[:top_n]


# ----------------------------------------------------------------------
# 4. Direction-correct-but-target-missed diagnostic
# ----------------------------------------------------------------------
def compute_direction_target_diagnostic(
    df,
    condition_window=100,
    lag_depths=(1, 2, 3),
    min_fires=15,
    take_profit_pct=2.0,
    stop_loss_pct=1.0,
    max_hold_bars=20,
):
    """Per base condition (long and short pools, same construction as
    ComboBacktester's size-1 pass): directional_accuracy_pct (was the
    close-of-window move in the signal's favor, regardless of the fixed
    bracket) vs. target_hit_rate_pct (did the FIXED take_profit_pct actually
    get touched before the fixed stop_loss_pct). The gap between them is
    exactly "direction was right but our fixed target missed it" - and the
    median MFE/MAE gives a data-driven suggested TP/SL instead of the fixed
    one currently in use."""
    combo_bt = ComboBacktester(
        df, condition_window=condition_window, lag_depths=lag_depths, min_fires=min_fires,
    )
    long_pool, short_pool = combo_bt._build_pools()

    open_ = df["Open"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()

    rows = []
    for direction, pool, label in ((1, long_pool, "Long"), (-1, short_pool, "Short")):
        for name, mask in pool.items():
            sig = np.where(mask, direction, 0)
            trades = simulate_trades_mfe_mae(sig, open_, high, low, close, take_profit_pct, stop_loss_pct, max_hold_bars)
            n = len(trades)
            if n < min_fires:
                continue

            final_pcts = [t["final_pct"] for t in trades]
            mfes = [t["mfe_pct"] for t in trades]
            maes = [t["mae_pct"] for t in trades]
            target_hits = sum(1 for t in trades if t["target_hit"])

            directional_accuracy = sum(1 for p in final_pcts if p > 0) / n * 100
            target_hit_rate = target_hits / n * 100

            rows.append({
                "condition": name,
                "direction": label,
                "trades": n,
                "directional_accuracy_pct": round(directional_accuracy, 1),
                "target_hit_rate_pct": round(target_hit_rate, 1),
                "gap_pct": round(directional_accuracy - target_hit_rate, 1),
                "suggested_tp_pct": round(float(np.median(mfes)), 3),
                "suggested_sl_pct": round(float(np.median(maes)), 3),
            })

    rows.sort(key=lambda r: -r["gap_pct"])
    return rows
