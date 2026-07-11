"""Signal evaluation: does a PriceActionEngine sig_* signal's forward return (in ATR
units) actually pay off? Column names match the same vocabulary used across the
relevance and backtest reports (fires, hit_rate_pct, buy_match_pct, sell_match_pct)
so the same concept is never named differently in different tables:

  - fires          : how many candles this signal fired on.
  - hit_rate_pct    : of those fires, % where the forward return was profitable
                      (long signals measured long, shorts measured short).
  - buy_match_pct  : that same hit-rate, restricted to this signal's LONG fires only.
  - sell_match_pct : that same hit-rate, restricted to this signal's SHORT fires only.
  - avg_R          : average forward move in ATR units.
  - expectancy_R   : hit_rate x avg_win - (1-hit_rate) x avg_loss - the number that
                      actually matters: positive = a real edge, negative = it loses
                      on average.
"""

import numpy as np
import pandas as pd
from price_action_engine import PriceActionEngine


def evaluate_signals(df, forward_bars=10, min_fires=15):
    has_signals = any(c.startswith("sig_") for c in df.columns)
    out = df.copy() if has_signals else PriceActionEngine(df).build()

    close = out["Close"].to_numpy()
    atr = out["ATR_14"].to_numpy()
    n = len(out)

    # Forward return in ATR units, `forward_bars` ahead of each candle.
    fwd_idx = np.arange(n) + forward_bars
    fwd_idx = np.clip(fwd_idx, 0, n - 1)
    fwd_return_atr = (close[fwd_idx] - close) / np.where(atr == 0, np.nan, atr)

    sig_cols = [c for c in out.columns if c.startswith("sig_")]
    rows = []

    for col in sig_cols:
        sig = out[col].to_numpy()
        long_mask = sig == 1
        short_mask = sig == -1

        long_fwd = fwd_return_atr[long_mask]
        short_fwd = -fwd_return_atr[short_mask]  # flip sign so + is a win for shorts too
        long_fwd = long_fwd[~np.isnan(long_fwd)]
        short_fwd = short_fwd[~np.isnan(short_fwd)]

        combined = np.concatenate([long_fwd, short_fwd])

        n_fires = len(combined)
        if n_fires < min_fires:
            rows.append(
                {
                    "signal": col,
                    "fires": n_fires,
                    "hit_rate": None,
                    "buy_match_pct": None,
                    "sell_match_pct": None,
                    "avg_R": None,
                    "expectancy_R": None,
                }
            )
            continue

        wins = combined[combined > 0]
        losses = combined[combined <= 0]
        hit_rate = len(wins) / n_fires
        avg_win = wins.mean() if len(wins) else 0
        avg_loss = -losses.mean() if len(losses) else 0
        expectancy = hit_rate * avg_win - (1 - hit_rate) * avg_loss

        buy_match_pct = (long_fwd > 0).mean() * 100 if len(long_fwd) else None
        sell_match_pct = (short_fwd > 0).mean() * 100 if len(short_fwd) else None

        rows.append(
            {
                "signal": col,
                "fires": n_fires,
                "hit_rate": round(hit_rate, 3),
                "buy_match_pct": round(buy_match_pct, 1) if buy_match_pct is not None else None,
                "sell_match_pct": round(sell_match_pct, 1) if sell_match_pct is not None else None,
                "avg_R": round(combined.mean(), 3),
                "expectancy_R": round(expectancy, 3),
            }
        )

    result = pd.DataFrame(rows).sort_values("expectancy_R", ascending=False, na_position="last")
    return result.reset_index(drop=True)
