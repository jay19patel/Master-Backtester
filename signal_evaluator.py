import numpy as np
import pandas as pd
from price_action_engine import PriceActionEngine


def evaluate_signals(df, forward_bars=10, min_fires=15):
    engine = PriceActionEngine(df)
    out = engine.build()

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

        combined = np.concatenate([long_fwd, short_fwd])
        combined = combined[~np.isnan(combined)]

        n_fires = len(combined)
        if n_fires < min_fires:
            rows.append({"signal": col, "fires": n_fires, "hit_rate": None,
                         "avg_R": None, "expectancy_R": None})
            continue

        wins = combined[combined > 0]
        losses = combined[combined <= 0]
        hit_rate = len(wins) / n_fires
        avg_win = wins.mean() if len(wins) else 0
        avg_loss = -losses.mean() if len(losses) else 0
        expectancy = hit_rate * avg_win - (1 - hit_rate) * avg_loss

        rows.append({
            "signal": col,
            "fires": n_fires,
            "hit_rate": round(hit_rate, 3),
            "avg_R": round(combined.mean(), 3),
            "expectancy_R": round(expectancy, 3),
        })

    result = pd.DataFrame(rows).sort_values("expectancy_R", ascending=False, na_position="last")
    return result.reset_index(drop=True)
