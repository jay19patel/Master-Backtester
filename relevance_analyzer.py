"""Indicator relevance analysis: which indicators actually connect to the oracle's
BUY / SELL signal?

Every column except OHLCV and the oracle_* columns is treated as "an indicator".
For each one, this is measured against oracle_signal:

  - buy_match_pct  : of the bars where the oracle signal was BUY and the indicator
                      actually moved, % of time the indicator itself moved UP too.
  - sell_match_pct : of the bars where the oracle signal was SELL and the indicator
                      actually moved, % of time the indicator itself moved DOWN too.
  - dir_accuracy   : the combined hit-rate across both (50% = coin flip, further away
                      in either direction = a stronger connection to the signal).
  - corr_value / corr_change : Pearson correlation of the indicator's raw value / its
                      bar-to-bar change against the oracle's net edge (upside_pct +
                      downside_pct - positive when BUY-side dominates, negative when
                      SELL-side dominates), computed over every candle with a known
                      outcome (BUY, SELL and HOLD alike).

HOLD candles are excluded from the hit-rate checks (there's no decisive call to match
against) but are still included in the correlation, since the net edge is defined
for every known candle.
"""

import numpy as np
import pandas as pd

EXCLUDED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
EXCLUDED_PREFIXES = ("oracle_",)


class RelevanceAnalyzer:
    """Ranks indicator columns by how strongly they connect to the oracle's BUY/SELL signal.

    Usage:
        RelevanceAnalyzer(df).print_report()
    """

    def __init__(self, df, min_samples=50):
        self.df = df
        self.min_samples = min_samples

    def indicator_columns(self):
        return [
            col
            for col in self.df.columns
            if col not in EXCLUDED_COLUMNS and not col.startswith(EXCLUDED_PREFIXES)
        ]

    def analyze(self):
        """Return a DataFrame with one relevance row per indicator, most relevant first."""
        df = self.df
        signal = df["oracle_signal"]
        # Net edge: positive when the BUY side dominates, negative when the SELL side
        # dominates, continuous - lets us correlate against every known candle, not
        # just the decisive BUY/SELL ones.
        net_edge = df["oracle_upside_pct"] + df["oracle_downside_pct"]
        decisive_mask_base = signal.isin(["BUY", "SELL"])

        rows = []
        for col in self.indicator_columns():
            value = df[col]
            change = value.diff()

            value_mask = value.notna() & net_edge.notna()
            change_mask = change.notna() & net_edge.notna()
            decisive_mask = decisive_mask_base & change.notna()

            if value_mask.sum() < self.min_samples or decisive_mask.sum() < self.min_samples:
                continue

            corr_value = value[value_mask].corr(net_edge[value_mask])
            corr_change = (
                change[change_mask].corr(net_edge[change_mask]) if change_mask.sum() >= self.min_samples else np.nan
            )

            # Only bars where the indicator actually moved count toward accuracy - a
            # flat/unchanged bar is neither an "up" nor a "down" call, so it must not
            # be silently scored as a miss (that would unfairly punish step-like
            # indicators such as supertrend_direction or is_bullish that rarely flip).
            moved_mask = decisive_mask & (change != 0)
            move_rate_pct = moved_mask.sum() / decisive_mask.sum() * 100

            buy_mask = moved_mask & (signal == "BUY")
            sell_mask = moved_mask & (signal == "SELL")
            buy_match_pct = (change[buy_mask] > 0).mean() * 100 if buy_mask.sum() else np.nan
            sell_match_pct = (change[sell_mask] < 0).mean() * 100 if sell_mask.sum() else np.nan

            if moved_mask.sum() < self.min_samples:
                continue

            correct_calls = (change[buy_mask] > 0).sum() + (change[sell_mask] < 0).sum()
            dir_accuracy = correct_calls / moved_mask.sum() * 100

            rows.append(
                {
                    "indicator": col,
                    "dir_accuracy_pct": dir_accuracy,
                    "buy_match_pct": buy_match_pct,
                    "sell_match_pct": sell_match_pct,
                    "move_rate_pct": move_rate_pct,
                    "corr_value": corr_value,
                    "corr_change": corr_change,
                    "samples": int(moved_mask.sum()),
                }
            )

        result = pd.DataFrame(rows)
        if result.empty:
            return result

        # Relevance = how far the directional hit-rate strays from a 50/50 coin flip.
        result["relevance_score"] = (result["dir_accuracy_pct"] - 50).abs()
        result = result.sort_values("relevance_score", ascending=False).reset_index(drop=True)
        return result

    def print_report(self, top_n=15):
        result = self.analyze()

        print("\n" + "=" * 110)
        print("INDICATOR RELEVANCE REPORT (vs oracle BUY/SELL signal)")
        print("=" * 110)

        if result.empty:
            print("No indicator had enough valid samples to analyze.")
            print("=" * 110 + "\n")
            return result

        print(f"Indicators analyzed   : {len(result)}  (min {self.min_samples} samples required each)")
        print("dir_accuracy%          : of the bars where the indicator ACTUALLY moved on a BUY or SELL")
        print("                         candle, % of time it moved the same way the signal called (50% = coin flip)")
        print("buy_match% / sell_match%: that same hit-rate split out for BUY candles vs SELL candles")
        print("move_rate%             : % of BUY/SELL bars the indicator changed at all (flat bars are")
        print("                         excluded from accuracy so rarely-moving indicators aren't unfairly judged)")
        print("corr_value/corr_chg    : Pearson correlation of raw value / bar-to-bar change vs the oracle's")
        print("                         net edge % (positive = BUY-side bigger, negative = SELL-side bigger)")

        header = (
            f"{'#':>3} {'Indicator':<24} {'dir_acc%':>9} {'buy_match%':>11} "
            f"{'sell_match%':>12} {'move_%':>8} {'corr_val':>9} {'corr_chg':>9} {'n':>7}"
        )
        divider = "-" * len(header)

        def print_rows(rows_df, title):
            print(f"\n--- {title} ---")
            print(header)
            print(divider)
            for idx, row in rows_df.iterrows():
                print(
                    f"{idx + 1:>3} {row['indicator']:<24} "
                    f"{row['dir_accuracy_pct']:>9.1f} {row['buy_match_pct']:>11.1f} "
                    f"{row['sell_match_pct']:>12.1f} {row['move_rate_pct']:>8.1f} "
                    f"{row['corr_value']:>9.3f} {row['corr_change']:>9.3f} {row['samples']:>7}"
                )

        print_rows(result.head(top_n), f"Top {top_n} most relevant indicators")

        bottom = result.tail(top_n).iloc[::-1]
        print_rows(bottom, f"Bottom {top_n} least relevant indicators (closest to a coin flip)")

        print("=" * 110 + "\n")
        return result
