"""Indicator relevance analysis: how much does each indicator actually relate to
the oracle's ground-truth future movement?

Every column except OHLCV and the oracle_* columns is treated as "an indicator".
For each one, three things are measured against the oracle labels:

  - corr_value   : Pearson correlation between the indicator's raw value and the
                    oracle's actual future move (%) - a simple "value vs outcome" relation.
  - corr_change  : Pearson correlation between the indicator's bar-to-bar change and
                    the oracle's actual future move (%) - does the indicator's own
                    direction line up with what actually happens next?
  - dir_accuracy : when the oracle says UP, how often did the indicator itself move up
                    on that same bar (and the mirror check for DOWN)? 50% = coin flip,
                    the further from 50% (either direction) the more relevant/related it is.
"""

import numpy as np
import pandas as pd

EXCLUDED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
EXCLUDED_PREFIXES = ("oracle_",)


class RelevanceAnalyzer:
    """Ranks indicator columns by how strongly they relate to the oracle labels.

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
        move = df["oracle_actual_move_pct"]
        direction = df["oracle_direction"]
        dir_mask_base = direction.isin([1, -1])

        rows = []
        for col in self.indicator_columns():
            value = df[col]
            change = value.diff()

            value_mask = value.notna() & move.notna()
            change_mask = change.notna() & move.notna()
            dir_mask = dir_mask_base & change.notna()

            if value_mask.sum() < self.min_samples or dir_mask.sum() < self.min_samples:
                continue

            corr_value = value[value_mask].corr(move[value_mask])
            corr_change = (
                change[change_mask].corr(move[change_mask]) if change_mask.sum() >= self.min_samples else np.nan
            )

            # Only bars where the indicator actually moved count toward accuracy - a
            # flat/unchanged bar is neither an "up" nor a "down" call, so it must not
            # be silently scored as a miss (that would unfairly punish step-like
            # indicators such as supertrend_direction or is_bullish that rarely flip).
            moved_mask = dir_mask & (change != 0)
            move_rate_pct = moved_mask.sum() / dir_mask.sum() * 100

            up_mask = moved_mask & (direction == 1)
            down_mask = moved_mask & (direction == -1)
            up_match_pct = (change[up_mask] > 0).mean() * 100 if up_mask.sum() else np.nan
            down_match_pct = (change[down_mask] < 0).mean() * 100 if down_mask.sum() else np.nan

            if moved_mask.sum() < self.min_samples:
                continue

            correct_calls = (change[up_mask] > 0).sum() + (change[down_mask] < 0).sum()
            dir_accuracy = correct_calls / moved_mask.sum() * 100

            rows.append(
                {
                    "indicator": col,
                    "dir_accuracy_pct": dir_accuracy,
                    "up_match_pct": up_match_pct,
                    "down_match_pct": down_match_pct,
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

        print("\n" + "=" * 105)
        print("INDICATOR RELEVANCE REPORT (vs oracle ground truth)")
        print("=" * 105)

        if result.empty:
            print("No indicator had enough valid samples to analyze.")
            print("=" * 105 + "\n")
            return result

        print(f"Indicators analyzed : {len(result)}  (min {self.min_samples} samples required each)")
        print("dir_accuracy%        : of the bars where the indicator ACTUALLY moved, % where it moved")
        print("                        the same way the oracle's next move went (50% = coin flip)")
        print("up_match% / dn_match%: that same hit-rate split out for oracle UP bars vs DOWN bars")
        print("move_rate%           : % of bars the indicator changed at all (flat bars are excluded")
        print("                        from accuracy so rarely-moving indicators aren't unfairly judged)")
        print("corr_value/corr_chg  : Pearson correlation of raw value / bar-to-bar change vs the")
        print("                        oracle's actual move % (closer to +-1 = stronger relation)")

        header = (
            f"{'#':>3} {'Indicator':<24} {'dir_acc%':>9} {'up_match%':>10} "
            f"{'dn_match%':>10} {'move_%':>8} {'corr_val':>9} {'corr_chg':>9} {'n':>7}"
        )
        divider = "-" * len(header)

        def print_rows(rows_df, title):
            print(f"\n--- {title} ---")
            print(header)
            print(divider)
            for idx, row in rows_df.iterrows():
                print(
                    f"{idx + 1:>3} {row['indicator']:<24} "
                    f"{row['dir_accuracy_pct']:>9.1f} {row['up_match_pct']:>10.1f} "
                    f"{row['down_match_pct']:>10.1f} {row['move_rate_pct']:>8.1f} "
                    f"{row['corr_value']:>9.3f} {row['corr_change']:>9.3f} {row['samples']:>7}"
                )

        print_rows(result.head(top_n), f"Top {top_n} most relevant indicators")

        bottom = result.tail(top_n).iloc[::-1]
        print_rows(bottom, f"Bottom {top_n} least relevant indicators (closest to a coin flip)")

        print("=" * 105 + "\n")
        return result
