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
from rich.console import Console
from rich.table import Table

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

            # A zero-variance slice (indicator flat over the whole sample) makes
            # corr() divide by a zero std internally - guard it instead of letting
            # numpy raise a RuntimeWarning for an undefined correlation.
            corr_value = (
                value[value_mask].corr(net_edge[value_mask]) if value[value_mask].std() > 0 else np.nan
            )
            corr_change = (
                change[change_mask].corr(net_edge[change_mask])
                if change_mask.sum() >= self.min_samples and change[change_mask].std() > 0
                else np.nan
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

    def print_report(self):
        """Print every analyzed indicator in one table, most relevant first."""
        result = self.analyze()
        console = Console(width=220)

        if result.empty:
            console.print("[bold]INDICATOR RELEVANCE REPORT[/bold] - no indicator had enough valid samples.")
            return result

        table = Table(
            title=f"INDICATOR RELEVANCE REPORT vs oracle BUY/SELL signal "
            f"({len(result)} indicators, min {self.min_samples} samples each, most relevant first)",
            show_lines=False,
        )
        table.add_column("#", justify="right", style="dim")
        table.add_column("Indicator", style="bold")
        table.add_column("dir_acc%", justify="right")
        table.add_column("buy_match%", justify="right")
        table.add_column("sell_match%", justify="right")
        table.add_column("move_%", justify="right")
        table.add_column("corr_val", justify="right")
        table.add_column("corr_chg", justify="right")
        table.add_column("n", justify="right")

        for idx, row in result.iterrows():
            deviation = row["dir_accuracy_pct"] - 50
            acc_style = "green" if deviation > 0 else ("red" if deviation < 0 else "")
            table.add_row(
                str(idx + 1),
                row["indicator"],
                f"[{acc_style}]{row['dir_accuracy_pct']:.1f}[/{acc_style}]" if acc_style else f"{row['dir_accuracy_pct']:.1f}",
                f"{row['buy_match_pct']:.1f}",
                f"{row['sell_match_pct']:.1f}",
                f"{row['move_rate_pct']:.1f}",
                f"{row['corr_value']:.3f}",
                f"{row['corr_change']:.3f}",
                str(row["samples"]),
            )

        console.print(
            "\ndir_accuracy%: of the bars where the indicator ACTUALLY moved on a BUY/SELL candle, % of "
            "time it moved the same way the signal called (50% = coin flip)\n"
            "buy_match% / sell_match%: that same hit-rate split for BUY candles vs SELL candles\n"
            "move_%: % of BUY/SELL bars the indicator changed at all (flat bars excluded from accuracy)\n"
            "corr_val/corr_chg: Pearson correlation of raw value / bar-to-bar change vs the oracle's net edge"
        )
        console.print(table)
        return result
