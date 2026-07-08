"""Search combinations of simple, interpretable indicator conditions (EMA up, RSI
oversold, MACD bullish cross, ...) to find which ones best line up with the
oracle's BUY / SELL signal.

Trying every indicator at every possible threshold is a huge search space, so a
curated list of common, well-understood trading conditions is built first (see
`_build_conditions`). Those conditions are then AND-combined 1, 2 and up to
`max_combo_size` at a time, and each combination is scored by:

  - support   : how many candles the combination actually fires on (too few and
                the result is just noise/overfitting - see `min_support`).
  - precision : of the candles where the combination fired, what % were actually
                the target signal (BUY or SELL).
  - lift      : precision / base_rate - how many times better than random guessing
                this combination is (1.0x = no better than the overall BUY/SELL rate).
"""

import itertools

import pandas as pd


class ConditionFinder:
    """Finds which indicator-condition combinations best match the oracle signal.

    Usage:
        ConditionFinder(df).print_report()
    """

    def __init__(self, df, min_support=100, max_combo_size=3, top_n=15):
        self.df = df
        self.min_support = min_support
        self.max_combo_size = max_combo_size
        self.top_n = top_n
        self.conditions = self._build_conditions()

    def _build_conditions(self):
        """A curated set of {name: boolean_mask} conditions built from whichever
        indicator columns are actually present in the DataFrame."""
        df = self.df
        conditions = {}

        def add(name, required_cols, mask_fn):
            if all(c in df.columns for c in required_cols):
                conditions[name] = mask_fn()

        add("EMA5>EMA20 (short-term uptrend)", ["EMA_5", "EMA_20"], lambda: df["EMA_5"] > df["EMA_20"])
        add("EMA5<EMA20 (short-term downtrend)", ["EMA_5", "EMA_20"], lambda: df["EMA_5"] < df["EMA_20"])
        add("Close>EMA20 (above trend)", ["Close", "EMA_20"], lambda: df["Close"] > df["EMA_20"])
        add("Close<EMA20 (below trend)", ["Close", "EMA_20"], lambda: df["Close"] < df["EMA_20"])
        add("Close>EMA50 (above trend)", ["Close", "EMA_50"], lambda: df["Close"] > df["EMA_50"])
        add("Close<EMA50 (below trend)", ["Close", "EMA_50"], lambda: df["Close"] < df["EMA_50"])

        add("RSI14<30 (oversold)", ["RSI_14"], lambda: df["RSI_14"] < 30)
        add("RSI14>70 (overbought)", ["RSI_14"], lambda: df["RSI_14"] > 70)
        add("RSI14>50 (bullish momentum)", ["RSI_14"], lambda: df["RSI_14"] > 50)
        add("RSI14<50 (bearish momentum)", ["RSI_14"], lambda: df["RSI_14"] < 50)

        add("MACD>Signal (bullish cross)", ["MACD", "MACD_signal"], lambda: df["MACD"] > df["MACD_signal"])
        add("MACD<Signal (bearish cross)", ["MACD", "MACD_signal"], lambda: df["MACD"] < df["MACD_signal"])

        add("ADX14>25 (strong trend)", ["ADX_14"], lambda: df["ADX_14"] > 25)
        add("ADX14<20 (weak/no trend)", ["ADX_14"], lambda: df["ADX_14"] < 20)

        add("BB_position<0.2 (near lower band)", ["BB_position"], lambda: df["BB_position"] < 0.2)
        add("BB_position>0.8 (near upper band)", ["BB_position"], lambda: df["BB_position"] > 0.8)

        add("StochK<20 (oversold)", ["Stoch_K"], lambda: df["Stoch_K"] < 20)
        add("StochK>80 (overbought)", ["Stoch_K"], lambda: df["Stoch_K"] > 80)

        add("CCI20<-100 (oversold)", ["CCI_20"], lambda: df["CCI_20"] < -100)
        add("CCI20>100 (overbought)", ["CCI_20"], lambda: df["CCI_20"] > 100)

        add("WilliamsR<-80 (oversold)", ["WilliamsR_14"], lambda: df["WilliamsR_14"] < -80)
        add("WilliamsR>-20 (overbought)", ["WilliamsR_14"], lambda: df["WilliamsR_14"] > -20)

        add("Supertrend up", ["supertrend_direction"], lambda: df["supertrend_direction"] == 1)
        add("Supertrend down", ["supertrend_direction"], lambda: df["supertrend_direction"] == -1)

        add("VolumeRatio>1.5 (high volume)", ["volume_ratio"], lambda: df["volume_ratio"] > 1.5)
        add("VolumeRatio<0.7 (low volume)", ["volume_ratio"], lambda: df["volume_ratio"] < 0.7)

        add("Bullish candle", ["is_bullish"], lambda: df["is_bullish"] == 1)
        add("Bearish candle", ["is_bullish"], lambda: df["is_bullish"] == 0)

        return conditions

    def find_best_combinations(self, target):
        """Return (result_df, base_rate_pct) for combinations matching oracle_signal == target."""
        df = self.df
        known_mask = df["oracle_signal"].notna()
        target_mask = known_mask & (df["oracle_signal"] == target)
        total_known = int(known_mask.sum())
        base_rate = target_mask.sum() / total_known * 100

        names = list(self.conditions.keys())
        rows = []

        for size in range(1, self.max_combo_size + 1):
            for combo in itertools.combinations(names, size):
                combined_mask = known_mask
                for name in combo:
                    combined_mask = combined_mask & self.conditions[name]

                support = int(combined_mask.sum())
                if support < self.min_support:
                    continue

                hits = int((combined_mask & target_mask).sum())
                precision = hits / support * 100

                rows.append(
                    {
                        "conditions": " AND ".join(combo),
                        "size": size,
                        "support": support,
                        "precision_pct": precision,
                        "lift": precision / base_rate if base_rate else float("nan"),
                    }
                )

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("precision_pct", ascending=False).reset_index(drop=True)
        return result, base_rate

    def print_report(self):
        n_conditions = len(self.conditions)
        print("\n" + "=" * 100)
        print("CONDITION COMBINATION SEARCH (which indicator combos best match BUY/SELL)")
        print("=" * 100)
        print(f"Base conditions        : {n_conditions}")
        print(f"Combo sizes tried      : 1 to {self.max_combo_size}")
        print(f"Minimum support        : {self.min_support} candles (combos firing less often are dropped as noise)")
        print("precision%             : of the candles where ALL conditions in the combo were true,")
        print("                         what % actually got that oracle signal")
        print("lift                   : precision / base rate - how many times better than a random")
        print("                         guess this combo is (1.0x = no better than guessing)")

        for target in ["BUY", "SELL"]:
            result, base_rate = self.find_best_combinations(target)
            print(f"\n--- Best combinations for {target} (base rate: {base_rate:.1f}% of known candles) ---")

            if result.empty:
                print("  No combination met the minimum support.")
                continue

            header = f"{'#':>3} {'Conditions':<75} {'support':>8} {'precision%':>11} {'lift':>6}"
            print(header)
            print("-" * len(header))
            for i, row in result.head(self.top_n).iterrows():
                print(
                    f"{i + 1:>3} {row['conditions']:<75} {row['support']:>8} "
                    f"{row['precision_pct']:>10.1f}% {row['lift']:>5.2f}x"
                )

        print("\n" + "=" * 100 + "\n")
