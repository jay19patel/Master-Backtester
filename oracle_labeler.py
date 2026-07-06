"""Oracle labeling: forward-looking ground-truth bands computed directly from
historical OHLC data.

For every candle, this looks ahead `lookahead` candles and records what actually
happened next - the real max upside and max downside. That becomes the "source of
truth" used later to check how accurate any indicator/strategy/model prediction was,
since it is derived purely from the historical data itself, not from a prediction.

Example: close is 100, and over the next 20 candles price went as high as 150 and
as low as 80.
    oracle_upper_band   = 150            (the max it reached)
    oracle_lower_band   = 80             (the min it reached)
    oracle_range        = 150 - 80 = 70  (gap between the max and the min)
    oracle_upside_gap   = 150 - 100 = 50 (gap from close up to the max)
    oracle_downside_gap = 100 - 80  = 20 (gap from close down to the min)
Here the upside gap (50) is bigger than the downside gap (20), so this candle's
dominant/actual move is UP. Flip the example (close 100, max 120, min 60) and the
downside gap (100 - 60 = 40) beats the upside gap (120 - 100 = 20), so that candle's
dominant move is DOWN.

On top of that, `oracle_signal` turns the two gaps into a trade call using a minimum
reward:risk ratio (default 1:2 - the winning side must be at least double the losing
side): upside_gap 50 vs downside_gap 20 -> 50 >= 2 x 20 -> BUY. If it were upside_gap
30 vs downside_gap 20, that is less than double, so it is neither a clean BUY nor
SELL -> HOLD.
"""

import numpy as np
import pandas as pd


class OracleLabeler:
    """Adds forward-looking upper/lower bands and a ground-truth direction label.

    Usage:
        df = OracleLabeler(df, lookahead=20).label()
    """

    def __init__(self, df, lookahead=20, neutral_threshold_pct=0.5, min_reward_risk_ratio=2.0):
        self.df = df.copy()
        self.lookahead = lookahead
        self.neutral_threshold_pct = neutral_threshold_pct
        self.min_reward_risk_ratio = min_reward_risk_ratio

    def label(self):
        df = self.df
        lookahead = self.lookahead

        # Actual price levels reached in the NEXT `lookahead` candles (excludes the
        # current candle itself - only real future data).
        df["oracle_upper_band"] = self._forward_max(df["High"], lookahead)
        df["oracle_lower_band"] = self._forward_min(df["Low"], lookahead)

        # Raw price gaps (same units as Close, not %) - see the module docstring example.
        df["oracle_range"] = df["oracle_upper_band"] - df["oracle_lower_band"]
        df["oracle_upside_gap"] = df["oracle_upper_band"] - df["Close"]
        df["oracle_downside_gap"] = df["Close"] - df["oracle_lower_band"]

        # Max possible gain / loss (%) from the current close, based on what really happened.
        df["oracle_upside_pct"] = (df["oracle_upper_band"] - df["Close"]) / df["Close"] * 100
        df["oracle_downside_pct"] = (df["oracle_lower_band"] - df["Close"]) / df["Close"] * 100

        # The dominant, actual movement from this candle (signed): whichever of
        # upside/downside was bigger in magnitude. This is the single number that
        # tells you "what really moved next" for this candle - the source of truth.
        # (Comparing the % columns or the raw gap columns gives the same winner,
        # since both sides are divided by the same Close.)
        upside_dominant = df["oracle_upside_pct"] >= df["oracle_downside_pct"].abs()
        df["oracle_actual_move_pct"] = np.where(upside_dominant, df["oracle_upside_pct"], df["oracle_downside_pct"])

        df["oracle_direction"] = self._build_direction(df["oracle_actual_move_pct"])

        # Trade call: BUY/SELL only when one side's gap clears the other by at least
        # `min_reward_risk_ratio` (1:2 by default), otherwise HOLD - see module docstring.
        df["oracle_signal"] = self._build_signal(df["oracle_upside_gap"], df["oracle_downside_gap"])

        self.df = df
        known = df["oracle_direction"].notna().sum()
        print(
            f"[OracleLabeler] Labeled {len(df)} candles with a {lookahead}-candle forward window "
            f"({known} have a known outcome, last {len(df) - known} are undecided - no future data yet)."
        )
        return self.df

    def _build_direction(self, actual_move_pct):
        """1 = UP dominant, -1 = DOWN dominant, 0 = NEUTRAL/chop, NaN = unknown (tail rows)."""
        direction = pd.Series(np.nan, index=actual_move_pct.index)
        valid = actual_move_pct.notna()

        direction.loc[valid & (actual_move_pct >= self.neutral_threshold_pct)] = 1
        direction.loc[valid & (actual_move_pct <= -self.neutral_threshold_pct)] = -1
        direction.loc[valid & (actual_move_pct.abs() < self.neutral_threshold_pct)] = 0

        return direction

    def _build_signal(self, upside_gap, downside_gap):
        """BUY when the upside gap is at least `min_reward_risk_ratio` times the
        downside gap, SELL when it's the other way round, HOLD when neither side
        clears that bar, NaN for candles with no known outcome yet (tail rows)."""
        signal = pd.Series(np.nan, index=upside_gap.index, dtype=object)
        known = upside_gap.notna() & downside_gap.notna()

        buy = known & (upside_gap > 0) & (upside_gap >= self.min_reward_risk_ratio * downside_gap)
        sell = known & (downside_gap > 0) & (downside_gap >= self.min_reward_risk_ratio * upside_gap)
        hold = known & ~buy & ~sell

        signal.loc[buy] = "BUY"
        signal.loc[sell] = "SELL"
        signal.loc[hold] = "HOLD"

        return signal

    @staticmethod
    def _forward_max(series, window):
        """Max of the NEXT `window` values after each row (current row excluded)."""
        return series[::-1].rolling(window=window, min_periods=window).max()[::-1].shift(-1)

    @staticmethod
    def _forward_min(series, window):
        """Min of the NEXT `window` values after each row (current row excluded)."""
        return series[::-1].rolling(window=window, min_periods=window).min()[::-1].shift(-1)
