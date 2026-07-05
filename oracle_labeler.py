"""Oracle labeling: forward-looking ground-truth bands computed directly from
historical OHLC data.

For every candle, this looks ahead `lookahead` candles and records what actually
happened next - the real max upside and max downside. That becomes the "source of
truth" used later to check how accurate any indicator/strategy/model prediction was,
since it is derived purely from the historical data itself, not from a prediction.
"""

import numpy as np
import pandas as pd


class OracleLabeler:
    """Adds forward-looking upper/lower bands and a ground-truth direction label.

    Usage:
        df = OracleLabeler(df, lookahead=20).label()
    """

    def __init__(self, df, lookahead=20, neutral_threshold_pct=0.5):
        self.df = df.copy()
        self.lookahead = lookahead
        self.neutral_threshold_pct = neutral_threshold_pct

    def label(self):
        df = self.df
        lookahead = self.lookahead

        # Actual price levels reached in the NEXT `lookahead` candles (excludes the
        # current candle itself - only real future data).
        df["oracle_upper_band"] = self._forward_max(df["High"], lookahead)
        df["oracle_lower_band"] = self._forward_min(df["Low"], lookahead)

        # Max possible gain / loss (%) from the current close, based on what really happened.
        df["oracle_upside_pct"] = (df["oracle_upper_band"] - df["Close"]) / df["Close"] * 100
        df["oracle_downside_pct"] = (df["oracle_lower_band"] - df["Close"]) / df["Close"] * 100

        # The dominant, actual movement from this candle (signed): whichever of
        # upside/downside was bigger in magnitude. This is the single number that
        # tells you "what really moved next" for this candle - the source of truth.
        upside_dominant = df["oracle_upside_pct"] >= df["oracle_downside_pct"].abs()
        df["oracle_actual_move_pct"] = np.where(upside_dominant, df["oracle_upside_pct"], df["oracle_downside_pct"])

        df["oracle_direction"] = self._build_direction(df["oracle_actual_move_pct"])

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

    @staticmethod
    def _forward_max(series, window):
        """Max of the NEXT `window` values after each row (current row excluded)."""
        return series[::-1].rolling(window=window, min_periods=window).max()[::-1].shift(-1)

    @staticmethod
    def _forward_min(series, window):
        """Min of the NEXT `window` values after each row (current row excluded)."""
        return series[::-1].rolling(window=window, min_periods=window).min()[::-1].shift(-1)
