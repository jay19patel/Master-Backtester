"""Feature engineering: technical indicators and statistical features on OHLCV data."""

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats


class IndicatorEngine:
    """Builds a full technical-indicator / feature set on top of raw OHLCV data.

    Usage:
        df = IndicatorEngine(raw_df).build()
    """

    def __init__(self, df):
        self.df = df.copy()

    def build(self):
        """Run the full feature pipeline and return the enriched DataFrame."""
        steps = [
            ("basic price/volume features", self._add_basic_features),
            ("moving averages", self._add_moving_averages),
            ("momentum indicators", self._add_momentum_indicators),
            ("volatility indicators", self._add_volatility_indicators),
            ("trend indicators", self._add_trend_indicators),
            ("volume indicators", self._add_volume_indicators),
            ("candle features", self._add_candle_features),
            ("statistical features", self._add_statistical_features),
            ("advanced volatility features", self._add_advanced_volatility_features),
            ("advanced trend features", self._add_advanced_trend_features),
            ("information theory features", self._add_information_theory_features),
            ("microstructure features", self._add_microstructure_features),
            ("interaction features", self._add_interaction_features),
        ]

        for label, step in steps:
            print(f"[IndicatorEngine] Adding {label}...")
            step()

        self.df = self.df.bfill().ffill()
        print(f"[IndicatorEngine] Done. Total columns: {len(self.df.columns)}")
        return self.df

    # ------------------------------------------------------------------
    # Basic price & volume features
    # ------------------------------------------------------------------
    def _add_basic_features(self):
        df = self.df

        df["close_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["return_1"] = df["Close"].pct_change()
        df["return_5"] = df["Close"].pct_change(5)
        df["return_10"] = df["Close"].pct_change(10)
        df["return_20"] = df["Close"].pct_change(20)

        df["open_close_return"] = np.log(df["Close"] / df["Open"])
        # high_open_return dropped: unused by every strategy the combo search has
        # ever found useful, and redundant with open_close_return/low_open_return
        # (same corner-return family).
        df["low_open_return"] = np.log(df["Low"] / df["Open"])

        df["log_volume"] = np.log(df["Volume"] + 1)
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        # volume_std dropped: never a useful condition on its own, no downstream dependency.

        # daily_range and range_pct dropped: 99.5% correlated with each other
        # (High-Low over Close vs over Open - barely differs) and neither was
        # ever a useful condition in the combo search.

    # ------------------------------------------------------------------
    # Moving averages & trend
    # ------------------------------------------------------------------
    def _add_moving_averages(self):
        df = self.df

        # EMA_5 dropped: 0.93-correlated with EMA_10 as a >median condition
        # (near-duplicate search result) and never itself a useful condition.
        for period in [10, 20, 50, 100]:
            df[f"EMA_{period}"] = ta.ema(df["Close"], length=period).bfill()

        # SMA_200 dropped: never a useful condition; SMA_50 kept only as the
        # price_to_sma_50 dependency (SMA_20/100 are used directly).
        for period in [20, 50, 100]:
            df[f"SMA_{period}"] = ta.sma(df["Close"], length=period).bfill()

        df["price_to_ema_20"] = (df["Close"] - df["EMA_20"]) / df["EMA_20"]
        df["price_to_sma_50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]

        df["ema_10_20_cross"] = df["EMA_10"] - df["EMA_20"]

        df["VWAP"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]).bfill()
        df["price_to_vwap"] = (df["Close"] - df["VWAP"]) / df["VWAP"]

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------
    def _add_momentum_indicators(self):
        df = self.df

        # RSI_14 dropped: 0.76-0.87 correlated with RSI_7/RSI_21 as a >median
        # condition (near-duplicate) and never itself a useful condition -
        # RSI_7 and RSI_21 already cover the short/long ends of this family.
        for period in [7, 21]:
            df[f"RSI_{period}"] = ta.rsi(df["Close"], length=period).bfill()

        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        df["MACD"] = macd["MACD_12_26_9"].bfill()
        df["MACD_signal"] = macd["MACDs_12_26_9"].bfill()
        df["MACD_hist"] = macd["MACDh_12_26_9"].bfill()

        stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
        df["Stoch_K"] = stoch["STOCHk_14_3_3"].bfill()
        # Stoch_D dropped: never a useful condition (Stoch_K already covers this indicator).

        # CCI_20 dropped: never a useful condition.
        df["WilliamsR_14"] = ta.willr(df["High"], df["Low"], df["Close"], length=14).bfill()

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------
    def _add_volatility_indicators(self):
        df = self.df

        # ATR_7 dropped: 0.81-correlated with ATR_14 as a >median condition
        # and never itself a useful condition; ATR_14 stays as a dependency
        # for ATR_pct/stop_hunt_proxy, ATR_21 stays as its own condition.
        for period in [14, 21]:
            df[f"ATR_{period}"] = ta.atr(df["High"], df["Low"], df["Close"], length=period).bfill()

        df["ATR_pct"] = (df["ATR_14"] / df["Close"]) * 100

        bbands = ta.bbands(df["Close"], length=20, std=2)
        if bbands is not None:
            df["BB_lower"] = bbands.iloc[:, 0].bfill()
            df["BB_middle"] = bbands.iloc[:, 1].bfill()
            df["BB_upper"] = bbands.iloc[:, 2].bfill()

        # BB_width and BB_position dropped: neither was ever a useful condition.

        kc = ta.kc(df["High"], df["Low"], df["Close"], length=20, scalar=2)
        if kc is not None:
            df["KC_lower"] = kc.iloc[:, 0].bfill()
            df["KC_upper"] = kc.iloc[:, 2].bfill()

        df["volatility_10"] = df["Close"].pct_change().rolling(10).std()
        # volatility_20/volatility_50 dropped: never used as a standalone
        # condition or dependency; volatility_10 alone drives vol_regime,
        # fractal_proxy, and shock_elasticity.

    # ------------------------------------------------------------------
    # Trend strength
    # ------------------------------------------------------------------
    def _add_trend_indicators(self):
        df = self.df

        for period in [14, 20]:
            adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=period)
            df[f"ADX_{period}"] = adx_df[f"ADX_{period}"].bfill()
            df[f"DMP_{period}"] = adx_df[f"DMP_{period}"].bfill()
            df[f"DMN_{period}"] = adx_df[f"DMN_{period}"].bfill()

        df["directional_bias"] = df["DMP_14"] - df["DMN_14"]

        supertrend = ta.supertrend(df["High"], df["Low"], df["Close"], length=10, multiplier=3)
        df["supertrend"] = supertrend["SUPERT_10_3"].bfill()
        df["supertrend_direction"] = supertrend["SUPERTd_10_3"].bfill()

        df["st_flip"] = df["supertrend_direction"].diff().abs()
        df["bars_since_flip"] = df.groupby((df["st_flip"] == 2).cumsum()).cumcount()

        aroon = ta.aroon(df["High"], df["Low"], length=25)
        df["aroon_up"] = aroon["AROONU_25"].bfill()
        df["aroon_down"] = aroon["AROOND_25"].bfill()
        df["aroon_oscillator"] = df["aroon_up"] - df["aroon_down"]

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------
    def _add_volume_indicators(self):
        df = self.df

        # OBV/OBV_ema dropped: 0.80 condition-agreement with each other, and
        # neither was ever a useful condition. AD dropped: never used.
        df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20).bfill()
        df["MFI"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14).bfill()
        df["VPT"] = ta.pvt(df["Close"], df["Volume"]).bfill()

    # ------------------------------------------------------------------
    # Candle patterns
    # ------------------------------------------------------------------
    def _add_candle_features(self):
        df = self.df

        # body_size dropped: never a useful condition, no downstream dependency.
        df["upper_wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["lower_wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["total_wick"] = df["upper_wick"] + df["lower_wick"]

        df["wick_imbalance"] = (df["upper_wick"] - df["lower_wick"]) / df["Close"]
        df["wick_to_body"] = df["total_wick"] / (abs(df["Close"] - df["Open"]) + 0.0001)

        df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 0.0001)

        df["is_bullish"] = (df["Close"] > df["Open"]).astype(int)
        df["candle_strength"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 0.0001)

        df["gap_up"] = (df["Open"] > df["Close"].shift(1)).astype(int)
        df["gap_down"] = (df["Open"] < df["Close"].shift(1)).astype(int)
        df["gap_size"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # ------------------------------------------------------------------
    # Statistical
    # ------------------------------------------------------------------
    def _add_statistical_features(self):
        df = self.df

        # zscore_20 dropped: moderately correlated with kept zscore_10/zscore_50
        # and never itself a useful condition.
        for period in [10, 50]:
            df[f"zscore_{period}"] = (df["Close"] - df["Close"].rolling(period).mean()) / (
                df["Close"].rolling(period).std() + 0.0001
            )

        df["skew_20"] = df["return_1"].rolling(20).skew()
        df["kurt_20"] = df["return_1"].rolling(20).kurt()
        # percentile_rank_20 dropped: never a useful condition, and it was the
        # single slowest computation in the whole engine (scipy percentileofscore
        # inside a rolling(20).apply()).

    # ------------------------------------------------------------------
    # Advanced volatility microstructure
    # ------------------------------------------------------------------
    def _add_advanced_volatility_features(self):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        new_features["realized_var_20"] = (df["return_1"] ** 2).rolling(20).sum()
        new_features["bipower_var"] = (abs(df["return_1"]) * abs(df["return_1"].shift())).rolling(20).sum()
        new_features["jump_strength"] = new_features["realized_var_20"] - new_features["bipower_var"]
        # vol_cluster, range_compression, vol_reversion_speed dropped: none was
        # ever a useful condition and none has a downstream dependency.
        new_features["vol_regime"] = (df["volatility_10"] > df["volatility_10"].rolling(50).mean()).astype(int)
        rng = df["High"] - df["Low"]
        new_features["range_velocity"] = (rng - rng.shift(1)) / (rng.shift(1) + 0.0001)
        new_features["fractal_proxy"] = df["ATR_pct"] / (df["volatility_10"] + 0.0001)

        self.df = pd.concat([df, new_features], axis=1)

    # ------------------------------------------------------------------
    # Advanced trend & momentum microstructure
    # ------------------------------------------------------------------
    def _add_advanced_trend_features(self):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        new_features["efficiency_ratio"] = abs(df["Close"] - df["Close"].shift(10)) / (
            df["High"].rolling(10).max() - df["Low"].rolling(10).min() + 0.0001
        )
        new_features["trend_persistence"] = np.sign(df["return_1"]).rolling(10).sum()
        new_features["trend_smoothness"] = abs(df["Close"] - df["Close"].shift(20)) / (
            df["return_1"].rolling(20).std() + 0.0001
        )
        new_features["path_curvature"] = df["return_1"].diff().abs().rolling(10).mean()
        new_features["trend_strength"] = abs(df["Close"] - df["supertrend"]) / df["Close"]
        new_features["trend_acceleration"] = new_features["trend_strength"].diff()
        new_features["dir_entropy"] = df["return_1"].rolling(20).apply(
            lambda x: -np.mean(np.sign(x) * np.log(np.abs(np.sign(x)) + 1e-6))
        )

        self.df = pd.concat([df, new_features], axis=1)

    # ------------------------------------------------------------------
    # Information theory
    # ------------------------------------------------------------------
    def _add_information_theory_features(self):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        new_features["price_entropy"] = df["return_1"].rolling(20).apply(
            lambda x: stats.entropy(np.histogram(x, bins=5)[0] + 1) if len(x) > 0 else 0,
            raw=False,
        )
        new_features["surprise"] = (df["return_1"] - df["return_1"].rolling(20).mean()) / (
            df["return_1"].rolling(20).std() + 1e-6
        )
        new_features["shock_elasticity"] = df["return_1"].abs() / (df["volatility_10"] + 1e-6)

        self.df = pd.concat([df, new_features], axis=1)

    # ------------------------------------------------------------------
    # Market microstructure & liquidity
    # ------------------------------------------------------------------
    def _add_microstructure_features(self):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        # buy_pressure dropped: row-for-row identical formula to close_position
        # (Close-Low)/(High-Low), already added in _add_candle_features.
        new_features["slippage_proxy"] = (df["High"] - df["Low"]) / df["Close"].rolling(10).mean()
        new_features["stop_hunt_proxy"] = (df["High"] - df["Low"]) / (df["ATR_14"] + 0.0001)
        # amihud_illiquidity dropped: never a useful condition.

        self.df = pd.concat([df, new_features], axis=1)

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------
    def _add_interaction_features(self):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        # rsi_vol, rsi_atr, bb_rsi dropped: all depended on RSI_14/BB_position,
        # both already cut above, and none was ever a useful condition itself.
        new_features["trend_volume"] = df["trend_strength"] * df["volume_ratio"]
        new_features["adx_volume"] = df["ADX_14"] * df["volume_ratio"]
        new_features["vol_atr_ratio"] = df["volume_ratio"] / (df["ATR_pct"] + 0.0001)

        self.df = pd.concat([df, new_features], axis=1)
