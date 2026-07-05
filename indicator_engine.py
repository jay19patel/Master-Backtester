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
        df["high_open_return"] = np.log(df["High"] / df["Open"])
        df["low_open_return"] = np.log(df["Low"] / df["Open"])

        df["log_volume"] = np.log(df["Volume"] + 1)
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["volume_std"] = df["Volume"].rolling(20).std()

        df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["range_pct"] = ((df["High"] - df["Low"]) / df["Open"]) * 100

    # ------------------------------------------------------------------
    # Moving averages & trend
    # ------------------------------------------------------------------
    def _add_moving_averages(self):
        df = self.df

        for period in [5, 10, 20, 50, 100]:
            df[f"EMA_{period}"] = ta.ema(df["Close"], length=period).bfill()

        for period in [20, 50, 100, 200]:
            df[f"SMA_{period}"] = ta.sma(df["Close"], length=period).bfill()

        df["price_to_ema_5"] = (df["Close"] - df["EMA_5"]) / df["EMA_5"]
        df["price_to_ema_20"] = (df["Close"] - df["EMA_20"]) / df["EMA_20"]
        df["price_to_sma_50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]

        df["ema_5_10_cross"] = df["EMA_5"] - df["EMA_10"]
        df["ema_10_20_cross"] = df["EMA_10"] - df["EMA_20"]

        df["VWAP"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]).bfill()
        df["price_to_vwap"] = (df["Close"] - df["VWAP"]) / df["VWAP"]

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------
    def _add_momentum_indicators(self):
        df = self.df

        for period in [7, 14, 21]:
            df[f"RSI_{period}"] = ta.rsi(df["Close"], length=period).bfill()

        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        df["MACD"] = macd["MACD_12_26_9"].bfill()
        df["MACD_signal"] = macd["MACDs_12_26_9"].bfill()
        df["MACD_hist"] = macd["MACDh_12_26_9"].bfill()

        stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
        df["Stoch_K"] = stoch["STOCHk_14_3_3"].bfill()
        df["Stoch_D"] = stoch["STOCHd_14_3_3"].bfill()

        df["CCI_20"] = ta.cci(df["High"], df["Low"], df["Close"], length=20).bfill()
        df["WilliamsR_14"] = ta.willr(df["High"], df["Low"], df["Close"], length=14).bfill()
        df["ROC_10"] = ta.roc(df["Close"], length=10).bfill()
        df["ROC_20"] = ta.roc(df["Close"], length=20).bfill()

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------
    def _add_volatility_indicators(self):
        df = self.df

        for period in [7, 14, 21]:
            df[f"ATR_{period}"] = ta.atr(df["High"], df["Low"], df["Close"], length=period).bfill()

        df["ATR_pct"] = (df["ATR_14"] / df["Close"]) * 100

        bbands = ta.bbands(df["Close"], length=20, std=2)
        if bbands is not None:
            df["BB_lower"] = bbands.iloc[:, 0].bfill()
            df["BB_middle"] = bbands.iloc[:, 1].bfill()
            df["BB_upper"] = bbands.iloc[:, 2].bfill()

        df["BB_width"] = ((df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]) * 100
        df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

        kc = ta.kc(df["High"], df["Low"], df["Close"], length=20, scalar=2)
        if kc is not None:
            df["KC_lower"] = kc.iloc[:, 0].bfill()
            df["KC_upper"] = kc.iloc[:, 2].bfill()

        df["volatility_10"] = df["Close"].pct_change().rolling(10).std()
        df["volatility_20"] = df["Close"].pct_change().rolling(20).std()
        df["volatility_50"] = df["Close"].pct_change().rolling(50).std()

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

        df["OBV"] = ta.obv(df["Close"], df["Volume"]).bfill()
        df["OBV_ema"] = ta.ema(df["OBV"], length=20).bfill()
        df["AD"] = ta.ad(df["High"], df["Low"], df["Close"], df["Volume"]).bfill()
        df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20).bfill()
        df["MFI"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14).bfill()
        df["VPT"] = ta.pvt(df["Close"], df["Volume"]).bfill()

    # ------------------------------------------------------------------
    # Candle patterns
    # ------------------------------------------------------------------
    def _add_candle_features(self):
        df = self.df

        df["body_size"] = abs(df["Close"] - df["Open"]) / df["Close"]
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

        for period in [10, 20, 50]:
            df[f"zscore_{period}"] = (df["Close"] - df["Close"].rolling(period).mean()) / (
                df["Close"].rolling(period).std() + 0.0001
            )

        df["skew_20"] = df["return_1"].rolling(20).skew()
        df["kurt_20"] = df["return_1"].rolling(20).kurt()
        df["percentile_rank_20"] = df["Close"].rolling(20).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 0 else 0.5
        )

    # ------------------------------------------------------------------
    # Advanced volatility microstructure
    # ------------------------------------------------------------------
    def _add_advanced_volatility_features(self):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        new_features["realized_var_20"] = (df["return_1"] ** 2).rolling(20).sum()
        new_features["bipower_var"] = (abs(df["return_1"]) * abs(df["return_1"].shift())).rolling(20).sum()
        new_features["jump_strength"] = new_features["realized_var_20"] - new_features["bipower_var"]
        new_features["vol_cluster"] = df["volatility_10"].rolling(20).std()
        new_features["vol_regime"] = (df["volatility_10"] > df["volatility_10"].rolling(50).mean()).astype(int)
        new_features["range_compression"] = (df["High"] - df["Low"]).rolling(10).mean() / (
            df["High"] - df["Low"]
        ).rolling(50).mean()
        new_features["range_velocity"] = (df["High"] - df["Low"]).pct_change()
        new_features["fractal_proxy"] = df["ATR_pct"] / (df["volatility_10"] + 0.0001)
        new_features["vol_reversion_speed"] = (df["volatility_10"] - df["volatility_10"].shift(10)) / 10

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

        new_features["buy_pressure"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 0.0001)
        new_features["slippage_proxy"] = (df["High"] - df["Low"]) / df["Close"].rolling(10).mean()
        new_features["stop_hunt_proxy"] = (df["High"] - df["Low"]) / (df["ATR_14"] + 0.0001)
        new_features["amihud_illiquidity"] = abs(df["return_1"]) / (df["Volume"] + 1)

        self.df = pd.concat([df, new_features], axis=1)

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------
    def _add_interaction_features(self):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        new_features["rsi_vol"] = df["RSI_14"] * df["volatility_10"]
        new_features["rsi_atr"] = df["RSI_14"] * df["ATR_pct"]
        new_features["trend_volume"] = df["trend_strength"] * df["volume_ratio"]
        new_features["adx_volume"] = df["ADX_14"] * df["volume_ratio"]
        new_features["bb_rsi"] = df["BB_position"] * df["RSI_14"]
        new_features["vol_atr_ratio"] = df["volume_ratio"] / (df["ATR_pct"] + 0.0001)

        self.df = pd.concat([df, new_features], axis=1)
