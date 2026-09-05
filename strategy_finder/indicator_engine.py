"""Feature engineering: technical indicators and statistical features on OHLCV data."""

import numpy as np
import pandas as pd
import pandas_ta as ta
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats


class IndicatorEngine:
    """Builds the full technical-indicator/feature set on top of raw OHLCV data."""

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
            ("spectral (FFT) features", self._add_spectral_features),
            ("quant regime features", self._add_quant_regime_features),
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

        # close_return dropped: r=1.0 with return_1 (log vs simple return of the same
        # single-bar move), no downstream use - return_1 is what everything else here uses
        df["return_1"] = df["Close"].pct_change()
        df["return_5"] = df["Close"].pct_change(5)
        df["return_10"] = df["Close"].pct_change(10)
        df["return_20"] = df["Close"].pct_change(20)

        df["open_close_return"] = np.log(df["Close"] / df["Open"])
        # high_open_return dropped: unused in practice, redundant with open/low_open_return
        df["low_open_return"] = np.log(df["Low"] / df["Open"])

        df["log_volume"] = np.log(df["Volume"] + 1)
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        # volume_std dropped: never useful, no downstream dependency

        # daily_range/range_pct dropped: 99.5% correlated with each other, never useful

    # ------------------------------------------------------------------
    # Moving averages & trend
    # ------------------------------------------------------------------
    def _add_moving_averages(self):
        df = self.df

        # EMA_5 dropped: 0.93-correlated with EMA_10, near-duplicate, never useful
        ema_20 = ta.ema(df["Close"], length=20).bfill()
        sma_50 = ta.sma(df["Close"], length=50).bfill()
        vwap = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]).bfill()

        df["price_to_ema_20"] = (df["Close"] - ema_20) / ema_20
        df["price_to_sma_50"] = (df["Close"] - sma_50) / sma_50
        df["price_to_vwap"] = (df["Close"] - vwap) / vwap

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------
    def _add_momentum_indicators(self):
        df = self.df

        # RSI_14 dropped: correlated with RSI_7/RSI_21, never itself useful
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

        # ATR_7 dropped: correlated with ATR_14, never useful; ATR_14 kept as a dependency
        # ATR_21 dropped: 0.991-correlated with ATR_14, no other downstream use
        df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14).bfill()

        df["ATR_pct"] = (df["ATR_14"] / df["Close"]) * 100

        # BB_lower/upper, BB_middle, KC_lower/upper dropped from this engine entirely:
        # nothing here uses them (only PriceActionEngine's squeeze_on does, and it
        # computes its own copy as a local dependency - see its _ensure_base_indicators).
        # BB_width and BB_position dropped: neither was ever a useful condition.

        df["volatility_10"] = df["Close"].pct_change().rolling(10).std()
        # volatility_20/50 dropped: unused; volatility_10 alone drives downstream features

    # ------------------------------------------------------------------
    # Trend strength
    # ------------------------------------------------------------------
    def _add_trend_indicators(self):
        df = self.df

        # ADX_20/DMP_20/DMN_20 dropped: 0.93-0.98-correlated with the _14 versions, no other downstream use
        adx_df = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        df["ADX_14"] = adx_df["ADX_14"].bfill()
        df["DMP_14"] = adx_df["DMP_14"].bfill()
        df["DMN_14"] = adx_df["DMN_14"].bfill()

        df["directional_bias"] = df["DMP_14"] - df["DMN_14"]

        supertrend = ta.supertrend(df["High"], df["Low"], df["Close"], length=10, multiplier=3)
        # raw supertrend line kept as an instance attribute (trend_strength dependency
        # below), never a df column - it's a price level, >=0.98-correlated with the
        # EMA/SMA/VWAP cluster, so exposing it would just add a near-duplicate condition
        self._supertrend = supertrend["SUPERT_10_3"].bfill()
        df["supertrend_direction"] = supertrend["SUPERTd_10_3"].bfill()

        # st_flip dropped: rare + negative lift in combo search (kept internally - bars_since_flip needs it)
        st_flip = df["supertrend_direction"].diff().abs()
        df["bars_since_flip"] = df.groupby((st_flip == 2).cumsum()).cumcount()

        aroon = ta.aroon(df["High"], df["Low"], length=25)
        df["aroon_up"] = aroon["AROONU_25"].bfill()
        df["aroon_down"] = aroon["AROOND_25"].bfill()
        df["aroon_oscillator"] = df["aroon_up"] - df["aroon_down"]

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------
    def _add_volume_indicators(self):
        df = self.df

        # OBV/OBV_ema/AD dropped: redundant with each other, never useful
        # VPT dropped: 0.98+-correlated with the price-level cluster (cumulative,
        # tracks price direction over time in trending data), no downstream use
        df["CMF"] = ta.cmf(df["High"], df["Low"], df["Close"], df["Volume"], length=20).bfill()
        df["MFI"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14).bfill()

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

        # Raw multi-candle building blocks - deliberately atomic (no baked-in breakout/direction
        # logic) so the combo search's own lag_depths AND-ing can discover multi-bar sequences
        # itself (e.g. a wide-range candle N bars back + an inside bar recently), rather than us
        # hand-coding one fixed version of a "mother candle" pattern.
        rng = df["High"] - df["Low"]
        df["is_inside_bar"] = ((df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))).astype(int)
        df["is_wide_range_candle"] = (rng > 1.5 * df["ATR_14"]).astype(int)
        df["is_narrow_range_candle"] = (rng == rng.rolling(7).min()).astype(int)

        df["gap_down"] = (df["Open"] < df["Close"].shift(1)).astype(int)
        df["gap_size"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # ------------------------------------------------------------------
    # Statistical
    # ------------------------------------------------------------------
    def _add_statistical_features(self):
        df = self.df

        # zscore_20 dropped: correlated with zscore_10/50, never useful
        for period in [10, 50]:
            df[f"zscore_{period}"] = (df["Close"] - df["Close"].rolling(period).mean()) / (
                df["Close"].rolling(period).std() + 0.0001
            )

        df["skew_20"] = df["return_1"].rolling(20).skew()
        df["kurt_20"] = df["return_1"].rolling(20).kurt()
        # percentile_rank_20 dropped: never useful, and was the slowest computation in the engine

    # ------------------------------------------------------------------
    # Advanced volatility microstructure
    # ------------------------------------------------------------------
    def _add_advanced_volatility_features(self):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        new_features["realized_var_20"] = (df["return_1"] ** 2).rolling(20).sum()
        new_features["bipower_var"] = (abs(df["return_1"]) * abs(df["return_1"].shift())).rolling(20).sum()
        new_features["jump_strength"] = new_features["realized_var_20"] - new_features["bipower_var"]
        # vol_cluster/range_compression/vol_reversion_speed dropped: never useful, no dependents
        new_features["vol_regime"] = (df["volatility_10"] > df["volatility_10"].rolling(50).mean()).astype(int)
        rng = df["High"] - df["Low"]
        new_features["range_velocity"] = (rng - rng.shift(1)) / (rng.shift(1) + 0.0001)

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
        new_features["trend_strength"] = abs(df["Close"] - self._supertrend) / df["Close"]
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

        # buy_pressure dropped: identical formula to close_position
        new_features["slippage_proxy"] = (df["High"] - df["Low"]) / df["Close"].rolling(10).mean()
        new_features["stop_hunt_proxy"] = (df["High"] - df["Low"]) / (df["ATR_14"] + 0.0001)
        # amihud_illiquidity dropped: never a useful condition.

        self.df = pd.concat([df, new_features], axis=1)

    # ------------------------------------------------------------------
    # Spectral (FFT) features
    #
    # Every rolling window is transformed in ONE batched numpy FFT call
    # (np.fft.rfft/irfft operate along the last axis of a 2D array of
    # windows at once) instead of a per-row Python loop - the same trick
    # simulate_trades() uses for its bracket touch/exit windows.
    # ------------------------------------------------------------------
    @staticmethod
    def _rolling_spectrum(returns, window):
        """FFT power spectrum of every trailing `window`-bar slice of
        `returns` (demeaned + Hann-tapered per window to reduce spectral
        leakage - standard practice since these windows are analyzed for
        their power distribution, not reconstructed back to a value).
        Returns (power_no_dc, freq_bins) where power_no_dc excludes the
        zero-frequency (DC/mean) bin - that bin just reflects the window's
        average return, not any cyclical structure, and would otherwise
        dominate the "dominant frequency" pick on trending windows."""
        n = len(returns)
        if n < window:
            return None, 0
        windows = sliding_window_view(returns, window)
        windows = windows - windows.mean(axis=1, keepdims=True)
        windows = windows * np.hanning(window)[None, :]
        spectrum = np.fft.rfft(windows, axis=1)
        power = np.abs(spectrum) ** 2
        return power[:, 1:], power.shape[1] - 1

    def _rolling_fft_trend(self, price, window, keep_harmonics=3):
        """Causal FFT low-pass reconstruction: per trailing window, fit and
        remove the window's own linear trend (so the FFT isn't fooled by
        the sharp jump between a trending window's start and end - FFT
        implicitly treats the window as periodic, and that jump would
        otherwise leak into every frequency bin as spurious high-frequency
        content, i.e. ringing right at the edge we actually care about),
        zero out all but the lowest `keep_harmonics` frequencies of the
        now-detrended residual, inverse-FFT, add the trend back, and keep
        only the LAST point of each window - the current bar's FFT-smoothed
        trend value, comparable to an EMA/SMA but built from the window's
        own frequency content instead of a fixed decay shape."""
        n = len(price)
        if n < window:
            return np.full(n, np.nan)
        windows = sliding_window_view(price, window).astype(np.float64)
        x = np.arange(window, dtype=np.float64)
        x_centered = x - x.mean()
        denom = (x_centered**2).sum()

        y_mean = windows.mean(axis=1, keepdims=True)
        slope = (windows * x_centered[None, :]).sum(axis=1, keepdims=True) / denom
        intercept = y_mean - slope * x.mean()
        trend = slope * x[None, :] + intercept
        residual = windows - trend

        spectrum = np.fft.rfft(residual, axis=1)
        spectrum[:, keep_harmonics:] = 0
        reconstructed = np.fft.irfft(spectrum, n=window, axis=1) + trend
        return np.concatenate([np.full(window - 1, np.nan), reconstructed[:, -1]])

    def _add_spectral_features(self, window=32):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        power_no_dc, n_bins = self._rolling_spectrum(df["return_1"].fillna(0).to_numpy(), window)
        pad = window - 1
        if power_no_dc is None:
            for col in ["fft_dominant_cycle", "fft_dominant_power_ratio", "fft_spectral_entropy", "fft_low_freq_power_ratio"]:
                new_features[col] = np.nan
        else:
            total_power = power_no_dc.sum(axis=1) + 1e-12
            dominant_bin = np.argmax(power_no_dc, axis=1) + 1  # +1: bin index in the original (DC-included) spectrum
            dominant_power = power_no_dc[np.arange(len(power_no_dc)), dominant_bin - 1]

            p = power_no_dc / total_power[:, None]
            spectral_entropy = -np.sum(p * np.log(p + 1e-12), axis=1) / np.log(n_bins)

            low_cut = max(1, n_bins // 4)
            low_freq_ratio = power_no_dc[:, :low_cut].sum(axis=1) / total_power

            new_features["fft_dominant_cycle"] = np.concatenate([np.full(pad, np.nan), window / dominant_bin])
            new_features["fft_dominant_power_ratio"] = np.concatenate([np.full(pad, np.nan), dominant_power / total_power])
            new_features["fft_spectral_entropy"] = np.concatenate([np.full(pad, np.nan), spectral_entropy])
            new_features["fft_low_freq_power_ratio"] = np.concatenate([np.full(pad, np.nan), low_freq_ratio])

        fft_trend = self._rolling_fft_trend(df["Close"].to_numpy(), window)
        new_features["price_to_fft_trend"] = (df["Close"].to_numpy() - fft_trend) / fft_trend

        self.df = pd.concat([df, new_features], axis=1)

    # ------------------------------------------------------------------
    # Quant regime features: trending-vs-mean-reverting classifiers used in
    # stat-arb / systematic trading, none of them specific to any one asset.
    # ------------------------------------------------------------------
    @staticmethod
    def _hurst_exponent(x, n_lags=8):
        """Generalized Hurst exponent via the variance-lag method (Di Matteo
        et al.): for fractional Brownian motion, std(x[t+lag]-x[t]) scales as
        lag^H, so H is the slope of log(std(diff)) vs log(lag). H > 0.5 =
        trending/persistent, H < 0.5 = mean-reverting, H ~= 0.5 = random walk.
        Cheaper and more stable on short windows than classical rescaled-range
        (R/S) analysis, at the cost of being a rougher estimate."""
        n = len(x)
        if n < 20 or np.all(x == x[0]):
            return np.nan
        lags = np.unique(np.floor(np.logspace(np.log10(2), np.log10(max(3, n // 2)), num=n_lags)).astype(int))
        lags = lags[lags >= 2]
        if len(lags) < 2:
            return np.nan
        tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
        valid = tau > 0
        if valid.sum() < 2:
            return np.nan
        slope, _intercept, _r, _p, _se = stats.linregress(np.log(lags[valid]), np.log(tau[valid]))
        return slope

    def _add_quant_regime_features(self, hurst_window=64, ou_window=30, fisher_window=10):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        # Hurst exponent: no closed-form rolling formula (needs a per-window
        # multi-lag fit), so this is a rolling-apply like price_entropy/
        # dir_entropy above - same accepted cost profile as those.
        new_features["hurst_exponent"] = df["Close"].rolling(hurst_window).apply(self._hurst_exponent, raw=True)

        # Ornstein-Uhlenbeck mean-reversion speed: rolling AR(1) regression of
        # Close[t]-Close[t-1] on Close[t-1] (pandas' rolling .cov()/.var() are
        # themselves vectorized, no python loop). lambda < 0 = mean-reverting
        # (more negative = faster reversion), lambda > 0 = trending/momentum.
        lag = df["Close"].shift(1)
        diff = df["Close"].diff()
        roll_cov = lag.rolling(ou_window).cov(diff)
        roll_var = lag.rolling(ou_window).var()
        ou_lambda = roll_cov / (roll_var + 1e-9)
        new_features["ou_lambda"] = ou_lambda
        # Half-life in bars if mean-reverting, hard-capped at 500 (not just
        # gated on lambda's sign - a lambda barely below zero gives a huge
        # half-life too) so this stays bounded instead of leaving NaN/inf
        # for downstream code to trip over, consistent with this file's
        # epsilon-guard convention.
        raw_half_life = np.where(ou_lambda < -1e-8, -np.log(2) / ou_lambda, 500.0)
        new_features["ou_half_life"] = np.minimum(raw_half_life, 500.0)

        # Fisher Transform (Ehlers): maps price's position within its recent
        # range through an inverse-hyperbolic-tangent-shaped curve, sharpening
        # turning points into more extreme, more clearly-separated values than
        # the raw stochastic-style position itself.
        llv = df["Low"].rolling(fisher_window).min()
        hhv = df["High"].rolling(fisher_window).max()
        raw_value = (2 * ((df["Close"] - llv) / (hhv - llv + 1e-9) - 0.5)).clip(-0.999, 0.999)
        fisher = 0.5 * np.log((1 + raw_value) / (1 - raw_value))
        new_features["fisher_transform"] = fisher.ewm(span=3, adjust=False).mean()

        self.df = pd.concat([df, new_features], axis=1)

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------
    def _add_interaction_features(self):
        df = self.df
        new_features = pd.DataFrame(index=df.index)

        # rsi_vol/rsi_atr/bb_rsi dropped: depended on already-cut RSI_14/BB_position
        new_features["trend_volume"] = df["trend_strength"] * df["volume_ratio"]
        new_features["adx_volume"] = df["ADX_14"] * df["volume_ratio"]
        new_features["vol_atr_ratio"] = df["volume_ratio"] / (df["ATR_pct"] + 0.0001)

        self.df = pd.concat([df, new_features], axis=1)
