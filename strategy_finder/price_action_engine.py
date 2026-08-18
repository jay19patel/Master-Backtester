"""Price-action engine: swing points, market structure (BOS/CHoCH), breakouts,
smart-money concepts (order blocks, FVG, sweeps), fibonacci, pullbacks, and
range/combination signals. Every strategy writes one `sig_<name>` column (+1
buy, -1 sell, 0 none), firing only once a setup is CONFIRMED - swing points
are shifted `right` bars after the pivot, so there's no lookahead.

Usage: `df = PriceActionEngine(df).build()`, or chain individual `add_*` methods.
"""

import numpy as np
import pandas_ta as ta


class PriceActionEngine:
    """Adds price-action structure columns and discrete trade signals (sig_*)."""

    def __init__(self, df, swing_left=3, swing_right=3, max_zone_age=50, max_active_zones=3):
        """max_zone_age: bars before an unfilled OB/FVG zone is dropped as stale.
        max_active_zones: concurrent zones tracked per side (bull/bear)."""
        self.df = df.copy()
        self.swing_left = swing_left
        self.swing_right = swing_right
        self.max_zone_age = max_zone_age
        self.max_active_zones = max_active_zones
        self._ensure_base_indicators()

    # ------------------------------------------------------------------
    # Base indicators (only computed if IndicatorEngine hasn't already)
    # ------------------------------------------------------------------
    def _ensure_base_indicators(self):
        """Columns that are also legitimate standalone search conditions (ATR_14,
        MACD family, ADX_14/DMP_14/DMN_14, supertrend_direction, volume_ratio) are
        written to df as usual. Raw price-level dependencies (EMA_10/20/50, RSI_14,
        VWAP, BB/KC bands) are kept as instance attributes instead - never written
        to df - since they're all >=0.98-correlated with each other (just different
        smoothings of Close) and would only add near-duplicate conditions to the
        combo search. Reused from df if some earlier step already computed them."""
        df = self.df

        self._ema_10 = df["EMA_10"] if "EMA_10" in df.columns else ta.ema(df["Close"], length=10).bfill()
        self._ema_20 = df["EMA_20"] if "EMA_20" in df.columns else ta.ema(df["Close"], length=20).bfill()
        self._ema_50 = df["EMA_50"] if "EMA_50" in df.columns else ta.ema(df["Close"], length=50).bfill()
        self._rsi_14 = df["RSI_14"] if "RSI_14" in df.columns else ta.rsi(df["Close"], length=14).bfill()
        self._vwap = (
            df["VWAP"] if "VWAP" in df.columns else ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]).bfill()
        )

        if "BB_upper" in df.columns:
            self._bb_lower, self._bb_upper = df["BB_lower"], df["BB_upper"]
        else:
            bb = ta.bbands(df["Close"], length=20, std=2)
            self._bb_lower, self._bb_upper = bb.iloc[:, 0].bfill(), bb.iloc[:, 2].bfill()

        if "KC_upper" in df.columns:
            self._kc_lower, self._kc_upper = df["KC_lower"], df["KC_upper"]
        else:
            kc = ta.kc(df["High"], df["Low"], df["Close"], length=20, scalar=2)
            self._kc_lower, self._kc_upper = kc.iloc[:, 0].bfill(), kc.iloc[:, 2].bfill()

        if "ATR_14" not in df.columns:
            df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14).bfill()
        if "MACD_hist" not in df.columns or "MACD" not in df.columns or "MACD_signal" not in df.columns:
            macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
            df["MACD"] = macd["MACD_12_26_9"].bfill()
            df["MACD_signal"] = macd["MACDs_12_26_9"].bfill()
            df["MACD_hist"] = macd["MACDh_12_26_9"].bfill()
        if "ADX_14" not in df.columns:
            adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
            df["ADX_14"] = adx["ADX_14"].bfill()
            df["DMP_14"] = adx["DMP_14"].bfill()
            df["DMN_14"] = adx["DMN_14"].bfill()
        if "supertrend_direction" not in df.columns:
            st = ta.supertrend(df["High"], df["Low"], df["Close"], length=10, multiplier=3)
            df["supertrend_direction"] = st["SUPERTd_10_3"].bfill()
        if "volume_ratio" not in df.columns:
            df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def build(self):
        steps = [
            ("swing highs/lows", self.add_swings),
            ("market structure (BOS/CHoCH, HH/HL/LH/LL)", self.add_market_structure),
            ("candlestick signals (engulfing, pin bar)", self.add_candlestick_signals),
            ("trend/momentum crossovers", self.add_crossover_signals),
            ("breakout signals (Donchian, S/R, squeeze)", self.add_breakout_signals),
            ("smart money (order blocks, FVG, sweeps)", self.add_smart_money_signals),
            ("pullback signals (EMA, golden zone, VWAP)", self.add_pullback_signals),
            ("fibonacci retracement/extension", self.add_fibonacci_signals),
            ("range contraction (inside bar, NR7)", self.add_range_signals),
            ("combination signals (confluence, BOS+retest)", self.add_combination_signals),
        ]
        for label, step in steps:
            print(f"[PriceActionEngine] Adding {label}...")
            step()

        sig_cols = [c for c in self.df.columns if c.startswith("sig_")]
        total_signals = int(self.df[sig_cols].abs().sum().sum())
        print(f"[PriceActionEngine] Done. {len(sig_cols)} signal columns, {total_signals} total signals fired.")
        return self.df

    # ------------------------------------------------------------------
    # 1. Swing highs / lows (fractal pivots)
    # ------------------------------------------------------------------
    def add_swings(self):
        """Fractal swing points, confirmed `right` bars after the pivot (zero lookahead).
        NOTE: a flat top/bottom can cause more than one bar to satisfy the pivot condition."""
        df = self.df
        left, right = self.swing_left, self.swing_right
        win = left + right + 1

        is_ph = df["High"] == df["High"].rolling(win, center=True, min_periods=win).max()
        is_pl = df["Low"] == df["Low"].rolling(win, center=True, min_periods=win).min()

        # pivot bar itself - plotting only, never used for signals
        df["swing_high_at_pivot"] = is_ph.astype(int)
        df["swing_low_at_pivot"] = is_pl.astype(int)

        # placed on the bar where the pivot becomes CONFIRMED
        confirmed_high = df["High"].where(is_ph).shift(right)
        confirmed_low = df["Low"].where(is_pl).shift(right)
        df["last_swing_high"] = confirmed_high.ffill()
        # last_swing_low is not exposed as a df column (rare + negative lift in combo search),
        # but is still needed internally - market structure, sweeps, pullbacks, fib, retest below.
        self._last_swing_low = confirmed_low.ffill()

        df["bars_since_swing_high"] = df.groupby(confirmed_high.notna().cumsum()).cumcount()
        df["bars_since_swing_low"] = df.groupby(confirmed_low.notna().cumsum()).cumcount()

        # used by add_market_structure below
        self._confirmed_high = confirmed_high
        self._confirmed_low = confirmed_low
        return self

    # ------------------------------------------------------------------
    # 2. Market structure: BOS / CHoCH + HH/HL/LH/LL
    # ------------------------------------------------------------------
    def add_market_structure(self):
        """BOS = break of the last swing level in the trend direction (continuation).
        CHoCH = break against the current trend (reversal, trend flips).
        swing_label combines high/low labels; if both confirm same bar, low label wins."""
        if "last_swing_high" not in self.df.columns:
            self.add_swings()
        df = self.df
        n = len(df)

        closes = df["Close"].to_numpy()
        lsh = df["last_swing_high"].to_numpy()
        lsl = self._last_swing_low.to_numpy()

        trend_arr = np.zeros(n)

        trend = 0
        consumed_high = np.nan
        consumed_low = np.nan

        for i in range(n):
            level_h, level_l = lsh[i], lsl[i]

            broke_up = (not np.isnan(level_h)) and level_h != consumed_high and closes[i] > level_h
            broke_dn = (not np.isnan(level_l)) and level_l != consumed_low and closes[i] < level_l

            if broke_up:
                trend = 1
                consumed_high = level_h
            if broke_dn:
                trend = -1
                consumed_low = level_l

            trend_arr[i] = trend

        df["structure_trend"] = trend_arr

        # separate arrays so a same-bar high+low don't clobber each other
        label_high = np.zeros(n)
        label_low = np.zeros(n)
        ch = self._confirmed_high.to_numpy()
        cl = self._confirmed_low.to_numpy()
        prev_high = np.nan
        prev_low = np.nan
        for i in range(n):
            if not np.isnan(ch[i]):
                if not np.isnan(prev_high):
                    label_high[i] = 2 if ch[i] > prev_high else -1  # HH or LH
                prev_high = ch[i]
            if not np.isnan(cl[i]):
                if not np.isnan(prev_low):
                    label_low[i] = 1 if cl[i] > prev_low else -2  # HL or LL
                prev_low = cl[i]
        df["swing_high_label"] = label_high
        df["swing_low_label"] = label_low
        df["swing_label"] = np.where(label_low != 0, label_low, label_high)
        return self

    # ------------------------------------------------------------------
    # 3. Candlestick reversal signals
    # ------------------------------------------------------------------
    def add_candlestick_signals(self):
        """Engulfing: body swallows the previous opposite-colour body.
        Pin bar: wick >= 2x body on one side + close near the other extreme (rejection)."""
        df = self.df
        body = (df["Close"] - df["Open"]).abs()
        prev_bear = df["Close"].shift(1) < df["Open"].shift(1)
        prev_bull = df["Close"].shift(1) > df["Open"].shift(1)

        bull_engulf = (
            (df["Close"] > df["Open"])
            & prev_bear
            & (df["Close"] >= df["Open"].shift(1))
            & (df["Open"] <= df["Close"].shift(1))
        )
        bear_engulf = (
            (df["Close"] < df["Open"])
            & prev_bull
            & (df["Close"] <= df["Open"].shift(1))
            & (df["Open"] >= df["Close"].shift(1))
        )
        df["sig_engulfing"] = np.where(bull_engulf, 1, np.where(bear_engulf, -1, 0))

        rng = df["High"] - df["Low"] + 1e-9
        upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)
        lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
        close_pos = (df["Close"] - df["Low"]) / rng  # 0 = at low, 1 = at high

        bull_pin = (lower_wick >= 2 * body) & (close_pos > 0.6)
        bear_pin = (upper_wick >= 2 * body) & (close_pos < 0.4)
        df["sig_pinbar"] = np.where(bull_pin, 1, np.where(bear_pin, -1, 0))
        return self

    # ------------------------------------------------------------------
    # 3b. Trend/momentum crossovers
    # ------------------------------------------------------------------
    def add_crossover_signals(self):
        """Fresh-cross events (fire only on the bar of the cross, +1/-1) for pairs
        not already captured by the level-based indicator>median conditions:
        EMA10/EMA20, EMA20/EMA50, MACD/MACD_signal, Close/VWAP, Close/EMA20,
        and RSI_14 crossing 50 (momentum flip)."""
        df = self.df

        def cross(fast, slow):
            up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
            dn = (fast < slow) & (fast.shift(1) >= slow.shift(1))
            return np.where(up, 1, np.where(dn, -1, 0))

        df["sig_ema_10_20_cross"] = cross(self._ema_10, self._ema_20)
        df["sig_ema_20_50_cross"] = cross(self._ema_20, self._ema_50)
        df["sig_macd_signal_cross"] = cross(df["MACD"], df["MACD_signal"])
        df["sig_price_vwap_cross"] = cross(df["Close"], self._vwap)
        df["sig_price_ema20_cross"] = cross(df["Close"], self._ema_20)

        rsi_up = (self._rsi_14 > 50) & (self._rsi_14.shift(1) <= 50)
        rsi_dn = (self._rsi_14 < 50) & (self._rsi_14.shift(1) >= 50)
        df["sig_rsi50_cross"] = np.where(rsi_up, 1, np.where(rsi_dn, -1, 0))
        return self

    # ------------------------------------------------------------------
    # 4. Breakout signals
    # ------------------------------------------------------------------
    def add_breakout_signals(self):
        """sig_donchian: 20-bar high/low break (turtle). squeeze_on: TTM squeeze
        (BB inside KC) state, used elsewhere as a volatility-regime filter."""
        if "last_swing_high" not in self.df.columns:
            self.add_swings()
        df = self.df

        # --- Donchian / turtle breakout (previous 20 bars, current excluded) ---
        dc_high = df["High"].rolling(20).max().shift(1)
        dc_low = df["Low"].rolling(20).min().shift(1)
        don_up = (df["Close"] > dc_high) & (df["Close"].shift(1) <= dc_high.shift(1))
        don_dn = (df["Close"] < dc_low) & (df["Close"].shift(1) >= dc_low.shift(1))
        df["sig_donchian"] = np.where(don_up, 1, np.where(don_dn, -1, 0))

        # --- Resistance level: strongest of the last 3 confirmed swing highs ---
        ch = self._confirmed_high.copy()
        resistance = ch.dropna().rolling(3).max().reindex(df.index).ffill()
        df["resistance_level"] = resistance

        # --- TTM squeeze state ---
        squeeze_on = (self._bb_upper < self._kc_upper) & (self._bb_lower > self._kc_lower)
        df["squeeze_on"] = squeeze_on.astype(int)
        return self

    # ------------------------------------------------------------------
    # 5. Smart money concepts: fair value gaps, liquidity sweeps
    # ------------------------------------------------------------------
    def add_smart_money_signals(self):
        """FVG: 3-candle gap (candle1.High < candle3.Low = bullish), multi-zone
        with age + invalidation handling. Liquidity sweep: wick takes the swing
        level but candle closes back inside (stop hunt)."""
        if "last_swing_high" not in self.df.columns:
            self.add_swings()
        df = self.df
        n = len(df)

        o = df["Open"].to_numpy()
        h = df["High"].to_numpy()
        low = df["Low"].to_numpy()
        c = df["Close"].to_numpy()
        atr = df["ATR_14"].to_numpy()
        max_age = self.max_zone_age
        max_zones = self.max_active_zones

        # --- Fair value gaps (multi-zone, age + invalidation aware) ---
        fvg_sig = np.zeros(n)
        bull_gaps = []
        bear_gaps = []

        for i in range(2, n):
            if low[i] > h[i - 2]:
                bull_gaps.append({"lo": h[i - 2], "hi": low[i], "birth": i})
                if len(bull_gaps) > max_zones:
                    bull_gaps.pop(0)
            if h[i] < low[i - 2]:
                bear_gaps.append({"lo": h[i], "hi": low[i - 2], "birth": i})
                if len(bear_gaps) > max_zones:
                    bear_gaps.pop(0)

            fired = False
            for gap in list(bull_gaps):
                if i - gap["birth"] > max_age or c[i] < gap["lo"] - atr[i]:
                    bull_gaps.remove(gap)
                    continue
                if not fired and low[i] <= gap["hi"] and c[i] > gap["hi"] and c[i] > o[i]:
                    fvg_sig[i] = 1
                    bull_gaps.remove(gap)
                    fired = True
            for gap in list(bear_gaps):
                if i - gap["birth"] > max_age or c[i] > gap["hi"] + atr[i]:
                    bear_gaps.remove(gap)
                    continue
                if not fired and h[i] >= gap["lo"] and c[i] < gap["lo"] and c[i] < o[i]:
                    fvg_sig[i] = -1
                    bear_gaps.remove(gap)
                    fired = True

        df["sig_fvg_fill"] = fvg_sig

        # --- Liquidity sweep / stop hunt (vectorised) ---
        lsl = self._last_swing_low
        lsh = df["last_swing_high"]
        bull_sweep = (df["Low"] < lsl) & (df["Close"] > lsl) & (df["Close"] > df["Open"])
        bear_sweep = (df["High"] > lsh) & (df["Close"] < lsh) & (df["Close"] < df["Open"])
        df["sig_sweep"] = np.where(bull_sweep, 1, np.where(bear_sweep, -1, 0))
        return self

    # ------------------------------------------------------------------
    # 6. Pullback / retracement entries
    # ------------------------------------------------------------------
    def add_pullback_signals(self):
        """sig_ema_pullback: dip touches EMA20 in an EMA20>EMA50 uptrend, closes back
        above with RSI still healthy. sig_golden_pullback: retrace into the fib
        50-61.8% "golden zone" of the last impulse."""
        if "structure_trend" not in self.df.columns:
            self.add_market_structure()
        df = self.df

        # --- EMA pullback ---
        uptrend = self._ema_20 > self._ema_50
        dntrend = self._ema_20 < self._ema_50
        touch_up = (df["Low"] <= self._ema_20) & (df["Close"] > self._ema_20) & (df["Close"] > df["Open"])
        touch_dn = (df["High"] >= self._ema_20) & (df["Close"] < self._ema_20) & (df["Close"] < df["Open"])
        df["sig_ema_pullback"] = np.where(
            uptrend & touch_up & (self._rsi_14 > 45), 1,
            np.where(dntrend & touch_dn & (self._rsi_14 < 55), -1, 0),
        )

        # --- Golden zone (fib 0.5 - 0.618 of the last confirmed impulse) ---
        lsh = df["last_swing_high"]
        lsl = self._last_swing_low
        impulse = lsh - lsl
        gz_top_up = lsh - 0.5 * impulse
        gz_bot_up = lsh - 0.618 * impulse
        gz_bot_dn = lsl + 0.5 * impulse
        gz_top_dn = lsl + 0.618 * impulse

        in_gz_up = (df["Low"] <= gz_top_up) & (df["Low"] >= gz_bot_up)
        in_gz_dn = (df["High"] >= gz_bot_dn) & (df["High"] <= gz_top_dn)
        bull_candle = df["Close"] > df["Open"]
        bear_candle = df["Close"] < df["Open"]

        df["sig_golden_pullback"] = np.where(
            (df["structure_trend"] == 1) & in_gz_up & bull_candle, 1,
            np.where((df["structure_trend"] == -1) & in_gz_dn & bear_candle, -1, 0),
        )
        return self

    # ------------------------------------------------------------------
    # 7. Fibonacci retracement
    # ------------------------------------------------------------------
    def add_fibonacci_signals(self):
        """sig_fib_retracement: pullback into the 38.2-78.6% zone of the last impulse,
        with structure_trend, continuation entry (overlaps sig_golden_pullback's
        narrower 50-61.8% band)."""
        if "structure_trend" not in self.df.columns:
            self.add_market_structure()
        df = self.df

        lsh = df["last_swing_high"]
        lsl = self._last_swing_low
        impulse = (lsh - lsl).abs()
        bull_candle = df["Close"] > df["Open"]
        bear_candle = df["Close"] < df["Open"]

        # --- Retracement zone (38.2% - 78.6%) ---
        up_top = lsh - 0.382 * impulse
        up_bot = lsh - 0.786 * impulse
        dn_bot = lsl + 0.382 * impulse
        dn_top = lsl + 0.786 * impulse

        in_zone_up = (df["Low"] <= up_top) & (df["Low"] >= up_bot)
        in_zone_dn = (df["High"] >= dn_bot) & (df["High"] <= dn_top)

        df["sig_fib_retracement"] = np.where(
            (df["structure_trend"] == 1) & in_zone_up & bull_candle, 1,
            np.where((df["structure_trend"] == -1) & in_zone_dn & bear_candle, -1, 0),
        )
        return self

    # ------------------------------------------------------------------
    # 8. Volatility-contraction range patterns
    # ------------------------------------------------------------------
    def add_range_signals(self):
        """sig_nr7_breakout: break of the narrowest-range-of-7 (NR7) candle's high/low."""
        df = self.df
        rng = df["High"] - df["Low"]

        is_nr7 = rng == rng.rolling(7).min()
        nr7_break_up = is_nr7.shift(1, fill_value=False) & (df["Close"] > df["High"].shift(1))
        nr7_break_dn = is_nr7.shift(1, fill_value=False) & (df["Close"] < df["Low"].shift(1))
        df["sig_nr7_breakout"] = np.where(nr7_break_up, 1, np.where(nr7_break_dn, -1, 0))
        return self

    # ------------------------------------------------------------------
    # 9. Advanced combination signals (confluence maths)
    # ------------------------------------------------------------------
    def add_combination_signals(self):
        """trend_score sums EMA stack + supertrend + MACD sign + ADX-filtered DI bias
        + VWAP side (range -5..+5) - a standalone trend-alignment indicator."""
        needed = ["structure_trend", "sig_sweep", "sig_engulfing"]
        if any(col not in self.df.columns for col in needed):
            self.add_market_structure()
            self.add_candlestick_signals()
            self.add_smart_money_signals()
        df = self.df

        votes = (
            np.sign(self._ema_20 - self._ema_50)
            + df["supertrend_direction"]
            + np.sign(df["MACD_hist"])
            + np.sign(df["DMP_14"] - df["DMN_14"]) * (df["ADX_14"] > 20)
            + np.sign(df["Close"] - self._vwap)
        )
        df["trend_score"] = votes
        return self
