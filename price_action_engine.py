"""Price-action engine: swing points, market structure (BOS/CHoCH), breakouts,
smart-money concepts (order blocks, FVG, sweeps), fibonacci, trendlines, and
range/divergence/combination signals. Every strategy writes one `sig_<name>`
column (+1 buy, -1 sell, 0 none), firing only once a setup is CONFIRMED -
swing points are shifted `right` bars after the pivot, so there's no lookahead.

Usage: `df = PriceActionEngine(df).build()`, or chain individual `add_*` methods.
"""

import numpy as np
import pandas as pd
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
        df = self.df

        if "EMA_10" not in df.columns:
            df["EMA_10"] = ta.ema(df["Close"], length=10).bfill()
        if "EMA_20" not in df.columns:
            df["EMA_20"] = ta.ema(df["Close"], length=20).bfill()
        if "EMA_50" not in df.columns:
            df["EMA_50"] = ta.ema(df["Close"], length=50).bfill()
        if "RSI_14" not in df.columns:
            df["RSI_14"] = ta.rsi(df["Close"], length=14).bfill()
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
        if "VWAP" not in df.columns:
            df["VWAP"] = ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"]).bfill()
        if "BB_upper" not in df.columns:
            bb = ta.bbands(df["Close"], length=20, std=2)
            df["BB_lower"] = bb.iloc[:, 0].bfill()
            df["BB_middle"] = bb.iloc[:, 1].bfill()
            df["BB_upper"] = bb.iloc[:, 2].bfill()
        if "KC_upper" not in df.columns:
            kc = ta.kc(df["High"], df["Low"], df["Close"], length=20, scalar=2)
            df["KC_lower"] = kc.iloc[:, 0].bfill()
            df["KC_upper"] = kc.iloc[:, 2].bfill()
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
            # sig_trendline_break dropped: rare + negative lift in combo search (also the
            # slowest step in the pipeline - skipping it saves time too)
            # ("dynamic trendline breaks", self.add_trendline_signals),
            ("range contraction (inside bar, NR7)", self.add_range_signals),
            # sig_rsi_divergence dropped: rare + negative lift in combo search
            # ("divergence signals (RSI vs price)", self.add_divergence_signals),
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
        # last_swing_low dropped: rare + negative lift in combo search. Kept internally (not a
        # df column) - still used by market structure, sweeps, pullbacks, fib, retest below.
        self._last_swing_low = confirmed_low.ffill()

        df["bars_since_swing_high"] = df.groupby(confirmed_high.notna().cumsum()).cumcount()
        df["bars_since_swing_low"] = df.groupby(confirmed_low.notna().cumsum()).cumcount()

        # sig_swing_break dropped: rare + negative lift in combo search
        # cross_up = (df["Close"] > df["last_swing_high"]) & (df["Close"].shift(1) <= df["last_swing_high"].shift(1))
        # cross_dn = (df["Close"] < self._last_swing_low) & (df["Close"].shift(1) >= self._last_swing_low.shift(1))
        # df["sig_swing_break"] = np.where(cross_up, 1, np.where(cross_dn, -1, 0))

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

        bos = np.zeros(n)
        choch = np.zeros(n)
        trend_arr = np.zeros(n)

        trend = 0
        consumed_high = np.nan
        consumed_low = np.nan

        for i in range(n):
            level_h, level_l = lsh[i], lsl[i]

            broke_up = (not np.isnan(level_h)) and level_h != consumed_high and closes[i] > level_h
            broke_dn = (not np.isnan(level_l)) and level_l != consumed_low and closes[i] < level_l

            if broke_up:
                if trend == -1:
                    choch[i] = 1  # bearish structure broken upward = CHoCH
                else:
                    bos[i] = 1
                trend = 1
                consumed_high = level_h
            if broke_dn:
                if trend == 1:
                    choch[i] = -1
                else:
                    bos[i] = -1
                trend = -1
                consumed_low = level_l

            trend_arr[i] = trend

        df["structure_trend"] = trend_arr
        # sig_bos / sig_choch dropped: rare + negative lift in combo search (also removes their
        # only consumers, sig_bos_retest and sig_super_confluence, in add_combination_signals)
        # df["sig_bos"] = bos
        # df["sig_choch"] = choch

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

        df["sig_ema_10_20_cross"] = cross(df["EMA_10"], df["EMA_20"])
        df["sig_ema_20_50_cross"] = cross(df["EMA_20"], df["EMA_50"])
        df["sig_macd_signal_cross"] = cross(df["MACD"], df["MACD_signal"])
        df["sig_price_vwap_cross"] = cross(df["Close"], df["VWAP"])
        df["sig_price_ema20_cross"] = cross(df["Close"], df["EMA_20"])

        rsi_up = (df["RSI_14"] > 50) & (df["RSI_14"].shift(1) <= 50)
        rsi_dn = (df["RSI_14"] < 50) & (df["RSI_14"].shift(1) >= 50)
        df["sig_rsi50_cross"] = np.where(rsi_up, 1, np.where(rsi_dn, -1, 0))
        return self

    # ------------------------------------------------------------------
    # 4. Breakout signals
    # ------------------------------------------------------------------
    def add_breakout_signals(self):
        """sig_donchian: 20-bar high/low break (turtle). sig_sr_breakout: break of the
        strongest of the last 3 swing pivots. sig_squeeze_breakout: TTM squeeze
        (BB inside KC) release, direction confirmed by volume."""
        if "last_swing_high" not in self.df.columns:
            self.add_swings()
        df = self.df

        # --- Donchian / turtle breakout (previous 20 bars, current excluded) ---
        dc_high = df["High"].rolling(20).max().shift(1)
        dc_low = df["Low"].rolling(20).min().shift(1)
        don_up = (df["Close"] > dc_high) & (df["Close"].shift(1) <= dc_high.shift(1))
        don_dn = (df["Close"] < dc_low) & (df["Close"].shift(1) >= dc_low.shift(1))
        df["sig_donchian"] = np.where(don_up, 1, np.where(don_dn, -1, 0))

        # --- S/R breakout: strongest of the last 3 confirmed pivots ---
        ch = self._confirmed_high.copy()
        resistance = ch.dropna().rolling(3).max().reindex(df.index).ffill()
        df["resistance_level"] = resistance
        # support_level / sig_sr_breakout dropped: rare + negative lift in combo search
        # cl = self._confirmed_low.copy()
        # support = cl.dropna().rolling(3).min().reindex(df.index).ffill()
        # df["support_level"] = support
        # res_break = (df["Close"] > resistance) & (df["Close"].shift(1) <= resistance.shift(1))
        # sup_break = (df["Close"] < support) & (df["Close"].shift(1) >= support.shift(1))
        # df["sig_sr_breakout"] = np.where(res_break, 1, np.where(sup_break, -1, 0))

        # --- TTM squeeze breakout ---
        squeeze_on = (df["BB_upper"] < df["KC_upper"]) & (df["BB_lower"] > df["KC_lower"])
        df["squeeze_on"] = squeeze_on.astype(int)
        # sig_squeeze_breakout dropped: rare + negative lift in combo search
        # squeeze_released = squeeze_on.shift(1, fill_value=False) & ~squeeze_on
        # momentum_up = df["Close"] > df["BB_middle"]
        # volume_ok = df["volume_ratio"] > 1.2
        # df["sig_squeeze_breakout"] = np.where(
        #     squeeze_released & momentum_up & volume_ok, 1,
        #     np.where(squeeze_released & ~momentum_up & volume_ok, -1, 0),
        # )
        return self

    # ------------------------------------------------------------------
    # 5. Smart money concepts: order blocks, FVG, liquidity sweeps
    # ------------------------------------------------------------------
    def add_smart_money_signals(self):
        """Order block: last opposite candle before an impulse (>1.5 ATR/3 candles);
        tracks up to max_active_zones per side, expires after max_zone_age bars or
        a 1-ATR close-through invalidation. FVG: 3-candle gap (candle1.High <
        candle3.Low = bullish), same multi-zone/expiry handling. Liquidity sweep:
        wick takes the swing level but candle closes back inside (stop hunt)."""
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

        # sig_ob_retest dropped: rare + negative lift in combo search
        # # --- Order blocks (multi-zone, age + invalidation aware) ---
        # ob_sig = np.zeros(n)
        # bull_zones = []  # each: {"lo":.., "hi":.., "birth":i}
        # bear_zones = []
        #
        # for i in range(3, n):
        #     if c[i] - c[i - 3] > 1.5 * atr[i]:
        #         for j in range(i - 1, max(i - 4, 0) - 1, -1):
        #             if c[j] < o[j]:
        #                 if not bull_zones or (low[j], h[j]) != (bull_zones[-1]["lo"], bull_zones[-1]["hi"]):
        #                     bull_zones.append({"lo": low[j], "hi": h[j], "birth": i})
        #                     if len(bull_zones) > max_zones:
        #                         bull_zones.pop(0)
        #                 break
        #     if c[i - 3] - c[i] > 1.5 * atr[i]:
        #         for j in range(i - 1, max(i - 4, 0) - 1, -1):
        #             if c[j] > o[j]:
        #                 if not bear_zones or (low[j], h[j]) != (bear_zones[-1]["lo"], bear_zones[-1]["hi"]):
        #                     bear_zones.append({"lo": low[j], "hi": h[j], "birth": i})
        #                     if len(bear_zones) > max_zones:
        #                         bear_zones.pop(0)
        #                 break
        #
        #     fired = False
        #     for zone in list(bull_zones):
        #         if i - zone["birth"] > max_age or c[i] < zone["lo"] - atr[i]:
        #             bull_zones.remove(zone)
        #             continue
        #         if not fired and low[i] <= zone["hi"] and c[i] > zone["hi"] and c[i] > o[i]:
        #             ob_sig[i] = 1
        #             bull_zones.remove(zone)
        #             fired = True
        #     for zone in list(bear_zones):
        #         if i - zone["birth"] > max_age or c[i] > zone["hi"] + atr[i]:
        #             bear_zones.remove(zone)
        #             continue
        #         if not fired and h[i] >= zone["lo"] and c[i] < zone["lo"] and c[i] < o[i]:
        #             ob_sig[i] = -1
        #             bear_zones.remove(zone)
        #             fired = True
        #
        # df["sig_ob_retest"] = ob_sig

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

        # --- Liquidity sweep / stop hunt (vectorised, unchanged) ---
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
        50-61.8% "golden zone" of the last impulse. sig_vwap_reclaim: VWAP cross
        back with a volume surge, structure-aligned."""
        if "structure_trend" not in self.df.columns:
            self.add_market_structure()
        df = self.df

        # --- EMA pullback ---
        uptrend = df["EMA_20"] > df["EMA_50"]
        dntrend = df["EMA_20"] < df["EMA_50"]
        touch_up = (df["Low"] <= df["EMA_20"]) & (df["Close"] > df["EMA_20"]) & (df["Close"] > df["Open"])
        touch_dn = (df["High"] >= df["EMA_20"]) & (df["Close"] < df["EMA_20"]) & (df["Close"] < df["Open"])
        df["sig_ema_pullback"] = np.where(
            uptrend & touch_up & (df["RSI_14"] > 45), 1,
            np.where(dntrend & touch_dn & (df["RSI_14"] < 55), -1, 0),
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

        # sig_vwap_reclaim dropped: rare + negative lift in combo search
        # cross_up = (df["Close"] > df["VWAP"]) & (df["Close"].shift(1) <= df["VWAP"].shift(1))
        # cross_dn = (df["Close"] < df["VWAP"]) & (df["Close"].shift(1) >= df["VWAP"].shift(1))
        # vol_surge = df["volume_ratio"] > 1.5
        # df["sig_vwap_reclaim"] = np.where(
        #     cross_up & vol_surge & (df["structure_trend"] >= 0), 1,
        #     np.where(cross_dn & vol_surge & (df["structure_trend"] <= 0), -1, 0),
        # )
        return self

    # ------------------------------------------------------------------
    # 7. Fibonacci retracement + extension (NEW)
    # ------------------------------------------------------------------
    def add_fibonacci_signals(self):
        """sig_fib_retracement: pullback into the 38.2-78.6% zone of the last impulse,
        with structure_trend, continuation entry (overlaps sig_golden_pullback's
        narrower 50-61.8% band). sig_fib_extension: push beyond the 127.2-161.8%
        extension, contrarian exhaustion/take-profit signal."""
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

        # sig_fib_extension dropped: rare + negative lift in combo search
        # ext_up_lo = lsh + 0.272 * impulse
        # ext_up_hi = lsh + 0.618 * impulse
        # ext_dn_hi = lsl - 0.272 * impulse
        # ext_dn_lo = lsl - 0.618 * impulse
        #
        # hit_ext_up = (df["High"] >= ext_up_lo) & (df["High"] <= ext_up_hi)
        # hit_ext_dn = (df["Low"] <= ext_dn_hi) & (df["Low"] >= ext_dn_lo)
        #
        # df["sig_fib_extension"] = np.where(
        #     (df["structure_trend"] == 1) & hit_ext_up & bear_candle, -1,
        #     np.where((df["structure_trend"] == -1) & hit_ext_dn & bull_candle, 1, 0),
        # )
        return self

    # ------------------------------------------------------------------
    # 8. Dynamic trendline breaks (NEW)
    # ------------------------------------------------------------------
    def add_trendline_signals(self, lookback_pivots=3):
        """Fits a least-squares trendline through the last `lookback_pivots` swing
        highs/lows, fires when close crosses the projected line.
        Note: refits every bar - the slowest step of the pipeline on 100k+ bar histories."""
        if "last_swing_high" not in self.df.columns:
            self.add_swings()
        df = self.df
        n = len(df)

        ch = self._confirmed_high.to_numpy()
        cl = self._confirmed_low.to_numpy()
        c = df["Close"].to_numpy()

        high_pivots = []  # (bar_index, price) of last confirmed swing highs
        low_pivots = []
        sig = np.zeros(n)

        for i in range(n):
            if i >= 1 and len(high_pivots) >= 2:
                xs = np.array([p[0] for p in high_pivots[-lookback_pivots:]], dtype=float)
                ys = np.array([p[1] for p in high_pivots[-lookback_pivots:]], dtype=float)
                slope, intercept = np.polyfit(xs, ys, 1)
                proj_now = slope * i + intercept
                proj_prev = slope * (i - 1) + intercept
                if slope < 0 and c[i] > proj_now and c[i - 1] <= proj_prev:
                    sig[i] = 1

            if i >= 1 and len(low_pivots) >= 2:
                xs = np.array([p[0] for p in low_pivots[-lookback_pivots:]], dtype=float)
                ys = np.array([p[1] for p in low_pivots[-lookback_pivots:]], dtype=float)
                slope, intercept = np.polyfit(xs, ys, 1)
                proj_now = slope * i + intercept
                proj_prev = slope * (i - 1) + intercept
                if slope > 0 and c[i] < proj_now and c[i - 1] >= proj_prev and sig[i] == 0:
                    sig[i] = -1

            if not np.isnan(ch[i]):
                high_pivots.append((i, ch[i]))
            if not np.isnan(cl[i]):
                low_pivots.append((i, cl[i]))

        df["sig_trendline_break"] = sig
        return self

    # ------------------------------------------------------------------
    # 9. Volatility-contraction range patterns (NEW)
    # ------------------------------------------------------------------
    def add_range_signals(self):
        """sig_inside_bar_breakout: break of an inside bar's high/low.
        sig_nr7_breakout: break of the narrowest-range-of-7 (NR7) candle's high/low."""
        df = self.df
        rng = df["High"] - df["Low"]

        # sig_inside_bar_breakout dropped: rare + negative lift in combo search
        # inside = (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))
        # break_up = inside.shift(1, fill_value=False) & (df["Close"] > df["High"].shift(1))
        # break_dn = inside.shift(1, fill_value=False) & (df["Close"] < df["Low"].shift(1))
        # df["sig_inside_bar_breakout"] = np.where(break_up, 1, np.where(break_dn, -1, 0))

        is_nr7 = rng == rng.rolling(7).min()
        nr7_break_up = is_nr7.shift(1, fill_value=False) & (df["Close"] > df["High"].shift(1))
        nr7_break_dn = is_nr7.shift(1, fill_value=False) & (df["Close"] < df["Low"].shift(1))
        df["sig_nr7_breakout"] = np.where(nr7_break_up, 1, np.where(nr7_break_dn, -1, 0))
        return self

    # ------------------------------------------------------------------
    # 10. RSI divergence at swing points
    # ------------------------------------------------------------------
    def add_divergence_signals(self):
        """Divergence at confirmed swing points: price makes a lower low (or higher
        high) while RSI makes the opposite - momentum no longer confirms price."""
        if "last_swing_high" not in self.df.columns:
            self.add_swings()
        df = self.df
        n = len(df)
        right = self.swing_right

        rsi = df["RSI_14"].to_numpy()
        ch = self._confirmed_high.to_numpy()
        cl = self._confirmed_low.to_numpy()

        sig = np.zeros(n)
        prev_low_price, prev_low_rsi = np.nan, np.nan
        prev_high_price, prev_high_rsi = np.nan, np.nan

        for i in range(n):
            if not np.isnan(cl[i]):
                pivot_rsi = rsi[i - right] if i - right >= 0 else np.nan
                if (
                    not np.isnan(prev_low_price)
                    and cl[i] < prev_low_price
                    and not np.isnan(pivot_rsi)
                    and pivot_rsi > prev_low_rsi
                ):
                    sig[i] = 1
                prev_low_price, prev_low_rsi = cl[i], pivot_rsi
            if not np.isnan(ch[i]):
                pivot_rsi = rsi[i - right] if i - right >= 0 else np.nan
                if (
                    not np.isnan(prev_high_price)
                    and ch[i] > prev_high_price
                    and not np.isnan(pivot_rsi)
                    and pivot_rsi < prev_high_rsi
                ):
                    sig[i] = -1
                prev_high_price, prev_high_rsi = ch[i], pivot_rsi

        df["sig_rsi_divergence"] = sig
        return self

    # ------------------------------------------------------------------
    # 11. Advanced combination signals (confluence maths)
    # ------------------------------------------------------------------
    def add_combination_signals(self):
        """trend_score sums EMA stack + supertrend + MACD sign + ADX-filtered DI bias
        + VWAP side (range -5..+5).
        sig_trend_confluence, sig_bos_retest, sig_sweep_reversal, sig_super_confluence all
        dropped: rare + negative lift in combo search (sig_bos_retest/sig_super_confluence
        also depended on the now-dropped sig_bos/sig_choch)."""
        needed = ["structure_trend", "sig_sweep", "sig_engulfing"]
        if any(col not in self.df.columns for col in needed):
            self.add_market_structure()
            self.add_candlestick_signals()
            self.add_smart_money_signals()
        df = self.df

        # --- Confluence score ---
        votes = (
            np.sign(df["EMA_20"] - df["EMA_50"])
            + df["supertrend_direction"]
            + np.sign(df["MACD_hist"])
            + np.sign(df["DMP_14"] - df["DMN_14"]) * (df["ADX_14"] > 20)
            + np.sign(df["Close"] - df["VWAP"])
        )
        df["trend_score"] = votes
        # fresh_up = (votes >= 4) & (votes.shift(1) < 4)
        # fresh_dn = (votes <= -4) & (votes.shift(1) > -4)
        # df["sig_trend_confluence"] = np.where(fresh_up, 1, np.where(fresh_dn, -1, 0))

        # --- BOS + retest ---
        # n = len(df)
        # c = df["Close"].to_numpy()
        # low = df["Low"].to_numpy()
        # h = df["High"].to_numpy()
        # o = df["Open"].to_numpy()
        # atr = df["ATR_14"].to_numpy()
        # bos = df["sig_bos"].to_numpy() + df["sig_choch"].to_numpy()
        # lsh = df["last_swing_high"].to_numpy()
        # lsl = self._last_swing_low.to_numpy()
        #
        # retest_sig = np.zeros(n)
        # pending_up_level = np.nan  # broken resistance waiting for a retest
        # pending_dn_level = np.nan
        # max_wait = 30  # candles a retest stays valid
        # up_age = dn_age = 0
        #
        # for i in range(n):
        #     if bos[i] > 0:
        #         pending_up_level, up_age = lsh[i], 0
        #         pending_dn_level = np.nan
        #     elif bos[i] < 0:
        #         pending_dn_level, dn_age = lsl[i], 0
        #         pending_up_level = np.nan
        #
        #     if not np.isnan(pending_up_level):
        #         up_age += 1
        #         touched = low[i] <= pending_up_level + 0.25 * atr[i]
        #         if touched and c[i] > pending_up_level and c[i] > o[i]:
        #             retest_sig[i] = 1
        #             pending_up_level = np.nan
        #         elif c[i] < pending_up_level - 0.5 * atr[i] or up_age > max_wait:
        #             pending_up_level = np.nan  # retest failed / expired
        #
        #     if not np.isnan(pending_dn_level):
        #         dn_age += 1
        #         touched = h[i] >= pending_dn_level - 0.25 * atr[i]
        #         if touched and c[i] < pending_dn_level and c[i] < o[i]:
        #             retest_sig[i] = -1
        #             pending_dn_level = np.nan
        #         elif c[i] > pending_dn_level + 0.5 * atr[i] or dn_age > max_wait:
        #             pending_dn_level = np.nan
        #
        # df["sig_bos_retest"] = retest_sig

        # --- Sweep + reversal confirmation ---
        # sweep = df["sig_sweep"]
        # confirm = df["sig_engulfing"] + df["sig_pinbar"]
        # bull = ((sweep == 1) | (sweep.shift(1) == 1)) & (confirm > 0)
        # bear = ((sweep == -1) | (sweep.shift(1) == -1)) & (confirm < 0)
        # df["sig_sweep_reversal"] = np.where(bull, 1, np.where(bear, -1, 0))

        # --- Super confluence: fresh structure break + everything agrees ---
        # break_evt = df["sig_bos"] + df["sig_choch"]
        # vol_ok = df["volume_ratio"] > 1.2
        # df["sig_super_confluence"] = np.where(
        #     (break_evt == 1) & (df["trend_score"] >= 3) & vol_ok, 1,
        #     np.where((break_evt == -1) & (df["trend_score"] <= -3) & vol_ok, -1, 0),
        # )
        return self
