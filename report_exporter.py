"""ReportExporter: runs the combo backtest (if not already computed) and writes
a single JSON file that `dashboard.html` reads and renders.

Usage:
    ReportExporter(df, config).save("report.json")

`config` is a plain dict of every setting main.py already has as module-level
constants (symbol, interval, backtest/combo parameters, which sections to run,
...) - this module stays independent of main.py so there's no circular import.

Optionally pass `precomputed` (a dict) with analysis results main.py already
computed for its own console report - this avoids re-running the same
(expensive) ComboBacktester search a second time just to export it. Any key
left out is computed fresh here instead. Recognized keys: combo_backtester
(instance), combo_profitable (DataFrame).
"""

import json
import math

import pandas as pd

from combo_backtester import ComboBacktester

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _json_safe(value):
    """Make a single value safe for json.dump: NaN/Inf/NaT/NA -> None, numpy/pandas
    scalars -> native Python types, Timestamps -> ISO strings."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar (int64, float64, bool_, ...)
        value = value.item()
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def records(df):
    """DataFrame -> list of JSON-safe dicts. None/empty-safe."""
    if df is None or len(df) == 0:
        return []
    return [{k: _json_safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


class ReportExporter:
    """Collects the combo backtest into one JSON-serializable dict and writes it out.

    Usage:
        ReportExporter(df, config).save("report.json")
    """

    def __init__(self, df, config, precomputed=None):
        self.df = df
        self.config = config
        self.precomputed = precomputed or {}

    def _column_groups(self):
        df = self.df
        ohlcv = [c for c in OHLCV_COLUMNS if c in df.columns]
        price_action = [c for c in df.columns if c.startswith("sig_")]
        indicator = [c for c in df.columns if c not in ohlcv and c not in price_action]
        return ohlcv, indicator, price_action

    def _dataset_section(self):
        df = self.df
        ohlcv, indicator, price_action = self._column_groups()
        return {
            "symbol": self.config.get("symbol"),
            "interval": self.config.get("interval"),
            "rows": len(df),
            "columns": len(df.columns),
            "date_start": df.index.min().isoformat(),
            "date_end": df.index.max().isoformat(),
            "missing_values": int(df.isna().sum().sum()),
            "column_groups": {"ohlcv": ohlcv, "indicator": indicator, "price_action": price_action},
        }

    def _combo_backtest_section(self):
        if not self.config.get("run_combo_backtest"):
            return None
        combo_bt = self.precomputed.get("combo_backtester")
        profitable = self.precomputed.get("combo_profitable")
        if combo_bt is None or profitable is None:
            combo_bt = ComboBacktester(
                self.df,
                initial_capital=self.config["backtest_initial_capital"],
                risk_per_trade_pct=self.config["backtest_risk_per_trade_pct"],
                stop_loss_pct=self.config["backtest_stop_loss_pct"],
                take_profit_pct=self.config["backtest_take_profit_pct"],
                max_hold_bars=self.config["backtest_max_hold_bars"],
                fee_pct=self.config["backtest_fee_pct"],
                min_combo_size=self.config.get("combo_min_size", 1),
                max_combo_size=self.config.get("combo_max_size", 8),
                min_fires=self.config.get("combo_min_fires", 15),
            )
            result = combo_bt.run()
            profitable = result[result["total_pnl"] > 0] if not result.empty else result

        # Saving every profitable combo (can be a lot at larger combo sizes) makes
        # report.json unusably large for a browser to fetch/parse - the JSON/dashboard
        # only ever need to answer "what are the best combos", so cap it here. The
        # console's own top-N slice is separate and unaffected.
        json_top_n = self.config.get("combo_json_top_n", 2000)
        saved = profitable.head(json_top_n)

        return {
            "config": {
                "min_combo_size": self.config.get("combo_min_size", 1),
                "max_combo_size": self.config.get("combo_max_size", 8),
                "min_fires": self.config.get("combo_min_fires", 15),
                "profitable_combos_found": len(profitable),
                "combos_saved_to_json": len(saved),
                **combo_bt.stats,
            },
            "combinations": records(saved),
        }

    def build(self):
        return {
            "generated_at": self.config.get("generated_at"),
            "config": {k: v for k, v in self.config.items() if k != "generated_at"},
            "dataset": self._dataset_section(),
            "combo_backtest": self._combo_backtest_section(),
        }

    def save(self, path="report.json"):
        report = self.build()
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=_json_safe)
        print(f"[ReportExporter] Saved -> {path}")
        return path
