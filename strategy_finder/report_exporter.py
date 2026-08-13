"""ReportExporter: runs the combo backtest (if not already computed) and writes
a single JSON file that the ui/ Streamlit app reads and renders.

Usage:
    ReportExporter(df, config).save("data/report.json")

`config` is a plain dict of main.py's settings, passed in to avoid a circular
import. Optionally pass `precomputed` (combo_backtester instance,
combo_profitable DataFrame) to skip re-running the search just to export it.
"""

import json
import math

import pandas as pd

from .combo_backtester import ComboBacktester
from .indicator_diagnostics import (
    compute_contribution_stats,
    compute_direction_target_diagnostic,
    compute_redundancy_pairs,
    find_perfect_predictors,
)

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


def compact_combo_records(df):
    """Every combo, compacted: condition/signal names dedup into one
    `condition_dictionary` array, each combo stores integer indices into it
    instead of a repeated " AND "-joined string - the display string is
    reconstructed on read (ui/app.py, backtesting/strategy_runner.py)."""
    if df is None or len(df) == 0:
        return [], []

    dictionary = sorted({name for conditions in df["conditions"] for name in conditions})
    index_of = {name: i for i, name in enumerate(dictionary)}

    rows = []
    for row in df.to_dict(orient="records"):
        rows.append({
            "direction": row["direction"],
            "size": _json_safe(row["size"]),
            "fires": _json_safe(row["fires"]),
            "trades": _json_safe(row["trades"]),
            "win_rate_pct": _json_safe(row["win_rate_pct"]),
            "final_equity": _json_safe(row["final_equity"]),
            "total_pnl": _json_safe(row["total_pnl"]),
            "return_pct": _json_safe(row["return_pct"]),
            "conditions": [index_of[name] for name in row["conditions"]],
        })
    return dictionary, rows


class ReportExporter:
    """Collects the combo backtest into one JSON-serializable dict and writes it out.

    Usage:
        ReportExporter(df, config).save("data/report.json")
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

        condition_dictionary, combinations = compact_combo_records(profitable)

        _, indicator_cols, _ = self._column_groups()
        contribution = compute_contribution_stats(condition_dictionary, combinations)
        redundancy = compute_redundancy_pairs(self.df, indicator_cols, threshold=0.9)
        perfect_predictors = find_perfect_predictors(combinations, condition_dictionary)
        direction_target_diagnostic = compute_direction_target_diagnostic(
            self.df,
            condition_window=self.config.get("combo_condition_window", 100),
            lag_depths=tuple(self.config.get("combo_lag_depths", (1, 2, 3))),
            min_fires=self.config.get("combo_min_fires", 15),
            take_profit_pct=self.config["backtest_take_profit_pct"],
            stop_loss_pct=self.config["backtest_stop_loss_pct"],
            max_hold_bars=self.config["backtest_max_hold_bars"],
        )

        return {
            "config": {
                "min_combo_size": self.config.get("combo_min_size", 1),
                "max_combo_size": self.config.get("combo_max_size", 8),
                "min_fires": self.config.get("combo_min_fires", 15),
                "profitable_combos_found": len(profitable),
                "combos_saved_to_json": len(combinations),
                **combo_bt.stats,
            },
            "condition_dictionary": condition_dictionary,
            "combinations": combinations,
            "indicator_contribution": contribution,
            "indicator_redundancy": redundancy,
            "perfect_predictors": perfect_predictors,
            "direction_target_diagnostic": direction_target_diagnostic,
        }

    def build(self):
        return {
            "generated_at": self.config.get("generated_at"),
            "config": {k: v for k, v in self.config.items() if k != "generated_at"},
            "dataset": self._dataset_section(),
            "combo_backtest": self._combo_backtest_section(),
        }

    def save(self, path="data/report.json"):
        """Compact JSON (no indent) - this file can be hundreds of MB with the
        full combo search, and it's read by the ui/ Streamlit app, not by hand."""
        report = self.build()
        with open(path, "w") as f:
            json.dump(report, f, default=_json_safe)
        print(f"[ReportExporter] Saved -> {path}")
        return path
