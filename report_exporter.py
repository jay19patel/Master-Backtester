"""ReportExporter: runs every analysis in this project once and writes a single
JSON file that `dashboard.html` reads and renders.

Usage:
    ReportExporter(df, config).save("report.json")

`config` is a plain dict of every setting main.py already has as module-level
constants (symbol, interval, backtest/portfolio parameters, which sections to
run, ...) - this module stays independent of main.py so there's no circular import.
"""

import json
import math

import pandas as pd

from backtester import Backtester
from condition_finder import ConditionFinder
from oracle_backtester import OracleBacktester
from portfolio_manager import PortfolioManager
from relevance_analyzer import RelevanceAnalyzer
from signal_combo_backtester import SignalComboBacktester
from signal_evaluator import evaluate_signals

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


def safe_round(value, ndigits=2):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(value) or math.isinf(value)) else round(value, ndigits)


def records(df):
    """DataFrame -> list of JSON-safe dicts. None/empty-safe."""
    if df is None or len(df) == 0:
        return []
    return [{k: _json_safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


class ReportExporter:
    """Collects every analysis into one JSON-serializable dict and writes it out.

    Usage:
        ReportExporter(df, config).save("report.json")
    """

    def __init__(self, df, config):
        self.df = df
        self.config = config

    def _column_groups(self):
        df = self.df
        ohlcv = [c for c in OHLCV_COLUMNS if c in df.columns]
        oracle = [c for c in df.columns if c.startswith("oracle_")]
        indicator = [c for c in df.columns if c not in ohlcv and c not in oracle]
        return ohlcv, oracle, indicator

    def _dataset_section(self):
        df = self.df
        ohlcv, oracle, indicator = self._column_groups()
        return {
            "symbol": self.config.get("symbol"),
            "interval": self.config.get("interval"),
            "rows": len(df),
            "columns": len(df.columns),
            "date_start": df.index.min().isoformat(),
            "date_end": df.index.max().isoformat(),
            "missing_values": int(df.isna().sum().sum()),
            "column_groups": {"ohlcv": ohlcv, "oracle": oracle, "indicator": indicator},
        }

    def _oracle_section(self):
        df = self.df
        if "oracle_signal" not in df.columns:
            return None

        counts = df["oracle_signal"].value_counts(dropna=False)
        signal_counts = {(k if pd.notna(k) else "UNKNOWN"): int(v) for k, v in counts.items()}

        known = df[df["oracle_signal"].notna()]
        stats_cols = ["oracle_upside_pct", "oracle_downside_pct", "oracle_upside_gap", "oracle_downside_gap"]
        avg = known.groupby("oracle_signal")[stats_cols].mean()

        signal_meaning = {}
        for signal in ["BUY", "HOLD", "SELL"]:
            if signal not in avg.index:
                continue
            row = avg.loc[signal]
            signal_meaning[signal] = {
                "candles": int((known["oracle_signal"] == signal).sum()),
                "avg_upside_pct": safe_round(row["oracle_upside_pct"], 3),
                "avg_downside_pct": safe_round(row["oracle_downside_pct"], 3),
                "avg_upside_gap": safe_round(row["oracle_upside_gap"], 4),
                "avg_downside_gap": safe_round(row["oracle_downside_gap"], 4),
            }

        return {
            "min_reward_risk_ratio": self.config.get("oracle_min_reward_risk_ratio"),
            "signal_counts": signal_counts,
            "signal_meaning": signal_meaning,
        }

    def _ema_signal_section(self):
        df = self.df
        ema_col = self.config.get("ema_column", "EMA_20")
        if ema_col not in df.columns or "oracle_signal" not in df.columns:
            return None

        known = df[df["oracle_signal"].notna() & df[ema_col].notna()].copy()
        above, below = f"Above {ema_col}", f"Below {ema_col}"
        known["ema_state"] = (known["Close"] >= known[ema_col]).map({True: above, False: below})

        row_order = [name for name in [above, below] if name in known["ema_state"].unique()]
        col_order = [s for s in ["BUY", "HOLD", "SELL"] if s in known["oracle_signal"].unique()]

        counts = pd.crosstab(known["ema_state"], known["oracle_signal"]).reindex(
            index=row_order, columns=col_order, fill_value=0
        )
        row_pct = counts.div(counts.sum(axis=1), axis=0) * 100

        return {
            "ema_column": ema_col,
            "rows": [
                {
                    "state": state,
                    "counts": {c: int(counts.loc[state, c]) for c in col_order},
                    "row_pct": {c: safe_round(row_pct.loc[state, c], 1) for c in col_order},
                }
                for state in row_order
            ],
        }

    def _relevance_section(self):
        if not self.config.get("run_relevance_analysis"):
            return None
        result = RelevanceAnalyzer(self.df).analyze()
        return records(result)

    def _condition_section(self):
        if not self.config.get("run_condition_search"):
            return None
        finder = ConditionFinder(
            self.df,
            min_support=self.config.get("condition_min_support", 100),
            max_combo_size=self.config.get("condition_max_combo_size", 3),
        )
        out = {}
        for target in ["BUY", "SELL"]:
            result, base_rate = finder.find_best_combinations(target)
            out[target] = {"base_rate_pct": safe_round(base_rate, 2), "combinations": records(result)}
        return out

    def _price_action_section(self):
        if not self.config.get("run_price_action"):
            return None
        result = evaluate_signals(
            self.df,
            forward_bars=self.config.get("price_action_forward_bars", 10),
            min_fires=self.config.get("price_action_min_fires", 15),
        )
        return records(result)

    def _backtest_section(self):
        if not self.config.get("run_backtest"):
            return None, None

        bt = Backtester(
            self.df,
            initial_capital=self.config["backtest_initial_capital"],
            risk_per_trade_pct=self.config["backtest_risk_per_trade_pct"],
            stop_loss_pct=self.config["backtest_stop_loss_pct"],
            take_profit_pct=self.config["backtest_take_profit_pct"],
            max_hold_bars=self.config["backtest_max_hold_bars"],
            fee_pct=self.config["backtest_fee_pct"],
        )
        result = bt.run()
        comparison = bt.run_no_fee_comparison()

        best_signal_curve = []
        best_signal_name = None
        if not result.empty:
            best_signal_name = result.iloc[0]["signal"]
            best_trades, _ = bt._simulate_signal(best_signal_name)
            best_signal_curve = [self.config["backtest_initial_capital"]] + [
                round(t["equity_after"], 2) for t in best_trades
            ]

        section = {
            "config": {
                "initial_capital": self.config["backtest_initial_capital"],
                "risk_per_trade_pct": self.config["backtest_risk_per_trade_pct"],
                "stop_loss_pct": self.config["backtest_stop_loss_pct"],
                "take_profit_pct": self.config["backtest_take_profit_pct"],
                "max_hold_bars": self.config["backtest_max_hold_bars"],
                "fee_pct": self.config["backtest_fee_pct"],
            },
            "breakeven_win_rate_pct": safe_round(bt.breakeven_win_rate_pct, 2),
            "fee_adjusted_breakeven_win_rate_pct": safe_round(bt.fee_adjusted_breakeven_win_rate_pct, 2),
            "results": records(result),
            "fee_comparison": records(comparison),
            "best_signal_name": best_signal_name,
            "best_signal_equity_curve": best_signal_curve,
        }
        return section, result

    def _portfolio_section(self, backtest_result):
        if not self.config.get("run_portfolio") or backtest_result is None or backtest_result.empty:
            return None

        profitable = backtest_result.loc[backtest_result["total_pnl"] > 0, "signal"].tolist()
        if not profitable:
            return {"signals_used": [], "message": "No standalone-profitable signals found."}

        pm = PortfolioManager(
            self.df,
            signals=profitable,
            initial_capital=self.config["backtest_initial_capital"],
            risk_per_trade_pct=self.config["backtest_risk_per_trade_pct"],
            stop_loss_pct=self.config["backtest_stop_loss_pct"],
            take_profit_pct=self.config["backtest_take_profit_pct"],
            max_hold_bars=self.config["backtest_max_hold_bars"],
            fee_pct=self.config["backtest_fee_pct"],
            max_concurrent_trades=self.config["portfolio_max_concurrent_trades"],
            portfolio_risk_cap_pct=self.config["portfolio_risk_cap_pct"],
            drawdown_throttle_trigger_pct=self.config["portfolio_drawdown_throttle_trigger_pct"],
            drawdown_recovery_pct=self.config["portfolio_drawdown_recovery_pct"],
            throttled_risk_pct=self.config["portfolio_throttled_risk_pct"],
        )
        trades, final_equity, equity_curve = pm.run()
        n_trades = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]

        exit_reasons = {}
        per_signal = {}
        for t in trades:
            exit_reasons[t["exit_reason"]] = exit_reasons.get(t["exit_reason"], 0) + 1
            row = per_signal.setdefault(t["signal"], {"trades": 0, "pnl": 0.0})
            row["trades"] += 1
            row["pnl"] += t["pnl"]

        return {
            "signals_used": profitable,
            "config": {
                "max_concurrent_trades": self.config["portfolio_max_concurrent_trades"],
                "portfolio_risk_cap_pct": self.config["portfolio_risk_cap_pct"],
                "drawdown_throttle_trigger_pct": self.config["portfolio_drawdown_throttle_trigger_pct"],
                "drawdown_recovery_pct": self.config["portfolio_drawdown_recovery_pct"],
                "throttled_risk_pct": self.config["portfolio_throttled_risk_pct"],
            },
            "summary": {
                "trades": n_trades,
                "win_rate_pct": safe_round(len(wins) / n_trades * 100, 1) if n_trades else None,
                "final_equity": safe_round(final_equity, 2),
                "total_pnl": safe_round(final_equity - self.config["backtest_initial_capital"], 2),
                "max_drawdown_pct": safe_round(PortfolioManager._max_drawdown_pct(equity_curve), 1),
            },
            "exit_reasons": exit_reasons,
            "per_signal_contribution": [
                {"signal": s, "trades": v["trades"], "pnl": safe_round(v["pnl"], 2)}
                for s, v in sorted(per_signal.items(), key=lambda kv: -kv[1]["pnl"])
            ],
            "equity_curve": [round(v, 2) for v in equity_curve],
        }

    def _signal_combo_section(self):
        if not self.config.get("run_signal_combo_backtest"):
            return None
        combo_bt = SignalComboBacktester(
            self.df,
            initial_capital=self.config["backtest_initial_capital"],
            risk_per_trade_pct=self.config["backtest_risk_per_trade_pct"],
            stop_loss_pct=self.config["backtest_stop_loss_pct"],
            take_profit_pct=self.config["backtest_take_profit_pct"],
            max_hold_bars=self.config["backtest_max_hold_bars"],
            fee_pct=self.config["backtest_fee_pct"],
            min_combo_size=self.config.get("signal_combo_min_size", 2),
            max_combo_size=self.config.get("signal_combo_max_size", 4),
            min_fires=self.config.get("signal_combo_min_fires", 15),
        )
        result = combo_bt.run()
        return {
            "config": {
                "min_combo_size": self.config.get("signal_combo_min_size", 2),
                "max_combo_size": self.config.get("signal_combo_max_size", 4),
                "min_fires": self.config.get("signal_combo_min_fires", 15),
            },
            "combinations": records(result),
        }

    def _oracle_backtest_section(self):
        if not self.config.get("run_oracle_backtest") or "oracle_signal" not in self.df.columns:
            return None
        ob = OracleBacktester(
            self.df,
            initial_capital=self.config["backtest_initial_capital"],
            risk_per_trade_pct=self.config["backtest_risk_per_trade_pct"],
            stop_loss_pct=self.config["backtest_stop_loss_pct"],
            take_profit_pct=self.config["backtest_take_profit_pct"],
            max_hold_bars=self.config["backtest_max_hold_bars"],
            fee_pct=self.config["backtest_fee_pct"],
            portfolio_kwargs=dict(
                max_concurrent_trades=self.config["portfolio_max_concurrent_trades"],
                portfolio_risk_cap_pct=self.config["portfolio_risk_cap_pct"],
                drawdown_throttle_trigger_pct=self.config["portfolio_drawdown_throttle_trigger_pct"],
                drawdown_recovery_pct=self.config["portfolio_drawdown_recovery_pct"],
                throttled_risk_pct=self.config["portfolio_throttled_risk_pct"],
            ),
        )
        standalone, managed = ob.run()
        return {
            "warning": (
                "NOT a tradeable strategy - oracle_signal is built from the next "
                f"{self.config.get('oracle_lookahead')} candles' actual future high/low, so this shows "
                "the theoretical best case under this bracket/fee setup, not something you can trade live."
            ),
            "standalone": standalone,
            "managed": managed,
        }

    def build(self):
        backtest_section, backtest_df = self._backtest_section()
        return {
            "generated_at": self.config.get("generated_at"),
            "config": {k: v for k, v in self.config.items() if k != "generated_at"},
            "dataset": self._dataset_section(),
            "oracle": self._oracle_section(),
            "ema_signal_relationship": self._ema_signal_section(),
            "relevance": self._relevance_section(),
            "condition_search": self._condition_section(),
            "price_action_evaluation": self._price_action_section(),
            "backtest": backtest_section,
            "portfolio": self._portfolio_section(backtest_df),
            "signal_combo_backtest": self._signal_combo_section(),
            "oracle_backtest": self._oracle_backtest_section(),
        }

    def save(self, path="report.json"):
        report = self.build()
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=_json_safe)
        print(f"[ReportExporter] Saved -> {path}")
        return path
