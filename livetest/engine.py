"""Live paper-trading engine.

Design: rather than hand-rolling fragile incremental bar-by-bar state
tracking, each cycle re-fetches EVERY candle since a fixed `anchor_date`
(persisted once, the moment live testing first started) and re-runs the
exact same PortfolioManager simulation used for backtesting, from
initial_capital forward. This is deterministic and self-healing - the
"current live state" (balance, open positions, pending signal) is always
just whatever that full-history simulation says right now, never state that
can drift or get corrupted by a missed cycle. New trades (and a freshly
fired, not-yet-filled signal) are detected by diffing against what was
already recorded, then appended to trades.json / signals.json.

State lives in two small JSON files (data/state.json, data/trades.json,
data/signals.json) - no database, matches the rest of this project's
lightweight, file-based approach.
"""

import json
import os
from datetime import datetime, timedelta, timezone

from data_fetcher import DataFetcher
from portfolio_manager import PortfolioManager
from strategies import STRATEGIES, build_direction_array, build_features

SYMBOL = "ETHUSD"
INTERVAL = "1h"
ACTIVE_STRATEGY_NAMES = ["strategy_01", "strategy_02"]

INITIAL_CAPITAL = 100.0
RISK_PER_TRADE_PCT = 2.0
STOP_LOSS_PCT = 1
TAKE_PROFIT_PCT = 3
MAX_HOLD_BARS = 20
FEE_PCT = 0.05
MAX_LEVERAGE = 2.0

PORTFOLIO_MAX_CONCURRENT_TRADES = 5
PORTFOLIO_RISK_CAP_PCT = 10.0
PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT = 10.0
PORTFOLIO_DRAWDOWN_RECOVERY_PCT = 5.0
PORTFOLIO_THROTTLED_RISK_PCT = 1.0

CYCLE_INTERVAL_MINUTES = 20

# Days of history fetched purely so indicators (some need 100+ bars) are
# warmed up by the time trading actually starts - never traded on directly.
WARMUP_DAYS = 120
# On the very first run, how many of the most-recent days become the initial
# live trading window (grows from there every cycle after).
INITIAL_TRADE_WINDOW_DAYS = 3

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
TRADES_PATH = os.path.join(DATA_DIR, "trades.json")
SIGNALS_PATH = os.path.join(DATA_DIR, "signals.json")


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _save_json(path, payload):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _trade_key(t):
    return f"{t['strategy']}|{t['entry_time']}|{t['exit_time']}"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_state():
    return _load_json(STATE_PATH, {})


def load_trades():
    return _load_json(TRADES_PATH, [])


def load_signals():
    return _load_json(SIGNALS_PATH, [])


def run_cycle():
    """Fetch fresh data, re-run the full simulation, persist any NEW trades
    and any freshly-fired (not yet filled) signal. Returns a short summary
    dict for logging - the dashboard reads the JSON files directly."""
    state = load_state()
    first_run = "anchor_date" not in state

    if first_run:
        # Bootstrap: fetch warmup + a small initial trading window, THEN anchor
        # to (last available candle - INITIAL_TRADE_WINDOW_DAYS). Anchoring to
        # "now" directly doesn't work - the newest candle is always somewhat
        # older than "now" (the current hour isn't complete yet), so there'd be
        # zero candles at/after a "now" anchor on day one.
        total_days = WARMUP_DAYS + INITIAL_TRADE_WINDOW_DAYS
        df = DataFetcher(symbol=SYMBOL, interval=INTERVAL, total_days=total_days).fetch(force_refresh=True)
        if df.empty:
            return {"ok": False, "error": "no data returned from fetch"}
        anchor_date = df.index[-1].to_pydatetime() - timedelta(days=INITIAL_TRADE_WINDOW_DAYS)
        state["anchor_date"] = anchor_date.isoformat()
    else:
        anchor_date = datetime.fromisoformat(state["anchor_date"])
        # Fetch WARMUP_DAYS of history BEFORE the anchor too, so indicators
        # (some need ~100+ bars) are fully warmed up by the time trading
        # actually starts at anchor_date - only candles from anchor_date
        # onward are ever traded (see the slice below); the warmup days are
        # lookback-only, never traded on directly.
        total_days = WARMUP_DAYS + max(1, (datetime.now(timezone.utc) - anchor_date).days + 2)
        df = DataFetcher(symbol=SYMBOL, interval=INTERVAL, total_days=total_days).fetch(force_refresh=True)
        if df.empty:
            return {"ok": False, "error": "no data returned from fetch"}

    df = build_features(df)

    active = [s for s in STRATEGIES if s["name"] in ACTIVE_STRATEGY_NAMES]
    strategy_arrays_full = [
        {"name": s["name"], "combo": s["combo"], "direction_array": build_direction_array(df, s)} for s in active
    ]

    # Trade only from anchor_date forward - the warmup days before it were
    # fetched purely to give indicators real history, not to be traded on.
    anchor_local = anchor_date.astimezone(df.index.tz)
    trade_mask = df.index >= anchor_local
    sim_df = df[trade_mask].copy()
    if sim_df.empty:
        return {"ok": False, "error": "no candles at/after anchor_date yet"}

    strategy_arrays = [{**s, "direction_array": s["direction_array"][trade_mask]} for s in strategy_arrays_full]

    pm = PortfolioManager(
        sim_df,
        strategy_arrays,
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade_pct=RISK_PER_TRADE_PCT,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT,
        max_hold_bars=MAX_HOLD_BARS,
        fee_pct=FEE_PCT,
        max_leverage=MAX_LEVERAGE,
        max_concurrent_trades=PORTFOLIO_MAX_CONCURRENT_TRADES,
        portfolio_risk_cap_pct=PORTFOLIO_RISK_CAP_PCT,
        drawdown_throttle_trigger_pct=PORTFOLIO_DRAWDOWN_THROTTLE_TRIGGER_PCT,
        drawdown_recovery_pct=PORTFOLIO_DRAWDOWN_RECOVERY_PCT,
        throttled_risk_pct=PORTFOLIO_THROTTLED_RISK_PCT,
    )
    trades, equity, equity_curve, open_positions, pending_entries = pm.run()

    # --- Diff against what's already recorded, append only what's NEW ---
    known_keys = set(state.get("recorded_trade_keys", []))
    new_trades = [t for t in trades if _trade_key(t) not in known_keys]
    if new_trades:
        existing_trades = load_trades()
        existing_trades.extend(new_trades)
        _save_json(TRADES_PATH, existing_trades)
        known_keys |= {_trade_key(t) for t in new_trades}

    last_candle_time = str(sim_df.index[-1])
    if pending_entries and state.get("last_signal_candle_time") != last_candle_time:
        existing_signals = load_signals()
        for name, direction in pending_entries:
            existing_signals.append(
                {
                    "strategy": name,
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "signal_candle_time": last_candle_time,
                    "detected_at": _now_iso(),
                    "status": "pending_fill",  # will appear as an actual trade next cycle once filled
                }
            )
        _save_json(SIGNALS_PATH, existing_signals)
        state["last_signal_candle_time"] = last_candle_time

    # --- Persist current live state ---
    state["recorded_trade_keys"] = sorted(known_keys)
    state["balance"] = round(equity, 2)
    state["open_positions"] = [
        {
            "strategy": p["strategy"],
            "direction": "LONG" if p["direction"] == 1 else "SHORT",
            "entry_time": str(sim_df.index[p["entry_i"]]),
            "entry_price": round(p["entry_price"], 6),
            "stop_price": round(p["stop_price"], 6),
            "target_price": round(p["target_price"], 6),
            "position_size": round(p["position_size"], 6),
        }
        for p in open_positions
    ]
    state["last_candle_time"] = last_candle_time
    state["last_run_at"] = _now_iso()
    state["total_trades"] = len(trades)
    _save_json(STATE_PATH, state)

    return {
        "ok": True,
        "balance": state["balance"],
        "new_trades": len(new_trades),
        "open_positions": len(open_positions),
        "pending_signals": len(pending_entries),
        "last_candle_time": last_candle_time,
    }


if __name__ == "__main__":
    result = run_cycle()
    print(json.dumps(result, indent=2))
