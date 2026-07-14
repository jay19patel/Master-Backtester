"""Live paper-trading engine.

Design: keep a small ROLLING WINDOW of candles (BUFFER_DAYS, enough for every
indicator to be fully warmed up) in memory only - raw OHLCV data is never
written to disk. On the very first run (or after a process restart, since the
in-memory buffer is lost) a one-time larger fetch builds this buffer. Every
cycle after that needs only ONE lightweight API call (a couple of days of
candles - comfortably one HTTP request, see DataFetcher.CHUNK_FREQ), merged
into the buffer and trimmed back down to BUFFER_DAYS.

Trading state (balance, open positions, pending signal, drawdown-throttle
status, and exactly which candle has already been processed) DOES persist to
a small JSON file, so a restart resumes real trading progress even though the
candle buffer itself has to be rebuilt. Positions are tracked by absolute
entry_time (not a bar index), which is exactly what lets PortfolioManager
resume a simulation on a completely different (freshly re-fetched) DataFrame
each cycle - see portfolio_manager.PortfolioManager.run_incremental().

State lives in two small JSON files (data/state.json, data/trades.json,
data/signals.json) - no database, matches the rest of this project's
lightweight, file-based approach.
"""

import json
import os

import pandas as pd

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

# The in-memory rolling candle buffer, kept small on purpose - big enough for
# every indicator's own lookback (the slowest need ~100 bars = ~4 days) plus
# comfortable margin, never written to disk.
BUFFER_DAYS = 25
# Steady-state fetch: just enough overlap to safely catch new candles even if
# a cycle or two was missed, small enough to stay ONE API call (well under
# DataFetcher.CHUNK_FREQ's 7-day chunk size).
STEADY_STATE_FETCH_DAYS = 2

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
TRADES_PATH = os.path.join(DATA_DIR, "trades.json")
SIGNALS_PATH = os.path.join(DATA_DIR, "signals.json")

# In-memory only - deliberately NOT persisted. Lost on process restart, which
# just costs one bootstrap-sized fetch to rebuild; trading state is unaffected
# since that lives in STATE_PATH instead.
_buffer_df = None


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _save_json(path, payload):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _now_iso():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def load_state():
    return _load_json(STATE_PATH, {})


def load_trades():
    return _load_json(TRADES_PATH, [])


def load_signals():
    return _load_json(SIGNALS_PATH, [])


def _serialize_position(p):
    p = dict(p)
    p["entry_time"] = str(p["entry_time"])
    return p


def _deserialize_position(p):
    p = dict(p)
    p["entry_time"] = pd.Timestamp(p["entry_time"])
    return p


def _merge_and_trim(buffer_df, fresh_df, buffer_days):
    combined = pd.concat([buffer_df, fresh_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    cutoff = combined.index[-1] - pd.Timedelta(days=buffer_days)
    return combined[combined.index >= cutoff]


def run_cycle():
    """Refresh the in-memory candle buffer (one small API call in steady
    state), re-derive signals, advance the live simulation by whatever new
    candles arrived, and persist any new trades/signals. Returns a short
    summary dict for logging - the dashboard reads the JSON files directly."""
    global _buffer_df

    state = load_state()
    is_first_run_ever = "last_processed_time" not in state

    if _buffer_df is None:
        # First run ever, OR the process just (re)started and lost its
        # in-memory buffer - one larger fetch to (re)build proper indicator
        # history. Trading state itself (below) is untouched by this.
        _buffer_df = DataFetcher(symbol=SYMBOL, interval=INTERVAL, total_days=BUFFER_DAYS).fetch_nocache()
        if _buffer_df is None or _buffer_df.empty:
            _buffer_df = None
            return {"ok": False, "error": "no data returned from bootstrap fetch"}
    else:
        fresh = DataFetcher(symbol=SYMBOL, interval=INTERVAL, total_days=STEADY_STATE_FETCH_DAYS).fetch_nocache()
        if fresh.empty:
            return {"ok": False, "error": "no data returned from fetch"}
        _buffer_df = _merge_and_trim(_buffer_df, fresh, BUFFER_DAYS)

    features_df = build_features(_buffer_df)

    if is_first_run_ever:
        # Establish the indicator baseline only - don't retroactively "trade"
        # the whole bootstrap window. Live trading starts from the next
        # genuinely new candle onward.
        state = {
            "balance": INITIAL_CAPITAL,
            "peak_equity": INITIAL_CAPITAL,
            "throttled": False,
            "open_positions": [],
            "pending_entries": [],
            "last_processed_time": str(features_df.index[-1]),
            "last_run_at": _now_iso(),
        }
        _save_json(STATE_PATH, state)
        return {
            "ok": True,
            "bootstrap": True,
            "balance": INITIAL_CAPITAL,
            "new_trades": 0,
            "open_positions": 0,
            "pending_signals": 0,
            "last_candle_time": state["last_processed_time"],
        }

    active = [s for s in STRATEGIES if s["name"] in ACTIVE_STRATEGY_NAMES]
    strategy_arrays = [
        {"name": s["name"], "combo": s["combo"], "direction_array": build_direction_array(features_df, s)}
        for s in active
    ]

    pm = PortfolioManager(
        features_df,
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

    prior_pending = [tuple(p) for p in state.get("pending_entries", [])]
    prior_state = {
        "balance": state["balance"],
        "peak_equity": state.get("peak_equity", state["balance"]),
        "throttled": state.get("throttled", False),
        "open_positions": [_deserialize_position(p) for p in state.get("open_positions", [])],
        "pending_entries": prior_pending,
        "last_processed_time": pd.Timestamp(state["last_processed_time"]),
    }

    trades, equity, _, open_positions, pending_entries, peak_equity, throttled = pm.run_incremental(prior_state)

    if trades:
        existing_trades = load_trades()
        existing_trades.extend(trades)
        _save_json(TRADES_PATH, existing_trades)

    if pending_entries and pending_entries != prior_pending:
        existing_signals = load_signals()
        for name, direction in pending_entries:
            existing_signals.append(
                {
                    "strategy": name,
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "detected_at": _now_iso(),
                    "status": "pending_fill",  # will appear as an actual trade next cycle once filled
                }
            )
        _save_json(SIGNALS_PATH, existing_signals)

    state["balance"] = round(equity, 2)
    state["peak_equity"] = round(peak_equity, 2)
    state["throttled"] = throttled
    state["open_positions"] = [_serialize_position(p) for p in open_positions]
    state["pending_entries"] = list(pending_entries)
    state["last_processed_time"] = str(features_df.index[-1])
    state["last_run_at"] = _now_iso()
    _save_json(STATE_PATH, state)

    return {
        "ok": True,
        "balance": state["balance"],
        "new_trades": len(trades),
        "open_positions": len(open_positions),
        "pending_signals": len(pending_entries),
        "last_candle_time": state["last_processed_time"],
    }


if __name__ == "__main__":
    result = run_cycle()
    print(json.dumps(result, indent=2))
