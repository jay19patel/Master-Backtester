"""Flask + HTMX live-trading dashboard.

Two independent things run in this one process:
  1. An APScheduler background job that calls engine.run_cycle() every
     CYCLE_INTERVAL_MINUTES - fetches fresh data, re-simulates, persists any
     new trades/signals to JSON.
  2. A Flask web server that just READS those JSON files and renders them -
     it never fetches data or runs the simulation itself, so viewing the
     dashboard is always instant regardless of how slow a fetch cycle is.

Usage:
    uv run --project .. python app.py
    -> open http://127.0.0.1:5000
"""

from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, render_template, request

import engine

app = Flask(__name__)

PAGE_SIZE = 15


def compute_stats():
    state = engine.load_state()
    trades = engine.load_trades()

    balance = state.get("balance", engine.INITIAL_CAPITAL)
    open_positions = state.get("open_positions", [])
    closed_trades = len(trades)
    open_trades = len(open_positions)

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    long_closed = [t for t in trades if t["direction"] == "LONG"]
    short_closed = [t for t in trades if t["direction"] == "SHORT"]
    long_open = [p for p in open_positions if p["direction"] == "LONG"]
    short_open = [p for p in open_positions if p["direction"] == "SHORT"]

    return {
        "balance": round(balance, 2),
        "initial_capital": engine.INITIAL_CAPITAL,
        "total_pnl": round(balance - engine.INITIAL_CAPITAL, 2),
        "return_pct": round((balance - engine.INITIAL_CAPITAL) / engine.INITIAL_CAPITAL * 100, 1),
        "total_trades": closed_trades + open_trades,
        "closed_trades": closed_trades,
        "open_trades": open_trades,
        "long_trades": len(long_closed) + len(long_open),
        "short_trades": len(short_closed) + len(short_open),
        "win_rate_pct": round(len(wins) / closed_trades * 100, 1) if closed_trades else None,
        "total_profit": round(sum(t["pnl"] for t in wins), 2),
        "total_loss": round(sum(t["pnl"] for t in losses), 2),
        "last_run_at": state.get("last_run_at"),
        "last_candle_time": state.get("last_candle_time"),
        "anchor_date": state.get("anchor_date"),
        "open_positions": open_positions,
    }


def build_equity_svg(trades, width=700, height=220):
    """Minimal dependency-free SVG line chart of the equity curve - same
    self-contained approach as the backtest dashboard's charts, no JS libs."""
    values = [engine.INITIAL_CAPITAL] + [t["equity_after"] for t in trades]
    if len(values) < 2:
        return None

    padding = 28
    lo, hi = min(values + [engine.INITIAL_CAPITAL]), max(values + [engine.INITIAL_CAPITAL])
    span = (hi - lo) or 1
    step_x = (width - 2 * padding) / max(1, len(values) - 1)

    def point(i, v):
        x = padding + i * step_x
        y = height - padding - ((v - lo) / span) * (height - 2 * padding)
        return f"{x:.2f},{y:.2f}"

    points = " ".join(point(i, v) for i, v in enumerate(values))
    baseline_y = height - padding - ((engine.INITIAL_CAPITAL - lo) / span) * (height - 2 * padding)
    color = "#3ecf8e" if values[-1] >= engine.INITIAL_CAPITAL else "#f2545b"

    return {
        "width": width,
        "height": height,
        "points": points,
        "baseline_y": round(baseline_y, 2),
        "color": color,
        "min_val": round(lo, 2),
        "max_val": round(hi, 2),
    }


@app.route("/")
def index():
    stats = compute_stats()
    all_trades = engine.load_trades()
    chart = build_equity_svg(all_trades)

    recent_trades = list(reversed(all_trades))
    total = len(recent_trades)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    return render_template(
        "dashboard.html",
        stats=stats,
        chart=chart,
        cycle_minutes=engine.CYCLE_INTERVAL_MINUTES,
        trades=recent_trades[:PAGE_SIZE],
        page=1,
        total_pages=total_pages,
        total=total,
    )


@app.route("/fragments/stats")
def fragment_stats():
    return render_template("_stats.html", stats=compute_stats())


@app.route("/fragments/chart")
def fragment_chart():
    trades = engine.load_trades()
    return render_template("_chart.html", chart=build_equity_svg(trades))


@app.route("/fragments/trades")
def fragment_trades():
    page = max(1, int(request.args.get("page", 1)))
    trades = list(reversed(engine.load_trades()))  # most recent first
    total = len(trades)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages)
    start = (page - 1) * PAGE_SIZE
    page_trades = trades[start : start + PAGE_SIZE]
    return render_template(
        "_trades.html", trades=page_trades, page=page, total_pages=total_pages, total=total
    )


@app.route("/run-now", methods=["POST"])
def run_now():
    engine.run_cycle()
    return render_template("_stats.html", stats=compute_stats())


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        engine.run_cycle,
        "interval",
        minutes=engine.CYCLE_INTERVAL_MINUTES,
        next_run_time=datetime.now(),  # run once immediately on startup, then every N minutes
        id="live_cycle",
    )
    scheduler.start()
    return scheduler


if __name__ == "__main__":
    start_scheduler()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
