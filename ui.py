"""ui.py: Flask app serving the dashboard (dashboard.html).

report.json can be hundreds of MB once the full combo search runs (600k+
profitable combos) - fetching that straight into a browser tab is what made
the static dashboard.html "never load" (huge fetch + JSON.parse, easily
freezing/crashing the tab). This app loads report.json once server-side
(cached, reloaded automatically if the file's mtime changes) and exposes
small, paginated/filtered/sorted API endpoints instead - the browser only
ever receives one page of rows at a time.

Usage:
    python3 ui.py
    open http://127.0.0.1:5000

Does NOT run main.py or finalbacktesting.py - it only reads whatever
report.json / final_backtest_data.json already exist in this folder.
"""

import json
import os
import time

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, "report.json")
FINAL_BACKTEST_PATH = os.path.join(BASE_DIR, "final_backtest_data.json")

SORT_COLUMNS = {"size", "fires", "trades", "win_rate_pct", "final_equity", "total_pnl", "return_pct"}

app = Flask(__name__)

_cache = {
    "report": None, "report_mtime": None, "dictionary": [], "combos": [],
    "final_backtest": None, "final_backtest_mtime": None,
}


def _load_report():
    """Loads + caches report.json, reloading only if its mtime changed."""
    if not os.path.exists(REPORT_PATH):
        return None
    mtime = os.path.getmtime(REPORT_PATH)
    if _cache["report"] is not None and _cache["report_mtime"] == mtime:
        return _cache["report"]

    print(f"[ui.py] Loading {REPORT_PATH} ...")
    t0 = time.monotonic()
    with open(REPORT_PATH) as f:
        report = json.load(f)
    cb = report.get("combo_backtest") or {}

    _cache["report"] = report
    _cache["report_mtime"] = mtime
    _cache["dictionary"] = cb.get("condition_dictionary", [])
    _cache["combos"] = cb.get("combinations", [])
    print(f"[ui.py] Loaded {len(_cache['combos']):,} combos in {time.monotonic() - t0:.1f}s")
    return report


def _load_final_backtest():
    if not os.path.exists(FINAL_BACKTEST_PATH):
        return None
    mtime = os.path.getmtime(FINAL_BACKTEST_PATH)
    if _cache["final_backtest"] is not None and _cache["final_backtest_mtime"] == mtime:
        return _cache["final_backtest"]
    with open(FINAL_BACKTEST_PATH) as f:
        data = json.load(f)
    _cache["final_backtest"] = data
    _cache["final_backtest_mtime"] = mtime
    return data


def _combo_text(condition_indices):
    dictionary = _cache["dictionary"]
    return " AND ".join(dictionary[i] for i in condition_indices)


def _combo_to_row(c):
    return {
        "direction": c["direction"],
        "combo": _combo_text(c["conditions"]),
        "size": c["size"],
        "fires": c["fires"],
        "trades": c["trades"],
        "win_rate_pct": c["win_rate_pct"],
        "final_equity": c["final_equity"],
        "total_pnl": c["total_pnl"],
        "return_pct": c["return_pct"],
    }


def _filtered_combos(args):
    """All server-side filtering for the combos tab - direction/size/trades/
    win-rate from the filter toolbar, plus free-text search (matched against
    condition NAMES via the small dictionary, not by materializing every
    combo's full text, which would be the expensive part at 600k+ rows)."""
    combos = _cache["combos"]
    direction = args.get("direction")
    min_size = args.get("min_size", type=float)
    max_size = args.get("max_size", type=float)
    min_trades = args.get("min_trades", type=float)
    min_winrate = args.get("min_winrate", type=float)
    search = (args.get("search") or "").strip().lower()

    matching_idx = None
    if search:
        dictionary = _cache["dictionary"]
        matching_idx = {i for i, name in enumerate(dictionary) if search in name.lower()}

    def keep(c):
        if direction in ("Long", "Short") and c["direction"] != direction:
            return False
        if min_size is not None and c["size"] < min_size:
            return False
        if max_size is not None and c["size"] > max_size:
            return False
        if min_trades is not None and c["trades"] < min_trades:
            return False
        if min_winrate is not None and c["win_rate_pct"] < min_winrate:
            return False
        if matching_idx is not None and matching_idx.isdisjoint(c["conditions"]):
            return False
        return True

    return [c for c in combos if keep(c)]


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/api/summary")
def api_summary():
    report = _load_report()
    if report is None:
        return jsonify({"error": "report.json not found - run `python3 main.py` first."}), 404

    cb = report.get("combo_backtest") or {}
    combos = _cache["combos"]
    best = _combo_to_row(max(combos, key=lambda c: c["total_pnl"])) if combos else None

    return jsonify({
        "generated_at": report.get("generated_at"),
        "config": report.get("config"),
        "dataset": report.get("dataset"),
        "combo_config": cb.get("config"),
        "profitable_count": len(combos),
        "best_combo": best,
        "final_backtest_available": os.path.exists(FINAL_BACKTEST_PATH),
    })


@app.route("/api/combos")
def api_combos():
    if _load_report() is None:
        return jsonify({"error": "report.json not found - run `python3 main.py` first."}), 404

    filtered = _filtered_combos(request.args)
    total = len(filtered)

    sort_key = request.args.get("sort", "total_pnl")
    if sort_key not in SORT_COLUMNS:
        sort_key = "total_pnl"
    reverse = request.args.get("dir", "desc") != "asc"
    filtered.sort(key=lambda c: c[sort_key], reverse=reverse)

    page = max(0, request.args.get("page", 0, type=int))
    page_size = min(200, max(1, request.args.get("page_size", 50, type=int)))
    page_rows = filtered[page * page_size: (page + 1) * page_size]

    return jsonify({
        "rows": [_combo_to_row(c) for c in page_rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    })


@app.route("/api/combos/chart")
def api_combos_chart():
    if _load_report() is None:
        return jsonify({"error": "report.json not found - run `python3 main.py` first."}), 404

    filtered = _filtered_combos(request.args)
    kind = request.args.get("kind", "scatter")

    if kind == "bar":
        top = sorted(filtered, key=lambda c: c["total_pnl"], reverse=True)[:10]
        points = [{"label": f"[{c['direction']}] {_combo_text(c['conditions'])}", "value": c["total_pnl"]} for c in top]
        return jsonify({"points": points, "total": len(filtered)})

    # scatter: same best-first stride-sampling the old client-side code did,
    # just done server-side now so the browser never sees more than max_points.
    max_points = 1500
    sample = filtered
    sampled = False
    if len(filtered) > max_points:
        ordered = sorted(filtered, key=lambda c: c["return_pct"], reverse=True)
        stride = max(1, len(ordered) // max_points)
        sample = ordered[::stride][:max_points]
        sampled = True

    points = [
        {
            "x": c["win_rate_pct"], "y": c["return_pct"], "direction": c["direction"],
            "label": f"[{c['direction']}] {_combo_text(c['conditions'])}",
            "trades": c["trades"], "size": c["size"],
        }
        for c in sample
    ]
    return jsonify({"points": points, "total": len(filtered), "sampled": sampled})


@app.route("/api/indicators")
def api_indicators():
    report = _load_report()
    if report is None:
        return jsonify({"error": "report.json not found - run `python3 main.py` first."}), 404
    cb = report.get("combo_backtest") or {}
    return jsonify({
        "contribution": cb.get("indicator_contribution") or {"rows": [], "overall_mean_pnl": 0, "total_combos": 0},
        "redundancy": cb.get("indicator_redundancy", []),
        "perfect_predictors": cb.get("perfect_predictors", []),
        "direction_target_diagnostic": cb.get("direction_target_diagnostic", []),
    })


@app.route("/api/final_backtest")
def api_final_backtest():
    return jsonify(_load_final_backtest())


if __name__ == "__main__":
    print(f"[ui.py] Serving from {BASE_DIR}")
    print("[ui.py] Open http://127.0.0.1:8005")
    app.run(port=8005, debug=False)
