"""Simple Flask viewer for data/combo_results.db.

Run from the project root:
    python3 -m strategy_finder.webapp

Read-only - just queries results_db and renders them. The combo search
itself (strategy_finder.main) is what populates the database.
"""

from flask import Flask, jsonify, render_template, request

from . import db as results_db

app = Flask(__name__)

PER_PAGE = 50


@app.route("/api/chart-data")
def chart_data():
    top_combos = results_db.get_top_combos_for_chart(limit=15)
    return jsonify(top_combos)


@app.route("/")
def index():

    direction = request.args.get("direction") or None
    min_size = request.args.get("min_size", type=int)
    min_trades = request.args.get("min_trades", type=int)
    sort_by = request.args.get("sort_by", "total_pnl")
    sort_dir = request.args.get("sort_dir", "desc")
    page = max(1, request.args.get("page", 1, type=int))

    rows, total = results_db.fetch_results(
        direction=direction,
        min_size=min_size,
        min_trades=min_trades,
        sort_by=sort_by,
        sort_dir=sort_dir,
        limit=PER_PAGE,
        offset=(page - 1) * PER_PAGE,
    )
    stats = results_db.summary_stats()
    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)

    return render_template(
        "index.html",
        rows=rows,
        stats=stats,
        total=total,
        page=page,
        total_pages=total_pages,
        direction=direction or "",
        min_size=min_size or "",
        min_trades=min_trades or "",
        sort_by=sort_by,
        sort_dir=sort_dir,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
