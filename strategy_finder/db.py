"""SQLite persistence for combo backtest results.

Single source of truth for two things:
  1. Crash-safe checkpointing during a run - ComboBacktester inserts each
     batch of rows as soon as it's computed, so a crash/kill mid-run loses
     at most the last in-flight batch, not the whole search.
  2. Ad-hoc querying of past runs' results straight from this database.

Every new run starts by wiping the table (clear_results()) - only the
latest run's results are ever kept, by design.
"""

import json
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = "data/combo_results.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS combo_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    direction TEXT NOT NULL,
    combo TEXT NOT NULL,
    conditions TEXT NOT NULL,
    size INTEGER NOT NULL,
    fires INTEGER NOT NULL,
    trades INTEGER NOT NULL,
    win_rate_pct REAL NOT NULL,
    avg_sl_pct REAL NOT NULL DEFAULT 0.0,
    avg_tp_pct REAL NOT NULL DEFAULT 0.0,
    final_equity REAL NOT NULL,
    total_pnl REAL NOT NULL,
    return_pct REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_combo_results_total_pnl ON combo_results (total_pnl);
CREATE INDEX IF NOT EXISTS idx_combo_results_direction ON combo_results (direction);
"""

# Whitelist - never interpolate a user-supplied column name directly into SQL.
SORTABLE_COLUMNS = {
    "total_pnl", "return_pct", "win_rate_pct", "trades", "fires", "size", "final_equity", "avg_sl_pct", "avg_tp_pct", "id",
}



@contextmanager
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        # Migration check for pre-existing databases created before avg_sl_pct/avg_tp_pct were added
        cursor = conn.execute("PRAGMA table_info(combo_results)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        if "avg_sl_pct" not in existing_cols:
            conn.execute("ALTER TABLE combo_results ADD COLUMN avg_sl_pct REAL NOT NULL DEFAULT 0.0")
        if "avg_tp_pct" not in existing_cols:
            conn.execute("ALTER TABLE combo_results ADD COLUMN avg_tp_pct REAL NOT NULL DEFAULT 0.0")
        conn.commit()



def clear_results():
    """Wipe every row from the previous run - call once at the start of a
    new backtest so the table only ever holds the latest run's results."""
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM combo_results")
        conn.commit()


def insert_results(rows):
    """Append-only incremental insert, called repeatedly during a run as
    each batch of combos finishes simulating."""
    if not rows:
        return
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO combo_results
                (direction, combo, conditions, size, fires, trades, win_rate_pct, avg_sl_pct, avg_tp_pct, final_equity, total_pnl, return_pct)
            VALUES (:direction, :combo, :conditions, :size, :fires, :trades, :win_rate_pct, :avg_sl_pct, :avg_tp_pct, :final_equity, :total_pnl, :return_pct)
            """,
            [
                {
                    "direction": r["direction"],
                    "combo": r["combo"],
                    "conditions": json.dumps(r["conditions"]),
                    "size": r["size"],
                    "fires": r["fires"],
                    "trades": r["trades"],
                    "win_rate_pct": r["win_rate_pct"],
                    "avg_sl_pct": r.get("avg_sl_pct", 0.0),
                    "avg_tp_pct": r.get("avg_tp_pct", 0.0),
                    "final_equity": r["final_equity"],
                    "total_pnl": r["total_pnl"],
                    "return_pct": r["return_pct"],
                }
                for r in rows
            ],
        )

        conn.commit()


def fetch_results(direction=None, min_size=None, min_trades=None, sort_by="total_pnl", sort_dir="desc", limit=50, offset=0):
    """Filtered/sorted/paginated read of past runs' results."""
    init_db()
    sort_by = sort_by if sort_by in SORTABLE_COLUMNS else "total_pnl"
    sort_dir = "ASC" if str(sort_dir).lower() == "asc" else "DESC"


    where, params = [], {}
    if direction in ("Long", "Short"):
        where.append("direction = :direction")
        params["direction"] = direction
    if min_size:
        where.append("size >= :min_size")
        params["min_size"] = min_size
    if min_trades:
        where.append("trades >= :min_trades")
        params["min_trades"] = min_trades
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM combo_results
            {where_sql}
            ORDER BY {sort_by} {sort_dir}
            LIMIT :limit OFFSET :offset
            """,
            {**params, "limit": limit, "offset": offset},
        ).fetchall()
        total = conn.execute(f"SELECT COUNT(*) AS n FROM combo_results {where_sql}", params).fetchone()["n"]

    results = []
    for row in rows:
        d = dict(row)
        d["conditions"] = json.loads(d["conditions"])
        d["avg_sl_pct"] = d.get("avg_sl_pct", 0.0) or 0.0
        d["avg_tp_pct"] = d.get("avg_tp_pct", 0.0) or 0.0
        results.append(d)
    return results, total



def get_top_combos_for_chart(limit=15):
    """Fetch top N combos by total_pnl for visual UI chart rendering."""
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT combo, direction, total_pnl, win_rate_pct, return_pct, trades
            FROM combo_results
            ORDER BY total_pnl DESC
            LIMIT :limit
            """,
            {"limit": limit},
        ).fetchall()
    return [dict(r) for r in rows]


def summary_stats():
    """Overview numbers for the top of the viewer page including Long and Short splits."""
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_combos,
                SUM(CASE WHEN direction = 'Long' THEN 1 ELSE 0 END) AS long_combos,
                SUM(CASE WHEN direction = 'Short' THEN 1 ELSE 0 END) AS short_combos,
                MAX(total_pnl) AS best_pnl,
                AVG(win_rate_pct) AS avg_win_rate,
                MAX(size) AS max_size
            FROM combo_results
            """
        ).fetchone()
        if not row or not row["total_combos"]:
            return None

        best_overall = conn.execute("SELECT * FROM combo_results ORDER BY total_pnl DESC LIMIT 1").fetchone()
        best_long = conn.execute("SELECT * FROM combo_results WHERE direction = 'Long' ORDER BY total_pnl DESC LIMIT 1").fetchone()
        best_short = conn.execute("SELECT * FROM combo_results WHERE direction = 'Short' ORDER BY total_pnl DESC LIMIT 1").fetchone()

        stats = dict(row)
        stats["best_combo"] = dict(best_overall) if best_overall else None
        stats["best_long_combo"] = dict(best_long) if best_long else None
        stats["best_short_combo"] = dict(best_short) if best_short else None
        return stats

