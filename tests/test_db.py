"""Unit tests for strategy_finder.db module."""

import json
import pytest
from strategy_finder import db


@pytest.fixture(autouse=True)
def setup_tmp_db(tmp_path, monkeypatch):
    """Use an isolated SQLite database file for each test."""
    test_db_path = str(tmp_path / "test_combo_results.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield
    db.clear_results()


def test_init_db_and_clear_results():
    db.clear_results()
    results, total = db.fetch_results()
    assert results == []
    assert total == 0


def test_insert_and_fetch_results():
    sample_rows = [
        {
            "direction": "Long",
            "combo": "RSI_14>median AND EMA_20>median",
            "conditions": ["RSI_14>median", "EMA_20>median"],
            "size": 2,
            "fires": 50,
            "trades": 25,
            "win_rate_pct": 60.0,
            "final_equity": 1200.5,
            "total_pnl": 200.5,
            "return_pct": 20.05,
        },
        {
            "direction": "Short",
            "combo": "MACD<median",
            "conditions": ["MACD<median"],
            "size": 1,
            "fires": 30,
            "trades": 15,
            "win_rate_pct": 40.0,
            "final_equity": 950.0,
            "total_pnl": -50.0,
            "return_pct": -5.0,
        },
    ]

    db.insert_results(sample_rows)
    results, total = db.fetch_results()

    assert total == 2
    assert len(results) == 2
    assert results[0]["combo"] == "RSI_14>median AND EMA_20>median"
    assert results[0]["conditions"] == ["RSI_14>median", "EMA_20>median"]
    assert results[0]["total_pnl"] == 200.5


def test_summary_stats_long_short_split():
    sample_rows = [
        {
            "direction": "Long",
            "combo": "LongCombo1",
            "conditions": ["c1"],
            "size": 1,
            "fires": 20,
            "trades": 10,
            "win_rate_pct": 70.0,
            "final_equity": 1300.0,
            "total_pnl": 300.0,
            "return_pct": 30.0,
        },
        {
            "direction": "Short",
            "combo": "ShortCombo1",
            "conditions": ["c2"],
            "size": 1,
            "fires": 30,
            "trades": 15,
            "win_rate_pct": 50.0,
            "final_equity": 1100.0,
            "total_pnl": 100.0,
            "return_pct": 10.0,
        },
    ]

    db.insert_results(sample_rows)
    stats = db.summary_stats()

    assert stats is not None
    assert stats["total_combos"] == 2
    assert stats["long_combos"] == 1
    assert stats["short_combos"] == 1
    assert stats["best_pnl"] == 300.0
    assert stats["best_long_combo"]["combo"] == "LongCombo1"
    assert stats["best_long_combo"]["total_pnl"] == 300.0
    assert stats["best_short_combo"]["combo"] == "ShortCombo1"
    assert stats["best_short_combo"]["total_pnl"] == 100.0


def test_get_top_combos_for_chart():
    rows = [
        {
            "direction": "Long",
            "combo": f"Combo_{i}",
            "conditions": [f"c_{i}"],
            "size": 1,
            "fires": 20,
            "trades": 10,
            "win_rate_pct": 50.0 + i,
            "final_equity": 1000.0 + i * 10,
            "total_pnl": float(i * 10),
            "return_pct": float(i),
        }
        for i in range(1, 20)
    ]
    db.insert_results(rows)
    chart_data = db.get_top_combos_for_chart(limit=5)

    assert len(chart_data) == 5
    assert chart_data[0]["total_pnl"] == 190.0
    assert chart_data[4]["total_pnl"] == 150.0
