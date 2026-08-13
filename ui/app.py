"""Streamlit dashboard for the strategy search + single-strategy backtesting.

Usage:
    streamlit run ui/app.py

Reads data/report.json (written by `python3 -m strategy_finder.main`) and, on
demand in the "Backtest a Strategy" tab, rebuilds the dataset once (cached)
to run one selected strategy standalone via backtesting.strategy_runner.
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backtesting.strategy_runner import run_strategy  # noqa: E402
from strategy_finder.main import (  # noqa: E402
    BACKTEST_FEE_PCT,
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_MAX_HOLD_BARS,
    BACKTEST_RISK_PER_TRADE_PCT,
    BACKTEST_STOP_LOSS_PCT,
    BACKTEST_TAKE_PROFIT_PCT,
    build_dataset,
)

REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "report.json")
TOP_N_TABLE = 500
TOP_N_BACKTEST_CHOICES = 200

st.set_page_config(page_title="Master Backtester", layout="wide")


@st.cache_data(show_spinner="Loading data/report.json...")
def load_report(_mtime):
    with open(REPORT_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner="Building dataset (indicators + price action)...")
def get_dataset():
    return build_dataset()


def combo_text(dictionary, indices):
    return " AND ".join(dictionary[i] for i in indices)


def truncate(text, n=70):
    return text if len(text) <= n else text[: n - 1] + "…"


def render_overview(report):
    dataset = report.get("dataset") or {}
    config = report.get("config") or {}
    cb_config = (report.get("combo_backtest") or {}).get("config") or {}

    st.subheader("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol / Interval", f"{dataset.get('symbol')} / {dataset.get('interval')}")
    c2.metric("Rows", f"{dataset.get('rows', 0):,}")
    c3.metric("Columns", dataset.get("columns"))
    c4.metric("Missing values", dataset.get("missing_values"))
    st.caption(f"Date range: {str(dataset.get('date_start'))[:10]} → {str(dataset.get('date_end'))[:10]}")

    groups = dataset.get("column_groups") or {}
    st.caption(
        f"{len(groups.get('ohlcv', []))} OHLCV, {len(groups.get('indicator', []))} indicator, "
        f"{len(groups.get('price_action', []))} price-action columns."
    )

    st.subheader("Backtest config")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stop / Target", f"{config.get('backtest_stop_loss_pct')}% / {config.get('backtest_take_profit_pct')}%")
    c2.metric("Risk / trade", f"{config.get('backtest_risk_per_trade_pct')}%")
    c3.metric("Combo sizes", f"{cb_config.get('min_combo_size')} - {cb_config.get('max_combo_size')}")
    c4.metric("Min fires", cb_config.get("min_fires"))

    st.subheader("Search funnel")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Long pool", cb_config.get("long_pool_size"))
    c2.metric("Short pool", cb_config.get("short_pool_size"))
    c3.metric("Combos tested", f"{cb_config.get('combos_tested', 0):,}")
    c4.metric("Cleared min fires", f"{cb_config.get('combos_cleared_min_fires', 0):,}")
    c5.metric("Profitable found", f"{cb_config.get('profitable_combos_found', 0):,}")


def render_combos(combos, dictionary):
    st.subheader("Filter")
    c1, c2, c3, c4 = st.columns(4)
    direction = c1.selectbox("Direction", ["All", "Long", "Short"])
    min_size = c2.number_input("Min size", min_value=0, value=0, step=1)
    min_trades = c3.number_input("Min trades", min_value=0, value=0, step=10)
    min_winrate = c4.number_input("Min win rate %", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    filtered = combos
    if direction != "All":
        filtered = [c for c in filtered if c["direction"] == direction]
    if min_size:
        filtered = [c for c in filtered if c["size"] >= min_size]
    if min_trades:
        filtered = [c for c in filtered if c["trades"] >= min_trades]
    if min_winrate:
        filtered = [c for c in filtered if c["win_rate_pct"] >= min_winrate]

    st.caption(f"{len(filtered):,} of {len(combos):,} combos match")
    if not filtered:
        st.info("No combos match these filters.")
        return

    shown = sorted(filtered, key=lambda c: c["total_pnl"], reverse=True)[:TOP_N_TABLE]
    if len(filtered) > TOP_N_TABLE:
        st.caption(f"Showing top {TOP_N_TABLE} of {len(filtered):,} matching, ranked by total PnL.")

    df = pd.DataFrame([
        {
            "direction": c["direction"],
            "combo": combo_text(dictionary, c["conditions"]),
            "size": c["size"], "fires": c["fires"], "trades": c["trades"],
            "win_rate_pct": c["win_rate_pct"], "final_equity": c["final_equity"],
            "total_pnl": c["total_pnl"], "return_pct": c["return_pct"],
        }
        for c in shown
    ])

    st.subheader("Top 10 by total PnL")
    top10 = df.head(10).copy()
    top10["label"] = top10["combo"].map(truncate)
    st.bar_chart(top10.set_index("label")["total_pnl"])

    st.subheader("Win rate vs return")
    st.scatter_chart(df, x="win_rate_pct", y="return_pct", color="direction")

    st.subheader(f"Combinations ({len(shown)} shown)")
    st.dataframe(df, width="stretch", height=420)


def render_indicators(cb):
    contribution = cb.get("indicator_contribution") or {"rows": [], "overall_mean_pnl": 0, "total_combos": 0}
    redundancy = cb.get("indicator_redundancy") or []
    perfect_predictors = cb.get("perfect_predictors") or []
    direction_target = cb.get("direction_target_diagnostic") or []

    st.subheader("Indicator contribution")
    st.caption(
        "How often each base indicator/signal appears across every profitable combo, and its "
        '"lift" - how much a combo\'s total PnL differs from the overall average when present. '
        "High appearance + positive lift = a real driver. High appearance + negative lift = "
        "along for the ride, not actually helping."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Indicators/signals tracked", len(contribution["rows"]))
    c2.metric("Profitable combos analyzed", f"{contribution.get('total_combos', 0):,}")
    c3.metric("Overall mean PnL / combo", f"${contribution.get('overall_mean_pnl', 0):.2f}")

    contrib_df = pd.DataFrame(contribution["rows"])
    if not contrib_df.empty:
        by_lift = contrib_df.sort_values("lift_vs_overall_mean", ascending=False)
        top_bottom = pd.concat([by_lift.head(15), by_lift.tail(15)]).drop_duplicates(subset="name")
        st.subheader("Top / bottom 15 by lift")
        st.bar_chart(top_bottom.set_index("name")["lift_vs_overall_mean"])

        st.subheader("Contribution table")
        st.dataframe(contrib_df.sort_values("appears_in", ascending=False), width="stretch", height=350)

    st.subheader("Possibly redundant indicator pairs (|r| ≥ 0.9)")
    st.caption("Numeric indicator pairs this correlated (e.g. RSI_7 vs RSI_14) carry less independent signal than the raw column count suggests.")
    if redundancy:
        st.dataframe(pd.DataFrame(redundancy), width="stretch", height=250)
    else:
        st.caption("None found.")

    st.subheader("Perfect-predictor scan (100% win rate)")
    st.caption("Ranked by a Wilson-score confidence lower bound, not the raw 100% - a 9-trade fluke shouldn't read the same as an 80-trade result.")
    if perfect_predictors:
        st.dataframe(pd.DataFrame(perfect_predictors), width="stretch", height=250)
    else:
        st.caption(
            "No combo achieved a literal 100% win rate in this run - expected/healthy given "
            "realistic fees and finite sample sizes, not a bug."
        )

    st.subheader("Direction-right-but-target-missed diagnostic")
    st.caption(
        "gap_pct = directional_accuracy_pct − target_hit_rate_pct: the condition correctly predicts "
        "direction far more often than our FIXED take-profit actually gets touched. suggested_tp_pct/"
        "suggested_sl_pct are the condition's own median favorable/adverse move size - a data-driven "
        "bracket instead of the fixed one. Sorted by gap% to surface where the fixed bracket is failing worst."
    )
    if direction_target:
        dt_df = pd.DataFrame(direction_target).sort_values("gap_pct", ascending=False)
        st.dataframe(dt_df, width="stretch", height=420)


def render_backtest(combos, dictionary):
    st.subheader("Backtest a strategy")
    st.caption("Pick one strategy the search found and run it standalone to see its full trade book.")

    if not combos:
        st.info("No combos available.")
        return

    top_combos = sorted(combos, key=lambda c: c["total_pnl"], reverse=True)[:TOP_N_BACKTEST_CHOICES]
    labels = [
        f"[{c['direction']}] {truncate(combo_text(dictionary, c['conditions']), 90)}  "
        f"(PnL {c['total_pnl']:+.2f}, {c['trades']} trades, {c['win_rate_pct']:.1f}% win)"
        for c in top_combos
    ]
    st.caption(f"Choosing from the top {len(top_combos)} combos by total PnL.")
    choice = st.selectbox("Strategy", options=range(len(top_combos)), format_func=lambda i: labels[i])

    if st.button("Run backtest", type="primary"):
        chosen = top_combos[choice]
        conditions = [dictionary[i] for i in chosen["conditions"]]
        with st.spinner("Building dataset + simulating..."):
            df = get_dataset()
            result = run_strategy(
                chosen["direction"], conditions, df,
                initial_capital=BACKTEST_INITIAL_CAPITAL, risk_per_trade_pct=BACKTEST_RISK_PER_TRADE_PCT,
                stop_loss_pct=BACKTEST_STOP_LOSS_PCT, take_profit_pct=BACKTEST_TAKE_PROFIT_PCT,
                max_hold_bars=BACKTEST_MAX_HOLD_BARS, fee_pct=BACKTEST_FEE_PCT,
            )
        s = result["summary"]
        st.markdown(f"**[{s['direction']}]** {s['combo']}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final equity", f"${s['final_equity']:,.2f}")
        c2.metric("Total PnL", f"{s['total_pnl']:+.2f}", f"{s['return_pct']:+.1f}%")
        c3.metric("Trades", s["trades"])
        c4.metric("Win rate", f"{s['win_rate_pct']:.1f}%")
        c5.metric("Max drawdown", f"{s['max_drawdown_pct']:.1f}%")

        st.subheader("Equity curve")
        st.line_chart(pd.Series(result["equity_curve"], name="equity"))

        st.subheader(f"Trade book ({len(result['trades'])} trades)")
        st.dataframe(pd.DataFrame(result["trades"]), width="stretch", height=420)


def main():
    st.title("Master Backtester")

    if not os.path.exists(REPORT_PATH):
        st.error(f"`{REPORT_PATH}` not found - run `python3 -m strategy_finder.main` first.")
        return

    report = load_report(os.path.getmtime(REPORT_PATH))
    cb = report.get("combo_backtest") or {}
    dictionary = cb.get("condition_dictionary", [])
    combos = cb.get("combinations", [])

    tab_overview, tab_combos, tab_indicators, tab_backtest = st.tabs(
        ["Overview", "Combo Search Results", "Indicators", "Backtest a Strategy"]
    )
    with tab_overview:
        render_overview(report)
    with tab_combos:
        render_combos(combos, dictionary)
    with tab_indicators:
        render_indicators(cb)
    with tab_backtest:
        render_backtest(combos, dictionary)


main()
