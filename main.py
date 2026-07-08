"""Entry point: fetch data, engineer indicators, add oracle labels, and report stats."""

import pandas as pd

from condition_finder import ConditionFinder
from data_fetcher import DataFetcher
from indicator_engine import IndicatorEngine
from oracle_labeler import OracleLabeler
from relevance_analyzer import RelevanceAnalyzer
from visualizer import HeatmapVisualizer

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

SYMBOL = "ETHUSD"
INTERVAL = "15m"
TOTAL_DAYS = 100
ORACLE_LOOKAHEAD = 20
ORACLE_MIN_REWARD_RISK_RATIO = 2.0  # 1:2 minimum - the winning side must be at least double the losing side

# Indicators are needed now (the EMA vs signal heatmap depends on EMA_20). The
# relevance analysis is a separate, heavier report - keep it off unless asked for.
INCLUDE_INDICATORS = True
RUN_RELEVANCE_ANALYSIS = True
RUN_CONDITION_SEARCH = True

CONDITION_MIN_SUPPORT = 100
CONDITION_MAX_COMBO_SIZE = 3

EMA_COLUMN = "EMA_20"
EMA_SIGNAL_HEATMAP_PATH = "signal_ema_heatmap.jpg"

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def build_dataset():
    fetcher = DataFetcher(symbol=SYMBOL, interval=INTERVAL, total_days=TOTAL_DAYS)
    df = fetcher.fetch()

    if df.empty:
        raise RuntimeError("No data fetched - aborting.")

    if INCLUDE_INDICATORS:
        df = IndicatorEngine(df).build()

    df = OracleLabeler(
        df, lookahead=ORACLE_LOOKAHEAD, min_reward_risk_ratio=ORACLE_MIN_REWARD_RISK_RATIO
    ).label()
    return df


def column_groups(df):
    """Split columns into OHLCV / oracle / indicator buckets for reporting."""
    oracle_cols = [c for c in df.columns if c.startswith("oracle_")]
    ohlcv_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
    indicator_cols = [c for c in df.columns if c not in oracle_cols and c not in ohlcv_cols]
    return ohlcv_cols, indicator_cols, oracle_cols


def ema_signal_crosstab(df, ema_col=EMA_COLUMN):
    """Cross-tab of "price vs EMA" trend state against oracle_signal.

    Returns None if either column is missing, else (counts, row_pct, row_order, col_order).
    """
    if ema_col not in df.columns or "oracle_signal" not in df.columns:
        return None

    known = df[df["oracle_signal"].notna() & df[ema_col].notna()].copy()

    above_label = f"Above {ema_col}"
    below_label = f"Below {ema_col}"
    known["ema_state"] = (known["Close"] >= known[ema_col]).map({True: above_label, False: below_label})

    row_order = [name for name in [above_label, below_label] if name in known["ema_state"].unique()]
    col_order = [s for s in ["BUY", "HOLD", "SELL"] if s in known["oracle_signal"].unique()]

    counts = pd.crosstab(known["ema_state"], known["oracle_signal"]).reindex(
        index=row_order, columns=col_order, fill_value=0
    )
    row_pct = counts.div(counts.sum(axis=1), axis=0) * 100
    return counts, row_pct, row_order, col_order


def print_ema_signal_relationship(df):
    """Show how price-vs-EMA trend state lines up with oracle_signal, so you can see
    which trend state tends to produce which kind of signal."""
    crosstab = ema_signal_crosstab(df)
    if crosstab is None:
        return
    counts, row_pct, row_order, col_order = crosstab

    print(f"\n--- {EMA_COLUMN} trend vs Signal relationship ---")
    print(f"(of the candles where price was above/below {EMA_COLUMN}, what % got each signal)")

    print("\nCounts:")
    print(counts)

    print("\nRow % (within each trend state):")
    print(row_pct.round(1))

    print("\nIn words:")
    for state_name in row_order:
        total = counts.loc[state_name].sum()
        parts = ", ".join(f"{col} {row_pct.loc[state_name, col]:.1f}%" for col in col_order)
        print(f"  When price was {state_name:<16} ({total:>5} candles) -> {parts}")


def save_ema_signal_heatmap(df):
    """Render the EMA-trend vs signal row-% table as a JPG heatmap."""
    crosstab = ema_signal_crosstab(df)
    if crosstab is None:
        return None
    _, row_pct, _, _ = crosstab

    return HeatmapVisualizer(EMA_SIGNAL_HEATMAP_PATH).plot_percentage_heatmap(
        row_pct,
        title=f"Signal vs {EMA_COLUMN} trend (row %)",
        x_label="Signal",
        y_label="Price vs EMA",
    )


def print_signal_meaning(df):
    """Explain in plain language + real averages what a BUY/HOLD/SELL signal means."""
    if "oracle_signal" not in df.columns:
        return

    known = df[df["oracle_signal"].notna()]
    stats_cols = ["oracle_upside_pct", "oracle_downside_pct", "oracle_upside_gap", "oracle_downside_gap"]
    avg = known.groupby("oracle_signal")[stats_cols].mean()
    counts = known["oracle_signal"].value_counts()

    print(f"\n--- What each signal means (avg over the next {ORACLE_LOOKAHEAD} candles) ---")
    for signal in ["BUY", "HOLD", "SELL"]:
        if signal not in avg.index:
            continue
        row = avg.loc[signal]
        n = int(counts.get(signal, 0))
        print(f"\n{signal} ({n} candles, {n / len(known) * 100:.1f}% of known candles):")
        print(f"  Avg potential upside   : +{row['oracle_upside_pct']:.2f}%  (price gap {row['oracle_upside_gap']:.4f})")
        print(f"  Avg potential downside : {row['oracle_downside_pct']:.2f}%  (price gap {row['oracle_downside_gap']:.4f})")

        if signal == "BUY":
            print(
                f"  Meaning: going forward {ORACLE_LOOKAHEAD} candles, the upside "
                f"(+{row['oracle_upside_pct']:.2f}%) was at least {ORACLE_MIN_REWARD_RISK_RATIO:.0f}x "
                f"bigger than the downside ({row['oracle_downside_pct']:.2f}%) -> a long trade had a "
                f"clear, favorable reward:risk edge."
            )
        elif signal == "SELL":
            print(
                f"  Meaning: going forward {ORACLE_LOOKAHEAD} candles, the downside "
                f"({row['oracle_downside_pct']:.2f}%) was at least {ORACLE_MIN_REWARD_RISK_RATIO:.0f}x "
                f"bigger than the upside (+{row['oracle_upside_pct']:.2f}%) -> a short trade had a "
                f"clear, favorable reward:risk edge."
            )
        else:
            print(
                "  Meaning: neither side cleared the "
                f"{ORACLE_MIN_REWARD_RISK_RATIO:.0f}x bar over the other -> no clean edge, best to stay flat."
            )


def print_report(df):
    print("\n" + "=" * 70)
    print("DATASET REPORT")
    print("=" * 70)
    print(f"Symbol / Interval      : {SYMBOL} / {INTERVAL}")
    print(f"Rows (candles)         : {len(df)}")
    print(f"Columns                : {len(df.columns)}")
    print(f"Date range             : {df.index.min()}  ->  {df.index.max()}")
    print(f"Missing values (total) : {int(df.isna().sum().sum())}")

    if "oracle_signal" in df.columns:
        print(f"\n--- Oracle signal (min reward:risk = 1:{ORACLE_MIN_REWARD_RISK_RATIO:.0f}) ---")
        counts = df["oracle_signal"].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            name = value if pd.notna(value) else "UNKNOWN (tail, no future data yet)"
            pct = count / len(df) * 100
            print(f"  {name:<28}: {count:>6}  ({pct:5.1f}%)")

    print_signal_meaning(df)
    print_ema_signal_relationship(df)

    ohlcv_cols, indicator_cols, oracle_cols = column_groups(df)

    print(f"\n--- OHLCV columns ({len(ohlcv_cols)}) ---")
    for i, col in enumerate(ohlcv_cols, start=1):
        print(f"  {i:>3}. {col}")

    print(f"\n--- Oracle columns ({len(oracle_cols)}) ---")
    for i, col in enumerate(oracle_cols, start=1):
        print(f"  {i:>3}. {col}")

    print(f"\n--- Indicator columns ({len(indicator_cols)}) ---")
    for i, col in enumerate(indicator_cols, start=1):
        print(f"  {i:>3}. {col}")

    preview_cols = [
        c
        for c in ["Close", EMA_COLUMN, "oracle_range", "oracle_upside_gap", "oracle_downside_gap", "oracle_signal"]
        if c in df.columns
    ]
    if "oracle_signal" in df.columns:
        known_rows = df[df["oracle_signal"].notna()]
        print(f"\n--- Last 5 candles with a known oracle outcome (out of {len(known_rows)}) ---")
        print(known_rows[preview_cols].tail(5))
    print("=" * 70 + "\n")


def main():
    df = build_dataset()
    print_report(df)
    save_ema_signal_heatmap(df)
    if RUN_RELEVANCE_ANALYSIS:
        RelevanceAnalyzer(df).print_report(top_n=15)
    if RUN_CONDITION_SEARCH:
        ConditionFinder(
            df, min_support=CONDITION_MIN_SUPPORT, max_combo_size=CONDITION_MAX_COMBO_SIZE
        ).print_report()


if __name__ == "__main__":
    main()
