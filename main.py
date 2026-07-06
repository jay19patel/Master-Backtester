"""Entry point: fetch data, add oracle labels, and report stats."""

import pandas as pd

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

# For now we only want to look at the oracle data itself, so the indicator engine
# (and the relevance analysis that depends on it) is switched off. Flip this back
# to True once indicators are needed again.
INCLUDE_INDICATORS = False

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
ORACLE_LABEL_NAMES = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}
DIRECTION_SIGNAL_HEATMAP_PATH = "direction_signal_heatmap.jpg"


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


def direction_signal_crosstab(df):
    """Cross-tab of oracle_direction vs oracle_signal: counts and row-percentages.

    Returns None if either column is missing, else (counts, row_pct, row_order, col_order).
    """
    if "oracle_direction" not in df.columns or "oracle_signal" not in df.columns:
        return None

    known = df[df["oracle_direction"].notna() & df["oracle_signal"].notna()].copy()
    known["direction_name"] = known["oracle_direction"].map(ORACLE_LABEL_NAMES)

    row_order = [name for name in ["UP", "DOWN", "NEUTRAL"] if name in known["direction_name"].unique()]
    col_order = [s for s in ["BUY", "HOLD", "SELL"] if s in known["oracle_signal"].unique()]

    counts = pd.crosstab(known["direction_name"], known["oracle_signal"]).reindex(
        index=row_order, columns=col_order, fill_value=0
    )
    row_pct = counts.div(counts.sum(axis=1), axis=0) * 100
    return counts, row_pct, row_order, col_order


def print_direction_signal_relationship(df):
    """Show how oracle_direction (what actually happened) lines up with oracle_signal
    (the reward:risk based call), so you can see which direction tends to produce
    which kind of signal."""
    crosstab = direction_signal_crosstab(df)
    if crosstab is None:
        return
    counts, row_pct, row_order, col_order = crosstab

    print("\n--- Direction vs Signal relationship ---")
    print("(of the candles where the oracle direction was X, what % got each signal)")

    print("\nCounts:")
    print(counts)

    print("\nRow % (within each direction):")
    print(row_pct.round(1))

    print("\nIn words:")
    for direction_name in row_order:
        total = counts.loc[direction_name].sum()
        parts = ", ".join(f"{col} {row_pct.loc[direction_name, col]:.1f}%" for col in col_order)
        print(f"  When direction was {direction_name:<7} ({total:>5} candles) -> {parts}")


def save_direction_signal_heatmap(df):
    """Render the direction vs signal row-% table as a JPG heatmap."""
    crosstab = direction_signal_crosstab(df)
    if crosstab is None:
        return None
    _, row_pct, _, _ = crosstab

    return HeatmapVisualizer(DIRECTION_SIGNAL_HEATMAP_PATH).plot_percentage_heatmap(
        row_pct,
        title="Oracle Direction vs Signal (row %)",
        x_label="Signal",
        y_label="Direction",
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

    if "oracle_direction" in df.columns:
        print(f"\n--- Oracle direction (lookahead={ORACLE_LOOKAHEAD} candles, source of truth) ---")
        counts = df["oracle_direction"].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            name = ORACLE_LABEL_NAMES.get(value, "UNKNOWN (tail, no future data yet)")
            pct = count / len(df) * 100
            print(f"  {name:<28}: {count:>6}  ({pct:5.1f}%)")

    if "oracle_signal" in df.columns:
        print(f"\n--- Oracle signal (min reward:risk = 1:{ORACLE_MIN_REWARD_RISK_RATIO:.0f}) ---")
        counts = df["oracle_signal"].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            name = value if pd.notna(value) else "UNKNOWN (tail, no future data yet)"
            pct = count / len(df) * 100
            print(f"  {name:<28}: {count:>6}  ({pct:5.1f}%)")

    print_direction_signal_relationship(df)

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
        for c in [
            "Close",
            "oracle_range",
            "oracle_upside_gap",
            "oracle_downside_gap",
            "oracle_actual_move_pct",
            "oracle_direction",
            "oracle_signal",
        ]
        if c in df.columns
    ]
    if "oracle_direction" in df.columns:
        known_rows = df[df["oracle_direction"].notna()]
        print(f"\n--- Last 5 candles with a known oracle outcome (out of {len(known_rows)}) ---")
        print(known_rows[preview_cols].tail(5))
    print("=" * 70 + "\n")


def main():
    df = build_dataset()
    print_report(df)
    save_direction_signal_heatmap(df)
    if INCLUDE_INDICATORS:
        RelevanceAnalyzer(df).print_report(top_n=15)


if __name__ == "__main__":
    main()
