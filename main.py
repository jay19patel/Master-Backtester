"""Entry point: fetch data, engineer indicators, add oracle labels, and report stats."""

from data_fetcher import DataFetcher
from indicator_engine import IndicatorEngine
from oracle_labeler import OracleLabeler
from relevance_analyzer import RelevanceAnalyzer

SYMBOL = "ADAUSD"
INTERVAL = "15m"
TOTAL_DAYS = 100
ORACLE_LOOKAHEAD = 20

ORACLE_LABEL_NAMES = {1: "UP", -1: "DOWN", 0: "NEUTRAL"}


def build_dataset():
    fetcher = DataFetcher(symbol=SYMBOL, interval=INTERVAL, total_days=TOTAL_DAYS)
    df = fetcher.fetch()

    if df.empty:
        raise RuntimeError("No data fetched - aborting.")

    df = IndicatorEngine(df).build()
    df = OracleLabeler(df, lookahead=ORACLE_LOOKAHEAD).label()
    return df


def print_report(df):
    print("\n" + "=" * 70)
    print("DATASET REPORT")
    print("=" * 70)
    print(f"Symbol / Interval      : {SYMBOL} / {INTERVAL}")
    print(f"Rows (candles)         : {len(df)}")
    print(f"Columns                : {len(df.columns)}")
    print(f"Date range             : {df.index.min()}  ->  {df.index.max()}")
    print(f"Missing values (total) : {int(df.isna().sum().sum())}")

    print("\n--- OHLCV summary ---")
    print(df[["Open", "High", "Low", "Close", "Volume"]].describe().T)

    if "oracle_direction" in df.columns:
        print(f"\n--- Oracle labels (lookahead={ORACLE_LOOKAHEAD} candles, source of truth) ---")
        counts = df["oracle_direction"].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            name = ORACLE_LABEL_NAMES.get(value, "UNKNOWN (tail, no future data yet)")
            pct = count / len(df) * 100
            print(f"  {name:<28}: {count:>6}  ({pct:5.1f}%)")

    print(f"\n--- All {len(df.columns)} columns ---")
    for i, col in enumerate(df.columns, start=1):
        print(f"  {i:>3}. {col}")

    preview_cols = [
        c
        for c in ["Close", "oracle_upper_band", "oracle_lower_band", "oracle_actual_move_pct", "oracle_direction"]
        if c in df.columns
    ]
    print("\n--- Last 5 rows (oracle preview) ---")
    print(df[preview_cols].tail(5))
    print("=" * 70 + "\n")


def main():
    df = build_dataset()
    print_report(df)
    RelevanceAnalyzer(df).print_report(top_n=15)


if __name__ == "__main__":
    main()
