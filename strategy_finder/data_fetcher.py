"""Fetch and cache OHLCV candle data from the Delta Exchange public API."""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


class DataFetcher:
    """Downloads OHLCV candles from Delta Exchange and caches them locally as CSV.

    On repeated calls with the same symbol/interval/total_days, data is loaded
    from the local cache instead of hitting the API again.
    """

    API_URL = "https://api.india.delta.exchange/v2/history/candles"
    CHUNK_FREQ = "30D"
    TIMEZONE = "Asia/Kolkata"

    def __init__(self, symbol="ADAUSD", interval="15m", total_days=100, cache_dir="data"):
        self.symbol = symbol
        self.interval = interval
        self.total_days = total_days
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def cache_path(self):
        filename = f"{self.symbol}_{self.interval}_{self.total_days}d.csv"
        return os.path.join(self.cache_dir, filename)

    def fetch(self, force_refresh=False):
        """Return the OHLCV DataFrame, using the local cache unless force_refresh is set."""
        if not force_refresh and os.path.exists(self.cache_path):
            print(f"[DataFetcher] Loading cached data: {self.cache_path}")
            return pd.read_csv(self.cache_path, index_col=0, parse_dates=True)

        print(f"[DataFetcher] Fetching {self.symbol} ({self.interval}) for last {self.total_days} days...")
        df = self._fetch_from_api()

        if df.empty:
            print("[DataFetcher] Warning: no data returned from API.")
            return df

        df.to_csv(self.cache_path)
        print(f"[DataFetcher] Saved {len(df)} rows to {self.cache_path}")
        return df

    def _fetch_from_api(self):
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=self.total_days)
        date_ranges = pd.date_range(start=start_date, end=end_date, freq=self.CHUNK_FREQ)

        chunks = []
        for i in range(len(date_ranges)):
            chunk_start = date_ranges[i]
            chunk_end = date_ranges[i + 1] if i + 1 < len(date_ranges) else end_date
            chunk_df = self._fetch_chunk(chunk_start, chunk_end)
            if chunk_df is not None:
                chunks.append(chunk_df)

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        return df

    def _fetch_chunk(self, chunk_start, chunk_end, retries=3):
        params = {
            "resolution": self.interval,
            "symbol": self.symbol,
            "start": str(int(chunk_start.timestamp())),
            "end": str(int(chunk_end.timestamp())),
        }
        print(f"[DataFetcher] Fetching {chunk_start.date()} -> {chunk_end.date()}")

        for attempt in range(1, retries + 1):
            try:
                response = requests.get(
                    self.API_URL, params=params, headers={"Accept": "application/json"}, timeout=10
                )
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("success") and payload.get("result"):
                        return self._to_dataframe(payload["result"])
                    return None
            except requests.RequestException as exc:
                print(f"[DataFetcher] Attempt {attempt}/{retries} failed: {exc}")
            time.sleep(1)
        return None

    def _to_dataframe(self, candles):
        rows = [
            {
                "time": c["time"],
                "Open": float(c["open"]),
                "High": float(c["high"]),
                "Low": float(c["low"]),
                "Close": float(c["close"]),
                "Volume": float(c["volume"] or 0),
            }
            for c in candles
        ]
        df = pd.DataFrame(rows)
        df["DateTime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(self.TIMEZONE)
        df = df.drop(columns=["time"]).set_index("DateTime")
        return df
