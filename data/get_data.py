"""
Pull ~8 years of hourly XAU/USD (gold) from Dukascopy -> data/xauusd_h1.csv.

The paper uses 47,304 hourly XAU/USD observations (~8 years). Dukascopy provides
free, deep, broker-grade history with no API key. The pull is reproducible: fixed
UTC start/end, committed CSV, SHA-256 printed at the end.

    python data/get_data.py                 # default 2017-01-01 .. 2025-01-01
    python data/get_data.py --start 2016-01-01 --end 2025-01-01
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "xauusd_h1.csv"
OHLCV = ["open", "high", "low", "close", "volume"]


def _utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def fetch(start: str, end: str) -> pd.DataFrame:
    import dukascopy_python
    from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

    df = dukascopy_python.fetch(
        INSTRUMENT_FX_METALS_XAU_USD,
        dukascopy_python.INTERVAL_HOUR_1,
        dukascopy_python.OFFER_SIDE_BID,
        _utc(start),
        _utc(end),
    )
    df.columns = [c.lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df[OHLCV]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2025-01-01")
    args = ap.parse_args()

    print(f"Fetching XAU/USD H1 {args.start} -> {args.end} from Dukascopy ...")
    df = fetch(args.start, args.end)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["open", "high", "low", "close"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)

    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"Saved {OUT}")
    print(f"{len(df):,} bars | {df.index[0]} -> {df.index[-1]}")
    print(f"SHA-256: {sha}")


if __name__ == "__main__":
    main()
