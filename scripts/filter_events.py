"""Filter a watchlist based on upcoming earnings or macro events."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def load_events(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"])


def filter_watchlist(
    watchlist: list[str],
    events: pd.DataFrame,
    now: datetime,
    window_minutes: int = 30,
) -> list[str]:
    window = timedelta(minutes=window_minutes)
    upcoming = events.loc[(events["time"] - now).abs() <= window, "symbol"]
    blocked = set(upcoming)
    if "*" in blocked:  # macro event affecting all symbols
        return []
    return [sym for sym in watchlist if sym not in blocked]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", required=True, help="path to text file with one symbol per line")
    ap.add_argument("--events", required=True, help="CSV with columns symbol,time")
    ap.add_argument("--now", help="ISO timestamp, defaults to current UTC time")
    ap.add_argument("--window", type=int, default=30, help="filter window in minutes")
    args = ap.parse_args()

    with open(args.watchlist) as f:
        watch = [ln.strip() for ln in f if ln.strip()]
    events = load_events(Path(args.events))
    now = datetime.fromisoformat(args.now) if args.now else datetime.utcnow()
    allowed = filter_watchlist(watch, events, now, args.window)
    for sym in allowed:
        print(sym)
