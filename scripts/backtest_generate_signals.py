#!/usr/bin/env python
import argparse, pandas as pd, yfinance as yf
from swing_agent.data import VALID_INTERVALS
from swing_agent.agent import SwingAgent
from swing_agent.storage import record_signal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", default="30m", choices=["15m","30m","1h","1d"])
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--warmup-bars", type=int, default=80)
    ap.add_argument("--db", default="data/signals.sqlite")
    ap.add_argument("--vec-db", default="data/vec_store.sqlite")
    ap.add_argument("--sector", default="XLK")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    df = yf.download(args.symbol, interval=VALID_INTERVALS[args.interval],
                     period=f"{args.lookback_days}d", auto_adjust=False,
                     progress=False, prepost=False, threads=True)
    if df is None or df.empty:
        raise SystemExit("No data")
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")]

    agent = SwingAgent(interval=args.interval, lookback_days=args.lookback_days,
                       log_db=None, vec_db=args.vec_db,
                       use_llm=not args.no_llm, llm_extras=False, sector_symbol=args.sector)

    start = max(args.warmup_bars, 60)
    for i in range(start, len(df)-1):
        sub = df.iloc[:i+1].copy()
        try:
            sig = agent.analyze_df(args.symbol, sub)
        except Exception:
            continue
        sig.asof = sub.index[-1].isoformat()
        record_signal(sig, args.db)
        print(f"Saved signal @ {sig.asof} | side={sig.entry.side.value if sig.entry else 'none'}")

if __name__ == "__main__":
    main()
