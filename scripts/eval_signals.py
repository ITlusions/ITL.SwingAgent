import argparse, sqlite3
from pathlib import Path
import pandas as pd
import yfinance as yf

from swing_agent.backtester import simulate_trade
from swing_agent.data import VALID_INTERVALS
from swing_agent.vectorstore import update_vector_payload

def fetch_ohlcv(symbol: str, start_iso: str, interval: str, days_ahead: int = 5) -> pd.DataFrame:
    start = pd.Timestamp(start_iso).tz_convert("UTC")
    end = start + pd.Timedelta(days=days_ahead)
    df = yf.download(tickers=symbol, interval=VALID_INTERVALS.get(interval, "30m"),
                     start=start.tz_convert(None).to_pydatetime(), end=end.tz_convert(None).to_pydatetime(),
                     auto_adjust=False, progress=False, prepost=False, threads=True)
    if df is None or df.empty: return pd.DataFrame()
    df = df.rename(columns=str.lower); df.index = pd.to_datetime(df.index, utc=True)
    return df[~df.index.duplicated(keep="last")]

def bars_per_day(interval: str) -> int:
    return {"15m": int(6.5*4), "30m": int(6.5*2), "1h": int(6.5)}.get(interval, 13)

def main():
    ap = argparse.ArgumentParser(description="Evaluate stored signals and update vector payload with hold times.")
    ap.add_argument("--db", default="data/signals.sqlite")
    ap.add_argument("--max-hold-days", type=float, default=2.0)
    args = ap.parse_args()
    db = Path(args.db); db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db) as con:
        rows = con.execute("""
            SELECT id, symbol, timeframe, asof, side, entry_price, stop_price, take_profit
            FROM signals WHERE evaluated = 0 AND side IS NOT NULL AND entry_price IS NOT NULL
        """).fetchall()

    if not rows:
        print("No pending signals to evaluate."); return

    for (sid, symbol, tf, asof, side, entry, stop, tp) in rows:
        df = fetch_ohlcv(symbol, start_iso=asof, interval=tf, days_ahead=5)
        if df.empty:
            with sqlite3.connect(db) as con:
                con.execute("UPDATE signals SET evaluated=1, exit_reason='NODATA' WHERE id=?", (sid,))
            continue
        try:
            open_idx = df.index.get_indexer([pd.Timestamp(asof).tz_convert("UTC")], method="nearest")[0]
        except Exception:
            open_idx = 0
        max_hold_bars = int(bars_per_day(tf) * args.max_hold_days)
        from swing_agent.models import SignalSide
        exit_idx, reason, exit_price = simulate_trade(df, open_idx, SignalSide(side), float(entry), float(stop), float(tp), max_hold_bars)
        risk = abs(float(entry) - float(stop)) or 1e-9
        r_mult = (exit_price - float(entry))/risk if side=="long" else (float(entry) - exit_price)/risk
        exit_time = df.index[min(exit_idx, len(df)-1)].isoformat()

        from swing_agent.storage import mark_evaluation
        mark_evaluation(sid, db_path=db, exit_reason=reason, exit_price=float(exit_price), exit_time_utc=exit_time, realized_r=float(r_mult))

        try:
            hold_minutes = max(0, (pd.Timestamp(exit_time) - pd.Timestamp(asof)).total_seconds() / 60.0)
            bph = {"15m": 4, "30m": 2, "1h": 1, "1d": 1/6.5}.get(tf, 2)
            hold_bars = int(round((hold_minutes / 60.0) * bph))
            update_vector_payload(db_path=str(db).replace("signals.sqlite", "vec_store.sqlite"), vid=f"{symbol}-{asof}", merge={"hold_minutes": hold_minutes, "hold_bars": hold_bars, "exit_reason": reason})
        except Exception:
            pass

        print(f"[{symbol} {asof}] -> {reason} @ {exit_price:.2f} (R={r_mult:.2f})")

if __name__ == "__main__":
    main()
