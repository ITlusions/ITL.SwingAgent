#!/usr/bin/env python
import argparse, sqlite3, json
from pathlib import Path
import pandas as pd
import yfinance as yf

from swing_agent.features import build_setup_vector, vol_regime_from_series
from swing_agent.vectorstore import add_vector
from swing_agent.data import VALID_INTERVALS

def fetch_slice(symbol: str, asof_iso: str, interval: str):
    asof = pd.Timestamp(asof_iso).tz_convert("UTC")
    span_days = 5 if interval in ("15m","30m","1h") else 60
    start = asof - pd.Timedelta(days=span_days)
    end = asof + pd.Timedelta(days=1)
    df = yf.download(tickers=symbol, interval=VALID_INTERVALS.get(interval, "30m"),
                     start=start.tz_convert(None).to_pydatetime(), end=end.tz_convert(None).to_pydatetime(),
                     auto_adjust=False, progress=False, prepost=False, threads=True)
    if df is None or df.empty: return pd.DataFrame()
    df = df.rename(columns=str.lower); df.index = pd.to_datetime(df.index, utc=True)
    return df[~df.index.duplicated(keep="last")]

def compute_context(df: pd.DataFrame, asof_iso: str):
    asof = pd.Timestamp(asof_iso).tz_convert("UTC")
    if df.empty: return None
    try: idx = df.index.get_indexer([asof], method="nearest")[0]
    except Exception: idx = len(df) - 1
    if idx < 2: return None
    pc = float(df["close"].iloc[idx-1]); phi=float(df["high"].iloc[idx-1]); plo=float(df["low"].iloc[idx-1])
    o = float(df["open"].iloc[idx]); c = float(df["close"].iloc[idx])
    prev_range_pct = (phi - plo)/max(1e-9, pc); gap_pct = (o - pc)/max(1e-9, pc)
    prev_c = df["close"].iloc[: idx + 1].shift(1)
    tr = (pd.concat([(df["high"].iloc[:idx+1]-df["low"].iloc[:idx+1]), (df["high"].iloc[:idx+1]-prev_c).abs(), (df["low"].iloc[:idx+1]-prev_c).abs()], axis=1)).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
    atr_pct = float(atr14/max(1e-9, c))
    return {"price": c, "prev_range_pct": prev_range_pct, "gap_pct": gap_pct, "atr_pct": atr_pct}

def main():
    ap = argparse.ArgumentParser(description="Backfill vec_store.sqlite from signals.sqlite (includes vol_regime payload).")
    ap.add_argument("--signals-db", default="data/swing_agent.sqlite")
    ap.add_argument("--vec-db", default="data/swing_agent.sqlite")
    args = ap.parse_args()
    sigdb = Path(args.signals_db)
    if not sigdb.exists(): print(f"Signals DB not found: {sigdb}"); exit(1)
    with sqlite3.connect(sigdb) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT id, symbol, timeframe, asof, trend_label, ema_slope, price_above_ema, rsi14,
                   side, entry_price, r_multiple, fib_golden_low, fib_golden_high,
                   llm_vote_json, evaluated, exit_reason, exit_price, realized_r, exit_time_utc
            FROM signals ORDER BY asof ASC
        """).fetchall()
    for r in rows:
        symbol, tf, asof = r["symbol"], r["timeframe"], r["asof"]
        df = fetch_slice(symbol, asof, tf); ctx = compute_context(df, asof)
        if not ctx: continue
        trend = type("Trend", (), {"label": r["trend_label"], "ema_slope": float(r["ema_slope"]), "price_above_ema": bool(r["price_above_ema"]), "rsi_14": float(r["rsi14"])})()
        entry = None
        if r["side"] and r["entry_price"]:
            entry = type("Entry", (), {"fib_golden_low": r["fib_golden_low"], "fib_golden_high": r["fib_golden_high"], "r_multiple": r["r_multiple"] or 0.0})()
        llm_conf = 0.0
        try:
            lv = json.loads(r["llm_vote_json"]) if r["llm_vote_json"] else None
            if lv and "confidence" in lv: llm_conf = float(lv["confidence"])
        except Exception: pass
        price = float(ctx["price"])
        vec = build_setup_vector(price=price, trend=trend, entry=entry, prev_range_pct=float(ctx["prev_range_pct"]), gap_pct=float(ctx["gap_pct"]), atr_pct=float(ctx["atr_pct"]), session_bin=1, llm_conf=llm_conf)
        vol_reg = vol_regime_from_series(df["close"])
        add_vector(args.vec_db, vid=f"{symbol}-{asof}", ts_utc=asof, symbol=symbol, timeframe=tf, vec=vec, realized_r=(float(r["realized_r"]) if r["realized_r"] is not None else None), exit_reason=r["exit_reason"], payload={"vector_version":"v1.6.1","signal_id":r["id"],"hold_minutes":None,"hold_bars":None,"exit_reason":r["exit_reason"],"vol_regime":vol_reg})
    print(f"Backfilled vectors into {args.vec_db}")

if __name__ == "__main__":
    main()
