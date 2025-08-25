import argparse, sqlite3, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/swing_agent.sqlite")
    args = ap.parse_args()
    db = Path(args.db)
    with sqlite3.connect(db) as con:
        df = pd.read_sql_query("""
            SELECT symbol, timeframe, asof, side, expected_r, expected_winrate,
                   expected_hold_bars, realized_r, exit_reason, mtf_alignment, rs_sector_20, vol_regime
            FROM signals WHERE evaluated = 1
        """, con, parse_dates=["asof"])
    if df.empty:
        print("No evaluated signals yet."); return
    g = df.groupby(["vol_regime"]).agg(
        n=("symbol","count"),
        mean_exp_r=("expected_r","mean"),
        mean_realized_r=("realized_r","mean"),
        winrate=("realized_r", lambda s: (s>0).mean())
    ).reset_index()
    print("=== Performance by vol_regime ===")
    print(g.to_string(index=False))
    df = df.dropna(subset=["expected_winrate"])
    if not df.empty:
        df["bin"] = (df["expected_winrate"]*10).clip(0,9).astype(int)/10.0
        cal = df.groupby("bin").agg(n=("symbol","count"), pred=("expected_winrate","mean"), emp=("realized_r", lambda s: (s>0).mean())).reset_index().sort_values("bin")
        print("\n=== Calibration (winrate) ===")
        print(cal.to_string(index=False))

if __name__ == "__main__":
    main()
