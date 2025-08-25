import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from swing_agent.calibration import calibrated_winrate, load_calibrator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/swing_agent.sqlite")
    args = ap.parse_args()
    db = Path(args.db)
    with sqlite3.connect(db) as con:
        df = pd.read_sql_query(
            """
            SELECT symbol, timeframe, asof, side, expected_r, expected_winrate,
                   expected_hold_bars, realized_r, exit_reason, mtf_alignment,
                   rs_sector_20, vol_regime
            FROM signals WHERE evaluated = 1
            """,
            con,
            parse_dates=["asof"],
        )
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
    df = df.dropna(subset=["expected_winrate", "realized_r"])
    if df.empty:
        return

    y_true = (df["realized_r"] > 0).astype(int)
    y_pred = df["expected_winrate"].clip(0, 1)
    brier_raw = brier_score_loss(y_true, y_pred)

    # Fit calibrator and apply if available
    model = load_calibrator(db)
    if model is not None:
        df["calibrated"] = df["expected_winrate"].apply(lambda p: calibrated_winrate(p, db))
        brier_cal = brier_score_loss(y_true, df["calibrated"])
        print(f"\nBrier score (raw/calibrated): {brier_raw:.4f}/{brier_cal:.4f}")
        prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_pred, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, df["calibrated"], n_bins=10)
        plt.plot(prob_pred_raw, prob_true_raw, marker="o", label="raw")
        plt.plot(prob_pred_cal, prob_true_cal, marker="o", label="calibrated")
    else:
        print(f"\nBrier score: {brier_raw:.4f}")
        prob_true_raw, prob_pred_raw = calibration_curve(y_true, y_pred, n_bins=10)
        plt.plot(prob_pred_raw, prob_true_raw, marker="o", label="raw")

    plt.plot([0, 1], [0, 1], "k--", label="ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical probability")
    plt.title("Calibration curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
