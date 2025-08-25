"""Simple portfolio-level backtest using generated signals.

This script expects a CSV file with the following columns:

```
date,symbol,entry,stop,exit
```

The backtest is intentionally lightweight; it applies the sizing rules
from :mod:`swing_agent.risk` and accumulates PnL after costs and
slippage.
"""

import argparse
from pathlib import Path

import pandas as pd

from swing_agent.risk import sized_quantity


def run_backtest(
    signals_path: Path,
    equity: float = 100_000.0,
    fee_per_trade: float = 1.0,
    slippage_bp: float = 5.0,
) -> None:
    df = pd.read_csv(signals_path, parse_dates=["date"])
    trades = 0
    pnl = 0.0
    for row in df.itertuples():
        qty = sized_quantity(equity, row.entry, row.stop)
        if qty == 0:
            continue
        trades += 1
        slippage = row.entry * qty * slippage_bp / 10_000
        trade_pnl = (row.exit - row.entry) * qty - 2 * fee_per_trade - slippage
        pnl += trade_pnl
        equity += trade_pnl
    if trades:
        print(f"Trades: {trades}  PnL: {pnl:.2f}  Final equity: {equity:.2f}")
    else:
        print("No trades generated.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True, help="CSV file with columns date,symbol,entry,stop,exit")
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--fee", type=float, default=1.0, help="per-trade fee")
    ap.add_argument("--slippage-bp", type=float, default=5.0, help="slippage in basis points")
    args = ap.parse_args()
    run_backtest(Path(args.signals), equity=args.equity, fee_per_trade=args.fee, slippage_bp=args.slippage_bp)
