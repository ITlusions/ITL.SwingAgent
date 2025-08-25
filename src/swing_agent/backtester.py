from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from .models import SignalSide

@dataclass
class TradeResult:
    symbol: str; open_time: pd.Timestamp; close_time: pd.Timestamp; side: str
    entry: float; stop: float; target: float; exit_price: float; exit_reason: str; r_multiple: float

def simulate_trade(df: pd.DataFrame, open_idx: int, side: SignalSide, entry: float, stop: float, target: float, max_hold_bars: int) -> Tuple[int, str, float]:
    for i in range(open_idx + 1, min(len(df), open_idx + 1 + max_hold_bars)):
        high = df["high"].iloc[i]; low = df["low"].iloc[i]
        if side == SignalSide.LONG:
            if low <= stop: return i, "SL", float(stop)
            if high >= target: return i, "TP", float(target)
        else:
            if high >= stop: return i, "SL", float(stop)
            if low <= target: return i, "TP", float(target)
    exit_idx = min(len(df) - 1, open_idx + max_hold_bars)
    return exit_idx, "TIME", float(df["close"].iloc[exit_idx])
