from __future__ import annotations
from typing import Optional
import numpy as np, pandas as pd
from .models import TrendState, EntryPlan, TrendLabel
from .indicators import ema, rsi, atr, bollinger_width

def fib_position(price: float, gp_low: Optional[float], gp_high: Optional[float]) -> float:
    if gp_low is None or gp_high is None or gp_high <= gp_low: return 0.5
    return max(0.0, min(1.0, (price - gp_low) / (gp_high - gp_low)))

def time_of_day_bucket(ts: pd.Timestamp) -> str:
    hour = ts.hour; minute = ts.minute
    mins = hour*60 + minute
    open_mins = 9*60+30; close_mins = 16*60
    if mins <= open_mins + 90: return "open"
    if mins >= close_mins - 60: return "close"
    return "mid"

def vol_regime_from_series(price: pd.Series) -> str:
    bw = bollinger_width(price, length=20, ndev=2.0)
    recent = bw.dropna().iloc[-60:] if len(bw.dropna())>=60 else bw.dropna()
    if recent.empty: return "M"
    q33, q66 = recent.quantile(0.33), recent.quantile(0.66)
    now = recent.iloc[-1]
    return "L" if now <= q33 else ("H" if now >= q66 else "M")

def build_setup_vector(*, price: float, trend: TrendState, entry: Optional[EntryPlan], prev_range_pct: float=0.0, gap_pct: float=0.0, atr_pct: float=0.0, session_bin: int=0, llm_conf: float=0.0):
    t_up = 1.0 if trend.label in (TrendLabel.UP, TrendLabel.STRONG_UP) else 0.0
    t_down = 1.0 if trend.label in (TrendLabel.DOWN, TrendLabel.STRONG_DOWN) else 0.0
    t_side = 1.0 if trend.label == TrendLabel.SIDEWAYS else 0.0
    sb0 = 1.0 if session_bin in (0,2) else 0.0
    sb1 = 1.0 if session_bin in (1,2) else 0.0
    fib_pos = fib_position(price, getattr(entry, "fib_golden_low", None), getattr(entry, "fib_golden_high", None))
    in_golden = 1.0 if (entry and entry.fib_golden_low and entry.fib_golden_high and entry.fib_golden_low <= price <= entry.fib_golden_high) else 0.0
    r_mult = float(getattr(entry, "r_multiple", 0.0) or 0.0)
    vec = np.array([float(trend.ema_slope), float(trend.rsi_14)/100.0, atr_pct, 1.0 if trend.price_above_ema else 0.0, prev_range_pct, gap_pct, fib_pos, in_golden, r_mult/5.0, t_up, t_down, t_side, sb0, sb1, llm_conf, 1.0], dtype=float)
    n = np.linalg.norm(vec); return vec if n==0 else vec/n
