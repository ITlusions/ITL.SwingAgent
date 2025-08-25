from __future__ import annotations
from typing import Optional
import pandas as pd
from .indicators import ema, rsi, atr, ema_slope, fibonacci_range

from .models import EntryPlan, SignalSide, TrendLabel, TrendState

def label_trend(df: pd.DataFrame) -> TrendState:
    price = df["close"]; ema20 = ema(price, 20)
    slope = ema_slope(price, 20, lookback=6)
    rsi14 = rsi(price, 14).iloc[-1]
    price_above = price.iloc[-1] > ema20.iloc[-1]
    if slope > 0.01 and price_above and rsi14 >= 60: label = TrendLabel.STRONG_UP if slope > 0.02 else TrendLabel.UP
    elif slope < -0.01 and (not price_above) and rsi14 <= 40: label = TrendLabel.STRONG_DOWN if slope < -0.02 else TrendLabel.DOWN
    else: label = TrendLabel.SIDEWAYS
    return TrendState(label=label, ema_slope=slope, price_above_ema=price_above, rsi_14=float(rsi14))

def build_entry(df: pd.DataFrame, trend: TrendState) -> Optional[EntryPlan]:
    price = df["close"].iloc[-1]; hi, lo = df["high"].iloc[-1], df["low"].iloc[-1]
    atr14 = float(atr(df, 14).iloc[-1]); prev_hi = float(df["high"].iloc[-2]); prev_lo = float(df["low"].iloc[-2])
    fib = fibonacci_range(df, lookback=40); gp_lo, gp_hi = fib.golden_low, fib.golden_high
    ext_1272, ext_1618 = fib.levels["1.272"], fib.levels["1.618"]
    def _plan(side: SignalSide, entry: float, sl: float, tp: float, note: str):
        r_mult = (tp - entry)/(entry - sl) if side==SignalSide.LONG else (entry - tp)/(sl - entry)
        return EntryPlan(side=side, entry_price=float(entry), stop_price=float(sl), take_profit=float(tp),
                         r_multiple=float(max(0.0, r_mult)), comment=note,
                         fib_golden_low=float(gp_lo), fib_golden_high=float(gp_hi),
                         fib_target_1=float(ext_1272), fib_target_2=float(ext_1618))
    if trend.label in (TrendLabel.UP, TrendLabel.STRONG_UP) and fib.dir_up:
        if gp_lo <= price <= gp_hi:
            entry = price; sl = min(lo, gp_lo) - 0.2*atr14; tp = max(ext_1272, prev_hi)
            return _plan(SignalSide.LONG, entry, sl, tp, "Fibonacci golden-pocket LONG pullback")
    if trend.label in (TrendLabel.DOWN, TrendLabel.STRONG_DOWN) and (not fib.dir_up):
        if gp_lo <= price <= gp_hi:
            entry = price; sl = max(hi, gp_hi) + 0.2*atr14; tp = min(ext_1272, prev_lo)
            return _plan(SignalSide.SHORT, entry, sl, tp, "Fibonacci golden-pocket SHORT pullback")
    if trend.label in (TrendLabel.UP, TrendLabel.STRONG_UP):
        entry = max(prev_hi, price); sl = entry - 1.2*atr14; tp = max(ext_1272, entry + 2.0*atr14)
        return _plan(SignalSide.LONG, entry, sl, tp, "Momentum continuation LONG over prior high")
    if trend.label in (TrendLabel.DOWN, TrendLabel.STRONG_DOWN):
        entry = min(prev_lo, price); sl = entry + 1.2*atr14; tp = min(ext_1272, entry - 2.0*atr14)
        return _plan(SignalSide.SHORT, entry, sl, tp, "Momentum continuation SHORT under prior low")
    rsi_now = trend.rsi_14
    if trend.label == TrendLabel.SIDEWAYS and rsi_now < 35:
        entry = price; sl = lo - 1.0*atr14; tp = entry + 1.5*atr14
        return _plan(SignalSide.LONG, entry, sl, tp, "Mean-reversion LONG from oversold RSI")
    if trend.label == TrendLabel.SIDEWAYS and rsi_now > 65:
        entry = price; sl = hi + 1.0*atr14; tp = entry - 1.5*atr14
        return _plan(SignalSide.SHORT, entry, sl, tp, "Mean-reversion SHORT from overbought RSI")
    return None
