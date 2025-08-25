from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = (d.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l),(h-pc).abs(),(l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def bollinger_width(price: pd.Series, length: int = 20, ndev: float = 2.0) -> pd.Series:
    ma = price.rolling(length).mean()
    std = price.rolling(length).std(ddof=0)
    upper = ma + ndev*std
    lower = ma - ndev*std
    width = (upper - lower) / price
    return width

def ema_slope(series: pd.Series, span: int = 20, lookback: int = 6) -> float:
    e = ema(series, span)
    if len(e) < lookback + 1: return 0.0
    d = e.iloc[-1] - e.iloc[-(lookback + 1)]
    return float(d / max(1e-9, series.iloc[-1]))

FIBS = {"0.236":0.236,"0.382":0.382,"0.5":0.5,"0.618":0.618,"0.65":0.65,"0.786":0.786,"1.0":1.0,"1.272":1.272,"1.414":1.414,"1.618":1.618}

@dataclass
class FibRange:
    start: float; end: float; dir_up: bool; levels: dict; golden_low: float; golden_high: float

def recent_swing(df: pd.DataFrame, lookback: int = 40):
    window = df.iloc[-lookback:]
    hi_idx = window["high"].idxmax(); lo_idx = window["low"].idxmin()
    hi = float(window.loc[hi_idx, "high"]); lo = float(window.loc[lo_idx, "low"])
    dir_up = lo_idx < hi_idx
    lo2, hi2 = (lo, hi) if lo < hi else (hi, lo)
    return lo2, hi2, dir_up

def fibonacci_range(df: pd.DataFrame, lookback: int = 40) -> FibRange:
    lo, hi, dir_up = recent_swing(df, lookback)
    rng = hi - lo if hi != lo else 1e-9
    levels = {k: (lo + v*rng) if dir_up else (hi - v*rng) for k,v in FIBS.items()}
    if not dir_up: levels["1.0"] = lo
    gp_low = min(levels["0.618"], levels["0.65"]); gp_high = max(levels["0.618"], levels["0.65"])
    return FibRange(start=lo if dir_up else hi, end=hi if dir_up else lo, dir_up=dir_up, levels=levels, golden_low=gp_low, golden_high=gp_high)
