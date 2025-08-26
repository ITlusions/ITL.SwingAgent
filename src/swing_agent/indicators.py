from __future__ import annotations
from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass


def ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average.
    
    Args:
        series: Price series.
        span: EMA period.
        
    Returns:
        pd.Series: EMA values.
    """
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.
    
    Args:
        series: Price series.
        length: RSI calculation period.
        
    Returns:
        pd.Series: RSI values [0, 100].
    """
    d = series.diff()
    up = (d.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Calculate Average True Range.
    
    Args:
        df: OHLC DataFrame.
        length: ATR calculation period.
        
    Returns:
        pd.Series: ATR values.
    """
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()


def bollinger_width(price: pd.Series, length: int = 20, ndev: float = 2.0) -> pd.Series:
    """Calculate Bollinger Band width as percentage of price.
    
    Args:
        price: Price series.
        length: Moving average period.
        ndev: Number of standard deviations for bands.
        
    Returns:
        pd.Series: Bollinger Band width as percentage of price.
    """
    ma = price.rolling(length).mean()
    std = price.rolling(length).std(ddof=0)
    upper = ma + ndev * std
    lower = ma - ndev * std
    width = (upper - lower) / price
    return width


def ema_slope(series: pd.Series, span: int = 20, lookback: int = 6) -> float:
    """Calculate EMA slope as percentage change over lookback period.
    
    Args:
        series: Price series.
        span: EMA period.
        lookback: Number of bars to calculate slope over.
        
    Returns:
        float: EMA slope as percentage of current price.
    """
    e = ema(series, span)
    if len(e) < lookback + 1:
        return 0.0
    d = e.iloc[-1] - e.iloc[-(lookback + 1)]
    return float(d / max(1e-9, series.iloc[-1]))


# Fibonacci retracement and extension levels
FIBS: Dict[str, float] = {
    "0.236": 0.236,
    "0.382": 0.382,
    "0.5": 0.5,
    "0.618": 0.618,
    "0.65": 0.65,
    "0.786": 0.786,
    "1.0": 1.0,
    "1.272": 1.272,
    "1.414": 1.414,
    "1.618": 1.618
}


@dataclass
class FibRange:
    """Fibonacci retracement and extension analysis results.
    
    Attributes:
        start: Starting price level (swing low for uptrend, swing high for downtrend).
        end: Ending price level (swing high for uptrend, swing low for downtrend).
        dir_up: True if trend is up (from low to high), False if down.
        levels: Dict of Fibonacci level names to price values.
        golden_low: Lower bound of golden pocket (0.618-0.65 retracement).
        golden_high: Upper bound of golden pocket (0.618-0.65 retracement).
    """
    start: float
    end: float
    dir_up: bool
    levels: Dict[str, float]
    golden_low: float
    golden_high: float


def recent_swing(df: pd.DataFrame, lookback: int = 40) -> Tuple[float, float, bool]:
    """Find most recent swing high and low within lookback period.
    
    Identifies significant price swings for Fibonacci analysis by finding
    the absolute highest and lowest points within the specified lookback.
    Direction is determined by which extreme occurred first.
    
    Args:
        df: OHLC DataFrame with required columns ['high', 'low'].
        lookback: Number of bars to scan for swing extremes.
        
    Returns:
        Tuple containing:
        - swing_low: Absolute lowest price in lookback period
        - swing_high: Absolute highest price in lookback period  
        - direction_up: True if low occurred before high (uptrend swing)
        
    Example:
        >>> df = load_ohlcv("AAPL", "30m", 50)
        >>> low, high, up = recent_swing(df, lookback=40)
        >>> print(f"Swing: {low:.2f} to {high:.2f}")
        >>> print(f"Direction: {'UP' if up else 'DOWN'}")
        >>> swing_size = ((high - low) / low) * 100
        >>> print(f"Swing size: {swing_size:.1f}%")
        
    Note:
        For edge case where high == low (no meaningful swing), 
        returns (low, high, True) to default to upward direction.
        Used internally by fibonacci_range() for level calculations.
    """
    window = df.iloc[-lookback:]
    hi_idx = window["high"].idxmax()
    lo_idx = window["low"].idxmin()
    hi = float(window.loc[hi_idx, "high"])
    lo = float(window.loc[lo_idx, "low"])
    dir_up = lo_idx < hi_idx
    lo2, hi2 = (lo, hi) if lo < hi else (hi, lo)
    return lo2, hi2, dir_up


def fibonacci_range(df: pd.DataFrame, lookback: int = 40) -> FibRange:
    """Calculate Fibonacci retracement and extension levels from recent swing.
    
    Identifies the most significant swing high/low within the lookback period
    and calculates comprehensive Fibonacci levels for entry, stop, and target
    analysis. Core component of the golden pocket strategy.
    
    Args:
        df: OHLC DataFrame with minimum lookback bars of data.
            Must contain columns: ['high', 'low']
        lookback: Number of bars to scan for swing points (default: 40).
        
    Returns:
        FibRange: Object containing:
            - start: Swing start price (low for uptrend, high for downtrend)
            - end: Swing end price (high for uptrend, low for downtrend)  
            - dir_up: True if uptrend (low to high), False if downtrend
            - levels: Dict of all Fibonacci levels by name
            - golden_low: Lower golden pocket boundary (0.618 level)
            - golden_high: Upper golden pocket boundary (0.65 level)
            
    Example:
        >>> df = load_ohlcv("AAPL", "30m", 50)
        >>> fib = fibonacci_range(df, lookback=40)
        >>> print(f"Direction: {'UP' if fib.dir_up else 'DOWN'}")
        >>> print(f"Golden Pocket: {fib.golden_low:.2f} - {fib.golden_high:.2f}")
        >>> print(f"61.8% level: {fib.levels['0.618']:.2f}")
        >>> print(f"127.2% extension: {fib.levels['1.272']:.2f}")
        >>> 
        >>> # Check if current price is in golden pocket
        >>> current = df["close"].iloc[-1]
        >>> in_gp = fib.golden_low <= current <= fib.golden_high
        >>> print(f"In golden pocket: {in_gp}")
        
    Note:
        Golden pocket (61.8%-65% retracement) is the highest probability
        entry zone for pullback strategies. Extensions (127.2%, 161.8%) 
        provide target levels for breakout strategies.
    """
    the swing range. For uptrends, retracements are calculated from the high.
    For downtrends, retracements are calculated from the low.
    
    Args:
        df: OHLC DataFrame with at least 'lookback' bars.
        lookback: Number of bars to analyze for swing points.
        
    Returns:
        FibRange: Complete Fibonacci analysis with levels and golden pocket.
        
    Examples:
        >>> fib = fibonacci_range(df, lookback=40)
        >>> print(f"Golden pocket: {fib.golden_low:.2f} - {fib.golden_high:.2f}")
        >>> print(f"1.272 extension: {fib.levels['1.272']:.2f}")
    """
    lo, hi, dir_up = recent_swing(df, lookback)
    rng = hi - lo if hi != lo else 1e-9
    
    # Calculate Fibonacci levels based on trend direction
    if dir_up:
        # Uptrend: retracements from high, extensions above high
        levels = {k: lo + v * rng for k, v in FIBS.items()}
    else:
        # Downtrend: retracements from low, extensions below low
        levels = {k: hi - v * rng for k, v in FIBS.items()}
        levels["1.0"] = lo  # 100% retracement goes to the swing low
    
    # Golden pocket is between 61.8% and 65% retracement
    gp_low = min(levels["0.618"], levels["0.65"])
    gp_high = max(levels["0.618"], levels["0.65"])
    
    return FibRange(
        start=lo if dir_up else hi,
        end=hi if dir_up else lo,
        dir_up=dir_up,
        levels=levels,
        golden_low=gp_low,
        golden_high=gp_high
    )
