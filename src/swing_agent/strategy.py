from __future__ import annotations

import pandas as pd

from .config import get_config
from .indicators import atr, ema, ema_slope, fibonacci_range, rsi
from .models import EntryPlan, SignalSide, TrendLabel, TrendState


def label_trend(df: pd.DataFrame) -> TrendState:
    """Label the current trend based on EMA slope, price position, and RSI.
    
    Uses three main trend categories:
    1. Strong/Up trends: EMA20 slope positive, price above EMA, RSI >= 60
    2. Strong/Down trends: EMA20 slope negative, price below EMA, RSI <= 40  
    3. Sideways: All other conditions
    
    Args:
        df: OHLCV DataFrame with minimum 20 bars for EMA calculation.
            Must contain columns: ['close']
        
    Returns:
        TrendState: Object containing trend label, EMA slope, price vs EMA, and RSI.
        
    Example:
        >>> df = load_ohlcv("AAPL", "30m", 30)
        >>> trend = label_trend(df)
        >>> print(f"Trend: {trend.label.value}, RSI: {trend.rsi_14:.1f}")
    """
    cfg = get_config()

    price = df["close"]
    ema20 = ema(price, cfg.EMA20_PERIOD)
    slope = ema_slope(price, cfg.EMA20_PERIOD, lookback=cfg.EMA_SLOPE_LOOKBACK)
    rsi14 = rsi(price, cfg.RSI_PERIOD).iloc[-1]
    price_above = price.iloc[-1] > ema20.iloc[-1]

    # Determine trend label using configuration thresholds
    if (slope > cfg.EMA_SLOPE_THRESHOLD_UP and price_above and
        rsi14 >= cfg.RSI_TREND_UP_MIN):
        label = (TrendLabel.STRONG_UP if slope > cfg.EMA_SLOPE_THRESHOLD_STRONG
                else TrendLabel.UP)
    elif (slope < cfg.EMA_SLOPE_THRESHOLD_DOWN and not price_above and
          rsi14 <= cfg.RSI_TREND_DOWN_MAX):
        label = (TrendLabel.STRONG_DOWN if slope < cfg.EMA_SLOPE_THRESHOLD_STRONG_DOWN
                else TrendLabel.DOWN)
    else:
        label = TrendLabel.SIDEWAYS

    return TrendState(
        label=label,
        ema_slope=slope,
        price_above_ema=price_above,
        rsi_14=float(rsi14)
    )

def build_entry(df: pd.DataFrame, trend: TrendState) -> EntryPlan | None:
    """Generate entry plan based on trend analysis and Fibonacci levels.
    
    Uses three main entry strategies:
    1. Fibonacci golden pocket pullbacks (highest probability)
    2. Momentum continuation breakouts  
    3. Mean reversion from extreme RSI levels
    
    Args:
        df: OHLCV DataFrame with at least 40 bars for Fibonacci calculation.
            Must contain columns: ['open', 'high', 'low', 'close', 'volume']
        trend: Current trend state from label_trend()
        
    Returns:
        EntryPlan with entry, stop, target prices and risk metrics, or None if no setup
        
    Risk Management:
        - Stops: Golden pocket boundary + ATR buffer, or 1.2*ATR from entry
        - Targets: Previous swing points or Fibonacci extensions
        
    Examples:
        >>> df = load_ohlcv("AAPL", "30m", 30)
        >>> trend = label_trend(df)
        >>> entry = build_entry(df, trend)
        >>> if entry:
        ...     print(f"Entry: {entry.side.value} at {entry.entry_price}")
    """
    cfg = get_config()

    # Current market data
    price = df["close"].iloc[-1]
    hi, lo = df["high"].iloc[-1], df["low"].iloc[-1]
    atr14 = float(atr(df, cfg.ATR_PERIOD).iloc[-1])
    prev_hi = float(df["high"].iloc[-2])
    prev_lo = float(df["low"].iloc[-2])

    # Fibonacci analysis
    fib = fibonacci_range(df, lookback=cfg.FIB_LOOKBACK)
    gp_lo, gp_hi = fib.golden_low, fib.golden_high
    ext_1272, ext_1618 = fib.levels["1.272"], fib.levels["1.618"]

    def _plan(side: SignalSide, entry: float, sl: float, tp: float, note: str) -> EntryPlan:
        """Helper to create EntryPlan with R-multiple calculation."""
        r_mult = ((tp - entry)/(entry - sl) if side == SignalSide.LONG
                 else (entry - tp)/(sl - entry))
        return EntryPlan(
            side=side,
            entry_price=float(entry),
            stop_price=float(sl),
            take_profit=float(tp),
            r_multiple=float(max(0.0, r_mult)),
            comment=note,
            fib_golden_low=float(gp_lo),
            fib_golden_high=float(gp_hi),
            fib_target_1=float(ext_1272),
            fib_target_2=float(ext_1618)
        )

    # Strategy 1: Fibonacci golden pocket pullbacks
    if trend.label in (TrendLabel.UP, TrendLabel.STRONG_UP) and fib.dir_up:
        if gp_lo <= price <= gp_hi:
            entry = price
            sl = min(lo, gp_lo) - cfg.ATR_STOP_BUFFER * atr14
            tp = max(ext_1272, prev_hi)
            return _plan(SignalSide.LONG, entry, sl, tp,
                        "Fibonacci golden-pocket LONG pullback")

    if trend.label in (TrendLabel.DOWN, TrendLabel.STRONG_DOWN) and not fib.dir_up:
        if gp_lo <= price <= gp_hi:
            entry = price
            sl = max(hi, gp_hi) + cfg.ATR_STOP_BUFFER * atr14
            tp = min(ext_1272, prev_lo)
            return _plan(SignalSide.SHORT, entry, sl, tp,
                        "Fibonacci golden-pocket SHORT pullback")

    # Strategy 2: Momentum continuation breakouts
    if trend.label in (TrendLabel.UP, TrendLabel.STRONG_UP):
        entry = max(prev_hi, price)
        sl = entry - cfg.ATR_STOP_MULTIPLIER * atr14
        tp = max(ext_1272, entry + cfg.ATR_TARGET_MULTIPLIER * atr14)
        return _plan(SignalSide.LONG, entry, sl, tp,
                    "Momentum continuation LONG over prior high")

    if trend.label in (TrendLabel.DOWN, TrendLabel.STRONG_DOWN):
        entry = min(prev_lo, price)
        sl = entry + cfg.ATR_STOP_MULTIPLIER * atr14
        tp = min(ext_1272, entry - cfg.ATR_TARGET_MULTIPLIER * atr14)
        return _plan(SignalSide.SHORT, entry, sl, tp,
                    "Momentum continuation SHORT under prior low")

    # Strategy 3: Mean reversion from extreme RSI levels
    rsi_now = trend.rsi_14
    if trend.label == TrendLabel.SIDEWAYS and rsi_now < cfg.RSI_OVERSOLD_THRESHOLD:
        entry = price
        sl = lo - cfg.ATR_MEAN_REVERSION_STOP * atr14
        tp = entry + cfg.ATR_MEAN_REVERSION_TARGET * atr14
        return _plan(SignalSide.LONG, entry, sl, tp,
                    "Mean-reversion LONG from oversold RSI")

    if trend.label == TrendLabel.SIDEWAYS and rsi_now > cfg.RSI_OVERBOUGHT_THRESHOLD:
        entry = price
        sl = hi + cfg.ATR_MEAN_REVERSION_STOP * atr14
        tp = entry - cfg.ATR_MEAN_REVERSION_TARGET * atr14
        return _plan(SignalSide.SHORT, entry, sl, tp,
                    "Mean-reversion SHORT from overbought RSI")

    return None
