from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .models import SignalSide


@dataclass
class TradeResult:
    symbol: str; open_time: pd.Timestamp; close_time: pd.Timestamp; side: str
    entry: float; stop: float; target: float; exit_price: float; exit_reason: str; r_multiple: float

def simulate_trade(df: pd.DataFrame, open_idx: int, side: SignalSide, entry: float, stop: float, target: float, max_hold_bars: int) -> tuple[int, str, float]:
    """Simulate trade execution with stop loss, take profit, and time exit logic.
    
    Executes a simulated trade through historical price data to determine
    actual exit conditions and R-multiple outcomes. Essential for backtesting
    and outcome evaluation.
    
    Args:
        df: OHLC price DataFrame with required columns ['high', 'low', 'close'].
        open_idx: Index where trade opens (next bar after signal).
        side: Trade direction (LONG or SHORT).
        entry: Entry price level.
        stop: Stop loss price level.
        target: Take profit price level.
        max_hold_bars: Maximum bars to hold before time exit.
        
    Returns:
        Tuple containing:
        - exit_idx: Bar index where trade exited
        - exit_reason: Exit type ("TP", "SL", "TIME")
        - exit_price: Actual exit price achieved
        
    Example:
        >>> df = load_ohlcv("AAPL", "30m", 100)
        >>> exit_idx, reason, price = simulate_trade(
        ...     df=df,
        ...     open_idx=50,
        ...     side=SignalSide.LONG,
        ...     entry=150.0,
        ...     stop=148.0,
        ...     target=154.0,
        ...     max_hold_bars=48  # 2 days at 30min bars
        ... )
        >>> r_multiple = (price - 150.0) / (150.0 - 148.0)
        >>> print(f"Exit: {reason} at {price:.2f}, R = {r_multiple:.2f}")
        
    Note:
        Assumes fills at exact stop/target levels. For long trades, checks
        stop before target on same bar. For short trades, checks stop before
        target. Uses close price for time-based exits.
    """
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
