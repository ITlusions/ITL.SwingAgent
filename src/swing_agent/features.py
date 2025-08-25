from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from .models import TrendState, EntryPlan, TrendLabel
from .indicators import ema, rsi, atr, bollinger_width
from .config import get_config

def fib_position(price: float, gp_low: Optional[float], gp_high: Optional[float]) -> float:
    """Calculate relative position within Fibonacci golden pocket.
    
    Args:
        price: Current price.
        gp_low: Golden pocket low level.
        gp_high: Golden pocket high level.
        
    Returns:
        float: Position within golden pocket [0, 1], or 0.5 if invalid levels.
    """
    if gp_low is None or gp_high is None or gp_high <= gp_low:
        return 0.5
    return max(0.0, min(1.0, (price - gp_low) / (gp_high - gp_low)))


def time_of_day_bucket(ts: pd.Timestamp) -> str:
    """Determine trading session bucket based on market hours.
    
    Args:
        ts: Timestamp in market timezone (Eastern).
        
    Returns:
        str: One of "open", "mid", or "close".
    """
    hour = ts.hour
    minute = ts.minute
    mins = hour * 60 + minute
    open_mins = 9 * 60 + 30  # 9:30 AM
    close_mins = 16 * 60     # 4:00 PM
    
    if mins <= open_mins + 90:  # First 90 minutes
        return "open"
    if mins >= close_mins - 60:  # Last 60 minutes
        return "close"
    return "mid"


def vol_regime_from_series(price: pd.Series) -> str:
    """Determine volatility regime based on Bollinger Band width.
    
    Args:
        price: Price series for volatility analysis.
        
    Returns:
        str: Volatility regime - "L" (low), "M" (medium), or "H" (high).
    """
    cfg = get_config()
    
    # Calculate Bollinger Band width
    bw = bollinger_width(price, length=20, ndev=2.0)
    recent = (bw.dropna().iloc[-cfg.VOL_REGIME_LOOKBACK:] 
             if len(bw.dropna()) >= cfg.VOL_REGIME_LOOKBACK 
             else bw.dropna())
    
    if recent.empty:
        return "M"
    
    # Determine regime using configuration percentiles
    q_low = recent.quantile(cfg.VOL_LOW_PERCENTILE)
    q_high = recent.quantile(cfg.VOL_HIGH_PERCENTILE)
    now = recent.iloc[-1]
    
    if now <= q_low:
        return "L"
    elif now >= q_high:
        return "H"
    else:
        return "M"

def build_setup_vector(*, price: float, trend: TrendState, entry: Optional[EntryPlan], 
                      prev_range_pct: float = 0.0, gap_pct: float = 0.0, 
                      atr_pct: float = 0.0, session_bin: int = 0, 
                      llm_conf: float = 0.0) -> np.ndarray:
    """Build normalized feature vector for ML pattern matching.
    
    Creates a compact feature vector representing the current market setup
    for similarity matching in the vector database.
    
    Args:
        price: Current price.
        trend: Trend state with label, slope, RSI, etc.
        entry: Entry plan if available.
        prev_range_pct: Previous bar range as % of close.
        gap_pct: Gap from previous close as % of close.
        atr_pct: ATR as % of current price.
        session_bin: Trading session (0=open, 1=mid, 2=close).
        llm_conf: LLM confidence score [0, 1].
        
    Returns:
        np.ndarray: Normalized feature vector for ML analysis.
    """
    # Trend direction encoding
    t_up = 1.0 if trend.label in (TrendLabel.UP, TrendLabel.STRONG_UP) else 0.0
    t_down = 1.0 if trend.label in (TrendLabel.DOWN, TrendLabel.STRONG_DOWN) else 0.0
    t_side = 1.0 if trend.label == TrendLabel.SIDEWAYS else 0.0
    
    # Session encoding (binary features for open/close vs mid)
    sb0 = 1.0 if session_bin in (0, 2) else 0.0  # Open or close session
    sb1 = 1.0 if session_bin in (1, 2) else 0.0  # Mid or close session
    
    # Fibonacci position and golden pocket status
    fib_pos = fib_position(
        price, 
        getattr(entry, "fib_golden_low", None), 
        getattr(entry, "fib_golden_high", None)
    )
    
    in_golden = 0.0
    if (entry and entry.fib_golden_low and entry.fib_golden_high and 
        entry.fib_golden_low <= price <= entry.fib_golden_high):
        in_golden = 1.0
    
    # Risk-reward ratio (capped at 5.0 for normalization)
    r_mult = float(getattr(entry, "r_multiple", 0.0) or 0.0)
    
    # Build feature vector
    vec = np.array([
        float(trend.ema_slope),          # EMA slope (momentum)
        float(trend.rsi_14) / 100.0,     # RSI normalized [0, 1]
        atr_pct,                         # Volatility measure
        1.0 if trend.price_above_ema else 0.0,  # Price vs EMA
        prev_range_pct,                  # Previous bar range
        gap_pct,                         # Gap magnitude
        fib_pos,                         # Fibonacci position
        in_golden,                       # In golden pocket flag
        r_mult / 5.0,                    # R-multiple normalized
        t_up,                            # Uptrend flag
        t_down,                          # Downtrend flag
        t_side,                          # Sideways flag
        sb0,                             # Open/close session
        sb1,                             # Mid/close session
        llm_conf,                        # LLM confidence
        1.0                              # Constant term
    ], dtype=float)
    
    # L2 normalization for cosine similarity
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm
