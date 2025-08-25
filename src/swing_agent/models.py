from __future__ import annotations
from enum import Enum
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field

class TrendLabel(str, Enum):
    STRONG_UP = "strong_up"
    UP = "up"
    SIDEWAYS = "sideways"
    DOWN = "down"
    STRONG_DOWN = "strong_down"

class SignalSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"

class EntryPlan(BaseModel):
    side: SignalSide
    entry_price: float = Field(..., gt=0)
    stop_price: float = Field(..., gt=0)
    take_profit: float = Field(..., gt=0)
    r_multiple: float = Field(..., gt=0)
    comment: str = ""
    fib_golden_low: Optional[float] = None
    fib_golden_high: Optional[float] = None
    fib_target_1: Optional[float] = None
    fib_target_2: Optional[float] = None

class TrendState(BaseModel):
    label: TrendLabel
    ema_slope: float
    price_above_ema: bool
    rsi_14: float

class TradeSignal(BaseModel):
    symbol: str
    timeframe: Literal["15m", "30m", "1h", "1d"] = "30m"
    asof: str
    trend: TrendState
    entry: Optional[EntryPlan] = None
    confidence: float = 0.0
    reasoning: str = ""

    # Expectation metrics
    expected_r: float | None = None
    expected_winrate: float | None = None
    expected_source: str | None = None
    expected_notes: str | None = None

    # Holding time priors
    expected_hold_bars: int | None = None
    expected_hold_days: float | None = None
    expected_win_hold_bars: int | None = None
    expected_loss_hold_bars: int | None = None

    # LLM transparency & plan
    llm_vote: Optional[Dict[str, Any]] = None
    llm_explanation: Optional[str] = None
    action_plan: Optional[str] = None
    risk_notes: Optional[str] = None
    scenarios: Optional[List[str]] = None

    # enrichments
    mtf_15m_trend: Optional[str] = None
    mtf_1h_trend: Optional[str] = None
    mtf_alignment: Optional[int] = None
    rs_sector_20: Optional[float] = None
    rs_spy_20: Optional[float] = None
    sector_symbol: Optional[str] = None
    tod_bucket: Optional[str] = None
    atr_pct: Optional[float] = None
    vol_regime: Optional[str] = None
