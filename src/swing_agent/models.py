from __future__ import annotations

from enum import Enum
from typing import Any, Literal

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
    fib_golden_low: float | None = None
    fib_golden_high: float | None = None
    fib_target_1: float | None = None
    fib_target_2: float | None = None

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
    entry: EntryPlan | None = None
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
    llm_vote: dict[str, Any] | None = None
    llm_explanation: str | None = None
    action_plan: str | None = None
    risk_notes: str | None = None
    scenarios: list[str] | None = None

    # enrichments
    mtf_15m_trend: str | None = None
    mtf_1h_trend: str | None = None
    mtf_alignment: int | None = None
    rs_sector_20: float | None = None
    rs_spy_20: float | None = None
    sector_symbol: str | None = None
    tod_bucket: str | None = None
    atr_pct: float | None = None
    vol_regime: str | None = None
