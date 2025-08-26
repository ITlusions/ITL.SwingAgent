"""
SQLAlchemy models for the swing agent database.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.ext.hybrid import hybrid_property

from .database import Base


class Signal(Base):
    """SQLAlchemy model for trade signals."""
    __tablename__ = "signals"

    # Primary key
    id = Column(String, primary_key=True)
    created_at_utc = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Basic signal info
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    asof = Column(String, nullable=False)  # ISO timestamp as string for compatibility

    # Trend information
    trend_label = Column(String, nullable=False)
    ema_slope = Column(Float, nullable=False)
    price_above_ema = Column(Integer, nullable=False)  # Boolean as integer
    rsi14 = Column(Float, nullable=False)

    # Entry information
    side = Column(String)  # 'LONG' or 'SHORT'
    entry_price = Column(Float)
    stop_price = Column(Float)
    take_profit = Column(Float)
    r_multiple = Column(Float)

    # Fibonacci levels
    fib_golden_low = Column(Float)
    fib_golden_high = Column(Float)
    fib_target_1 = Column(Float)
    fib_target_2 = Column(Float)

    # Signal quality
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text)
    llm_vote_json = Column(Text)
    llm_explanation = Column(Text)

    # Expectations & plan
    expected_r = Column(Float)
    expected_winrate = Column(Float)
    expected_hold_bars = Column(Integer)
    expected_hold_days = Column(Float)
    expected_win_hold_bars = Column(Integer)
    expected_loss_hold_bars = Column(Integer)
    action_plan = Column(Text)
    risk_notes = Column(Text)
    scenarios_json = Column(Text)

    # Enrichments
    mtf_15m_trend = Column(String)
    mtf_1h_trend = Column(String)
    mtf_alignment = Column(Integer)
    rs_sector_20 = Column(Float)
    rs_spy_20 = Column(Float)
    sector_symbol = Column(String)
    tod_bucket = Column(String)
    atr_pct = Column(Float)
    vol_regime = Column(String)

    # Evaluation results
    evaluated = Column(Integer, default=0)
    exit_reason = Column(String)
    exit_price = Column(Float)
    exit_time_utc = Column(String)
    realized_r = Column(Float)

    @hybrid_property
    def llm_vote(self) -> dict[str, Any] | None:
        """Get LLM vote as parsed JSON."""
        if self.llm_vote_json:
            try:
                return json.loads(self.llm_vote_json)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    @llm_vote.setter
    def llm_vote(self, value: dict[str, Any] | None):
        """Set LLM vote as JSON string."""
        if value is not None:
            self.llm_vote_json = json.dumps(value)
        else:
            self.llm_vote_json = None

    @hybrid_property
    def scenarios(self) -> list | None:
        """Get scenarios as parsed JSON."""
        if self.scenarios_json:
            try:
                return json.loads(self.scenarios_json)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    @scenarios.setter
    def scenarios(self, value: list | None):
        """Set scenarios as JSON string."""
        if value is not None:
            self.scenarios_json = json.dumps(value)
        else:
            self.scenarios_json = None


class VectorStore(Base):
    """SQLAlchemy model for vector store."""
    __tablename__ = "vec_store"

    # Primary key
    id = Column(String, primary_key=True)
    ts_utc = Column(String, nullable=False)  # ISO timestamp as string

    # Basic info
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)

    # Vector data
    vec_json = Column(Text, nullable=False)  # JSON array of floats

    # Results
    realized_r = Column(Float)
    exit_reason = Column(String)

    # Additional payload
    payload_json = Column(Text)

    @hybrid_property
    def payload(self) -> dict[str, Any] | None:
        """Get payload as parsed JSON."""
        if self.payload_json:
            try:
                return json.loads(self.payload_json)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    @payload.setter
    def payload(self, value: dict[str, Any] | None):
        """Set payload as JSON string."""
        if value is not None:
            self.payload_json = json.dumps(value)
        else:
            self.payload_json = None


# Create indexes for better query performance
Index('idx_signals_symbol', Signal.symbol)
Index('idx_signals_asof', Signal.asof)
Index('idx_signals_evaluated', Signal.evaluated)
Index('idx_signals_symbol_asof', Signal.symbol, Signal.asof)
Index('idx_vec_store_symbol', VectorStore.symbol)
Index('idx_vec_store_symbol_ts', VectorStore.symbol, VectorStore.ts_utc)
