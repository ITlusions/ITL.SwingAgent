from __future__ import annotations
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from typing import Union, Optional
from .models import TradeSignal
from .database import get_database_config, init_database, get_session
from .models_db import Signal


def _ensure_db(db_path: Union[str, Path]):
    """Ensure database exists. For backward compatibility, convert file path to database URL."""
    if isinstance(db_path, (str, Path)):
        path = Path(db_path)
        if path.suffix == '.sqlite':
            # Extract directory to use as centralized database location
            data_dir = path.parent
            data_dir.mkdir(parents=True, exist_ok=True)
            # Use centralized database regardless of the input path
            database_url = f"sqlite:///{data_dir / 'swing_agent.sqlite'}"
            init_database(database_url)
        else:
            # Assume it's a database URL
            init_database(str(db_path))
    else:
        init_database()


def record_signal(ts: TradeSignal, db_path: Union[str, Path]) -> str:
    """Record a complete trade signal to the centralized database.
    
    Persists all signal components including trend analysis, entry plan,
    ML expectations, LLM insights, and enrichments for later evaluation
    and analysis. Returns unique signal ID for tracking.
    
    Args:
        ts: Complete TradeSignal object with all components populated.
        db_path: Database path or URL (converted to centralized database).
        
    Returns:
        str: Unique signal identifier (UUID) for tracking and evaluation.
        
    Example:
        >>> signal = agent.analyze_df("AAPL", df)
        >>> signal_id = record_signal(signal, "data/signals.sqlite")
        >>> print(f"Signal recorded with ID: {signal_id}")
        >>> 
        >>> # Later, mark evaluation results
        >>> mark_evaluation(
        ...     "data/signals.sqlite",
        ...     signal_id=signal_id,
        ...     realized_r=1.85,
        ...     exit_reason="TP",
        ...     exit_price=154.50,
        ...     hold_bars=36
        ... )
        
    Note:
        Uses centralized database architecture for all signal storage.
        All signal components are serialized to JSON for flexibility.
        Created timestamp is automatically set to current UTC time.
    """
    _ensure_db(db_path)
    
    sid = str(uuid4())
    
    with get_session() as session:
        signal = Signal(
            id=sid,
            created_at_utc=datetime.utcnow(),
            symbol=ts.symbol,
            timeframe=ts.timeframe,
            asof=ts.asof,
            trend_label=ts.trend.label.value,
            ema_slope=float(ts.trend.ema_slope),
            price_above_ema=1 if ts.trend.price_above_ema else 0,
            rsi14=float(ts.trend.rsi_14),
            side=ts.entry.side.value if ts.entry else None,
            entry_price=ts.entry.entry_price if ts.entry else None,
            stop_price=ts.entry.stop_price if ts.entry else None,
            take_profit=ts.entry.take_profit if ts.entry else None,
            r_multiple=ts.entry.r_multiple if ts.entry else None,
            fib_golden_low=ts.entry.fib_golden_low if ts.entry else None,
            fib_golden_high=ts.entry.fib_golden_high if ts.entry else None,
            fib_target_1=ts.entry.fib_target_1 if ts.entry else None,
            fib_target_2=ts.entry.fib_target_2 if ts.entry else None,
            confidence=float(ts.confidence),
            reasoning=ts.reasoning,
            llm_vote=ts.llm_vote,
            llm_explanation=ts.llm_explanation,
            expected_r=ts.expected_r,
            expected_winrate=ts.expected_winrate,
            expected_hold_bars=ts.expected_hold_bars,
            expected_hold_days=ts.expected_hold_days,
            expected_win_hold_bars=ts.expected_win_hold_bars,
            expected_loss_hold_bars=ts.expected_loss_hold_bars,
            action_plan=ts.action_plan,
            risk_notes=ts.risk_notes,
            scenarios=ts.scenarios,
            mtf_15m_trend=ts.mtf_15m_trend,
            mtf_1h_trend=ts.mtf_1h_trend,
            mtf_alignment=ts.mtf_alignment,
            rs_sector_20=ts.rs_sector_20,
            rs_spy_20=ts.rs_spy_20,
            sector_symbol=ts.sector_symbol,
            tod_bucket=ts.tod_bucket,
            atr_pct=ts.atr_pct,
            vol_regime=ts.vol_regime
        )
        
        session.add(signal)
        session.commit()
    
    return sid


def mark_evaluation(
    signal_id: str, 
    *, 
    db_path: Union[str, Path], 
    exit_reason: str, 
    exit_price: Optional[float], 
    exit_time_utc: Optional[str], 
    realized_r: Optional[float]
):
    """Mark a signal as evaluated with actual trading outcome results.
    
    Updates a previously recorded signal with backtesting or live trading
    results including exit conditions, realized returns, and performance metrics.
    
    Args:
        signal_id: Unique signal identifier from record_signal().
        db_path: Database path or URL (converted to centralized database).
        exit_reason: How trade exited ("TP", "SL", "TIME").
        exit_price: Actual exit price achieved.
        exit_time_utc: UTC timestamp of exit (ISO format).
        realized_r: Actual R-multiple return achieved.
        
    Example:
        >>> # After backtesting or live trading
        >>> mark_evaluation(
        ...     signal_id="abc123-def456-ghi789",
        ...     db_path="data/signals.sqlite",
        ...     exit_reason="TP",
        ...     exit_price=154.50,
        ...     exit_time_utc="2024-01-16T11:45:00Z",
        ...     realized_r=1.85
        ... )
        >>> print("Signal evaluation recorded")
        
    Note:
        Silently succeeds if signal_id doesn't exist in database.
        Used by eval_signals.py script for batch outcome evaluation.
    """
    _ensure_db(db_path)
    
    with get_session() as session:
        signal = session.query(Signal).filter(Signal.id == signal_id).first()
        if signal:
            signal.evaluated = 1
            signal.exit_reason = exit_reason
            signal.exit_price = exit_price
            signal.exit_time_utc = exit_time_utc
            signal.realized_r = realized_r
            session.commit()


# Keep the old SCHEMA for reference/migration purposes
SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
  id TEXT PRIMARY KEY,
  created_at_utc TEXT NOT NULL,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  asof TEXT NOT NULL,
  trend_label TEXT NOT NULL,
  ema_slope REAL NOT NULL,
  price_above_ema INTEGER NOT NULL,
  rsi14 REAL NOT NULL,
  side TEXT,
  entry_price REAL,
  stop_price REAL,
  take_profit REAL,
  r_multiple REAL,
  fib_golden_low REAL,
  fib_golden_high REAL,
  fib_target_1 REAL,
  fib_target_2 REAL,
  confidence REAL NOT NULL,
  reasoning TEXT,
  llm_vote_json TEXT,
  llm_explanation TEXT,

  -- expectations & plan
  expected_r REAL,
  expected_winrate REAL,
  expected_hold_bars INTEGER,
  expected_hold_days REAL,
  expected_win_hold_bars INTEGER,
  expected_loss_hold_bars INTEGER,
  action_plan TEXT,
  risk_notes TEXT,
  scenarios_json TEXT,

  -- enrichments
  mtf_15m_trend TEXT,
  mtf_1h_trend TEXT,
  mtf_alignment INTEGER,
  rs_sector_20 REAL,
  rs_spy_20 REAL,
  sector_symbol TEXT,
  tod_bucket TEXT,
  atr_pct REAL,
  vol_regime TEXT,

  evaluated INTEGER DEFAULT 0,
  exit_reason TEXT,
  exit_price REAL,
  exit_time_utc TEXT,
  realized_r REAL
);
"""
