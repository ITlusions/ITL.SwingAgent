from __future__ import annotations
import sqlite3, json
from pathlib import Path
from uuid import uuid4
from datetime import datetime
from .models import TradeSignal

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

def _ensure_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA)

def record_signal(ts: TradeSignal, db_path: str | Path) -> str:
    db_path = Path(db_path); _ensure_db(db_path)
    sid = str(uuid4())
    with sqlite3.connect(db_path) as con:
        con.execute("""
            INSERT INTO signals (
              id, created_at_utc, symbol, timeframe, asof,
              trend_label, ema_slope, price_above_ema, rsi14,
              side, entry_price, stop_price, take_profit, r_multiple,
              fib_golden_low, fib_golden_high, fib_target_1, fib_target_2,
              confidence, reasoning, llm_vote_json, llm_explanation,
              expected_r, expected_winrate, expected_hold_bars, expected_hold_days,
              expected_win_hold_bars, expected_loss_hold_bars,
              action_plan, risk_notes, scenarios_json,
              mtf_15m_trend, mtf_1h_trend, mtf_alignment, rs_sector_20, rs_spy_20, sector_symbol,
              tod_bucket, atr_pct, vol_regime
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sid, datetime.utcnow().isoformat(timespec="seconds"),
            ts.symbol, ts.timeframe, ts.asof,
            ts.trend.label.value, float(ts.trend.ema_slope), 1 if ts.trend.price_above_ema else 0, float(ts.trend.rsi_14),
            (ts.entry.side.value if ts.entry else None),
            (ts.entry.entry_price if ts.entry else None),
            (ts.entry.stop_price if ts.entry else None),
            (ts.entry.take_profit if ts.entry else None),
            (ts.entry.r_multiple if ts.entry else None),
            (ts.entry.fib_golden_low if ts.entry else None),
            (ts.entry.fib_golden_high if ts.entry else None),
            (ts.entry.fib_target_1 if ts.entry else None),
            (ts.entry.fib_target_2 if ts.entry else None),
            float(ts.confidence), ts.reasoning,
            json.dumps(ts.llm_vote) if ts.llm_vote else None, ts.llm_explanation,
            ts.expected_r, ts.expected_winrate, ts.expected_hold_bars, ts.expected_hold_days,
            ts.expected_win_hold_bars, ts.expected_loss_hold_bars,
            ts.action_plan, ts.risk_notes,
            json.dumps(ts.scenarios) if ts.scenarios else None,
            ts.mtf_15m_trend, ts.mtf_1h_trend, ts.mtf_alignment, ts.rs_sector_20, ts.rs_spy_20, ts.sector_symbol,
            ts.tod_bucket, ts.atr_pct, ts.vol_regime
        ))
    return sid

def mark_evaluation(signal_id: str, *, db_path: str | Path, exit_reason: str, exit_price: float | None, exit_time_utc: str | None, realized_r: float | None):
    db_path = Path(db_path); _ensure_db(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute("UPDATE signals SET evaluated=1, exit_reason=?, exit_price=?, exit_time_utc=?, realized_r=? WHERE id=?", (exit_reason, exit_price, exit_time_utc, realized_r, signal_id))
