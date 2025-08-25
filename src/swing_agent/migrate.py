"""
Migration utility to move data from separate SQLite files to centralized database.
"""
import sqlite3
import argparse
from pathlib import Path
from typing import Optional
from .database import get_database_config, init_database, get_session
from .models_db import Signal, VectorStore


def migrate_signals(old_signals_db: Path, target_db_url: Optional[str] = None):
    """Migrate signals from old SQLite file to centralized database."""
    if not old_signals_db.exists():
        print(f"Old signals database not found: {old_signals_db}")
        return
    
    # Initialize target database
    if target_db_url:
        init_database(target_db_url)
    else:
        init_database()
    
    migrated_count = 0
    
    with get_session() as session:
        # Read from old database
        with sqlite3.connect(old_signals_db) as old_conn:
            old_conn.row_factory = sqlite3.Row
            cursor = old_conn.execute("SELECT * FROM signals")
            
            for row in cursor.fetchall():
                # Check if signal already exists
                existing = session.query(Signal).filter(Signal.id == row["id"]).first()
                if existing:
                    print(f"Signal {row['id']} already exists, skipping")
                    continue
                
                # Create new signal
                signal = Signal(
                    id=row["id"],
                    created_at_utc=row["created_at_utc"],
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    asof=row["asof"],
                    trend_label=row["trend_label"],
                    ema_slope=row["ema_slope"],
                    price_above_ema=row["price_above_ema"],
                    rsi14=row["rsi14"],
                    side=row["side"],
                    entry_price=row["entry_price"],
                    stop_price=row["stop_price"],
                    take_profit=row["take_profit"],
                    r_multiple=row["r_multiple"],
                    fib_golden_low=row["fib_golden_low"],
                    fib_golden_high=row["fib_golden_high"],
                    fib_target_1=row["fib_target_1"],
                    fib_target_2=row["fib_target_2"],
                    confidence=row["confidence"],
                    reasoning=row["reasoning"],
                    llm_vote_json=row["llm_vote_json"],
                    llm_explanation=row["llm_explanation"],
                    expected_r=row["expected_r"],
                    expected_winrate=row["expected_winrate"],
                    expected_hold_bars=row["expected_hold_bars"],
                    expected_hold_days=row["expected_hold_days"],
                    expected_win_hold_bars=row["expected_win_hold_bars"],
                    expected_loss_hold_bars=row["expected_loss_hold_bars"],
                    action_plan=row["action_plan"],
                    risk_notes=row["risk_notes"],
                    scenarios_json=row["scenarios_json"],
                    mtf_15m_trend=row["mtf_15m_trend"],
                    mtf_1h_trend=row["mtf_1h_trend"],
                    mtf_alignment=row["mtf_alignment"],
                    rs_sector_20=row["rs_sector_20"],
                    rs_spy_20=row["rs_spy_20"],
                    sector_symbol=row["sector_symbol"],
                    tod_bucket=row["tod_bucket"],
                    atr_pct=row["atr_pct"],
                    vol_regime=row["vol_regime"],
                    evaluated=row["evaluated"],
                    exit_reason=row["exit_reason"],
                    exit_price=row["exit_price"],
                    exit_time_utc=row["exit_time_utc"],
                    realized_r=row["realized_r"]
                )
                
                session.add(signal)
                migrated_count += 1
        
        session.commit()
    
    print(f"Migrated {migrated_count} signals from {old_signals_db}")


def migrate_vectors(old_vectors_db: Path, target_db_url: Optional[str] = None):
    """Migrate vectors from old SQLite file to centralized database."""
    if not old_vectors_db.exists():
        print(f"Old vectors database not found: {old_vectors_db}")
        return
    
    # Initialize target database
    if target_db_url:
        init_database(target_db_url)
    else:
        init_database()
    
    migrated_count = 0
    
    with get_session() as session:
        # Read from old database
        with sqlite3.connect(old_vectors_db) as old_conn:
            old_conn.row_factory = sqlite3.Row
            cursor = old_conn.execute("SELECT * FROM vec_store")
            
            for row in cursor.fetchall():
                # Check if vector already exists
                existing = session.query(VectorStore).filter(VectorStore.id == row["id"]).first()
                if existing:
                    print(f"Vector {row['id']} already exists, skipping")
                    continue
                
                # Create new vector
                vector = VectorStore(
                    id=row["id"],
                    ts_utc=row["ts_utc"],
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    vec_json=row["vec_json"],
                    realized_r=row["realized_r"],
                    exit_reason=row["exit_reason"],
                    payload_json=row["payload_json"]
                )
                
                session.add(vector)
                migrated_count += 1
        
        session.commit()
    
    print(f"Migrated {migrated_count} vectors from {old_vectors_db}")


def migrate_data(
    data_dir: Path, 
    target_db_url: Optional[str] = None,
    signals_filename: str = "signals.sqlite",
    vectors_filename: str = "vec_store.sqlite"
):
    """Migrate data from separate files to centralized database."""
    old_signals = data_dir / signals_filename
    old_vectors = data_dir / vectors_filename
    
    print(f"Starting migration from {data_dir}")
    print(f"Target database: {target_db_url or 'default centralized database'}")
    
    migrate_signals(old_signals, target_db_url)
    migrate_vectors(old_vectors, target_db_url)
    
    print("Migration completed!")


def main():
    """Command line interface for migration."""
    parser = argparse.ArgumentParser(description="Migrate data to centralized database")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), 
                       help="Directory containing old database files")
    parser.add_argument("--target-db", type=str, 
                       help="Target database URL (default: centralized SQLite)")
    parser.add_argument("--signals-file", default="signals.sqlite",
                       help="Name of old signals database file")
    parser.add_argument("--vectors-file", default="vec_store.sqlite", 
                       help="Name of old vectors database file")
    
    args = parser.parse_args()
    
    migrate_data(
        data_dir=args.data_dir,
        target_db_url=args.target_db,
        signals_filename=args.signals_file,
        vectors_filename=args.vectors_file
    )


if __name__ == "__main__":
    main()