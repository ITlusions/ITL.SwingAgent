#!/usr/bin/env python3
"""
Simple test to verify the database centralization works.
"""
import sys
import os
import tempfile
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from swing_agent.database import init_database, get_session
from swing_agent.models_db import Signal, VectorStore
import json
import numpy as np
from datetime import datetime


def test_centralized_database():
    """Test that we can create and use the centralized database."""
    # Use a temporary database for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_db = Path(temp_dir) / "test_swing_agent.sqlite"
        db_url = f"sqlite:///{test_db}"
        
        print(f"Testing with database: {test_db}")
        
        # Initialize database
        init_database(db_url)
        print("✓ Database initialized")
        
        # Test adding a signal
        with get_session() as session:
            signal = Signal(
                id="test-signal-1",
                created_at_utc=datetime.utcnow(),
                symbol="AAPL",
                timeframe="30m",
                asof="2024-01-01T10:00:00Z",
                trend_label="BULLISH",
                ema_slope=0.5,
                price_above_ema=1,
                rsi14=60.0,
                confidence=0.8
            )
            session.add(signal)
            session.commit()
            
            # Verify signal was added
            retrieved = session.query(Signal).filter(Signal.id == "test-signal-1").first()
            assert retrieved is not None
            assert retrieved.symbol == "AAPL"
            print("✓ Signal storage works")
        
        # Test adding a vector
        with get_session() as session:
            test_vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            vector = VectorStore(
                id="test-vector-1",
                ts_utc="2024-01-01T10:00:00Z",
                symbol="AAPL",
                timeframe="30m",
                vec_json=json.dumps(test_vector.tolist()),
                payload={"test": "data"}
            )
            session.add(vector)
            session.commit()
            
            # Verify vector was added
            retrieved = session.query(VectorStore).filter(VectorStore.id == "test-vector-1").first()
            assert retrieved is not None
            assert retrieved.symbol == "AAPL"
            assert retrieved.payload == {"test": "data"}
            print("✓ Vector storage works")
        
        # Test that both tables exist in the same database
        with get_session() as session:
            signal_count = session.query(Signal).count()
            vector_count = session.query(VectorStore).count()
            assert signal_count == 1
            assert vector_count == 1
            print("✓ Both tables coexist in centralized database")
        
        print("✓ All tests passed!")


if __name__ == "__main__":
    test_centralized_database()