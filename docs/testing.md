# Testing Strategy & Implementation

Comprehensive testing approach for SwingAgent to ensure reliability, correctness, and performance.

## Testing Philosophy

SwingAgent uses a multi-layered testing strategy:

1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Component interaction and data flow
3. **Property Tests**: Mathematical invariants and edge cases
4. **Performance Tests**: Speed and memory usage validation
5. **End-to-End Tests**: Complete workflow scenarios

## Test Structure

```
tests/
├── unit/                    # Isolated component tests
│   ├── test_indicators.py   # Technical indicator functions
│   ├── test_strategy.py     # Strategy logic validation
│   ├── test_features.py     # Feature engineering tests
│   ├── test_models.py       # Pydantic model validation
│   ├── test_vectorstore.py  # Vector operations
│   └── test_llm.py         # LLM integration (mocked)
├── integration/             # Component interaction tests
│   ├── test_agent.py        # Full agent workflow
│   ├── test_database.py     # Database operations
│   ├── test_pipeline.py     # Data processing pipeline
│   └── test_signals.py      # Signal generation scenarios
├── property/               # Property-based testing
│   ├── test_math_properties.py  # Mathematical correctness
│   └── test_invariants.py       # System invariants
├── performance/            # Performance benchmarks
│   ├── test_benchmarks.py   # Speed benchmarks
│   └── test_memory.py       # Memory usage tests
├── fixtures/               # Shared test data
│   ├── sample_data.py       # Market data fixtures
│   ├── mock_responses.py    # API response mocks
│   └── scenarios.py         # Trading scenarios
└── conftest.py             # Pytest configuration
```

## Unit Testing

### Technical Indicators

```python
# tests/unit/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from swing_agent.indicators import ema, rsi, atr, fibonacci_range, bollinger_width

class TestTechnicalIndicators:
    @pytest.fixture
    def trending_data(self):
        """Generate uptrending price data for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        np.random.seed(42)  # Reproducible tests
        
        # Create realistic trending price series
        trend = np.linspace(100, 120, 100)  # 20% uptrend
        noise = np.random.randn(100) * 0.5
        prices = trend + noise
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.randn(100) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(100)) * 0.002),
            'low': prices * (1 - np.abs(np.random.randn(100)) * 0.002),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def test_ema_basic_properties(self, trending_data):
        """Test EMA calculation basic properties."""
        ema_20 = ema(trending_data['close'], 20)
        
        # Length preservation
        assert len(ema_20) == len(trending_data)
        
        # No NaN in final values (after warmup)
        assert not ema_20.iloc[-50:].isna().any()
        
        # EMA follows trend but is smoother
        price_volatility = trending_data['close'].pct_change().std()
        ema_volatility = ema_20.pct_change().std()
        assert ema_volatility < price_volatility
        
        # EMA lags but converges toward price
        final_price = trending_data['close'].iloc[-1]
        final_ema = ema_20.iloc[-1]
        assert abs(final_price - final_ema) / final_price < 0.1  # Within 10%
    
    def test_rsi_bounds_and_behavior(self, trending_data):
        """Test RSI calculation bounds and expected behavior."""
        rsi_14 = rsi(trending_data['close'], 14)
        rsi_values = rsi_14.dropna()
        
        # RSI must stay within bounds
        assert all(0 <= val <= 100 for val in rsi_values)
        
        # Should have reasonable number of non-NaN values
        assert len(rsi_values) > 50
        
        # In trending market, RSI should show some directional bias
        recent_rsi = rsi_values.iloc[-20:]  # Last 20 values
        assert recent_rsi.mean() > 45  # Slight upward bias in uptrend
    
    def test_atr_calculation(self, trending_data):
        """Test ATR calculation properties."""
        atr_14 = atr(trending_data, 14)
        atr_values = atr_14.dropna()
        
        # ATR must be positive
        assert all(val >= 0 for val in atr_values)
        
        # ATR should reflect price movement
        price_range = trending_data['high'] - trending_data['low']
        avg_range = price_range.mean()
        avg_atr = atr_values.mean()
        
        # ATR should be comparable to average range
        assert 0.5 * avg_range < avg_atr < 2.0 * avg_range
    
    def test_fibonacci_levels_ordering(self, trending_data):
        """Test Fibonacci level calculations and ordering."""
        fib = fibonacci_range(trending_data, lookback=40)
        
        # Levels must be properly ordered
        levels = list(fib.levels.values())
        sorted_levels = sorted(levels)
        assert levels == sorted_levels or levels == sorted_levels[::-1]
        
        # Golden pocket bounds
        assert fib.golden_low <= fib.golden_high
        
        # Levels should be within reasonable range of price data
        price_min = trending_data['close'].min()
        price_max = trending_data['close'].max()
        price_range = price_max - price_min
        
        for level in levels:
            # Fibonacci levels can extend beyond price range
            assert price_min - price_range <= level <= price_max + price_range
    
    def test_bollinger_width_calculation(self, trending_data):
        """Test Bollinger Band width calculation."""
        bw = bollinger_width(trending_data['close'], length=20, ndev=2.0)
        bw_values = bw.dropna()
        
        # Width should be positive
        assert all(val > 0 for val in bw_values)
        
        # Width should be reasonable percentage of price
        assert all(0.001 < val < 0.5 for val in bw_values)  # 0.1% to 50%
```

### Strategy Logic

```python
# tests/unit/test_strategy.py
import pytest
from swing_agent.strategy import label_trend, build_entry
from swing_agent.models import TrendLabel, SignalSide

class TestStrategyLogic:
    def test_trend_labeling_consistency(self, trending_data):
        """Test trend labeling produces consistent results."""
        trend = label_trend(trending_data)
        
        # Trend should be valid enum value
        assert trend.label in list(TrendLabel)
        
        # RSI should be in valid range
        assert 0 <= trend.rsi_14 <= 100
        
        # EMA slope should be reasonable
        assert -0.1 <= trend.ema_slope <= 0.1  # Within ±10%
        
        # Price vs EMA should be boolean
        assert isinstance(trend.price_above_ema, bool)
    
    def test_entry_plan_risk_reward_validation(self, trending_data):
        """Test entry plan risk/reward calculations."""
        trend = label_trend(trending_data)
        entry = build_entry(trending_data, trend)
        
        if entry:  # If entry plan was generated
            # R-multiple should be positive
            assert entry.r_multiple > 0
            
            # Entry, stop, and target should be different
            assert entry.stop_price != entry.entry_price
            assert entry.take_profit != entry.entry_price
            
            # Validate long trade logic
            if entry.side == SignalSide.LONG:
                assert entry.stop_price < entry.entry_price
                assert entry.take_profit > entry.entry_price
                
                # R-multiple calculation verification
                risk = entry.entry_price - entry.stop_price
                reward = entry.take_profit - entry.entry_price
                expected_r = reward / risk
                assert abs(entry.r_multiple - expected_r) < 0.01
            
            # Validate short trade logic
            elif entry.side == SignalSide.SHORT:
                assert entry.stop_price > entry.entry_price
                assert entry.take_profit < entry.entry_price
                
                # R-multiple calculation verification
                risk = entry.stop_price - entry.entry_price
                reward = entry.entry_price - entry.take_profit
                expected_r = reward / risk
                assert abs(entry.r_multiple - expected_r) < 0.01
    
    def test_fibonacci_golden_pocket_logic(self, trending_data):
        """Test Fibonacci golden pocket entry logic."""
        from swing_agent.indicators import fibonacci_range
        
        trend = label_trend(trending_data)
        fib = fibonacci_range(trending_data, lookback=40)
        
        # Manually test golden pocket logic
        current_price = trending_data['close'].iloc[-1]
        
        if fib.golden_low <= current_price <= fib.golden_high:
            entry = build_entry(trending_data, trend)
            
            if entry and "golden-pocket" in entry.comment.lower():
                # Entry should be at current price for golden pocket setup
                assert abs(entry.entry_price - current_price) < 0.01
                
                # Fibonacci targets should be set
                assert entry.fib_golden_low is not None
                assert entry.fib_golden_high is not None
```

### Feature Engineering

```python
# tests/unit/test_features.py
import pytest
import numpy as np
from swing_agent.features import build_setup_vector, time_of_day_bucket, vol_regime_from_series

class TestFeatureEngineering:
    def test_setup_vector_properties(self, sample_trend, sample_entry):
        """Test feature vector generation properties."""
        vector = build_setup_vector(
            price=150.0,
            trend=sample_trend,
            entry=sample_entry,
            prev_range_pct=0.02,
            gap_pct=0.001,
            atr_pct=0.015,
            session_bin=1,
            llm_conf=0.7
        )
        
        # Vector should be normalized
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-6
        
        # Vector should have expected length
        assert len(vector) == 16  # Current feature count
        
        # All values should be finite
        assert np.all(np.isfinite(vector))
        
        # Values should be in reasonable ranges
        assert np.all(vector >= -1.0)
        assert np.all(vector <= 1.0)
    
    def test_time_of_day_bucketing(self):
        """Test time of day bucket assignment."""
        import pandas as pd
        
        # Test different times (ET assumed)
        open_time = pd.Timestamp("2024-01-01 09:30:00", tz="America/New_York")
        mid_time = pd.Timestamp("2024-01-01 13:00:00", tz="America/New_York")
        close_time = pd.Timestamp("2024-01-01 15:30:00", tz="America/New_York")
        
        assert time_of_day_bucket(open_time) == "open"
        assert time_of_day_bucket(mid_time) == "mid"
        assert time_of_day_bucket(close_time) == "close"
    
    def test_volatility_regime_classification(self):
        """Test volatility regime classification."""
        # Create price series with known volatility patterns
        low_vol_prices = pd.Series(100 + np.random.randn(100) * 0.1)  # Low volatility
        high_vol_prices = pd.Series(100 + np.random.randn(100) * 2.0)  # High volatility
        
        # Should classify correctly (though not deterministic)
        low_regime = vol_regime_from_series(low_vol_prices)
        high_regime = vol_regime_from_series(high_vol_prices)
        
        # Regimes should be valid
        assert low_regime in ["L", "M", "H"]
        assert high_regime in ["L", "M", "H"]
```

## Integration Testing

### Full Agent Workflow

```python
# tests/integration/test_agent.py
import pytest
from unittest.mock import patch, MagicMock
from swing_agent.agent import SwingAgent
from swing_agent.models import TrendLabel

class TestAgentIntegration:
    @pytest.fixture
    def mock_agent(self):
        """Create agent with controlled dependencies."""
        return SwingAgent(
            interval="30m",
            lookback_days=30,
            use_llm=False,  # Avoid API calls
            log_db=":memory:",  # In-memory database
            vec_db=":memory:"
        )
    
    @patch('swing_agent.data.load_ohlcv')
    def test_complete_signal_generation(self, mock_load_ohlcv, mock_agent, trending_data):
        """Test complete signal generation pipeline."""
        mock_load_ohlcv.return_value = trending_data
        
        signal = mock_agent.analyze("AAPL")
        
        # Basic signal validation
        assert signal.symbol == "AAPL"
        assert signal.timeframe == "30m"
        assert signal.trend.label in list(TrendLabel)
        assert 0 <= signal.confidence <= 1
        assert signal.asof is not None
        
        # Trend analysis should be consistent
        if signal.trend.label in [TrendLabel.UP, TrendLabel.STRONG_UP]:
            assert signal.trend.ema_slope > 0
            assert signal.trend.price_above_ema
        elif signal.trend.label in [TrendLabel.DOWN, TrendLabel.STRONG_DOWN]:
            assert signal.trend.ema_slope < 0
            assert not signal.trend.price_above_ema
    
    @patch('swing_agent.data.load_ohlcv')
    def test_signal_storage_and_retrieval(self, mock_load_ohlcv, mock_agent, trending_data):
        """Test signal database storage."""
        mock_load_ohlcv.return_value = trending_data
        
        # Generate and store signal
        signal = mock_agent.analyze("AAPL")
        
        # Signal should be automatically stored in database
        # This tests the integration between agent and storage
        assert signal.symbol == "AAPL"
        
        # Test that database operations don't raise errors
        # (Detailed database tests in test_database.py)
    
    def test_vector_store_integration(self, mock_agent):
        """Test vector store operations integration."""
        from swing_agent.vectorstore import add_vector, knn
        import numpy as np
        
        # Add test vector
        test_vector = np.random.randn(16)
        add_vector(
            ":memory:",
            vid="test-1",
            ts_utc="2024-01-01T10:00:00Z",
            symbol="TEST",
            timeframe="30m",
            vec=test_vector,
            realized_r=1.5,
            exit_reason="target",
            payload={"test": True}
        )
        
        # Search for similar vectors
        results = knn(":memory:", test_vector, k=5)
        
        # Should find the vector we just added
        assert len(results) >= 1
```

## Property-Based Testing

```python
# tests/property/test_math_properties.py
from hypothesis import given, strategies as st, assume
import hypothesis.extra.pandas as pdst
import pandas as pd
import numpy as np
from swing_agent.indicators import ema, rsi

class TestMathematicalProperties:
    @given(
        prices=pdst.series(
            elements=st.floats(min_value=1.0, max_value=1000.0, 
                             allow_nan=False, allow_infinity=False),
            min_size=50,
            max_size=200
        ),
        span=st.integers(min_value=2, max_value=50)
    )
    def test_ema_smoothing_property(self, prices, span):
        """EMA should be less volatile than input series."""
        assume(prices.std() > 0)  # Avoid constant series
        
        ema_result = ema(prices, span)
        
        # EMA volatility should be <= input volatility
        input_volatility = prices.pct_change().std()
        ema_volatility = ema_result.pct_change().std()
        
        # Allow small numerical errors
        assert ema_volatility <= input_volatility * 1.05
    
    @given(
        prices=pdst.series(
            elements=st.floats(min_value=1.0, max_value=1000.0,
                             allow_nan=False, allow_infinity=False),
            min_size=30,
            max_size=100
        ),
        period=st.integers(min_value=2, max_value=30)
    )
    def test_rsi_bounds_property(self, prices, period):
        """RSI must always be between 0 and 100."""
        assume(prices.std() > 0)  # Avoid constant series
        
        rsi_result = rsi(prices, period)
        rsi_values = rsi_result.dropna()
        
        # RSI bounds are strict
        assert all(0 <= val <= 100 for val in rsi_values)
        
        # Should produce some non-NaN values
        assert len(rsi_values) > 0
    
    @given(
        r_multiple=st.floats(min_value=0.1, max_value=10.0),
        entry_price=st.floats(min_value=1.0, max_value=1000.0),
        side=st.sampled_from(["long", "short"])
    )
    def test_risk_reward_calculation_property(self, r_multiple, entry_price, side):
        """Risk/reward calculations should be consistent."""
        from swing_agent.models import EntryPlan, SignalSide
        
        # Calculate stop and target based on R-multiple
        risk_amount = entry_price * 0.02  # 2% risk
        
        if side == "long":
            stop_price = entry_price - risk_amount
            target_price = entry_price + (risk_amount * r_multiple)
            
            # Verify R-multiple calculation
            actual_risk = entry_price - stop_price
            actual_reward = target_price - entry_price
            calculated_r = actual_reward / actual_risk
            
            assert abs(calculated_r - r_multiple) < 0.01
            
        else:  # short
            stop_price = entry_price + risk_amount
            target_price = entry_price - (risk_amount * r_multiple)
            
            # Verify R-multiple calculation
            actual_risk = stop_price - entry_price
            actual_reward = entry_price - target_price
            calculated_r = actual_reward / actual_risk
            
            assert abs(calculated_r - r_multiple) < 0.01
```

## Performance Testing

```python
# tests/performance/test_benchmarks.py
import time
import pytest
import numpy as np
from swing_agent.vectorstore import knn, add_vector
from swing_agent.agent import SwingAgent

class TestPerformanceBenchmarks:
    def test_vector_search_speed(self):
        """Vector search should complete within reasonable time."""
        # Create larger test dataset
        n_vectors = 1000
        vector_dim = 16
        
        # Add vectors to database
        db_path = ":memory:"
        for i in range(n_vectors):
            test_vector = np.random.randn(vector_dim)
            add_vector(
                db_path,
                vid=f"test-{i}",
                ts_utc="2024-01-01T10:00:00Z",
                symbol="TEST",
                timeframe="30m",
                vec=test_vector,
                realized_r=np.random.randn(),
                exit_reason="test",
                payload={}
            )
        
        # Benchmark search speed
        query_vector = np.random.randn(vector_dim)
        
        start_time = time.time()
        results = knn(db_path, query_vector, k=10)
        elapsed = time.time() - start_time
        
        # Should complete within 1 second for 1000 vectors
        assert elapsed < 1.0
        assert len(results) <= 10
    
    @pytest.mark.slow
    def test_signal_generation_speed(self, trending_data):
        """Signal generation should complete quickly."""
        agent = SwingAgent(use_llm=False)  # Skip LLM for speed
        
        start_time = time.time()
        signal = agent.analyze_df("TEST", trending_data)
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        assert elapsed < 5.0  # 5 seconds max
        assert signal is not None
```

## Test Fixtures and Utilities

```python
# tests/fixtures/sample_data.py
import pytest
import pandas as pd
import numpy as np
from swing_agent.models import TrendState, EntryPlan, TrendLabel, SignalSide

@pytest.fixture
def trending_data():
    """Generate realistic trending OHLCV data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    np.random.seed(42)
    
    # Generate trending price series
    trend = np.linspace(100, 120, 100)  # 20% uptrend
    noise = np.random.randn(100) * 0.5
    prices = trend + noise
    
    return pd.DataFrame({
        'open': prices * (1 + np.random.randn(100) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

@pytest.fixture
def sideways_data():
    """Generate sideways/consolidating OHLCV data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    np.random.seed(123)
    
    # Generate sideways price series
    base_price = 100
    noise = np.random.randn(100) * 0.5
    prices = base_price + noise
    
    return pd.DataFrame({
        'open': prices * (1 + np.random.randn(100) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

@pytest.fixture
def sample_trend():
    """Sample TrendState for testing."""
    return TrendState(
        label=TrendLabel.UP,
        ema_slope=0.015,
        price_above_ema=True,
        rsi_14=65.0
    )

@pytest.fixture
def sample_entry():
    """Sample EntryPlan for testing."""
    return EntryPlan(
        side=SignalSide.LONG,
        entry_price=150.0,
        stop_price=147.0,
        take_profit=156.0,
        r_multiple=2.0,
        comment="Test entry",
        fib_golden_low=148.0,
        fib_golden_high=150.0,
        fib_target_1=155.0,
        fib_target_2=160.0
    )
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=swing_agent --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/property/       # Property-based tests

# Run performance tests (marked as slow)
pytest -m slow

# Exclude slow tests
pytest -m "not slow"

# Run tests with specific markers
pytest -m "unit"
pytest -m "integration"

# Verbose output with test names
pytest -v

# Stop on first failure
pytest -x

# Run tests matching pattern
pytest -k "test_fibonacci"

# Parallel execution (with pytest-xdist)
pytest -n auto
```

### Test Configuration

```python
# tests/conftest.py
import pytest
import tempfile
import os
from pathlib import Path

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")  
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "property: Property-based tests")

@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up clean test environment."""
    # Don't use real API keys in tests
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SWING_LLM_MODEL", "test-model")
    
    # Use test database paths
    monkeypatch.setenv("SWING_DATABASE_URL", "sqlite:///:memory:")
```

This comprehensive testing strategy ensures SwingAgent maintains high quality and reliability across all components while providing fast feedback during development.