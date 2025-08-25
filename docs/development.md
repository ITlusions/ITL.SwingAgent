# Development Guide

Complete guide for SwingAgent development, testing, and code quality.

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Virtual environment tool (venv, conda, etc.)

### Local Development Environment

```bash
# Clone and setup
git clone https://github.com/ITlusions/ITL.SwingAgent.git
cd ITL.SwingAgent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black ruff mypy pre-commit

# Set up pre-commit hooks
pre-commit install
```

### Environment Configuration

Create `.env` file for local development:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
SWING_LLM_MODEL=gpt-4o-mini
SWING_DATABASE_URL=sqlite:///data/swing_agent_dev.sqlite
SWING_LOG_LEVEL=DEBUG
```

## Code Quality Standards

### Code Style

We use Ruff for fast linting and Black for code formatting:

```bash
# Check code quality
ruff check src/ scripts/
ruff check --fix src/ scripts/  # Auto-fix issues

# Format code
black src/ scripts/

# Type checking
mypy src/
```

### Ruff Configuration

Located in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings  
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "N",  # flake8-naming
    "UP", # pyupgrade
    "C4", # flake8-comprehensions
    "PIE", # flake8-pie
]

ignore = [
    "E501",  # Line too long (handled by black)
    "B008",  # Do not perform function calls in argument defaults
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]
"scripts/*.py" = ["T201"]  # Allow print statements in scripts
```

### Type Checking

All new code should include type hints:

```python
# Good
def fibonacci_range(df: pd.DataFrame, lookback: int = 40) -> FibRange:
    """Calculate Fibonacci retracement levels."""
    # Implementation

# Avoid
def fibonacci_range(df, lookback=40):
    # Missing type information
```

## Testing Strategy

### Test Structure

```
tests/
├── unit/
│   ├── test_indicators.py
│   ├── test_strategy.py
│   ├── test_features.py
│   └── test_models.py
├── integration/
│   ├── test_agent.py
│   ├── test_vectorstore.py
│   └── test_storage.py
├── fixtures/
│   ├── sample_data.py
│   └── mock_responses.py
└── conftest.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=swing_agent --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run tests matching pattern
pytest -k "test_fibonacci"

# Verbose output
pytest -v
```

### Test Examples

#### Unit Test Example

```python
# tests/unit/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from swing_agent.indicators import ema, rsi, atr, fibonacci_range

class TestIndicators:
    @pytest.fixture
    def sample_ohlcv(self):
        """Generate realistic OHLCV test data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        np.random.seed(42)  # Reproducible tests
        
        # Generate trending price series
        trend = np.linspace(100, 110, 100)
        noise = np.random.randn(100) * 0.5
        prices = trend + noise
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.randn(100) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(100)) * 0.002),
            'low': prices * (1 - np.abs(np.random.randn(100)) * 0.002), 
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def test_ema_calculation(self, sample_ohlcv):
        """Test EMA calculation correctness and properties."""
        ema_20 = ema(sample_ohlcv['close'], 20)
        
        # Basic validation
        assert len(ema_20) == len(sample_ohlcv)
        assert not ema_20.isna().all()
        assert ema_20.iloc[-1] > 0
        
        # EMA should be smooth (less volatile than price)
        price_volatility = sample_ohlcv['close'].pct_change().std()
        ema_volatility = ema_20.pct_change().std()
        assert ema_volatility < price_volatility
        
        # EMA should lag price but follow trend
        final_price = sample_ohlcv['close'].iloc[-1]
        final_ema = ema_20.iloc[-1]
        assert abs(final_price - final_ema) < final_price * 0.1  # Within 10%
    
    def test_rsi_bounds(self, sample_ohlcv):
        """Test RSI stays within valid bounds."""
        rsi_14 = rsi(sample_ohlcv['close'], 14)
        rsi_values = rsi_14.dropna()
        
        assert all(0 <= val <= 100 for val in rsi_values)
        assert len(rsi_values) > 0
    
    def test_fibonacci_levels_ordering(self, sample_ohlcv):
        """Test Fibonacci levels are properly ordered."""
        fib = fibonacci_range(sample_ohlcv, lookback=40)
        
        # Levels should be ordered
        assert fib.levels["0.236"] < fib.levels["0.618"]
        assert fib.levels["0.618"] < fib.levels["1.0"]
        assert fib.levels["1.0"] < fib.levels["1.272"]
        
        # Golden pocket should be within range
        assert fib.golden_low <= fib.golden_high
        assert fib.start <= fib.golden_low <= fib.end
        assert fib.start <= fib.golden_high <= fib.end
```

#### Integration Test Example

```python
# tests/integration/test_agent.py
import pytest
from unittest.mock import patch, MagicMock
from swing_agent.agent import SwingAgent
from swing_agent.models import TrendLabel

class TestSwingAgentIntegration:
    @pytest.fixture
    def mock_agent(self):
        """Create agent with mocked external dependencies."""
        return SwingAgent(
            interval="30m",
            lookback_days=30,
            use_llm=False,  # Avoid API calls in tests
            log_db=":memory:",  # In-memory SQLite
            vec_db=":memory:"
        )
    
    @patch('swing_agent.data.load_ohlcv')
    def test_signal_generation_pipeline(self, mock_load_ohlcv, mock_agent, sample_ohlcv):
        """Test complete signal generation without external dependencies."""
        mock_load_ohlcv.return_value = sample_ohlcv
        
        signal = mock_agent.analyze("AAPL")
        
        # Validate signal structure
        assert signal.symbol == "AAPL"
        assert signal.timeframe == "30m"
        assert signal.trend.label in list(TrendLabel)
        assert 0 <= signal.confidence <= 1
        assert signal.asof is not None
        
        # If entry plan exists, validate risk/reward
        if signal.entry:
            assert signal.entry.r_multiple > 0
            assert signal.entry.stop_price != signal.entry.entry_price
            assert signal.entry.take_profit != signal.entry.entry_price
            
            # Validate long/short trade logic
            if signal.entry.side.value == "long":
                assert signal.entry.stop_price < signal.entry.entry_price
                assert signal.entry.take_profit > signal.entry.entry_price
            elif signal.entry.side.value == "short":
                assert signal.entry.stop_price > signal.entry.entry_price
                assert signal.entry.take_profit < signal.entry.entry_price
```

### Property-Based Testing

For mathematical functions, use hypothesis for property-based testing:

```python
# tests/unit/test_properties.py
from hypothesis import given, strategies as st
import hypothesis.extra.pandas as pdst
from swing_agent.indicators import ema, rsi

class TestIndicatorProperties:
    @given(
        prices=pdst.series(
            elements=st.floats(min_value=1.0, max_value=1000.0),
            min_size=50,
            max_size=200
        ),
        span=st.integers(min_value=2, max_value=50)
    )
    def test_ema_properties(self, prices, span):
        """Test EMA properties hold for any valid input."""
        ema_result = ema(prices, span)
        
        # EMA should have same length as input
        assert len(ema_result) == len(prices)
        
        # EMA should not be more volatile than input
        input_std = prices.pct_change().std()
        ema_std = ema_result.pct_change().std()
        
        # Allow for numerical precision issues
        assert ema_std <= input_std * 1.1
```

## Database Development

### Database Migrations

When modifying database schema:

```python
# Add to migration script
def upgrade_schema_to_v1_6_2():
    """Upgrade database schema to version 1.6.2."""
    with get_session() as session:
        # Add new columns with appropriate defaults
        session.execute("""
            ALTER TABLE signals 
            ADD COLUMN new_field TEXT DEFAULT '';
        """)
        
        # Update schema version
        session.execute("""
            INSERT OR REPLACE INTO migrations (version, applied_at, description)
            VALUES ('1.6.2', datetime('now'), 'Added new_field to signals table');
        """)
        session.commit()
```

### Database Testing

Use in-memory SQLite for fast database tests:

```python
@pytest.fixture
def temp_database():
    """Create temporary database for testing."""
    from swing_agent.database import init_database
    db_url = "sqlite:///:memory:"
    init_database(db_url)
    return db_url
```

## Performance Testing

### Benchmarking

```python
# tests/performance/test_benchmarks.py
import time
import pytest
from swing_agent.vectorstore import knn

class TestPerformance:
    def test_vector_search_performance(self, sample_vectors):
        """Ensure vector search completes within acceptable time."""
        start_time = time.time()
        
        results = knn(":memory:", sample_vectors[0], k=10)
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete within 1 second
        assert len(results) <= 10
    
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Generate large test dataset
        # Run performance benchmarks
        pass
```

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def build_entry(df: pd.DataFrame, trend: TrendState) -> Optional[EntryPlan]:
    """Generate entry plan based on trend analysis and Fibonacci levels.
    
    This function implements three primary entry strategies:
    1. Fibonacci golden pocket pullbacks (highest probability)
    2. Momentum continuation breakouts
    3. Mean reversion from extreme RSI levels
    
    Args:
        df: OHLCV price data with at least 40 bars for Fibonacci calculation
        trend: Current trend state from label_trend()
        
    Returns:
        EntryPlan with entry, stop, target prices and risk metrics, or None if no setup
        
    Raises:
        ValueError: If df has insufficient data for analysis
        
    Examples:
        >>> df = load_ohlcv("AAPL", "30m", 30)
        >>> trend = label_trend(df)
        >>> entry = build_entry(df, trend)
        >>> if entry:
        ...     print(f"Entry: {entry.side} at {entry.entry_price}")
        
    Note:
        - All price levels are absolute values, not relative to current price
        - R-multiple calculation assumes proper position sizing
        
    See Also:
        label_trend(): Trend classification function
        fibonacci_range(): Fibonacci level calculation
    """
```

### API Documentation

Generate API docs with Sphinx:

```bash
# Install sphinx and extensions
pip install sphinx sphinx-autodoc-typehints

# Generate documentation
sphinx-quickstart docs/
sphinx-apidoc -o docs/source/ src/swing_agent/
cd docs && make html
```

## Debugging

### Logging Configuration

```python
# For development debugging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Component-specific debugging
logger = logging.getLogger("swing_agent.strategy")
logger.setLevel(logging.DEBUG)
```

### Debug Utilities

```python
# Debug signal generation
def debug_signal(symbol: str):
    """Generate signal with debug information."""
    agent = SwingAgent(use_llm=False)  # Avoid API costs
    
    # Load data
    df = load_ohlcv(symbol, "30m", 30)
    print(f"Data shape: {df.shape}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Analyze trend
    trend = label_trend(df)
    print(f"Trend: {trend.label.value}, RSI: {trend.rsi_14:.1f}")
    
    # Generate signal
    signal = agent.analyze_df(symbol, df)
    print(f"Signal: {signal.model_dump_json(indent=2)}")
    
    return signal
```

## Release Process

### Version Management

1. Update version in `pyproject.toml`
2. Update `__version__` in `src/swing_agent/__init__.py`
3. Create release notes in `CHANGELOG.md`
4. Tag release: `git tag v1.6.2`

### Pre-release Checklist

- [ ] All tests pass: `pytest`
- [ ] Code quality checks: `ruff check && black --check .`
- [ ] Type checking: `mypy src/`
- [ ] Documentation builds: `sphinx-build docs/ docs/_build/`
- [ ] Manual testing of core scenarios
- [ ] Security review of changes
- [ ] Performance impact assessment

### Continuous Integration

Recommended GitHub Actions workflow:

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest ruff black mypy
        
    - name: Lint with ruff
      run: ruff check src/ scripts/
      
    - name: Check formatting
      run: black --check src/ scripts/
      
    - name: Type check
      run: mypy src/
      
    - name: Run tests
      run: pytest --cov=swing_agent
```

This development guide ensures consistent code quality, comprehensive testing, and maintainable documentation across the SwingAgent project.