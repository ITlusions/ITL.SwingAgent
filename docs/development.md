# Development Guide

Complete guide for SwingAgent v1.6.1 development, architecture, and code quality standards.

## Architecture Overview

SwingAgent v1.6.1 follows a **modular, layered architecture** with centralized database management:

```
┌─────────────────────────────────────┐
│            API Layer                │  
│  (Scripts + Agent Interface)        │
├─────────────────────────────────────┤
│          Business Logic             │
│  (Strategy + Indicators + Features) │
├─────────────────────────────────────┤
│         Data & ML Layer             │
│  (Vector Store + LLM + Storage)     │
├─────────────────────────────────────┤
│      Database Layer (NEW)           │
│  (SQLAlchemy ORM + Migration)       │
├─────────────────────────────────────┤
│        Infrastructure              │
│  (Data Fetching + Configuration)    │
└─────────────────────────────────────┘
```

### Key Architecture Changes in v1.6.1

1. **Centralized Database Layer**: Single database for all storage needs
2. **SQLAlchemy ORM**: Type-safe database operations with relationship support
3. **Multiple Database Backends**: SQLite, PostgreSQL, MySQL, CloudNativePG
4. **Configuration Management**: Centralized `TradingConfig` class
5. **Migration Framework**: Automated migration from legacy databases

### Core Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Configurable Parameters**: All magic numbers centralized in `TradingConfig`
3. **Comprehensive Error Handling**: Custom exceptions with detailed error context
4. **Type Safety**: Complete type annotations throughout the codebase
5. **Database Agnostic**: Support for multiple backends via environment configuration
6. **Backward Compatibility**: Migration tools for legacy database formats
5. **Documentation**: Comprehensive docstrings with examples for all public APIs

### Key Components

#### 1. `SwingAgent` (Orchestrator)
**File**: `src/swing_agent/agent.py`

Main orchestrator that coordinates the entire analysis pipeline through focused methods:

- `_build_market_context()` - Market data collection and enrichment
- `_perform_technical_analysis()` - Core technical analysis
- `_get_multitimeframe_analysis()` - MTF trend alignment
- `_calculate_confidence()` - Base confidence scoring
- `_get_ml_expectations()` - Vector similarity analysis
- `_get_llm_insights()` - LLM analysis and action plans
- `_assemble_signal()` - Final signal creation

**Benefits of Refactored Design**:
- Each method has single responsibility (~20-40 lines vs 160+ original)
- Easy to test individual components
- Clear error handling and debugging
- Better code reusability

#### 2. `TradingConfig` (Configuration Management)
**File**: `src/swing_agent/config.py`

Centralized configuration eliminates magic numbers and provides easy parameter tuning:

```python
from swing_agent.config import get_config, update_config

# Access configuration
config = get_config()
print(f"EMA threshold: {config.EMA_SLOPE_THRESHOLD_UP}")

# Update parameters
update_config(RSI_PERIOD=21, ATR_STOP_MULTIPLIER=1.5)
```

**Configuration Groups**:
- **Trend Detection**: EMA/RSI thresholds
- **Risk Management**: ATR multipliers
- **Fibonacci Analysis**: Lookback periods
- **Volatility Analysis**: Percentile thresholds
- **Confidence Scoring**: Base levels and bonuses

#### 3. Strategy Functions (Enhanced)
**File**: `src/swing_agent/strategy.py`

- `label_trend()` - Uses configurable thresholds for trend classification
- `build_entry()` - Three main strategies with comprehensive documentation:
  1. Fibonacci golden pocket pullbacks
  2. Momentum continuation breakouts
  3. Mean reversion from extremes

#### 4. Database Layer (NEW in v1.6.1)
**Files**: `src/swing_agent/database.py`, `src/swing_agent/models_db.py`, `src/swing_agent/migrate.py`

Centralized database management using SQLAlchemy ORM:

```python
from swing_agent.database import get_session, init_database
from swing_agent.models_db import Signal, VectorStore

# Initialize database (creates tables if missing)
init_database()

# Use database session
with get_session() as session:
    signals = session.query(Signal).filter(
        Signal.symbol == "AAPL"
    ).order_by(Signal.created_at_utc.desc()).limit(10).all()
```

**Database Features**:
- **Multiple Backends**: SQLite (dev), PostgreSQL/MySQL (prod), CNPG (K8s)
- **Connection Management**: SQLAlchemy engine with connection pooling
- **Type Safety**: Pydantic models with SQLAlchemy ORM
- **Migration Support**: Automated migration from legacy databases

**Configuration Examples**:
```bash
# SQLite (default)
export SWING_DATABASE_URL="sqlite:///data/swing_agent.sqlite"

# PostgreSQL  
export SWING_DATABASE_URL="postgresql://user:pass@host:5432/swing_agent"

# CloudNativePG
export SWING_DB_TYPE="cnpg"
export CNPG_CLUSTER_NAME="swing-postgres"
```

#### 5. Enhanced Error Handling
**File**: `src/swing_agent/data.py`

Custom `SwingAgentDataError` provides context-specific error handling:

```python
try:
    df = load_ohlcv("INVALID_SYMBOL")
except SwingAgentDataError as e:
    if "not found" in str(e):
        # Handle invalid symbol
    elif "rate limit" in str(e):
        # Handle API rate limiting
```

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

## Extending the System

### Adding New Indicators

1. **Add calculation function to `indicators.py`**:

```python
def your_indicator(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate your custom indicator.
    
    Args:
        series: Price series.
        period: Calculation period.
        
    Returns:
        pd.Series: Indicator values.
        
    Examples:
        >>> indicator_values = your_indicator(df['close'], 20)
        >>> current_value = indicator_values.iloc[-1]
    """
    # Implementation here
    return result
```

2. **Add to strategy logic in `strategy.py`**:

```python
def label_trend(df: pd.DataFrame) -> TrendState:
    # Existing code...
    your_value = your_indicator(df['close'], 20).iloc[-1]
    
    # Use in trend logic...
    # Update trend classification if needed
```

3. **Update feature vector in `features.py`** (if needed for ML):

```python
def build_setup_vector(...):
    # Add new feature
    your_feature = your_value / 100.0  # Normalize appropriately
    
    vec = np.array([
        # ... existing features ...
        your_feature,  # Add your feature
        1.0  # Keep constant term last
    ], dtype=float)
```

4. **Update configuration** (if parameters needed):

```python
# In config.py TradingConfig class
YOUR_INDICATOR_PERIOD: int = 20
YOUR_INDICATOR_THRESHOLD: float = 0.5
```

### Adding New Entry Strategies

1. **Add strategy to `build_entry()` in `strategy.py`**:

```python
def build_entry(df: pd.DataFrame, trend: TrendState) -> Optional[EntryPlan]:
    # ... existing strategies ...
    
    # Your new strategy
    if your_strategy_condition(df, trend):
        entry = calculate_entry_price(df)
        sl = calculate_stop_loss(df, entry)
        tp = calculate_take_profit(df, entry)
        return _plan(SignalSide.LONG, entry, sl, tp, "Your strategy description")
    
    return None
```

2. **Add supporting functions**:

```python
def your_strategy_condition(df: pd.DataFrame, trend: TrendState) -> bool:
    """Check if your strategy conditions are met."""
    cfg = get_config()
    # Use configuration parameters
    # Return True if setup is valid

def calculate_entry_price(df: pd.DataFrame) -> float:
    """Calculate optimal entry price for your strategy."""
    # Implementation
```

### Customizing Configuration

1. **Update `config.py`** with new parameters:

```python
@dataclass
class TradingConfig:
    # ... existing parameters ...
    
    # Your new parameters
    YOUR_STRATEGY_THRESHOLD: float = 0.75
    YOUR_STRATEGY_LOOKBACK: int = 14
    YOUR_STRATEGY_MULTIPLIER: float = 1.5
```

2. **Use in your code**:

```python
from swing_agent.config import get_config

def your_function():
    cfg = get_config()
    if some_value > cfg.YOUR_STRATEGY_THRESHOLD:
        # Use configured threshold
```

3. **Update at runtime**:

```python
from swing_agent.config import update_config

# Tune parameters for different market conditions
update_config(
    YOUR_STRATEGY_THRESHOLD=0.8,  # More conservative
    ATR_STOP_MULTIPLIER=1.0       # Tighter stops
)
```

### Error Handling Best Practices

1. **Use appropriate exception types**:

```python
from swing_agent.data import SwingAgentDataError

def your_data_function(symbol: str):
    try:
        # Data processing
        result = process_data(symbol)
    except Exception as e:
        raise SwingAgentDataError(
            f"Failed to process {symbol}: {e}",
            symbol=symbol
        ) from e
```

2. **Graceful degradation**:

```python
def your_optional_feature():
    try:
        # Try enhanced feature
        return enhanced_calculation()
    except Exception:
        # Fallback to basic version
        return basic_calculation()
```

3. **Detailed error context**:

```python
def validate_data(df: pd.DataFrame, symbol: str):
    if df.empty:
        raise SwingAgentDataError(
            f"No data available for {symbol}. Check if symbol is valid "
            f"and markets are open.",
            symbol=symbol
        )
    
    if len(df) < 20:
        raise SwingAgentDataError(
            f"Insufficient data for {symbol}: only {len(df)} bars. "
            f"Need at least 20 bars for analysis.",
            symbol=symbol
        )
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