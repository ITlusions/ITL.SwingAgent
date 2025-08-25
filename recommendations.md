# SwingAgent Code Review & Recommendations

## Executive Summary

The SwingAgent system is a sophisticated 1-2 day swing trading platform that demonstrates strong technical foundations in combining traditional technical analysis, machine learning pattern matching, and LLM-driven insights. The codebase shows good architectural thinking with clear separation of concerns across data management, technical analysis, ML vector stores, and AI integration.

**Strengths:**
- Well-structured modular architecture with clear separation of concerns
- Comprehensive technical analysis implementation with Fibonacci, RSI, EMA, ATR
- Innovative ML pattern matching via vector similarity using SQLAlchemy ORM
- Thoughtful LLM integration without letting AI control risk management
- Proper data models using Pydantic v2 for type safety and validation
- Complete signal tracking and evaluation framework with enrichments
- Recently centralized database architecture improving maintainability
- Good use of type hints and modern Python 3.12+ features

**Recent Improvements (v1.6.1):**
- ✅ Database centralization with SQLAlchemy ORM replacing raw SQL
- ✅ External database support (PostgreSQL, MySQL, CNPG)
- ✅ Enhanced multi-timeframe analysis and enrichment features
- ✅ Comprehensive documentation structure

**Critical Areas for Improvement:**
- Code complexity and maintainability (some functions >100 lines)
- Missing comprehensive testing infrastructure
- Limited error handling and recovery mechanisms
- Performance optimization opportunities in vector operations
- Configuration management and environment setup
- Security hardening for production deployment
- Monitoring and observability capabilities

## Detailed Recommendations

### 1. Code Quality & Maintainability

#### 1.1 Function Complexity - HIGH PRIORITY

**Issue**: The `analyze_df` method in `agent.py` is 160+ lines with multiple responsibilities.

**Current Code Pattern**:
```python
def analyze_df(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
    # Data preprocessing (15 lines)
    # Technical analysis (20 lines)
    # Multi-timeframe analysis (25 lines)
    # Vector store lookup (30 lines)
    # LLM integration (40 lines)
    # Signal assembly (30 lines)
```

**Recommendation**: Break into smaller, focused methods:
```python
class SwingAgent:
    def analyze_df(self, symbol: str, df: pd.DataFrame) -> TradeSignal:
        # Orchestrate analysis pipeline
        context = self._build_market_context(symbol, df)
        trend, entry = self._perform_technical_analysis(df)
        expectations = self._get_ml_expectations(context, trend, entry)
        llm_insights = self._get_llm_insights(context, trend, entry)
        return self._assemble_signal(symbol, df, trend, entry, expectations, llm_insights)
    
    def _build_market_context(self, symbol: str, df: pd.DataFrame) -> MarketContext:
        # Extract market context and enrichments
    
    def _perform_technical_analysis(self, df: pd.DataFrame) -> Tuple[TrendState, EntryPlan]:
        # Trend labeling and entry planning
    
    def _get_ml_expectations(self, context: MarketContext, trend: TrendState, entry: EntryPlan) -> MLExpectations:
        # Vector store lookup and statistical analysis
```

**Benefits**:
- Easier testing of individual components
- Better code reusability
- Clearer error handling and debugging
- Simplified maintenance

#### 1.2 Magic Numbers - MEDIUM PRIORITY

**Issue**: Hardcoded values throughout the codebase without clear explanation.

**Examples**:
```python
# strategy.py
if slope > 0.01 and price_above and rsi14 >= 60:  # Why 0.01? Why 60?
sl = min(lo, gp_lo) - 0.2*atr14  # Why 0.2?

# features.py  
recent = bw.dropna().iloc[-60:] if len(bw.dropna())>=60  # Why 60?
q33, q66 = recent.quantile(0.33), recent.quantile(0.66)  # Why 33/66?
```

**Recommendation**: Create a configuration class:
```python
@dataclass
class TradingConfig:
    # Trend thresholds
    EMA_SLOPE_THRESHOLD_UP: float = 0.01
    EMA_SLOPE_THRESHOLD_STRONG: float = 0.02
    RSI_TREND_UP_MIN: float = 60.0
    RSI_TREND_DOWN_MAX: float = 40.0
    
    # Risk management
    ATR_STOP_BUFFER: float = 0.2
    ATR_STOP_MULTIPLIER: float = 1.2
    ATR_TARGET_MULTIPLIER: float = 2.0
    
    # Volatility regime
    VOL_REGIME_LOOKBACK: int = 60
    VOL_LOW_PERCENTILE: float = 0.33
    VOL_HIGH_PERCENTILE: float = 0.66
    
    # Fibonacci
    FIB_LOOKBACK: int = 40
    GOLDEN_POCKET_LOW: float = 0.618
    GOLDEN_POCKET_HIGH: float = 0.65

config = TradingConfig()

# Usage in strategy.py
if slope > config.EMA_SLOPE_THRESHOLD_UP and price_above and rsi14 >= config.RSI_TREND_UP_MIN:
    label = TrendLabel.STRONG_UP if slope > config.EMA_SLOPE_THRESHOLD_STRONG else TrendLabel.UP
```

#### 1.3 Error Handling - HIGH PRIORITY

**Issue**: Limited error handling with potential for silent failures.

**Current Pattern**:
```python
# llm_predictor.py - No error handling
def llm_extra_prediction(**features) -> LlmVote:
    agent = _make_agent(model_name, sys, LlmVote)
    res = agent.run(user_message="...", input=features)
    return res.data  # Could fail silently
```

**Recommendation**: Implement comprehensive error handling:
```python
import logging
from typing import Optional
from enum import Enum

class ErrorSeverity(Enum):
    LOW = "low"       # Degraded functionality, continue
    MEDIUM = "medium" # Significant impact, log warning
    HIGH = "high"     # Critical failure, raise exception

class SwingAgentError(Exception):
    def __init__(self, message: str, severity: ErrorSeverity, component: str):
        super().__init__(message)
        self.severity = severity
        self.component = component

def safe_llm_prediction(**features) -> Optional[LlmVote]:
    try:
        agent = _make_agent(model_name, sys, LlmVote)
        res = agent.run(user_message="...", input=features)
        return res.data
    except openai.RateLimitError as e:
        logging.warning(f"LLM rate limit hit: {e}")
        return None  # Graceful degradation
    except openai.AuthenticationError as e:
        raise SwingAgentError(f"LLM auth failed: {e}", ErrorSeverity.HIGH, "llm")
    except Exception as e:
        logging.error(f"Unexpected LLM error: {e}")
        return None

def safe_data_fetch(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    try:
        return load_ohlcv(symbol, interval, lookback_days)
    except Exception as e:
        if "No data" in str(e):
            raise SwingAgentError(f"Invalid symbol or no data: {symbol}", ErrorSeverity.HIGH, "data")
        else:
            raise SwingAgentError(f"Data fetch failed: {e}", ErrorSeverity.HIGH, "data")
```

#### 1.4 Type Annotations - MEDIUM PRIORITY

**Issue**: Missing type hints in several functions.

**Current**:
```python
def _context_from_df(df):  # Missing return type
def _clip01(x):           # Missing parameter and return types
```

**Recommendation**: Add complete type annotations:
```python
def _context_from_df(df: pd.DataFrame) -> Dict[str, Any]:
def _clip01(x: float) -> float:
def fibonacci_range(df: pd.DataFrame, lookback: int = 40) -> FibRange:
```

#### 1.5 Documentation - MEDIUM PRIORITY

**Issue**: Limited docstrings and inline documentation.

**Recommendation**: Add comprehensive docstrings:
```python
def build_entry(df: pd.DataFrame, trend: TrendState) -> Optional[EntryPlan]:
    """
    Generate entry plan based on trend analysis and Fibonacci levels.
    
    Uses three main strategies:
    1. Fibonacci golden pocket pullbacks (highest probability)
    2. Momentum continuation breakouts  
    3. Mean reversion from extreme RSI levels
    
    Args:
        df: OHLCV price data with at least 40 bars for Fibonacci calculation
        trend: Current trend state from label_trend()
        
    Returns:
        EntryPlan with entry, stop, target prices and risk metrics, or None if no setup
        
    Risk Management:
        - Stops: Golden pocket boundary + ATR buffer, or 1.2*ATR from entry
        - Targets: Previous swing points or Fibonacci extensions
        - R-multiple calculated as (target-entry)/(entry-stop)
        
    Examples:
        >>> df = load_ohlcv("AAPL", "30m", 30)
        >>> trend = label_trend(df)
        >>> entry = build_entry(df, trend)
        >>> if entry:
        ...     print(f"Entry: {entry.side} @ {entry.entry_price}")
    """
```

### 2. Architecture Improvements

#### 2.1 Database Schema Evolution - MEDIUM PRIORITY

**Current State**: Recently centralized with SQLAlchemy ORM, good foundation.

**Additional Findings**:
```python
# models_db.py - Good practices observed
class Signal(Base):
    __tablename__ = "signals"
    # Proper use of SQLAlchemy constraints and types
    # JSON fields handled correctly with hybrid properties
    
class VectorStore(Base):
    __tablename__ = "vec_store"
    # Appropriate indexes for performance
```

**Recommendations**:
1. **Migration Strategy**: Add schema versioning for future changes
2. **Performance Indexes**: Consider composite indexes for common query patterns
3. **Data Validation**: Add database-level constraints matching Pydantic models

```python
# Add to models_db.py
__version__ = "1.6.1"

class Migration(Base):
    __tablename__ = "migrations"
    version = Column(String, primary_key=True)
    applied_at = Column(DateTime, default=datetime.utcnow)
    description = Column(String)
```

#### 2.2 Configuration Management - HIGH PRIORITY

**Issue**: Configuration scattered across multiple files and environment variables.

**Current Pattern**:
```python
# agent.py
model_name = os.getenv("SWING_LLM_MODEL", "gpt-4o-mini")

# Multiple hardcoded defaults throughout codebase
interval: str = "30m"
lookback_days: int = 30
```

**Recommendation**: Centralized configuration system:
```python
# config.py
from pydantic import BaseSettings

class SwingAgentConfig(BaseSettings):
    # Database
    database_url: str = "sqlite:///data/swing_agent.sqlite"
    
    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_enabled: bool = True
    openai_api_key: str = ""
    
    # Trading
    default_interval: str = "30m"
    default_lookback_days: int = 30
    default_sector: str = "XLK"
    
    # Technical Analysis Thresholds
    ema_slope_threshold: float = 0.01
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    atr_stop_multiplier: float = 1.2
    
    class Config:
        env_prefix = "SWING_"
        env_file = ".env"

# Usage throughout codebase
config = SwingAgentConfig()
```

#### 2.3 Error Recovery and Resilience - HIGH PRIORITY

**Issue**: Limited error handling could cause silent failures in production.

**Current Gaps Identified**:
```python
# llm_predictor.py - No error handling
def llm_extra_prediction(**features) -> LlmVote:
    agent = _make_agent(model_name, sys, LlmVote)
    res = agent.run(user_message="...", input=features)
    return res.data  # Could raise unhandled exceptions

# data.py - Basic error handling but could be improved
def load_ohlcv(symbol: str, interval: str = "30m", lookback_days: int = 30):
    # Handles empty data but not network timeouts, rate limits, etc.
```

**Recommendation**: Comprehensive error handling strategy:
```python
# errors.py
from enum import Enum
from typing import Optional
import logging

class ErrorSeverity(Enum):
    LOW = "low"       # Degraded functionality, continue operation
    MEDIUM = "medium" # Significant impact, log warning, use fallback
    HIGH = "high"     # Critical failure, raise exception
    CRITICAL = "critical" # System failure, immediate attention

class SwingAgentError(Exception):
    def __init__(self, message: str, severity: ErrorSeverity, component: str, 
                 original_error: Optional[Exception] = None):
        super().__init__(message)
        self.severity = severity
        self.component = component
        self.original_error = original_error
        
        # Log based on severity
        logger = logging.getLogger(f"swing_agent.{component}")
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(message, exc_info=original_error)
        elif severity == ErrorSeverity.HIGH:
            logger.error(message, exc_info=original_error)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(message)
        else:
            logger.info(message)

# Enhanced error handling example
def safe_llm_prediction(**features) -> Optional[LlmVote]:
    try:
        agent = _make_agent(model_name, sys, LlmVote)
        res = agent.run(user_message="...", input=features)
        return res.data
    except openai.RateLimitError as e:
        raise SwingAgentError(
            "LLM rate limit exceeded, consider request throttling",
            ErrorSeverity.MEDIUM, "llm", e
        )
    except openai.AuthenticationError as e:
        raise SwingAgentError(
            "LLM authentication failed, check API key",
            ErrorSeverity.HIGH, "llm", e
        )
    except Exception as e:
        raise SwingAgentError(
            f"Unexpected LLM error: {e}",
            ErrorSeverity.MEDIUM, "llm", e
        )
```

#### 2.1 Dependency Injection - MEDIUM PRIORITY

**Issue**: Tight coupling between components makes testing difficult.

**Current Pattern**:
```python
class SwingAgent:
    def __init__(self, ...):
        # Hardcoded dependencies
        self.data_loader = load_ohlcv
        self.llm_enabled = use_llm
```

**Recommendation**: Use dependency injection:
```python
from abc import ABC, abstractmethod

class DataProvider(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        pass

class LLMProvider(ABC):
    @abstractmethod  
    def get_prediction(self, **features) -> Optional[LlmVote]:
        pass

class YahooDataProvider(DataProvider):
    def fetch_ohlcv(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        return load_ohlcv(symbol, interval, days)

class OpenAIProvider(LLMProvider):
    def get_prediction(self, **features) -> Optional[LlmVote]:
        return llm_extra_prediction(**features)

class SwingAgent:
    def __init__(
        self, 
        data_provider: DataProvider,
        llm_provider: Optional[LLMProvider] = None,
        **config
    ):
        self.data_provider = data_provider
        self.llm_provider = llm_provider
        self.config = TradingConfig(**config)
```

**Benefits**:
- Easy unit testing with mock providers
- Support for multiple data sources
- Runtime provider switching
- Clear interface contracts

#### 2.2 Configuration Management - HIGH PRIORITY

**Issue**: No centralized configuration system.

**Recommendation**: Implement hierarchical configuration:
```python
from dataclasses import dataclass, field
from typing import Optional
import os
import yaml

@dataclass
class DatabaseConfig:
    signals_path: str = "data/signals.sqlite"
    vectors_path: str = "data/vec_store.sqlite"
    backup_enabled: bool = True
    backup_interval_hours: int = 24

@dataclass 
class LLMConfig:
    enabled: bool = True
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    
@dataclass
class TradingConfig:
    default_interval: str = "30m"
    default_lookback_days: int = 30
    max_hold_days: float = 2.0
    # ... other trading parameters

@dataclass
class SwingAgentConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SwingAgentConfig':
        """Load configuration from YAML file."""
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> 'SwingAgentConfig':
        """Load configuration from environment variables."""
        llm_config = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("SWING_LLM_MODEL", "gpt-4o-mini")
        )
        return cls(llm=llm_config)

# Usage
config = SwingAgentConfig.from_env()
agent = SwingAgent(config=config)
```

#### 2.3 Event-Driven Architecture - LOW PRIORITY

**Issue**: Monolithic signal generation process.

**Recommendation**: Consider event-driven pattern for extensibility:
```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class SignalEvent:
    symbol: str
    timestamp: str
    event_type: str
    data: dict

class EventHandler(Protocol):
    def handle(self, event: SignalEvent) -> None: ...

class SignalPipeline:
    def __init__(self):
        self.handlers = []
    
    def add_handler(self, handler: EventHandler):
        self.handlers.append(handler)
    
    def process(self, symbol: str) -> TradeSignal:
        events = []
        
        # Emit events at each stage
        data_event = SignalEvent(symbol, now(), "data_loaded", {"df": df})
        for handler in self.handlers:
            handler.handle(data_event)
            
        # Continue through pipeline...
```

### 3. Performance Optimizations

#### 3.1 Vector Store Indexing - HIGH PRIORITY

**Issue**: KNN search loads all vectors into memory and calculates similarities.

**Current Implementation**:
```python
def knn(db_path, query_vec, k=50, symbol=None):
    with sqlite3.connect(db_path) as con:
        if symbol:
            rows = con.execute("SELECT * FROM vec_store WHERE symbol=?", (symbol,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM vec_store").fetchall()  # Loads everything!
    
    # Calculate similarities for all rows
    for row in rows:
        vec = np.array(json.loads(row[4]))
        similarity = cosine(query_vec, vec)
```

**Recommendation**: Implement approximate nearest neighbors:
```python
# Option 1: Use Faiss for large-scale similarity search
import faiss

class FaissVectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index
        self.metadata = []
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[dict]):
        self.index.add(vectors.astype('float32'))
        self.metadata.extend(metadata)
    
    def search(self, query: np.ndarray, k: int) -> List[dict]:
        similarities, indices = self.index.search(query.reshape(1, -1).astype('float32'), k)
        return [self.metadata[i] for i in indices[0]]

# Option 2: Pre-compute embeddings and use database indexes
class OptimizedVectorStore:
    def add_vector_with_hash(self, vec: np.ndarray, metadata: dict):
        # Create locality-sensitive hash for approximate matching
        vec_hash = self._hash_vector(vec)
        
        # Store with spatial index
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                INSERT INTO vec_store_indexed 
                (id, vec_hash, vec_json, metadata_json) 
                VALUES (?, ?, ?, ?)
            """, (metadata['id'], vec_hash, json.dumps(vec.tolist()), json.dumps(metadata)))
    
    def _hash_vector(self, vec: np.ndarray, num_bits: int = 64) -> str:
        # Simple random projection hash
        random_vectors = np.random.randn(len(vec), num_bits)
        hash_bits = np.dot(vec, random_vectors) > 0
        return ''.join(['1' if bit else '0' for bit in hash_bits])
```

#### 3.2 Data Caching - MEDIUM PRIORITY

**Issue**: Redundant API calls for same data.

**Recommendation**: Implement multi-level caching:
```python
import redis
from functools import wraps
from datetime import datetime, timedelta

class DataCache:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        self.memory_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def get_ohlcv(self, symbol: str, interval: str, lookback_days: int) -> Optional[pd.DataFrame]:
        cache_key = f"ohlcv:{symbol}:{interval}:{lookback_days}"
        
        # Try Redis first
        if self.redis_client:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pd.read_json(cached_data)
        
        # Try memory cache
        if cache_key in self.memory_cache:
            data, timestamp = self.memory_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return data
        
        return None
    
    def set_ohlcv(self, symbol: str, interval: str, lookback_days: int, df: pd.DataFrame):
        cache_key = f"ohlcv:{symbol}:{interval}:{lookback_days}"
        
        # Store in Redis
        if self.redis_client:
            self.redis_client.setex(cache_key, self.cache_ttl, df.to_json())
        
        # Store in memory
        self.memory_cache[cache_key] = (df, datetime.now())

def cached_load_ohlcv(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    cache = DataCache()
    
    # Try cache first
    df = cache.get_ohlcv(symbol, interval, lookback_days)
    if df is not None:
        return df
    
    # Fetch and cache
    df = load_ohlcv(symbol, interval, lookback_days)
    cache.set_ohlcv(symbol, interval, lookback_days, df)
    return df
```

#### 3.3 Database Optimization - MEDIUM PRIORITY

**Current Schema Issues**:
- No indexes on commonly queried columns
- TEXT storage for JSON data (should use JSON type in newer SQLite)
- No partitioning for large datasets

**Recommendations**:
```sql
-- Add indexes for better query performance
CREATE INDEX idx_signals_symbol_asof ON signals(symbol, asof);
CREATE INDEX idx_signals_evaluated ON signals(evaluated) WHERE evaluated = 0;
CREATE INDEX idx_signals_timeframe ON signals(timeframe);
CREATE INDEX idx_vectors_symbol_ts ON vec_store(symbol, ts_utc);

-- Use JSON columns for better performance (SQLite 3.38+)
ALTER TABLE signals ADD COLUMN llm_vote_json_typed JSON;
UPDATE signals SET llm_vote_json_typed = json(llm_vote_json);

-- Partition large tables by date
CREATE TABLE signals_2024 (
    CHECK (asof >= '2024-01-01' AND asof < '2025-01-01')
) INHERITS (signals);
```

#### 3.4 Parallel Processing - LOW PRIORITY

**Issue**: Sequential processing for multiple symbols.

**Recommendation**: Add concurrent processing:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelSwingAgent:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.base_agent = SwingAgent()
    
    def analyze_symbols(self, symbols: List[str]) -> Dict[str, TradeSignal]:
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.base_agent.analyze, symbol): symbol 
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signal = future.result(timeout=30)
                    results[symbol] = signal
                except Exception as e:
                    logging.error(f"Failed to analyze {symbol}: {e}")
                    results[symbol] = None
        
        return results

# Usage
agent = ParallelSwingAgent(max_workers=8)
signals = agent.analyze_symbols(["AAPL", "MSFT", "GOOGL", "TSLA"])
```

### 4. Testing Infrastructure

#### 4.1 Unit Testing - HIGH PRIORITY

**Issue**: No visible test infrastructure.

**Recommendation**: Implement comprehensive test suite:
```python
# tests/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from swing_agent.indicators import ema, rsi, atr, fibonacci_range

class TestIndicators:
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        # Generate realistic price series
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.randn(100) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(100)) * 0.002),
            'low': prices * (1 - np.abs(np.random.randn(100)) * 0.002),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_ema_calculation(self, sample_data):
        """Test EMA calculation correctness."""
        close_prices = sample_data['close']
        ema_20 = ema(close_prices, 20)
        
        # Basic validations
        assert len(ema_20) == len(close_prices)
        assert not ema_20.isna().all()
        assert ema_20.iloc[-1] > 0
        
        # EMA should be less volatile than raw prices
        ema_volatility = ema_20.pct_change().std()
        price_volatility = close_prices.pct_change().std()
        assert ema_volatility < price_volatility
    
    def test_rsi_bounds(self, sample_data):
        """Test RSI stays within bounds."""
        rsi_values = rsi(sample_data['close'], 14)
        
        # RSI should be between 0 and 100
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
        
        # Should have valid values after warmup period
        assert not rsi_values.iloc[20:].isna().any()
    
    def test_fibonacci_range_properties(self, sample_data):
        """Test Fibonacci calculation properties."""
        fib = fibonacci_range(sample_data, lookback=20)
        
        # Basic structure tests
        assert fib.start < fib.end or fib.start > fib.end  # Valid swing
        assert 0.618 in [float(k) for k in fib.levels.keys()]
        assert fib.golden_low <= fib.golden_high
        
        # Level ordering tests
        if fib.dir_up:
            assert fib.levels["0.236"] < fib.levels["0.618"]
            assert fib.levels["1.0"] < fib.levels["1.272"]

# tests/test_strategy.py
class TestStrategy:
    def test_trend_labeling_consistency(self, sample_data):
        """Test trend labeling logic."""
        from swing_agent.strategy import label_trend
        
        trend = label_trend(sample_data)
        
        # Trend should be one of valid labels
        valid_labels = ["strong_up", "up", "sideways", "down", "strong_down"]
        assert trend.label in valid_labels
        
        # RSI should be reasonable
        assert 0 <= trend.rsi_14 <= 100
        
        # Price vs EMA should be boolean
        assert isinstance(trend.price_above_ema, bool)
    
    def test_entry_plan_risk_reward(self, sample_data):
        """Test entry plan risk/reward calculations."""
        from swing_agent.strategy import label_trend, build_entry
        
        trend = label_trend(sample_data)
        entry = build_entry(sample_data, trend)
        
        if entry:  # If entry plan generated
            # R-multiple should be positive
            assert entry.r_multiple > 0
            
            # Stop should be different from entry
            assert entry.stop_price != entry.entry_price
            
            # Target should be different from entry
            assert entry.take_profit != entry.entry_price
            
            # Long trade validation
            if entry.side == "long":
                assert entry.stop_price < entry.entry_price
                assert entry.take_profit > entry.entry_price
            
            # Short trade validation
            elif entry.side == "short":
                assert entry.stop_price > entry.entry_price
                assert entry.take_profit < entry.entry_price

# tests/test_integration.py
class TestIntegration:
    @pytest.mark.integration
    def test_full_signal_generation(self):
        """Test complete signal generation pipeline."""
        from swing_agent.agent import SwingAgent
        
        # Test with known good symbol
        agent = SwingAgent(
            interval="1d",
            lookback_days=30,
            use_llm=False,  # Skip LLM for integration test
            log_db=None,
            vec_db=None
        )
        
        signal = agent.analyze("AAPL")
        
        # Basic signal validation
        assert signal.symbol == "AAPL"
        assert signal.timeframe == "1d"
        assert signal.trend is not None
        assert 0 <= signal.confidence <= 1
        
        # If entry generated, validate structure
        if signal.entry:
            assert signal.entry.r_multiple > 0
            assert signal.entry.side in ["long", "short", "none"]

# Run tests
# pytest tests/ -v --cov=swing_agent --cov-report=html
```

#### 4.2 Property-Based Testing - MEDIUM PRIORITY

**Recommendation**: Use Hypothesis for robust testing:
```python
from hypothesis import given, strategies as st
import hypothesis.extra.pandas as hpd

@given(hpd.data_frames(
    columns=hpd.columns(['open', 'high', 'low', 'close', 'volume'], dtype=float),
    rows=st.integers(min_value=50, max_value=200)  # Enough data for indicators
))
def test_indicators_never_crash(df):
    """Property test: indicators should never crash on valid data."""
    # Ensure OHLC relationships are valid
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    df = df[df['close'] > 0]  # Positive prices only
    
    if len(df) >= 20:  # Minimum for indicators
        # These should not crash
        ema_result = ema(df['close'], 20)
        rsi_result = rsi(df['close'], 14)
        atr_result = atr(df, 14)
        
        # Basic sanity checks
        assert len(ema_result) == len(df)
        assert all(0 <= r <= 100 for r in rsi_result.dropna())
        assert all(atr_val >= 0 for atr_val in atr_result.dropna())
```

#### 4.3 Performance Testing - LOW PRIORITY

**Recommendation**: Add performance benchmarks:
```python
import time
import pytest
from swing_agent.agent import SwingAgent

class TestPerformance:
    @pytest.mark.performance
    def test_signal_generation_speed(self):
        """Signal generation should complete within reasonable time."""
        agent = SwingAgent(use_llm=False)
        
        start_time = time.time()
        signal = agent.analyze("AAPL")
        elapsed = time.time() - start_time
        
        # Should complete within 10 seconds
        assert elapsed < 10.0, f"Signal generation took {elapsed:.2f}s"
    
    @pytest.mark.performance
    def test_vector_store_performance(self):
        """Vector store operations should be fast."""
        # Test with 1000 vectors
        vectors = np.random.rand(1000, 10)
        
        start_time = time.time()
        # Add vectors and search
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0, f"Vector operations took {elapsed:.2f}s"
```

### 5. Operational Improvements

#### 5.1 Logging and Monitoring - HIGH PRIORITY

**Issue**: No structured logging or monitoring.

**Recommendation**: Implement comprehensive observability:
```python
import logging
import structlog
from prometheus_client import Counter, Histogram, start_http_server

# Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Metrics
SIGNALS_GENERATED = Counter('swingagent_signals_total', 'Total signals generated', ['symbol', 'timeframe'])
SIGNAL_GENERATION_TIME = Histogram('swingagent_signal_duration_seconds', 'Signal generation time')
LLM_CALLS = Counter('swingagent_llm_calls_total', 'LLM API calls', ['model', 'status'])
DATABASE_OPERATIONS = Histogram('swingagent_db_operation_duration_seconds', 'Database operation time', ['operation'])

class MonitoredSwingAgent(SwingAgent):
    def analyze(self, symbol: str) -> TradeSignal:
        with SIGNAL_GENERATION_TIME.time():
            logger.info("Starting signal analysis", symbol=symbol, timeframe=self.interval)
            
            try:
                signal = super().analyze(symbol)
                
                SIGNALS_GENERATED.labels(symbol=symbol, timeframe=self.interval).inc()
                logger.info(
                    "Signal generated successfully",
                    symbol=symbol,
                    trend=signal.trend.label,
                    confidence=signal.confidence,
                    entry_side=signal.entry.side if signal.entry else None
                )
                
                return signal
                
            except Exception as e:
                logger.error("Signal generation failed", symbol=symbol, error=str(e), exc_info=True)
                raise

# Start metrics server
start_http_server(8000)
```

#### 5.2 Health Checks - MEDIUM PRIORITY

**Recommendation**: Implement health monitoring:
```python
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    details: Dict = None

class HealthMonitor:
    def __init__(self, agent: SwingAgent):
        self.agent = agent
    
    def check_data_connectivity(self) -> HealthCheck:
        """Check if market data is accessible."""
        try:
            df = load_ohlcv("AAPL", "1d", 1)
            if len(df) > 0:
                return HealthCheck("data_connectivity", HealthStatus.HEALTHY, "Market data accessible")
            else:
                return HealthCheck("data_connectivity", HealthStatus.UNHEALTHY, "No market data received")
        except Exception as e:
            return HealthCheck("data_connectivity", HealthStatus.UNHEALTHY, f"Data fetch failed: {e}")
    
    def check_database_connectivity(self) -> HealthCheck:
        """Check database accessibility."""
        try:
            if self.agent.log_db:
                with sqlite3.connect(self.agent.log_db) as conn:
                    conn.execute("SELECT 1").fetchone()
                return HealthCheck("database", HealthStatus.HEALTHY, "Database accessible")
            else:
                return HealthCheck("database", HealthStatus.DEGRADED, "Database not configured")
        except Exception as e:
            return HealthCheck("database", HealthStatus.UNHEALTHY, f"Database error: {e}")
    
    def check_llm_connectivity(self) -> HealthCheck:
        """Check LLM service availability."""
        if not self.agent.use_llm:
            return HealthCheck("llm", HealthStatus.DEGRADED, "LLM disabled")
        
        try:
            # Simple LLM test
            result = llm_extra_prediction(symbol="TEST", price=100.0, trend_label="up")
            return HealthCheck("llm", HealthStatus.HEALTHY, "LLM accessible")
        except Exception as e:
            return HealthCheck("llm", HealthStatus.DEGRADED, f"LLM error: {e}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        checks = [
            self.check_data_connectivity(),
            self.check_database_connectivity(), 
            self.check_llm_connectivity()
        ]
        
        # Determine overall status
        statuses = [check.status for check in checks]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return {
            "status": overall.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "details": check.details
                }
                for check in checks
            ]
        }

# Health check endpoint (if using web framework)
from flask import Flask, jsonify

app = Flask(__name__)
health_monitor = HealthMonitor(agent)

@app.route('/health')
def health_check():
    return jsonify(health_monitor.get_health_report())
```

#### 5.3 Graceful Degradation - MEDIUM PRIORITY

**Recommendation**: Implement fallback strategies:
```python
class RobustSwingAgent(SwingAgent):
    def analyze(self, symbol: str) -> TradeSignal:
        """Analyze with graceful degradation."""
        try:
            return super().analyze(symbol)
        except Exception as e:
            logger.warning(f"Full analysis failed for {symbol}: {e}")
            return self._generate_fallback_signal(symbol)
    
    def _generate_fallback_signal(self, symbol: str) -> TradeSignal:
        """Generate basic signal without advanced features."""
        try:
            # Try with minimal data
            df = load_ohlcv(symbol, self.interval, 7)  # Reduced lookback
            trend = label_trend(df)
            
            return TradeSignal(
                symbol=symbol,
                timeframe=self.interval,
                asof=datetime.utcnow().isoformat(),
                trend=trend,
                confidence=0.1,  # Low confidence for fallback
                reasoning="Fallback signal - limited analysis available"
            )
        except Exception as e:
            logger.error(f"Fallback signal generation failed: {e}")
            # Return minimal signal
            return TradeSignal(
                symbol=symbol,
                timeframe=self.interval,
                asof=datetime.utcnow().isoformat(),
                trend=TrendState(
                    label=TrendLabel.SIDEWAYS,
                    ema_slope=0.0,
                    price_above_ema=False,
                    rsi_14=50.0
                ),
                confidence=0.0,
                reasoning="System degraded - no analysis available"
            )
```

### 6. Security Hardening

#### 6.1 Input Validation - HIGH PRIORITY

**Issue**: Limited validation of user inputs.

**Recommendation**: Implement comprehensive validation:
```python
import re
from typing import Union

class InputValidator:
    # Valid symbol patterns
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.]{1,10}$')
    
    # Valid intervals
    VALID_INTERVALS = {"15m", "30m", "1h", "1d"}
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate and sanitize trading symbol."""
        if not isinstance(symbol, str):
            raise ValueError("Symbol must be a string")
        
        symbol = symbol.upper().strip()
        
        if not InputValidator.SYMBOL_PATTERN.match(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        # Check against known bad patterns
        if symbol in ["", "NULL", "NONE", "TEST"]:
            raise ValueError(f"Invalid symbol: {symbol}")
        
        return symbol
    
    @staticmethod
    def validate_interval(interval: str) -> str:
        """Validate trading interval."""
        if interval not in InputValidator.VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {InputValidator.VALID_INTERVALS}")
        return interval
    
    @staticmethod
    def validate_lookback_days(days: Union[int, str]) -> int:
        """Validate lookback period."""
        try:
            days = int(days)
        except (ValueError, TypeError):
            raise ValueError("Lookback days must be an integer")
        
        if days < 1:
            raise ValueError("Lookback days must be positive")
        
        if days > 365:
            raise ValueError("Lookback days cannot exceed 365")
        
        return days

class ValidatedSwingAgent(SwingAgent):
    def analyze(self, symbol: str) -> TradeSignal:
        # Validate inputs
        symbol = InputValidator.validate_symbol(symbol)
        self.interval = InputValidator.validate_interval(self.interval)
        self.lookback_days = InputValidator.validate_lookback_days(self.lookback_days)
        
        return super().analyze(symbol)
```

#### 6.2 SQL Injection Prevention - MEDIUM PRIORITY

**Issue**: Some SQL queries use string formatting.

**Current Risky Pattern**:
```python
# In vectorstore.py (hypothetical risk)
query = f"SELECT * FROM vec_store WHERE symbol='{symbol}'"  # DON'T DO THIS
```

**Recommendation**: Use parameterized queries everywhere:
```python
class SecureVectorStore:
    def knn(self, db_path: str, query_vec: np.ndarray, k: int = 50, symbol: Optional[str] = None):
        with sqlite3.connect(db_path) as con:
            if symbol:
                # Parameterized query - GOOD
                rows = con.execute(
                    "SELECT * FROM vec_store WHERE symbol = ? ORDER BY ts_utc DESC LIMIT ?", 
                    (symbol, k * 10)  # Get more candidates for better similarity search
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM vec_store ORDER BY ts_utc DESC LIMIT ?", 
                    (k * 10,)
                ).fetchall()
        
        # Continue with similarity calculation...
```

#### 6.3 API Key Security - HIGH PRIORITY

**Issue**: API keys stored in environment with no rotation strategy.

**Recommendation**: Implement secure key management:
```python
import boto3
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

class SecretManager:
    def __init__(self, provider: str = "env"):
        self.provider = provider
        self._secrets_cache = {}
        self._cache_ttl = 3600  # 1 hour
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret with caching and rotation support."""
        if key in self._secrets_cache:
            value, timestamp = self._secrets_cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
        
        if self.provider == "aws":
            value = self._get_aws_secret(key)
        elif self.provider == "azure":
            value = self._get_azure_secret(key)
        else:
            value = os.getenv(key)
        
        if value:
            self._secrets_cache[key] = (value, time.time())
        
        return value
    
    def _get_aws_secret(self, key: str) -> Optional[str]:
        try:
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId=key)
            return response['SecretString']
        except Exception as e:
            logger.error(f"Failed to get AWS secret {key}: {e}")
            return None
    
    def _get_azure_secret(self, key: str) -> Optional[str]:
        try:
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url="https://your-vault.vault.azure.net/", credential=credential)
            secret = client.get_secret(key)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to get Azure secret {key}: {e}")
            return None

# Usage
secret_manager = SecretManager("aws")  # or "azure" or "env"
openai_key = secret_manager.get_secret("OPENAI_API_KEY")
```

#### 6.4 Rate Limiting - MEDIUM PRIORITY

**Recommendation**: Implement API rate limiting:
```python
import time
from collections import defaultdict, deque
from threading import Lock

class RateLimiter:
    def __init__(self, max_calls_per_minute: int = 60):
        self.max_calls = max_calls_per_minute
        self.calls = defaultdict(deque)
        self.lock = Lock()
    
    def wait_if_needed(self, key: str = "default"):
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Remove old calls
            while self.calls[key] and self.calls[key][0] < minute_ago:
                self.calls[key].popleft()
            
            # Check if we're at limit
            if len(self.calls[key]) >= self.max_calls:
                wait_time = 60 - (now - self.calls[key][0])
                if wait_time > 0:
                    time.sleep(wait_time)
                    # Recursive call after waiting
                    return self.wait_if_needed(key)
            
            # Record this call
            self.calls[key].append(now)

class RateLimitedLLMProvider:
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    def get_prediction(self, **features) -> Optional[LlmVote]:
        self.rate_limiter.wait_if_needed("openai")
        return llm_extra_prediction(**features)
```

### 7. Data Quality and Validation

#### 7.1 Market Data Validation - HIGH PRIORITY

**Issue**: No validation of data quality from Yahoo Finance.

**Recommendation**: Implement data quality checks:
```python
class DataQualityChecker:
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean OHLCV data."""
        original_len = len(df)
        
        # 1. Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # 2. Remove rows with invalid OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Removing {invalid_ohlc.sum()} rows with invalid OHLC for {symbol}")
            df = df[~invalid_ohlc]
        
        # 3. Remove rows with zero or negative prices
        invalid_prices = (
            (df['open'] <= 0) |
            (df['high'] <= 0) |
            (df['low'] <= 0) |
            (df['close'] <= 0)
        )
        
        if invalid_prices.any():
            logger.warning(f"Removing {invalid_prices.sum()} rows with invalid prices for {symbol}")
            df = df[~invalid_prices]
        
        # 4. Check for extreme price movements (> 50% in one bar)
        price_changes = df['close'].pct_change().abs()
        extreme_moves = price_changes > 0.5
        
        if extreme_moves.any():
            logger.warning(f"Found {extreme_moves.sum()} extreme price movements for {symbol}")
            # Don't remove, but flag for investigation
        
        # 5. Check for gaps in time series
        if hasattr(df.index, 'freq') and df.index.freq:
            expected_len = (df.index[-1] - df.index[0]) / df.index.freq + 1
            if len(df) < expected_len * 0.9:  # Allow 10% missing data
                logger.warning(f"Potential data gaps in {symbol}: {len(df)} vs expected {expected_len}")
        
        # 6. Minimum data length check
        if len(df) < 20:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} bars")
        
        logger.info(f"Data validation complete for {symbol}: {original_len} -> {len(df)} rows")
        return df

def validated_load_ohlcv(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    """Load and validate OHLCV data."""
    df = load_ohlcv(symbol, interval, lookback_days)
    return DataQualityChecker.validate_ohlcv(df, symbol)
```

#### 7.2 Signal Validation - MEDIUM PRIORITY

**Recommendation**: Validate generated signals:
```python
class SignalValidator:
    @staticmethod
    def validate_signal(signal: TradeSignal) -> TradeSignal:
        """Validate signal consistency and quality."""
        
        # 1. Basic field validation
        if not signal.symbol:
            raise ValueError("Signal missing symbol")
        
        if not signal.asof:
            raise ValueError("Signal missing timestamp")
        
        # 2. Trend validation
        if signal.trend.rsi_14 < 0 or signal.trend.rsi_14 > 100:
            logger.warning(f"Invalid RSI value: {signal.trend.rsi_14}")
            signal.trend.rsi_14 = max(0, min(100, signal.trend.rsi_14))
        
        # 3. Entry plan validation
        if signal.entry:
            entry = signal.entry
            
            # Risk-reward validation
            if entry.r_multiple <= 0:
                logger.warning(f"Invalid R-multiple: {entry.r_multiple}")
                signal.confidence *= 0.5  # Reduce confidence
            
            # Price relationship validation
            if entry.side == "long":
                if entry.stop_price >= entry.entry_price:
                    logger.error("Long trade: stop >= entry price")
                    signal.entry = None  # Invalidate entry
                
                if entry.take_profit <= entry.entry_price:
                    logger.error("Long trade: target <= entry price")
                    signal.entry = None
            
            elif entry.side == "short":
                if entry.stop_price <= entry.entry_price:
                    logger.error("Short trade: stop <= entry price")
                    signal.entry = None
                
                if entry.take_profit >= entry.entry_price:
                    logger.error("Short trade: target >= entry price") 
                    signal.entry = None
        
        # 4. Confidence validation
        signal.confidence = max(0.0, min(1.0, signal.confidence))
        
        # 5. Expected values validation
        if signal.expected_winrate:
            signal.expected_winrate = max(0.0, min(1.0, signal.expected_winrate))
        
        return signal

class ValidatedSwingAgent(SwingAgent):
    def analyze(self, symbol: str) -> TradeSignal:
        signal = super().analyze(symbol)
        return SignalValidator.validate_signal(signal)
```

### 8. Documentation and Maintenance

#### 8.1 API Documentation - MEDIUM PRIORITY

**Recommendation**: Generate comprehensive API docs:
```python
# Use Sphinx with autodoc for automatic documentation generation
# sphinx-apidoc -o docs/ src/swing_agent/

# Example enhanced docstring
def build_entry(df: pd.DataFrame, trend: TrendState) -> Optional[EntryPlan]:
    """
    Generate entry plan based on trend analysis and Fibonacci levels.
    
    This function implements three primary entry strategies:
    
    1. **Fibonacci Golden Pocket Pullbacks** (Highest Priority)
       - Triggers when price is within 0.618-0.65 retracement
       - Used for trend continuation trades
       - Stop: Below golden pocket + ATR buffer
       - Target: Previous swing extreme or 1.272 extension
    
    2. **Momentum Continuation Breakouts**
       - Triggers on break above/below previous high/low
       - Used when no Fibonacci setup available
       - Stop: 1.2 × ATR from entry
       - Target: 2.0 × ATR or Fibonacci extension
    
    3. **Mean Reversion from Extremes**
       - Triggers on RSI < 35 (oversold) or RSI > 65 (overbought)
       - Used in sideways markets only
       - Stop: 1.0 × ATR from entry
       - Target: 1.5 × ATR mean reversion
    
    Args:
        df: OHLCV DataFrame with minimum 40 bars for Fibonacci calculation.
            Must contain columns: ['open', 'high', 'low', 'close', 'volume']
        trend: TrendState object from label_trend() containing:
            - label: Current trend classification
            - ema_slope: EMA20 slope (risk-adjusted)
            - price_above_ema: Current price vs EMA20
            - rsi_14: Current RSI(14) value
    
    Returns:
        EntryPlan with entry/stop/target prices and risk metrics, or None if no
        valid setup found. EntryPlan includes:
        - side: "long", "short", or "none"
        - entry_price: Recommended entry price
        - stop_price: Stop loss price
        - take_profit: Take profit target
        - r_multiple: Risk-reward ratio (target-entry)/(entry-stop)
        - Fibonacci levels for reference
    
    Raises:
        ValueError: If DataFrame is too short (< 40 bars) or missing required columns
        TypeError: If trend parameter is not TrendState object
    
    Risk Management:
        - All stops include ATR-based buffer to avoid premature exit
        - R-multiples typically range from 1.0 to 3.0
        - Position sizing not included (handled externally)
    
    Examples:
        >>> # Basic usage
        >>> df = load_ohlcv("AAPL", "30m", 30)
        >>> trend = label_trend(df)
        >>> entry = build_entry(df, trend)
        >>> if entry and entry.side != "none":
        ...     print(f"Entry: {entry.side} @ ${entry.entry_price:.2f}")
        ...     print(f"Stop: ${entry.stop_price:.2f}")
        ...     print(f"Target: ${entry.take_profit:.2f}")
        ...     print(f"R-Multiple: {entry.r_multiple:.2f}")
        
        >>> # Risk validation
        >>> if entry and entry.r_multiple < 1.5:
        ...     print("Low reward-to-risk ratio, consider skipping")
    
    Notes:
        - Function assumes market data is clean and validated
        - Fibonacci calculations use 40-bar lookback by default
        - ATR calculations use 14-period default
        - All price levels are absolute values, not relative to current price
        
    See Also:
        label_trend(): Trend classification function
        fibonacci_range(): Fibonacci level calculation
        TrendState: Trend analysis data structure
        EntryPlan: Entry plan data structure
    """
```

#### 8.2 Developer Documentation - LOW PRIORITY

**Recommendation**: Create comprehensive developer guides:
```markdown
# Developer Guide

## Architecture Overview

SwingAgent follows a layered architecture:

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
│        Infrastructure              │
│  (Data Fetching + Database)         │
└─────────────────────────────────────┘
```

## Adding New Indicators

1. Add calculation function to `indicators.py`:
```python
def your_indicator(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate your custom indicator."""
    # Implementation here
    return result
```

2. Add to strategy logic in `strategy.py`:
```python
def label_trend(df: pd.DataFrame) -> TrendState:
    # Existing code...
    your_value = your_indicator(df['close'], 20).iloc[-1]
    # Use in trend logic...
```

3. Update feature vector in `features.py`:
```python
def build_setup_vector(...):
    # Add new feature
    your_feature = your_value / 100.0  # Normalize
    return np.array([..., your_feature])
```

## Testing Guidelines

- Unit tests for all new functions
- Integration tests for strategy changes  
- Property-based tests for mathematical functions
- Performance tests for optimization work

## Additional Code Review Findings (v1.6.1 Analysis)

### 10. Code Quality Assessment

#### 10.1 Positive Patterns Identified - MAINTAINED

**Modern Python Practices**:
```python
# Good use of type hints throughout
def label_trend(df: pd.DataFrame) -> TrendState:
def build_entry(df: pd.DataFrame, trend: TrendState) -> Optional[EntryPlan]:

# Proper use of Pydantic v2 models
class TradeSignal(BaseModel):
    symbol: str
    timeframe: Literal["15m", "30m", "1h", "1d"] = "30m"
    # Field validation and serialization

# Clean enum usage
class TrendLabel(str, Enum):
    STRONG_UP = "strong_up"
    UP = "up"
    SIDEWAYS = "sideways"
```

**Database Architecture**:
```python
# Recently improved SQLAlchemy integration
from .database import get_session
from .models_db import Signal, VectorStore

# Proper session management with context managers
with get_session() as session:
    signal = Signal(...)
    session.add(signal)
    session.commit()
```

#### 10.2 Areas Needing Attention - NEW FINDINGS

**Function Complexity**:
- `SwingAgent.analyze()` method: ~160 lines, multiple responsibilities
- `build_entry()` in strategy.py: ~50 lines with complex branching logic
- Vector similarity calculations could be optimized

**Error Handling Gaps**:
```python
# llm_predictor.py - Potential silent failures
def llm_extra_prediction(**features) -> LlmVote:
    # No try/catch for API failures, rate limits, or validation errors
    
# data.py - Basic handling but missing edge cases
def load_ohlcv(symbol: str, interval: str = "30m", lookback_days: int = 30):
    # Handles empty data but not delisted symbols, weekend calls, etc.
```

**Performance Considerations**:
```python
# vectorstore.py - Vector operations not optimized
def knn(db_path, query_vec, k=10):
    # Loading all vectors into memory for similarity calculation
    # Could benefit from approximate nearest neighbor algorithms
```

#### 10.3 Security Review - NEW SECTION

**API Key Management**:
- ✅ Environment variable usage: `os.getenv("OPENAI_API_KEY")`
- ⚠️ No key validation or rotation support
- ⚠️ No secrets masking in logs

**Database Security**:
- ✅ SQLAlchemy ORM prevents basic SQL injection
- ⚠️ No database connection encryption configuration
- ⚠️ No audit logging for data modifications

**Input Validation**:
```python
# Current validation patterns are good but could be enhanced
def load_ohlcv(symbol: str, interval: str = "30m", lookback_days: int = 30):
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    # Could add symbol format validation, reasonable lookback limits
```

#### 10.4 Documentation Quality - ASSESSMENT

**Strengths**:
- Comprehensive README with quickstart
- Well-structured docs/ directory
- API reference with examples
- Deployment guides for multiple environments

**Gaps Identified**:
- Missing inline docstrings for complex functions
- No architecture decision records (ADRs)
- Limited troubleshooting scenarios for common issues
- No performance tuning guide

### 11. Development Workflow Improvements

#### 11.1 Code Quality Tools - RECOMMENDED SETUP

**Linting Configuration** (pyproject.toml):
```toml
[tool.ruff]
line-length = 100
target-version = "py312"
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "N",  # flake8-naming
    "UP", # pyupgrade
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]

[tool.black]
line-length = 100
target-version = ['py312']

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
```

**Pre-commit Configuration** (.pre-commit-config.yaml):
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
  
  - repo: https://github.com/psf/black
    rev: 23.10.1
    hooks:
      - id: black
```

#### 11.2 Testing Infrastructure - IMPLEMENTATION PLAN

**Phase 1: Unit Tests** (High Priority)
```python
# tests/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from swing_agent.indicators import ema, rsi, atr, fibonacci_range

class TestIndicators:
    @pytest.fixture
    def sample_ohlcv(self):
        """Generate realistic OHLCV data for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.randn(100) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(100)) * 0.002),
            'low': prices * (1 - np.abs(np.random.randn(100)) * 0.002),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def test_ema_calculation(self, sample_ohlcv):
        """Test EMA calculation correctness."""
        ema_20 = ema(sample_ohlcv['close'], 20)
        
        # Basic validation
        assert len(ema_20) == len(sample_ohlcv)
        assert not ema_20.isna().all()
        assert ema_20.iloc[-1] > 0
        
        # EMA should be smooth (less volatile than price)
        price_volatility = sample_ohlcv['close'].pct_change().std()
        ema_volatility = ema_20.pct_change().std()
        assert ema_volatility < price_volatility
```

**Phase 2: Integration Tests** (Medium Priority)
```python
# tests/test_integration.py
class TestSwingAgentIntegration:
    def test_full_signal_generation(self, mock_data):
        """Test complete signal generation pipeline."""
        agent = SwingAgent(interval="30m", lookback_days=30, use_llm=False)
        signal = agent.analyze_df("AAPL", mock_data)
        
        # Validate signal structure
        assert signal.symbol == "AAPL"
        assert signal.trend.label in list(TrendLabel)
        assert 0 <= signal.confidence <= 1
        
        # If entry plan exists, validate risk/reward
        if signal.entry:
            assert signal.entry.r_multiple > 0
            assert signal.entry.stop_price != signal.entry.entry_price
```

### 12. Monitoring and Observability

#### 12.1 Logging Strategy - NEW RECOMMENDATION

**Structured Logging Implementation**:
```python
# logging_config.py
import logging
import json
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    def __init__(self, component: str):
        self.logger = logging.getLogger(f"swing_agent.{component}")
        self.component = component
    
    def log_signal_generated(self, symbol: str, signal: TradeSignal):
        self.logger.info("signal_generated", extra={
            "event_type": "signal_generated",
            "symbol": symbol,
            "trend_label": signal.trend.label.value,
            "entry_side": signal.entry.side.value if signal.entry else None,
            "confidence": signal.confidence,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def log_vector_search(self, query_type: str, results_count: int, search_time_ms: float):
        self.logger.info("vector_search", extra={
            "event_type": "vector_search",
            "query_type": query_type,
            "results_count": results_count,
            "search_time_ms": search_time_ms
        })
```

#### 12.2 Performance Metrics - NEW RECOMMENDATION

**Key Metrics to Track**:
```python
# metrics.py
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class PerformanceMetrics:
    signal_generation_time_ms: float
    vector_search_time_ms: float
    llm_response_time_ms: float
    database_query_time_ms: float
    total_analysis_time_ms: float

class MetricsCollector:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        # Implementation for performance tracking
```

### 13. Production Readiness Checklist

#### 13.1 Environment Configuration - NEW SECTION

**Development Environment**:
```bash
# .env.development
SWING_DATABASE_URL=sqlite:///data/swing_agent_dev.sqlite
SWING_LLM_MODEL=gpt-4o-mini
SWING_LOG_LEVEL=DEBUG
SWING_ENABLE_METRICS=true
```

**Production Environment**:
```bash
# .env.production
SWING_DATABASE_URL=postgresql://user:pass@db:5432/swing_agent
SWING_LLM_MODEL=gpt-4o
SWING_LOG_LEVEL=INFO
SWING_ENABLE_METRICS=true
SWING_RATE_LIMIT_ENABLED=true
```

#### 13.2 Deployment Considerations - NEW SECTION

**Docker Configuration**:
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app/src
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python scripts/health_check.py

CMD ["python", "scripts/run_swing_agent.py"]
```

**Health Check Implementation**:
```python
# scripts/health_check.py
def health_check() -> bool:
    """Verify system health for monitoring."""
    try:
        # Test database connection
        with get_session() as session:
            session.execute("SELECT 1")
        
        # Test data provider
        load_ohlcv("SPY", "1d", 1)
        
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
```
```

### 9. Future Enhancements

#### 9.1 Machine Learning Improvements - LOW PRIORITY

**Current State**: Basic cosine similarity with manual feature engineering.

**Recommendations**:
1. **Advanced ML Models**: Random Forest or XGBoost for pattern classification
2. **Feature Learning**: Autoencoders for automatic feature discovery  
3. **Time Series Models**: LSTM/GRU for sequence modeling
4. **Ensemble Methods**: Combine multiple models for better predictions

#### 9.2 Real-time Processing - LOW PRIORITY

**Current State**: Batch processing via scripts.

**Recommendations**:
1. **Streaming Data**: Apache Kafka for real-time price feeds
2. **Event Processing**: Apache Flink or Spark Streaming
3. **Websocket APIs**: Real-time signal delivery
4. **Live Trading**: Integration with broker APIs

#### 9.3 Portfolio Management - LOW PRIORITY

**Current State**: Individual signal analysis.

**Recommendations**:
1. **Position Sizing**: Kelly criterion or fixed fractional
2. **Portfolio Risk**: Correlation analysis and diversification
3. **Drawdown Control**: Dynamic position sizing based on equity curve
4. **Multi-timeframe Coordination**: Coordinate signals across timeframes

## Implementation Priority

### Phase 1: Critical Fixes (4-6 weeks)
1. ✅ Function complexity reduction (high impact, medium effort)
2. ✅ Configuration management (high impact, low effort)  
3. ✅ Error handling improvements (high impact, medium effort)
4. ✅ Input validation (high impact, low effort)
5. ✅ Basic testing infrastructure (high impact, high effort)

### Phase 2: Performance & Operations (6-8 weeks)  
1. ✅ Vector store optimization (medium impact, medium effort)
2. ✅ Data caching (medium impact, low effort)
3. ✅ Logging and monitoring (medium impact, medium effort)
4. ✅ Health checks (medium impact, low effort)
5. ✅ Database optimization (medium impact, medium effort)

### Phase 3: Advanced Features (8-12 weeks)
1. ✅ Advanced ML models (low impact, high effort)
2. ✅ Real-time processing (low impact, very high effort)
3. ✅ Portfolio management (low impact, high effort)
4. ✅ API documentation (low impact, medium effort)

## Conclusion

The SwingAgent system demonstrates solid foundations in quantitative trading strategy development. The main areas for improvement focus on:

1. **Code maintainability** through better structure and testing
2. **Operational robustness** through monitoring and error handling  
3. **Performance optimization** through caching and database improvements
4. **Security hardening** through input validation and secret management

The recommended changes maintain the core strengths of the system while addressing operational challenges that would arise in production deployment. The phased approach allows for incremental improvements while maintaining system stability.