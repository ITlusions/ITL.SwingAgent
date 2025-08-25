# Configuration Guide

This guide covers all configuration options for the SwingAgent system.

## Environment Setup

### Required Environment Variables

```bash
# OpenAI API Configuration (required for LLM features)
export OPENAI_API_KEY="sk-..."                # Your OpenAI API key
export SWING_LLM_MODEL="gpt-4o-mini"          # LLM model selection

# Optional: Centralized database path  
export SWING_DATABASE_URL="sqlite:///data/swing_agent.sqlite"

# Legacy environment variables (for migration)
export SWING_SIGNALS_DB="data/signals.sqlite"  
export SWING_VECTOR_DB="data/vec_store.sqlite"
```

### Supported LLM Models

- `gpt-4o-mini` (recommended): Cost-effective, fast responses
- `gpt-4o`: Higher quality, more expensive
- `gpt-4`: Legacy model, good performance
- `gpt-3.5-turbo`: Fastest, lowest cost

### Installation

```bash
# Basic installation
pip install -e .

# Development installation with optional dependencies
pip install -e ".[dev,test]"
```

## Agent Configuration

### SwingAgent Parameters

```python
from swing_agent.agent import SwingAgent

agent = SwingAgent(
    interval="30m",              # Trading timeframe
    lookback_days=30,            # Historical data period
    log_db="data/signals.sqlite", # Signal storage database
    vec_db="data/vec_store.sqlite", # Vector store database
    use_llm=True,                # Enable LLM integration
    llm_extras=True,             # Enable additional LLM features
    sector_symbol="XLK"          # Sector ETF for relative strength
)
```

**Parameter Details:**

#### `interval` (str)
Trading timeframe for analysis.
- **Options**: "15m", "30m", "1h", "1d"
- **Default**: "30m"
- **Impact**: Affects bar counts, holding time calculations, session buckets

#### `lookback_days` (int)  
Historical data period for analysis.
- **Range**: 7-365 days
- **Default**: 30
- **Recommendations**:
  - 15m/30m: 30-60 days
  - 1h: 60-90 days  
  - 1d: 180-365 days

#### `log_db` (str | None)
Path to centralized database file.
- **Default**: "data/swing_agent.sqlite" (centralized database)
- **Format**: "path/to/swing_agent.sqlite" 
- **Auto-creation**: Database and tables created if missing
- **Note**: Same file as vec_db in centralized setup

#### `vec_db` (str | None)
Path to centralized database file (same as log_db).
- **Default**: "data/swing_agent.sqlite" (centralized database)
- **Format**: "path/to/swing_agent.sqlite"
- **Impact**: Enables historical pattern matching and statistical expectations
- **Note**: Same file as log_db in centralized setup

#### `use_llm` (bool)
Enable OpenAI LLM integration.
- **Default**: True
- **Requirements**: OPENAI_API_KEY environment variable
- **Impact**: Provides trend confirmation and explanations

#### `llm_extras` (bool)
Enable additional LLM features.
- **Default**: True
- **Impact**: Generates action plans and scenario analysis
- **Cost**: Increases API usage

#### `sector_symbol` (str)
Sector ETF for relative strength analysis.
- **Default**: "XLK" (Technology)
- **Fallback**: SPY if sector ETF fails
- **Popular Options**:
  - XLK: Technology
  - XLF: Financials
  - XLE: Energy
  - XLV: Healthcare
  - QQQ: Nasdaq 100

## Technical Indicator Configuration

### EMA Settings

EMA calculations use hardcoded parameters that can be customized:

```python
# In strategy.py - trend labeling
ema_period = 20              # EMA period for trend
slope_lookback = 6           # Bars for slope calculation
slope_threshold_up = 0.01    # Minimum slope for up trend
slope_threshold_strong = 0.02 # Minimum slope for strong trend
```

### RSI Settings

```python
# In indicators.py
rsi_period = 14              # RSI calculation period
rsi_oversold = 35           # Oversold threshold for mean reversion
rsi_overbought = 65         # Overbought threshold for mean reversion
rsi_trend_up = 60           # Minimum RSI for up trend
rsi_trend_down = 40         # Maximum RSI for down trend
```

### ATR Settings

```python
# In indicators.py and strategy.py
atr_period = 14             # ATR calculation period
atr_stop_multiplier = 1.2   # ATR multiplier for stops
atr_buffer = 0.2            # ATR buffer for Fibonacci stops
atr_target_multiplier = 2.0 # ATR multiplier for targets
```

### Fibonacci Settings

```python
# In indicators.py
fib_lookback = 40           # Bars to search for swing high/low
golden_pocket_low = 0.618   # Lower bound of golden pocket
golden_pocket_high = 0.65   # Upper bound of golden pocket

# Available levels
fib_levels = {
    "0.236": 0.236,
    "0.382": 0.382, 
    "0.5": 0.5,
    "0.618": 0.618,
    "0.65": 0.65,
    "0.786": 0.786,
    "1.0": 1.0,
    "1.272": 1.272,         # Primary target
    "1.414": 1.414,
    "1.618": 1.618          # Extended target
}
```

## Vector Store Configuration

### Feature Vector Components

```python
# In features.py - build_setup_vector
vector_components = [
    "trend_up",              # Binary: 1 if UP/STRONG_UP
    "trend_down",            # Binary: 1 if DOWN/STRONG_DOWN  
    "trend_sideways",        # Binary: 1 if SIDEWAYS
    "rsi_normalized",        # RSI / 100
    "price_vs_ema",          # (price - ema) / price
    "fib_position",          # Position within golden pocket
    "in_golden_pocket",      # Binary: 1 if in golden pocket
    "r_multiple",            # Expected risk-reward ratio
    "prev_range_pct",        # Previous bar range / close
    "gap_pct",               # Gap from previous close
    "atr_pct",               # ATR / close
    "session_open",          # Binary: 1 if open session
    "session_mid",           # Binary: 1 if mid session
    "llm_confidence"         # LLM confidence (0-1)
]
```

### KNN Search Parameters

```python
# In vectorstore.py
knn_default_k = 50          # Number of neighbors to find
similarity_threshold = 0.0   # Minimum cosine similarity (not enforced)
```

### Volatility Regime Classification

```python
# In features.py - vol_regime_from_series
bollinger_period = 20       # Bollinger band period
bollinger_std = 2.0         # Standard deviations
vol_lookback = 60           # Bars for regime classification
vol_low_percentile = 33     # Low volatility threshold
vol_high_percentile = 66    # High volatility threshold
```

## Multi-timeframe Configuration

### Timeframe Mappings

```python
# In data.py
VALID_INTERVALS = {
    "15m": "15m",           # Yahoo Finance format
    "30m": "30m", 
    "1h": "60m",            # Note: 60m for Yahoo Finance
    "1d": "1d"
}
```

### Session Time Buckets

```python
# In features.py - time_of_day_bucket
# Times in market timezone (US/Eastern)
session_open_end = "11:30"     # First 2 hours
session_close_start = "14:30"   # Last 2 hours
# Mid session: 11:30 - 14:30
```

### Bars Per Day Calculation

```python
# Used for holding time calculations
bars_per_day = {
    "15m": 26,              # 6.5 hours * 4 bars/hour
    "30m": 13,              # 6.5 hours * 2 bars/hour
    "1h": 7,                # ~6.5 hours (rounded)
    "1d": 1
}
```

## Risk Management Configuration

### Position Sizing (Not Implemented)

Currently, position sizing is not implemented. All analysis assumes:
- Fixed R-multiple targets (typically 1-3R)
- Percentage risk per trade (not specified)
- No portfolio-level risk management

### Holding Time Limits

```python
# In backtesting and evaluation
max_hold_days = 2.0         # Maximum trade duration
max_hold_bars = max_hold_days * bars_per_day[interval]
```

### Stop Loss Methods

1. **Fibonacci-based**: Below golden pocket + ATR buffer
2. **ATR-based**: Entry ± ATR multiplier
3. **Technical**: Below/above recent swing points

## Database Configuration

### Centralized Database (v1.6.1+)

Starting with v1.6.1, SwingAgent uses a **centralized SQLite database** instead of separate files:

- **Default**: `data/swing_agent.sqlite` 
- **Contains**: Both signals and vector store tables in one file
- **Technology**: SQLAlchemy ORM instead of raw sqlite3
- **Benefits**: 
  - Simplified data management
  - Better query performance with proper indexes
  - Easier backup and deployment
  - Type-safe database operations

```python
# Modern usage - centralized database
agent = SwingAgent(
    log_db="data/swing_agent.sqlite",    # Same file
    vec_db="data/swing_agent.sqlite",    # Same file
)

# Or simply use defaults (recommended)
agent = SwingAgent()  # Uses data/swing_agent.sqlite automatically
```

### Migration from Legacy Setup

If you have existing separate database files, use the migration utility:

```bash
# Migrate from separate files to centralized database
python -m swing_agent.migrate --data-dir data/

# Or migrate specific files
python -m swing_agent.migrate \
    --data-dir data/ \
    --signals-file old_signals.sqlite \
    --vectors-file old_vectors.sqlite
```

### Signals Database Schema

The signals database automatically creates tables with the following key settings:

```sql
-- Primary table
CREATE TABLE signals (
    id TEXT PRIMARY KEY,           -- UUID for each signal
    created_at_utc TEXT NOT NULL,  -- ISO timestamp
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    -- ... full schema in storage.py
);

-- Indexes for performance
CREATE INDEX idx_signals_symbol ON signals(symbol);
CREATE INDEX idx_signals_asof ON signals(asof);
CREATE INDEX idx_signals_evaluated ON signals(evaluated);
```

### Vector Store Schema

```sql
CREATE TABLE vec_store (
    id TEXT PRIMARY KEY,           -- Signal ID reference
    ts_utc TEXT NOT NULL,         -- ISO timestamp
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    vec_json TEXT NOT NULL,       -- Feature vector as JSON array
    realized_r REAL,              -- Actual R-multiple outcome
    exit_reason TEXT,             -- TP/SL/TIME
    payload_json TEXT             -- Additional context data
);

CREATE INDEX vec_store_symbol_idx ON vec_store(symbol);
```

## Script Configuration

### run_swing_agent.py

```bash
python scripts/run_swing_agent.py \
  --symbol AAPL \                    # Required: symbol to analyze
  --interval 30m \                   # Trading timeframe
  --lookback-days 30 \               # Historical data days
  --db data/signals.sqlite \         # Signal storage (optional)
  --vec-db data/vec_store.sqlite \   # Vector store (optional)  
  --sector XLK \                     # Sector ETF
  --no-llm \                         # Disable LLM (optional)
  --no-llm-extras                    # Disable LLM extras (optional)
```

### backtest_generate_signals.py

```bash
python scripts/backtest_generate_signals.py \
  --symbol AAPL \                    # Required: symbol to backtest
  --interval 30m \                   # Trading timeframe
  --lookback-days 180 \              # Historical period
  --warmup-bars 80 \                 # Skip initial bars
  --db data/signals.sqlite \         # Signal storage
  --vec-db data/vec_store.sqlite \   # Vector store
  --sector XLK \                     # Sector ETF
  --no-llm                           # Recommended for backtesting
```

### eval_signals.py

```bash
python scripts/eval_signals.py \
  --db data/signals.sqlite \         # Signal database
  --max-hold-days 2.0                # Maximum trade duration
```

### analyze_performance.py

```bash
python scripts/analyze_performance.py \
  --db data/signals.sqlite           # Signal database
```

## Performance Tuning

### Data Fetching Optimization

```python
# Reduce lookback for faster data fetching
agent = SwingAgent(lookback_days=15)  # vs default 30

# Cache data externally for multiple symbols
from swing_agent.data import load_ohlcv
data_cache = {}
for symbol in symbols:
    data_cache[symbol] = load_ohlcv(symbol, "30m", 30)
```

### Vector Store Optimization

```python
# Limit KNN search for faster lookups
neighbors = knn(vec_db, query_vec, k=25)  # vs default 50

# Symbol-specific searches when possible
neighbors = knn(vec_db, query_vec, k=50, symbol="AAPL")
```

### LLM Cost Control

```python
# Disable LLM for backtesting
agent = SwingAgent(use_llm=False)

# Disable expensive LLM extras
agent = SwingAgent(use_llm=True, llm_extras=False)

# Use cheaper models
export SWING_LLM_MODEL="gpt-3.5-turbo"
```

## Security Configuration

### API Key Management

```bash
# Store in environment file
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc

# Or use secret management
export OPENAI_API_KEY=$(cat /secrets/openai_key)

# Rotate keys regularly
# OpenAI Dashboard -> API Keys -> Rotate
```

### Database Security

```python
# Use parameterized queries (already implemented)
# Store databases in secure location
os.chmod("data/signals.sqlite", 0o600)  # Owner read/write only
```

### Input Validation

```python
# Validate symbols before processing
import re
if not re.match(r'^[A-Z0-9.]{1,10}$', symbol):
    raise ValueError("Invalid symbol format")

# Validate timeframes
if interval not in ["15m", "30m", "1h", "1d"]:
    raise ValueError("Unsupported timeframe")
```

## Custom Configuration Example

```python
# config.py - Custom configuration module
import os
from swing_agent.agent import SwingAgent

class TradingConfig:
    # Environment
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("SWING_LLM_MODEL", "gpt-4o-mini")
    
    # Agent settings
    DEFAULT_INTERVAL = "30m"
    DEFAULT_LOOKBACK = 30
    DEFAULT_SECTOR = "XLK"
    
    # Database paths - Now centralized!
    CENTRALIZED_DB = "data/swing_agent.sqlite"  # Single database for both signals and vectors
    
    # Legacy paths (for migration only)
    SIGNALS_DB = "data/signals.sqlite"  # Old signals database
    VECTOR_DB = "data/vec_store.sqlite"  # Old vector database
    
    # Risk management
    MAX_HOLD_DAYS = 2.0
    
    # Performance tuning
    KNN_NEIGHBORS = 50
    ENABLE_LLM_EXTRAS = True

def create_agent(symbol_type="tech"):
    """Create pre-configured agent based on symbol type."""
    sector_map = {
        "tech": "XLK",
        "finance": "XLF", 
        "energy": "XLE",
        "health": "XLV"
    }
    
    return SwingAgent(
        interval=TradingConfig.DEFAULT_INTERVAL,
        lookback_days=TradingConfig.DEFAULT_LOOKBACK,
        log_db=TradingConfig.CENTRALIZED_DB,
        vec_db=TradingConfig.CENTRALIZED_DB,
        use_llm=bool(TradingConfig.OPENAI_API_KEY),
        llm_extras=TradingConfig.ENABLE_LLM_EXTRAS,
        sector_symbol=sector_map.get(symbol_type, TradingConfig.DEFAULT_SECTOR)
    )

# Usage
agent = create_agent("tech")
signal = agent.analyze("AAPL")
```

## Troubleshooting Configuration Issues

### Common Environment Issues

```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $SWING_LLM_MODEL

# Verify Python path
python -c "import swing_agent; print(swing_agent.__file__)"

# Test database creation
python -c "from swing_agent.storage import record_signal"
```

### Database Permission Issues

```bash
# Check file permissions
ls -la data/
chmod 755 data/
chmod 644 data/*.sqlite
```

### LLM Configuration Issues

```python
# Test LLM connectivity
from swing_agent.llm_predictor import llm_extra_prediction
try:
    result = llm_extra_prediction(symbol="AAPL", price=150.0, trend_label="up")
    print("LLM working")
except Exception as e:
    print(f"LLM error: {e}")
```

### Data Fetching Issues

```python
# Test data fetching
from swing_agent.data import load_ohlcv
try:
    df = load_ohlcv("AAPL", "30m", 5)
    print(f"Data shape: {df.shape}")
except Exception as e:
    print(f"Data error: {e}")
```