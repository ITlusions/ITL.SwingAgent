# System Architecture

This document provides a detailed overview of the SwingAgent v1.6.1 system architecture, component responsibilities, and data flow.

## High-Level Architecture

```mermaid
graph TB
    A[Market Data] --> B[SwingAgent Core]
    B --> C[Technical Analysis]
    B --> D[ML Pattern Matching]
    B --> E[LLM Integration]
    C --> F[Signal Generation]
    D --> F
    E --> F
    F --> G[Centralized Database]
    G --> H[Vector Store]
    G --> I[Signal Storage]
    H --> J[Performance Tracking]
    I --> J
    J --> K[Backtesting & Analytics]
```

## Database Architecture

SwingAgent v1.6.1 introduces a centralized database architecture using SQLAlchemy ORM:

```mermaid
graph LR
    A[SwingAgent] --> B[Database Layer]
    B --> C[SQLite Default]
    B --> D[PostgreSQL]
    B --> E[MySQL] 
    B --> F[CNPG Kubernetes]
    
    G[Legacy Migration] --> B
    G --> H[Signals DB]
    G --> I[Vector DB]
    H --> B
    I --> B
```

### Database Features
- **Unified Storage**: Single database for signals and vector patterns
- **Multiple Backends**: SQLite, PostgreSQL, MySQL, CloudNativePG
- **Migration Tools**: Automated migration from legacy separate databases
- **SQLAlchemy ORM**: Type-safe database operations
- **Production Ready**: Connection pooling, SSL support, Kubernetes integration

## Core Components

### 1. SwingAgent Core (`agent.py`)

The main orchestrator that coordinates all system components.

**Key Responsibilities:**
- Data fetching and preprocessing
- Component orchestration
- Multi-timeframe analysis coordination
- Signal assembly and validation
- Database session management

**Key Methods:**
- `analyze(symbol)`: Main entry point for signal generation
- `analyze_df(symbol, df)`: Core analysis with pre-loaded data
- `_rel_strength()`: Relative strength calculation
- `_context_from_df()`: Market context extraction
- `_multitimeframe_analysis()`: MTF trend alignment

**Data Flow:**
```
Input: Symbol + Configuration
↓
Fetch OHLCV data (current + 15m/1h timeframes)
↓
Technical analysis (trend, indicators, Fibonacci)
↓
Multi-timeframe alignment check
↓
Entry plan generation
↓
Vector pattern matching via centralized database
↓
LLM analysis (if enabled)
↓
Signal assembly and database storage
```

### 2. Database Layer (`database.py`, `models_db.py`)

Centralized database management using SQLAlchemy ORM.

**Key Components:**
- `database.py`: Session management, connection configuration
- `models_db.py`: SQLAlchemy models for signals and vectors
- `migrate.py`: Migration utilities for legacy databases

**Supported Backends:**
- **SQLite**: Default for development (`data/swing_agent.sqlite`)
- **PostgreSQL**: Production deployments with `psycopg2`
- **MySQL**: Alternative production backend with `PyMySQL`
- **CNPG**: CloudNativePG for Kubernetes deployments

**Configuration Options:**
```python
# Environment variable configurations
SWING_DATABASE_URL="postgresql://user:pass@host:5432/db"
SWING_DB_TYPE="postgresql"  # sqlite, postgresql, mysql, cnpg
SWING_DB_HOST="localhost"
SWING_DB_NAME="swing_agent"
SWING_DB_USER="username"
SWING_DB_PASSWORD="password"
```

### 3. Configuration Management (`config.py`)

Centralized parameter management for all trading thresholds.

**Key Features:**
- Eliminates magic numbers throughout codebase
- Type-safe configuration with dataclasses
- Consistent parameter access via `get_config()`

**Configuration Categories:**
- Trend detection thresholds
- RSI momentum parameters
- Risk management (ATR multipliers)
- Volatility regime classification
- Fibonacci analysis parameters
- Multi-timeframe settings
- Vector store configuration
↓
Vector store lookup for historical patterns
↓
LLM analysis (optional)
↓
Signal assembly and storage
```

### 2. Strategy Engine (`strategy.py`)

Implements the core trading logic for trend identification and entry planning.

**Trend Labeling Logic:**
```python
if slope > 0.01 and price_above_ema and rsi >= 60:
    label = STRONG_UP if slope > 0.02 else UP
elif slope < -0.01 and not price_above_ema and rsi <= 40:
    label = STRONG_DOWN if slope < -0.02 else DOWN
else:
    label = SIDEWAYS
```

**Entry Strategies:**

1. **Fibonacci Golden Pocket**
   - Entry: Within 0.618-0.65 retracement
   - Stop: Below golden pocket + 0.2*ATR buffer
   - Target: Previous high/low or 1.272 extension

2. **Momentum Continuation**
   - Entry: Break above/below previous high/low
   - Stop: 1.2*ATR from entry
   - Target: 2.0*ATR or Fibonacci extension

3. **Mean Reversion**
   - Entry: RSI < 35 (oversold) or RSI > 65 (overbought)
   - Stop: 1.0*ATR from entry
   - Target: 1.5*ATR mean reversion

### 3. Technical Indicators (`indicators.py`)

Provides all technical analysis calculations.

**Indicators Implemented:**
- **EMA**: Exponential Moving Average with configurable span
- **RSI**: Relative Strength Index (14-period default)
- **ATR**: Average True Range for volatility measurement
- **Bollinger Width**: Volatility regime classification
- **Fibonacci Levels**: Retracement and extension calculations

**Fibonacci Calculation:**
```python
def fibonacci_range(df, lookback=40):
    # Find recent swing high/low within lookback period
    # Calculate retracement levels (0.236, 0.382, 0.5, 0.618, 0.786)
    # Calculate extension levels (1.272, 1.414, 1.618)
    # Determine golden pocket (0.618-0.65 range)
```

### 4. Feature Engineering (`features.py`)

Converts market data into ML-ready feature vectors.

**Feature Vector Components:**
- Trend direction (up/down/sideways binary encoding)
- RSI momentum (normalized 0-1)
- Price position relative to EMAs
- Fibonacci position within golden pocket
- ATR percentage (volatility)
- Session timing (open/mid/close buckets)
- Gap analysis
- LLM confidence (if available)

**Session Bucketing:**
- Open: First 2 hours of trading
- Mid: Middle hours  
- Close: Last 2 hours of trading

**Volatility Regime:**
- Low: Bottom 33% of 60-period Bollinger width
- Medium: Middle 33%
- High: Top 33%

### 5. Vector Store (`vectorstore.py`)

SQLite-based similarity search for historical pattern matching.

**Database Schema:**
```sql
CREATE TABLE vec_store (
  id TEXT PRIMARY KEY,
  ts_utc TEXT NOT NULL,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  vec_json TEXT NOT NULL,     -- Feature vector as JSON
  realized_r REAL,            -- Actual R-multiple outcome
  exit_reason TEXT,           -- TP/SL/TIME
  payload_json TEXT           -- Additional context data
);
```

**KNN Search Process:**
1. Encode current market state into feature vector
2. Calculate cosine similarity with all stored vectors
3. Return top-k most similar historical setups
4. Filter by symbol (optional) and volatility regime
5. Calculate statistical expectations from neighbors

**Statistical Metrics:**
- Win rate (% of positive R outcomes)
- Average R-multiple (risk-adjusted returns)
- Median holding time (bars and days)
- Profit factor (gross profit / gross loss)
- Exit reason distribution (TP/SL/TIME)

### 6. Signal Storage (`storage.py`)

Comprehensive signal database with expectations and outcomes.

**Database Schema:**
```sql
CREATE TABLE signals (
  -- Core signal data
  id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  asof TEXT NOT NULL,
  trend_label TEXT NOT NULL,
  
  -- Technical indicators
  ema_slope REAL NOT NULL,
  price_above_ema INTEGER NOT NULL,
  rsi14 REAL NOT NULL,
  
  -- Entry plan
  side TEXT,
  entry_price REAL,
  stop_price REAL,
  take_profit REAL,
  r_multiple REAL,
  
  -- Fibonacci levels
  fib_golden_low REAL,
  fib_golden_high REAL,
  fib_target_1 REAL,
  fib_target_2 REAL,
  
  -- ML expectations
  expected_r REAL,
  expected_winrate REAL,
  expected_hold_bars INTEGER,
  expected_hold_days REAL,
  
  -- LLM outputs
  llm_vote_json TEXT,
  llm_explanation TEXT,
  action_plan TEXT,
  risk_notes TEXT,
  scenarios_json TEXT,
  
  -- Enrichments
  mtf_15m_trend TEXT,
  mtf_1h_trend TEXT,
  mtf_alignment INTEGER,
  rs_sector_20 REAL,
  rs_spy_20 REAL,
  sector_symbol TEXT,
  tod_bucket TEXT,
  atr_pct REAL,
  vol_regime TEXT,
  
  -- Evaluation results
  evaluated INTEGER DEFAULT 0,
  exit_reason TEXT,
  exit_price REAL,
  exit_time_utc TEXT,
  realized_r REAL
);
```

### 7. LLM Integration (`llm_predictor.py`)

OpenAI integration for trade explanations and action plans.

**LLM Vote Structure:**
```python
class LlmVote(BaseModel):
    trend_label: Literal["strong_up", "up", "sideways", "down", "strong_down"]
    entry_bias: Literal["long", "short", "none"]
    entry_window_low: Optional[float]
    entry_window_high: Optional[float]
    confidence: float  # 0-1
    rationale: str
```

**Action Plan Structure:**
```python
class LlmActionPlan(BaseModel):
    action_plan: str          # Step-by-step execution plan
    risk_notes: str           # Risk management and invalidations
    scenarios: List[str]      # 2-4 potential outcomes
    tone: Literal["conservative", "balanced", "aggressive"]
```

**LLM Usage Pattern:**
1. Technical analysis completed first (no LLM influence on math)
2. LLM provides additional confidence and explanation
3. LLM generates structured action plans with scenarios
4. All LLM outputs stored for transparency and analysis

### 8. Data Management (`data.py`)

Market data fetching and preprocessing via Yahoo Finance.

**Supported Timeframes:**
- 15m: 15-minute bars
- 30m: 30-minute bars  
- 1h: 1-hour bars
- 1d: Daily bars

**Data Quality Checks:**
- Remove duplicate timestamps
- Validate OHLCV column presence
- Convert timezone to UTC
- Handle missing data gracefully

### 9. Backtesting Engine (`backtester.py`)

Trade simulation for historical performance analysis.

**Simulation Logic:**
```python
def simulate_trade(df, open_idx, side, entry, stop, target, max_hold_bars):
    for i in range(open_idx + 1, min(len(df), open_idx + 1 + max_hold_bars)):
        high, low = df["high"].iloc[i], df["low"].iloc[i]
        
        if side == LONG:
            if low <= stop: return i, "SL", stop
            if high >= target: return i, "TP", target
        else:  # SHORT
            if high >= stop: return i, "SL", stop
            if low <= target: return i, "TP", target
    
    # Time exit if max holding period reached
    exit_idx = min(len(df) - 1, open_idx + max_hold_bars)
    return exit_idx, "TIME", df["close"].iloc[exit_idx]
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Strategy
    participant Indicators
    participant VectorStore
    participant LLM
    participant Storage
    
    User->>Agent: analyze(symbol)
    Agent->>Data: fetch_ohlcv(symbol)
    Data-->>Agent: OHLCV DataFrame
    
    Agent->>Indicators: calculate_all(df)
    Indicators-->>Agent: EMA, RSI, ATR, Fibonacci
    
    Agent->>Strategy: label_trend(df)
    Strategy-->>Agent: TrendState
    
    Agent->>Strategy: build_entry(df, trend)
    Strategy-->>Agent: EntryPlan
    
    Agent->>Features: build_setup_vector(...)
    Features-->>Agent: feature_vector
    
    Agent->>VectorStore: knn(feature_vector)
    VectorStore-->>Agent: similar_patterns
    
    Agent->>VectorStore: extended_stats(patterns)
    VectorStore-->>Agent: expectations
    
    opt LLM enabled
        Agent->>LLM: llm_extra_prediction(features)
        LLM-->>Agent: LlmVote
        
        Agent->>LLM: llm_build_action_plan(signal)
        LLM-->>Agent: LlmActionPlan
    end
    
    Agent->>Storage: record_signal(trade_signal)
    Storage-->>Agent: signal_id
    
    Agent-->>User: TradeSignal
```

## Configuration Management

Currently, configuration is handled through:

1. **Command-line arguments**: Script-specific parameters
2. **Environment variables**: API keys and model settings
3. **Hardcoded constants**: Technical indicator parameters

**Environment Variables:**
```bash
OPENAI_API_KEY=sk-...           # OpenAI API key
SWING_LLM_MODEL=gpt-4o-mini     # LLM model selection
```

## Error Handling Strategy

**Current Approach:**
- Basic try/catch around external API calls
- Graceful degradation when LLM unavailable
- Data validation at component boundaries

**Areas for Improvement:**
- Centralized error logging
- Retry logic for network failures
- Input validation and sanitization
- Structured error reporting

## Performance Considerations

**Bottlenecks:**
1. **Vector Store Lookups**: Loading all vectors for each search
2. **LLM API Calls**: Network latency and rate limits
3. **Data Fetching**: Yahoo Finance API limitations
4. **Historical Backtesting**: Processing large datasets

**Optimization Opportunities:**
1. Vector store indexing and caching
2. Batch LLM processing
3. Data caching and incremental updates
4. Parallel processing for backtests

## Security Architecture

**Current Security Measures:**
- Environment variable for API keys
- No hardcoded credentials
- SQL parameterized queries (mostly)

**Security Gaps:**
- No API key rotation
- Limited input validation
- No rate limiting on external calls
- Missing audit logging