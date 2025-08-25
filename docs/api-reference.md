# API Reference

Complete API documentation for all SwingAgent components.

## Core Classes

### SwingAgent

Main orchestrator class that coordinates all system components using a modular, well-structured approach.

```python
class SwingAgent:
    def __init__(
        self,
        interval: str = "30m",
        lookback_days: int = 30,
        use_llm: bool = True,
        llm_extras: bool = True,
        sector_symbol: str = "XLK"
    )
```

**Parameters:**
- `interval`: Trading timeframe ("15m", "30m", "1h", "1d")
- `lookback_days`: Historical data period for analysis
- `use_llm`: Enable OpenAI LLM integration
- `llm_extras`: Enable additional LLM features (action plans, risk scenarios)
- `sector_symbol`: Sector ETF for relative strength analysis (default: XLK)

**Note**: Database configuration is now handled via environment variables. See [Configuration Guide](configuration.md) for details.

**Methods:**

#### `analyze(symbol: str) -> TradeSignal`
Generate a comprehensive trading signal for the given symbol.

```python
agent = SwingAgent(interval="30m", lookback_days=30)
signal = agent.analyze("AAPL")
print(f"Signal: {signal.entry.side if signal.entry else 'No setup'}")
print(f"Confidence: {signal.confidence:.1%}")
```

#### `analyze_df(symbol: str, df: pd.DataFrame) -> TradeSignal`
Generate a signal using pre-loaded market data. This method orchestrates the entire analysis pipeline:

1. **Market Context Building** - Volatility, gaps, time of day, relative strength
2. **Technical Analysis** - Trend labeling and entry plan generation
3. **Multi-timeframe Analysis** - Trend alignment across 15m, 30m, 1h timeframes
4. **ML Expectations** - Vector similarity analysis for statistical priors
5. **LLM Insights** - AI-generated analysis and action plans
6. **Signal Assembly** - Final signal creation and persistence

```python
df = load_ohlcv("AAPL", "30m", 30)
signal = agent.analyze_df("AAPL", df)
```

## Configuration Management

### TradingConfig

Centralized configuration class that eliminates magic numbers throughout the codebase.

```python
from swing_agent.config import get_config, update_config

# Get current configuration
config = get_config()
print(f"EMA slope threshold: {config.EMA_SLOPE_THRESHOLD_UP}")
print(f"RSI trend threshold: {config.RSI_TREND_UP_MIN}")

# Update configuration parameters
update_config(
    EMA_SLOPE_THRESHOLD_UP=0.015,  # More sensitive trend detection
    RSI_PERIOD=21,                 # Longer RSI period
    ATR_STOP_MULTIPLIER=1.5       # Tighter stops
)
```

**Key Configuration Groups:**

- **Trend Detection**: EMA slope thresholds, RSI levels
- **Risk Management**: ATR multipliers, stop/target ratios
- **Fibonacci Analysis**: Lookback periods, golden pocket levels  
- **Volatility Regimes**: Lookback periods, percentile thresholds
- **Confidence Scoring**: Base levels, bonus factors
- **Vector Analysis**: KNN parameters, similarity thresholds

## Error Handling

Enhanced error handling with custom exceptions and detailed error messages:

```python
from swing_agent.data import SwingAgentDataError

try:
    signal = agent.analyze("INVALID_SYMBOL")
except SwingAgentDataError as e:
    print(f"Data error for {e.symbol}: {e}")
    # Handle specific data issues (invalid symbol, no data, etc.)
except RuntimeError as e:
    print(f"Analysis failed: {e}")
    # Handle general analysis errors
```

**Common Error Scenarios:**
- Invalid or delisted symbols
- Insufficient historical data
- Network connectivity issues
- Rate limiting from data providers
- Market closure during data requests

## Data Models

### TradeSignal

Complete trading signal with technical analysis, ML expectations, and LLM insights.

```python
class TradeSignal(BaseModel):
    symbol: str
    timeframe: Literal["15m", "30m", "1h", "1d"] = "30m"
    asof: str  # Timestamp of analysis
    trend: TrendState  # Technical trend analysis
    entry: Optional[EntryPlan] = None  # Entry plan if signal found
    confidence: float = 0.0  # Overall signal confidence 0-1
    reasoning: str = ""  # Analysis reasoning
    
    # ML-derived expectations from vector similarity
    expected_r: Optional[float] = None  # Expected R-multiple return
    expected_winrate: Optional[float] = None  # Win rate from similar patterns
    expected_source: Optional[str] = None  # Source of expectations
    expected_notes: Optional[str] = None  # Additional context
    
    # Holding time priors from historical patterns
    expected_hold_bars: Optional[int] = None  # Expected bars to hold
    expected_hold_days: Optional[float] = None  # Expected days to hold
    expected_win_hold_bars: Optional[int] = None  # Bars for winning trades
    expected_loss_hold_bars: Optional[int] = None  # Bars for losing trades
    
    # LLM insights and action planning
    llm_vote: Optional[Dict[str, Any]] = None  # LLM trend/bias assessment
    llm_explanation: Optional[str] = None  # LLM reasoning
    action_plan: Optional[str] = None  # Execution checklist
    risk_notes: Optional[str] = None  # Risk and invalidation scenarios
    scenarios: Optional[List[str]] = None  # Possible outcomes
    
    # Market context enrichments
    mtf_15m_trend: Optional[str] = None  # 15-minute trend
    mtf_1h_trend: Optional[str] = None  # 1-hour trend
    mtf_alignment: Optional[int] = None  # Multi-timeframe alignment score
    rs_sector_20: Optional[float] = None  # Relative strength vs sector
    rs_spy_20: Optional[float] = None  # Relative strength vs SPY
    sector_symbol: Optional[str] = None  # Sector ETF symbol
    tod_bucket: Optional[str] = None  # Time of day ("open", "mid", "close")
    atr_pct: Optional[float] = None  # ATR as percentage of price
    vol_regime: Optional[str] = None  # Volatility regime ("L", "M", "H")
```

**Example Usage:**
```python
agent = SwingAgent()
signal = agent.analyze("AAPL")

print(f"Symbol: {signal.symbol}")
print(f"Trend: {signal.trend.label.value}")
print(f"Confidence: {signal.confidence:.2f}")

if signal.entry:
    print(f"Entry: {signal.entry.side.value} at ${signal.entry.entry_price:.2f}")
    print(f"Stop: ${signal.entry.stop_price:.2f}")
    print(f"Target: ${signal.entry.take_profit:.2f}")
    print(f"R-Multiple: {signal.entry.r_multiple:.2f}")

if signal.expected_r:
    print(f"Expected Return: {signal.expected_r:.2f}R")
    print(f"Expected Win Rate: {signal.expected_winrate:.1%}")
```

### TrendState

Technical trend analysis with key momentum indicators.

```python
class TrendState(BaseModel):
    label: TrendLabel  # Classified trend direction
    ema_slope: float  # EMA20 slope as percentage
    price_above_ema: bool  # Current price vs EMA20
    rsi_14: float  # 14-period RSI value (0-100)
```

**TrendLabel Classifications:**
- `STRONG_UP`: Strong uptrend (slope > 2%, RSI ≥ 60, price above EMA)  
- `UP`: Uptrend (slope > 1%, RSI ≥ 60, price above EMA)
- `SIDEWAYS`: Consolidation or unclear direction
- `DOWN`: Downtrend (slope < -1%, RSI ≤ 40, price below EMA)
- `STRONG_DOWN`: Strong downtrend (slope < -2%, RSI ≤ 40, price below EMA)

### EntryPlan  

Trade entry specification with risk management levels.

```python
class EntryPlan(BaseModel):
    side: SignalSide  # "long", "short", or "none"
    entry_price: float  # Specific entry price level
    stop_price: float  # Stop loss price for risk management
    take_profit: float  # Initial profit target
    r_multiple: float  # Risk/reward ratio (target/risk)
    comment: str  # Strategy description and context
    
    # Fibonacci analysis context
    fib_golden_low: Optional[float] = None  # 0.618 retracement level
    fib_golden_high: Optional[float] = None  # 0.65 retracement level  
    fib_target_1: Optional[float] = None  # 1.272 extension target
    fib_target_2: Optional[float] = None  # 1.618 extension target
```

**Entry Strategy Categories:**
1. **Fibonacci Golden Pocket**: Pullback to 0.618-0.65 retracement zone
2. **Momentum Continuation**: Breakout above/below prior swing extremes
3. **Mean Reversion**: RSI extreme reversals in sideways markets

**Risk Management Validation:**
```python
# Long trade validation
if entry.side == "long":
    assert entry.stop_price < entry.entry_price
    assert entry.take_profit > entry.entry_price
    
# Short trade validation  
if entry.side == "short":
    assert entry.stop_price > entry.entry_price
    assert entry.take_profit < entry.entry_price
```

### TradeSignal

Complete trading signal with technical analysis, ML expectations, and LLM insights.

```python
class TradeSignal(BaseModel):
    # Core identification
    symbol: str
    timeframe: Literal["15m", "30m", "1h", "1d"] = "30m"
    asof: str                              # ISO timestamp
    
    # Technical analysis
    trend: TrendState
    entry: Optional[EntryPlan] = None
    confidence: float = 0.0
    reasoning: str = ""
    
    # ML expectations
    expected_r: float | None = None
    expected_winrate: float | None = None
    expected_source: str | None = None
    expected_notes: str | None = None
    expected_hold_bars: int | None = None
    expected_hold_days: float | None = None
    expected_win_hold_bars: int | None = None
    expected_loss_hold_bars: int | None = None
    
    # LLM outputs
    llm_vote: Optional[Dict[str, Any]] = None
    llm_explanation: Optional[str] = None
    action_plan: Optional[str] = None
    risk_notes: Optional[str] = None
    scenarios: Optional[List[str]] = None
    
    # Enrichments
    mtf_15m_trend: Optional[str] = None
    mtf_1h_trend: Optional[str] = None
    mtf_alignment: Optional[int] = None      # 0-2 timeframes aligned
    rs_sector_20: Optional[float] = None     # Relative strength vs sector
    rs_spy_20: Optional[float] = None        # Relative strength vs SPY
    sector_symbol: Optional[str] = None
    tod_bucket: Optional[str] = None         # "open", "mid", "close"
    atr_pct: Optional[float] = None
    vol_regime: Optional[str] = None         # "L", "M", "H"
```

### TrendState

Technical trend analysis results.

```python
class TrendState(BaseModel):
    label: TrendLabel                        # Trend classification
    ema_slope: float                         # EMA20 slope (normalized)
    price_above_ema: bool                    # Price vs EMA20 position
    rsi_14: float                           # RSI(14) current value
```

### EntryPlan

Trade entry plan with risk management.

```python
class EntryPlan(BaseModel):
    side: SignalSide                         # "long", "short", "none"
    entry_price: float = Field(..., gt=0)
    stop_price: float = Field(..., gt=0)
    take_profit: float = Field(..., gt=0)
    r_multiple: float = Field(..., gt=0)     # Risk-reward ratio
    comment: str = ""                        # Strategy description
    
    # Fibonacci levels
    fib_golden_low: Optional[float] = None
    fib_golden_high: Optional[float] = None
    fib_target_1: Optional[float] = None     # 1.272 extension
    fib_target_2: Optional[float] = None     # 1.618 extension
```

### Enums

```python
class TrendLabel(str, Enum):
    STRONG_UP = "strong_up"
    UP = "up"
    SIDEWAYS = "sideways" 
    DOWN = "down"
    STRONG_DOWN = "strong_down"

class SignalSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"
```

## Database Management

### Database Configuration

```python
from swing_agent.database import (
    get_database_config,
    init_database,
    get_session,
    create_postgresql_url,
    create_mysql_url,
    create_cnpg_url
)

# Initialize database tables
init_database()

# Get database session for queries
with get_session() as session:
    # Database operations here
    signals = session.query(Signal).all()

# Create connection URLs
postgres_url = create_postgresql_url(
    host="localhost",
    database="swing_agent", 
    username="user",
    password="pass"
)

mysql_url = create_mysql_url(
    host="localhost",
    database="swing_agent",
    username="user", 
    password="pass"
)

# CloudNativePG URL from environment
cnpg_url = create_cnpg_url()
```

### Database Models

SQLAlchemy models for centralized storage.

```python
from swing_agent.models_db import Signal, VectorStore

# Query signals with filters
with get_session() as session:
    signals = session.query(Signal).filter(
        Signal.symbol == "AAPL",
        Signal.timeframe == "30m"
    ).order_by(Signal.created_at_utc.desc()).limit(10).all()
    
    # Query vector patterns with outcomes
    vectors = session.query(VectorStore).filter(
        VectorStore.realized_r.isnot(None),
        VectorStore.symbol == "AAPL"
    ).all()
```

**Signal Model Fields:**
- `id`: Unique signal identifier
- `symbol`, `timeframe`, `asof`: Basic signal info
- `trend_label`, `ema_slope`, `rsi14`: Technical analysis
- `entry_price`, `stop_price`, `take_profit`: Entry plan
- `confidence`, `reasoning`: Signal quality
- `expected_r`, `expected_winrate`: ML expectations
- `action_plan`, `risk_notes`: LLM insights
- `mtf_alignment`, `vol_regime`: Market context
- `realized_r`, `exit_reason`: Trade outcomes

**VectorStore Model Fields:**
- `vid`: Unique vector identifier
- `ts_utc`: Vector timestamp
- `symbol`, `timeframe`: Market context
- `vector`: Feature vector as JSON array
- `realized_r`, `exit_reason`: Outcomes
- `payload_json`: Additional metadata

### Migration Utilities

```python
from swing_agent.migrate import (
    migrate_signals,
    migrate_vectors, 
    migrate_all_data
)

# Migrate from legacy separate databases
migrate_all_data(
    data_dir=Path("data/"),
    target_db_url="postgresql://user:pass@host:5432/swing_agent"
)

# Migrate specific components
migrate_signals(
    old_signals_db=Path("data/signals.sqlite"),
    target_db_url="postgresql://..."
)

migrate_vectors(
    old_vectors_db=Path("data/vec_store.sqlite"),
    target_db_url="postgresql://..."
)
```

## Configuration Management

### TradingConfig

Centralized configuration for all trading parameters.

```python
from swing_agent.config import get_config, TradingConfig

cfg = get_config()

# Access trend detection thresholds
ema_threshold_up = cfg.EMA_SLOPE_THRESHOLD_UP      # 0.01
ema_threshold_strong = cfg.EMA_SLOPE_THRESHOLD_STRONG  # 0.02
ema_lookback = cfg.EMA_SLOPE_LOOKBACK              # 6

# RSI parameters
rsi_oversold = cfg.RSI_OVERSOLD_THRESHOLD          # 35.0
rsi_overbought = cfg.RSI_OVERBOUGHT_THRESHOLD      # 65.0
rsi_period = cfg.RSI_PERIOD                        # 14

# Risk management
atr_stop_mult = cfg.ATR_STOP_MULTIPLIER            # 1.2
atr_target_mult = cfg.ATR_TARGET_MULTIPLIER        # 2.0
atr_period = cfg.ATR_PERIOD                        # 14

# Fibonacci settings
fib_lookback = cfg.FIB_LOOKBACK                    # 40
golden_low = cfg.GOLDEN_POCKET_LOW                 # 0.618
golden_high = cfg.GOLDEN_POCKET_HIGH               # 0.65
```

**Configuration Categories:**
- **Trend Detection**: EMA slope thresholds, RSI levels
- **Risk Management**: ATR multipliers for stops/targets
- **Volatility Analysis**: Regime classification percentiles
- **Fibonacci**: Lookback periods, golden pocket bounds
- **Multi-timeframe**: Alignment scoring parameters
- **Vector Store**: KNN search parameters

## Technical Indicators

### Core Functions

#### `ema(series: pd.Series, span: int) -> pd.Series`
Exponential Moving Average calculation.

```python
from swing_agent.indicators import ema

close_prices = df["close"]
ema_20 = ema(close_prices, 20)
```

#### `rsi(series: pd.Series, length: int = 14) -> pd.Series`
Relative Strength Index momentum oscillator.

```python
from swing_agent.indicators import rsi

rsi_values = rsi(df["close"], 14)
current_rsi = rsi_values.iloc[-1]
```

#### `atr(df: pd.DataFrame, length: int = 14) -> pd.Series`
Average True Range volatility indicator.

```python
from swing_agent.indicators import atr

atr_values = atr(df, 14)
current_atr = atr_values.iloc[-1]
```

#### `fibonacci_range(df: pd.DataFrame, lookback: int = 40) -> FibRange`
Calculate Fibonacci retracement and extension levels.

```python
from swing_agent.indicators import fibonacci_range

fib = fibonacci_range(df, lookback=40)
golden_pocket = (fib.golden_low, fib.golden_high)
extension_1272 = fib.levels["1.272"]
```

### FibRange Object

```python
@dataclass
class FibRange:
    start: float                    # Swing start price
    end: float                      # Swing end price  
    dir_up: bool                    # True if upward swing
    levels: dict                    # All Fibonacci levels
    golden_low: float               # 0.618 level
    golden_high: float              # 0.65 level
```

**Available Fibonacci Levels:**
- Retracements: 0.236, 0.382, 0.5, 0.618, 0.65, 0.786
- Extensions: 1.0, 1.272, 1.414, 1.618

## Strategy Functions

### `label_trend(df: pd.DataFrame) -> TrendState`
Classify market trend based on EMA slope, price position, and RSI using configurable thresholds.

```python
from swing_agent.strategy import label_trend

trend = label_trend(df)
print(f"Trend: {trend.label}, RSI: {trend.rsi_14:.1f}")
print(f"EMA Slope: {trend.ema_slope:.4f}")
print(f"Price above EMA: {trend.price_above_ema}")
```

**Trend Classification Logic:**
Uses three main trend categories with configurable thresholds from `TradingConfig`:

1. **Strong/Up Trends**:
   - EMA20 slope > `EMA_SLOPE_THRESHOLD_UP` (default: 0.01)
   - Price above EMA20
   - RSI ≥ `RSI_TREND_UP_MIN` (default: 60)
   - **STRONG_UP** if slope > `EMA_SLOPE_THRESHOLD_STRONG` (default: 0.02)

2. **Strong/Down Trends**:
   - EMA20 slope < `EMA_SLOPE_THRESHOLD_DOWN` (default: -0.01)
   - Price below EMA20
   - RSI ≤ `RSI_TREND_DOWN_MAX` (default: 40)
   - **STRONG_DOWN** if slope < `EMA_SLOPE_THRESHOLD_STRONG_DOWN` (default: -0.02)

3. **SIDEWAYS**: All other conditions

### `build_entry(df: pd.DataFrame, trend: TrendState) -> Optional[EntryPlan]`
Generate entry plan using three main strategies with configurable risk management.

```python
from swing_agent.strategy import build_entry, label_trend

trend = label_trend(df)
entry = build_entry(df, trend)

if entry:
    print(f"Strategy: {entry.comment}")
    print(f"Entry: {entry.side.value} @ {entry.entry_price:.2f}")
    print(f"Stop: {entry.stop_price:.2f}, Target: {entry.take_profit:.2f}")
    print(f"R-Multiple: {entry.r_multiple:.2f}")
    print(f"Golden Pocket: {entry.fib_golden_low:.2f} - {entry.fib_golden_high:.2f}")
```

**Entry Strategies:**

1. **Fibonacci Golden Pocket Pullbacks** (Highest Probability):
   - Triggers when price is within 61.8%-65% retracement of recent swing
   - **LONG**: In uptrend, price pullback to golden pocket
   - **SHORT**: In downtrend, price pullback to golden pocket
   - Stop: Golden pocket boundary + `ATR_STOP_BUFFER` × ATR (default: 0.2)
   - Target: Previous swing point or Fibonacci extension

2. **Momentum Continuation Breakouts**:
   - **LONG**: In uptrend, breakout above previous high
   - **SHORT**: In downtrend, breakdown below previous low
   - Stop: Entry ± `ATR_STOP_MULTIPLIER` × ATR (default: 1.2)
   - Target: Fibonacci extension or `ATR_TARGET_MULTIPLIER` × ATR (default: 2.0)

3. **Mean Reversion from Extremes**:
   - Triggers in sideways markets only
   - **LONG**: RSI < `RSI_OVERSOLD_THRESHOLD` (default: 35)
   - **SHORT**: RSI > `RSI_OVERBOUGHT_THRESHOLD` (default: 65)
   - Stop: `ATR_MEAN_REVERSION_STOP` × ATR (default: 1.0)
   - Target: `ATR_MEAN_REVERSION_TARGET` × ATR (default: 1.5)

**Risk Management Features:**
- All stops and targets use ATR-based calculations
- R-multiple automatically calculated for each setup
- Fibonacci levels included for context and additional targets
- Configuration allows easy tuning of all parameters

## Vector Store API

### `add_vector(db_path, vid, ts_utc, symbol, timeframe, vec, realized_r, exit_reason, payload)`
Store a feature vector with outcomes.

```python
from swing_agent.vectorstore import add_vector
import numpy as np

vector = np.array([0.1, 0.8, 0.3, 0.6, 0.2])
add_vector(
    db_path="data/vec_store.sqlite",
    vid="AAPL-2024-01-15T15:30:00Z",
    ts_utc="2024-01-15T15:30:00Z",
    symbol="AAPL",
    timeframe="30m",
    vec=vector,
    realized_r=1.5,
    exit_reason="TP",
    payload={"vol_regime": "M", "mtf_alignment": 2}
)
```

### `knn(db_path, query_vec, k=50, symbol=None) -> List[Dict]`
Find k most similar historical patterns.

```python
from swing_agent.vectorstore import knn
import numpy as np

query_vector = np.array([0.1, 0.8, 0.3, 0.6, 0.2])
neighbors = knn(
    db_path="data/vec_store.sqlite",
    query_vec=query_vector,
    k=50,
    symbol="AAPL"  # Optional: filter by symbol
)

for neighbor in neighbors[:5]:
    print(f"Similarity: {neighbor['similarity']:.3f}, R: {neighbor['realized_r']}")
```

### `extended_stats(neighbors: List[Dict]) -> Dict`
Calculate statistical expectations from similar patterns.

```python
from swing_agent.vectorstore import extended_stats

stats = extended_stats(neighbors)
print(f"Win Rate: {stats['p_win']:.2%}")
print(f"Average R: {stats['avg_R']:.2f}")
print(f"Median Hold: {stats['median_hold_days']} days")
```

**Returned Statistics:**
- `n`: Number of neighbors
- `p_win`: Win rate (0-1)
- `avg_R`: Average R-multiple
- `avg_win_R`: Average winning R-multiple
- `avg_loss_R`: Average losing R-multiple
- `median_hold_bars`: Median holding period in bars
- `median_hold_days`: Median holding period in days
- `median_win_hold_bars`: Median winning trade duration
- `median_loss_hold_bars`: Median losing trade duration
- `profit_factor`: Gross profit / gross loss
- `tp`: Count of take-profit exits
- `sl`: Count of stop-loss exits
- `time`: Count of time-based exits

## Feature Engineering

### `build_setup_vector(price, trend, entry, prev_range_pct, gap_pct, atr_pct, session_bin, llm_conf)`
Convert market state into ML feature vector.

```python
from swing_agent.features import build_setup_vector

vector = build_setup_vector(
    price=185.50,
    trend=trend_state,
    entry=entry_plan,
    prev_range_pct=0.015,
    gap_pct=0.002,
    atr_pct=0.012,
    session_bin=1,  # 0=open, 1=mid, 2=close
    llm_conf=0.75
)
```

### `time_of_day_bucket(ts: pd.Timestamp) -> str`
Classify timestamp into trading session.

```python
from swing_agent.features import time_of_day_bucket

bucket = time_of_day_bucket(pd.Timestamp("2024-01-15 10:30:00", tz="US/Eastern"))
# Returns: "open", "mid", or "close"
```

### `vol_regime_from_series(price: pd.Series) -> str`
Classify volatility regime from price series.

```python
from swing_agent.features import vol_regime_from_series

regime = vol_regime_from_series(df["close"])
# Returns: "L" (low), "M" (medium), or "H" (high)
```

## Data Management

### `load_ohlcv(symbol, interval="30m", lookback_days=30) -> pd.DataFrame`
Fetch market data via Yahoo Finance.

```python
from swing_agent.data import load_ohlcv

# Fetch 30 days of 30-minute AAPL data
df = load_ohlcv("AAPL", "30m", 30)

# Available intervals: "15m", "30m", "1h", "1d"
daily_data = load_ohlcv("AAPL", "1d", 90)
```

**Returned DataFrame:**
- Index: UTC timestamps
- Columns: open, high, low, close, volume (lowercase)
- Data cleaning: duplicates removed, timezone normalized

## LLM Integration

### `llm_extra_prediction(**features) -> LlmVote`
Get LLM analysis of current market setup.

```python
from swing_agent.llm_predictor import llm_extra_prediction

vote = llm_extra_prediction(
    symbol="AAPL",
    price=185.50,
    trend_label="up",
    rsi_14=62.3,
    ema_slope=0.0156,
    fib_golden_low=184.20,
    fib_golden_high=186.10
)

print(f"LLM Trend: {vote.trend_label}")
print(f"Entry Bias: {vote.entry_bias}")
print(f"Confidence: {vote.confidence}")
print(f"Rationale: {vote.rationale}")
```

### `llm_build_action_plan(signal_json, style="balanced") -> LlmActionPlan`
Generate structured action plan for a trade signal.

```python
from swing_agent.llm_predictor import llm_build_action_plan

plan = llm_build_action_plan(
    signal_json=signal.model_dump(),
    style="balanced"  # "conservative", "balanced", "aggressive"
)

print("Action Plan:")
print(plan.action_plan)
print("\nRisk Notes:")
print(plan.risk_notes)
print("\nScenarios:")
for scenario in plan.scenarios:
    print(f"- {scenario}")
```

## Storage API

### `record_signal(ts: TradeSignal, db_path: str) -> str`
Store a complete trading signal in the database.

```python
from swing_agent.storage import record_signal

signal_id = record_signal(trade_signal, "data/signals.sqlite")
print(f"Stored signal: {signal_id}")
```

### `mark_evaluation(signal_id, db_path, exit_reason, exit_price, exit_time_utc, realized_r)`
Update signal with trade outcome.

```python
from swing_agent.storage import mark_evaluation

mark_evaluation(
    signal_id="abc123",
    db_path="data/signals.sqlite",
    exit_reason="TP",
    exit_price=190.80,
    exit_time_utc="2024-01-16T14:30:00Z",
    realized_r=1.61
)
```

## Backtesting

### `simulate_trade(df, open_idx, side, entry, stop, target, max_hold_bars)`
Simulate a trade execution with realistic fills.

```python
from swing_agent.backtester import simulate_trade
from swing_agent.models import SignalSide

exit_idx, exit_reason, exit_price = simulate_trade(
    df=price_data,
    open_idx=100,              # Bar index for trade entry
    side=SignalSide.LONG,
    entry=185.50,
    stop=182.20,
    target=190.80,
    max_hold_bars=26           # 1 day = ~13 bars for 30m
)

print(f"Exit: {exit_reason} @ {exit_price} after {exit_idx-100} bars")
```

**Exit Reasons:**
- `"TP"`: Take profit hit
- `"SL"`: Stop loss hit  
- `"TIME"`: Maximum holding period reached

## Error Handling

All functions include basic error handling:

```python
try:
    signal = agent.analyze("INVALID_SYMBOL")
except RuntimeError as e:
    print(f"Analysis failed: {e}")

# LLM functions gracefully degrade
try:
    vote = llm_extra_prediction(**features)
except Exception:
    vote = None  # Continue without LLM
```

## Environment Variables

```bash
# Required for LLM features
export OPENAI_API_KEY="sk-..."
export SWING_LLM_MODEL="gpt-4o-mini"  # or "gpt-4", "gpt-3.5-turbo"
```

## Usage Examples

### Complete Signal Generation

```python
from swing_agent.agent import SwingAgent

# Initialize agent
agent = SwingAgent(
    interval="30m",
    lookback_days=30,
    log_db="data/signals.sqlite",
    vec_db="data/vec_store.sqlite",
    use_llm=True,
    sector_symbol="QQQ"
)

# Generate signal
signal = agent.analyze("AAPL")

# Access results
if signal.entry:
    print(f"Entry: {signal.entry.side} @ {signal.entry.entry_price}")
    print(f"Stop: {signal.entry.stop_price}")
    print(f"Target: {signal.entry.take_profit}")
    print(f"R-Multiple: {signal.entry.r_multiple}")

if signal.expected_r:
    print(f"Expected R: {signal.expected_r}")
    print(f"Win Rate: {signal.expected_winrate:.2%}")

if signal.action_plan:
    print(f"Action Plan: {signal.action_plan}")
```

### Custom Analysis Pipeline

```python
from swing_agent.data import load_ohlcv
from swing_agent.strategy import label_trend, build_entry
from swing_agent.indicators import fibonacci_range
from swing_agent.features import build_setup_vector

# 1. Load data
df = load_ohlcv("AAPL", "30m", 30)

# 2. Technical analysis
trend = label_trend(df)
entry = build_entry(df, trend)
fib = fibonacci_range(df, lookback=40)

# 3. Feature engineering
vector = build_setup_vector(
    price=df["close"].iloc[-1],
    trend=trend,
    entry=entry,
    atr_pct=0.012,
    session_bin=1,
    llm_conf=0.0
)

print(f"Trend: {trend.label}")
print(f"Entry: {entry.side if entry else 'None'}")
print(f"Feature Vector: {vector}")
```