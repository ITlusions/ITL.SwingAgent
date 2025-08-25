# SwingAgent v1.6.1

A sophisticated 1-2 day swing trading system that combines technical analysis, machine learning pattern matching, and LLM-generated action plans for systematic trading decisions.

## Overview

SwingAgent is a comprehensive trading system designed for short-term swing trades with holding periods of 1-2 days. The system integrates multiple analytical approaches:

- **Technical Analysis**: EMA trends, RSI momentum, ATR volatility, Fibonacci retracements
- **Machine Learning**: Vector-based KNN for historical pattern matching with centralized storage
- **LLM Integration**: OpenAI models for trade explanations and structured action plans
- **Multi-timeframe Analysis**: 15-minute and 1-hour trend alignment
- **Risk Management**: Systematic stop-loss and take-profit calculations
- **Centralized Database**: SQLAlchemy ORM with support for SQLite, PostgreSQL, MySQL, and Kubernetes CNPG

## Key Features

### 🎯 Fibonacci Golden Pocket Strategy
- Uses 0.618-0.65 retracement levels for high-probability entry points
- Combines with momentum and mean-reversion setups
- Dynamic risk-reward calculation based on ATR

### 🧠 ML Pattern Recognition
- Centralized SQLite/PostgreSQL-based vector store for historical pattern matching
- Cosine similarity search across feature vectors
- Statistical expectations based on similar historical setups

### 📊 Multi-timeframe Confluence
- 15-minute and 1-hour trend alignment filtering
- Relative strength analysis vs sector ETF and SPY
- Volatility regime classification (Low/Medium/High)

### 🤖 LLM-Powered Insights
- OpenAI integration for trade explanations
- Structured action plans with entry/exit scenarios
- Risk assessment and invalidation conditions

### 📈 Comprehensive Tracking
- Complete signal database with expectations vs outcomes
- Performance analytics by volatility regime
- Calibration analysis for prediction accuracy
- Centralized storage with migration tools from legacy databases

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ITlusions/ITL.SwingAgent.git
cd ITL.SwingAgent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Optional: Set up OpenAI for LLM features
export OPENAI_API_KEY="sk-..."
export SWING_LLM_MODEL="gpt-4o-mini"
```

### Basic Usage

#### Generate a Live Signal

```bash
python scripts/run_swing_agent.py \
  --symbol AAPL \
  --interval 30m \
  --lookback-days 30 \
  --sector QQQ
```

#### Generate Historical Signals (Backtesting)

```bash
python scripts/backtest_generate_signals.py \
  --symbol AAPL \
  --interval 30m \
  --lookback-days 180 \
  --warmup-bars 80 \
  --sector QQQ \
  --no-llm
```

#### Evaluate Signal Performance

```bash
python scripts/eval_signals.py \
  --max-hold-days 2.0
```

#### Analyze Performance

```bash
python scripts/analyze_performance.py
```

## System Architecture

The system is built with a modular architecture:

```
SwingAgent v1.6.1
├── Core Engine
│   ├── agent.py          # Main orchestrator
│   ├── strategy.py       # Trend & entry logic
│   ├── indicators.py     # Technical calculations
│   └── config.py         # Centralized configuration
├── Database Layer
│   ├── database.py       # SQLAlchemy session management
│   ├── models_db.py      # Database models
│   ├── storage.py        # Signal storage operations
│   └── migrate.py        # Migration utilities
├── Machine Learning
│   ├── features.py       # Feature engineering
│   └── vectorstore.py    # Pattern matching & KNN
├── AI Integration
│   └── llm_predictor.py  # OpenAI integration
├── Data & Backtesting
│   ├── data.py           # Market data fetching
│   └── backtester.py     # Trade simulation
└── Scripts
    ├── run_swing_agent.py
    ├── backtest_generate_signals.py
    ├── eval_signals.py
    ├── analyze_performance.py
    ├── backfill_vector_store.py
    ├── db_info.py
    └── test_cnpg.py
```

## Signal Generation Process

1. **Data Collection**: Fetch OHLCV data via Yahoo Finance
2. **Technical Analysis**: Calculate EMA trends, RSI, ATR, Fibonacci levels
3. **Multi-timeframe Check**: Verify 15m/1h alignment and relative strength
4. **Entry Logic**: Apply Fibonacci golden pocket or momentum strategies
5. **Vector Lookup**: Find similar historical patterns via KNN
6. **LLM Analysis**: Generate explanations and action plans
7. **Signal Storage**: Record complete signal with expectations
8. **Performance Tracking**: Evaluate outcomes and update patterns

## Output Example

```json
{
  "symbol": "AAPL",
  "timeframe": "30m",
  "asof": "2024-01-15T15:30:00+00:00",
  "trend": {
    "label": "up",
    "ema_slope": 0.0156,
    "price_above_ema": true,
    "rsi_14": 62.3
  },
  "entry": {
    "side": "long",
    "entry_price": 185.50,
    "stop_price": 182.20,
    "take_profit": 190.80,
    "r_multiple": 1.61,
    "fib_golden_low": 184.20,
    "fib_golden_high": 186.10
  },
  "confidence": 0.72,
  "expected_r": 0.95,
  "expected_winrate": 0.58,
  "expected_hold_days": 1.2,
  "action_plan": "Monitor for entry between $184.20-$186.10...",
  "mtf_alignment": 2,
  "vol_regime": "M"
}
```

## Getting Started

### For New Users
- **[Getting Started Guide](getting-started.md)** - Complete setup and your first signal
- **[Tutorial](tutorial.md)** - Hands-on walkthrough of your first week
- **[FAQ](faq.md)** - Quick answers to common questions

### For Active Traders  
- **[Use Cases](use-cases.md)** - Real-world trading scenarios and strategies
- **[Best Practices](best-practices.md)** - Professional tips and risk management
- **[Glossary](glossary.md)** - Trading and technical terms explained

### For Technical Users
- **[API Reference](api-reference.md)** - Complete function documentation  
- **[Configuration](configuration.md)** - Setup and customization options
- **[Architecture](architecture.md)** - Deep dive into system design

### For Deployment
- **[Deployment](deployment.md)** - Production deployment guide
- **[Security](security.md)** - Security best practices
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## License

This project is provided as-is for educational and research purposes.