# Getting Started with SwingAgent

Welcome to SwingAgent v1.6.1! This guide will help you get up and running with the swing trading system in just a few steps.

## What is SwingAgent?

SwingAgent is an automated swing trading system designed for 1-2 day trades. It combines:

- **Technical Analysis**: Fibonacci retracements, trend analysis, momentum indicators
- **Machine Learning**: Pattern recognition based on historical similar setups
- **AI Integration**: OpenAI-powered explanations and action plans
- **Risk Management**: Automatic stop-loss and take-profit calculations

## Prerequisites

Before you begin, make sure you have:

- **Python 3.10 or higher** installed on your computer
- **Basic familiarity with trading concepts** (entry, stop-loss, take-profit)
- **Optional**: OpenAI API key for enhanced AI features

## Quick Installation

### Step 1: Download SwingAgent

```bash
# Option A: Clone from GitHub (recommended)
git clone https://github.com/ITlusions/ITL.SwingAgent.git
cd ITL.SwingAgent

# Option B: Download as ZIP and extract
# Download from: https://github.com/ITlusions/ITL.SwingAgent/archive/main.zip
```

### Step 2: Set Up Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv swingagent-env

# Activate the environment
# On Windows:
swingagent-env\Scripts\activate
# On macOS/Linux:
source swingagent-env/bin/activate

# Install SwingAgent
pip install -e .
```

### Step 3: Basic Configuration

Create a data directory for storing signals and results:

```bash
mkdir data
```

### Step 4: Optional - Set Up AI Features

If you want enhanced AI explanations and action plans:

```bash
# Get your API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-your-key-here"
export SWING_LLM_MODEL="gpt-4o-mini"
```

## Your First Signal

Let's generate your first swing trading signal:

```bash
python scripts/run_swing_agent.py --symbol AAPL --interval 30m --lookback-days 30
```

This command will:

1. **Download** 30 days of Apple (AAPL) stock data in 30-minute intervals
2. **Analyze** the current technical setup
3. **Search** for similar historical patterns
4. **Generate** a trading signal with entry, stop-loss, and take-profit levels
5. **Store** the signal in your local database

## Understanding Your First Signal

Your signal output will look something like this:

```json
{
  "symbol": "AAPL",
  "asof": "2024-01-15T15:30:00+00:00",
  "trend": {
    "label": "up",
    "price_above_ema": true,
    "rsi_14": 62.3
  },
  "entry": {
    "side": "long",
    "entry_price": 185.50,
    "stop_price": 182.20,
    "take_profit": 190.80,
    "r_multiple": 1.61
  },
  "confidence": 0.72,
  "expected_r": 0.95,
  "expected_winrate": 0.58,
  "action_plan": "Monitor for entry between $184.20-$186.10..."
}
```

### Key Information:

- **Entry Side**: "long" = buy, "short" = sell
- **Entry Price**: Recommended price to enter the trade
- **Stop Price**: Exit price if trade goes against you (limits loss)
- **Take Profit**: Target price to exit for profit
- **R Multiple**: Risk-to-reward ratio (1.61 means you risk $1 to potentially make $1.61)
- **Confidence**: System's confidence in the signal (0-1 scale)
- **Expected R**: Expected return based on historical similar setups
- **Expected Win Rate**: Percentage of similar historical setups that were profitable
- **Action Plan**: AI-generated explanation and trading plan

## Common Commands

### Generate Signals for Different Stocks

```bash
# Technology stock with tech sector comparison
python scripts/run_swing_agent.py --symbol NVDA --sector QQQ

# Financial stock with financial sector comparison
python scripts/run_swing_agent.py --symbol JPM --sector XLF

# Different timeframes
python scripts/run_swing_agent.py --symbol TSLA --interval 15m  # 15-minute bars
python scripts/run_swing_agent.py --symbol MSFT --interval 1h   # 1-hour bars
```

### View Historical Performance

```bash
# Analyze all stored signals
python scripts/analyze_performance.py
```

### Evaluate Signal Outcomes

After some time has passed, evaluate how your signals performed:

```bash
# Check outcomes of signals with 2-day maximum hold time
python scripts/eval_signals.py --max-hold-days 2.0
```

## What's Next?

Now that you have SwingAgent running, you might want to:

1. **[Read the Tutorial](tutorial.md)** - Learn to interpret and use signals effectively
2. **[Explore Use Cases](use-cases.md)** - See real-world trading scenarios
3. **[Review Best Practices](best-practices.md)** - Get tips for successful swing trading
4. **[Check the FAQ](faq.md)** - Find answers to common questions

## Need Help?

- **Technical Issues**: See the [Troubleshooting Guide](troubleshooting.md)
- **Trading Questions**: Check the [FAQ](faq.md) and [Glossary](glossary.md)
- **Advanced Setup**: Review [Configuration](configuration.md) options

## Safety Reminder

⚠️ **Important**: SwingAgent is a tool to assist with trading decisions. Always:

- Start with small position sizes while learning
- Understand that past performance doesn't guarantee future results
- Consider your risk tolerance and financial situation
- Never risk more than you can afford to lose
- Consider the signals as suggestions, not guaranteed outcomes

Happy trading! 🚀