# Real-World Use Cases for SwingAgent

This guide demonstrates how to use SwingAgent in different trading scenarios and market conditions.

## Use Case 1: Technology Stock Momentum Trading

### Scenario
You want to trade momentum breakouts in technology stocks during earnings season.

### Setup
```bash
# Monitor major tech stocks
python scripts/run_swing_agent.py --symbol AAPL --interval 30m --sector QQQ
python scripts/run_swing_agent.py --symbol MSFT --interval 30m --sector QQQ
python scripts/run_swing_agent.py --symbol NVDA --interval 30m --sector QQQ
python scripts/run_swing_agent.py --symbol GOOGL --interval 30m --sector QQQ
python scripts/run_swing_agent.py --symbol META --interval 30m --sector QQQ
```

### What to Look For
- **High confidence signals** (>75%) with strong momentum
- **MTF alignment** of 2 (both 15m and 1h trends aligned)
- **RSI between 50-80** (momentum without being overbought)
- **R-multiple > 1.5** for good risk/reward

### Example Signal Analysis
```json
{
  "symbol": "NVDA",
  "trend": {"label": "up", "rsi_14": 68.5},
  "entry": {
    "side": "long",
    "entry_price": 875.50,
    "stop_price": 855.20,
    "take_profit": 905.80,
    "r_multiple": 1.49
  },
  "confidence": 0.82,
  "mtf_alignment": 2,
  "vol_regime": "H",
  "action_plan": "Strong momentum breakout above $870 resistance..."
}
```

**Why This Works:**
- High confidence (82%) indicates strong pattern match
- MTF alignment confirms trend on multiple timeframes
- High volatility regime ("H") is typical during earnings season
- Clear risk management with defined stop and target

### Trading Plan
1. **Entry**: Between market open and 10:30 AM for best volume
2. **Position Size**: Reduce by 25% due to high volatility
3. **Management**: Take 50% profits at 1R, let rest run to target
4. **Exit**: Close position by end of day 2 regardless of target

## Use Case 2: Defensive Sector Rotation

### Scenario
Market uncertainty is increasing, and you want to trade defensive sectors while avoiding high-beta stocks.

### Setup
```bash
# Focus on defensive sectors
python scripts/run_swing_agent.py --symbol XLU --interval 30m  # Utilities
python scripts/run_swing_agent.py --symbol KO --interval 30m --sector XLP   # Consumer staples
python scripts/run_swing_agent.py --symbol JNJ --interval 30m --sector XLV  # Healthcare
python scripts/run_swing_agent.py --symbol PG --interval 30m --sector XLP   # Consumer staples
python scripts/run_swing_agent.py --symbol WMT --interval 30m --sector XLP  # Consumer staples
```

### What to Look For
- **Medium volatility regime** ("M") for stability
- **Steady uptrends** with consistent EMA support
- **Win rates > 60%** (defensive stocks tend to be more predictable)
- **Lower R-multiples acceptable** (1.2-1.5) due to lower volatility

### Example Signal Analysis
```json
{
  "symbol": "KO",
  "trend": {"label": "up", "rsi_14": 58.2},
  "entry": {
    "side": "long",
    "entry_price": 62.80,
    "stop_price": 61.20,
    "take_profit": 65.60,
    "r_multiple": 1.25
  },
  "confidence": 0.68,
  "expected_winrate": 0.64,
  "vol_regime": "M",
  "action_plan": "Steady uptrend in defensive name during market uncertainty..."
}
```

### Trading Plan
1. **Larger Position Sizes**: Lower volatility allows for larger positions
2. **Longer Holds**: Consider holding for full 2 days due to steady trends
3. **Multiple Positions**: Can hold 3-4 defensive positions simultaneously
4. **Stop Management**: Less aggressive stop management due to lower volatility

## Use Case 3: Mean Reversion in ETFs

### Scenario
Major ETFs have pulled back to support levels, and you want to trade the bounce.

### Setup
```bash
# Monitor major ETFs for mean reversion
python scripts/run_swing_agent.py --symbol SPY --interval 30m
python scripts/run_swing_agent.py --symbol QQQ --interval 30m
python scripts/run_swing_agent.py --symbol IWM --interval 30m
python scripts/run_swing_agent.py --symbol XLF --interval 30m
python scripts/run_swing_agent.py --symbol XLE --interval 30m
```

### What to Look For
- **Fibonacci golden pocket entries** (fib_golden_low to fib_golden_high)
- **RSI < 40** indicating oversold conditions
- **Recent downtrend** with price approaching EMA support
- **High R-multiples** (>2.0) due to strong support levels

### Example Signal Analysis
```json
{
  "symbol": "SPY",
  "trend": {"label": "down", "rsi_14": 35.8},
  "entry": {
    "side": "long",
    "entry_price": 445.20,
    "stop_price": 440.50,
    "take_profit": 455.80,
    "r_multiple": 2.25,
    "fib_golden_low": 444.50,
    "fib_golden_high": 446.10
  },
  "confidence": 0.71,
  "vol_regime": "M",
  "action_plan": "Oversold bounce from 61.8% Fibonacci level..."
}
```

### Trading Plan
1. **Wait for Golden Pocket**: Only enter within the Fibonacci zone
2. **Quick Exits**: Mean reversion trades can reverse quickly
3. **Scale Out**: Take 75% profits at 1.5R, let 25% run to target
4. **Tight Management**: Move stop to breakeven once trade moves 0.5R in your favor

## Use Case 4: Post-Earnings Continuation

### Scenario
A stock has reported earnings and moved strongly. You want to catch the continuation move.

### Setup
```bash
# After earnings announcement, generate fresh signal
python scripts/run_swing_agent.py --symbol AAPL --interval 15m --lookback-days 15 --sector QQQ
```

### What to Look For
- **Fresh signals** generated after earnings (not pre-earnings setups)
- **15-minute timeframe** for more precise entry timing
- **High volume** confirmation in the action plan
- **Strong MTF alignment** as trend establishes

### Example Signal Analysis
```json
{
  "symbol": "AAPL",
  "asof": "2024-01-25T10:15:00+00:00",  # Day after earnings
  "trend": {"label": "up", "rsi_14": 72.3},
  "entry": {
    "side": "long",
    "entry_price": 195.80,
    "stop_price": 190.20,
    "take_profit": 208.40,
    "r_multiple": 2.25
  },
  "confidence": 0.76,
  "vol_regime": "H",
  "action_plan": "Post-earnings momentum continuation above $195 with high volume confirmation..."
}
```

### Trading Plan
1. **Fast Execution**: Enter quickly as momentum can fade
2. **Volume Confirmation**: Ensure high volume supports the move
3. **Tight Timeframe**: Use 15m charts for precision
4. **Quick Management**: Take profits aggressively (50% at 1R)

## Use Case 5: Sector Rotation Strategy

### Scenario
You want to systematically rotate between sectors based on relative strength.

### Weekly Sector Scan
```bash
# Monday morning sector analysis
python scripts/run_swing_agent.py --symbol XLK --interval 1h --lookback-days 30  # Technology
python scripts/run_swing_agent.py --symbol XLF --interval 1h --lookback-days 30  # Financial
python scripts/run_swing_agent.py --symbol XLE --interval 1h --lookback-days 30  # Energy
python scripts/run_swing_agent.py --symbol XLV --interval 1h --lookback-days 30  # Healthcare
python scripts/run_swing_agent.py --symbol XLI --interval 1h --lookback-days 30  # Industrial
python scripts/run_swing_agent.py --symbol XLP --interval 1h --lookback-days 30  # Consumer Staples
python scripts/run_swing_agent.py --symbol XLRE --interval 1h --lookback-days 30 # Real Estate
python scripts/run_swing_agent.py --symbol XLU --interval 1h --lookback-days 30  # Utilities
python scripts/run_swing_agent.py --symbol XLB --interval 1h --lookback-days 30  # Materials
```

### Sector Selection Criteria
1. **Rank by confidence**: Start with highest confidence sectors
2. **Check relative strength**: Prefer sectors outperforming SPY
3. **Assess momentum**: Look for uptrending sectors with RSI 45-75
4. **Evaluate risk/reward**: Target R-multiples > 1.3

### Stock Selection Within Sector
```bash
# Once you identify strong sector (e.g., XLF), pick individual stocks
python scripts/run_swing_agent.py --symbol JPM --interval 30m --sector XLF
python scripts/run_swing_agent.py --symbol BAC --interval 30m --sector XLF
python scripts/run_swing_agent.py --symbol WFC --interval 30m --sector XLF
python scripts/run_swing_agent.py --symbol GS --interval 30m --sector XLF
```

## Use Case 6: Low Volatility Grinding Trades

### Scenario
Market is in a low volatility, steady uptrend. You want to capture consistent small gains.

### Setup
```bash
# Focus on steady, low-vol names
python scripts/run_swing_agent.py --symbol MSFT --interval 1h --lookback-days 45
python scripts/run_swing_agent.py --symbol AAPL --interval 1h --lookback-days 45
python scripts/run_swing_agent.py --symbol COST --interval 1h --lookback-days 45
python scripts/run_swing_agent.py --symbol HD --interval 1h --lookback-days 45
```

### What to Look For
- **Low volatility regime** ("L")
- **Consistent uptrends** with steady EMA support
- **High win rates** (>65%) even with lower R-multiples
- **Longer timeframes** (1h) for smoother signals

### Trading Strategy
1. **Size Up**: Lower volatility allows larger positions
2. **Hold Longer**: Can hold full 2 days comfortably
3. **Multiple Positions**: Run 4-5 positions simultaneously
4. **Target Management**: Take full profits at targets (less likely to extend)

## Use Case 7: Event-Driven Trading

### Scenario
Trading around known events like FDA approvals, product launches, or sector-specific news.

### Pre-Event Setup
```bash
# Day before FDA approval decision for biotech
python scripts/run_swing_agent.py --symbol GILD --interval 15m --lookback-days 10 --sector XBI
```

### Risk Management for Events
- **Reduce position size** by 50% due to binary outcome risk
- **Tighter stops** (risk only 1-2% per trade instead of 3-4%)
- **Quick exits** if event doesn't go as expected
- **Avoid holding through events** unless specifically trading the outcome

## Performance Tracking Across Use Cases

### Monthly Review
```bash
# Generate comprehensive analysis
python scripts/analyze_performance.py

# Look at performance by volatility regime
# This tells you which use cases (high/medium/low vol) work best for you
```

### Key Metrics to Track
1. **Win Rate by Use Case**: Which scenarios work best for you?
2. **Average R by Volatility**: Are you adapting position size correctly?
3. **Time in Trade**: Are you holding too long or too short?
4. **Sector Performance**: Which sectors provide the most consistent signals?

### Adaptation Strategies
- **If tech momentum isn't working**: Focus more on defensive sectors
- **If mean reversion is failing**: Look for trending continuation setups
- **If win rate is low**: Increase minimum confidence threshold
- **If R-multiple is poor**: Be more selective with risk/reward ratios

## Common Mistakes to Avoid

### 1. Use Case Mixing
- Don't trade momentum strategies in low volatility environments
- Don't use mean reversion tactics during strong trends
- Match your strategy to current market conditions

### 2. Ignoring Volatility Regime
- "H" (High): Reduce position size, take profits faster
- "M" (Medium): Standard approach works well
- "L" (Low): Can size up, hold longer

### 3. Sector Blindness
- Technology stocks behave differently than utilities
- Adjust expectations based on sector characteristics
- Use appropriate sector ETFs for relative strength comparison

### 4. Timeframe Confusion
- 15m: Best for precise entries, requires active monitoring
- 30m: Good balance of precision and practicality
- 1h: Smoother signals, good for part-time traders

Remember: The key to successful use case application is matching the strategy to current market conditions and your own trading style and availability.