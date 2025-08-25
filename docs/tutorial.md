# SwingAgent Tutorial: Your First Week of Trading

This hands-on tutorial will walk you through a complete week of using SwingAgent, from generating your first signals to evaluating results.

## Day 1: Setting Up and Understanding Signals

### Morning Setup (5 minutes)

Let's start by scanning a few popular stocks for potential setups:

```bash
# Scan some technology stocks
python scripts/run_swing_agent.py --symbol AAPL --interval 30m --lookback-days 30 --sector QQQ
python scripts/run_swing_agent.py --symbol MSFT --interval 30m --lookback-days 30 --sector QQQ
python scripts/run_swing_agent.py --symbol GOOGL --interval 30m --lookback-days 30 --sector QQQ
```

### Understanding the Output

Let's say SwingAgent found a signal for AAPL:

```json
{
  "symbol": "AAPL",
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
  "mtf_alignment": 2,
  "vol_regime": "M",
  "action_plan": "Strong uptrend with price holding above EMA. RSI at 62 shows momentum without being overbought. Entry recommended in the Fibonacci golden pocket between $184.20-$186.10. Stop below recent swing low at $182.20 limits risk to $3.30 per share. Target $190.80 offers 1.61R reward-to-risk ratio. Monitor for entry between market open and 11 AM when volume is typically highest."
}
```

#### What This Tells You:

**The Setup:**
- AAPL is in an **uptrend** (trend.label = "up")
- Price is **above the moving average** (price_above_ema = true)
- **Momentum is healthy** (RSI = 62.3, not overbought)
- **Medium volatility** environment (vol_regime = "M")

**The Trade Plan:**
- **Direction**: Long (buy)
- **Entry Zone**: $184.20 - $186.10 (Fibonacci golden pocket)
- **Stop Loss**: $182.20 (risk = $3.30 per share if entering at $185.50)
- **Take Profit**: $190.80 (reward = $5.30 per share)
- **Risk/Reward**: 1.61 (risk $1 to potentially make $1.61)

**The Confidence:**
- **System Confidence**: 72% (fairly high)
- **Historical Expected Return**: 0.95R (historically, similar setups averaged 95% of risk as profit)
- **Win Rate**: 58% (similar setups were profitable 58% of the time)

### Your Decision Process

Ask yourself:
1. **Do I understand the trade?** ✓ Buy AAPL between $184.20-$186.10
2. **Is the risk acceptable?** If you're willing to lose $3.30 per share
3. **Does the timeframe fit?** This is a 1-2 day swing trade
4. **Do I like the setup?** 58% win rate with 1.61 reward-to-risk is reasonable

## Day 2: Monitoring and Entry

### Morning Check

Check if AAPL is trading in your entry zone:

```bash
# Generate a fresh signal to see current price
python scripts/run_swing_agent.py --symbol AAPL --interval 30m --lookback-days 30 --sector QQQ
```

### Entry Scenarios

**Scenario A: Price is at $184.80 (in the golden pocket)**
- ✅ **Enter the trade**
- Place stop at $182.20
- Set target at $190.80
- Position size based on your $3.30 risk per share

**Scenario B: Price is at $187.50 (above the entry zone)**
- ❌ **Skip this trade**
- The risk/reward is no longer favorable
- Wait for the next opportunity

**Scenario C: Price is at $180.00 (below the entry zone)**
- ⏳ **Wait and watch**
- The setup may be invalidated
- Generate a new signal to see if conditions changed

## Day 3: Position Management

Let's say you entered AAPL at $184.80 yesterday.

### Morning Check

```bash
# See how AAPL is performing
python scripts/run_swing_agent.py --symbol AAPL --interval 30m --lookback-days 30 --sector QQQ
```

### Management Scenarios

**Scenario A: AAPL is at $189.50 (near target)**
- Consider taking partial profits (sell 50% of position)
- Move stop loss to breakeven ($184.80)
- Let the rest run to full target

**Scenario B: AAPL is at $183.00 (near stop)**
- Prepare to exit if it hits $182.20
- Don't hope it will turn around
- Stick to your plan

**Scenario C: AAPL is at $186.20 (moving in your favor)**
- Hold the position
- Consider moving stop to $183.50 (original entry zone low)
- Let it run toward target

## Day 4: Generating New Signals

Whether your AAPL trade is still active or closed, let's find new opportunities:

```bash
# Scan different sectors
python scripts/run_swing_agent.py --symbol XLF --interval 30m --lookback-days 30  # Financial sector ETF
python scripts/run_swing_agent.py --symbol JPM --interval 30m --lookback-days 30 --sector XLF
python scripts/run_swing_agent.py --symbol BAC --interval 30m --lookback-days 30 --sector XLF

# Try energy sector
python scripts/run_swing_agent.py --symbol XLE --interval 30m --lookback-days 30  # Energy sector ETF
python scripts/run_swing_agent.py --symbol XOM --interval 30m --lookback-days 30 --sector XLE
```

### Signal Quality Assessment

Rate each signal:

**High Quality Signal:**
- Confidence > 70%
- R-multiple > 1.5
- Win rate > 55%
- Clear entry zone
- MTF alignment ≥ 2

**Medium Quality Signal:**
- Confidence 50-70%
- R-multiple 1.2-1.5
- Win rate 45-55%

**Low Quality Signal:**
- Confidence < 50%
- R-multiple < 1.2
- Win rate < 45%

## Day 5: Backtesting and Analysis

Let's generate some historical signals to see how the system performed:

```bash
# Generate 180 days of historical signals for AAPL (this takes a few minutes)
python scripts/backtest_generate_signals.py --symbol AAPL --interval 30m --lookback-days 180 --warmup-bars 80 --sector QQQ --no-llm
```

Then evaluate the historical performance:

```bash
# Evaluate all historical signals
python scripts/eval_signals.py --max-hold-days 2.0

# Analyze the results
python scripts/analyze_performance.py
```

### Understanding Backtest Results

The analysis will show you:

**Overall Performance:**
- Total number of signals
- Win rate percentage
- Average R per trade
- Best and worst trades

**By Volatility Regime:**
- How signals performed in Low/Medium/High volatility environments
- Which regimes were most profitable

**By Confidence Buckets:**
- How well the system's confidence predictions matched reality
- Whether high-confidence signals actually performed better

## Weekend: Review and Planning

### Weekly Review Questions

1. **What did I learn about the system?**
   - Which signal types worked best?
   - What market conditions were most favorable?

2. **How was my execution?**
   - Did I follow the entry rules?
   - Did I stick to stop losses?
   - Did I take profits at targets?

3. **What would I do differently?**
   - Position sizing adjustments?
   - Different stocks or sectors?
   - Different timeframes?

### Next Week Planning

```bash
# Create a watchlist of sectors to monitor
python scripts/run_swing_agent.py --symbol QQQ --interval 30m --lookback-days 30  # Tech
python scripts/run_swing_agent.py --symbol XLF --interval 30m --lookback-days 30  # Finance
python scripts/run_swing_agent.py --symbol XLE --interval 30m --lookback-days 30  # Energy
python scripts/run_swing_agent.py --symbol XLI --interval 30m --lookback-days 30  # Industrial
python scripts/run_swing_agent.py --symbol XLV --interval 30m --lookback-days 30  # Healthcare
```

## Pro Tips from Your First Week

### 1. Quality Over Quantity
- It's better to take one high-quality signal than three mediocre ones
- High confidence (>70%) signals with good R-multiples (>1.5) tend to work best

### 2. Respect the Stop Loss
- Always set your stop loss before entering
- Never move it against you
- The system calculates stops based on technical levels

### 3. Use the Action Plan
- Read the AI-generated action plan carefully
- It often contains important timing and context information
- Use it to understand the reasoning behind the signal

### 4. Track Your Results
- Keep a simple trading journal
- Note which signals you took and why
- Compare your actual results to the system's expectations

### 5. Start Small
- Use small position sizes while learning
- Focus on learning the system rather than making money initially
- Increase size only after you're comfortable with the process

## What's Next?

After your first week, you might want to explore:

- **[Use Cases](use-cases.md)** - Advanced trading scenarios
- **[Best Practices](best-practices.md)** - Professional trading tips
- **[Configuration](configuration.md)** - Customizing the system
- **[FAQ](faq.md)** - Common questions and solutions

Remember: The goal of this first week is to learn how SwingAgent works and develop confidence in using it. Focus on understanding the signals rather than maximizing profits.