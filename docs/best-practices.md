# Best Practices for SwingAgent Trading

This guide provides professional tips and proven strategies for getting the most out of SwingAgent while maintaining proper risk management.

## Signal Selection Best Practices

### Quality Over Quantity
The most successful SwingAgent users are highly selective about which signals they trade.

**High-Quality Signal Criteria:**
- **Confidence ≥ 70%**: Strong pattern match with historical data
- **R-Multiple ≥ 1.5**: Favorable risk/reward ratio
- **Expected Win Rate ≥ 55%**: Better than coin flip probability
- **MTF Alignment = 2**: Both timeframes confirm the setup
- **Clear Entry Zone**: Specific price range, not a single price point

**Example of a Premium Signal:**
```json
{
  "symbol": "AAPL",
  "confidence": 0.78,
  "entry": {
    "r_multiple": 1.82,
    "fib_golden_low": 184.20,
    "fib_golden_high": 186.10
  },
  "expected_winrate": 0.62,
  "mtf_alignment": 2,
  "vol_regime": "M"
}
```

### Signal Filtering Strategy
Don't take every signal. Use this hierarchy:

1. **Tier 1 (Take immediately)**: Confidence >80%, R-multiple >2.0
2. **Tier 2 (Take selectively)**: Confidence 70-80%, R-multiple 1.5-2.0
3. **Tier 3 (Consider carefully)**: Confidence 60-70%, R-multiple 1.3-1.5
4. **Tier 4 (Usually skip)**: Below Tier 3 thresholds

## Position Sizing Guidelines

### Base Position Size Formula
```
Position Size = (Account Risk %) ÷ (Entry Price - Stop Price) × Account Value
```

**Example:**
- Account: $10,000
- Risk per trade: 2%
- Entry: $100
- Stop: $96
- Position Size = 2% ÷ $4 × $10,000 = 50 shares

### Volatility Adjustments
Adjust your base position size based on volatility regime:

- **Low Volatility ("L")**: +25% of base size
- **Medium Volatility ("M")**: Base size
- **High Volatility ("H")**: -25% of base size

### Confidence Adjustments
Further adjust based on signal confidence:

- **Confidence >80%**: +20% of calculated size
- **Confidence 70-80%**: Calculated size
- **Confidence 60-70%**: -20% of calculated size
- **Confidence <60%**: Consider skipping

## Entry Execution Best Practices

### Timing Your Entries

**Best Entry Times:**
- **9:30-10:30 AM**: High volume, clear direction establishment
- **2:00-3:00 PM**: Institutional activity, trend continuation
- **Avoid**: First 5 minutes (too volatile), last 30 minutes (unpredictable)

### Entry Zone Strategy
SwingAgent provides entry ranges (e.g., $184.20-$186.10). Use this approach:

1. **Wait for the zone**: Don't chase above the high end
2. **Scale in**: Enter 60% at zone midpoint, 40% at favorable end
3. **Use limit orders**: Avoid market orders during volatile periods
4. **Set stops immediately**: Place stop-loss order as soon as filled

**Example Entry Plan:**
```bash
# Signal shows entry zone $184.20-$186.10
# Place limit orders:
# 60% of position at $185.15 (midpoint)
# 40% of position at $184.50 (favorable end)
# Stop loss at $182.20 for all shares
```

## Risk Management Best Practices

### The 2% Rule
Never risk more than 2% of your account on any single trade. This allows for a string of losses without devastating your account.

**Why 2% Works:**
- 10 consecutive losses = -18% account drawdown (recoverable)
- Allows for statistical edge to play out over time
- Prevents emotional decision-making

### Stop-Loss Discipline
SwingAgent calculates technical stop levels. Follow them religiously:

1. **Set stops immediately** upon entry
2. **Never move stops against you** (only in your favor)
3. **Don't hope for reversals** - cut losses quickly
4. **Honor the stop even if** you "know" the stock will bounce

### Position Correlation Management
Avoid taking multiple highly correlated positions:

**Poor Diversification:**
```bash
# All tech stocks - highly correlated
AAPL long
MSFT long  
GOOGL long
NVDA long
```

**Better Diversification:**
```bash
# Mixed sectors - lower correlation
AAPL long (tech)
JPM long (finance)
XLE long (energy)
WMT long (consumer staples)
```

## Profit-Taking Strategies

### The 50-25-25 Method
This is the most popular profit-taking strategy among successful SwingAgent users:

- **50% at 1R**: Lock in profit when trade moves in your favor by the amount you risked
- **25% at 1.5R**: Take additional profits as momentum continues
- **25% at target**: Let remaining position run to full target

**Example:**
- Entry: $100, Stop: $96, Target: $108 (Risk: $4, Reward: $8, R-multiple: 2.0)
- Take 50% profit at $104 (1R = $4 profit)
- Take 25% profit at $106 (1.5R = $6 profit)
- Let 25% run to $108 target (2R = $8 profit)

### Trailing Stop Strategy
For strong trending moves, consider trailing your stop:

1. **After 1R profit**: Move stop to breakeven
2. **After 1.5R profit**: Move stop to +0.5R
3. **After 2R profit**: Move stop to +1R

## Daily and Weekly Routines

### Morning Routine (15 minutes)
```bash
# 1. Check major market ETFs for overall direction
python scripts/run_swing_agent.py --symbol SPY --interval 30m
python scripts/run_swing_agent.py --symbol QQQ --interval 30m

# 2. Scan your watchlist sectors
python scripts/run_swing_agent.py --symbol XLF --interval 30m
python scripts/run_swing_agent.py --symbol XLE --interval 30m
python scripts/run_swing_agent.py --symbol XLV --interval 30m

# 3. Check individual opportunities
python scripts/run_swing_agent.py --symbol AAPL --interval 30m --sector QQQ
python scripts/run_swing_agent.py --symbol MSFT --interval 30m --sector QQQ
```

### Evening Routine (10 minutes)
```bash
# Review positions and update stops
# Plan tomorrow's potential entries
# Check any pending signals that may trigger
```

### Weekly Review (30 minutes)
```bash
# Evaluate closed trades
python scripts/eval_signals.py --max-hold-days 2.0

# Analyze overall performance
python scripts/analyze_performance.py

# Review and update watchlists
# Plan next week's focus sectors
```

## Market Condition Adaptations

### Bull Market Best Practices
- **Focus on momentum signals** (trend continuation)
- **Hold positions longer** (closer to full targets)
- **Size up slightly** on high-confidence long signals
- **Avoid shorting** unless extremely high confidence

### Bear Market Best Practices
- **Focus on mean reversion** and short signals
- **Take profits faster** (markets can reverse quickly)
- **Reduce position sizes** (higher overall volatility)
- **Watch for false breakouts** (more common in bear markets)

### Sideways Market Best Practices
- **Focus on range-bound strategies**
- **Quick profit-taking** (trends don't last long)
- **Higher selectivity** (fewer quality setups)
- **Consider ETF trading** (less single-stock risk)

## Advanced Techniques

### Sector Rotation Strategy
Monitor sector strength weekly:

```bash
# Weekly sector strength scan
for sector in QQQ XLF XLE XLV XLI XLP XLRE XLU XLB; do
  python scripts/run_swing_agent.py --symbol $sector --interval 1h --lookback-days 30
done
```

Focus your individual stock picks from the strongest sectors.

### Multi-Timeframe Confluence
For highest-probability signals, check multiple timeframes:

```bash
# Same stock, different timeframes
python scripts/run_swing_agent.py --symbol AAPL --interval 15m --lookback-days 10
python scripts/run_swing_agent.py --symbol AAPL --interval 30m --lookback-days 20
python scripts/run_swing_agent.py --symbol AAPL --interval 1h --lookback-days 40
```

Take positions only when 2+ timeframes agree.

### Earnings Season Adjustments
During earnings season (first 3 weeks of each quarter):

1. **Reduce position sizes** by 25-50%
2. **Avoid holding through earnings** unless specifically trading the event
3. **Focus on stocks that have already reported**
4. **Increase stop-loss monitoring** (gap risk is higher)

## Common Mistakes to Avoid

### Overtrading
**Problem**: Taking too many signals, especially marginal ones.
**Solution**: Set daily/weekly trade limits. Quality over quantity.

### Ignoring Volatility Regimes
**Problem**: Using same position size regardless of market volatility.
**Solution**: Always adjust size based on vol_regime (L/M/H).

### Emotional Override
**Problem**: Overriding system signals based on "gut feeling."
**Solution**: Trust the system or don't trade. Consistency is key.

### Poor Record Keeping
**Problem**: Not tracking results systematically.
**Solution**: Use a simple spreadsheet or the built-in evaluation tools.

### Revenge Trading
**Problem**: Increasing size after losses to "get even."
**Solution**: Stick to position sizing rules. One trade doesn't affect the next.

## Performance Tracking

### Key Metrics to Monitor
1. **Overall Win Rate**: Should match system expectations
2. **Average R per Trade**: Aim for >0.3R over time
3. **Maximum Drawdown**: Keep under 10-15%
4. **Trade Frequency**: 3-8 trades per week typically optimal
5. **Sector Performance**: Which sectors work best for you?

### Monthly Review Questions
1. Are you following entry and exit rules consistently?
2. Is your actual win rate close to expected win rates?
3. Are you sizing positions appropriately for volatility?
4. Which signal types are working best for you?
5. What can you improve next month?

## Psychology and Discipline

### Maintain Trading Discipline
- **Pre-plan every trade**: Entry, stop, target, size
- **Accept losses as part of the business**
- **Don't let emotions drive decisions**
- **Take breaks after emotional trades**
- **Focus on process, not profits**

### Building Confidence
- **Start small**: Use minimal position sizes while learning
- **Paper trade first**: Practice without money at risk
- **Track your improvement**: See your skills develop over time
- **Focus on consistency**: Consistent profits beat home runs

### Handling Drawdowns
Every trader experiences losing streaks:

1. **Expect them**: 5-10 consecutive losses can happen
2. **Don't change the system**: Maintain discipline
3. **Reduce size if needed**: But don't stop trading completely
4. **Review but don't overthink**: Learn but don't over-optimize
5. **Focus on process**: Trust that edge will reassert itself

Remember: SwingAgent provides the analysis, but your discipline and consistency in following best practices will determine your long-term success.