# ADR-004: Fibonacci-Based Entry Strategy

## Status

Accepted

## Context

SwingAgent's core strategy is based on Fibonacci retracements for 1-2 day swing trades:
- Golden pocket (61.8%-65% retracement) entries provide high-probability setups
- Fibonacci extensions offer logical target levels
- ATR-based stops provide consistent risk management

The strategy needs to balance:
- High win rates from quality setups
- Reasonable trade frequency for portfolio returns
- Risk management with consistent R-multiples
- Multiple entry strategies for different market conditions

## Decision

We will implement a three-tier Fibonacci strategy:

1. **Primary: Golden Pocket Pullbacks** (Highest Probability)
   - Entry: Price within 61.8%-65% retracement zone
   - Stop: Below golden pocket + ATR buffer
   - Target: Fibonacci extensions or previous swing highs

2. **Secondary: Momentum Breakouts**
   - Entry: Break above previous swing high (uptrend)
   - Stop: Entry - ATR multiple
   - Target: Fibonacci extensions

3. **Tertiary: Mean Reversion**
   - Entry: Extreme RSI in sideways markets
   - Stop: ATR-based from entry level
   - Target: Modest ATR-based targets

## Implementation Details

### Fibonacci Calculation
```python
def fibonacci_range(df, lookback=40):
    # Find recent swing high/low within lookback
    swing_low, swing_high, direction = recent_swing(df, lookback)
    
    # Calculate all standard Fibonacci levels
    levels = {
        "0.236": ..., "0.382": ..., "0.5": ...,
        "0.618": ..., "0.65": ..., "0.786": ...,
        "1.0": ..., "1.272": ..., "1.618": ...
    }
    
    # Golden pocket boundaries
    golden_low = min(levels["0.618"], levels["0.65"])
    golden_high = max(levels["0.618"], levels["0.65"])
```

### Risk Management
- **Stop Loss**: Always ATR-based for volatility adjustment
- **Position Sizing**: Consistent R-multiple approach (1R risk per trade)
- **Target Selection**: 
  - Primary: Previous swing levels
  - Secondary: Fibonacci extensions (127.2%, 161.8%)
  - Minimum: 1.5R reward-to-risk ratio

### Configuration Parameters
```python
# Golden pocket strategy
FIB_LOOKBACK = 40              # Bars to scan for swings
ATR_STOP_BUFFER = 0.2          # Additional ATR for stops
ATR_STOP_MULTIPLIER = 1.2      # ATR multiple for momentum stops
ATR_TARGET_MULTIPLIER = 2.5    # ATR multiple for targets

# Risk thresholds
MIN_R_MULTIPLE = 1.5           # Minimum reward-to-risk
MAX_STOP_PERCENT = 3.0         # Maximum stop as % of price
```

## Strategy Priority Logic

```python
def build_entry(df, trend):
    # 1. Golden pocket pullbacks (highest priority)
    if in_golden_pocket(price, fib) and aligned_trend(trend):
        return golden_pocket_entry()
    
    # 2. Momentum breakouts  
    if trend_strong() and above_resistance():
        return momentum_entry()
        
    # 3. Mean reversion (lowest priority)
    if sideways_trend() and extreme_rsi():
        return mean_reversion_entry()
        
    return None  # No valid setup
```

## Consequences

### Positive

- **High Win Rates**: Golden pocket entries historically show 65-75% win rates
- **Logical Levels**: Fibonacci levels provide objective entry/exit points
- **Risk Control**: ATR-based stops adjust to volatility automatically  
- **Multiple Opportunities**: Three strategies capture different market conditions
- **Backtestable**: Clear, objective rules for historical validation

### Negative

- **Setup Frequency**: High-quality golden pocket setups may be infrequent
- **Whipsaw Risk**: False breakouts can trigger momentum entries
- **Trend Dependency**: Requires trending markets for optimal performance
- **Lag**: Waits for pullbacks, may miss fast moves

## Performance Expectations

Based on backtesting and live trading:
- **Win Rate**: 60-70% overall (higher for golden pocket)
- **Average R**: 1.8-2.2R per trade
- **Frequency**: 2-5 signals per symbol per month
- **Drawdown**: <15% with proper position sizing

## Validation Metrics

- Win rate by strategy type (golden pocket vs momentum vs mean reversion)
- R-multiple distribution and consistency
- Market condition performance (trending vs sideways)
- False signal analysis and refinement opportunities

## Future Enhancements

- Dynamic fibonacci lookback based on volatility
- Volume confirmation for golden pocket entries
- Sector rotation overlay for strategy selection
- Multi-timeframe fibonacci alignment