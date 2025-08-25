# Trading and Technical Glossary

This glossary explains trading terms, technical analysis concepts, and SwingAgent-specific terminology.

## SwingAgent-Specific Terms

### Action Plan
AI-generated explanation and trading strategy for a signal. Includes reasoning behind the setup, timing considerations, and risk management guidance.

### Confidence Score
A percentage (0-100%) indicating how closely the current setup matches historically profitable patterns. Higher confidence means more similar successful historical patterns were found.

### Entry Zone / Golden Pocket
The price range where SwingAgent recommends entering a trade, typically based on Fibonacci retracement levels (61.8% to 65%). Provides flexibility rather than a single entry price.

### Expected R
The average return (as a multiple of risk) that similar historical setups have produced. An expected R of 0.8 means similar setups averaged 80% of the risk amount as profit.

### Expected Win Rate
The percentage of similar historical setups that were profitable. Based on pattern matching in SwingAgent's historical database.

### MTF Alignment
Multi-Timeframe Alignment score showing how many timeframes (15-minute and 1-hour) agree on trend direction. Score of 2 = both aligned, 1 = partial alignment, 0 = conflicting.

### R-Multiple
Risk-to-reward ratio. If you risk $100 and target $150 profit, the R-multiple is 1.5. Higher R-multiples indicate better potential reward relative to risk.

### Volatility Regime
Market volatility classification:
- **L (Low)**: Calm, steady markets
- **M (Medium)**: Normal volatility conditions  
- **H (High)**: Elevated volatility, often around events/earnings

## Trading Terms

### Ask/Offer
The price at which someone is willing to sell a security. Always higher than the bid price.

### Average True Range (ATR)
Measure of price volatility over a specific period. Higher ATR means larger price swings. Used for position sizing and stop-loss placement.

### Bid
The price at which someone is willing to buy a security. Always lower than the ask price.

### Bid-Ask Spread
The difference between the bid and ask prices. Wider spreads indicate less liquid securities.

### Breakout
When price moves above resistance or below support levels, often signaling the start of a new trend.

### Day Trading
Opening and closing positions within the same trading day. Different from swing trading which holds positions 1-2 days.

### Entry Price
The price at which you buy or sell to initiate a position.

### Fill
When your order is executed at a specific price. "Getting filled" means your order was completed.

### Gap
A price difference between consecutive trading periods where no trading occurred at intermediate prices.

### Limit Order
An order to buy/sell at a specific price or better. Provides price control but no guarantee of execution.

### Long Position
Owning a security with the expectation that its price will rise. "Going long" means buying.

### Market Order
An order to buy/sell immediately at the current market price. Guarantees execution but not price.

### Position Size
The number of shares or dollar amount invested in a particular trade.

### Short Position
Selling a borrowed security with the expectation of buying it back at a lower price. "Going short" means selling.

### Slippage
The difference between expected trade price and actual execution price, usually due to market movement or low liquidity.

### Stop-Loss Order
An order to sell/buy when the price reaches a certain level, designed to limit losses.

### Take-Profit Order
An order to close a position when it reaches a target profit level.

## Technical Analysis Terms

### Bollinger Bands
Technical indicator showing price channels based on standard deviations from a moving average. Used to identify overbought/oversold conditions.

### Candlestick
Chart representation showing open, high, low, and close prices for a time period in a single bar.

### Exponential Moving Average (EMA)
A moving average that gives more weight to recent prices, making it more responsive to new information than simple moving averages.

### Fibonacci Retracement
Technical analysis tool based on the mathematical Fibonacci sequence. Key levels are 23.6%, 38.2%, 50%, 61.8%, and 78.6%.

### Golden Pocket
The area between 61.8% and 65% Fibonacci retracement levels, often considered high-probability reversal zones.

### Moving Average
The average price over a specific number of periods, used to smooth price action and identify trends.

### Relative Strength Index (RSI)
Momentum oscillator (0-100) measuring speed and change of price movements. Values above 70 suggest overbought conditions, below 30 suggest oversold.

### Resistance
A price level where selling pressure historically emerges, preventing further upward movement.

### Support
A price level where buying interest historically emerges, preventing further downward movement.

### Trend
The general direction of price movement over time. Can be upward (bullish), downward (bearish), or sideways (neutral).

### Volume
The number of shares traded during a specific period. Higher volume often confirms price movements.

## Risk Management Terms

### Drawdown
The decline from a peak to a trough in account value, typically expressed as a percentage.

### Kelly Criterion
Mathematical formula for determining optimal position size based on win rate and average win/loss ratio.

### Maximum Adverse Excursion (MAE)
The largest loss a position experienced before it was closed, regardless of final outcome.

### Position Correlation
The degree to which different positions move in the same direction. High correlation increases overall portfolio risk.

### Risk-Adjusted Return
Returns measured relative to the risk taken to achieve them. Higher risk-adjusted returns are better.

### Risk of Ruin
The probability of losing your entire trading capital given your win rate, average win/loss, and position sizing.

### Sharpe Ratio
Measure of risk-adjusted performance calculated as (return - risk-free rate) / standard deviation of returns.

### Value at Risk (VaR)
Statistical measure estimating the maximum potential loss over a specific time period at a given confidence level.

## Market Structure Terms

### After-Hours Trading
Trading that occurs outside regular market hours (9:30 AM - 4:00 PM ET for US stocks).

### Circuit Breaker
Automatic trading halt triggered when markets decline by specified percentages.

### Extended Hours
Trading sessions before market open (pre-market) and after market close (after-hours).

### Market Cap
Total value of a company's outstanding shares. Categories include small-cap, mid-cap, and large-cap.

### Pre-Market Trading
Trading that occurs before regular market hours, typically 4:00 AM - 9:30 AM ET.

### Sector Rotation
Investment strategy of moving money between different market sectors based on economic cycles.

## Order Types

### All-or-None (AON)
Order that must be executed in its entirety or not at all.

### Fill-or-Kill (FOK)
Order that must be executed immediately and completely or canceled.

### Good-Till-Canceled (GTC)
Order that remains active until executed or manually canceled.

### Immediate-or-Cancel (IOC)
Order that must be executed immediately; any unfilled portion is canceled.

### Stop-Limit Order
Combines stop and limit orders; becomes a limit order when the stop price is reached.

### Trailing Stop
Stop order that adjusts with favorable price movement while maintaining a fixed distance from the current price.

## Time Frames and Periods

### Intraday
Referring to price movements within a single trading day.

### Swing Trading
Trading style holding positions for 1-10 days, capturing short-term price swings.

### Day Trading
Opening and closing positions within the same trading day.

### Position Trading
Long-term trading holding positions for weeks to months.

### Scalping
Very short-term trading holding positions for seconds to minutes.

## Statistical and Performance Terms

### Alpha
Measure of an investment's performance relative to a market benchmark.

### Beta
Measure of a security's volatility relative to the overall market.

### Correlation
Statistical measure (-1 to +1) showing how two securities move relative to each other.

### Standard Deviation
Measure of price volatility; higher values indicate more volatile price movements.

### Win Rate
Percentage of trades that are profitable.

### Profit Factor
Ratio of gross profit to gross loss. Values above 1.0 indicate profitability.

### Average Win/Average Loss Ratio
Comparison of average profitable trade size to average losing trade size.

## Market Conditions

### Bull Market
Extended period of generally rising prices and investor optimism.

### Bear Market
Extended period of generally falling prices and investor pessimism.

### Sideways Market
Market characterized by horizontal price movement with no clear trend direction.

### Consolidation
Period where price moves within a defined range, often before a breakout.

### Correction
A decline of 10-20% from recent highs, typically temporary.

### Recession
Economic downturn lasting at least two consecutive quarters.

## Common Acronyms

**OHLC**: Open, High, Low, Close - the four key price points for any time period
**P&L**: Profit and Loss
**ROI**: Return on Investment
**ROE**: Return on Equity
**YTD**: Year-to-Date
**QoQ**: Quarter-over-Quarter
**MoM**: Month-over-Month
**AUM**: Assets Under Management
**NAV**: Net Asset Value
**IPO**: Initial Public Offering
**ETF**: Exchange-Traded Fund
**REIT**: Real Estate Investment Trust
**GDP**: Gross Domestic Product
**CPI**: Consumer Price Index
**Fed**: Federal Reserve
**FOMC**: Federal Open Market Committee

This glossary covers the essential terms you'll encounter when using SwingAgent and trading in general. Understanding these concepts will help you better interpret signals and make informed trading decisions.