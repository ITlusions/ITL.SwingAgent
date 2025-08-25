# Frequently Asked Questions (FAQ)

## Getting Started

### Q: What exactly is SwingAgent?
**A:** SwingAgent is an automated trading system designed for 1-2 day swing trades. It combines technical analysis (Fibonacci levels, moving averages, momentum indicators), machine learning pattern recognition, and AI-powered explanations to generate trading signals with specific entry, stop-loss, and take-profit levels.

### Q: Do I need programming experience to use SwingAgent?
**A:** No. While SwingAgent is built in Python, you only need to run simple command-line scripts. The [Getting Started guide](getting-started.md) walks you through everything step-by-step.

### Q: What's the minimum amount of money I need to start?
**A:** SwingAgent doesn't trade for you - it generates signals that you can trade with any broker and any account size. However, since it's designed for individual stock positions, having at least $1,000-2,000 per position is practical for proper risk management.

### Q: Do I need to pay for data feeds?
**A:** No. SwingAgent uses free Yahoo Finance data, which is sufficient for swing trading timeframes (15-minute, 30-minute, 1-hour bars).

## Technical Setup

### Q: What if I don't have an OpenAI API key?
**A:** The system works without OpenAI - you just won't get the AI-generated action plans and explanations. All the technical analysis, pattern matching, and signal generation still function normally. You can add the API key later.

### Q: Can I run this on Windows?
**A:** Yes, SwingAgent works on Windows, macOS, and Linux. The commands in the documentation show both Windows and Unix-style examples.

### Q: How much disk space does SwingAgent need?
**A:** Very little. The application itself is small, and the database files typically use only a few MB even after generating hundreds of signals.

### Q: Do I need to keep my computer running all the time?
**A:** No. SwingAgent generates signals when you run it - it's not a continuously running system. Most traders run it once or twice per day to scan for new opportunities.

## Understanding Signals

### Q: What does the "confidence" score mean?
**A:** Confidence (0-100%) represents how similar the current setup is to historical profitable patterns. Higher confidence means the system found many similar historical setups that worked well. 70%+ is generally considered high confidence.

### Q: What's an R-multiple and why does it matter?
**A:** R-multiple is your reward-to-risk ratio. If you risk $100 (difference between entry and stop), an R-multiple of 1.5 means you could potentially make $150. Higher R-multiples mean better risk/reward ratios. Look for signals with R > 1.3.

### Q: How accurate are the "expected" win rates?
**A:** Expected win rates are based on historical patterns similar to the current setup. They're generally reliable for understanding the probability profile of trades, but remember past performance doesn't guarantee future results.

### Q: What does "MTF alignment" mean?
**A:** Multi-Timeframe (MTF) alignment shows how many timeframes (15-minute and 1-hour) agree on the trend direction. A score of 2 means both timeframes are aligned, which generally produces stronger signals.

### Q: What's the difference between volatility regimes (L, M, H)?
**A:** 
- **L (Low)**: Calm markets, trends are steady, can use larger position sizes
- **M (Medium)**: Normal market conditions, standard approach works well  
- **H (High)**: Volatile markets (often around earnings), reduce position sizes and take profits faster

## Trading with Signals

### Q: Do I have to take every signal?
**A:** Absolutely not. Signals are suggestions. You should only take signals that:
- Meet your quality criteria (confidence, R-multiple, win rate)
- Fit your risk tolerance
- Make sense to you personally
- Align with your market view

### Q: What if the stock price has moved away from the entry by the time I see the signal?
**A:** Don't chase. If the price is outside the recommended entry zone, skip the trade. SwingAgent calculates risk/reward based on specific entry levels. Entering elsewhere changes the math.

### Q: Should I always hold for the full target?
**A:** Not necessarily. Many traders take partial profits (e.g., 50% at 1R) and let the rest run to the target. You can also exit early if you see the setup deteriorating or if you're uncomfortable with the position.

### Q: What if a trade hits the stop loss?
**A:** That's normal and expected. With a 60% win rate, 40% of trades will be losers. The key is that your average winner (when you hit targets) should be larger than your average loser (when you hit stops). This is why R-multiples matter.

### Q: Can I modify the stop loss or target?
**A:** You can, but be careful. SwingAgent calculates these levels based on technical analysis and historical patterns. If you modify them, you're changing the risk/reward profile that the system used to generate its expectations.

## System Behavior

### Q: Why do some stocks never generate signals?
**A:** SwingAgent only generates signals when it finds high-quality setups that meet its criteria:
- Clear trend or mean-reversion setup
- Good risk/reward ratio (>1.2 typically)
- Similar historical patterns in its database
- Proper technical indicators alignment

Not all stocks will have these conditions at any given time.

### Q: Why do I get different signals when I run the same command twice?
**A:** If you run commands very close together (within the same minute), you might get identical signals. However, as new price data comes in or if you run during different market hours, signals can change. Always use the most recent signal.

### Q: How does the system learn and improve?
**A:** As you use SwingAgent and evaluate signals (using `eval_signals.py`), it builds a database of outcomes. This helps improve pattern matching for future signals. The more signals you evaluate, the better the historical expectations become.

### Q: What happens if my internet connection is slow?
**A:** SwingAgent downloads data from Yahoo Finance, so it needs internet access. Slow connections will just make the commands take longer to run. If downloads fail, you'll get error messages and can retry.

## Performance and Results

### Q: What kind of returns can I expect?
**A:** This depends on many factors: how selective you are with signals, your position sizing, execution quality, and market conditions. SwingAgent provides tools for analysis, but results depend on your trading decisions. Focus on learning the system before expecting consistent profits.

### Q: How do I know if I'm using the system correctly?
**A:** Good signs include:
- Your actual win rate is close to the system's expected win rate
- You're following the entry, stop, and target levels consistently
- You're selective about signal quality
- You're tracking and learning from your results

### Q: Why are my results different from the backtest?
**A:** Backtests assume perfect execution at exact prices. Real trading involves:
- Slippage (getting slightly different prices)
- Emotional decisions
- Missing signals due to timing
- Different position sizing

This is normal - focus on being consistent rather than matching backtest results exactly.

### Q: Should I trade every day?
**A:** No. Some days there are no good signals. It's better to wait for high-quality setups than force trades. Good swing traders might only make 1-3 trades per week.

## Advanced Questions

### Q: Can I customize the technical parameters?
**A:** The core system uses proven parameters, but you can see [Configuration](configuration.md) for some customization options. For most users, the default settings work well.

### Q: Can I use this for options trading?
**A:** SwingAgent generates signals for stock prices. You could potentially use these signals to inform options trades, but you'd need to adapt the risk management and timing for options characteristics.

### Q: Can I run this for crypto or forex?
**A:** Currently, SwingAgent is designed for US stocks and ETFs using Yahoo Finance data. The technical analysis principles could apply to other markets, but the data sources and some features are stock-specific.

### Q: How do I add new stocks to scan?
**A:** Just use any valid stock symbol in the commands. SwingAgent will download the data and analyze it. Popular symbols include:
- Individual stocks: AAPL, MSFT, GOOGL, NVDA, TSLA, etc.
- Sector ETFs: QQQ, XLF, XLE, XLV, XLI, etc.
- Market ETFs: SPY, IWM, DIA, etc.

### Q: Can I automate the trading based on signals?
**A:** SwingAgent generates signals but doesn't execute trades. Some traders connect it to brokers via APIs, but this requires programming knowledge and careful risk management. Most users trade the signals manually.

## Troubleshooting

### Q: I get "Permission denied" errors?
**A:** See the [Troubleshooting guide](troubleshooting.md) for detailed solutions. Usually this is fixed by:
```bash
chmod 755 data/
chmod 644 data/*.sqlite
```

### Q: The commands run but produce no output?
**A:** This usually means:
1. No signal was generated (no good setup found)
2. The stock symbol doesn't exist or has no data
3. There's an error - check for error messages

### Q: Yahoo Finance data seems delayed?
**A:** Yahoo Finance data can have 15-20 minute delays during market hours. For swing trading (1-2 day holds), this delay doesn't materially affect signal quality.

### Q: How do I get help if something isn't working?
**A:** 
1. Check the [Troubleshooting guide](troubleshooting.md)
2. Review the [Getting Started guide](getting-started.md) to ensure proper setup
3. Check that your command syntax matches the examples exactly
4. Verify your Python installation and dependencies

## Trading Psychology

### Q: What if I'm scared to take the first trade?
**A:** This is normal. Start with very small position sizes or even paper trading. Focus on learning how the signals work rather than making money initially. Confidence comes with experience.

### Q: What if I keep second-guessing the signals?
**A:** Develop clear criteria for which signals you'll take (e.g., confidence >70%, R-multiple >1.5) and stick to them. The system is designed to remove emotion from analysis - trust the process.

### Q: Should I always wait for perfect signals?
**A:** There's no such thing as a perfect signal. Set reasonable quality standards and be consistent. Better to take consistently good signals than wait forever for perfect ones.

Remember: SwingAgent is a tool to help with trading decisions, not a guarantee of profits. Always trade responsibly and never risk more than you can afford to lose.