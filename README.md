# Swing Agent v1.6.1

**Adds** Copilot-friendly `instructions.md` and `scripts/backtest_generate_signals.py` (walk-forward historical signal generation). Based on v1.6 enrichments:
- Multi-timeframe alignment (15m + 1h)
- Relative strength vs sector ETF (default `XLK`) + SPY fallback
- Time-of-day bucket (open/mid/close)
- Volatility regime filter for KNN priors
- Signals DB stores expectations + LLM plan + enrichments

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# optional LLM
export OPENAI_API_KEY="sk-..."
export SWING_LLM_MODEL="gpt-4o-mini"
```

## Run live signal
```bash
python scripts/run_swing_agent.py   --symbol AMD --interval 30m --lookback-days 30   --db data/signals.sqlite --vec-db data/vec_store.sqlite   --sector XLK
```

## Generate historical signals (no look-ahead)
```bash
python scripts/backtest_generate_signals.py   --symbol AMD --interval 30m --lookback-days 180   --warmup-bars 80   --db data/signals.sqlite --vec-db data/vec_store.sqlite   --sector XLK --no-llm
```

## Evaluate stored signals
```bash
python scripts/eval_signals.py --db data/signals.sqlite --max-hold-days 2.0
```

## Backfill vector store from signal history
```bash
PYTHONPATH=src python scripts/backfill_vector_store.py   --signals-db data/signals.sqlite --vec-db data/vec_store.sqlite
```

## Performance snapshot
```bash
python scripts/analyze_performance.py --db data/signals.sqlite
```
