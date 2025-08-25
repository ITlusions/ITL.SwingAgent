# Copilot Instructions — Swing Agent v1.6.1

You are helping maintain a Python package for 1–2 day swing trading with Fib-based entries, vector KNN priors, and OpenAI-generated action plans.

## Project map
- `src/swing_agent/` — core library
  - `agent.py` — orchestrates data → features → priors → signal JSON (enrichments included)
  - `strategy.py` — trend labeling + entry building (Fibonacci golden pocket logic)
  - `indicators.py` — EMA/RSI/ATR/Bollinger/Fibonacci utilities
  - `features.py` — vectorization + time-of-day + vol regime
  - `vectorstore.py` — lightweight cosine KNN over SQLite
  - `storage.py` — signals SQLite schema (expectations + LLM + enrichments)
  - `llm_predictor.py` — OpenAI models: voting + action plan
- `scripts/`
  - `run_swing_agent.py` — generate a live signal
  - `backtest_generate_signals.py` — create historical signals (no look-ahead)
  - `eval_signals.py` — compute outcomes (TP/SL/TIME) & update vector payloads
  - `backfill_vector_store.py` — rebuild vector DB from signals DB
  - `analyze_performance.py` — simple calibration & bucket stats

## Coding style
- Python 3.12+, type hints, Pydantic v2 models.
- Keep the **feature vector** compact; put extra context in `payload_json` in the vector DB.
- Never let LLM change math; LLMs should **explain and plan**, not decide SL/TP.

## Common tasks (prompt me like this)
- *"Add earnings proximity to enrichments and include it in KNN filter. Update schema and analyzer."*
- *"Implement peers-based relative strength and use it as a confidence bump only when > 1.05."*
- *"Add plot script to visualize calibration and R-distribution by vol regime."*

## Guardrails
- No look-ahead in backtests. When generating historical signals, slice data to the current bar.
- Never store API keys; read them from env (`OPENAI_API_KEY`, `SWING_LLM_MODEL`).
- Respect v1.6 schema fields; any schema change must include migration or recreate instructions.

## Runbook
1. Generate or fetch signals (live or historical).
2. Evaluate with `eval_signals.py` (fills `realized_r` and exit info).
3. Analyze with `analyze_performance.py` (calibration & regime buckets).
4. Iterate on features; keep vectors lean and payloads rich.