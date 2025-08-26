# ADR-002: Vector Store Design for Pattern Matching

## Status

Accepted

## Context

SwingAgent uses machine learning for pattern matching by:
- Encoding market setups as feature vectors
- Finding similar historical patterns via cosine similarity 
- Predicting outcomes based on historical performance

Key requirements:
- Fast similarity search over thousands of vectors
- Flexible metadata storage for filtering
- Outcome tracking for backtesting validation
- Cross-symbol pattern matching capabilities

## Decision

We will implement a custom vector store with:

1. **Compact Feature Vectors**: 16-dimensional normalized vectors for core market features
2. **Rich Metadata Payloads**: JSON storage for additional context (vol regime, MTF alignment, etc.)
3. **Cosine Similarity Search**: L2-normalized vectors for angle-based similarity
4. **Outcome Integration**: Direct linkage to realized R-multiples and exit reasons
5. **Context Filtering**: Ability to filter by market conditions (volatility, etc.)

## Vector Composition

```python
# 16-dimensional feature vector
[
    ema_slope,           # Momentum indicator
    rsi_normalized,      # Overbought/oversold
    atr_pct,            # Volatility measure  
    price_above_ema,    # Trend position
    prev_range_pct,     # Previous bar range
    gap_pct,            # Gap size
    fib_position,       # Fibonacci position
    in_golden_pocket,   # Golden pocket flag
    r_multiple/5.0,     # Risk/reward normalized
    trend_up,           # Uptrend flag
    trend_down,         # Downtrend flag
    trend_sideways,     # Sideways flag
    session_open_close, # Session encoding
    session_mid_close,  # Session encoding
    llm_confidence,     # LLM confidence
    constant_term       # Bias term
]
```

## Storage Schema

```sql
CREATE TABLE vec_store (
    id TEXT PRIMARY KEY,           -- Unique vector ID
    ts_utc TEXT NOT NULL,         -- Timestamp
    symbol TEXT NOT NULL,         -- Trading symbol
    timeframe TEXT NOT NULL,      -- Timeframe
    vec_json TEXT NOT NULL,       -- Vector as JSON array
    realized_r REAL,              -- Actual outcome
    exit_reason TEXT,             -- Exit type
    payload JSON                  -- Additional metadata
);
```

## Similarity Algorithm

```python
def cosine(u, v):
    """Cosine similarity with L2 normalization"""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
```

## Consequences

### Positive

- **Efficient Search**: Cosine similarity scales well to thousands of vectors
- **Flexible Metadata**: JSON payloads allow rich contextual filtering
- **Cross-Symbol Learning**: Patterns work across different symbols
- **Outcome Integration**: Direct backtesting validation of predictions
- **Compact Storage**: 16 dimensions keep memory usage reasonable

### Negative

- **Vector Dimensionality**: Fixed at 16 dimensions, adding features requires migration
- **No Indexing**: Linear search may become slow with 100k+ vectors
- **Feature Engineering**: Careful normalization required for meaningful similarity

## Performance Considerations

- **Memory Usage**: ~64 bytes per vector (16 * 4 bytes float)
- **Search Time**: O(n) linear search, acceptable for <50k vectors
- **Storage**: JSON encoding adds ~2x overhead vs binary

## Future Enhancements

- Vector indexing (FAISS, Annoy) for sub-linear search times
- Dynamic vector dimensions via versioning
- Distributed vector storage for massive datasets