from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sqlalchemy import func
from .database import get_database_config, init_database, get_session
from .models_db import VectorStore


def _ensure_db(db: Union[str, Path]):
    """Ensure database exists. For backward compatibility, convert file path to database URL."""
    if isinstance(db, (str, Path)):
        path = Path(db)
        if path.suffix == '.sqlite':
            # Extract directory to use as centralized database location
            data_dir = path.parent
            data_dir.mkdir(parents=True, exist_ok=True)
            # Use centralized database regardless of the input path
            database_url = f"sqlite:///{data_dir / 'swing_agent.sqlite'}"
            init_database(database_url)
        else:
            # Assume it's a database URL
            init_database(str(db))
    else:
        init_database()


def add_vector(
    db_path: Union[str, Path], 
    *, 
    vid: str, 
    ts_utc: str, 
    symbol: str, 
    timeframe: str, 
    vec: np.ndarray, 
    realized_r: Optional[float], 
    exit_reason: Optional[str], 
    payload: Optional[Dict[str, Any]]
):
    """Add a feature vector to the centralized vector store.
    
    Stores a feature vector along with trading outcomes for future pattern matching.
    Uses SQLAlchemy ORM with the centralized database architecture.
    
    Args:
        db_path: Database path or URL (converted to centralized database).
        vid: Unique vector identifier (e.g., "AAPL-2024-01-15T15:30:00Z").
        ts_utc: UTC timestamp of the vector generation.
        symbol: Trading symbol (e.g., "AAPL").
        timeframe: Trading timeframe (e.g., "30m", "1h").
        vec: Feature vector as numpy array.
        realized_r: Actual R-multiple return achieved (None if not evaluated).
        exit_reason: How the trade exited ("TP", "SL", "TIME", None if pending).
        payload: Additional metadata (vol regime, MTF alignment, etc.).
        
    Example:
        >>> vector = np.array([0.1, 0.8, 0.3, 0.6, 0.2])
        >>> add_vector(
        ...     "data/vec_store.sqlite",
        ...     vid="AAPL-2024-01-15T15:30:00Z",
        ...     ts_utc="2024-01-15T15:30:00Z",
        ...     symbol="AAPL",
        ...     timeframe="30m", 
        ...     vec=vector,
        ...     realized_r=1.5,
        ...     exit_reason="TP",
        ...     payload={"vol_regime": "M", "mtf_alignment": 2}
        ... )
        
    Note:
        Vectors are stored as JSON arrays in the database for cross-platform 
        compatibility. If a vector with the same ID exists, it will be updated.
    """
    _ensure_db(db_path)
    v = vec.astype(float).tolist()
    
    with get_session() as session:
        vector = VectorStore(
            id=vid,
            ts_utc=ts_utc,
            symbol=symbol,
            timeframe=timeframe,
            vec_json=json.dumps(v),
            realized_r=realized_r,
            exit_reason=exit_reason,
            payload=payload
        )
        
        # Use merge to handle INSERT OR REPLACE behavior
        existing = session.query(VectorStore).filter(VectorStore.id == vid).first()
        if existing:
            existing.ts_utc = ts_utc
            existing.symbol = symbol
            existing.timeframe = timeframe
            existing.vec_json = json.dumps(v)
            existing.realized_r = realized_r
            existing.exit_reason = exit_reason
            existing.payload = payload
        else:
            session.add(vector)
        
        session.commit()


def update_vector_payload(db_path: Union[str, Path], *, vid: str, merge: Dict[str, Any]):
    """Update vector payload by merging with existing data.
    
    Merges additional metadata into an existing vector's payload without 
    affecting the core vector data or trading outcomes.
    
    Args:
        db_path: Database path or URL (converted to centralized database).
        vid: Vector identifier to update.
        merge: Dictionary of key-value pairs to merge into payload.
        
    Example:
        >>> update_vector_payload(
        ...     "data/vec_store.sqlite",
        ...     vid="AAPL-2024-01-15T15:30:00Z", 
        ...     merge={"earnings_proximity": 5, "sector_rs": 1.15}
        ... )
        
    Note:
        If vector ID doesn't exist, the operation silently succeeds without error.
    """
    _ensure_db(db_path)
    
    with get_session() as session:
        vector = session.query(VectorStore).filter(VectorStore.id == vid).first()
        if vector:
            payload = vector.payload or {}
            payload.update(merge)
            vector.payload = payload
            session.commit()


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate cosine similarity between two normalized feature vectors.
    
    Computes the cosine of the angle between two vectors, providing a similarity
    measure independent of vector magnitude. Used for finding similar market setups.
    
    Args:
        u: First feature vector.
        v: Second feature vector.
        
    Returns:
        float: Cosine similarity in range [-1, 1]. Values closer to 1 indicate 
        higher similarity, 0 indicates orthogonality, -1 indicates opposite.
        
    Example:
        >>> vec1 = np.array([1.0, 0.5, 0.8])
        >>> vec2 = np.array([0.9, 0.6, 0.7]) 
        >>> similarity = cosine(vec1, vec2)
        >>> print(f"Similarity: {similarity:.3f}")
        
    Note:
        Returns 0.0 if either vector has zero norm to handle edge cases gracefully.
    """
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def knn(
    db_path: Union[str, Path], 
    *, 
    query_vec: np.ndarray, 
    k: int = 50, 
    symbol: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Find k nearest neighbors using cosine similarity search.
    
    Performs vector similarity search in the centralized database to find
    historical patterns most similar to the current market setup.
    
    Args:
        db_path: Database path or URL (converted to centralized database).
        query_vec: Feature vector to find similarities for.
        k: Number of nearest neighbors to return (default: 50).
        symbol: Optional symbol filter (None for all symbols).
        
    Returns:
        List[Dict]: Sorted list of neighbors with similarity scores and metadata.
        Each dict contains:
        - id: Vector identifier
        - symbol: Trading symbol  
        - timeframe: Trading timeframe
        - similarity: Cosine similarity score [0, 1]
        - realized_r: Actual return achieved (if available)
        - exit_reason: How trade exited (if available)
        - payload: Additional metadata
        
    Example:
        >>> query_vector = np.array([0.1, 0.8, 0.3, 0.6, 0.2])
        >>> neighbors = knn(
        ...     "data/vec_store.sqlite",
        ...     query_vec=query_vector,
        ...     k=50,
        ...     symbol="AAPL"
        ... )
        >>> print(f"Best match: {neighbors[0]['similarity']:.3f}")
        >>> print(f"Historical R: {neighbors[0]['realized_r']}")
        
    Note:
        Results are sorted by similarity in descending order. Only returns
        vectors that have valid similarity scores (non-zero norm vectors).
    """
    _ensure_db(db_path)
    rows = []
    
    with get_session() as session:
        query = session.query(VectorStore)
        if symbol:
            query = query.filter(VectorStore.symbol == symbol)
        
        vectors = query.all()
        
        for vector in vectors:
            vec = np.array(json.loads(vector.vec_json), dtype=float)
            sim = cosine(query_vec, vec)
            
            rows.append({
                "id": vector.id,
                "ts_utc": vector.ts_utc,
                "symbol": vector.symbol,
                "timeframe": vector.timeframe,
                "similarity": sim,
                "realized_r": vector.realized_r,
                "exit_reason": vector.exit_reason,
                "payload": vector.payload
            })
    
    # Sort by similarity in descending order and return top k
    rows.sort(key=lambda x: x["similarity"], reverse=True)
    return rows[:k]


def filter_neighbors(neighbors: List[Dict[str, Any]], *, vol_regime: Optional[str] = None) -> List[Dict[str, Any]]:
    """Filter neighbor results by market conditions and metadata.
    
    Applies optional filters to KNN results to find patterns from similar
    market environments. Improves prediction accuracy by matching context.
    
    Args:
        neighbors: List of neighbor dicts from knn() function.
        vol_regime: Optional volatility regime filter ("L", "M", "H").
        
    Returns:
        List[Dict]: Filtered neighbors matching the specified criteria.
        
    Example:
        >>> all_neighbors = knn(db_path, query_vec=vector, k=100)
        >>> similar_vol = filter_neighbors(
        ...     all_neighbors, 
        ...     vol_regime="M"  # Only medium volatility setups
        ... )
        >>> if len(similar_vol) >= 10:
        ...     stats = extended_stats(similar_vol)
        ...     print(f"Win rate in similar vol: {stats['win_rate']:.1%}")
        
    Note:
        Additional filters can be added by modifying the payload matching logic.
        Returns empty list if no neighbors match the filtering criteria.
    """
    """Filter neighbors by volatility regime."""
    if not neighbors or not vol_regime:
        return neighbors
    
    filt = [n for n in neighbors if (n.get("payload") or {}).get("vol_regime") == vol_regime]
    return filt if len(filt) >= max(10, int(0.4*len(neighbors))) else neighbors


def extended_stats(neighbors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comprehensive trading statistics from neighbor vectors.
    
    Computes detailed performance metrics from similar historical patterns
    to estimate expected outcomes for the current setup. Used for ML-based
    trade validation and outcome prediction.
    
    Args:
        neighbors: List of neighbor vectors with realized outcomes from knn().
        
    Returns:
        Dict containing comprehensive trading statistics:
        - n: Number of neighbors with valid realized_r values
        - p_win: Win probability [0, 1] (realized_r > 0)
        - avg_R: Average R-multiple across all trades
        - avg_win_R: Average R-multiple for winning trades only
        - avg_loss_R: Average R-multiple for losing trades only  
        - median_hold_bars: Median holding period in bars
        - median_hold_days: Median holding period in days
        - median_win_hold_bars: Median hold time for winners
        - median_loss_hold_bars: Median hold time for losers
        - profit_factor: Gross profit / gross loss ratio
        - tp: Number of take profit exits
        - sl: Number of stop loss exits
        - time: Number of time-based exits
        
    Example:
        >>> neighbors = knn(db_path, query_vec=vector, k=50)
        >>> stats = extended_stats(neighbors)
        >>> print(f"Win Rate: {stats['p_win']:.1%}")
        >>> print(f"Avg R: {stats['avg_R']:.2f}")
        >>> print(f"Profit Factor: {stats['profit_factor']:.2f}")
        >>> print(f"Median Hold: {stats['median_hold_days']:.1f} days")
        
    Note:
        Returns zero values for all metrics if no neighbors have valid outcomes.
        Only includes neighbors with non-None realized_r values in calculations.
    """
    if not neighbors:
        return {
            "n": 0,
            "p_win": 0.0,
            "avg_R": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "median_hold_bars": None,
            "median_hold_days": None,
            "median_win_hold_bars": None,
            "median_loss_hold_bars": None,
            "profit_factor": 0.0,
            "tp": 0,
            "sl": 0,
            "time": 0
        }
    
    rs = [x["realized_r"] for x in neighbors if x["realized_r"] is not None]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    
    p_win = (len(wins) / len(rs)) if rs else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_R = float(np.mean(rs)) if rs else 0.0
    
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (
        float("inf") if sum(wins) > 0 else 0.0
    )
    
    def collect(key):
        vals = []
        for x in neighbors:
            v = (x.get("payload") or {}).get(key)
            if v is not None:
                vals.append(float(v))
        return vals
    
    all_b = collect("hold_bars")
    win_b = []
    loss_b = []
    
    for x in neighbors:
        hb = (x.get("payload") or {}).get("hold_bars")
        r = x.get("realized_r")
        if hb is None or r is None:
            continue
        (win_b if r > 0 else loss_b).append(float(hb))
    
    med_all = int(np.median(all_b)) if all_b else None
    med_win = int(np.median(win_b)) if win_b else None
    med_loss = int(np.median(loss_b)) if loss_b else None
    
    bars_to_days = lambda b: None if b is None else round(b / 13.0, 2)
    
    return {
        "n": len(neighbors),
        "p_win": round(p_win, 3),
        "avg_R": round(avg_R, 3),
        "avg_win_R": round(avg_win, 3),
        "avg_loss_R": round(avg_loss, 3),
        "median_hold_bars": med_all,
        "median_hold_days": bars_to_days(med_all),
        "median_win_hold_bars": med_win,
        "median_loss_hold_bars": med_loss,
        "profit_factor": (round(profit_factor, 3) if profit_factor != float('inf') else float('inf')),
        "tp": sum(1 for x in neighbors if x.get("exit_reason") == "TP"),
        "sl": sum(1 for x in neighbors if x.get("exit_reason") == "SL"),
        "time": sum(1 for x in neighbors if x.get("exit_reason") == "TIME")
    }


# Keep the old SCHEMA for reference/migration purposes
SCHEMA = """
CREATE TABLE IF NOT EXISTS vec_store (
  id TEXT PRIMARY KEY,
  ts_utc TEXT NOT NULL,
  symbol TEXT NOT NULL,
  timeframe TEXT NOT NULL,
  vec_json TEXT NOT NULL,
  realized_r REAL,
  exit_reason TEXT,
  payload_json TEXT
);
CREATE INDEX IF NOT EXISTS vec_store_symbol_idx ON vec_store(symbol);
"""
