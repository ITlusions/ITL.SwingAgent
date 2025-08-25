# Troubleshooting Guide

Common issues and solutions for the SwingAgent system.

## Installation Issues

### Python Version Compatibility

**Problem**: `ImportError` or syntax errors during installation.

```bash
python -c "import sys; print(sys.version)"
# Should show Python 3.10 or higher
```

**Solutions**:
```bash
# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3.10-dev

# Create virtual environment with correct Python
python3.10 -m venv .venv
source .venv/bin/activate

# Verify version in virtual environment
python --version
```

### Dependency Installation Failures

**Problem**: Package installation errors, especially with numpy/pandas.

**Solutions**:
```bash
# Update pip first
pip install --upgrade pip setuptools wheel

# Install system dependencies (Ubuntu/Debian)
sudo apt install build-essential python3-dev

# Install packages individually if bulk install fails
pip install numpy>=1.26
pip install pandas>=2.1
pip install pydantic>=2.6
pip install yfinance>=0.2.40
pip install -e .
```

### Network/Proxy Issues

**Problem**: Cannot reach PyPI or external APIs.

**Solutions**:
```bash
# Configure pip for proxy
pip install --proxy http://proxy.company.com:8080 -e .

# Use alternative PyPI index
pip install -i https://pypi.org/simple/ -e .

# Test network connectivity
curl -I https://pypi.org
curl -I https://query1.finance.yahoo.com
curl -I https://api.openai.com
```

## Data Fetching Issues

### Yahoo Finance API Problems

**Problem**: `RuntimeError: No data for SYMBOL @ interval`

**Diagnosis**:
```python
import yfinance as yf

# Test direct yfinance access
ticker = yf.Ticker("AAPL")
info = ticker.info
print(f"Symbol exists: {info.get('symbol', 'Not found')}")

# Test data download
data = yf.download("AAPL", period="5d", interval="30m")
print(f"Data shape: {data.shape}")
```

**Solutions**:
```python
# 1. Check symbol validity
valid_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
# Invalid: "INVALID", delisted stocks, wrong exchanges

# 2. Use proper timeframe combinations
valid_combinations = {
    "15m": "5d",    # Max 60 days
    "30m": "60d",   # Max 60 days  
    "1h": "730d",   # Max 2 years
    "1d": "max"     # No limit
}

# 3. Handle market holidays
import pandas as pd
from datetime import datetime, timedelta

end = datetime.now()
start = end - timedelta(days=7)  # Use longer period for holidays

# 4. Add retry logic
import time
def fetch_with_retry(symbol, interval, retries=3):
    for i in range(retries):
        try:
            return load_ohlcv(symbol, interval, 30)
        except Exception as e:
            if i == retries - 1:
                raise e
            time.sleep(5 * (i + 1))  # Exponential backoff
```

### Missing OHLCV Columns

**Problem**: `RuntimeError: Missing column 'X' in data`

**Solutions**:
```python
# Debug missing columns
df = yf.download("AAPL", period="5d")
print(f"Columns: {list(df.columns)}")

# Handle multi-index columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
    
# Rename columns to lowercase
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Verify required columns
required = ['open', 'high', 'low', 'close', 'volume']
missing = [col for col in required if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
```

### Timezone Issues

**Problem**: Inconsistent timestamps or timezone errors.

**Solutions**:
```python
import pandas as pd
from datetime import timezone

# Ensure UTC timezone
df.index = pd.to_datetime(df.index, utc=True)

# Convert from market timezone
market_tz = "US/Eastern"
df.index = df.index.tz_localize(market_tz).tz_convert('UTC')

# Remove timezone info if needed
df.index = df.index.tz_localize(None)
```

## Database Issues

### SQLite Permission Errors

**Problem**: `sqlite3.OperationalError: unable to open database file`

**Solutions**:
```bash
# Check file permissions
ls -la data/
chmod 755 data/
chmod 644 data/*.sqlite

# Check parent directory permissions
mkdir -p data/
touch data/test.sqlite
rm data/test.sqlite

# Use absolute paths
export SWING_SIGNALS_DB="/full/path/to/signals.sqlite"
```

### Database Corruption

**Problem**: `sqlite3.DatabaseError: database disk image is malformed`

**Solutions**:
```bash
# Check database integrity
sqlite3 data/signals.sqlite "PRAGMA integrity_check;"

# Attempt repair
sqlite3 data/signals.sqlite ".recover" | sqlite3 data/signals_recovered.sqlite

# Backup before recovery
cp data/signals.sqlite data/signals.backup

# Rebuild from scratch if necessary
rm data/signals.sqlite
python -c "from swing_agent.storage import record_signal; print('Database recreated')"
```

### Database Lock Errors

**Problem**: `sqlite3.OperationalError: database is locked`

**Solutions**:
```python
# Use connection with timeout
import sqlite3
conn = sqlite3.connect("data/signals.sqlite", timeout=30.0)

# Check for hung processes
ps aux | grep python | grep swing

# Kill hung processes
pkill -f "python.*swing"

# Use WAL mode for better concurrency
sqlite3 data/signals.sqlite "PRAGMA journal_mode=WAL;"
```

## LLM Integration Issues

### API Key Problems

**Problem**: `openai.AuthenticationError: Incorrect API key`

**Solutions**:
```bash
# Verify environment variable
echo $OPENAI_API_KEY
# Should start with 'sk-'

# Test API key directly
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# Load from file if needed
export OPENAI_API_KEY=$(cat ~/.openai_key)
```

### Rate Limiting

**Problem**: `openai.RateLimitError: Rate limit exceeded`

**Solutions**:
```python
import time
import openai
from functools import wraps

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except openai.RateLimitError as e:
                    if i == retries - 1:
                        raise e
                    wait_time = backoff_in_seconds * (2 ** i)
                    time.sleep(wait_time)
        return wrapper
    return decorator

# Apply to LLM functions
@retry_with_backoff(retries=3, backoff_in_seconds=5)
def llm_call_with_retry(**kwargs):
    return llm_extra_prediction(**kwargs)
```

### Model Not Found

**Problem**: `openai.NotFoundError: The model 'gpt-X' does not exist`

**Solutions**:
```bash
# Check available models
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models | jq '.data[].id' | grep gpt

# Use supported models
export SWING_LLM_MODEL="gpt-4o-mini"  # Most cost-effective
export SWING_LLM_MODEL="gpt-4o"       # Best performance
export SWING_LLM_MODEL="gpt-3.5-turbo" # Fastest
```

### LLM Response Parsing Errors

**Problem**: `pydantic.ValidationError` when parsing LLM responses.

**Solutions**:
```python
# Add fallback parsing
def safe_llm_prediction(**features):
    try:
        return llm_extra_prediction(**features)
    except Exception as e:
        print(f"LLM error: {e}")
        # Return default response
        return LlmVote(
            trend_label="sideways",
            entry_bias="none",
            confidence=0.0,
            rationale="LLM unavailable"
        )

# Validate JSON before parsing
import json
def validate_llm_response(response_text):
    try:
        data = json.loads(response_text)
        return LlmVote(**data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Invalid LLM response: {e}")
        return None
```

## Performance Issues

### Slow Signal Generation

**Problem**: Signal generation takes too long (>30 seconds).

**Diagnosis**:
```python
import time
import cProfile

def profile_analysis():
    start = time.time()
    
    # Profile each component
    data_start = time.time()
    df = load_ohlcv("AAPL", "30m", 30)
    print(f"Data fetch: {time.time() - data_start:.2f}s")
    
    analysis_start = time.time()
    agent = SwingAgent(use_llm=False)  # Test without LLM first
    signal = agent.analyze_df("AAPL", df)
    print(f"Analysis: {time.time() - analysis_start:.2f}s")
    
    print(f"Total: {time.time() - start:.2f}s")

# Use cProfile for detailed analysis
cProfile.run('agent.analyze("AAPL")', 'profile_output.prof')
```

**Solutions**:
```python
# 1. Reduce lookback period
agent = SwingAgent(lookback_days=15)  # vs default 30

# 2. Disable LLM for speed testing
agent = SwingAgent(use_llm=False)

# 3. Cache data for multiple symbols
data_cache = {}
symbols = ["AAPL", "MSFT", "GOOGL"]
for symbol in symbols:
    if symbol not in data_cache:
        data_cache[symbol] = load_ohlcv(symbol, "30m", 30)
    signal = agent.analyze_df(symbol, data_cache[symbol])

# 4. Use smaller KNN neighborhoods
# Modify vectorstore.py knn() function default k=25 instead of 50
```

### Memory Usage Issues

**Problem**: High memory usage or out-of-memory errors.

**Solutions**:
```python
# Monitor memory usage
import psutil
import os

def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"{label}: {memory_mb:.1f} MB")

print_memory_usage("Start")
df = load_ohlcv("AAPL", "30m", 30)
print_memory_usage("After data load")

# Optimize data types
df = df.astype({
    'open': 'float32',
    'high': 'float32', 
    'low': 'float32',
    'close': 'float32',
    'volume': 'int32'
})

# Clear variables when done
del df
import gc
gc.collect()
```

### Vector Store Performance

**Problem**: Slow KNN searches in vector store.

**Solutions**:
```python
# 1. Add indexes to vector store
import sqlite3
conn = sqlite3.connect("data/vec_store.sqlite")
conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_symbol_ts ON vec_store(symbol, ts_utc);")
conn.close()

# 2. Limit search scope
neighbors = knn(
    db_path="data/vec_store.sqlite",
    query_vec=vector,
    k=25,  # Reduce from default 50
    symbol="AAPL"  # Symbol-specific search
)

# 3. Filter by time period
def recent_knn(db_path, query_vec, k=50, days_back=365):
    cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
    # Add WHERE clause to filter by ts_utc > cutoff_date
```

## Runtime Errors

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'swing_agent'`

**Solutions**:
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Add src directory to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/ITL.SwingAgent/src"

# Or install in development mode
pip install -e .

# Verify installation
python -c "import swing_agent; print(swing_agent.__file__)"
```

### Attribute Errors

**Problem**: `AttributeError: 'DataFrame' object has no attribute 'X'`

**Solutions**:
```python
# Debug DataFrame structure
print(f"DataFrame columns: {list(df.columns)}")
print(f"DataFrame index: {df.index}")
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame dtypes:\n{df.dtypes}")

# Check for empty DataFrames
if df.empty:
    print("DataFrame is empty!")
    
# Verify required columns
required_columns = ['open', 'high', 'low', 'close', 'volume']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
```

### Index Errors

**Problem**: `IndexError: single positional indexer is out-of-bounds`

**Solutions**:
```python
# Check DataFrame length before indexing
if len(df) < 50:
    print(f"Warning: DataFrame too short ({len(df)} rows)")
    
# Use safe indexing
def safe_iloc(series, index, default=None):
    try:
        return series.iloc[index]
    except (IndexError, KeyError):
        return default

# Example usage
last_close = safe_iloc(df['close'], -1, 0.0)
prev_close = safe_iloc(df['close'], -2, 0.0)
```

## Configuration Issues

### Environment Variables Not Loading

**Problem**: Environment variables not recognized.

**Solutions**:
```bash
# Check current environment
env | grep SWING
env | grep OPENAI

# Source environment file
source ~/.bashrc
source .env

# Set variables for current session
export OPENAI_API_KEY="sk-your-key"
export SWING_LLM_MODEL="gpt-4o-mini"

# Verify in Python
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'Not set'))"
```

### Path Issues

**Problem**: Files not found or incorrect paths.

**Solutions**:
```python
import os
from pathlib import Path

# Use absolute paths
project_root = Path(__file__).parent.parent
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)

signals_db = data_dir / "signals.sqlite"
vectors_db = data_dir / "vec_store.sqlite"

# Verify paths exist
print(f"Project root: {project_root}")
print(f"Data directory exists: {data_dir.exists()}")
print(f"Signals DB: {signals_db}")
```

## Testing and Validation

### Minimal Test Script

```python
#!/usr/bin/env python3
"""
Minimal test script to validate SwingAgent installation and basic functionality.
"""

def test_imports():
    """Test all required imports."""
    try:
        import swing_agent
        from swing_agent.agent import SwingAgent
        from swing_agent.data import load_ohlcv
        from swing_agent.strategy import label_trend, build_entry
        from swing_agent.indicators import ema, rsi, atr
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_fetch():
    """Test data fetching."""
    try:
        from swing_agent.data import load_ohlcv
        df = load_ohlcv("AAPL", "1d", 5)  # 5 days of daily data
        print(f"✓ Data fetch successful: {df.shape}")
        return True
    except Exception as e:
        print(f"✗ Data fetch error: {e}")
        return False

def test_technical_analysis():
    """Test technical indicators."""
    try:
        from swing_agent.data import load_ohlcv
        from swing_agent.strategy import label_trend, build_entry
        
        df = load_ohlcv("AAPL", "1d", 30)
        trend = label_trend(df)
        entry = build_entry(df, trend)
        
        print(f"✓ Technical analysis successful")
        print(f"  Trend: {trend.label}, RSI: {trend.rsi_14:.1f}")
        print(f"  Entry: {entry.side if entry else 'None'}")
        return True
    except Exception as e:
        print(f"✗ Technical analysis error: {e}")
        return False

def test_agent_basic():
    """Test basic agent functionality without LLM."""
    try:
        from swing_agent.agent import SwingAgent
        
        agent = SwingAgent(
            interval="1d",
            lookback_days=30,
            use_llm=False,  # Skip LLM for basic test
            log_db=None,
            vec_db=None
        )
        
        signal = agent.analyze("AAPL")
        print(f"✓ Agent analysis successful")
        print(f"  Symbol: {signal.symbol}")
        print(f"  Trend: {signal.trend.label}")
        print(f"  Confidence: {signal.confidence}")
        return True
    except Exception as e:
        print(f"✗ Agent analysis error: {e}")
        return False

def test_llm_optional():
    """Test LLM functionality if API key available."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("~ LLM test skipped (no API key)")
        return True
        
    try:
        from swing_agent.llm_predictor import llm_extra_prediction
        
        vote = llm_extra_prediction(
            symbol="AAPL",
            price=150.0,
            trend_label="up",
            rsi_14=60.0
        )
        print(f"✓ LLM test successful")
        print(f"  Trend: {vote.trend_label}")
        print(f"  Confidence: {vote.confidence}")
        return True
    except Exception as e:
        print(f"✗ LLM test error: {e}")
        return False

def main():
    """Run all tests."""
    print("SwingAgent System Test")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_data_fetch,
        test_technical_analysis,
        test_agent_basic,
        test_llm_optional
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! SwingAgent is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
```

### Run the Test

```bash
# Save as test_system.py and run
python test_system.py

# Expected output:
# SwingAgent System Test
# ==============================
# ✓ All imports successful
# ✓ Data fetch successful: (5, 5)
# ✓ Technical analysis successful
#   Trend: up, RSI: 62.3
#   Entry: long
# ✓ Agent analysis successful
#   Symbol: AAPL
#   Trend: up
#   Confidence: 0.72
# ✓ LLM test successful
#   Trend: up
#   Confidence: 0.8
# 
# Test Results: 5/5 passed
# 🎉 All tests passed! SwingAgent is ready to use.
```

## Getting Help

### Debug Information Collection

```python
def collect_debug_info():
    """Collect system information for debugging."""
    import sys
    import platform
    import pandas as pd
    import numpy as np
    
    info = {
        "Python version": sys.version,
        "Platform": platform.platform(),
        "Pandas version": pd.__version__,
        "Numpy version": np.__version__,
        "Working directory": os.getcwd(),
        "Python path": sys.path[:3]  # First 3 entries
    }
    
    # Environment variables
    env_vars = ["OPENAI_API_KEY", "SWING_LLM_MODEL", "PYTHONPATH"]
    for var in env_vars:
        value = os.getenv(var, "Not set")
        if "API_KEY" in var and value != "Not set":
            value = f"{value[:8]}..."  # Hide most of API key
        info[var] = value
    
    print("Debug Information:")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key}: {value}")

# Run when reporting issues
collect_debug_info()
```

### Additional Troubleshooting Scenarios

#### Database Migration Issues

**Problem**: Error migrating from separate SQLite files to centralized database.

```bash
Error: duplicate column name: id
```

**Solutions**:
```bash
# Check current database schema
python -c "
from swing_agent.database import get_session
with get_session() as session:
    result = session.execute('PRAGMA table_info(signals)').fetchall()
    print('Signals table columns:', result)
"

# If migration fails, backup and recreate
mv data/swing_agent.sqlite data/swing_agent_backup.sqlite
python -m swing_agent.migrate --data-dir data/ --force-recreate
```

#### Vector Store Performance Issues

**Problem**: Slow vector similarity search with large datasets.

**Solutions**:
```python
# Check vector store size
from swing_agent.vectorstore import get_vector_count
count = get_vector_count("data/swing_agent.sqlite")
print(f"Vector count: {count}")

# If >10,000 vectors, consider pruning old vectors
from swing_agent.vectorstore import prune_old_vectors
prune_old_vectors("data/swing_agent.sqlite", keep_days=365)

# Or rebuild with performance indexes
python scripts/backfill_vector_store.py --rebuild-indexes
```

#### Configuration Conflicts

**Problem**: Environment variables not taking effect.

**Solutions**:
```bash
# Check environment variable precedence
python -c "
import os
from swing_agent.database import get_database_config
config = get_database_config()
print('Database URL:', config.database_url)
print('Environment SWING_DATABASE_URL:', os.getenv('SWING_DATABASE_URL'))
"

# Clear any cached configurations
rm -rf __pycache__
rm -rf src/swing_agent/__pycache__
```

#### Memory Usage Issues

**Problem**: High memory usage during backtesting.

**Solutions**:
```python
# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Reduce batch size for large backtests
python scripts/backtest_generate_signals.py --batch-size 100 --symbol AAPL
```

#### LLM Rate Limiting

**Problem**: OpenAI API rate limits during batch processing.

**Solutions**:
```python
# Add rate limiting
import time
from functools import wraps

def rate_limit(calls_per_minute=60):
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = 60.0 / calls_per_minute - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Apply to LLM functions
@rate_limit(calls_per_minute=30)
def llm_with_rate_limit(**features):
    return llm_extra_prediction(**features)
```

### Development Troubleshooting

#### IDE Setup Issues

**Problem**: VSCode/PyCharm not recognizing swing_agent imports.

**Solutions**:
```bash
# Install in development mode
pip install -e .

# Set Python interpreter to virtual environment
# VSCode: Ctrl+Shift+P -> "Python: Select Interpreter"
# PyCharm: File -> Settings -> Project -> Python Interpreter

# Add src to Python path in IDE settings
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### Testing Framework Issues

**Problem**: Tests not discovering or running properly.

**Solutions**:
```bash
# Ensure pytest is installed
pip install pytest pytest-cov

# Run from project root
cd /path/to/ITL.SwingAgent
pytest tests/

# Check test discovery
pytest --collect-only

# If modules not found, install package
pip install -e .
```

#### Git and Version Control

**Problem**: Large database files in git history.

**Solutions**:
```bash
# Add to .gitignore if not already there
echo "data/*.sqlite" >> .gitignore
echo "logs/" >> .gitignore
echo "__pycache__/" >> .gitignore

# Remove from git history (careful!)
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch data/*.sqlite' \
--prune-empty --tag-name-filter cat -- --all
```

### Performance Optimization Troubleshooting

#### Slow Data Fetching

**Problem**: yfinance data fetching is slow or unreliable.

**Solutions**:
```python
# Add caching for development
import functools
import pickle
from pathlib import Path

@functools.lru_cache(maxsize=128)
def cached_load_ohlcv(symbol, interval, lookback_days):
    cache_file = Path(f"cache/{symbol}_{interval}_{lookback_days}.pkl")
    
    if cache_file.exists():
        # Check if cache is recent (< 1 hour)
        import time
        if time.time() - cache_file.stat().st_mtime < 3600:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Fetch fresh data
    df = load_ohlcv(symbol, interval, lookback_days)
    
    # Cache for next time
    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f)
    
    return df
```

#### Database Performance

**Problem**: Slow database queries with large signal history.

**Solutions**:
```sql
-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_signals_symbol_asof ON signals(symbol, asof);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at_utc);
CREATE INDEX IF NOT EXISTS idx_vectorstore_symbol_ts ON vec_store(symbol, ts_utc);

-- Analyze query performance
EXPLAIN QUERY PLAN SELECT * FROM signals WHERE symbol = 'AAPL' ORDER BY asof DESC LIMIT 10;
```

### Common Error Messages and Solutions

#### Import Errors

```
ModuleNotFoundError: No module named 'swing_agent'
```
**Solution**: Install package with `pip install -e .`

#### Database Errors

```
sqlalchemy.exc.OperationalError: no such table: signals
```
**Solution**: Initialize database with `python -m swing_agent.database --init`

#### API Errors

```
openai.error.AuthenticationError: Invalid API key
```
**Solution**: Check `OPENAI_API_KEY` environment variable

#### Permission Errors

```
PermissionError: [Errno 13] Permission denied: 'data/swing_agent.sqlite'
```
**Solution**: Check file permissions with `ls -la data/` and fix with `chmod 600 data/swing_agent.sqlite`

### Debug Information Collection

Create a comprehensive debug script to gather system information: