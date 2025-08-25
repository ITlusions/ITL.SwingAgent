# Database Centralization (v1.6.1)

## What Changed

SwingAgent v1.6.1 introduces **centralized database management** with these key improvements:

### Before (v1.6.0 and earlier)
- 🔀 **Separate files**: `data/signals.sqlite` + `data/vec_store.sqlite`  
- 🔧 **Raw sqlite3**: Direct SQL queries with potential inconsistencies
- 📁 **Manual management**: Separate backup/deployment of multiple files

### After (v1.6.1+)
- 🎯 **Single file**: `data/swing_agent.sqlite` (contains both signals and vectors)
- 🏗️ **SQLAlchemy ORM**: Type-safe, consistent database operations  
- 📦 **Simplified management**: One file to backup, deploy, and manage

## Migration Guide

### Automatic Migration

Run this once to migrate your existing data:

```bash
# From project root
python -m swing_agent.migrate --data-dir data/
```

This will:
1. Create new centralized database: `data/swing_agent.sqlite`
2. Copy all signals from `data/signals.sqlite` 
3. Copy all vectors from `data/vec_store.sqlite`
4. Preserve your existing files (no data loss)

### Manual Usage

All scripts now default to the centralized database:

```bash
# These all use data/swing_agent.sqlite by default now
python scripts/run_swing_agent.py --symbol AAPL
python scripts/eval_signals.py  
python scripts/analyze_performance.py
```

### Backward Compatibility

You can still specify separate databases if needed:

```bash
# Use legacy separate files
python scripts/run_swing_agent.py \
  --db data/signals.sqlite \
  --vec-db data/vec_store.sqlite
```

But we recommend migrating to the centralized approach for better performance and management.

## Benefits

✅ **Simpler deployment**: One database file instead of two  
✅ **Better performance**: Proper indexes and query optimization  
✅ **Type safety**: SQLAlchemy models prevent data inconsistencies  
✅ **Easier backup**: Just copy one file  
✅ **Future-ready**: Easier to migrate to PostgreSQL/MySQL later if needed  

## Troubleshooting

### Migration Issues

If migration fails, check:

```python
# Verify old files exist
ls -la data/signals.sqlite data/vec_store.sqlite

# Check permissions
chmod 644 data/*.sqlite

# Manual migration with custom paths
python -m swing_agent.migrate \
  --data-dir /custom/path \
  --signals-file my_signals.sqlite \
  --vectors-file my_vectors.sqlite
```

### Import Errors

If you get SQLAlchemy import errors, ensure dependencies are installed:

```bash
pip install sqlalchemy>=2.0.0
# or
pip install -e .  # Install with all dependencies
```