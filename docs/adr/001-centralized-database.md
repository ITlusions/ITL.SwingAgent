# ADR-001: Centralized Database Architecture

## Status

Accepted

## Context

The SwingAgent system generates multiple types of data:
- Trading signals with extensive metadata 
- Feature vectors for pattern matching
- Evaluation results and outcomes
- Configuration and enrichment data

Previously, this data was scattered across multiple SQLite files, making:
- Data consistency challenging
- Cross-dataset queries complex
- Deployment and backup procedures error-prone
- Development setup cumbersome

## Decision

We will adopt a centralized database architecture where:

1. **Single Database Instance**: All data (signals, vectors, evaluations) stored in one database
2. **SQLAlchemy ORM**: Use SQLAlchemy for all database operations with proper schema management
3. **Multiple Backend Support**: Support SQLite (development), PostgreSQL, and MySQL via connection strings
4. **Backward Compatibility**: Automatically migrate from old multi-file approach
5. **Centralized Configuration**: Single database URL configuration via environment variables

## Implementation Details

```python
# Centralized database configuration
database_url = get_database_config().database_url  # From environment or default

# All operations use same session factory
with get_session() as session:
    # Signals, vectors, and all data operations
```

**Schema Organization**:
- `signals` table: Complete trading signals with all metadata
- `vec_store` table: Feature vectors with outcomes for ML
- Foreign key relationships where appropriate
- JSON columns for flexible metadata storage

## Consequences

### Positive

- **Simplified Operations**: Single database to backup, migrate, and monitor
- **ACID Compliance**: All related data changes in same transaction
- **Better Performance**: Joins and complex queries possible across all data
- **Production Ready**: Easy to deploy with external databases (PostgreSQL/MySQL)
- **Development Velocity**: Simplified setup with single database file

### Negative

- **Migration Complexity**: Existing installations need data migration
- **Single Point of Failure**: Database issues affect entire system
- **Lock Contention**: High concurrency may require connection pooling

## Migration Strategy

```python
# Automatic migration in _ensure_db() functions
if old_sqlite_files_detected:
    migrate_to_centralized_database()
```

## Monitoring

- Database connection health checks
- Query performance monitoring  
- Storage space utilization alerts
- Backup verification procedures