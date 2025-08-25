"""
Centralized database configuration and session management using SQLAlchemy.
Replaces the previous separate SQLite files approach with a single database.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Generator, Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class DatabaseConfig:
    """Centralized database configuration."""
    
    def __init__(self, database_url: Optional[str] = None):
        # Default to a single SQLite file in data directory
        if database_url is None:
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            database_path = data_dir / "swing_agent.sqlite"
            database_url = f"sqlite:///{database_path}"
        
        self.database_url = database_url
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
    
    @property
    def engine(self) -> Engine:
        """Get the database engine."""
        if self._engine is None:
            if self.database_url.startswith("sqlite"):
                # SQLite-specific configuration
                self._engine = create_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 30
                    },
                    echo=False
                )
            else:
                # For other databases like PostgreSQL
                self._engine = create_engine(self.database_url, echo=False)
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory
    
    def create_all_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def ensure_db(self):
        """Ensure database and tables exist."""
        self.create_all_tables()


# Global database instance
_db_config: Optional[DatabaseConfig] = None


def get_database_config(database_url: Optional[str] = None) -> DatabaseConfig:
    """Get the global database configuration."""
    global _db_config
    if _db_config is None or database_url is not None:
        _db_config = DatabaseConfig(database_url)
    return _db_config


def get_session() -> Generator[Session, None, None]:
    """Get a database session from the global configuration."""
    db_config = get_database_config()
    yield from db_config.get_session()


def init_database(database_url: Optional[str] = None):
    """Initialize the database with the given URL."""
    db_config = get_database_config(database_url)
    db_config.ensure_db()