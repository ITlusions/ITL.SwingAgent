"""
Centralized database configuration and session management using SQLAlchemy.
Supports both local SQLite and external databases (PostgreSQL, MySQL, etc.).
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Generator, Optional, Dict, Any
from urllib.parse import urlparse
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import StaticPool, QueuePool


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


def create_postgresql_url(
    host: str,
    database: str,
    username: str,
    password: str,
    port: int = 5432,
    driver: str = "psycopg2"
) -> str:
    """Create PostgreSQL database URL."""
    return f"postgresql+{driver}://{username}:{password}@{host}:{port}/{database}"


def create_mysql_url(
    host: str,
    database: str,
    username: str,
    password: str,
    port: int = 3306,
    driver: str = "pymysql"
) -> str:
    """Create MySQL database URL."""
    return f"mysql+{driver}://{username}:{password}@{host}:{port}/{database}"


def from_env_config() -> Optional[str]:
    """Create database URL from environment variables.
    
    Supports multiple configuration patterns:
    1. SWING_DATABASE_URL - Full database URL
    2. Individual components for PostgreSQL:
       - SWING_DB_TYPE=postgresql
       - SWING_DB_HOST, SWING_DB_PORT, SWING_DB_NAME
       - SWING_DB_USER, SWING_DB_PASSWORD
    3. Individual components for MySQL:
       - SWING_DB_TYPE=mysql
       - SWING_DB_HOST, SWING_DB_PORT, SWING_DB_NAME
       - SWING_DB_USER, SWING_DB_PASSWORD
    """
    # Direct URL takes precedence
    direct_url = os.getenv("SWING_DATABASE_URL")
    if direct_url:
        return direct_url
    
    # Build from components
    db_type = os.getenv("SWING_DB_TYPE", "").lower()
    if not db_type:
        return None
    
    if db_type in ["postgresql", "postgres"]:
        host = os.getenv("SWING_DB_HOST")
        database = os.getenv("SWING_DB_NAME")
        username = os.getenv("SWING_DB_USER")
        password = os.getenv("SWING_DB_PASSWORD")
        port = int(os.getenv("SWING_DB_PORT", "5432"))
        
        if all([host, database, username, password]):
            return create_postgresql_url(host, database, username, password, port)
    
    elif db_type == "mysql":
        host = os.getenv("SWING_DB_HOST")
        database = os.getenv("SWING_DB_NAME")
        username = os.getenv("SWING_DB_USER")
        password = os.getenv("SWING_DB_PASSWORD")
        port = int(os.getenv("SWING_DB_PORT", "3306"))
        
        if all([host, database, username, password]):
            return create_mysql_url(host, database, username, password, port)
    
    return None


class DatabaseConfig:
    """Centralized database configuration supporting SQLite and external databases."""
    
    def __init__(self, database_url: Optional[str] = None):
        # Default to a single SQLite file in data directory
        if database_url is None:
            database_url = self._get_default_database_url()
        
        self.database_url = database_url
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._db_type = self._detect_database_type()
    
    def _get_default_database_url(self) -> str:
        """Get default database URL from environment or fallback to SQLite."""
        # Check for environment configuration first
        env_url = from_env_config()
        if env_url:
            return env_url
        
        # Fallback to local SQLite
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        database_path = data_dir / "swing_agent.sqlite"
        return f"sqlite:///{database_path}"
    
    def _detect_database_type(self) -> str:
        """Detect database type from URL."""
        parsed = urlparse(self.database_url)
        return parsed.scheme.split('+')[0]  # Handle cases like postgresql+psycopg2
    
    def _get_sqlite_config(self) -> Dict[str, Any]:
        """Get SQLite-specific configuration."""
        return {
            "poolclass": StaticPool,
            "connect_args": {
                "check_same_thread": False,
                "timeout": 30
            },
            "echo": os.getenv("SWING_DB_ECHO", "").lower() == "true"
        }
    
    def _get_postgresql_config(self) -> Dict[str, Any]:
        """Get PostgreSQL-specific configuration."""
        return {
            "poolclass": QueuePool,
            "pool_size": int(os.getenv("SWING_DB_POOL_SIZE", "5")),
            "max_overflow": int(os.getenv("SWING_DB_MAX_OVERFLOW", "10")),
            "pool_timeout": int(os.getenv("SWING_DB_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.getenv("SWING_DB_POOL_RECYCLE", "3600")),
            "pool_pre_ping": True,
            "echo": os.getenv("SWING_DB_ECHO", "").lower() == "true"
        }
    
    def _get_mysql_config(self) -> Dict[str, Any]:
        """Get MySQL-specific configuration."""
        return {
            "poolclass": QueuePool,
            "pool_size": int(os.getenv("SWING_DB_POOL_SIZE", "5")),
            "max_overflow": int(os.getenv("SWING_DB_MAX_OVERFLOW", "10")),
            "pool_timeout": int(os.getenv("SWING_DB_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.getenv("SWING_DB_POOL_RECYCLE", "3600")),
            "pool_pre_ping": True,
            "echo": os.getenv("SWING_DB_ECHO", "").lower() == "true"
        }
    
    def _get_engine_config(self) -> Dict[str, Any]:
        """Get database-specific engine configuration."""
        if self._db_type == "sqlite":
            return self._get_sqlite_config()
        elif self._db_type == "postgresql":
            return self._get_postgresql_config()
        elif self._db_type == "mysql":
            return self._get_mysql_config()
        else:
            # Generic configuration for other databases
            return {
                "poolclass": QueuePool,
                "pool_size": int(os.getenv("SWING_DB_POOL_SIZE", "5")),
                "max_overflow": int(os.getenv("SWING_DB_MAX_OVERFLOW", "10")),
                "pool_timeout": int(os.getenv("SWING_DB_POOL_TIMEOUT", "30")),
                "pool_recycle": int(os.getenv("SWING_DB_POOL_RECYCLE", "3600")),
                "pool_pre_ping": True,
                "echo": os.getenv("SWING_DB_ECHO", "").lower() == "true"
            }
    
    @property
    def engine(self) -> Engine:
        """Get the database engine."""
        if self._engine is None:
            config = self._get_engine_config()
            self._engine = create_engine(self.database_url, **config)
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
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self._db_type == "sqlite"
    
    @property
    def is_external(self) -> bool:
        """Check if using external database (non-SQLite)."""
        return self._db_type != "sqlite"
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database configuration."""
        parsed = urlparse(self.database_url)
        return {
            "type": self._db_type,
            "host": parsed.hostname,
            "port": parsed.port,
            "database": parsed.path.lstrip('/') if parsed.path else None,
            "is_sqlite": self.is_sqlite,
            "is_external": self.is_external,
            "url_masked": self._mask_credentials(self.database_url)
        }
    
    def _mask_credentials(self, url: str) -> str:
        """Mask credentials in database URL for logging."""
        parsed = urlparse(url)
        if parsed.username:
            masked_netloc = parsed.netloc.replace(
                f"{parsed.username}:{parsed.password}@" if parsed.password else f"{parsed.username}@",
                "***:***@" if parsed.password else "***@"
            )
            return url.replace(parsed.netloc, masked_netloc)
        return url


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


def get_database_info() -> Dict[str, Any]:
    """Get information about the current database configuration."""
    db_config = get_database_config()
    return db_config.get_database_info()