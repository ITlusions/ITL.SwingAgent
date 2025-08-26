"""
Centralized database configuration and session management using SQLAlchemy.
Supports both local SQLite and external databases (PostgreSQL, MySQL, etc.).
"""
from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool


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


def create_cnpg_url() -> str | None:
    """Create CNPG (CloudNativePG) database URL from environment variables.
    
    CNPG provides multiple connection methods:
    1. Direct cluster service: <cluster-name>-rw.<namespace>.svc.cluster.local
    2. Read-only service: <cluster-name>-ro.<namespace>.svc.cluster.local
    3. Read-write service: <cluster-name>-rw.<namespace>.svc.cluster.local
    
    Environment variables:
    - CNPG_CLUSTER_NAME or SWING_CNPG_CLUSTER: CNPG cluster name
    - CNPG_NAMESPACE or SWING_CNPG_NAMESPACE: Kubernetes namespace (default: default)
    - CNPG_SERVICE_TYPE or SWING_CNPG_SERVICE: rw (read-write) or ro (read-only), default: rw
    - SWING_DB_NAME: Database name
    - SWING_DB_USER: Database username
    - SWING_DB_PASSWORD: Database password
    - SWING_DB_PORT: Database port (default: 5432)
    - CNPG_SSL_MODE or SWING_CNPG_SSL_MODE: SSL mode (default: require)
    """
    # Get cluster configuration
    cluster_name = os.getenv("CNPG_CLUSTER_NAME") or os.getenv("SWING_CNPG_CLUSTER")
    if not cluster_name:
        return None

    namespace = os.getenv("CNPG_NAMESPACE") or os.getenv("SWING_CNPG_NAMESPACE", "default")
    service_type = os.getenv("CNPG_SERVICE_TYPE") or os.getenv("SWING_CNPG_SERVICE", "rw")

    # Build CNPG service hostname
    host = f"{cluster_name}-{service_type}.{namespace}.svc.cluster.local"

    # Get database connection details
    database = os.getenv("SWING_DB_NAME")
    username = os.getenv("SWING_DB_USER")
    password = os.getenv("SWING_DB_PASSWORD")
    port = int(os.getenv("SWING_DB_PORT", "5432"))

    if not all([database, username, password]):
        return None

    # SSL configuration
    ssl_mode = os.getenv("CNPG_SSL_MODE") or os.getenv("SWING_CNPG_SSL_MODE", "require")

    # Build PostgreSQL URL with SSL parameters
    base_url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"

    # Add SSL parameters
    ssl_params = f"?sslmode={ssl_mode}"

    # Add certificate paths if provided
    ssl_cert = os.getenv("CNPG_SSL_CERT") or os.getenv("SWING_CNPG_SSL_CERT")
    ssl_key = os.getenv("CNPG_SSL_KEY") or os.getenv("SWING_CNPG_SSL_KEY")
    ssl_ca = os.getenv("CNPG_SSL_CA") or os.getenv("SWING_CNPG_SSL_CA")

    if ssl_cert:
        ssl_params += f"&sslcert={ssl_cert}"
    if ssl_key:
        ssl_params += f"&sslkey={ssl_key}"
    if ssl_ca:
        ssl_params += f"&sslrootcert={ssl_ca}"

    return base_url + ssl_params


def from_env_config() -> str | None:
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
    4. CNPG (CloudNativePG) specific:
       - SWING_DB_TYPE=cnpg
       - CNPG_CLUSTER_NAME or SWING_CNPG_CLUSTER
       - Standard PostgreSQL credentials
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

    elif db_type == "cnpg":
        # CNPG specific configuration
        return create_cnpg_url()

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

    def __init__(self, database_url: str | None = None):
        # Default to a single SQLite file in data directory
        if database_url is None:
            database_url = self._get_default_database_url()

        self.database_url = database_url
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
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
        scheme = parsed.scheme.split('+')[0]  # Handle cases like postgresql+psycopg2

        # Check if this is a CNPG cluster connection
        if scheme == "postgresql" and parsed.hostname and "svc.cluster.local" in parsed.hostname:
            return "cnpg"

        return scheme

    def _get_sqlite_config(self) -> dict[str, Any]:
        """Get SQLite-specific configuration."""
        return {
            "poolclass": StaticPool,
            "connect_args": {
                "check_same_thread": False,
                "timeout": 30
            },
            "echo": os.getenv("SWING_DB_ECHO", "").lower() == "true"
        }

    def _get_postgresql_config(self) -> dict[str, Any]:
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

    def _get_mysql_config(self) -> dict[str, Any]:
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

    def _get_cnpg_config(self) -> dict[str, Any]:
        """Get CNPG (CloudNativePG) specific configuration."""
        return {
            "poolclass": QueuePool,
            "pool_size": int(os.getenv("SWING_DB_POOL_SIZE", "10")),  # Higher default for CNPG
            "max_overflow": int(os.getenv("SWING_DB_MAX_OVERFLOW", "20")),
            "pool_timeout": int(os.getenv("SWING_DB_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.getenv("SWING_DB_POOL_RECYCLE", "1800")),  # Shorter for k8s
            "pool_pre_ping": True,
            "pool_reset_on_return": "commit",  # Important for CNPG
            "echo": os.getenv("SWING_DB_ECHO", "").lower() == "true",
            # CNPG specific connection args
            "connect_args": {
                "connect_timeout": int(os.getenv("CNPG_CONNECT_TIMEOUT", "10")),
                "application_name": os.getenv("CNPG_APP_NAME", "swing-agent"),
            }
        }

    def _get_engine_config(self) -> dict[str, Any]:
        """Get database-specific engine configuration."""
        if self._db_type == "sqlite":
            return self._get_sqlite_config()
        elif self._db_type == "postgresql":
            return self._get_postgresql_config()
        elif self._db_type == "cnpg":
            return self._get_cnpg_config()
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

    @contextmanager
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

    @property
    def is_cnpg(self) -> bool:
        """Check if using CNPG (CloudNativePG) database."""
        return self._db_type == "cnpg"

    def get_database_info(self) -> dict[str, Any]:
        """Get information about the current database configuration."""
        parsed = urlparse(self.database_url)
        info = {
            "type": self._db_type,
            "host": parsed.hostname,
            "port": parsed.port,
            "database": parsed.path.lstrip('/') if parsed.path else None,
            "is_sqlite": self.is_sqlite,
            "is_external": self.is_external,
            "is_cnpg": self.is_cnpg,
            "url_masked": self._mask_credentials(self.database_url)
        }

        # Add CNPG specific information
        if self.is_cnpg and parsed.hostname:
            hostname_parts = parsed.hostname.split('.')
            if len(hostname_parts) >= 5:  # cluster-service.namespace.svc.cluster.local
                cluster_parts = hostname_parts[0].split('-')
                if len(cluster_parts) >= 2:
                    info["cnpg_cluster"] = '-'.join(cluster_parts[:-1])
                    info["cnpg_service"] = cluster_parts[-1]
                    info["cnpg_namespace"] = hostname_parts[1]

        return info

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
_db_config: DatabaseConfig | None = None


def get_database_config(database_url: str | None = None) -> DatabaseConfig:
    """Get the global database configuration.
    
    Returns a singleton DatabaseConfig instance that manages the centralized
    database connection for the entire SwingAgent system.
    
    Args:
        database_url: Optional database URL to override default configuration.
                     If None, uses environment variables or defaults to SQLite.
                     
    Returns:
        DatabaseConfig: Configured database instance with connection management.
        
    Example:
        >>> config = get_database_config()
        >>> print(f"Database type: {config.get_database_info()['type']}")
        >>> print(f"Is external: {config.is_external}")
    """
    global _db_config
    if _db_config is None or database_url is not None:
        _db_config = DatabaseConfig(database_url)
    return _db_config


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session from the global configuration.
    
    Provides a SQLAlchemy session with automatic transaction management.
    Sessions are automatically committed on success and rolled back on exceptions.
    
    Yields:
        Session: SQLAlchemy database session with automatic cleanup.
        
    Example:
        >>> from swing_agent.models_db import Signal
        >>> with get_session() as session:
        ...     signals = session.query(Signal).filter(
        ...         Signal.symbol == "AAPL"
        ...     ).all()
        ...     print(f"Found {len(signals)} signals")
        
    Note:
        This is the recommended way to interact with the database. The session
        is automatically committed if no exceptions occur, and rolled back if
        any exception is raised.
    """
    db_config = get_database_config()
    with db_config.get_session() as session:
        yield session


def init_database(database_url: str | None = None):
    """Initialize the database with tables and configuration.
    
    Creates all necessary database tables if they don't exist. This is 
    typically called once during setup or when migrating to a new database.
    
    Args:
        database_url: Optional database URL. If None, uses environment 
                     configuration or defaults to SQLite.
                     
    Example:
        >>> # Initialize with default configuration
        >>> init_database()
        
        >>> # Initialize with specific PostgreSQL database
        >>> init_database("postgresql://user:pass@host:5432/swing_agent")
        
        >>> # Initialize with environment variables
        >>> import os
        >>> os.environ["SWING_DATABASE_URL"] = "postgresql://..."
        >>> init_database()
        
    Note:
        This function is idempotent - it's safe to call multiple times.
        Tables that already exist will not be modified.
    """
    db_config = get_database_config(database_url)
    db_config.ensure_db()


def get_database_info() -> dict[str, Any]:
    """Get information about the current database configuration."""
    db_config = get_database_config()
    return db_config.get_database_info()
