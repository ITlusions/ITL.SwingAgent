# External Database Configuration

SwingAgent supports external databases including PostgreSQL and MySQL for production deployments. This guide covers setup and configuration options.

## Quick Start

### PostgreSQL Setup

```bash
# Set environment variables
export SWING_DB_TYPE=postgresql
export SWING_DB_HOST=localhost
export SWING_DB_PORT=5432
export SWING_DB_NAME=swing_agent
export SWING_DB_USER=your_username
export SWING_DB_PASSWORD=your_password

# Or use a direct database URL
export SWING_DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/swing_agent"
```

### MySQL Setup

```bash
# Set environment variables
export SWING_DB_TYPE=mysql
export SWING_DB_HOST=localhost
export SWING_DB_PORT=3306
export SWING_DB_NAME=swing_agent
export SWING_DB_USER=your_username
export SWING_DB_PASSWORD=your_password

# Or use a direct database URL
export SWING_DATABASE_URL="mysql+pymysql://user:pass@localhost:3306/swing_agent"
```

## Environment Variables

### Database Connection

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SWING_DATABASE_URL` | Complete database URL | SQLite file | `postgresql://user:pass@host:5432/db` |
| `SWING_DB_TYPE` | Database type | `sqlite` | `postgresql`, `mysql` |
| `SWING_DB_HOST` | Database host | `localhost` | `db.example.com` |
| `SWING_DB_PORT` | Database port | DB-specific | `5432`, `3306` |
| `SWING_DB_NAME` | Database name | - | `swing_agent` |
| `SWING_DB_USER` | Database username | - | `swing_user` |
| `SWING_DB_PASSWORD` | Database password | - | `secure_password` |

### Connection Pooling

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SWING_DB_POOL_SIZE` | Number of connections to maintain | `5` | `10` |
| `SWING_DB_MAX_OVERFLOW` | Max additional connections | `10` | `20` |
| `SWING_DB_POOL_TIMEOUT` | Connection timeout (seconds) | `30` | `60` |
| `SWING_DB_POOL_RECYCLE` | Connection recycle time (seconds) | `3600` | `7200` |
| `SWING_DB_ECHO` | Enable SQL query logging | `false` | `true` |

## Database Setup

### PostgreSQL

1. **Install PostgreSQL and Python driver:**
   ```bash
   # Install PostgreSQL (varies by OS)
   # Ubuntu/Debian: sudo apt-get install postgresql
   # macOS: brew install postgresql
   
   # Install Python driver
   pip install psycopg2-binary
   ```

2. **Create database and user:**
   ```sql
   -- Connect to PostgreSQL as superuser
   psql -U postgres
   
   -- Create database and user
   CREATE DATABASE swing_agent;
   CREATE USER swing_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE swing_agent TO swing_user;
   
   -- Grant schema permissions
   \c swing_agent
   GRANT ALL ON SCHEMA public TO swing_user;
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO swing_user;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO swing_user;
   ```

3. **Initialize tables:**
   ```python
   from swing_agent.database import init_database
   init_database()  # Will create all tables
   ```

### MySQL

1. **Install MySQL and Python driver:**
   ```bash
   # Install MySQL (varies by OS)
   # Ubuntu/Debian: sudo apt-get install mysql-server
   # macOS: brew install mysql
   
   # Install Python driver
   pip install PyMySQL
   ```

2. **Create database and user:**
   ```sql
   -- Connect to MySQL as root
   mysql -u root -p
   
   -- Create database and user
   CREATE DATABASE swing_agent;
   CREATE USER 'swing_user'@'%' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON swing_agent.* TO 'swing_user'@'%';
   FLUSH PRIVILEGES;
   ```

3. **Initialize tables:**
   ```python
   from swing_agent.database import init_database
   init_database()  # Will create all tables
   ```

## Migration from SQLite

To migrate existing SQLite data to an external database:

```python
from swing_agent.migrate import migrate_to_external_db
from swing_agent.database import create_postgresql_url

# Create external database URL
external_url = create_postgresql_url(
    host="localhost",
    database="swing_agent",
    username="swing_user",
    password="your_password"
)

# Migrate data
migrate_to_external_db(
    sqlite_path="data/swing_agent.sqlite",
    external_url=external_url
)
```

## Usage Examples

### Basic Usage with Environment Variables

```python
# Set environment variables first
import os
os.environ["SWING_DB_TYPE"] = "postgresql"
os.environ["SWING_DB_HOST"] = "localhost"
os.environ["SWING_DB_NAME"] = "swing_agent"
os.environ["SWING_DB_USER"] = "swing_user"
os.environ["SWING_DB_PASSWORD"] = "your_password"

# Use SwingAgent normally - it will automatically use external DB
from swing_agent import SwingAgent
agent = SwingAgent(symbol="AMD")
signal = agent.analyze("AMD")
```

### Explicit Database URL

```python
from swing_agent import SwingAgent
from swing_agent.database import create_postgresql_url

# Create database URL
db_url = create_postgresql_url(
    host="localhost",
    database="swing_agent",
    username="swing_user",
    password="your_password"
)

# Initialize database configuration
from swing_agent.database import init_database
init_database(db_url)

# Use SwingAgent with external database
agent = SwingAgent()
signal = agent.analyze("AMD")
```

### Database Information

```python
from swing_agent.database import get_database_info

# Get current database configuration
info = get_database_info()
print(f"Database type: {info['type']}")
print(f"Is external: {info['is_external']}")
print(f"Connection: {info['url_masked']}")
```

## Production Considerations

### Security
- Store credentials in environment variables or secure secret management
- Use SSL/TLS connections for external databases
- Configure proper firewall rules and network security
- Use dedicated database users with minimal required permissions

### Performance
- Configure connection pooling based on your workload
- Set appropriate pool sizes and timeouts
- Consider read replicas for read-heavy workloads
- Monitor database performance and optimize queries

### Backup and Recovery
- Implement regular database backups
- Test backup restoration procedures
- Consider point-in-time recovery for critical data
- Document disaster recovery procedures

### Monitoring
- Enable database query logging for troubleshooting (`SWING_DB_ECHO=true`)
- Monitor connection pool usage and performance
- Set up database monitoring and alerting
- Track query performance and optimize slow queries

## Troubleshooting

### Connection Issues

1. **Check credentials and network connectivity:**
   ```python
   from swing_agent.database import get_database_config
   
   config = get_database_config()
   try:
       engine = config.engine
       with engine.connect() as conn:
           print("Connection successful!")
   except Exception as e:
       print(f"Connection failed: {e}")
   ```

2. **Verify environment variables:**
   ```python
   import os
   print("Database config:")
   for key in os.environ:
       if key.startswith("SWING_DB"):
           print(f"{key}: {os.environ[key]}")
   ```

### Performance Issues

1. **Enable query logging:**
   ```bash
   export SWING_DB_ECHO=true
   ```

2. **Adjust connection pool settings:**
   ```bash
   export SWING_DB_POOL_SIZE=10
   export SWING_DB_MAX_OVERFLOW=20
   ```

3. **Check database indexes:**
   The application automatically creates performance indexes on commonly queried columns.

### Migration Issues

1. **Check existing data before migration:**
   ```python
   from swing_agent.storage import get_signals_count
   from swing_agent.vectorstore import get_vectors_count
   
   print(f"Signals to migrate: {get_signals_count()}")
   print(f"Vectors to migrate: {get_vectors_count()}")
   ```

2. **Verify migration success:**
   ```python
   # After migration, check external database
   from swing_agent.database import get_database_info
   info = get_database_info()
   print(f"Using external database: {info['is_external']}")
   ```