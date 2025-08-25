# Security & Best Practices Guide

Security considerations and best practices for SwingAgent v1.6.1 deployment and development.

## Security Overview

SwingAgent handles sensitive financial data and integrates with external APIs. This guide covers security considerations for:

- API key management and rotation
- Database security and encryption  
- Multi-backend database security (SQLite, PostgreSQL, MySQL, CNPG)
- Input validation and sanitization
- Network security and TLS
- Logging and audit trails
- Production deployment security
- Kubernetes security for CNPG deployments

## Database Security

### Connection Security

#### PostgreSQL Security
```bash
# Use SSL connections in production
export SWING_DATABASE_URL="postgresql://user:pass@host:5432/swing_agent?sslmode=require"

# Certificate-based authentication
export SWING_DB_SSL_CERT="/path/to/client-cert.pem"
export SWING_DB_SSL_KEY="/path/to/client-key.pem"
export SWING_DB_SSL_CA="/path/to/ca-cert.pem"
```

#### CloudNativePG Security
```bash
# CNPG with SSL configuration
export CNPG_SSL_MODE="require"
export CNPG_SSL_CERT="/var/secrets/client-cert.pem"
export CNPG_SSL_KEY="/var/secrets/client-key.pem"
export CNPG_SSL_CA="/var/secrets/ca-cert.pem"
```

#### Database Credentials Management
```python
# Use environment variables, never hardcode
from swing_agent.database import get_database_config

# Recommended: Use secrets management
import os
from pathlib import Path

def load_db_password():
    """Load database password from secure source."""
    # Option 1: Kubernetes secret
    secret_path = Path("/var/secrets/db-password")
    if secret_path.exists():
        return secret_path.read_text().strip()
    
    # Option 2: Environment variable (development)
    return os.getenv("SWING_DB_PASSWORD")

# Set password securely
os.environ["SWING_DB_PASSWORD"] = load_db_password()
```

### Data Encryption

#### Database-Level Encryption
```sql
-- PostgreSQL: Enable transparent data encryption
ALTER DATABASE swing_agent SET default_text_search_config = 'pg_catalog.english';

-- Enable row-level security for sensitive data
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY signals_policy ON signals FOR ALL TO swing_user;
```

#### Application-Level Encryption
```python
# Encrypt sensitive fields before storage
from cryptography.fernet import Fernet
import os

class SecureStorage:
    def __init__(self):
        key = os.getenv("SWING_ENCRYPTION_KEY")
        if not key:
            raise ValueError("SWING_ENCRYPTION_KEY required")
        self.cipher = Fernet(key.encode())
    
    def encrypt_llm_data(self, data: str) -> str:
        """Encrypt LLM responses before database storage."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_llm_data(self, encrypted_data: str) -> str:
        """Decrypt LLM responses after database retrieval."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
```

## API Key Security

### Current Implementation Review

**Strengths:**
```python
# Environment variable usage (good practice)
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("SWING_LLM_MODEL", "gpt-4o-mini")
```

**Areas for Improvement:**

### 1. API Key Validation

```python
# security/api_keys.py
import os
import re
from typing import Optional
import logging

class APIKeyManager:
    def __init__(self):
        self.logger = logging.getLogger("swing_agent.security")
    
    def validate_openai_key(self, api_key: Optional[str]) -> bool:
        """Validate OpenAI API key format."""
        if not api_key:
            return False
            
        # OpenAI keys start with 'sk-' and have specific length
        pattern = r'^sk-[A-Za-z0-9]{48}$'
        if not re.match(pattern, api_key):
            self.logger.warning("Invalid OpenAI API key format")
            return False
            
        return True
    
    def get_validated_key(self, env_var: str) -> Optional[str]:
        """Get and validate API key from environment."""
        key = os.getenv(env_var)
        
        if env_var == "OPENAI_API_KEY":
            if not self.validate_openai_key(key):
                raise ValueError("Invalid or missing OpenAI API key")
                
        return key
    
    def mask_key_for_logging(self, key: str) -> str:
        """Mask API key for safe logging."""
        if len(key) <= 8:
            return "*" * len(key)
        return key[:4] + "*" * (len(key) - 8) + key[-4:]

# Usage in llm_predictor.py
api_manager = APIKeyManager()
api_key = api_manager.get_validated_key("OPENAI_API_KEY")
```

### 2. Key Rotation Support

```python
# security/key_rotation.py
from datetime import datetime, timedelta
import json
from pathlib import Path

class KeyRotationManager:
    def __init__(self, config_path: str = "data/key_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load key rotation configuration."""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"last_rotation": None, "rotation_interval_days": 90}
    
    def _save_config(self):
        """Save key rotation configuration."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self.config, indent=2))
    
    def should_rotate_key(self) -> bool:
        """Check if key should be rotated based on policy."""
        if not self.config.get("last_rotation"):
            return True
            
        last_rotation = datetime.fromisoformat(self.config["last_rotation"])
        rotation_interval = timedelta(days=self.config["rotation_interval_days"])
        
        return datetime.utcnow() - last_rotation > rotation_interval
    
    def record_key_rotation(self):
        """Record that key was rotated."""
        self.config["last_rotation"] = datetime.utcnow().isoformat()
        self._save_config()
```

## Database Security

### Connection Security

```python
# security/database.py
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import ssl

class SecureDatabaseConfig:
    @staticmethod
    def create_secure_engine(database_url: str) -> Engine:
        """Create database engine with security best practices."""
        
        # SSL/TLS configuration for PostgreSQL
        if database_url.startswith("postgresql"):
            connect_args = {
                "sslmode": "require",
                "sslcert": os.getenv("DB_SSL_CERT"),
                "sslkey": os.getenv("DB_SSL_KEY"),
                "sslrootcert": os.getenv("DB_SSL_ROOT_CERT"),
            }
            
            # Remove None values
            connect_args = {k: v for k, v in connect_args.items() if v is not None}
            
            return create_engine(
                database_url,
                connect_args=connect_args,
                echo=False,  # Don't log SQL in production
                pool_pre_ping=True,  # Verify connections
                pool_recycle=3600,  # Recycle connections hourly
            )
        
        # MySQL with SSL
        elif database_url.startswith("mysql"):
            connect_args = {
                "ssl_disabled": False,
                "ssl_ca": os.getenv("DB_SSL_CA"),
                "ssl_cert": os.getenv("DB_SSL_CERT"),
                "ssl_key": os.getenv("DB_SSL_KEY"),
            }
            
            connect_args = {k: v for k, v in connect_args.items() if v is not None}
            
            return create_engine(
                database_url,
                connect_args=connect_args,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        
        # SQLite with security considerations
        elif database_url.startswith("sqlite"):
            # Ensure database file has proper permissions
            db_path = database_url.replace("sqlite:///", "")
            if os.path.exists(db_path):
                os.chmod(db_path, 0o600)  # Owner read/write only
                
            return create_engine(
                database_url,
                echo=False,
                pool_pre_ping=True,
            )
        
        else:
            raise ValueError(f"Unsupported database type: {database_url}")
```

### Data Encryption

```python
# security/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    def __init__(self, password: Optional[str] = None):
        self.password = password or os.getenv("SWING_ENCRYPTION_KEY")
        if not self.password:
            raise ValueError("Encryption password required")
        
        self.fernet = self._create_fernet_key()
    
    def _create_fernet_key(self) -> Fernet:
        """Create Fernet key from password."""
        password_bytes = self.password.encode()
        salt = b'swing_agent_salt_v1'  # In production, use random salt per database
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive string data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive string data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

# Usage in models_db.py for sensitive fields
class Signal(Base):
    # ... other fields ...
    
    # Encrypted fields for sensitive data
    _encrypted_notes = Column(String)  # Store encrypted
    
    @hybrid_property
    def notes(self) -> Optional[str]:
        if self._encrypted_notes:
            encryptor = DataEncryption()
            return encryptor.decrypt_sensitive_data(self._encrypted_notes)
        return None
    
    @notes.setter
    def notes(self, value: Optional[str]):
        if value:
            encryptor = DataEncryption()
            self._encrypted_notes = encryptor.encrypt_sensitive_data(value)
        else:
            self._encrypted_notes = None
```

## Input Validation and Sanitization

### Enhanced Validation

```python
# security/validation.py
import re
from typing import Any, List, Dict
from pydantic import BaseModel, validator, Field
import pandas as pd

class SecurityValidator:
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate stock symbol format."""
        # Allow alphanumeric, dots, dashes for international symbols
        pattern = r'^[A-Z0-9.-]{1,12}$'
        if not re.match(pattern, symbol.upper()):
            raise ValueError(f"Invalid symbol format: {symbol}")
        return symbol.upper()
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> str:
        """Validate timeframe against allowed values."""
        allowed = ["15m", "30m", "1h", "1d"]
        if timeframe not in allowed:
            raise ValueError(f"Invalid timeframe: {timeframe}. Allowed: {allowed}")
        return timeframe
    
    @staticmethod
    def validate_lookback_days(days: int) -> int:
        """Validate lookback days within reasonable limits."""
        if not 1 <= days <= 365:
            raise ValueError(f"Lookback days must be 1-365, got: {days}")
        return days
    
    @staticmethod
    def sanitize_llm_input(text: str) -> str:
        """Sanitize text input for LLM to prevent injection."""
        # Remove potential prompt injection patterns
        dangerous_patterns = [
            r'ignore\s+previous\s+instructions',
            r'system\s*:',
            r'assistant\s*:',
            r'user\s*:',
            r'<\|.*?\|>',  # Special tokens
        ]
        
        sanitized = text
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Limit length to prevent excessively long inputs
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000]
        
        return sanitized.strip()

# Enhanced data models with validation
class SecureTradeSignal(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=12)
    timeframe: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return SecurityValidator.validate_symbol(v)
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        return SecurityValidator.validate_timeframe(v)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        # Additional business logic validation
        if v > 0.95:
            # Log unusually high confidence for review
            import logging
            logging.getLogger("swing_agent.security").info(
                f"High confidence signal: {v:.3f}"
            )
        return v
```

### SQL Injection Prevention

```python
# security/sql_security.py
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List, Any

class SecureQueryBuilder:
    @staticmethod
    def safe_symbol_query(session: Session, symbols: List[str]) -> List[Any]:
        """Execute symbol query with parameterization."""
        # Validate all symbols first
        validated_symbols = [
            SecurityValidator.validate_symbol(symbol) 
            for symbol in symbols
        ]
        
        # Use parameterized query
        query = text("""
            SELECT * FROM signals 
            WHERE symbol = ANY(:symbol_list)
            ORDER BY created_at_utc DESC
        """)
        
        return session.execute(
            query, 
            {"symbol_list": validated_symbols}
        ).fetchall()
    
    @staticmethod
    def safe_date_range_query(session: Session, start_date: str, end_date: str) -> List[Any]:
        """Execute date range query safely."""
        # Validate date formats
        import datetime
        try:
            datetime.datetime.fromisoformat(start_date)
            datetime.datetime.fromisoformat(end_date)
        except ValueError:
            raise ValueError("Invalid date format. Use ISO format: YYYY-MM-DD")
        
        query = text("""
            SELECT * FROM signals 
            WHERE asof >= :start_date 
            AND asof <= :end_date
            ORDER BY asof DESC
        """)
        
        return session.execute(
            query,
            {"start_date": start_date, "end_date": end_date}
        ).fetchall()
```

## Network Security

### TLS Configuration

```python
# security/network.py
import ssl
import requests
from urllib3.util.ssl_ import create_urllib3_context

class SecureHTTPClient:
    def __init__(self):
        self.session = requests.Session()
        self._configure_tls()
    
    def _configure_tls(self):
        """Configure secure TLS settings."""
        # Create secure SSL context
        context = create_urllib3_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        # Configure adapter
        adapter = requests.adapters.HTTPAdapter()
        adapter.init_poolmanager(ssl_context=context)
        
        self.session.mount('https://', adapter)
        
        # Set timeouts
        self.session.timeout = (10, 30)  # Connect, read timeouts
    
    def secure_get(self, url: str, **kwargs) -> requests.Response:
        """Make secure GET request."""
        # Validate URL
        if not url.startswith('https://'):
            raise ValueError("Only HTTPS URLs allowed")
        
        return self.session.get(url, **kwargs)

# Usage in data fetching
class SecureDataProvider:
    def __init__(self):
        self.http_client = SecureHTTPClient()
    
    def fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch market data with secure HTTP client."""
        # Use yfinance with secure session
        import yfinance as yf
        
        # Validate symbol first
        symbol = SecurityValidator.validate_symbol(symbol)
        
        # Configure yfinance to use secure session
        ticker = yf.Ticker(symbol, session=self.http_client.session)
        return ticker.history(period="30d", interval="1h")
```

## Logging and Audit Trails

### Secure Logging

```python
# security/audit_logging.py
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

class AuditLogger:
    def __init__(self, log_file: str = "logs/audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure secure logging
        self.logger = logging.getLogger("swing_agent.audit")
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Set secure permissions on log file
        self.log_file.chmod(0o600)
    
    def log_signal_generation(self, user_id: str, symbol: str, 
                            signal_data: Dict[str, Any]):
        """Log signal generation event."""
        audit_event = {
            "event_type": "signal_generated",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "symbol": symbol,
            "trend_label": signal_data.get("trend", {}).get("label"),
            "confidence": signal_data.get("confidence"),
            "entry_side": signal_data.get("entry", {}).get("side"),
            "ip_address": self._get_client_ip(),
        }
        
        self.logger.info(json.dumps(audit_event))
    
    def log_database_access(self, user_id: str, action: str, 
                          table: str, record_id: Optional[str] = None):
        """Log database access events."""
        audit_event = {
            "event_type": "database_access",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,  # CREATE, READ, UPDATE, DELETE
            "table": table,
            "record_id": record_id,
            "ip_address": self._get_client_ip(),
        }
        
        self.logger.info(json.dumps(audit_event))
    
    def log_api_key_usage(self, service: str, operation: str, 
                         success: bool, tokens_used: Optional[int] = None):
        """Log API key usage for billing and security monitoring."""
        audit_event = {
            "event_type": "api_key_usage",
            "timestamp": datetime.utcnow().isoformat(),
            "service": service,
            "operation": operation,
            "success": success,
            "tokens_used": tokens_used,
        }
        
        self.logger.info(json.dumps(audit_event))
    
    def log_security_event(self, event_type: str, severity: str, 
                          details: Dict[str, Any]):
        """Log security-related events."""
        audit_event = {
            "event_type": "security_event",
            "timestamp": datetime.utcnow().isoformat(),
            "security_event_type": event_type,
            "severity": severity,
            "details": details,
            "ip_address": self._get_client_ip(),
        }
        
        self.logger.warning(json.dumps(audit_event))
    
    def _get_client_ip(self) -> Optional[str]:
        """Get client IP address if available."""
        # Implementation depends on deployment context
        # For web apps, use request.remote_addr
        # For CLI tools, this might not be relevant
        return None

# Secure logging configuration
def configure_secure_logging():
    """Configure logging with security best practices."""
    
    # Create secure log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_dir.chmod(0o700)  # Owner access only
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/swing_agent.log', mode='a'),
            logging.StreamHandler()  # Console output for development
        ]
    )
    
    # Set secure permissions on log files
    for log_file in log_dir.glob("*.log"):
        log_file.chmod(0o600)
    
    # Filter sensitive information from logs
    class SensitiveDataFilter(logging.Filter):
        def filter(self, record):
            # Mask API keys in log messages
            if hasattr(record, 'msg'):
                record.msg = re.sub(
                    r'sk-[A-Za-z0-9]{48}', 
                    'sk-****', 
                    str(record.msg)
                )
            return True
    
    # Add filter to all handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(SensitiveDataFilter())
```

## Production Deployment Security

### Environment Security

```python
# security/environment.py
import os
from typing import Dict, List
from pathlib import Path

class EnvironmentSecurity:
    REQUIRED_PROD_VARS = [
        "SWING_DATABASE_URL",
        "OPENAI_API_KEY", 
        "SWING_ENCRYPTION_KEY",
        "SWING_LOG_LEVEL",
    ]
    
    SENSITIVE_VARS = [
        "OPENAI_API_KEY",
        "SWING_ENCRYPTION_KEY", 
        "DB_PASSWORD",
        "DATABASE_URL",
    ]
    
    @classmethod
    def validate_production_environment(cls) -> Dict[str, List[str]]:
        """Validate production environment configuration."""
        issues = {
            "missing_vars": [],
            "weak_config": [],
            "security_warnings": []
        }
        
        # Check required variables
        for var in cls.REQUIRED_PROD_VARS:
            if not os.getenv(var):
                issues["missing_vars"].append(var)
        
        # Check for weak configurations
        if os.getenv("SWING_LOG_LEVEL") == "DEBUG":
            issues["weak_config"].append("DEBUG logging enabled in production")
        
        if os.getenv("SWING_DATABASE_URL", "").startswith("sqlite://"):
            issues["security_warnings"].append("SQLite not recommended for production")
        
        # Check file permissions
        sensitive_files = [
            ".env",
            "data/swing_agent.sqlite",
            "logs/audit.log"
        ]
        
        for file_path in sensitive_files:
            path = Path(file_path)
            if path.exists():
                stat = path.stat()
                if stat.st_mode & 0o077:  # Group/other permissions
                    issues["security_warnings"].append(
                        f"File {file_path} has overly permissive permissions"
                    )
        
        return issues
    
    @classmethod
    def secure_environment_setup(cls):
        """Set up secure environment configuration."""
        
        # Set secure file creation mask
        os.umask(0o077)
        
        # Create secure directories
        secure_dirs = ["data", "logs", "backups"]
        for dir_name in secure_dirs:
            path = Path(dir_name)
            path.mkdir(exist_ok=True)
            path.chmod(0o700)  # Owner only
        
        # Validate environment
        issues = cls.validate_production_environment()
        
        if any(issues.values()):
            import logging
            logger = logging.getLogger("swing_agent.security")
            
            for category, problems in issues.items():
                for problem in problems:
                    logger.warning(f"Environment {category}: {problem}")
    
    @classmethod
    def sanitize_environment_for_logging(cls) -> Dict[str, str]:
        """Get environment variables safe for logging."""
        env_vars = {}
        
        for key, value in os.environ.items():
            if key.startswith("SWING_"):
                if any(sensitive in key for sensitive in cls.SENSITIVE_VARS):
                    env_vars[key] = "***MASKED***"
                else:
                    env_vars[key] = value
        
        return env_vars

# Production deployment checker
def production_security_check():
    """Run comprehensive security check for production deployment."""
    print("SwingAgent Production Security Check")
    print("=" * 40)
    
    # Environment validation
    issues = EnvironmentSecurity.validate_production_environment()
    
    if not any(issues.values()):
        print("✅ Environment configuration passed security check")
    else:
        print("⚠️  Security issues found:")
        for category, problems in issues.items():
            if problems:
                print(f"\n{category.upper()}:")
                for problem in problems:
                    print(f"  - {problem}")
    
    # File permission check
    print("\n🔒 File Permission Check:")
    sensitive_files = [".env", "data/", "logs/"]
    
    for file_path in sensitive_files:
        path = Path(file_path)
        if path.exists():
            stat = path.stat()
            perms = oct(stat.st_mode)[-3:]
            
            if file_path == "data/" or file_path == "logs/":
                expected = "700"
            else:
                expected = "600"
            
            if perms == expected:
                print(f"  ✅ {file_path}: {perms}")
            else:
                print(f"  ⚠️  {file_path}: {perms} (recommended: {expected})")
    
    # Database connection security
    db_url = os.getenv("SWING_DATABASE_URL", "")
    print(f"\n🗄️  Database Security:")
    
    if db_url.startswith("postgresql://") and "sslmode=require" in db_url:
        print("  ✅ PostgreSQL with SSL required")
    elif db_url.startswith("sqlite://"):
        print("  ⚠️  SQLite - ensure file permissions are secure")
    else:
        print("  ⚠️  Database connection security unclear")
    
    print("\n" + "=" * 40)
    print("Security check complete. Address any warnings before production deployment.")

if __name__ == "__main__":
    production_security_check()
```

## Security Checklist

### Pre-Production Checklist

- [ ] **API Keys**
  - [ ] All API keys stored in environment variables
  - [ ] API key validation implemented
  - [ ] Key rotation policy documented
  - [ ] Keys masked in all logs

- [ ] **Database Security**
  - [ ] SSL/TLS enabled for database connections
  - [ ] Database credentials not hardcoded
  - [ ] Sensitive data encrypted at rest
  - [ ] Database backups encrypted

- [ ] **Input Validation**
  - [ ] All user inputs validated and sanitized
  - [ ] SQL injection protection verified
  - [ ] LLM prompt injection protection implemented
  - [ ] File upload restrictions (if applicable)

- [ ] **Network Security**
  - [ ] HTTPS required for all external communications
  - [ ] TLS 1.2+ enforced
  - [ ] Secure ciphers configured
  - [ ] Request timeouts configured

- [ ] **Logging and Monitoring**
  - [ ] Audit logging implemented
  - [ ] Log files have secure permissions
  - [ ] Sensitive data filtered from logs
  - [ ] Security events monitored

- [ ] **Environment**
  - [ ] Production environment variables validated
  - [ ] File permissions secured (600/700)
  - [ ] Debug logging disabled
  - [ ] Error messages don't expose sensitive data

### Regular Security Maintenance

**Monthly:**
- Review audit logs for suspicious activity
- Check for outdated dependencies with security vulnerabilities
- Validate API key rotation compliance

**Quarterly:**
- Review and update security policies
- Penetration testing of public interfaces
- Security training for development team

**Annually:**
- Comprehensive security audit
- Disaster recovery testing
- Security policy review and updates

This security guide provides a comprehensive framework for securing SwingAgent in production environments while maintaining functionality and ease of development.