#!/usr/bin/env python3
"""
CNPG Connection Test Utility

This script helps verify CNPG (CloudNativePG) configuration and connectivity.
Run it with environment variables set to test your CNPG setup.

Examples:
    # Test CNPG configuration
    export SWING_DB_TYPE=cnpg
    export CNPG_CLUSTER_NAME=swing-postgres
    export CNPG_NAMESPACE=default
    export SWING_DB_NAME=swing_agent
    export SWING_DB_USER=swing_user
    export SWING_DB_PASSWORD=your_password
    python scripts/test_cnpg.py

    # Test with custom SSL settings
    export CNPG_SSL_MODE=require
    export CNPG_SSL_CERT=/ssl/client.crt
    python scripts/test_cnpg.py
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    # Import database module directly to avoid package-level dependencies
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "database", 
        Path(__file__).parent.parent / "src" / "swing_agent" / "database.py"
    )
    database = importlib.util.module_from_spec(spec)
    sys.modules['database'] = database
    spec.loader.exec_module(database)
    
    create_cnpg_url = database.create_cnpg_url
    from_env_config = database.from_env_config
    DatabaseConfig = database.DatabaseConfig
    
except ImportError as e:
    print(f"❌ Failed to import database modules: {e}")
    print("Make sure SQLAlchemy is installed: pip install sqlalchemy")
    sys.exit(1)


def test_cnpg_url():
    """Test CNPG URL creation from environment variables."""
    print("🔗 Testing CNPG URL creation...")
    
    url = create_cnpg_url()
    if url is None:
        print("❌ Failed to create CNPG URL")
        print("Required environment variables:")
        print("  - CNPG_CLUSTER_NAME (or SWING_CNPG_CLUSTER)")
        print("  - SWING_DB_NAME")
        print("  - SWING_DB_USER") 
        print("  - SWING_DB_PASSWORD")
        return False
    
    print(f"✅ CNPG URL created successfully")
    print(f"   URL (masked): {mask_password(url)}")
    return True


def test_database_config():
    """Test database configuration and type detection."""
    print("\n🔧 Testing database configuration...")
    
    try:
        # Test environment-based configuration
        db_url = from_env_config()
        if not db_url:
            print("❌ No database URL found from environment")
            return False
        
        print(f"✅ Database URL created from environment")
        print(f"   URL (masked): {mask_password(db_url)}")
        
        # Test database config creation
        db_config = DatabaseConfig(db_url)
        info = db_config.get_database_info()
        
        print(f"✅ Database configuration loaded")
        print(f"   Type: {info['type']}")
        print(f"   Is CNPG: {info.get('is_cnpg', False)}")
        print(f"   Is External: {info.get('is_external', False)}")
        
        if info.get('is_cnpg'):
            print(f"   CNPG Cluster: {info.get('cnpg_cluster', 'Unknown')}")
            print(f"   Service Type: {info.get('cnpg_service', 'Unknown')}")
            print(f"   Namespace: {info.get('cnpg_namespace', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database configuration failed: {e}")
        return False


def test_connection():
    """Test actual database connection (requires running CNPG cluster)."""
    print("\n🔌 Testing database connection...")
    
    try:
        db_url = from_env_config()
        if not db_url:
            print("❌ No database URL available for connection test")
            return False
            
        db_config = DatabaseConfig(db_url)
        
        # Test engine creation
        engine = db_config.engine
        print("✅ Database engine created")
        
        # Test basic connection
        with engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            row = result.fetchone()
            if row and row[0] == 1:
                print("✅ Database connection successful")
            else:
                print("❌ Unexpected query result")
                return False
        
        # Test PostgreSQL-specific query
        with engine.connect() as conn:
            result = conn.execute("SELECT version(), current_database(), current_user")
            row = result.fetchone()
            print(f"   PostgreSQL version: {row[0][:50]}...")
            print(f"   Database: {row[1]}")
            print(f"   User: {row[2]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   This is expected if the CNPG cluster is not running or accessible")
        return False


def print_environment():
    """Print relevant environment variables."""
    print("📋 Environment Configuration:")
    
    env_vars = [
        'SWING_DB_TYPE',
        'CNPG_CLUSTER_NAME', 'SWING_CNPG_CLUSTER',
        'CNPG_NAMESPACE', 'SWING_CNPG_NAMESPACE',
        'CNPG_SERVICE_TYPE', 'SWING_CNPG_SERVICE',
        'SWING_DB_NAME', 'SWING_DB_USER', 'SWING_DB_PASSWORD',
        'CNPG_SSL_MODE', 'SWING_CNPG_SSL_MODE',
        'CNPG_SSL_CERT', 'CNPG_SSL_KEY', 'CNPG_SSL_CA',
        'CNPG_CONNECT_TIMEOUT', 'CNPG_APP_NAME',
        'SWING_DB_POOL_SIZE', 'SWING_DB_MAX_OVERFLOW'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'PASSWORD' in var:
                value = '*' * len(value)
            print(f"   {var}: {value}")


def mask_password(url):
    """Mask password in database URL for safe display."""
    if '://' in url:
        protocol, rest = url.split('://', 1)
        if '@' in rest:
            auth, rest = rest.split('@', 1)
            if ':' in auth:
                user, password = auth.split(':', 1)
                masked_auth = f"{user}:{'*' * len(password)}"
                return f"{protocol}://{masked_auth}@{rest}"
    return url


def main():
    """Run CNPG configuration tests."""
    print("🧪 CNPG Configuration Test Utility")
    print("=" * 50)
    
    # Print environment
    print_environment()
    print()
    
    # Check if CNPG is configured
    if os.getenv('SWING_DB_TYPE') != 'cnpg':
        print("ℹ️  SWING_DB_TYPE is not set to 'cnpg'")
        print("   Set SWING_DB_TYPE=cnpg to test CNPG configuration")
        print()
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_cnpg_url():
        tests_passed += 1
    
    if test_database_config():
        tests_passed += 1
    
    if test_connection():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! CNPG configuration is working correctly.")
    elif tests_passed >= 2:
        print("⚠️  Basic configuration works, but connection failed.")
        print("   Make sure your CNPG cluster is running and accessible.")
    else:
        print("❌ Configuration issues detected. Check your environment variables.")
    
    return 0 if tests_passed >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())