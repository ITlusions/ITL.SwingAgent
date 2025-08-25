#!/usr/bin/env python3
"""
Database information and configuration utility for SwingAgent.
"""
import argparse
import json
from swing_agent.database import get_database_info, get_database_config


def show_database_info():
    """Show current database configuration information."""
    info = get_database_info()
    
    print("SwingAgent Database Configuration")
    print("=" * 40)
    print(f"Database Type: {info['type']}")
    print(f"Is External: {info['is_external']}")
    print(f"Connection: {info['url_masked']}")
    
    if info['host']:
        print(f"Host: {info['host']}")
        print(f"Port: {info['port']}")
        print(f"Database: {info['database']}")
    
    print()
    print("Configuration Details:")
    print("-" * 20)
    
    # Test connection
    try:
        config = get_database_config()
        engine = config.engine
        with engine.connect() as conn:
            print("✓ Connection: Successful")
            
            # Check if tables exist
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            if 'signals' in tables:
                print("✓ Signals table: Exists")
            else:
                print("✗ Signals table: Missing")
            
            if 'vec_store' in tables:
                print("✓ Vector store table: Exists")
            else:
                print("✗ Vector store table: Missing")
                
    except Exception as e:
        print(f"✗ Connection: Failed - {e}")


def test_connection():
    """Test database connection."""
    try:
        config = get_database_config()
        engine = config.engine
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1")).fetchone()
            print("✓ Database connection successful")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def init_tables():
    """Initialize database tables."""
    from swing_agent.database import init_database
    
    try:
        init_database()
        print("✓ Database tables initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize tables: {e}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="SwingAgent database utility")
    parser.add_argument("--info", action="store_true", 
                       help="Show database configuration information")
    parser.add_argument("--test", action="store_true",
                       help="Test database connection")
    parser.add_argument("--init", action="store_true",
                       help="Initialize database tables")
    parser.add_argument("--json", action="store_true",
                       help="Output information as JSON")
    
    args = parser.parse_args()
    
    if args.info:
        if args.json:
            info = get_database_info()
            print(json.dumps(info, indent=2))
        else:
            show_database_info()
    elif args.test:
        test_connection()
    elif args.init:
        init_tables()
    else:
        # Default: show info
        show_database_info()


if __name__ == "__main__":
    main()