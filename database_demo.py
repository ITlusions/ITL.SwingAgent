#!/usr/bin/env python3
"""
Example script showing how to use the centralized database and migration.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from swing_agent.migrate import migrate_data
    from swing_agent.database import init_database
    print("✓ Successfully imported database modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Note: This would work with SQLAlchemy installed")


def show_migration_example():
    """Show example of how to migrate data."""
    print("\n--- Migration Example ---")
    print("To migrate from separate databases to centralized:")
    print("python -m swing_agent.migrate --data-dir data/")
    print("or programmatically:")
    print("""
from swing_agent.migrate import migrate_data
from pathlib import Path

# Migrate existing data
migrate_data(Path("data"))
""")


def show_usage_example():
    """Show example of centralized database usage."""
    print("\n--- Centralized Database Usage ---")
    print("Now all scripts use the same database file by default:")
    print("  - Signals: data/swing_agent.sqlite")
    print("  - Vectors: data/swing_agent.sqlite (same file!)")
    print()
    print("Script examples:")
    print("  python scripts/run_swing_agent.py --symbol AAPL")
    print("  python scripts/eval_signals.py")
    print("  python scripts/analyze_performance.py")
    print()
    print("All will use data/swing_agent.sqlite unless overridden.")


def main():
    parser = argparse.ArgumentParser(description="Database centralization demo")
    parser.add_argument("--migrate", action="store_true", help="Show migration example")
    parser.add_argument("--usage", action="store_true", help="Show usage example")
    args = parser.parse_args()
    
    if args.migrate:
        show_migration_example()
    elif args.usage:
        show_usage_example()
    else:
        print("Database Centralization Complete!")
        print("Use --migrate or --usage for examples")
        print("\nKey changes:")
        print("✓ Single SQLite file: data/swing_agent.sqlite")
        print("✓ SQLAlchemy ORM instead of raw sqlite3")
        print("✓ Centralized database configuration")
        print("✓ Migration utility for existing data")
        print("✓ Backward compatible function signatures")


if __name__ == "__main__":
    main()