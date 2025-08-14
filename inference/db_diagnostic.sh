#!/bin/bash
# Simple diagnostic script to check database status in Kubernetes pod

# Make sure we're in the right directory
cd /app || cd /workspace || cd /

# Enable error reporting
set -e

echo "Starting database diagnostics..."
echo

# Check if DB_CONNECTION_STRING is set
echo "Checking environment variables..."
if [ -z "$DB_CONNECTION_STRING" ]; then
    echo "ERROR: DB_CONNECTION_STRING is not set!"
else
    echo "DB_CONNECTION_STRING is set (value hidden)"
fi
echo

# Check if PostgreSQL is accessible
echo "Checking PostgreSQL connection..."
if command -v pg_isready &> /dev/null; then
    # Extract host and port from DB_CONNECTION_STRING
    if [[ "$DB_CONNECTION_STRING" =~ postgresql://[^@]+@([^:]+):([0-9]+)/ ]]; then
        HOST="${BASH_REMATCH[1]}"
        PORT="${BASH_REMATCH[2]}"
        echo "Trying to connect to PostgreSQL at $HOST:$PORT..."
        pg_isready -h "$HOST" -p "$PORT"
    else
        echo "Could not parse host/port from DB_CONNECTION_STRING"
    fi
else
    echo "pg_isready not available, skipping direct connection test"
fi
echo

# Check SQL files
echo "Checking SQL files..."
SQL_DIR="/app/server/db/sql"
if [ -d "$SQL_DIR" ]; then
    echo "SQL directory exists at $SQL_DIR"
    echo "SQL subdirectories:"
    find "$SQL_DIR" -type d | sort
    echo
    echo "User SQL files:"
    find "$SQL_DIR/user" -name "*.sql" -type f 2>/dev/null | sort
else
    echo "ERROR: SQL directory not found at $SQL_DIR"
    # Try to find it elsewhere
    FOUND_SQL=$(find / -name "sql" -type d -path "*/server/db/*" 2>/dev/null || echo "not found")
    echo "Searched for SQL directory, found: $FOUND_SQL"
fi
echo

# Create a simple Python diagnostic script
echo "Running Python diagnostic..."
cat > /tmp/db_check.py << 'EOL'
import os
import sys
import asyncio

async def check_db():
    print("Python diagnostic starting...")

    # Try to import required modules
    try:
        print("Importing modules...")
        import asyncpg
        from server.db import storage
        from server.db.queries import get_query, get_loader
        from server.config import DB_CONNECTION_STRING
        print("All modules imported successfully")
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)

    # Check DB connection string
    print(f"DB_CONNECTION_STRING exists: {bool(DB_CONNECTION_STRING)}")

    # Check loader
    try:
        print("Checking SQL loader...")
        loader = get_loader()
        print(f"SQL queries loaded: {len(loader.queries)}")
        critical_queries = [
            "user.create_users_table",
            "user.get_all_users",
            "user.get_config"
        ]
        for key in critical_queries:
            print(f"Query '{key}' exists: {key in loader.queries}")
    except Exception as e:
        print(f"Error checking loader: {e}")

    # Check storage initialization
    print(f"Storage initialized: {storage.initialized}")
    print(f"Storage pool exists: {storage.pool is not None}")
    print(f"Storage user_config exists: {storage.user_config is not None}")

    # Try to initialize if needed
    if not storage.initialized and DB_CONNECTION_STRING:
        try:
            print("Attempting to initialize storage...")
            await storage.initialize(DB_CONNECTION_STRING)
            print(f"Storage initialized after attempt: {storage.initialized}")
        except Exception as e:
            print(f"Error initializing storage: {e}")

asyncio.run(check_db())
EOL

# Execute the Python script
echo "Executing Python diagnostic script..."
python /tmp/db_check.py || echo "Python script failed"
echo

echo "Diagnostics complete"
