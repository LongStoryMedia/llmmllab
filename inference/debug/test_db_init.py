#!/usr/bin/env python3
"""
Test script to verify the new database table structure.
"""

import asyncio
import sys
from db import storage
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_db_init")

async def test_database_initialization():
    """Test that the database initializes without foreign key constraint errors."""
    try:
        # Build connection string from environment variables (like other debug scripts)
        import os
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_user = os.getenv("DB_USER", "lsm")
        db_password = os.getenv("DB_PASSWORD", "")
        db_name = os.getenv("DB_NAME", "llmmll")
        db_sslmode = os.getenv("DB_SSLMODE", "disable")

        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_sslmode}"
        
        logger.info("Testing database initialization...")
        await storage.initialize(connection_string)
        
        logger.info("✅ Database initialization completed successfully")
        
        # Test that the new storage services are available
        if storage.thought and storage.analysis and storage.tool_call:
            logger.info("✅ All new storage services are available")
        else:
            logger.error("❌ Some storage services are missing")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False
    finally:
        if storage.pool:
            await storage.close()

if __name__ == "__main__":
    success = asyncio.run(test_database_initialization())
    sys.exit(0 if success else 1)