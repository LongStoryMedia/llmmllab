#!/usr/bin/env python3
"""
Test script to verify checkpoint table initialization works correctly
after removing redundant table creation steps.
"""

import asyncio
import os
from db import Storage
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_checkpoint_init")

async def test_checkpoint_initialization():
    """Test that checkpoint storage can be initialized without errors."""
    try:
        # Get database connection string from environment
        db_host = os.getenv("DB_HOST", "psql.psql.svc.cluster.local")
        db_port = os.getenv("DB_PORT", "5432") 
        db_name = os.getenv("DB_NAME", "llmmll")
        db_user = os.getenv("DB_USER", "lsm")
        db_password = os.getenv("DB_PASSWORD", "password")
        
        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        logger.info("🔄 Testing checkpoint initialization...")
        logger.info(f"Using connection: {db_host}:{db_port}/{db_name}")
        
        # Initialize storage (this will create checkpoint tables)
        storage = Storage()
        await storage.initialize(connection_string)
        
        logger.info("✅ Checkpoint initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Checkpoint initialization failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_checkpoint_initialization())
    exit(0 if success else 1)