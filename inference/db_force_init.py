#!/usr/bin/env python
"""
Force database initialization script.
Run this script to initialize the database directly.
"""

import asyncio
import os
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db-force-init")


async def main():
    """Force database initialization"""
    logger.info("Starting database force initialization...")

    # Import after setting up logging
    from server.db import storage
    from server.db.init_db import initialize_database

    # Get connection string
    connection_string = os.environ.get("DB_CONNECTION_STRING")

    # If not set directly, build it from components
    if not connection_string:
        logger.info("DB_CONNECTION_STRING not set, building from components...")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "llmmll")
        db_user = os.environ.get("DB_USER", "postgres")
        db_password = os.environ.get("DB_PASSWORD", "")

        connection_string = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
        logger.info(
            f"Built connection string: postgresql://{db_user}:***@{db_host}:{db_port}/{db_name}"
        )

    # Initialize storage
    logger.info("Initializing database connection...")
    try:
        await storage.initialize(connection_string)

        if storage.initialized and storage.pool:
            logger.info("Database connection established, initializing schema...")
            schema_initialized = await initialize_database(storage.pool)
            if schema_initialized:
                logger.info("Database schema initialized successfully")
            else:
                logger.warning("Database schema initialization skipped or failed")
        else:
            logger.error("Failed to initialize database connection")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

    logger.info("Database force initialization complete")


if __name__ == "__main__":
    asyncio.run(main())
