"""
Debugging script to check database initialization state.
"""

import asyncio
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("db_debug")


async def check_database_init():
    """Check database initialization status and configuration."""
    try:
        # Import modules from server
        from server.db import storage
        from server.config import DB_CONNECTION_STRING
        from server.db.init_db import initialize_database

        # Log configuration
        logger.info(f"DB_CONNECTION_STRING exists: {bool(DB_CONNECTION_STRING)}")

        if not DB_CONNECTION_STRING:
            logger.error("DB_CONNECTION_STRING is not set or empty")
            return

        # Check if the connection is already initialized
        logger.info(f"Storage initialized flag: {storage.initialized}")
        logger.info(f"Storage pool exists: {bool(storage.pool)}")
        logger.info(f"Storage user_config exists: {bool(storage.user_config)}")

        # Try to initialize the connection
        if not storage.initialized:
            logger.info("Attempting to initialize storage connection...")
            try:
                await storage.initialize(DB_CONNECTION_STRING)
                logger.info(f"Storage initialization result: {storage.initialized}")
                logger.info(f"Storage pool after init: {bool(storage.pool)}")
                logger.info(
                    f"Storage user_config after init: {bool(storage.user_config)}"
                )
            except Exception as e:
                logger.error(f"Error initializing storage: {e}")

        # Try to initialize the database schema
        if storage.pool:
            logger.info("Initializing database schema...")
            try:
                schema_initialized = await initialize_database(storage.pool)
                logger.info(f"Schema initialization result: {schema_initialized}")
            except Exception as e:
                logger.error(f"Error initializing schema: {e}")

            # Test user_config
            if storage.initialized and storage.user_config:
                try:
                    logger.info("Testing user_config.get_all_users()")
                    users = await storage.user_config.get_all_users()
                    logger.info(f"Found {len(users)} users")
                    for user in users:
                        logger.info(f"User: {user}")
                except Exception as e:
                    logger.error(f"Error fetching users: {e}")
            else:
                logger.error("Cannot test user_config because it's not initialized")

            # Check queries
            try:
                from server.db.queries import get_loader

                loader = get_loader()
                logger.info(f"Total queries loaded: {len(loader.queries)}")
                logger.info("Checking critical queries:")
                critical_queries = [
                    "user.create_users_table",
                    "user.get_all_users",
                    "conversation.create_conversations_table",
                    "message.create_messages_table",
                    "summary.create_summaries_table",
                ]
                for query_key in critical_queries:
                    exists = query_key in loader.queries
                    logger.info(f"  - {query_key}: {'✅' if exists else '❌'}")
                    if not exists:
                        logger.error(f"Critical query missing: {query_key}")
            except Exception as e:
                logger.error(f"Error checking queries: {e}")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(check_database_init())
