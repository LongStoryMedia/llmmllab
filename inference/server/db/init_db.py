"""
Database initialization utilities.
"""

import asyncio
from typing import Any

from .queries import get_query
import server.config

logger = server.config.logger  # Use the logger from config


async def initialize_database(connection_pool: Any) -> bool:
    """
    Initialize the database schema.

    Args:
        connection_pool: The database connection pool.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Get a connection from the pool
        connection = await connection_pool.acquire()

        try:
            # Create the tables in the correct order
            await connection.execute(get_query("schema.create_conversations_table"))
            await connection.execute(get_query("schema.create_messages_table"))
            await connection.execute(get_query("schema.create_summaries_table"))

            # For vector support, we need to check if the pgvector extension is available
            try:
                # Try to create the pgvector extension
                await connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                # Create the memories table (which depends on vector extension)
                await connection.execute(get_query("schema.create_memories_table"))
            except (asyncio.TimeoutError, RuntimeError) as e:
                logger.warning(f"Could not create pgvector extension: {str(e)}")
                logger.warning(
                    "Memory search functionality will be limited. Install pgvector extension for full functionality."
                )

            # Return success
            logger.info("Database schema initialized successfully.")
            return True

        except (asyncio.TimeoutError, RuntimeError, ValueError) as e:
            logger.error(f"Error initializing database schema: {str(e)}")
            return False
        finally:
            # Release the connection back to the pool
            await connection_pool.release(connection)

    except (asyncio.TimeoutError, ConnectionError) as e:
        logger.error(f"Error acquiring database connection: {str(e)}")
        return False
