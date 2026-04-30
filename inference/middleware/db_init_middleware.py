"""
Middleware for database initialization.

This middleware ensures the database is initialized before handling any request.
"""

from fastapi import Request

from db import storage
from config import (
    DB_CONNECTION_STRING,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="db_init_middleware")


async def db_init_middleware(request: Request, call_next):
    """
    Middleware that ensures database connection is initialized.
    """
    if not storage.initialized:
        logger.info("Database not initialized, initializing from middleware...")

        connection_string = DB_CONNECTION_STRING
        # Check if connection_string is valid
        if not connection_string:
            logger.warning(
                "DB_CONNECTION_STRING not found in environment, building from components..."
            )
            if all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
                connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
                logger.info(
                    f"Built connection string: postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}"
                )
            else:
                logger.error(
                    "Cannot build connection string, missing required environment variables"
                )
                for var, name in [
                    (DB_HOST, "DB_HOST"),
                    (DB_PORT, "DB_PORT"),
                    (DB_NAME, "DB_NAME"),
                    (DB_USER, "DB_USER"),
                    (DB_PASSWORD, "DB_PASSWORD"),
                ]:
                    logger.error(f"  {name}: {'✓' if var else '✗'}")

        # Initialize database if we have a connection string
        if connection_string:
            try:
                await storage.initialize(connection_string)

                if storage.initialized:
                    logger.info("Database initialized successfully from middleware")
                else:
                    logger.error("Database initialization failed from middleware")
            except Exception as e:
                logger.error(f"Error initializing database from middleware: {e}")

    # Continue with request handling
    response = await call_next(request)
    return response
