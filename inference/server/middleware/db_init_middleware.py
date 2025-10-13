"""
Middleware for database initialization.

This middleware ensures the database is initialized before handling any request.
"""

from fastapi import Request
import os

from db import storage
from server.config import DB_CONNECTION_STRING
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
            db_host = os.environ.get("DB_HOST")
            db_port = os.environ.get("DB_PORT")
            db_name = os.environ.get("DB_NAME")
            db_user = os.environ.get("DB_USER")
            db_password = os.environ.get("DB_PASSWORD")

            if all([db_host, db_port, db_name, db_user, db_password]):
                connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                logger.info(
                    f"Built connection string: postgresql://{db_user}:***@{db_host}:{db_port}/{db_name}"
                )
            else:
                logger.error(
                    "Cannot build connection string, missing required environment variables"
                )
                for var, name in [
                    (db_host, "DB_HOST"),
                    (db_port, "DB_PORT"),
                    (db_name, "DB_NAME"),
                    (db_user, "DB_USER"),
                    (db_password, "DB_PASSWORD"),
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
