"""
Database initialization utilities.
"""

from typing import Any, List, Tuple

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
    logger.info("=== Starting database schema initialization ===")
    try:
        # Get a connection from the pool
        logger.info("Acquiring database connection from pool")
        connection = await connection_pool.acquire()
        logger.info("Database connection acquired successfully")

        try:
            # Initialize extensions first
            try:
                # Create extensions
                await connection.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                await connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("Database extensions initialized successfully")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Could not create all extensions: {str(e)}")
                logger.warning(
                    "Some functionality may be limited without required extensions."
                )

            # Initialize tables in correct dependency order
            initialization_steps = [
                # Step 1: Create base tables
                (
                    "Creating users table",
                    [
                        ("user.create_users_table", []),
                        ("user.create_user_check_trigger", []),
                        ("user.user_delete_cascade", []),
                    ],
                ),
                # Step 2: Create model tables
                (
                    "Creating model tables",
                    [
                        ("modelprofile.create_model_profiles_table", []),
                        ("modelprofile.create_model_profiles_index", []),
                        ("modelprofile.add_circuit_breaker_column", []),
                        ("modelprofile.add_gpu_config_column", []),
                    ],
                ),
                # Step 3: Create conversation tables
                (
                    "Creating conversation tables",
                    [
                        ("conversation.create_conversations_table", []),
                        ("conversation.create_conversations_indexes", []),
                        (
                            "conversation.create_conversations_hypertable",
                            ["timescaledb"],
                        ),
                        ("conversation.create_cascade_delete_trigger", []),
                        ("conversation.create_conversation_update_trigger", []),
                        (
                            "conversation.conversations_compression_policy",
                            ["timescaledb"],
                        ),
                        (
                            "conversation.conversations_retention_policy",
                            ["timescaledb"],
                        ),
                        (
                            "conversation.enable_conversations_compression",
                            ["timescaledb"],
                        ),
                    ],
                ),
                # Step 4: Create message tables
                (
                    "Creating message tables",
                    [
                        ("message.create_messages_table", []),
                        ("message.create_message_content_table", []),
                        ("message.create_message_contents_hypertable", ["timescaledb"]),
                        (
                            "message.message_contents_compression_policy",
                            ["timescaledb"],
                        ),
                        ("message.message_contents_retention_policy", ["timescaledb"]),
                        (
                            "message.enable_message_contents_compression",
                            ["timescaledb"],
                        ),
                    ],
                ),
                # Step 5: Create summary tables
                (
                    "Creating summary tables",
                    [
                        ("summary.create_summaries_table", []),
                        ("summary.create_summaries_hypertable", ["timescaledb"]),
                        ("summary.create_summaries_indexes", []),
                        ("summary.enable_summaries_compression", ["timescaledb"]),
                    ],
                ),
                # Step 6: Create search tables
                (
                    "Creating search tables",
                    [
                        ("search.create_search_topic_synthesis_table", []),
                        (
                            "search.create_search_topic_synthesis_hypertable",
                            ["timescaledb"],
                        ),
                    ],
                ),
                # Step 7: Create memory tables (requires vector extension)
                (
                    "Creating memory tables",
                    [
                        ("memory.init_memory_schema", ["vector", "timescaledb"]),
                        (
                            "memory.create_memory_indexes",
                            ["vector"],
                        ),  # Create indexes before enabling compression
                        ("memory.create_memory_cascade_delete_triggers", []),
                        (
                            "memory.enable_memories_compression",
                            ["timescaledb"],
                        ),  # Enable compression after indexes
                        (
                            "memory.memories_compression_policy",
                            ["timescaledb"],
                        ),  # Set policy after enabling compression
                        ("memory.memories_retention_policy", ["timescaledb"]),
                    ],
                ),
                # Step 8: Create image tables
                ("Creating image tables", [("images.create_images_schema", [])]),
                # Step 9: Create dynamic tools tables
                (
                    "Creating dynamic tools tables",
                    [
                        ("tool.create_tool_table", []),
                        ("tool.migrate_basetool_interface", []),
                        ("tool.create_embedding_index", ["vector"]),
                    ],
                ),
                # Step 10: Create research tables
                (
                    "Creating research tables",
                    [
                        ("research.create_research_tasks_table", []),
                        ("research.create_research_subtasks_table", []),
                    ],
                ),
            ]

            # Execute all initialization steps
            logger.info("Starting to execute all initialization steps...")
            step_count = len(initialization_steps)
            for idx, (step_description, queries) in enumerate(initialization_steps, 1):
                logger.info(f"Step {idx}/{step_count}: {step_description}")
                success = await execute_initialization_step(connection, queries)
                if success:
                    logger.info(f"✅ Step {idx}/{step_count} completed successfully")
                else:
                    logger.warning(f"⚠️ Step {idx}/{step_count} completed with warnings")

            # Return success
            logger.info("Database schema initialized successfully.")
            return True

        except (RuntimeError, ValueError) as e:
            logger.error(f"Error initializing database schema: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during schema initialization: {str(e)}")
            return False
        finally:
            # Release the connection back to the pool
            await connection_pool.release(connection)

    except (ConnectionError, ValueError, RuntimeError) as e:
        logger.error(f"Error acquiring database connection: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error acquiring database connection: {str(e)}")
        return False


async def execute_initialization_step(
    connection: Any, queries: List[Tuple[str, List[str]]]
) -> bool:
    """
    Execute a group of SQL initialization queries.

    Args:
        connection: The database connection.
        queries: List of tuples containing (query_key, required_extensions).

    Returns:
        bool: True if all queries executed successfully, False if any failed
    """
    all_successful = True
    extension_check_cache = {}  # Cache extension check results

    for query_key, required_extensions in queries:
        try:
            # Skip if missing required extensions
            skip_query = False
            if required_extensions:
                for ext in required_extensions:
                    # Use cached result if available
                    if ext in extension_check_cache:
                        is_installed = extension_check_cache[ext]
                    else:
                        # Check if extension is installed
                        try:
                            is_installed = await connection.fetchval(
                                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = $1)",
                                ext,
                            )
                            extension_check_cache[ext] = is_installed
                        except Exception:
                            logger.warning(
                                f"Could not check if extension {ext} is installed"
                            )
                            is_installed = False
                            extension_check_cache[ext] = False

                    if not is_installed:
                        logger.warning(
                            f"Skipping {query_key}, requires extension {ext}"
                        )
                        skip_query = True
                        break

                if skip_query:
                    continue

            # Try to get and execute the query
            try:
                query = get_query(query_key)
                await connection.execute(query)
                logger.info(f"Successfully executed {query_key}")
            except KeyError:
                logger.error(f"SQL query not found: {query_key}")
                all_successful = False
            except Exception as e:
                # Specific handling for common SQL errors
                error_msg = str(e)
                if "already exists" in error_msg:
                    logger.info(f"Object in {query_key} already exists, continuing")
                else:
                    logger.error(f"Unexpected error executing {query_key}: {error_msg}")
                    all_successful = False
                # Continue with other queries to ensure we create as much as possible
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Error executing {query_key}: {str(e)}")
            all_successful = False
            # Continue with other queries to ensure we create as much as possible
        except Exception as e:
            logger.error(f"Unexpected error executing {query_key}: {str(e)}")
            all_successful = False
            # Continue with other queries to ensure we create as much as possible

    return all_successful
