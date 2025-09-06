"""
Storage implementation for DynamicTool objects.
"""

from typing import List, Optional, Tuple
import asyncpg
import uuid
import json
import logging
from models.dynamic_tool import DynamicTool
from models.pagination import PaginationSchema
from server.db.db_utils import typed_pool
from utils.serialization import serialize_to_json
from .memory_storage import MemoryStorage

logger = logging.getLogger(__name__)


class DynamicToolStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def get_tool_by_id(
        self, tool_id: uuid.UUID, user_id: str
    ) -> Optional[DynamicTool]:
        """Get a dynamic tool by ID for a specific user"""
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("tool.get_tool_by_id"), tool_id, user_id
            )
            if not row:
                return None

            try:
                tool_data = dict(row)

                # Parse parameters if stored as JSON string
                if isinstance(tool_data.get("parameters"), str):
                    try:
                        tool_data["parameters"] = json.loads(tool_data["parameters"])
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse parameters JSON for tool {tool_id}: {e}"
                        )

                return DynamicTool(**tool_data)
            except Exception as e:
                logger.error(f"Error creating DynamicTool from database: {e}")
                return None

    async def list_all_tools(
        self, limit: int = 10, offset: int = 0
    ) -> Tuple[List[DynamicTool], PaginationSchema]:
        """
        List all dynamic tools in the system with pagination

        Args:
            limit: Maximum number of tools to return, defaults to 10
            offset: Number of tools to skip for pagination, defaults to 0

        Returns:
            Tuple containing list of tools and pagination metadata
        """
        async with self.typed_pool.acquire() as conn:
            # Get total count for pagination
            count_row = await conn.fetchrow(self.get_query("tool.count_all_tools"))
            total_count = count_row["total_count"] if count_row else 0

            # Fetch paginated results
            rows = await conn.fetch(
                self.get_query("tool.list_all_tools"), limit, offset
            )
            tools = []

            for row in rows:
                try:
                    tool_data = dict(row)

                    # Parse parameters if stored as JSON string
                    if isinstance(tool_data.get("parameters"), str):
                        try:
                            tool_data["parameters"] = json.loads(
                                tool_data["parameters"]
                            )
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse parameters JSON for tool: {e}"
                            )
                            continue

                    tools.append(DynamicTool(**tool_data))
                except Exception as e:
                    logger.error(f"Error creating DynamicTool from database: {e}")

            # Create pagination metadata
            pagination = PaginationSchema(
                total_count=total_count,
                limit=limit,
                offset=offset,
                has_more=offset + len(tools) < total_count,
            )

            return tools, pagination

    async def list_tools_by_user(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> Tuple[List[DynamicTool], PaginationSchema]:
        """
        List all dynamic tools for a specific user with pagination

        Args:
            user_id: The ID of the user whose tools to list
            limit: Maximum number of tools to return, defaults to 10
            offset: Number of tools to skip for pagination, defaults to 0

        Returns:
            Tuple containing list of tools and pagination metadata
        """
        async with self.typed_pool.acquire() as conn:
            # Get total count for pagination
            count_row = await conn.fetchrow(
                self.get_query("tool.count_tools_by_user"), user_id
            )
            total_count = count_row["total_count"] if count_row else 0

            # Fetch paginated results
            rows = await conn.fetch(
                self.get_query("tool.list_tools_by_user"), user_id, limit, offset
            )
            tools = []

            for row in rows:
                try:
                    tool_data = dict(row)

                    # Parse parameters if stored as JSON string
                    if isinstance(tool_data.get("parameters"), str):
                        try:
                            tool_data["parameters"] = json.loads(
                                tool_data["parameters"]
                            )
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse parameters JSON for tool: {e}"
                            )
                            continue

                    tools.append(DynamicTool(**tool_data))
                except Exception as e:
                    logger.error(f"Error creating DynamicTool from database: {e}")

            # Create pagination metadata
            pagination = PaginationSchema(
                total_count=total_count,
                limit=limit,
                offset=offset,
                has_more=offset + len(tools) < total_count,
            )

            return tools, pagination

    async def create_tool(self, tool: DynamicTool) -> DynamicTool:
        """Create a new dynamic tool"""
        # Serialize parameters to JSON if needed
        params_json = "{}"
        if tool.parameters:
            params_json = serialize_to_json(tool.parameters)

        # Convert embedding to database format if provided
        embedding = tool.embedding if tool.embedding is not None else None

        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("tool.create_tool"),
                tool.id,
                tool.user_id,
                tool.name,
                tool.description,
                tool.code,
                tool.function_name,
                embedding,
                params_json,
            )

            # Update the tool with current timestamps (which are set by the database)
            if row:
                tool_data = dict(row)
                tool.created_at = tool_data.get("created_at")
                tool.updated_at = tool_data.get("updated_at")

            return tool

    async def update_tool(self, tool: DynamicTool) -> Optional[DynamicTool]:
        """Update an existing dynamic tool"""
        # Serialize parameters to JSON if needed
        params_json = "{}"
        if tool.parameters:
            params_json = serialize_to_json(tool.parameters)

        # Convert embedding to database format if provided
        embedding = tool.embedding if tool.embedding is not None else None

        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("tool.update_tool"),
                tool.id,
                tool.user_id,
                tool.name,
                tool.description,
                tool.code,
                tool.function_name,
                embedding,
                params_json,
            )

            if not row:
                # Tool wasn't found or user doesn't have permission
                return None

            # Update the tool with current timestamp (which is set by the database)
            tool_data = dict(row)
            tool.updated_at = tool_data.get("updated_at")

            return tool

    async def delete_tool(self, tool_id: uuid.UUID, user_id: str) -> bool:
        """Delete a dynamic tool"""
        async with self.typed_pool.acquire() as conn:
            result = await conn.execute(
                self.get_query("tool.delete_tool"), tool_id, user_id
            )

            return "DELETE" in result

    async def search_tools_by_embedding(
        self,
        query_embedding: List[float],
        similarity_threshold: float = 0.7,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[DynamicTool], PaginationSchema]:
        """
        Search for dynamic tools based on embedding similarity

        Args:
            query_embedding: The query embedding vector to search with
            similarity_threshold: Minimum similarity score (0-1), defaults to 0.7
            limit: Maximum number of tools to return, defaults to 10
            offset: Number of tools to skip for pagination, defaults to 0

        Returns:
            Tuple containing list of tools and pagination metadata
        """
        vector_str = MemoryStorage.format_embedding_for_pgvector(query_embedding)

        async with self.typed_pool.acquire() as conn:
            # Get total count for pagination
            count_row = await conn.fetchrow(
                self.get_query("tool.count_tools_by_embedding"),
                vector_str,
                similarity_threshold,
            )
            total_count = count_row["total_count"] if count_row else 0

            # Fetch paginated results
            rows = await conn.fetch(
                self.get_query("tool.search_tools_by_embedding"),
                vector_str,
                similarity_threshold,
                limit,
                offset,
            )
            tools = []

            for row in rows:
                try:
                    tool_data = dict(row)

                    # Store the similarity score separately
                    similarity_score = tool_data.pop("similarity_score", None)

                    # Parse parameters if stored as JSON string
                    if isinstance(tool_data.get("parameters"), str):
                        try:
                            tool_data["parameters"] = json.loads(
                                tool_data["parameters"]
                            )
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse parameters JSON for tool: {e}"
                            )
                            continue

                    # Create the tool and add similarity score as an attribute
                    tool = DynamicTool(**tool_data)
                    setattr(tool, "similarity_score", similarity_score)
                    tools.append(tool)
                except Exception as e:
                    logger.error(f"Error creating DynamicTool from database: {e}")

            # Create pagination metadata
            pagination = PaginationSchema(
                total_count=total_count,
                limit=limit,
                offset=offset,
                has_more=offset + len(tools) < total_count,
            )

            return tools, pagination

    async def search_user_tools_by_embedding(
        self,
        user_id: str,
        query_embedding: List[float],
        similarity_threshold: float = 0.7,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[DynamicTool], PaginationSchema]:
        """
        Search for a specific user's dynamic tools based on embedding similarity

        Args:
            user_id: The ID of the user whose tools to search
            query_embedding: The query embedding vector to search with
            similarity_threshold: Minimum similarity score (0-1), defaults to 0.7
            limit: Maximum number of tools to return, defaults to 10
            offset: Number of tools to skip for pagination, defaults to 0

        Returns:
            Tuple containing list of tools and pagination metadata
        """
        async with self.typed_pool.acquire() as conn:
            # Get total count for pagination
            count_row = await conn.fetchrow(
                self.get_query("tool.count_user_tools_by_embedding"),
                user_id,
                query_embedding,
                similarity_threshold,
            )
            total_count = count_row["total_count"] if count_row else 0

            # Fetch paginated results
            rows = await conn.fetch(
                self.get_query("tool.search_user_tools_by_embedding"),
                user_id,
                query_embedding,
                similarity_threshold,
                limit,
                offset,
            )
            tools = []

            for row in rows:
                try:
                    tool_data = dict(row)

                    # Store the similarity score separately
                    similarity_score = tool_data.pop("similarity_score", None)

                    # Parse parameters if stored as JSON string
                    if isinstance(tool_data.get("parameters"), str):
                        try:
                            tool_data["parameters"] = json.loads(
                                tool_data["parameters"]
                            )
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse parameters JSON for tool: {e}"
                            )
                            continue

                    # Create the tool and add similarity score as an attribute
                    tool = DynamicTool(**tool_data)
                    setattr(tool, "similarity_score", similarity_score)
                    tools.append(tool)
                except Exception as e:
                    logger.error(f"Error creating DynamicTool from database: {e}")

            # Create pagination metadata
            pagination = PaginationSchema(
                total_count=total_count,
                limit=limit,
                offset=offset,
                has_more=offset + len(tools) < total_count,
            )

            return tools, pagination
