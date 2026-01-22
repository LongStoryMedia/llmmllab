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
from db.db_utils import typed_pool
from .serialization import serialize_to_json
from .memory_storage import MemoryStorage

logger = logging.getLogger(__name__)


class DynamicToolStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    def _parse_tool_from_row(self, row) -> DynamicTool:
        """Parse a DynamicTool from database row data with proper field handling"""
        tool_data = dict(row)

        # Parse JSON fields
        if isinstance(tool_data.get("parameters"), str):
            try:
                tool_data["parameters"] = json.loads(tool_data["parameters"])
            except (json.JSONDecodeError, TypeError):
                tool_data["parameters"] = None

        if isinstance(tool_data.get("args_schema"), str):
            try:
                tool_data["args_schema"] = json.loads(tool_data["args_schema"])
            except (json.JSONDecodeError, TypeError):
                tool_data["args_schema"] = None

        if isinstance(tool_data.get("metadata"), str):
            try:
                tool_data["metadata"] = json.loads(tool_data["metadata"])
            except (json.JSONDecodeError, TypeError):
                tool_data["metadata"] = {}

        # Parse error handler fields (convert back from string to appropriate type)
        for field in ["handle_tool_error", "handle_validation_error"]:
            value = tool_data.get(field)
            if value is not None and isinstance(value, str):
                if value.lower() in ["true", "false"]:
                    tool_data[field] = value.lower() == "true"
                elif value.lower() == "none":
                    tool_data[field] = None
                # Otherwise keep as string

        # Ensure default values for optional fields
        tool_data.setdefault("tags", [])
        tool_data.setdefault("metadata", {})
        tool_data.setdefault("return_direct", False)
        tool_data.setdefault("verbose", False)
        tool_data.setdefault("response_format", "content")

        return DynamicTool(**tool_data)

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
                return self._parse_tool_from_row(row)
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
                    tools.append(self._parse_tool_from_row(row))
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
                    tools.append(self._parse_tool_from_row(row))
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
        """Create a new dynamic tool with LangChain BaseTool interface support"""
        # Serialize complex fields to JSON if needed
        params_json = serialize_to_json(tool.parameters) if tool.parameters else None
        args_schema_json = (
            serialize_to_json(tool.args_schema) if tool.args_schema else None
        )
        metadata_json = serialize_to_json(tool.metadata) if tool.metadata else "{}"

        # Convert embedding to database format if provided
        embedding = tool.embedding if tool.embedding is not None else None

        # Handle error handler fields (can be boolean, string, or None)
        handle_tool_error = (
            str(tool.handle_tool_error) if tool.handle_tool_error is not None else None
        )
        handle_validation_error = (
            str(tool.handle_validation_error)
            if tool.handle_validation_error is not None
            else None
        )

        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("tool.create_tool"),
                tool.user_id,  # $1
                tool.name,  # $2
                tool.description,  # $3
                tool.code,  # $4
                tool.function_name,  # $5
                embedding,  # $6
                args_schema_json,  # $7
                tool.return_direct,  # $8
                tool.tags or [],  # $9
                metadata_json,  # $10
                handle_tool_error,  # $11
                handle_validation_error,  # $12
                tool.response_format,  # $13
                params_json,  # $14
            )

            # Update the tool with the generated ID and timestamps
            if row:
                tool_data = dict(row)
                tool.id = tool_data.get("id")  # Set the auto-generated ID
                tool.created_at = tool_data.get("created_at")
                tool.updated_at = tool_data.get("updated_at")

            return tool

    async def update_tool(self, tool: DynamicTool) -> Optional[DynamicTool]:
        """Update an existing dynamic tool with LangChain BaseTool interface support"""
        # Serialize complex fields to JSON if needed
        params_json = serialize_to_json(tool.parameters) if tool.parameters else None
        args_schema_json = (
            serialize_to_json(tool.args_schema) if tool.args_schema else None
        )
        metadata_json = serialize_to_json(tool.metadata) if tool.metadata else "{}"

        # Handle error handler fields (can be boolean, string, or None)
        handle_tool_error = (
            str(tool.handle_tool_error) if tool.handle_tool_error is not None else None
        )
        handle_validation_error = (
            str(tool.handle_validation_error)
            if tool.handle_validation_error is not None
            else None
        )

        # Convert embedding to database format if provided
        embedding = tool.embedding if tool.embedding is not None else None

        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("tool.update_tool"),
                tool.id,  # $1
                tool.user_id,  # $2
                tool.name,  # $3
                tool.description,  # $4
                tool.code,  # $5
                tool.function_name,  # $6
                embedding,  # $7
                args_schema_json,  # $8
                tool.return_direct,  # $9
                tool.tags or [],  # $11
                metadata_json,  # $12
                handle_tool_error,  # $13
                handle_validation_error,  # $14
                tool.response_format,  # $15
                params_json,  # $16
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
                    # Store the similarity score separately before parsing
                    row_dict = dict(row)
                    similarity_score = row_dict.pop("similarity_score", None)

                    # Create the tool and add similarity score as an attribute
                    tool = self._parse_tool_from_row(row_dict)
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
        # Format embedding for PostgreSQL vector extension
        vector_str = MemoryStorage.format_embedding_for_pgvector(query_embedding)

        async with self.typed_pool.acquire() as conn:
            # Get total count for pagination
            count_row = await conn.fetchrow(
                self.get_query("tool.count_user_tools_by_embedding"),
                user_id,
                vector_str,
                similarity_threshold,
            )
            total_count = count_row["total_count"] if count_row else 0

            # Fetch paginated results
            rows = await conn.fetch(
                self.get_query("tool.search_user_tools_by_embedding"),
                user_id,
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
