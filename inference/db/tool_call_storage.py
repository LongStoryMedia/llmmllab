"""
Storage service for managing tool call entities in the database.
Tool calls represent execution results from tools associated with messages.
"""

import asyncpg
from typing import List, Optional
from datetime import datetime, timezone
from models.tool_call import ToolCall
from db.db_utils import TypedConnection, typed_pool
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="tool_call_storage")


class ToolCallStorage:
    """Storage service for tool call entities with CRUD operations."""

    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = llmmllogger.bind(component="tool_call_storage_instance")

    async def add_tool_call(
        self,
        tool_call: ToolCall,
        conn: Optional[TypedConnection] = None,
    ) -> Optional[int]:
        """
        Add a new tool call to the database.

        Args:
            message_id: ID of the associated message
            tool_execution_result: The tool execution result data
            created_at: Optional timestamp (defaults to NOW())

        Returns:
            The ID of the created tool call, or None on failure
        """
        try:
            # Use provided connection or acquire a new one
            if conn is None:
                async with self.typed_pool.acquire() as connection:
                    return await self._add_tool_call(tool_call, connection)
            else:
                return await self._add_tool_call(tool_call, conn)

        except Exception as e:
            self.logger.error(
                f"Error adding tool call for message {tool_call.message_id}: {e}"
            )
            return None

    async def _add_tool_call(
        self,
        tool_call: ToolCall,
        conn: TypedConnection,
    ) -> Optional[int]:
        """Internal method to add tool call using a specific connection."""
        import json

        # Convert optional dict fields to JSON strings
        args_json = json.dumps(tool_call.args) if tool_call.args else None
        result_data_json = (
            json.dumps(tool_call.result_data) if tool_call.result_data else None
        )
        resource_usage_json = (
            json.dumps(tool_call.resource_usage.dict())
            if tool_call.resource_usage
            else None
        )

        row = await conn.fetchrow(
            self.get_query("tool_call.add_tool_call"),
            tool_call.message_id,  # $1
            tool_call.name,  # $2
            tool_call.execution_id,  # $3
            tool_call.success,  # $4
            args_json,  # $5
            result_data_json,  # $6
            tool_call.error_message,  # $7
            tool_call.execution_time_ms,  # $8
            resource_usage_json,  # $9
            tool_call.created_at,  # $10
        )

        if row:
            tool_call_id = row["id"]
            self.logger.info(
                f"Added tool call {tool_call_id} ({tool_call.name}) for message {tool_call.message_id}"
            )
            return tool_call_id
        else:
            self.logger.error(
                f"Failed to add tool call for message {tool_call.message_id}"
            )
            return None

    async def get_tool_calls_by_message(self, message_id: int) -> List[ToolCall]:
        """
        Retrieve all tool calls associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            List of ToolCall objects
        """
        try:
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(
                    self.get_query("tool_call.get_by_message"), message_id
                )

                tool_calls = []
                for row in rows:
                    # Parse JSON fields back to dict/objects
                    args = row["args"] if row["args"] else {}
                    result_data = row["result_data"] if row["result_data"] else None
                    resource_usage = (
                        row["resource_usage"] if row["resource_usage"] else None
                    )

                    tool_execution_result = ToolCall(
                        name=row["name"],
                        execution_id=row["execution_id"],
                        success=row["success"],
                        args=args,
                        result_data=result_data,
                        error_message=row["error_message"],
                        execution_time_ms=(
                            float(row["execution_time_ms"])
                            if row["execution_time_ms"]
                            else None
                        ),
                        resource_usage=resource_usage,
                        message_id=message_id,
                    )
                    tool_calls.append(tool_execution_result)

                self.logger.debug(
                    f"Retrieved {len(tool_calls)} tool calls for message {message_id}"
                )
                return tool_calls

        except Exception as e:
            self.logger.error(
                f"Error retrieving tool calls for message {message_id}: {e}"
            )
            return []
