"""
Storage service for managing tool call entities in the database.
Tool calls represent execution results from tools associated with messages.
"""

import asyncpg
from typing import List, Optional
from datetime import datetime, timezone
from models.tool_execution_result import ToolExecutionResult
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
        message_id: int,
        tool_execution_result: ToolExecutionResult,
        created_at: Optional[datetime] = None,
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
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        try:
            async with self.typed_pool.acquire() as conn:
                import json

                # Convert optional dict fields to JSON strings
                args_json = (
                    json.dumps(tool_execution_result.args)
                    if tool_execution_result.args
                    else None
                )
                result_data_json = (
                    json.dumps(tool_execution_result.result_data)
                    if tool_execution_result.result_data
                    else None
                )
                resource_usage_json = (
                    json.dumps(tool_execution_result.resource_usage.dict())
                    if tool_execution_result.resource_usage
                    else None
                )

                row = await conn.fetchrow(
                    self.get_query("tool_call.add_tool_call"),
                    message_id,  # $1
                    tool_execution_result.tool_name,  # $2
                    tool_execution_result.execution_id,  # $3
                    tool_execution_result.success,  # $4
                    args_json,  # $5
                    result_data_json,  # $6
                    tool_execution_result.error_message,  # $7
                    tool_execution_result.execution_time_ms,  # $8
                    resource_usage_json,  # $9
                    created_at,  # $10
                )

                if row:
                    tool_call_id = row["id"]
                    self.logger.info(
                        f"Added tool call {tool_call_id} ({tool_execution_result.tool_name}) for message {message_id}"
                    )
                    return tool_call_id
                else:
                    self.logger.error(
                        f"Failed to add tool call for message {message_id}"
                    )
                    return None

        except Exception as e:
            self.logger.error(f"Error adding tool call for message {message_id}: {e}")
            return None

    async def add_tool_call_legacy(
        self,
        message_id: int,
        tool_data: dict,
        created_at: Optional[datetime] = None,
    ) -> Optional[int]:
        """
        Add a new tool call to the database using legacy tool_data format.
        This method converts the legacy format to ToolExecutionResult.

        Args:
            message_id: ID of the associated message
            tool_data: The tool execution data as dict (legacy format)
            created_at: Optional timestamp (defaults to NOW())

        Returns:
            The ID of the created tool call, or None on failure
        """
        try:
            # Convert legacy format to ToolExecutionResult
            tool_execution_result = ToolExecutionResult(
                tool_name=tool_data.get("tool_name", "unknown"),
                execution_id=tool_data.get("execution_id"),
                success=tool_data.get("success", False),
                args=tool_data.get("args"),
                result_data=tool_data.get("result_data"),
                error_message=tool_data.get("error_message"),
                execution_time_ms=tool_data.get("execution_time_ms"),
                resource_usage=tool_data.get("resource_usage"),
            )

            return await self.add_tool_call(
                message_id, tool_execution_result, created_at
            )

        except Exception as e:
            self.logger.error(
                f"Error converting legacy tool call for message {message_id}: {e}"
            )
            return None

    async def get_tool_calls_by_message(
        self, message_id: int
    ) -> List[ToolExecutionResult]:
        """
        Retrieve all tool calls associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            List of ToolExecutionResult objects
        """
        try:
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(
                    self.get_query("tool_call.get_by_message"), message_id
                )

                tool_calls = []
                for row in rows:
                    # Parse JSON fields back to dict/objects
                    args = row["args"] if row["args"] else None
                    result_data = row["result_data"] if row["result_data"] else None
                    resource_usage = (
                        row["resource_usage"] if row["resource_usage"] else None
                    )

                    tool_execution_result = ToolExecutionResult(
                        tool_name=row["tool_name"],
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

    async def get_tool_calls_by_message_legacy(self, message_id: int) -> List[dict]:
        """
        Retrieve all tool calls associated with a message in legacy dict format.

        Args:
            message_id: ID of the message

        Returns:
            List of tool call dictionaries (legacy format)
        """
        try:
            tool_execution_results = await self.get_tool_calls_by_message(message_id)

            # Convert ToolExecutionResult objects back to legacy dict format
            tool_calls = []
            for ter in tool_execution_results:
                tool_call = {
                    "tool_name": ter.tool_name,
                    "execution_id": ter.execution_id,
                    "success": ter.success,
                    "args": ter.args,
                    "result_data": ter.result_data,
                    "error_message": ter.error_message,
                    "execution_time_ms": ter.execution_time_ms,
                    "resource_usage": (
                        ter.resource_usage.dict() if ter.resource_usage else None
                    ),
                }
                tool_calls.append(tool_call)

            return tool_calls

        except Exception as e:
            self.logger.error(
                f"Error retrieving legacy tool calls for message {message_id}: {e}"
            )
            return []

    async def delete_tool_calls_by_message(self, message_id: int) -> bool:
        """
        Delete all tool calls associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            async with self.typed_pool.acquire() as conn:
                result = await conn.execute(
                    self.get_query("tool_call.delete_by_message"), message_id
                )

                self.logger.info(f"Deleted tool calls for message {message_id}")
                return True

        except Exception as e:
            self.logger.error(
                f"Error deleting tool calls for message {message_id}: {e}"
            )
            return False
