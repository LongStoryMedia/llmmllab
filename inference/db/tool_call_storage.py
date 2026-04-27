"""
Storage service for managing tool call entities in the database.
Tool calls represent execution results from tools associated with messages.
"""

import json
from typing import List, Optional
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from models.tool_call import ToolCall
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="tool_call_storage")

ADD_TOOL_CALL_SQL = text("""
    INSERT INTO tool_calls(
        message_id, tool_name, execution_id, success, args, result_data,
        error_message, execution_time_ms, resource_usage, created_at
    ) VALUES (
        :message_id, :tool_name, :execution_id, :success, :args, :result_data,
        :error_message, :execution_time_ms, :resource_usage, :created_at
    )
    RETURNING id
""")

GET_BY_MESSAGE_SQL = text("""
    SELECT
        id, message_id, tool_name, execution_id, success, args, result_data,
        error_message, execution_time_ms, resource_usage, created_at
    FROM tool_calls
    WHERE message_id = :message_id
    ORDER BY created_at ASC
""")

DELETE_BY_MESSAGE_SQL = text("""
    DELETE FROM tool_calls
    WHERE message_id = :message_id
""")

UPDATE_RESULT_SQL = text("""
    UPDATE tool_calls
    SET
        result_data = :result_data,
        success = :success,
        error_message = :error_message,
        execution_time_ms = :execution_time_ms
    WHERE message_id = :message_id
      AND execution_id = :execution_id
    RETURNING
        id, message_id, tool_name AS name, execution_id, success, args,
        result_data, error_message, execution_time_ms, resource_usage, created_at
""")


class ToolCallStorage:
    """Storage service for tool call entities with CRUD operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory
        self.logger = llmmllogger.bind(component="tool_call_storage_instance")

    async def add_tool_call(
        self,
        tool_call: ToolCall,
    ) -> Optional[int]:
        """
        Add a new tool call to the database.

        Args:
            tool_call: The ToolCall object to persist

        Returns:
            The ID of the created tool call, or None on failure
        """
        try:
            async with self.session_factory() as session:
                tool_call_id = await self._add_tool_call(tool_call, session)
                await session.commit()
                return tool_call_id
        except Exception as e:
            self.logger.error(
                f"Error adding tool call for message {tool_call.message_id}: {e}"
            )
            return None

    async def _add_tool_call(
        self,
        tool_call: ToolCall,
        session: AsyncSession,
    ) -> Optional[int]:
        """Internal method to add tool call using a specific session."""
        # Convert optional dict fields to JSON strings with safe serialization
        try:
            args_json = json.dumps(tool_call.args) if tool_call.args else "{}"
        except (TypeError, ValueError) as e:
            self.logger.error(
                f"Failed to serialize tool_call.args: {e}, args: {tool_call.args}"
            )
            if tool_call.args and isinstance(tool_call.args, dict):
                safe_args = {
                    k: (
                        str(v)
                        if not isinstance(v, (str, int, float, bool, list, dict))
                        else v
                    )
                    for k, v in tool_call.args.items()
                }
                args_json = json.dumps(safe_args)
            else:
                args_json = "{}"

        try:
            result_data_json = (
                json.dumps(tool_call.result_data) if tool_call.result_data else "{}"
            )
        except (TypeError, ValueError) as e:
            self.logger.error(
                f"Failed to serialize tool_call.result_data: {e}, result_data: {tool_call.result_data}"
            )
            if tool_call.result_data and isinstance(tool_call.result_data, dict):
                safe_result = {
                    k: (
                        str(v)
                        if not isinstance(v, (str, int, float, bool, list, dict))
                        else v
                    )
                    for k, v in tool_call.result_data.items()
                }
                result_data_json = json.dumps(safe_result)
            else:
                result_data_json = "{}"

        try:
            resource_usage_json = (
                tool_call.resource_usage.model_dump_json()
                if tool_call.resource_usage
                else "{}"
            )
        except (TypeError, ValueError) as e:
            self.logger.error(f"Failed to serialize tool_call.resource_usage: {e}")
            resource_usage_json = "{}"

        row = await session.execute(
            ADD_TOOL_CALL_SQL,
            {
                "message_id": tool_call.message_id,
                "tool_name": tool_call.name,
                "execution_id": tool_call.execution_id,
                "success": tool_call.success if tool_call.success is not None else False,
                "args": args_json,
                "result_data": result_data_json,
                "error_message": tool_call.error_message,
                "execution_time_ms": tool_call.execution_time_ms,
                "resource_usage": resource_usage_json,
                "created_at": tool_call.created_at or datetime.now(timezone.utc),
            },
        )
        tool_call_id = row.scalar()

        if tool_call_id:
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
            async with self.session_factory() as session:
                result = await session.execute(
                    GET_BY_MESSAGE_SQL, {"message_id": message_id}
                )

                tool_calls = []
                for row in result.mappings():
                    # Parse JSON fields back to dict/objects
                    args = row["args"]
                    if isinstance(args, str):
                        args = json.loads(args) if args.strip() else {}
                    elif args is None:
                        args = {}

                    result_data = row["result_data"]
                    if isinstance(result_data, str):
                        result_data = (
                            json.loads(result_data) if result_data.strip() else {}
                        )
                    elif result_data is None or result_data == {}:
                        result_data = None

                    resource_usage_data = row["resource_usage"]
                    if isinstance(resource_usage_data, str):
                        resource_usage_data = (
                            json.loads(resource_usage_data)
                            if resource_usage_data.strip()
                            else {}
                        )

                    resource_usage = None
                    if resource_usage_data and isinstance(resource_usage_data, dict):
                        from models.resource_usage import ResourceUsage

                        try:
                            resource_usage = ResourceUsage(**resource_usage_data)
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to parse resource_usage: {e}"
                            )
                            resource_usage = None

                    created_at = row["created_at"]
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at).replace(
                            tzinfo=timezone.utc
                        )

                    tool_execution_result = ToolCall(
                        name=row["tool_name"],
                        execution_id=row["execution_id"],
                        success=row["success"],
                        args=args,
                        result_data=result_data,
                        error_message=row["error_message"],
                        execution_time_ms=int(row["execution_time_ms"])
                        if row["execution_time_ms"]
                        else None,
                        resource_usage=resource_usage,
                        message_id=message_id,
                        created_at=created_at,
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

    async def delete_tool_calls_by_message(self, message_id: int) -> bool:
        """
        Delete all tool calls associated with a message.

        Args:
            message_id: ID of the message

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    DELETE_BY_MESSAGE_SQL, {"message_id": message_id}
                )
                await session.commit()
                count = result.rowcount  # type: ignore[attr-defined]
                self.logger.info(
                    f"Deleted {count} tool calls for message {message_id}"
                )
                return True
        except Exception as e:
            self.logger.error(
                f"Error deleting tool calls for message {message_id}: {e}"
            )
            return False

    async def update_tool_call_result(
        self,
        message_id: int,
        execution_id: str,
        result_data: Optional[dict] = None,
        success: Optional[bool] = None,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
    ) -> Optional[ToolCall]:
        """
        Update the result of an existing tool call.

        Args:
            message_id: ID of the associated message
            execution_id: The execution ID of the tool call
            result_data: New result data
            success: New success status
            error_message: New error message
            execution_time_ms: New execution time in milliseconds

        Returns:
            The updated ToolCall object, or None on failure
        """
        try:
            async with self.session_factory() as session:
                result_data_json = (
                    json.dumps(result_data) if result_data else "{}"
                )

                result = await session.execute(
                    UPDATE_RESULT_SQL,
                    {
                        "message_id": message_id,
                        "execution_id": execution_id,
                        "result_data": result_data_json,
                        "success": success,
                        "error_message": error_message,
                        "execution_time_ms": execution_time_ms,
                    },
                )
                await session.commit()

                row = result.mappings().one_or_none()
                if row is None:
                    self.logger.warning(
                        f"No tool call found for message {message_id} "
                        f"with execution_id {execution_id}"
                    )
                    return None

                # Parse the returned row into a ToolCall object
                args = row["args"]
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                elif args is None:
                    args = {}

                rd = row["result_data"]
                if isinstance(rd, str):
                    rd = json.loads(rd) if rd.strip() else {}
                elif rd is None or rd == {}:
                    rd = None

                resource_usage_data = row["resource_usage"]
                if isinstance(resource_usage_data, str):
                    resource_usage_data = (
                        json.loads(resource_usage_data)
                        if resource_usage_data.strip()
                        else {}
                    )

                resource_usage = None
                if resource_usage_data and isinstance(resource_usage_data, dict):
                    from models.resource_usage import ResourceUsage

                    try:
                        resource_usage = ResourceUsage(**resource_usage_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse resource_usage: {e}")

                created_at = row["created_at"]
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at).replace(
                        tzinfo=timezone.utc
                    )

                updated = ToolCall(
                    name=row["name"],
                    execution_id=row["execution_id"],
                    success=row["success"],
                    args=args,
                    result_data=rd,
                    error_message=row["error_message"],
                    execution_time_ms=(
                        int(row["execution_time_ms"])
                        if row["execution_time_ms"]
                        else None
                    ),
                    resource_usage=resource_usage,
                    message_id=message_id,
                    created_at=created_at,
                )
                self.logger.info(
                    f"Updated tool call result for message {message_id} "
                    f"execution {execution_id}"
                )
                return updated

        except Exception as e:
            self.logger.error(
                f"Error updating tool call result for message {message_id} "
                f"execution {execution_id}: {e}"
            )
            return None
